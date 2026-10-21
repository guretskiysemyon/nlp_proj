#!/usr/bin/env python
"""Last-mile checks of the confirmatory expansion (amendment 9).

    python -m experiments.varwidth.last_mile floor --concept C      # GPU: nonce / deletion floors of efficacy
    python -m experiments.varwidth.last_mile gen --concept C        # GPU: neutral-prompt generations (premise, D3)
    python -m experiments.varwidth.last_mile judge-export           # CPU: blind OK/BROKEN batches of the generations
    python -m experiments.varwidth.last_mile report                 # CPU: -> collateral/last_mile.md

floor: the UNEDITED model on the efficacy test questions (qa_test) with every edited token of a (run, variant) made
unreadable in the text: replaced by a nonce ("nonce") or removed ("del"). Efficacy of that text = how much of an
edit's efficacy plain unreadability of its tokens would give. Options are kept in the original shuffle.
gen: greedy generations (80 tokens) of 20 neutral prompts per concept on the unedited model, each variant at its
collateral-rule point and EMBER / CINDER-z at the paper-rule point, at the concept's first seed where EMBER edits.
"""
from __future__ import annotations

import argparse
import copy
import json
import random

import numpy as np

from experiments.varwidth.collateral import EXP_VARIANTS, EXPANSION, OUT, edit_sets, seeds
from experiments.varwidth.common import slug

LM = OUT / "last_mile"
GEN_EXTRA = [("sub:ember", "paper"), ("sub:cinder_z", "paper")]


def unreadable(text, tok, ids, mode, banned):
    """text with every token whose id is in `ids` replaced by a nonce of its surface form, or removed."""
    from experiments.varwidth.integrity import nonce
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = [(a, b) for t, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]) if t in ids and b > a]
    for a, b in reversed(spans):
        piece = text[a:b]
        lead = piece[: len(piece) - len(piece.lstrip())]
        rep = "" if mode == "del" else lead + nonce(piece.strip() or "x", tok, banned)
        text = text[:a] + rep + text[b:]
    return text, len(spans)


def floor(concept):
    from ember.evals.mc import evaluate_mc_generation, prepare_mc_items
    from ember.local_datasets import load_mc_qa_items
    from experiments.varwidth.erase import load_model
    _, w = load_model()
    tok = w.tokenizer
    items = prepare_mc_items(load_mc_qa_items("qa_test", concept), seed=42)
    out_f = LM / slug(concept) / "floor.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    if "unedited" not in out:
        c, _, _ = evaluate_mc_generation(w, items)
        out["unedited"] = c / len(items)
    for s in seeds(concept):
        sets = edit_sets(concept, s)
        for v in EXP_VARIANTS:
            ids = set(int(t) for t in sets[v.split(":")[1]]["ids"])
            k = f"{s}|{v}"
            if not ids or k in out:
                continue
            res = {"n_edit": len(ids)}
            for mode in ("nonce", "del"):
                mod, touched = [], 0
                for it in items:
                    q, n = unreadable(it.question, tok, ids, mode, ids)
                    opts = {L: unreadable(o, tok, ids, mode, ids)[0] for L, o in it.options_by_letter.items()}
                    touched += n > 0
                    x = copy.copy(it)
                    x.question, x.options_by_letter = q, opts
                    x.correct_text = opts[it.correct_letter]
                    mod.append(x)
                c, _, _ = evaluate_mc_generation(w, mod)
                res[f"{mode}_acc"] = c / len(mod)
                res["q_touched"] = touched / len(mod)
            out[k] = res
            out_f.parent.mkdir(parents=True, exist_ok=True)
            out_f.write_text(json.dumps(out, indent=1))
            print(concept, k, res, flush=True)


def gen_seed(concept):
    return next(s for s in seeds(concept) if edit_sets(concept, s)["ember"]["ids"])


def gen(concept):
    from experiments.varwidth.erase_v2 import setup
    s = gen_seed(concept)
    P = json.loads((LM / "prompts.json").read_text())[concept]
    base, sets, model, w, ed, E = setup(concept, s)
    out_f = LM / slug(concept) / "gen.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    cfgs = [("unedited", None, None)]
    for v in EXP_VARIANTS:
        for rule, fname in (("sel", "result.json"), ("paper", "result_paper.json")):
            if rule == "paper" and (v, "paper") not in GEN_EXTRA:
                continue
            f = OUT / slug(concept) / f"s{s}" / v.replace(":", "_") / fname
            if f.exists() and json.loads(f.read_text()).get("param") is not None:
                cfgs.append((f"{v}@{rule}", v, json.loads(f.read_text())["param"]))
    for name, v, par in cfgs:
        if name in out:
            continue
        ed.restore()
        r = 0.0
        if v is not None:
            ids, new, r = E.rows(v, par)
            ed.set_rows(ids, new)
        gens = w.generate_multiple(P, max_new_tokens=80, do_sample=False, batch_size=10)
        out[name] = {"seed": s, "param": par, "r": r, "gens": gens}
        out_f.parent.mkdir(parents=True, exist_ok=True)
        out_f.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(concept, name, flush=True)
    ed.restore()


def wordfree(concept, seed):
    """Word-free efficacy (amendment 11) at the collateral-rule and paper-rule points of each variant.
    new: written items with no edited token id (union over the concept's variants and seeds) that the unedited model
    answers; old: qa_test items with none of the (run, variant)'s edited ids. -> <run dir>/<variant>/wordfree.json"""
    from ember.evals.mc import evaluate_mc_generation, prepare_mc_items
    from ember.local_datasets import MCQAItem, load_mc_qa_items
    from experiments.varwidth.erase_v2 import setup
    base, sets, model, w, ed, E = setup(concept, seed)
    tok = w.tokenizer
    union = set(json.loads((LM / "wordfree_union_ids.json").read_text())[concept])

    def ids_of(it):
        return set(tok(it.question, add_special_tokens=False)["input_ids"]).union(
            *[tok(o, add_special_tokens=False)["input_ids"] for o in it.options_by_letter.values()])
    df = LM / "wordfree_draft" / f"{slug(concept)}.json"      # Heroin, Nazism: no drafts (old items only)
    raw = json.loads(df.read_text())[concept] if df.exists() else []
    raw = [MCQAItem(concept, "QA", "wf", x["q"], x["options"], x["correct_answer"]) for x in raw
           if len(x.get("options", [])) == 4 and x.get("correct_answer") in x["options"]]
    new = [it for it in prepare_mc_items(raw, seed=42) if not ids_of(it) & union]
    ed.restore()
    _, _, recs = evaluate_mc_generation(w, new) if new else (0, 0, [])
    n_clean = len(new)
    new = [it for it, r in zip(new, recs) if r["is_correct"]]
    old_all = prepare_mc_items(load_mc_qa_items("qa_test", concept), seed=42)
    old_ids = [ids_of(it) for it in old_all]

    def acc(items):
        if not items:
            return float("nan")
        c, _, _ = evaluate_mc_generation(w, items)
        return c / len(items)
    for v in EXP_VARIANTS:
        d = OUT / slug(concept) / f"s{seed}" / v.replace(":", "_")
        out_f = d / "wordfree.json"
        if out_f.exists() or not (d / "result.json").exists():
            continue
        eid = set(int(t) for t in sets[v.split(":")[1]]["ids"])
        old = [it for it, s in zip(old_all, old_ids) if not s & eid]
        ed.restore()
        res = {"n_new_drafts": len(raw), "n_new_clean": n_clean, "n_new": len(new), "n_old": len(old),
               "old_base": acc(old)}
        for rule, fn in (("sel", "result.json"), ("paper", "result_paper.json")):
            if not (d / fn).exists():
                continue
            r = json.loads((d / fn).read_text())
            if not r.get("n_edit"):
                continue
            ids, newrows, rr = E.rows(v, r["param"])
            ed.set_rows(ids, newrows)
            res[rule] = {"param": r["param"], "new_acc": acc(new), "old_acc": acc(old)}
            ed.restore()
        out_f.write_text(json.dumps(res, indent=1))
        print(concept, seed, v, res, flush=True)


E3_RMAX = 2.5         # grid points with achieved r <= this get the copy test (copy is ~0 above, F9)
E3_COPY = 0.9


def copygrid(concept, seed):
    """E3 (amendment 11): T1 copy on every edited word (<= MAX_WORDS, seeded sample) at each grid point with
    r <= E3_RMAX, for the 5 variants. -> <run dir>/<variant>/copygrid.json {param: {r, copy, recs}}"""
    from experiments.varwidth.erase_v2 import setup
    from experiments.varwidth.integrity import COPY, MAX_WORDS, fill, source_words
    base, sets, model, w, ed, E = setup(concept, seed)
    tok = w.tokenizer
    for v in EXP_VARIANTS:
        d = OUT / slug(concept) / f"s{seed}" / v.replace(":", "_")
        if not (d / "grid.json").exists():
            continue
        out_f = d / "copygrid.json"
        out = json.loads(out_f.read_text()) if out_f.exists() else {}
        rows = [x for x in json.loads((d / "grid.json").read_text()) if x["r"] <= E3_RMAX]
        ids0 = E.comp(v)[0]
        if not len(ids0):
            continue
        sw = source_words(concept, tok, sorted(ids0))
        words = sorted({sw[t][1] for t in sw})
        if len(words) > MAX_WORDS:
            words = sorted(random.Random(seed).sample(words, MAX_WORDS))
        prompts = [(wd, f'Repeat the following sentence exactly, and nothing else: "{fill(c, wd)}"')
                   for wd in words for c in COPY]
        for x in rows:
            k = f"{x['param']:g}"
            if k in out:
                continue
            ids, new, rr = E.rows(v, x["param"])
            ed.set_rows(ids, new)
            outs = w.generate_multiple([p for _, p in prompts], max_new_tokens=40, do_sample=False, batch_size=32)
            ed.restore()
            recs = {}
            for (wd, _), o in zip(prompts, outs):
                recs.setdefault(wd, []).append(wd.strip('"') in o)
            out[k] = {"param": x["param"], "r": rr, "copy": float(np.mean([b for r in recs.values() for b in r])),
                      "recs": recs}
            out_f.write_text(json.dumps(out, ensure_ascii=False))
            print(concept, seed, v, k, f"r={rr:.3g} copy={out[k]['copy']:.2f}", flush=True)


ALPACA_SEED = 42
ALPACA_PAPER = ("sub:ember", "sub:cinder_z")


def alpaca(concept, rules=("sel", "paper")):
    """Alpaca coherence (Gemini, 50 prompts) at seed 42's collateral-rule points of the 5 variants and at the
    paper-rule points of EMBER and CINDER-z (amendment 10), or at the E3 points of the 5 variants (rules=("e3",),
    amendment 12; a point equal to the collateral-rule or paper-rule point reuses that score).
    -> <run dir>/alpaca_{sel,paper,e3}.json"""
    from experiments.varwidth.erase import Evals
    from experiments.varwidth.erase_v2 import setup
    s = ALPACA_SEED
    base, sets, model, w, ed, E = setup(concept, s)
    bi, bf = base["_global"]["alpaca50"]
    ev = Evals(None, 50)
    for v in EXP_VARIANTS:
        d = OUT / slug(concept) / f"s{s}" / v.replace(":", "_")
        for rule, fn in (("sel", "result.json"), ("paper", "result_paper.json"), ("e3", "result_e3.json")):
            if rule not in rules or (rule == "paper" and v not in ALPACA_PAPER):
                continue
            out = d / f"alpaca_{rule}.json"
            if out.exists() or not (d / fn).exists():
                continue
            r = json.loads((d / fn).read_text())
            if not r.get("n_edit"):
                continue
            same = [json.loads((d / f"alpaca_{q}.json").read_text()) for q in ("sel", "paper")
                    if q != rule and (d / f"alpaca_{q}.json").exists()]
            same = [x for x in same if x["param"] == r["param"]]
            if same:                              # same point already scored: reuse (amendment 12)
                res = dict(same[0], rule=rule, reused_from=same[0]["rule"])
                out.write_text(json.dumps(res, indent=1))
                print(concept, v, rule, "(reused)", res, flush=True)
                continue
            ids, new, rr = E.rows(v, r["param"])
            ed.set_rows(ids, new)
            ai, af = ev.alpaca(w)
            ed.restore()
            res = {"concept": concept, "seed": s, "variant": v, "rule": rule, "param": r["param"], "r": rr,
                   "alpaca_instr": ai, "alpaca_flu": af, "instr_frac": min(1.0, ai / bi), "flu_frac": min(1.0, af / bf)}
            from experiments.varwidth.erase import hm
            res["coherence"] = hm([res["instr_frac"], res["flu_frac"]])
            out.write_text(json.dumps(res, indent=1))
            print(concept, v, rule, res, flush=True)


def judge_export():
    """Blind batches: (prompt, reference answer, candidate answer) for every edited config; ids shuffled, the
    config kept in key.json. 10% duplicated into another batch for agreement."""
    J = LM / "judge"
    J.mkdir(parents=True, exist_ok=True)
    P = json.loads((LM / "prompts.json").read_text())
    items = []
    for c in EXPANSION:
        g = json.loads((LM / slug(c) / "gen.json").read_text())
        for name, x in g.items():
            if name == "unedited":
                continue
            for i, p in enumerate(P[c]):
                items.append({"concept": c, "config": name, "i": i, "prompt": p,
                              "reference": g["unedited"]["gens"][i], "candidate": x["gens"][i]})
    rng = random.Random(3)
    rng.shuffle(items)
    key = {}
    for n, it in enumerate(items):
        it["id"] = f"g{n:05d}"
        key[it["id"]] = {k: it[k] for k in ("concept", "config", "i")}
    dup = [dict(it, id=it["id"] + "d") for it in rng.sample(items, len(items) // 10)]
    nb = 6
    batches = [items[i::nb] for i in range(nb)]
    for j, d in enumerate(dup):
        b = (int(d["id"][1:6]) % nb + 1 + j % (nb - 1)) % nb      # never the original's batch
        batches[b].append(d)
    for b, its in enumerate(batches):
        rng.shuffle(its)
        (J / f"batch_{b}.json").write_text(json.dumps(
            [{k: it[k] for k in ("id", "prompt", "reference", "candidate")} for it in its], ensure_ascii=False, indent=1))
    (J / "key.json").write_text(json.dumps(key))
    print(f"{len(items)} items + {len(dup)} duplicates -> {nb} batches in {J}")


def ctx_check(res):
    """Context damage at the collateral-rule point (existing E2 outputs): context-MC and the answer-NLL excess over
    the nonce per variant against the references, and the paired contrast vs EMBER on context-MC."""
    from experiments.varwidth.collateral_report import NAMES
    from experiments.varwidth.confirm_stats import CONTRASTS, contrast
    from experiments.varwidth.integrity import IOUT, summarize
    S = {}
    for c in EXPANSION:
        for s in seeds(c):
            d = IOUT / slug(c) / f"s{s}"
            for v in EXP_VARIANTS:
                if (d / f"{v.replace(':', '_')}@sel.json").exists():
                    x = summarize(d, f"{v}@sel")["summary"]
                    S[(c, s, v)] = {"ctxmc": x["ctxmc"], "ctxmc_orig": x["ctxmc_orig"], "ctxmc_nonce": x["ctxmc_nonce"],
                                    "ctx_excess": x["ctx_excess"], "ctxmc_vs_orig": x["ctxmc"] - x["ctxmc_orig"],
                                    "n_edit": 1}
    if not S:
        return []
    L = ["", "## Context damage at the collateral-rule point (existing E2 outputs)", "",
         "Context-MC: an unrelated MMLU question asked right after a sentence that contains the edited word (same "
         "items for the edited model and the references). Answer-NLL excess: mean answer NLL after the word's "
         "sentence minus the same with the word's nonce (nats). Means over runs; the references are the same words "
         "in the unedited model.", "",
         "| Variant | runs | context-MC | ref: real word | ref: nonce | Δ vs real word | answer-NLL excess over nonce |",
         "|---|---|---|---|---|---|---|"]
    res["ctx"] = {}
    for v in EXP_VARIANTS:
        xs = [x for (c, s, vv), x in S.items() if vv == v]
        m = {k: float(np.mean([x[k] for x in xs])) for k in ("ctxmc", "ctxmc_orig", "ctxmc_nonce", "ctxmc_vs_orig",
                                                            "ctx_excess")}
        m["n"] = len(xs)
        res["ctx"][v] = m
        L.append(f"| {NAMES[v]} | {m['n']} | {m['ctxmc']:.3f} | {m['ctxmc_orig']:.3f} | {m['ctxmc_nonce']:.3f} | "
                 f"{m['ctxmc_vs_orig']:+.3f} | {m['ctx_excess']:+.3f} |")
    L += ["", "| Variant − EMBER | concepts (pairs) | Δ context-MC [95% CI] | p | Δ answer-NLL excess [95% CI] | p |",
          "|---|---|---|---|---|---|"]
    res["ctx_contrast"] = {}
    for v in CONTRASTS:
        a, b = contrast(S, v, "ctxmc"), contrast(S, v, "ctx_excess")
        res["ctx_contrast"][v] = {"ctxmc": a, "ctx_excess": b}
        if "mean" in a:
            L.append(f"| {NAMES[v]} | {a['n_concepts']} ({a['n_pairs']}) | {a['mean']:+.3f} [{a['ci'][0]:+.3f}, "
                     f"{a['ci'][1]:+.3f}] | {a['p']:.3f} | {b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] | "
                     f"{b['p']:.3f} |")
    return L


def amendment11(res):
    """Word-free efficacy and E3 sections (amendment 11)."""
    from experiments.varwidth.collateral_report import NAMES, renorm
    from experiments.varwidth.confirm_stats import CONTRASTS, contrast, holm, rates
    from experiments.varwidth.erase import hm, mc_norm
    from experiments.varwidth.integrity import token_class
    L = []
    # ---- word-free efficacy
    W = {}
    for c in EXPANSION:
        for s in seeds(c):
            for v in EXP_VARIANTS:
                d = OUT / slug(c) / f"s{s}" / v.replace(":", "_")
                if not (d / "wordfree.json").exists():
                    continue
                wf = json.loads((d / "wordfree.json").read_text())
                for rule, fn in (("sel", "result.json"), ("paper", "result_paper.json")):
                    if rule not in wf:
                        continue
                    r = {(c, s, v): json.loads((d / fn).read_text())}
                    renorm(r)
                    r = r[(c, s, v)]
                    x = wf[rule]
                    n1, n2 = wf["n_new"], wf["n_old"]
                    e_new = 1 - mc_norm(x["new_acc"], 1.0) if n1 else float("nan")
                    e_old = 1 - mc_norm(x["old_acc"], wf["old_base"]) if n2 and wf["old_base"] > 0.25 else float("nan")
                    acc = (x["new_acc"] * n1 + (x["old_acc"] * n2 if n2 else 0)) / max(1, n1 + n2)
                    base = (n1 + (wf["old_base"] * n2 if n2 else 0)) / max(1, n1 + n2)
                    e_all = 1 - mc_norm(acc, base)
                    W[(c, s, v, rule)] = {"eff": r["efficacy"], "eff_new": e_new, "eff_old": e_old, "eff_wf": e_all,
                                          "n_new": n1, "n_old": n2, "n_edit": 1, "H3v_rn": r["H3v_rn"],
                                          "H3v_wf": hm([e_all, r["specificity"], r["collateral_rn"]])}
    if W:
        L += ["", "## Word-free efficacy (amendment 11)", "",
              "Efficacy on questions that contain none of the edited tokens: new = written to avoid every token edited "
              "for the concept (kept if the unedited model answers them); old = existing test questions without the "
              "run's edited tokens; word-free = both pooled. Means over runs.", "",
              "| Variant | point | runs | items new / old (mean) | efficacy (all questions) | word-free: new | "
              "word-free: old | word-free: pooled | pooled / all |", "|---|---|---|---|---|---|---|---|---|"]
        res["wordfree"] = {}
        nm = lambda xs, k: float(np.nanmean([x[k] for x in xs])) if xs else float("nan")  # noqa: E731
        for v in EXP_VARIANTS:
            for rule in ("sel", "paper"):
                xs = [x for (c, s, vv, rr), x in W.items() if vv == v and rr == rule]
                if not xs:
                    continue
                m = {k: nm(xs, k) for k in ("eff", "eff_new", "eff_old", "eff_wf", "n_new", "n_old")}
                m["n"] = len(xs)
                res["wordfree"][f"{v}@{rule}"] = m
                L.append(f"| {NAMES[v]} | {rule} | {m['n']} | {m['n_new']:.0f} / {m['n_old']:.0f} | {m['eff']:.2f} | "
                         f"{m['eff_new']:.2f} | {m['eff_old']:.2f} | {m['eff_wf']:.2f} | "
                         f"{m['eff_wf'] / m['eff'] if m['eff'] else float('nan'):.2f} |")
        R = {(c, s, v): x for (c, s, v, rr), x in W.items() if rr == "sel"}
        L += ["", "Paired vs EMBER at the collateral-rule point (unit = concept, Wilcoxon; Holm over 4):", "",
              "| Variant − EMBER | concepts (pairs) | Δ word-free efficacy [95% CI] (p, Holm) | Δ H3v with word-free "
              "efficacy [95% CI] (p, Holm) |", "|---|---|---|---|"]
        C1 = {v: contrast(R, v, "eff_wf") for v in CONTRASTS}
        C2 = {v: contrast(R, v, "H3v_wf") for v in CONTRASTS}
        h1 = holm(np.array([C1[v].get("p", 1.0) for v in CONTRASTS]))
        h2 = holm(np.array([C2[v].get("p", 1.0) for v in CONTRASTS]))
        res["wordfree_contrast"] = {}
        for v, a, b in zip(CONTRASTS, h1, h2):
            x, y = C1[v], C2[v]
            res["wordfree_contrast"][v] = {"eff_wf": dict(x, p_holm=float(a)), "H3v_wf": dict(y, p_holm=float(b))}
            if "mean" not in x:
                continue
            L.append(f"| {NAMES[v]} | {x['n_concepts']} ({x['n_pairs']}) | {x['mean']:+.3f} [{x['ci'][0]:+.3f}, "
                     f"{x['ci'][1]:+.3f}] ({x['p']:.3f}, {a:.3f}) | {y['mean']:+.3f} [{y['ci'][0]:+.3f}, "
                     f"{y['ci'][1]:+.3f}] ({y['p']:.3f}, {b:.3f}) |")
    # ---- E3: copy-gated selection
    E = {}
    for c in EXPANSION:
        for s in seeds(c):
            for v in EXP_VARIANTS:
                f = OUT / slug(c) / f"s{s}" / v.replace(":", "_") / "result_e3.json"
                if f.exists():
                    E[(c, s, v)] = json.loads(f.read_text())
    if E:
        renorm(E)
        L += ["", "## E3: integrity-gated selection (copy ≥ 0.9; amendment 11)", "",
              "| Variant − EMBER | concepts (pairs) | mean ΔH3v | 95% CI | wins/losses | p | Holm p |",
              "|---|---|---|---|---|---|---|"]
        C = {v: contrast(E, v, "H3v_rn") for v in CONTRASTS}
        hh = holm(np.array([C[v].get("p", 1.0) for v in CONTRASTS]))
        res["e3"] = {"contrasts": {}, "rates": rates(E)}
        for v, a in zip(CONTRASTS, hh):
            x = C[v]
            res["e3"]["contrasts"][v] = dict(x, p_holm=float(a))
            if "mean" in x:
                L.append(f"| {NAMES[v]} | {x['n_concepts']} ({x['n_pairs']}) | {x['mean']:+.3f} | [{x['ci'][0]:+.3f}, "
                         f"{x['ci'][1]:+.3f}] | {x['wins']}/{x['losses']} | {x['p']:.3f} | {a:.3f} |")
        # exclusive-word copy at the selected point
        exc = {}
        for (c, s, v), r in E.items():
            if not r.get("n_edit"):
                continue
            cg = json.loads((OUT / slug(c) / f"s{s}" / v.replace(":", "_") / "copygrid.json").read_text())
            pt = cg.get(f"{r['param']:g}")
            if not pt:
                continue
            cls_tok = token_class(c)
            meta = json.loads((OUT / "integrity" / slug(c) / f"s{s}" / "meta.json").read_text())["words"] \
                if (OUT / "integrity" / slug(c) / f"s{s}" / "meta.json").exists() else {}
            for wd, b in pt["recs"].items():
                toks = meta.get(wd, {}).get("tokens", [])
                cl = "multi" if any(cls_tok.get(t) == "multi" for t in toks) else \
                    "exclusive" if any(cls_tok.get(t) == "exclusive" for t in toks) else "unclassified"
                exc.setdefault((v, cl), []).extend(b)
        L += ["", "| Variant | edit rate under E3 | mean efficacy | mean collateral | mean H3v | median r | copy, "
              "exclusive words | copy, multi-meaning words |", "|---|---|---|---|---|---|---|---|"]
        for v in EXP_VARIANTS:
            xs = [r for (c, s, vv), r in E.items() if vv == v and r.get("n_edit")]
            rt = res["e3"]["rates"][v]
            ce = exc.get((v, "exclusive"), [])
            cm = exc.get((v, "multi"), [])
            L.append(f"| {NAMES[v]} | {rt['edit_rate']:.2f} | {np.mean([x['efficacy'] for x in xs]):.2f} | "
                     f"{np.mean([x['collateral_rn'] for x in xs]):.2f} | {np.mean([x['H3v_rn'] for x in xs]):.2f} | "
                     f"{np.median([x['r'] for x in xs]):.2g} | {np.mean(ce) if ce else float('nan'):.2f} | "
                     f"{np.mean(cm) if cm else float('nan'):.2f} |")
    return L


def report():
    from experiments.varwidth.collateral_report import NAMES
    from experiments.varwidth.erase import mc_norm
    L = ["# Last-mile checks (PREREG amendment 9)", ""]
    # floor: efficacy of plain unreadability vs the efficacy at the two selected points
    rows = {}
    for c in EXPANSION:
        f = LM / slug(c) / "floor.json"
        if not f.exists():
            continue
        fl = json.loads(f.read_text())
        for k, x in fl.items():
            if k == "unedited":
                continue
            s, v = k.split("|")
            d = OUT / slug(c) / f"s{s}" / v.replace(":", "_")
            r = {m: 1 - mc_norm(x[f"{m}_acc"], fl["unedited"]) for m in ("nonce", "del")}
            r["q_touched"] = x["q_touched"]
            for rule, fn in (("sel", "result.json"), ("paper", "result_paper.json")):
                if (d / fn).exists() and json.loads((d / fn).read_text()).get("n_edit"):
                    r[rule] = json.loads((d / fn).read_text())["efficacy"]
            rows.setdefault(v, []).append(r)
    L += ["## Unreadability floor of efficacy", "",
          "Efficacy (1 − normalized accuracy) of the UNEDITED model when the edit set's tokens are replaced by a nonce "
          "or deleted in the efficacy test questions, against the edited model's efficacy at the collateral-rule "
          "point (sel) and the paper-rule point. Means over runs; 'share ≤ del' = share of runs whose efficacy is at "
          "or below the deletion floor.", "",
          "| Variant | runs | questions touched | floor: nonce | floor: deletion | efficacy @sel | share @sel ≤ del | "
          "efficacy @paper | share @paper ≤ del |", "|---|---|---|---|---|---|---|---|---|"]
    res = {"floor": {}}
    for v in EXP_VARIANTS:
        R = rows.get(v, [])
        if not R:
            continue
        m = lambda k: float(np.mean([r[k] for r in R if k in r])) if any(k in r for r in R) else float("nan")  # noqa
        sh = lambda k: float(np.mean([r[k] <= r["del"] + 1e-9 for r in R if k in r])) if any(k in r for r in R) \
            else float("nan")   # noqa: E731
        res["floor"][v] = {k: m(k) for k in ("q_touched", "nonce", "del", "sel", "paper")} | \
            {"share_sel_le_del": sh("sel"), "share_paper_le_del": sh("paper"), "n": len(R)}
        x = res["floor"][v]
        L.append(f"| {NAMES[v]} | {x['n']} | {x['q_touched']:.2f} | {x['nonce']:.2f} | {x['del']:.2f} | {x['sel']:.2f} | "
                 f"{x['share_sel_le_del']:.2f} | {x['paper']:.2f} | {x['share_paper_le_del']:.2f} |")
    # generation damage
    J = LM / "judge"
    if (J / "key.json").exists():
        key = json.loads((J / "key.json").read_text())
        lab = {}
        for f in sorted(J.glob("labels_*.json")):
            for k, x in json.loads(f.read_text()).items():
                lab.setdefault(k, x["label"] if isinstance(x, dict) else x)
        meta = json.loads((LM / "prompts_meta.json").read_text())
        agg = {}
        for i, k in key.items():
            if i not in lab:
                continue
            g = agg.setdefault(k["config"], {"all": [], "hit": []})
            g["all"].append(lab[i] == "BROKEN")
            if meta[k["concept"]][k["i"]]["token_hit"]:
                g["hit"].append(lab[i] == "BROKEN")
        batch_of = {it["id"]: int(f.stem.split("_")[1]) for f in J.glob("batch_*.json")
                    for it in json.loads(f.read_text())}
        dups = [(i, i + "d") for i in key if i in lab and i + "d" in lab and batch_of.get(i) != batch_of.get(i + "d")]
        agree = float(np.mean([lab[a] == lab[b] for a, b in dups])) if dups else float("nan")
        L += ["", "## Generation damage (neutral prompts, blind OK/BROKEN labels)", "",
              f"Labelled {sum(i in lab for i in key)}/{len(key)}; agreement on {len(dups)} cross-batch duplicates "
              f"{agree:.2f}. 'token present' = prompts whose target token id is in the tokenized prompt.", "",
              "| Config | prompts | broken | broken (token present) |", "|---|---|---|---|"]
        res["gen"] = {}
        for cfg in sorted(agg):
            g = agg[cfg]
            res["gen"][cfg] = {"n": len(g["all"]), "broken": float(np.mean(g["all"])),
                               "broken_hit": float(np.mean(g["hit"])) if g["hit"] else float("nan")}
            x = res["gen"][cfg]
            L.append(f"| {cfg} | {x['n']} | {x['broken']:.2f} | {x['broken_hit']:.2f} |")
        # the same on the concepts where every config exists (matched)
        have = {}
        for i, k in key.items():
            have.setdefault(k["concept"], set()).add(k["config"])
        full = set.union(*have.values())
        mc = sorted(c for c, s in have.items() if s == full)
        L += ["", f"Matched concepts (every config present: {', '.join(mc)}):", "", "| Config | prompts | broken |",
              "|---|---|---|"]
        res["gen_matched"] = {"concepts": mc}
        for cfg in sorted(full):
            xs = [lab[i] == "BROKEN" for i, k in key.items() if i in lab and k["config"] == cfg and k["concept"] in mc]
            res["gen_matched"][cfg] = float(np.mean(xs))
            L.append(f"| {cfg} | {len(xs)} | {np.mean(xs):.2f} |")
    # Alpaca coherence and H4 on seed 42 (amendment 10)
    from experiments.varwidth.collateral_report import renorm
    from experiments.varwidth.erase import hm
    A = {}
    for c in EXPANSION:
        for v in EXP_VARIANTS:
            d = OUT / slug(c) / f"s{ALPACA_SEED}" / v.replace(":", "_")
            for rule, fn in (("sel", "result.json"), ("paper", "result_paper.json"), ("e3", "result_e3.json")):
                if (d / f"alpaca_{rule}.json").exists() and (d / fn).exists():
                    r = {(c, ALPACA_SEED, v): json.loads((d / fn).read_text())}
                    renorm(r)
                    r = r[(c, ALPACA_SEED, v)]
                    a = json.loads((d / f"alpaca_{rule}.json").read_text())
                    A[(c, v, rule)] = {"coherence": a["coherence"], "instr_frac": a["instr_frac"],
                                       "flu_frac": a["flu_frac"], "efficacy": r["efficacy"],
                                       "H4": hm([r["efficacy"], r["specificity"], a["coherence"], r["collateral_rn"]]),
                                       "H": hm([r["efficacy"], r["specificity"], a["coherence"]])}
    if A:
        L += ["", "## Alpaca coherence and H4, seed 42 (amendment 10; descriptive)", "",
              "| Variant | point | runs | instr frac | fluency frac | coherence | share coherence < 0.9 | H (paper) | H4 |",
              "|---|---|---|---|---|---|---|---|---|"]
        res["alpaca"] = {}
        for v in EXP_VARIANTS:
            for rule in ("sel", "paper", "e3"):
                xs = [x for (c, vv, rr), x in A.items() if vv == v and rr == rule]
                if not xs:
                    continue
                m = {k: float(np.mean([x[k] for x in xs])) for k in ("instr_frac", "flu_frac", "coherence", "H", "H4")}
                m["low"] = float(np.mean([x["coherence"] < 0.9 for x in xs]))
                m["n"] = len(xs)
                res["alpaca"][f"{v}@{rule}"] = m
                L.append(f"| {NAMES[v]} | {rule} | {m['n']} | {m['instr_frac']:.2f} | {m['flu_frac']:.2f} | "
                         f"{m['coherence']:.2f} | {m['low']:.2f} | {m['H']:.2f} | {m['H4']:.2f} |")
        for rule, label in (("sel", "the collateral-rule point"), ("e3", "the E3 point (amendment 12)")):
            rows = []
            for v in EXP_VARIANTS[1:]:
                d = [A[(c, v, rule)]["H4"] - A[(c, "sub:ember", rule)]["H4"] for c in EXPANSION
                     if (c, v, rule) in A and (c, "sub:ember", rule) in A]
                if d:
                    res["alpaca"][f"paired_{v}" + ("" if rule == "sel" else f"@{rule}")] = {"n": len(d),
                                                                                       "mean": float(np.mean(d))}
                    rows.append(f"| {NAMES[v]} | {len(d)} | {np.mean(d):+.3f} | "
                                f"{sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} |")
            if rows:
                L += ["", f"Paired H4 vs EMBER at {label}, seed 42 (concepts where both edit):", "",
                      "| Variant − EMBER | concepts | mean ΔH4 | wins/losses |", "|---|---|---|---|"] + rows
    L += amendment11(res)
    L += ctx_check(res)
    (OUT / "last_mile.md").write_text("\n".join(L) + "\n")
    (OUT / "last_mile.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


def main():
    from dotenv import load_dotenv

    from experiments.varwidth.common import ROOT
    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("floor", "gen", "judge-export", "report", "alpaca", "copygrid", "wordfree"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--rules", nargs="*", default=["sel", "paper"], help="alpaca: points to score (sel paper e3)")
    ap.add_argument("--concept")
    a = ap.parse_args()
    {"floor": lambda: floor(a.concept), "gen": lambda: gen(a.concept), "judge-export": judge_export,
     "report": report, "alpaca": lambda: alpaca(a.concept, tuple(a.rules)),
     "copygrid": lambda: copygrid(a.concept, a.seed), "wordfree": lambda: wordfree(a.concept, a.seed)}[a.what]()


if __name__ == "__main__":
    main()
