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


ALPACA_SEED = 42
ALPACA_PAPER = ("sub:ember", "sub:cinder_z")


def alpaca(concept):
    """Alpaca coherence (Gemini, 50 prompts) at seed 42's collateral-rule points of the 5 variants and at the
    paper-rule points of EMBER and CINDER-z (amendment 10). -> <run dir>/alpaca_{sel,paper}.json"""
    from experiments.varwidth.erase import Evals
    from experiments.varwidth.erase_v2 import setup
    s = ALPACA_SEED
    base, sets, model, w, ed, E = setup(concept, s)
    bi, bf = base["_global"]["alpaca50"]
    ev = Evals(None, 50)
    for v in EXP_VARIANTS:
        d = OUT / slug(concept) / f"s{s}" / v.replace(":", "_")
        for rule, fn in (("sel", "result.json"), ("paper", "result_paper.json")):
            if rule == "paper" and v not in ALPACA_PAPER:
                continue
            out = d / f"alpaca_{rule}.json"
            if out.exists() or not (d / fn).exists():
                continue
            r = json.loads((d / fn).read_text())
            if not r.get("n_edit"):
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
            for rule, fn in (("sel", "result.json"), ("paper", "result_paper.json")):
                if (d / f"alpaca_{rule}.json").exists():
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
            for rule in ("sel", "paper"):
                xs = [x for (c, vv, rr), x in A.items() if vv == v and rr == rule]
                if not xs:
                    continue
                m = {k: float(np.mean([x[k] for x in xs])) for k in ("instr_frac", "flu_frac", "coherence", "H", "H4")}
                m["low"] = float(np.mean([x["coherence"] < 0.9 for x in xs]))
                m["n"] = len(xs)
                res["alpaca"][f"{v}@{rule}"] = m
                L.append(f"| {NAMES[v]} | {rule} | {m['n']} | {m['instr_frac']:.2f} | {m['flu_frac']:.2f} | "
                         f"{m['coherence']:.2f} | {m['low']:.2f} | {m['H']:.2f} | {m['H4']:.2f} |")
        L += ["", "Paired H4 vs EMBER at the collateral-rule point, seed 42 (concepts where both edit):", "",
              "| Variant − EMBER | concepts | mean ΔH4 | wins/losses |", "|---|---|---|---|"]
        for v in EXP_VARIANTS[1:]:
            d = [A[(c, v, "sel")]["H4"] - A[(c, "sub:ember", "sel")]["H4"] for c in EXPANSION
                 if (c, v, "sel") in A and (c, "sub:ember", "sel") in A]
            if d:
                res["alpaca"][f"paired_{v}"] = {"n": len(d), "mean": float(np.mean(d))}
                L.append(f"| {NAMES[v]} | {len(d)} | {np.mean(d):+.3f} | {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} |")
    (OUT / "last_mile.md").write_text("\n".join(L) + "\n")
    (OUT / "last_mile.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


def main():
    from dotenv import load_dotenv

    from experiments.varwidth.common import ROOT
    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("floor", "gen", "judge-export", "report", "alpaca"))
    ap.add_argument("--concept")
    a = ap.parse_args()
    {"floor": lambda: floor(a.concept), "gen": lambda: gen(a.concept), "judge-export": judge_export,
     "report": report, "alpaca": lambda: alpaca(a.concept)}[a.what]()


if __name__ == "__main__":
    main()
