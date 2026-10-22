#!/usr/bin/env python
"""Collateral-aware erasure (PREREG amendment 3) on gemma-2-2b-it, MC protocol, cheaper rule.

    python -m experiments.varwidth.erase_v2 grid  --concept C --seed S --group A|B [--qfile F] [--pilot]
    python -m experiments.varwidth.erase_v2 final --concept C --seed S [--alpaca VARIANT ...] [--qfile F]

Variants (edit sets from collateral.edit_sets; e, c in the TL-scaled embedding space, mu_t = |c_t| / |e_t|):
  sub:<set>         e - delta c                       sets: ember, cinder, cs_gs, ember_judge, ee, 1b, 1a, cross
  pow<m>:<set>      e - delta c^(m), c^(m) = sum_i sign(Y_it)|Y_it|^m z_i     (sets ember, cinder; m in 1.5, 2)
  excl:<set>        e - delta w_t c_t, w_t = exclusivity percentile of the token within the set
  noise:<set>       e + delta |c_t| u_t, u_t a fixed random unit vector (the paper's Noise)
  proj:<set>        e - alpha (e . c_hat) c_hat, alpha in PROJ (bounded: never past the token's own component)
  mean:<set>        e <- vocabulary mean embedding (the paper's Mean; one point)
Grid: target median relative perturbation r = median_t |e'_t - e_t| / |e_t| in REL (delta = r / median(mu)); every
row records the achieved r. Validation per row: MMLU train 50, qa_train 50, simdom_train 50, collateral validation
items (if the question file exists; `grid` fills missing collateral on rerun).
Selection (cheaper rule, as the anchor): gates mmlu_frac >= 0.9 and collateral >= COLL_GATE; pick max
HM(efficacy, specificity, collateral) (ties: smaller r; none passes: no edit). Test at the chosen point: qa_test,
simdom_test, MMLU test 1000, collateral test; Alpaca (50 test prompts) only for --alpaca variants.
H4 = HM(efficacy, specificity, coherence, collateral); H = the paper's HM(efficacy, specificity, coherence).
Outputs results/varwidth/layers/collateral/<slug>/s<seed>/<variant>/{grid.json, result.json}.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from experiments.varwidth.collateral import OUT, QFILE, Editor, edit_sets, exclusivity, load_fit
from experiments.varwidth.common import ROOT, slug
from experiments.varwidth.erase import BASE, Evals, baselines, hm, load_model, mc_norm, save

REL = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
PILOT_REL = [0.1, 1.0, 10.0, 100.0]
PROJ = [0.25, 0.5, 0.75, 1.0]
POW = [1.5, 2.0]
MIN_MMLU, COLL_GATE = 0.9, 0.8
SUB_SETS = ["ember", "cinder", "cs_gs", "ember_judge", "ee", "1b", "1a", "cross", "cinder_rs", "cinder_z", "cinder_rs_z"]
SHAPE_SETS = ["ember", "cinder"]


def variant_list(sets, group=None, pilot=False):
    """Variant names available for an edit-set dict; group A = sub + mean + noise, B = shapes (proj, excl, pow)."""
    have = {k for k, v in sets.items() if not k.startswith("_") and v["ids"]}
    # rungs with the same fit and F_C as another give identical edits: keep the first
    for dup, of in (("cs_gs", "cinder"), ("ember_judge", "ember")):
        if dup in have and of in have and sets[dup]["F"] == sets[of]["F"] and sets[dup]["fit"] == sets[of]["fit"]:
            have.discard(dup)
    if pilot:
        return [v for v in ["sub:ember", "sub:ee", "sub:cinder", "mean:ember", "noise:ember", "sub:1b", "proj:ember"]
                if v.split(":")[1] in have]
    A = [f"sub:{s}" for s in SUB_SETS if s in have] + [f"{k}:{s}" for k in ("mean", "noise") for s in SHAPE_SETS if s in have]
    B = [f"{k}:{s}" for s in SHAPE_SETS if s in have for k in ["proj", "excl"] + [f"pow{m}" for m in POW]]
    return {"A": A, "B": B, None: A + B}[group]


def grid_params(variant, pilot=False):
    kind = variant.split(":")[0]
    if kind == "mean":
        return [None]
    if kind == "proj":
        return [1.0] if pilot else PROJ
    return PILOT_REL if pilot else REL


class Edits:
    """Builds the new rows of a variant at a grid parameter."""

    def __init__(self, concept, seed, sets, editor, mean_row):
        self.c, self.s, self.sets, self.ed, self.mean_row = concept, seed, sets, editor, mean_row
        self._cache = {}

    def base(self, setname):
        st = self.sets[setname]
        E = self.ed.orig_scaled(st["ids"]).double().cpu()
        C = st["C"].double()
        return st, E, C

    def comp(self, variant):
        """(ids, E, D): D = the per-token direction to subtract at delta = 1 (for sub/pow/excl/noise magnitudes)."""
        if variant in self._cache:
            return self._cache[variant]
        kind, setname = variant.split(":")
        st, E, C = self.base(setname)
        if kind.startswith("pow"):
            m = float(kind[3:])
            prep, Z, Y, _, _, whit, _ = load_fit(self.c, st["fit"], self.s)
            Ym = Y[st["F"]][:, st["cols"]]
            C = (Z[:, st["F"]] @ (Ym.sign() * Ym.abs() ** m)).T
            if whit:
                C = C @ prep["T_inv"].double()
        elif kind == "excl":
            ex = exclusivity(self.c)
            x = np.array([ex.get(t, -9.0) for t in st["ids"]])
            w = (np.argsort(np.argsort(x)) + 1) / len(x)  # percentile within the set, (0, 1]
            C = C * torch.tensor(w, dtype=C.dtype)[:, None]
        mu = (C.norm(dim=1) / E.norm(dim=1)).numpy()
        self._cache[variant] = (st["ids"], E, C, float(np.median(mu)))
        return self._cache[variant]

    def rows(self, variant, p):
        kind = variant.split(":")[0]
        ids, E, C, med = self.comp(variant)
        if kind == "mean":
            new = self.mean_row.double().cpu()[None].expand_as(E).clone()
        elif kind == "proj":
            ch = C / C.norm(dim=1, keepdim=True)
            new = E - p * (E * ch).sum(1, keepdim=True) * ch
        elif kind == "noise":
            g = torch.Generator().manual_seed(1234 + self.s)
            u = torch.randn(E.shape, generator=g, dtype=E.dtype)
            u = u / u.norm(dim=1, keepdim=True)
            new = E + (p / med) * C.norm(dim=1, keepdim=True) * u
        else:
            new = E - (p / med) * C
        r = float(np.median(((new - E).norm(dim=1) / E.norm(dim=1)).numpy()))
        return ids, new, r


def load_questions(concept, qfile):
    """{split: [(token id, MCQAItem)]} of the collateral set, or None."""
    import os

    from ember.local_datasets import MCQAItem
    if not qfile or not os.path.exists(qfile):
        return None
    d = json.loads(open(qfile).read()).get(concept)
    if not d:
        return None
    out = {"val": [], "test": []}
    for tid, v in d["tokens"].items():
        for sp in ("val", "test"):
            for it in v[sp]:
                out[sp].append((tid, MCQAItem(concept, "Collateral", sp, it["q"], it["options"], it["correct_answer"])))
    return out


def coll_from_tok(acc, edited_ids):
    """(primary, union): primary = mean over the multi-meaning tokens the variant edits of mc_norm(token accuracy, 1)
    (1.0 when it edits none; decided on the pilot, amendment 3: the union mean is diluted by untouched tokens and
    depends on the pool); union = mean over all tokens of the set."""
    ed = {str(t) for t in edited_ids}
    sc = {t: mc_norm(a, 1.0) for t, a in acc.items()}
    mine = [v for t, v in sc.items() if t in ed]
    return (float(np.mean(mine)) if mine else 1.0), float(np.mean(list(sc.values()))), len(mine)


def coll_score(w, Q, split):
    """Per-token MC accuracy on the collateral items (kept items are all answered by the unedited model, so the
    baseline accuracy is 1). Returns {token: acc}."""
    from ember.evals.mc import evaluate_mc_generation, prepare_mc_items
    toks = [t for t, _ in Q[split]]
    mc = prepare_mc_items([it for _, it in Q[split]], seed=42)
    _, _, recs = evaluate_mc_generation(w, mc)
    per = {}
    for t, r in zip(toks, recs):
        per.setdefault(t, []).append(bool(r["is_correct"]))
    return {t: float(np.mean(v)) for t, v in per.items()}


def out_dir(concept, seed, variant):
    return OUT / slug(concept) / f"s{seed}" / variant.replace(":", "_")


def setup(concept, seed):
    base = json.loads(BASE.read_text())
    if concept not in base:
        base = baselines([concept], 50)
    sets = edit_sets(concept, seed)
    model, w = load_model()
    ed = Editor(model)
    W = model.get_input_embeddings().weight
    mean_row = W.detach().float().mean(0) * ed.scale
    from experiments.varwidth.erase import check_space
    for rung in ("L0", "L12"):                    # the plain and the CS fit: the edit space matches the live model
        check_space(concept, rung, seed, W, ed.scale)
    return base, sets, model, w, ed, Edits(concept, seed, sets, ed, mean_row)


def add_coll(r, acc, ids):
    r["coll_tok"] = acc
    r["collateral"], r["coll_all"], r["coll_n"] = coll_from_tok(acc, ids)


def score_val(ev, w, bc, bg, Q, r, ids=()):
    r["mmlu_acc"] = ev.mmlu_acc(w, "train")
    r["mmlu_frac"] = mc_norm(r["mmlu_acc"], bg["mmlu_train"])
    for n in ("qa", "simdom"):
        r[f"{n}_acc"] = ev.mc_acc(w, f"{n}_train")
        r[f"{n}_frac"] = mc_norm(r[f"{n}_acc"], bc[f"{n}_train"])
    r["efficacy"] = 1 - r["qa_frac"]
    r["specificity"] = hm([r["mmlu_frac"], r["simdom_frac"]])
    r["H2"] = hm([r["efficacy"], r["specificity"]])
    if Q:
        add_coll(r, coll_score(w, Q, "val"), ids)


def grid(concept, seed, group, qfile, pilot=False, variants=None):
    base, sets, model, w, ed, E = setup(concept, seed)
    bc, bg = base[concept], base["_global"]
    ev = Evals(concept, 0)
    Q = load_questions(concept, qfile)
    vs = [v for v in variants if sets[v.split(":")[1]]["ids"]] if variants else variant_list(sets, group, pilot)
    t0 = time.time()
    if pilot:   # the unedited model on the collateral items, and the anchor's sparse EMBER at delta 200 (reproduction)
        d = OUT / slug(concept) / f"s{seed}" / "_pilot"
        ed.restore()
        r = {"variant": "unedited"}
        score_val(ev, w, bc, bg, Q, r)
        st = sets["ember"]
        if st["ids"]:
            ed.subtract(st["ids"], st["C"], 200.0)
            rep = {"variant": "sub:ember@delta200", "qa_test": ev.mc_acc(w, "qa_test"),
                   "simdom_test": ev.mc_acc(w, "simdom_test"), "mmlu_test": ev.mmlu_acc(w, "test")}
        else:
            rep = {}
        save(d / "checks.json", {"unedited": r, "reproduction": rep, "info": sets["_info"]})
        print("pilot checks", json.dumps({"unedited": {k: v for k, v in r.items() if k != "coll_tok"}, "repro": rep}),
              flush=True)
    for v in vs:
        d = out_dir(concept, seed, v)
        gpath = d / "grid.json"
        rows = json.loads(gpath.read_text()) if gpath.exists() else []
        by = {json.dumps(x["param"]): x for x in rows}
        for p in grid_params(v, pilot):
            row = by.get(json.dumps(p))
            if row is not None and ("collateral" in row or not Q):
                continue
            ids, new, rr = E.rows(v, p)
            ed.set_rows(ids, new)
            if row is None:
                row = {"variant": v, "param": p, "r": rr, "n_edit": len(ids)}
                score_val(ev, w, bc, bg, Q, row, ids)
                rows.append(row)
            else:
                add_coll(row, coll_score(w, Q, "val"), ids)
            save(gpath, rows)
            print(f"{concept} s{seed} {v} p={p} r={rr:.3g} eff={row['efficacy']:.2f} spec={row['specificity']:.2f} "
                  f"coll={row.get('collateral', float('nan')):.2f} [{time.time() - t0:.0f}s]", flush=True)
    ed.restore()


def select(rows, gate=COLL_GATE):
    ok = [x for x in rows if x["mmlu_frac"] >= MIN_MMLU and x.get("collateral", 0) >= gate]
    for x in rows:
        x["H3v"] = hm([x["efficacy"], x["specificity"], x.get("collateral", 0.0)])
    if not ok:
        return None
    return sorted(ok, key=lambda x: (-x["H3v"], x["r"]))[0]


def select_paper(rows):
    """The paper-style rule (the anchor's cheaper rule): MMLU-frac >= 0.9, max HM(efficacy, specificity), no
    collateral term (amendment 7, "old evaluation")."""
    ok = [x for x in rows if x["mmlu_frac"] >= MIN_MMLU]
    return None if not ok else sorted(ok, key=lambda x: (-x["H2"], x["r"]))[0]


PAPER_VARIANTS = ["sub:ember", "sub:cinder", "sub:cinder_rs", "sub:cinder_z", "sub:cinder_rs_z", "sub:1b"]
TWOBYTWO = ["sub:ee", "sub:1b", "sub:1a", "sub:cross"]      # amendment 14: fixed-r test passes, no grid needed


def final(concept, seed, qfile, alpaca_variants, gate=COLL_GATE, rule="collateral"):
    base, sets, model, w, ed, E = setup(concept, seed)
    bc, bg = base[concept], base["_global"]
    ev = Evals(concept, 50)
    Q = load_questions(concept, qfile)
    assert Q, "final needs the frozen collateral set"
    bfile = OUT / slug(concept) / "unedited_test_coll.json"     # amendment 8: unedited per-token test accuracy
    if not bfile.exists():
        ed.restore()
        save(bfile, coll_score(w, Q, "test"))
    sc = OUT / "shape_choice.json"           # pooled alpha per token set (collateral_report shapes)
    alpha = json.loads(sc.read_text())["best_alpha"] if sc.exists() else {}
    for v in variant_list(sets):
        d = out_dir(concept, seed, v)
        fname = {"collateral": "result.json", "paper": "result_paper.json", "fixed05": "result_fixed05.json",
                 "fixed1": "result_fixed1.json", "e3": "result_e3.json"}[rule]
        if rule == "collateral" and gate != COLL_GATE:          # gate sensitivity: its own file, result.json untouched
            fname = f"result_g{gate:g}.json"
        fixed = {"fixed05": 0.5, "fixed1": 1.0}.get(rule)
        cell = fixed is not None and v in TWOBYTWO
        if (d / fname).exists() or not ((d / "grid.json").exists() or cell):
            continue
        if rule == "paper" and v not in PAPER_VARIANTS or rule == "fixed05" and v not in PAPER_VARIANTS + TWOBYTWO:
            continue
        if rule == "fixed1" and not cell:
            continue
        rows = json.loads((d / "grid.json").read_text()) if (d / "grid.json").exists() else []
        if v.startswith("proj:"):
            assert alpha, "run collateral_report shapes first (pooled alpha)"
            rows = [x for x in rows if x["param"] == float(alpha[v.split(':')[1]])]
        assert all("collateral" in x for x in rows), f"{v}: grid rows without collateral (run the phase-2 pass)"
        if rule == "collateral":
            best = select(rows, gate)
        elif rule == "paper":
            best = select_paper(rows)
        elif rule == "e3":                       # amendment 11: collateral rule + copy >= 0.9 (copygrid.json)
            cg = json.loads((d / "copygrid.json").read_text()) if (d / "copygrid.json").exists() else {}
            ok = {x["param"] for x in cg.values() if x["copy"] >= 0.9}
            best = select([x for x in rows if x["param"] in ok], gate)
        elif rows:                               # fixed r = 0.5 (amendment 8)
            best = next((x for x in rows if x["param"] == fixed), None)
        else:                                    # 2×2 cell without a grid (amendment 14): param = target r
            best = {"param": fixed}
        res = {"concept": concept, "seed": seed, "variant": v, "gate": gate, "rule": rule}
        par = None if best is None else best["param"]
        same = [json.loads((d / f).read_text()) for f in ("result.json", "result_paper.json", "result_fixed05.json",
                                                          "result_g0.7.json", "result_g0.9.json", "result_e3.json",
                                                          "result_fixed1.json")
                if f != fname and (d / f).exists()]
        same = [x for x in same if x.get("param") == par and "alpaca_instr" not in x]
        if same and v not in alpaca_variants:   # another rule chose the same point: deterministic eval, reuse it
            res.update({k: x for k, x in same[0].items() if k not in res})
            save(d / fname, res)
            print(json.dumps({k: v2 for k, v2 in res.items() if k != "coll_tok"}), "(reused)", flush=True)
            continue
        if best is None:
            ed.restore()
            res.update(param=None, r=0.0, n_edit=0)
        else:
            ids, new, rr = E.rows(v, best["param"])
            ed.set_rows(ids, new)
            res.update(param=best["param"], r=rr, n_edit=len(ids))
        res["qa_acc"], res["simdom_acc"] = ev.mc_acc(w, "qa_test"), ev.mc_acc(w, "simdom_test")
        res["mmlu_acc"] = ev.mmlu_acc(w, "test")
        res["qa_frac"] = mc_norm(res["qa_acc"], bc["qa_test"])
        res["simdom_frac"] = mc_norm(res["simdom_acc"], bc["simdom_test"])
        res["mmlu_frac"] = mc_norm(res["mmlu_acc"], bg["mmlu_test"])
        res["efficacy"] = 1 - res["qa_frac"]
        res["specificity"] = hm([res["mmlu_frac"], res["simdom_frac"]])
        add_coll(res, coll_score(w, Q, "test"), E.comp(v)[0])
        if v in alpaca_variants:
            bi, bf = bg["alpaca50"]
            ai, af = (bi, bf) if best is None else ev.alpaca(w)
            res.update(alpaca_instr=ai, alpaca_flu=af, instr_frac=min(1.0, ai / bi), flu_frac=min(1.0, af / bf))
            res["coherence"] = hm([res["instr_frac"], res["flu_frac"]])
            res["H"] = hm([res["efficacy"], res["specificity"], res["coherence"]])
            res["H4"] = hm([res["efficacy"], res["specificity"], res["coherence"], res["collateral"]])
            res["alpaca_flag"] = res["instr_frac"] < 0.9 or res["flu_frac"] < 0.9
        res["H3v"] = hm([res["efficacy"], res["specificity"], res["collateral"]])
        save(d / fname, res)
        print(json.dumps({k: v2 for k, v2 in res.items() if k != "coll_tok"}), flush=True)
    ed.restore()


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("grid", "final"))
    ap.add_argument("--concept", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--group", choices=("A", "B"))
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--qfile", default=str(QFILE))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--alpaca", nargs="*", default=[])
    ap.add_argument("--gate", type=float, default=COLL_GATE)
    ap.add_argument("--rule", choices=("collateral", "paper", "fixed05", "fixed1", "e3"), default="collateral")
    a = ap.parse_args()
    if a.what == "grid":
        grid(a.concept, a.seed, a.group, a.qfile, a.pilot, a.variants)
    else:
        final(a.concept, a.seed, a.qfile, a.alpaca, a.gate, a.rule)


if __name__ == "__main__":
    main()
