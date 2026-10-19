#!/usr/bin/env python
"""Report of the collateral-aware erasure stage (PREREG amendment 3): validation selection, pooled edit-shape choice,
test results, 2x2 effects, gate sensitivity, Pareto curves.

    python -m experiments.varwidth.collateral_report shapes   # pooled alpha / m / best edit shape (validation only)
    python -m experiments.varwidth.collateral_report report   # -> results/varwidth/layers/collateral/poc.md (+ pareto png)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from experiments.varwidth.collateral import OUT, POC, seeds
from experiments.varwidth.common import slug
from experiments.varwidth.erase import hm
from experiments.varwidth.erase_v2 import COLL_GATE, MIN_MMLU

NAMES = {"sub:ember": "EMBER", "sub:cinder": "CINDER", "sub:cs_gs": "+CS+GS", "sub:ember_judge": "EMBER + TJ∨WJ+share",
         "sub:ee": "2×2: EMBER tok × EMBER dir (dense)", "sub:1b": "2×2: EMBER tok × CINDER dir (1b)",
         "sub:1a": "2×2: CINDER top-N × CINDER dir (1a)", "sub:cross": "2×2: CINDER top-N × EMBER dir",
         "mean:ember": "Mean (EMBER tok)", "mean:cinder": "Mean (CINDER tok)", "noise:ember": "Noise (EMBER tok)",
         "noise:cinder": "Noise (CINDER tok)", "sub:cinder_rs": "CINDER-RS (CS + RS-1% + TWJ)"}


def name(v):
    if v in NAMES:
        return NAMES[v]
    k, s = v.split(":")
    return {"proj": "Projection", "excl": "Exclusivity-weighted", "pow1.5": "Power m=1.5", "pow2.0": "Power m=2"}.get(k, k) + \
        f" ({'EMBER' if s == 'ember' else 'CINDER'} tok)"


def runs():
    """[(concept, seed)] of the stage."""
    return [(c, s) for c in POC for s in seeds(c)]


def grids():
    """{(concept, seed, variant): rows}."""
    out = {}
    for c, s in runs():
        for f in glob.glob(str(OUT / slug(c) / f"s{s}" / "*" / "grid.json")):
            rows = json.load(open(f))
            if rows:
                out[(c, s, rows[0]["variant"])] = rows
    return out


def h3v(x):
    return hm([x["efficacy"], x["specificity"], x.get("collateral", 0.0)])


def passes(x, gate):
    return x["mmlu_frac"] >= MIN_MMLU and x.get("collateral", 0.0) >= gate


def select(rows, gate=COLL_GATE, param=None):
    ok = [x for x in rows if passes(x, gate) and (param is None or x["param"] == param)]
    return None if not ok else sorted(ok, key=lambda x: (-h3v(x), x["r"]))[0]


def complete(rows):
    return all("collateral" in x for x in rows)


def shapes(G=None, verbose=True):
    """Pooled choices on validation: alpha for proj, m for pow, then the best edit shape (family x token set).
    Score = mean over concept-seeds of the selected point's H3v (0 when no point passes the gates)."""
    G = G or grids()
    keys = [k for k in G if complete(G[k])]

    def pooled(variant, param=None):
        vals = [(h3v(b) if (b := select(G[k], param=param)) else 0.0) for k in keys if k[2] == variant]
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    res = {"alpha": {}, "shape": {}}
    for s in ("ember", "cinder"):
        res["alpha"][s] = {a: pooled(f"proj:{s}", a) for a in (0.25, 0.5, 0.75, 1.0)}
    best_alpha = {s: max(res["alpha"][s], key=lambda a: res["alpha"][s][a][0]) for s in res["alpha"]}
    for s in ("ember", "cinder"):
        for fam in ("proj", "excl", "pow1.5", "pow2.0"):
            v = f"{fam}:{s}"
            res["shape"][v] = pooled(v, best_alpha[s] if fam == "proj" else None)
    best = max(res["shape"], key=lambda v: res["shape"][v][0])
    res["best_alpha"], res["best_shape"] = best_alpha, best
    if verbose:
        print(json.dumps(res, indent=1, default=str))
    (OUT / "shape_choice.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def results():
    """{(concept, seed, variant): result.json}."""
    out = {}
    for c, s in runs():
        for f in glob.glob(str(OUT / slug(c) / f"s{s}" / "*" / "result.json")):
            r = json.load(open(f))
            out[(c, s, r["variant"])] = r
    return out


def unedited_test_acc():
    """{concept: {token: unedited test accuracy}} from any no-edit result (same items, same option order). The base
    filter ran on items shuffled as "val"; test items are shuffled differently, so the unedited test accuracy is
    < 1 for some tokens (erratum, amendment 6)."""
    out = {}
    for (c, s, v), r in results().items():
        if r.get("param") is None and r.get("n_edit", 1) == 0 and c not in out:
            out[c] = r["coll_tok"]
    return out


def renorm(R):
    """Adds collateral_rn = mean over the variant's edited multi-meaning tokens of mc_norm(acc, unedited acc)
    (tokens whose unedited test accuracy is <= 0.25 are unmeasurable and left out; no edit -> 1.0), and
    H3v_rn / H4_rn."""
    import torch

    from experiments.varwidth.erase import mc_norm
    base = unedited_test_acc()
    sets_cache = {}
    for (c, s, v), r in R.items():
        if (c, s) not in sets_cache:
            sets_cache[(c, s)] = torch.load(OUT / slug(c) / f"edits_s{s}.pt", weights_only=False)
        ids = {str(t) for t in sets_cache[(c, s)][v.split(":")[1]]["ids"]} if r.get("n_edit") else set()
        b = base.get(c, {})
        vals = [mc_norm(a, b[t]) for t, a in r["coll_tok"].items() if t in ids and b.get(t, 0) > 0.25]
        r["collateral_rn"] = float(np.mean(vals)) if vals else 1.0
        r["H3v_rn"] = hm([r["efficacy"], r["specificity"], r["collateral_rn"]])
        if "coherence" in r:
            r["H4_rn"] = hm([r["efficacy"], r["specificity"], r["coherence"], r["collateral_rn"]])
    return R


def fmt(x, n=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def pareto_png(G):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for c, s in runs():
        fig, ax = plt.subplots(figsize=(7, 5))
        for v in ("sub:ember", "sub:cinder", "sub:1b", "sub:1a", "noise:ember", "proj:ember", "excl:ember", "pow2.0:ember"):
            rows = G.get((c, s, v))
            if not rows or not complete(rows):
                continue
            rows = sorted(rows, key=lambda x: x["r"])
            ax.plot([x["collateral"] for x in rows], [x["efficacy"] for x in rows], "o-", label=name(v), ms=3)
        for v in ("mean:ember", "mean:cinder"):
            if (c, s, v) in G and complete(G[(c, s, v)]):
                x = G[(c, s, v)][0]
                ax.plot([x["collateral"]], [x["efficacy"]], "k*", ms=10, label=name(v))
        ax.axvline(COLL_GATE, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("collateral (validation, own edited tokens)")
        ax.set_ylabel("efficacy (validation)")
        ax.set_title(f"{c}, seed {s}")
        ax.legend(fontsize=7)
        p = OUT / "pareto" / f"{slug(c)}_s{s}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def report():
    G, R = grids(), renorm(results())
    sh = json.loads((OUT / "shape_choice.json").read_text()) if (OUT / "shape_choice.json").exists() else shapes(G, False)
    L = ["# Collateral-aware erasure: development stage (Harry Potter, Ancient Rome, Valentine's Day)", "",
         "Pre-registered in PREREG amendment 3. Question set `data/collateral_questions.json` (sha256[:16] 57b21bb9dfcecfc2).",
         "Selection on validation by HM(efficacy, specificity, collateral) under gates MMLU-frac ≥ 0.9 and collateral ≥ "
         f"{COLL_GATE}; the test split is scored at the chosen point. Collateral = mean over the variant's own edited "
         "multi-meaning tokens of mc_norm(token accuracy, 1). H = paper (efficacy, specificity, coherence); "
         "H4 adds collateral; H3v = HM(efficacy, specificity, collateral) (no Alpaca).", ""]
    # main table
    L += ["## Test results at the selected point", "",
          "Collateral (rn) is normalized by the unedited model's test accuracy per token (erratum, amendment 6); the raw "
          "column assumes a baseline of 1.", "",
          "| Concept | Seed | Variant | r | n edit | Efficacy | Specificity | Collateral (rn) | Coll. raw | Coherence | H | H4 (rn) | H3v (rn) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (c, s, v), r in sorted(R.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        L.append(f"| {c} | {s} | {name(v)} | {fmt(r.get('r'), 3)} | {r.get('n_edit')} | {fmt(r['efficacy'])} | "
                 f"{fmt(r['specificity'])} | {fmt(r['collateral_rn'])} | {fmt(r['collateral'])} | {fmt(r.get('coherence'))} | "
                 f"{fmt(r.get('H'))} | {fmt(r.get('H4_rn'))} | {fmt(r['H3v_rn'])} |")
    # means per variant over concept-seeds
    L += ["", "## Mean over concept-seeds (variants present in every run of the concept are averaged over those runs)", "",
          "| Variant | runs | Efficacy | Collateral (rn) | H3v (rn) | H4 rn (if Alpaca) | H3v raw | mean selected r |", "|---|---|---|---|---|---|---|---|"]
    by = {}
    for (c, s, v), r in R.items():
        by.setdefault(v, []).append(r)
    for v, rs in sorted(by.items(), key=lambda kv: -np.mean([r["H3v_rn"] for r in kv[1]])):
        h4 = [r["H4_rn"] for r in rs if "H4_rn" in r]
        L.append(f"| {name(v)} | {len(rs)} | {fmt(np.mean([r['efficacy'] for r in rs]))} | "
                 f"{fmt(np.mean([r['collateral_rn'] for r in rs]))} | {fmt(np.mean([r['H3v_rn'] for r in rs]))} | "
                 f"{fmt(np.mean(h4)) if h4 else '—'} | {fmt(np.mean([r['H3v'] for r in rs]))} | "
                 f"{fmt(np.mean([r.get('r') or 0 for r in rs]), 2)} |")
    # 2x2
    L += ["", "## 2×2: token set × direction (dense Y, N = EMBER's edit count), test H3v", "",
          "| Concept | Seed | EE | 1b (CINDER dir) | 1a (CINDER tok+dir) | cross (CINDER tok) | Δ direction | Δ tokens | Δ interaction |",
          "|---|---|---|---|---|---|---|---|---|"]
    d_dir, d_tok = [], []
    for c, s in runs():
        cells = {k: R.get((c, s, f"sub:{k}")) for k in ("ee", "1b", "1a", "cross")}
        if not all(cells.values()):
            continue
        h = {k: v["H3v_rn"] for k, v in cells.items()}
        dd = ((h["1b"] - h["ee"]) + (h["1a"] - h["cross"])) / 2
        dt = ((h["cross"] - h["ee"]) + (h["1a"] - h["1b"])) / 2
        di = (h["1a"] - h["cross"]) - (h["1b"] - h["ee"])
        d_dir.append(dd)
        d_tok.append(dt)
        L.append(f"| {c} | {s} | {fmt(h['ee'])} | {fmt(h['1b'])} | {fmt(h['1a'])} | {fmt(h['cross'])} | {dd:+.2f} | {dt:+.2f} | {di:+.2f} |")
    if d_dir:
        L.append(f"| **mean** | | | | | | **{np.mean(d_dir):+.2f}** | **{np.mean(d_tok):+.2f}** | |")
    # regime shift: selected r under the paper-style rule (no collateral) vs collateral rule, on validation
    L += ["", "## Regime shift: selected r without vs with the collateral term (validation)", "",
          "| Concept | Seed | Variant | r, H2 rule (paper-style, no collateral) | r, collateral rule | efficacy H2→coll | collateral H2→coll |",
          "|---|---|---|---|---|---|---|"]
    for (c, s, v), rows in sorted(G.items()):
        if v not in ("sub:ember", "sub:cinder", "sub:ember_judge") or not complete(rows):
            continue
        okp = [x for x in rows if x["mmlu_frac"] >= MIN_MMLU]
        a = sorted(okp, key=lambda x: (-x["H2"], x["r"]))[0] if okp else None
        b = select(rows)
        if a:
            L.append(f"| {c} | {s} | {name(v)} | {fmt(a['r'], 3)} | {fmt(b['r'], 3) if b else 'none passes'} | "
                     f"{fmt(a['efficacy'])}→{fmt(b['efficacy']) if b else '—'} | {fmt(a['collateral'])}→{fmt(b['collateral']) if b else '—'} |")
    # gate sensitivity
    L += ["", "## Gate sensitivity (validation H3v of the selected point, mean over concept-seeds)", "",
          "| Variant | gate 0.7 | gate 0.8 | gate 0.9 |", "|---|---|---|---|"]
    vs = sorted({k[2] for k in G})
    for v in vs:
        row = []
        for g in (0.7, 0.8, 0.9):
            vals = [(h3v(b) if (b := select(G[k], g)) else 0.0) for k in G if k[2] == v and complete(G[k])]
            row.append(fmt(np.mean(vals)) if vals else "—")
        L.append(f"| {name(v)} | " + " | ".join(row) + " |")
    L += ["", "## Pooled edit-shape choice (validation)", "", "```json", json.dumps(sh, indent=1, default=str), "```", ""]
    paths = pareto_png(G)
    L += ["## Pareto curves (validation)", ""] + [f"![{p.stem}](pareto/{p.name})" for p in paths]
    (OUT / "poc.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT / 'poc.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("shapes", "report"))
    a = ap.parse_args()
    {"shapes": shapes, "report": report}[a.what]()


if __name__ == "__main__":
    main()
