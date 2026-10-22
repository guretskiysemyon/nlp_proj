#!/usr/bin/env python
"""WP2 step 2, follow-up: is the top of the cosine distribution semantic, and is raw cos just word frequency?

    .venv/bin/python -m experiments.paper_analysis.feature_geometry.semantic_extra

Per run (all 570), pairs i < j, rel = MiniLM relatedness of the first-20 source-word lists (embed_text.py):
  lift      within-run percentile of rel averaged over the pairs in each within-run cos-percentile bin
            (fit-space signed cos; bins: top 0.5%, 0.5-2%, 2-10%, 10-50%, bottom 50%); 50 = no information.
  partial   Spearman(cos, rel) after regressing both ranks on the pair's frequency covariates (rank of the mean of
            the two features' |y|-weighted log frequency, rank of their absolute difference): does raw cos track
            relatedness beyond "both features are made of rare / frequent tokens"?
  freq      Spearman(cos, mean log frequency of the pair) and Spearman(rel, the same).
Concept = unit (mean over seeds); paired L0 vs L1 / L12.
Outputs: semantic_extra_runs.csv, semantic_extra_stats.json, fig_semantic_lift.{pdf,png} (+ _data.csv).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from experiments.paper_analysis.feature_geometry.common import (COLORS, CONFIGS, NAMES, OUT, boot_ci, load_features,
                                                                load_pairs, paired, savefig, setup_mpl, text_emb_path)

K = 100
BINS = [(99.5, 100.0, "top 0.5%"), (98.0, 99.5, "0.5–2%"), (90.0, 98.0, "2–10%"), (50.0, 90.0, "10–50%"),
        (0.0, 50.0, "bottom 50%")]


def pct(x):
    return 100.0 * (rankdata(x) - 0.5) / len(x)


def resid(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def main():
    F = load_features()
    E = np.load(text_emb_path())
    kw = {k: i for i, k in enumerate(E["keys_w"])}
    iu = np.triu_indices(K, 1)
    rows = []
    for (c, cfg, s), g in F.groupby(["concept", "cfg", "seed"], sort=False):
        P = load_pairs(c, cfg, s)
        V = E["w20"][[kw[f"{c}|{cfg}|{s}|{f}"] for f in range(K)]]
        rel = (V @ V.T)[iu]
        lf = g.logf_w.values
        fm, fd = (lf[iu[0]] + lf[iu[1]]) / 2, np.abs(lf[iu[0]] - lf[iu[1]])
        Xf = np.column_stack([rankdata(fm), rankdata(fd)])
        rr = pct(rel)
        r = {"concept": c, "cfg": cfg, "seed": s}
        for sp in ("fit", "raw", "rawc"):
            x = P[f"cos_{sp}"].astype(np.float64)
            cp = pct(x)
            for lo, hi, name in BINS:
                m = (cp > lo) & (cp <= hi)
                r[f"lift_{sp}_{name}"] = float(rr[m].mean())
            r[f"partial_{sp}"] = float(spearmanr(resid(rankdata(x), Xf), resid(rankdata(rel), Xf)).statistic)
            r[f"sp_cos_freq_{sp}"] = float(spearmanr(x, fm).statistic)
        r["sp_rel_freq"] = float(spearmanr(rel, fm).statistic)
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "semantic_extra_runs.csv", index=False)
    Cc = R.groupby(["concept", "cfg"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    metrics = [m for m in Cc.columns if m not in ("concept", "cfg")]
    st = {"n_concepts": int(Cc.concept.nunique()), "by_cfg": {}, "paired": {}}
    for cfg in CONFIGS:
        sub = Cc[Cc.cfg == cfg]
        st["by_cfg"][cfg] = {m: {"mean": float(sub[m].mean()), "ci95": boot_ci(sub[m])} for m in metrics}
    for m in metrics:
        p = Cc.pivot(index="concept", columns="cfg", values=m)
        st["paired"][m] = {f"L0->{b}": paired(p["L0"], p[b]) for b in ("L1", "L12")}
    (OUT / "semantic_extra_stats.json").write_text(json.dumps(st, indent=1, default=float))

    # ---------------- figure: relatedness percentile by cos-percentile bin (fit space; CS also mapped to raw)
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    out = []
    xs = np.arange(len(BINS))[::-1]
    for cfg in CONFIGS:
        for sp, ls, mk in (("fit", "-", "o"), ("raw", ":", None)):
            if cfg == "L0" and sp == "raw":
                continue
            sub = Cc[Cc.cfg == cfg]
            ys = [sub[f"lift_{sp}_{n}"].mean() for _, _, n in BINS]
            ci = [boot_ci(sub[f"lift_{sp}_{n}"]) for _, _, n in BINS]
            lab = NAMES[cfg] + ("" if sp == "fit" else ", mapped to raw")
            ax.plot(xs, ys, ls=ls, marker=mk, ms=3, lw=1.5 if sp == "fit" else 1.0, color=COLORS[cfg], label=lab)
            if sp == "fit":
                ax.fill_between(xs, [c[0] for c in ci], [c[1] for c in ci], color=COLORS[cfg], alpha=0.2, lw=0)
            out += [{"cfg": cfg, "space": sp, "cos_bin": n, "rel_pct_mean": y, "ci_lo": c[0], "ci_hi": c[1]}
                    for (_, _, n), y, c in zip(BINS, ys, ci)]
    ax.axhline(50, color="k", lw=0.5, alpha=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, _, n in BINS], fontsize=6)
    ax.set_xlabel("pairs by within-run percentile of cos(z_i, z_j)", fontsize=7)
    ax.set_ylabel("relatedness percentile\n(MiniLM, within run)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig_semantic_lift")
    pd.DataFrame(out).to_csv(OUT / "fig_semantic_lift_data.csv", index=False)
    for cfg, d in st["by_cfg"].items():
        print(cfg, {k: round(v["mean"], 3) for k, v in d.items()})
    for m in ("partial_fit", "partial_rawc", "lift_fit_top 0.5%", "lift_fit_0.5–2%", "lift_raw_top 0.5%"):
        print(m, {k: (round(v["diff"], 3), [round(x, 3) for x in v["ci95"]], f"{v['wilcoxon_p']:.1e}", v["n_b_gt_a"])
                  for k, v in st["paired"][m].items()})


if __name__ == "__main__":
    main()
