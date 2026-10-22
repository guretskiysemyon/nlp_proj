#!/usr/bin/env python
"""WP2 step 2: do cosines between feature directions track the semantic relatedness of the features?

    .venv/bin/python -m experiments.paper_analysis.feature_geometry.semantic

Relatedness rel(i, j) = cosine of the MiniLM embeddings (embed_text.py) of the two features' first-20 source-word
lists (all pairs) or of their stage-1 Gemini descriptions (pairs of described features, rho >= 2 only).
Geometry cos(i, j) = sign-aligned cosine of z_i, z_j in the fit space (and raw / CS metric as controls).
Per run: Spearman(cos, rel) over pairs; also over pairs with disjoint supports (Jaccard 0) to remove the token
overlap confound; NN@1: for each feature, the percentile of rel(i, nn(i)) among rel(i, .) where nn(i) is its
nearest direction by cos (50 = chance). Concept = unit (mean over seeds); paired L0 vs L1 / L12.
Outputs: semantic_runs.csv, semantic_stats.json, fig_cos_vs_semantic.{pdf,png} (+ _data.csv).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.paper_analysis.feature_geometry.common import (COLORS, CONFIGS, NAMES, OUT, SEEDS, boot_ci, full,
                                                                load_features, load_pairs, paired, savefig, setup_mpl,
                                                                text_emb_path)

K = 100


def _sp(x, y):
    if len(x) < 10:
        return np.nan
    return float(spearmanr(x, y).statistic)


def nn_percentile(C: np.ndarray, R: np.ndarray) -> float:
    C, R = C.copy(), R.copy()
    np.fill_diagonal(C, -np.inf)
    np.fill_diagonal(R, np.nan)
    out = []
    for i in range(len(C)):
        j = int(np.argmax(C[i]))
        r = R[i][~np.isnan(R[i])]
        out.append(100.0 * (r < R[i, j]).mean())
    return float(np.mean(out))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="recompute semantic_runs.csv even if present")
    recompute = ap.parse_args().recompute or not (OUT / "semantic_runs.csv").exists()
    F = load_features()
    E = np.load(text_emb_path())
    kw = {k: i for i, k in enumerate(E["keys_w"])}
    kd = {k: i for i, k in enumerate(E["keys_d"])}
    iu = np.triu_indices(K, 1)
    rows, pooled = [], {cfg: [] for cfg in CONFIGS}
    for (c, cfg, s), g in F.groupby(["concept", "cfg", "seed"], sort=False):
        if not recompute and s != SEEDS[0]:
            continue
        P = load_pairs(c, cfg, s)
        keys = [f"{c}|{cfg}|{s}|{f}" for f in range(K)]
        V = E["w20"][[kw[k] for k in keys]]
        V10 = E["w10"][[kw[k] for k in keys]]
        Rw = V @ V.T
        rel = Rw[iu]
        rel10 = (V10 @ V10.T)[iu]
        jac0 = P["jac"] == 0
        if s == SEEDS[0]:
            pooled[cfg].append(pd.DataFrame({"rel": rel, "cos": P["cos_fit"], "jac": P["jac"], "concept": c}))
        if not recompute:
            continue
        r = {"concept": c, "cfg": cfg, "seed": s, "n_pairs_jac0": int(jac0.sum())}
        for sp in ("fit", "raw", "cs", "rawc"):
            x = P[f"cos_{sp}"]
            r[f"sp_{sp}"] = _sp(x, rel)
            r[f"sp_{sp}_abs"] = _sp(np.abs(x), rel)
            r[f"sp_{sp}_jac0"] = _sp(x[jac0], rel[jac0])
            r[f"sp_{sp}_w10"] = _sp(x, rel10)
        r["sp_jac"] = _sp(P["jac"], rel)
        r["nn_pct_fit"] = nn_percentile(full(P["cos_fit"]), Rw)
        r["nn_pct_raw"] = nn_percentile(full(P["cos_raw"]), Rw)
        # descriptions (only described features)
        dk = [f for f in range(K) if keys[f] in kd]
        r["n_desc"] = len(dk)
        if len(dk) >= 10:
            D = E["desc"][[kd[keys[f]] for f in dk]]
            Rd = D @ D.T
            ii = np.triu_indices(len(dk), 1)
            Cf = full(P["cos_fit"])[np.ix_(dk, dk)]
            Cr = full(P["cos_raw"])[np.ix_(dk, dk)]
            J = full(P["jac"])[np.ix_(dk, dk)]
            m0 = J[ii] == 0
            r["sp_desc_fit"] = _sp(Cf[ii], Rd[ii])
            r["sp_desc_raw"] = _sp(Cr[ii], Rd[ii])
            r["sp_desc_fit_jac0"] = _sp(Cf[ii][m0], Rd[ii][m0])
            # same features, word-list relatedness (is the difference the measure or the subset?)
            r["sp_desc_subset_words_fit"] = _sp(Cf[ii], Rw[np.ix_(dk, dk)][ii])
        rows.append(r)
    if recompute:
        R = pd.DataFrame(rows)
        R.to_csv(OUT / "semantic_runs.csv", index=False)
    else:
        R = pd.read_csv(OUT / "semantic_runs.csv")
    Cc = R.groupby(["concept", "cfg"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    st = {"n_concepts": int(Cc.concept.nunique()), "by_cfg": {}, "paired": {}, "pooled_seed42": {}}
    metrics = [m for m in Cc.columns if m.startswith(("sp_", "nn_pct", "n_desc"))]
    for cfg in CONFIGS:
        sub = Cc[Cc.cfg == cfg]
        st["by_cfg"][cfg] = {m: {"mean": float(sub[m].mean()), "ci95": boot_ci(sub[m]),
                                 "n_concepts_pos": int((sub[m] > 0).sum())} for m in metrics}
    for m in metrics:
        p = Cc.pivot(index="concept", columns="cfg", values=m)
        st["paired"][m] = {f"L0->{b}": paired(p["L0"], p[b]) for b in ("L1", "L12")}
    for cfg, L in pooled.items():
        D = pd.concat(L)
        st["pooled_seed42"][cfg] = {"n_pairs": len(D), "spearman": _sp(D.cos, D.rel),
                                    "spearman_jac0": _sp(D.cos[D.jac == 0], D.rel[D.jac == 0])}
    st["per_concept"] = Cc[["concept", "cfg", "sp_fit", "sp_fit_jac0", "sp_raw", "nn_pct_fit"]].round(3) \
        .to_dict("records")
    (OUT / "semantic_stats.json").write_text(json.dumps(st, indent=1, default=float))
    Cc.to_csv(OUT / "semantic_concepts.csv", index=False)

    # ---------------- figure: cos vs relatedness (seed 42, pooled over concepts), one panel per config
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), sharex=True, sharey=True)
    out = []
    bins = np.linspace(-0.1, 1.0, 23)
    for ax, cfg in zip(axes, CONFIGS):
        D = pd.concat(pooled[cfg])
        ax.hexbin(D.rel, D.cos, gridsize=45, extent=(-0.1, 1.0, -0.4, 1.0), bins="log", cmap="Greys", mincnt=1,
                  linewidths=0)
        b = np.digitize(D.rel, bins)
        mids, med, q1, q3, ns = [], [], [], [], []
        for k in range(1, len(bins)):
            v = D.cos.values[b == k]
            if len(v) >= 30:
                mids.append((bins[k - 1] + bins[k]) / 2)
                med.append(np.median(v))
                q1.append(np.percentile(v, 25))
                q3.append(np.percentile(v, 75))
                ns.append(len(v))
        ax.plot(mids, med, color=COLORS[cfg], lw=2)
        ax.fill_between(mids, q1, q3, color=COLORS[cfg], alpha=0.25, lw=0)
        out += [{"cfg": cfg, "rel_bin_mid": m, "cos_median": a, "cos_q1": l, "cos_q3": h, "n": n}
                for m, a, l, h, n in zip(mids, med, q1, q3, ns)]
        rho = st["by_cfg"][cfg]["sp_fit"]["mean"]
        ax.text(0.03, 0.97, f"{NAMES[cfg]}\nSpearman {rho:+.2f}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel("relatedness of source words\n(MiniLM cosine)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylim(-0.4, 1.0)
        ax.set_xlim(-0.1, 1.0)
    axes[0].set_ylabel("cos(z_i, z_j), fit space")
    fig.tight_layout()
    savefig(fig, "fig_cos_vs_semantic")
    pd.DataFrame(out).to_csv(OUT / "fig_cos_vs_semantic_data.csv", index=False)
    print(json.dumps(st["by_cfg"], indent=1, default=float)[:6000])
    print(json.dumps(st["paired"], indent=1, default=float)[:5000])
    print(json.dumps(st["pooled_seed42"], indent=1, default=float))


if __name__ == "__main__":
    main()
