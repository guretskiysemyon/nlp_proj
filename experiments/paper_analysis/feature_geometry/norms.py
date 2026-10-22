#!/usr/bin/env python
"""WP2 step 3: the norm puzzle. Where does the concept feature rank in ||z||, raw vs CS, and why?

    .venv/bin/python -m experiments.paper_analysis.feature_geometry.norms

With unit-norm rows of Y, z_i absorbs the scale of feature i. For disjoint supports the ALS solution is the
projection z*_i = A y_i^T = sum_t y_it a_t, and exactly
    log ||z*_i|| = log l1_i + log nbar_i + log kappa_i,
l1 = sum_t |y_it| (<= sqrt(width)), nbar = |y|-weighted mean norm of the support's vectors in the fit space,
kappa = ||sum_t y_it a_t|| / sum_t |y_it| ||a_t|| in [0, 1] (coherence of the support). We check corr(log ||z*||,
log ||z||), decompose the variance of log ||z*|| over the three terms (Cov(term, y) / Var(y), within-run centered),
split the raw ||z||^2 into its background-mean part (z . mu_hat_B)^2 and the rest (mu_split; is the concept
feature's large raw norm its mu component?), regress log ||z|| on support properties (raw token norm, Wikipedia log frequency, width, rho; within-run
standardized), and report the concept feature's within-run z-score on every term.
Token level: Spearman of raw norm ||e_t||, background-whitened norm ||u_t|| and CS norm ||x_t|| with log frequency.
Outputs: norm_concepts.csv, norm_runs.csv, norm_stats.json, fig_norm_percentile.{pdf,png} (+ _data.csv),
fig_norm_terms.{pdf,png} (+ _data.csv).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.paper_analysis.feature_geometry.common import (COLORS, CONFIGS, NAMES, OUT, boot_ci,
                                                                concept_feature, load_features, paired, savefig,
                                                                setup_mpl)

TERMS = ("log_l1", "log_nbar", "log_kappa")


def add_logs(F: pd.DataFrame) -> pd.DataFrame:
    F = F.copy()
    F["log_z"] = np.log(F.znorm_fit)
    F["log_zraw"] = np.log(F.znorm_raw)
    F["log_pz"] = np.log(F.pnorm)
    F["log_l1"] = np.log(F.l1)
    F["log_nbar"] = np.log(F.nbar)
    F["log_kappa"] = np.log(F.kappa)
    F["log_enorm"] = np.log(F.e_norm_w)
    F["log_unorm"] = np.log(F.u_norm_w)
    F["log_width"] = np.log(F.width)
    F["log_rho"] = np.log(F.rho.clip(1e-3, 1e4))
    # raw-space split along the background mean (WP6 lead S3): ||z||^2 = (z . mu_hat)^2 + ||z_perp||^2
    F["z_mu"] = F.znorm_raw * F.cos_mu                     # signed mu coordinate of the sign-aligned raw direction
    F["z_perp"] = F.znorm_raw * np.sqrt((1 - F.cos_mu ** 2).clip(0))
    F["log_zperp"] = np.log(F.z_perp)
    return F


def within_z(F: pd.DataFrame, cols) -> pd.DataFrame:
    g = F.groupby(["concept", "cfg", "seed"])[list(cols)]
    return (F[list(cols)] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ b
    r2 = 1 - res.var() / y.var()
    # cluster-robust SEs would need run clusters; report plain SEs (features within a run are not independent)
    s2 = res @ res / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    return b[1:], float(r2), se[1:]


def main():
    F = add_logs(load_features())
    k = 100
    # ---------------- concept feature rank / percentile
    rows = []
    for (c, cfg, s), g in F.groupby(["concept", "cfg", "seed"], sort=False):
        g = g.reset_index(drop=True)
        zs = within_z(g.assign(concept=c, cfg=cfg, seed=s), ["log_z", "log_zraw", "log_pz", *TERMS, "log_enorm",
                                                              "log_unorm", "logf_w", "log_width"])
        for how in ("lex", "tj", "p"):
            f = concept_feature(g, how)
            if f is None:
                continue
            i = int(np.flatnonzero(g.feature.values == f)[0])
            r = {"concept": c, "cfg": cfg, "seed": s, "how": how, "feature": f, "lo": int(g.lo[i])}
            for col, name in (("znorm_fit", "fit"), ("znorm_raw", "raw"), ("znorm_cs", "cs"), ("pnorm", "proj")):
                v = g[col].values
                r[f"pct_{name}"] = 100.0 * (v < v[i]).sum() / (k - 1)
                r[f"rank_{name}"] = int((v > v[i]).sum()) + 1
                r[f"share_{name}"] = 100.0 * v[i] / v.sum()
            r |= {f"z_{col}": float(zs[col][i]) for col in zs.columns}
            zr, zmu, zp = g.znorm_raw.values, g.z_mu.values, g.z_perp.values
            r |= {"pct_perp": 100.0 * (zp < zp[i]).sum() / (k - 1),
                  "pct_absmu": 100.0 * (np.abs(zmu) < abs(zmu[i])).sum() / (k - 1),
                  "mu_share": float(zmu[i] ** 2 / zr[i] ** 2), "mu_share_med": float(np.median(zmu ** 2 / zr ** 2)),
                  "z_mu": float(zmu[i]), "z_mu_med": float(np.median(zmu)),
                  "excess": float(zr[i] ** 2 - np.mean(zr ** 2)), "excess_mu": float(zmu[i] ** 2 - np.mean(zmu ** 2)),
                  "excess_perp": float(zp[i] ** 2 - np.mean(zp ** 2))}
            r |= {"e_norm_w": g.e_norm_w[i], "logf_w": g.logf_w[i], "f0": g.f0[i], "kappa": g.kappa[i],
                  "nbar": g.nbar[i], "l1": g.l1[i], "width": int(g.width[i]), "u_norm_w": g.u_norm_w[i],
                  "x_norm_w": g.x_norm_w[i], "cos_mu": g.cos_mu[i]}
            rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "norm_runs.csv", index=False)
    C = R.groupby(["how", "concept", "cfg"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    C.to_csv(OUT / "norm_concepts.csv", index=False)
    st = {"concept_feature": {}, "paired": {}}
    for how in ("lex", "tj", "p"):
        sub = C[C.how == how]
        st["concept_feature"][how] = {
            cfg: {m: {"mean": float(sub[sub.cfg == cfg][m].mean()), "median": float(sub[sub.cfg == cfg][m].median()),
                      "ci95": boot_ci(sub[sub.cfg == cfg][m]), "n": int(sub[sub.cfg == cfg][m].notna().sum())}
                  for m in ("pct_fit", "pct_raw", "pct_cs", "share_fit", "z_log_l1", "z_log_nbar", "z_log_kappa",
                            "z_log_enorm", "z_log_unorm", "z_logf_w", "z_log_width")}
            for cfg in CONFIGS}
        st["concept_feature"][how]["share_top10pct"] = {
            cfg: float((sub[sub.cfg == cfg].pct_fit >= 90).mean()) for cfg in CONFIGS}
        for m in ("pct_fit", "pct_raw", "z_log_nbar", "z_log_kappa", "z_log_l1"):
            p = sub.pivot(index="concept", columns="cfg", values=m)
            st["paired"][f"{how}_{m}"] = {f"L0->{b}": paired(p["L0"], p[b]) for b in ("L1", "L12")}
    # per run: fraction of runs where the concept feature is in the top-10 by ||z||
    lex = R[R.how == "lex"]
    st["runs_top10"] = {cfg: {"n_runs": int((lex.cfg == cfg).sum()),
                              "frac_rank_le10": float((lex[lex.cfg == cfg].rank_fit <= 10).mean())} for cfg in CONFIGS}
    # ---- mean split (raw space): is the concept feature's large raw norm its mu_B component?
    st["mu_split"] = {}
    for how in ("lex", "tj"):
        sub = C[C.how == how]
        d = {}
        for cfg in CONFIGS:
            q = sub[sub.cfg == cfg]
            pos = q[q.excess > 0]
            d[cfg] = {"n": len(q), "pct_raw": float(q.pct_raw.mean()), "pct_perp": float(q.pct_perp.mean()),
                      "pct_absmu": float(q.pct_absmu.mean()),
                      "mu_share": float(q.mu_share.mean()), "mu_share_run_median": float(q.mu_share_med.mean()),
                      "z_mu": float(q.z_mu.mean()), "z_mu_run_median": float(q.z_mu_med.mean()),
                      "excess_from_mu_pooled": float(q.excess_mu.sum() / q.excess.sum()),
                      "n_excess_pos": len(pos),
                      "excess_from_mu_median_over_pos": float((pos.excess_mu / pos.excess).median()) if len(pos) else None}
        d["paired_pct_perp"] = {f"L0->{b}": paired(sub.pivot(index="concept", columns="cfg", values="pct_perp")["L0"],
                                                   sub.pivot(index="concept", columns="cfg", values="pct_perp")[b])
                                for b in ("L1", "L12")}
        d["L0_pct_raw_vs_perp"] = paired(sub[sub.cfg == "L0"].pct_raw, sub[sub.cfg == "L0"].pct_perp)
        st["mu_split"][how] = d
    # all features, within run: share of the variance of ||z_raw||^2 carried by the mu part
    st["mu_split"]["all_features"] = {}
    for cfg in CONFIGS:
        Fc = F[F.cfg == cfg]
        z2 = Fc.znorm_raw ** 2
        cen = lambda x: x - x.groupby([Fc.concept, Fc.seed]).transform("mean")  # noqa: E731
        y, m2, p2 = cen(z2), cen(Fc.z_mu ** 2), cen(Fc.z_perp ** 2)
        st["mu_split"]["all_features"][cfg] = {
            "median_mu_share": float((Fc.z_mu ** 2 / z2).median()),
            "var_share_mu": float(np.cov(m2, y)[0, 1] / y.var()), "var_share_perp": float(np.cov(p2, y)[0, 1] / y.var()),
            "frac_features_mu_coord_positive": float((Fc.z_mu > 0).mean())}
    hp = R[(R.concept == "Harry Potter")]
    st["harry_potter"] = hp[["cfg", "seed", "how", "feature", "rank_fit", "pct_fit", "share_fit", "pct_raw", "pct_perp",
                             "pct_absmu", "mu_share", "mu_share_med", "excess", "excess_mu",
                             "z_log_l1", "z_log_nbar", "z_log_kappa", "e_norm_w", "logf_w", "kappa"]].round(3) \
        .to_dict("records")

    # ---------------- decomposition and regression over all features
    st["decomp"], st["regression"] = {}, {}
    Z = within_z(F, ["log_z", "log_pz", *TERMS, "log_enorm", "logf_w", "log_width", "log_rho", "log_unorm"])
    for cfg in CONFIGS:
        m = (F.cfg == cfg).values
        Fc = F[m]
        cen = Fc[["log_pz", *TERMS, "log_z"]] - Fc.groupby(["concept", "seed"])[["log_pz", *TERMS, "log_z"]] \
            .transform("mean")
        y = cen.log_pz
        st["decomp"][cfg] = {"corr_logz_logpz": float(np.corrcoef(cen.log_z, y)[0, 1]),
                             **{f"share_{t}": float(np.cov(cen[t], y)[0, 1] / y.var()) for t in TERMS},
                             "sd_within_run": {t: float(cen[t].std()) for t in ("log_z", *TERMS)}}
        preds = ["log_enorm", "logf_w", "log_rho"] + (["log_width"] if cfg == "L12" else [])
        Zc = Z[m].dropna(subset=["log_z", *preds])
        b, r2, se = ols(Zc.log_z.values, Zc[preds].values)
        st["regression"][cfg] = {"predictors": preds, "beta_std": dict(zip(preds, b.round(3).tolist())),
                                 "se_naive": dict(zip(preds, se.round(3).tolist())), "R2": round(r2, 3),
                                 "n_features": len(Zc)}
        # the same with the mechanism terms (exact for the projection): how much of log ||z|| they carry
        b2, r22, _ = ols(Zc.log_z.values, Z[m].loc[Zc.index, list(TERMS)].values)
        st["regression"][cfg]["terms_R2"] = round(r22, 3)
        st["regression"][cfg]["terms_beta"] = dict(zip(TERMS, b2.round(3).tolist()))
        # which properties drive nbar and kappa
        for t in ("log_nbar", "log_kappa"):
            pr = ["log_enorm", "logf_w"]
            bb, rr, _ = ols(Zc[t].values if t in Zc else Z[m].loc[Zc.index, t].values, Zc[pr].values)
            st["regression"][cfg][f"{t}_on_props"] = {"beta_std": dict(zip(pr, bb.round(3).tolist())), "R2": round(rr, 3)}
        # raw-norm vs frequency at the feature level
        st["regression"][cfg]["spearman_enorm_logf_features"] = float(spearmanr(Fc.e_norm_w, Fc.logf_w).statistic)

    # ---------------- token level
    tok_rows = []
    for p in sorted((OUT / "tokens").glob("*.npz")):
        z = np.load(p)
        wc = np.load(OUT / "wiki_token_counts.npz")
        n = float(wc["n_tokens"])
        lf = np.log10((z["count"] + 1) / n * 1e6)
        tok_rows.append({"concept": p.stem,
                         "sp_e_logf": float(spearmanr(z["e_norm"], lf).statistic),
                         "sp_u_logf": float(spearmanr(z["u_norm"], lf).statistic),
                         "sp_x_logf": float(spearmanr(z["x_norm"], lf).statistic),
                         "sp_e_u": float(spearmanr(z["e_norm"], z["u_norm"]).statistic),
                         "e_C": float(z["e_norm"][z["codes"] == 1].mean()), "e_N": float(z["e_norm"][z["codes"] == 0].mean()),
                         "u_C": float(z["u_norm"][z["codes"] == 1].mean()), "u_N": float(z["u_norm"][z["codes"] == 0].mean()),
                         "x_C": float(z["x_norm"][z["codes"] == 1].mean()), "x_N": float(z["x_norm"][z["codes"] == 0].mean()),
                         "logf_C": float(lf[z["codes"] == 1].mean()), "logf_N": float(lf[z["codes"] == 0].mean()),
                         "cv_e": float(z["e_norm"].std() / z["e_norm"].mean()),
                         "cv_u": float(z["u_norm"].std() / z["u_norm"].mean()),
                         "cv_x": float(z["x_norm"].std() / z["x_norm"].mean())})
    T = pd.DataFrame(tok_rows)
    T.to_csv(OUT / "norm_tokens.csv", index=False)
    st["tokens"] = {m: {"mean": float(T[m].mean()), "ci95": boot_ci(T[m])} for m in T.columns if m != "concept"}
    # union of all V* tokens: binned raw norm by frequency decile
    allt = {}
    for p in sorted((OUT / "tokens").glob("*.npz")):
        z = np.load(p)
        for t, e, u, c in zip(z["vstar"], z["e_norm"], z["u_norm"], z["count"]):
            allt[int(t)] = (e, u, c)
    A = np.array(list(allt.values()))
    lf = np.log10((A[:, 2] + 1) / n * 1e6)
    dec = pd.qcut(lf, 10, labels=False, duplicates="drop")
    st["tokens_union"] = {"n_tokens": len(A), "sp_e_logf": float(spearmanr(A[:, 0], lf).statistic),
                          "sp_u_logf": float(spearmanr(A[:, 1], lf).statistic),
                          "e_norm_by_logf_decile": pd.Series(A[:, 0]).groupby(dec).mean().round(2).tolist(),
                          "u_norm_by_logf_decile": pd.Series(A[:, 1]).groupby(dec).mean().round(2).tolist(),
                          "logf_decile_edges": np.percentile(lf, np.arange(0, 101, 10)).round(2).tolist()}
    (OUT / "norm_stats.json").write_text(json.dumps(st, indent=1, default=float))

    # ---------------- figure 1: concept-feature ||z|| percentile per concept (lex identification)
    plt = setup_mpl()
    sub = C[C.how == "lex"].pivot(index="concept", columns="cfg", values="pct_fit")
    sub = sub.sort_values("L0", na_position="first")
    fig, ax = plt.subplots(figsize=(3.3, 5.0))
    y = np.arange(len(sub))
    mk = {"L0": "o", "L1": "s", "L12": "^"}
    for cfg in CONFIGS:
        ax.scatter(sub[cfg], y, s=14, marker=mk[cfg], color=COLORS[cfg], label=NAMES[cfg], zorder=3,
                   edgecolors="white", linewidths=0.4)
    for i, (_, r) in enumerate(sub.iterrows()):
        v = r[list(CONFIGS)].dropna()
        if len(v):
            ax.plot([v.min(), v.max()], [i, i], color="#BBBBBB", lw=0.6, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub.index, fontsize=6)
    ax.set_xlabel("percentile of the concept feature's ||z||\namong the k = 100 features (mean over seeds)")
    ax.axvline(50, color="k", lw=0.5, alpha=0.4)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.7, len(sub) - 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.06), ncol=3, frameon=False, fontsize=6, handletextpad=0.2,
              columnspacing=0.8)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    savefig(fig, "fig_norm_percentile")
    sub.reset_index().to_csv(OUT / "fig_norm_percentile_data.csv", index=False)

    # ---------------- figure 2: the concept feature's within-run z-score of each term
    lexc = C[C.how == "lex"]
    terms = [("z_log_l1", "Σ|y|"), ("z_log_nbar", "token\nnorm"), ("z_log_kappa", "coherence\nκ"), ("z_log_z", "||z||")]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    out = []
    w = 0.26
    for j, cfg in enumerate(CONFIGS):
        vals = [lexc[lexc.cfg == cfg][t] for t, _ in terms]
        means = [v.mean() for v in vals]
        cis = [boot_ci(v) for v in vals]
        x = np.arange(len(terms)) + (j - 1) * w
        ax.bar(x, means, width=w * 0.9, color=COLORS[cfg], label=NAMES[cfg])
        ax.errorbar(x, means, yerr=[[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]],
                    fmt="none", ecolor="k", lw=0.7, capsize=1.5)
        out += [{"cfg": cfg, "term": t, "mean_z": m, "ci_lo": c[0], "ci_hi": c[1]} for (t, _), m, c in
                zip(terms, means, cis)]
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(np.arange(len(terms)))
    ax.set_xticklabels([n for _, n in terms], fontsize=7)
    ax.set_ylabel("concept feature,\nwithin-run z-score")
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2,
              columnspacing=0.8)
    fig.tight_layout()
    savefig(fig, "fig_norm_terms")
    pd.DataFrame(out).to_csv(OUT / "fig_norm_terms_data.csv", index=False)
    print(json.dumps({k: st[k] for k in ("concept_feature", "runs_top10")}, indent=1, default=float)[:7000])
    print(json.dumps(st["mu_split"], indent=1, default=float))
    print(json.dumps({k: st[k] for k in ("decomp", "regression", "tokens_union")}, indent=1, default=float))
    print(json.dumps(st["tokens"], indent=1, default=float))


if __name__ == "__main__":
    main()
