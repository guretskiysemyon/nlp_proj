"""Figures (PDF + PNG, plotted data as CSV next to each) for WP1 quiet_concept.

    python -m experiments.paper_analysis.quiet_concept.figures

fig_rho_rank          rho rank of the judge-free concept feature (max lexicon hits, >= 5 to count as found; not found
                      plotted at 101) per concept (median over 5 seeds): raw / centred / whiten-only / CS / CS+GS
fig_direction_rank    PCA (energy) rank of selection-free concept directions per concept, raw vs CS (+ whiten-only)
fig_purity_value      fixed point of a single feature seeded on the concept's lexicon tokens: purity (share of
                      concept-only tokens) vs value (percentile among the fit's feature energies), per matrix
fig_svd_rank          truncated-SVD rank that matches the sparse-MF error, raw vs CS, per concept
fig_frequency         Wikipedia log-frequency vs embedding statistics (vocabulary bins; C vs N per concept)
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.paper_analysis.quiet_concept.common import COLORS, OUT  # noqa: E402

FIG = OUT / "figures"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
C = dict(COLORS)
C["ctr"] = "#B0B0B0"
C["rawp"] = "#4D4D4D"
NAMES = {"raw": "raw\n(EMBER)", "rawp": "raw\n⊥μ_B", "ctr": "centred", "wo": "whiten\nonly", "cs": "CS",
         "gs": "CS+GS"}
SHORT = {"raw": "raw", "rawp": "⊥μ̂", "ctr": "ctr", "wo": "wo", "cs": "CS", "gs": "GS"}
MARK = {"raw": "o", "rawp": "P", "ctr": "D", "wo": "s", "cs": "^", "gs": "v"}


def save(fig, name: str, data: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)


def slope(ax, D: pd.DataFrame, cols: list[str], log: bool = True, jitter: float = 0.06, highlight=None,
          lo=None, hi=None):
    rng = np.random.default_rng(0)
    x = np.arange(len(cols))
    for _, r in D.iterrows():
        jx = x + rng.uniform(-jitter, jitter, len(cols))
        hl = highlight is not None and r["concept"] == highlight
        ax.plot(jx, r[cols].values.astype(float), color="#000000" if hl else "#C8C8C8", lw=1.0 if hl else 0.5,
                zorder=3 if hl else 1)
        for i, c in enumerate(cols):
            ax.scatter(jx[i], r[c], s=9, color=C[c], marker=MARK[c], edgecolor="white", linewidth=0.3,
                       zorder=2 if not hl else 4)
    med = D[cols].median()
    ax.plot(x, med.values, color="black", lw=1.4, zorder=5)
    for i, c in enumerate(cols):
        ax.scatter(x[i], med[c], s=36, color=C[c], marker=MARK[c], edgecolor="black", linewidth=0.8, zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[c] for c in cols])
    ax.set_xlim(-0.4, len(cols) - 0.6)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color="#EEEEEE", lw=0.5)


def fig_rho_rank():
    """Runs (38 concepts x 5 seeds per matrix) by the rho rank of their judge-free concept feature."""
    F = pd.read_csv(OUT / "tables" / "features_all_runs.csv")
    cols = [c for c in ("raw", "rawp", "ctr", "wo", "cs", "gs") if c in set(F.cond)]
    cats = [("1", lambda r: r == 1), ("2–3", lambda r: (r >= 2) & (r <= 3)), ("4–20", lambda r: (r >= 4) & (r <= 20)),
            (">20", lambda r: (r > 20) & (r <= 100)), ("none", lambda r: r > 100)]
    shades = ["#08306B", "#2171B5", "#6BAED6", "#C6DBEF", "#E5E5E5"]
    rows = []
    for c in cols:
        rk = F[F.cond == c].rank_c.values if "rank_c" in F else None
        if rk is None:
            G = F[F.cond == c]
            rk = np.where(G.hits >= 5, G.rho_rank, 101)
        rows.append({"cond": c, "n_runs": len(rk), **{k: float(fn(rk).mean()) for k, fn in cats}})
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.3, 0.35 * len(cols) + 0.3))
    y = np.arange(len(cols))[::-1]
    left = np.zeros(len(cols))
    for (k, _), col in zip(cats, shades):
        ax.barh(y, D[k], left=left, color=col, edgecolor="white", linewidth=1.0, height=0.7, label=k)
        for yi, l, w in zip(y, left, D[k]):
            if w >= 0.07:
                ax.text(l + w / 2, yi, f"{w * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if col in shades[:2] else "#333333")
        left += D[k].values
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES[c].replace("\n", " ") for c in cols])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of runs (%) by ρ rank of the concept feature")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), handlelength=1.0,
              columnspacing=0.8, title=None)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_rho_rank", D)


def fig_feature_energy():
    """Norm puzzle across concepts: percentile of the concept feature's ||z||^2 among its fit's 100 features, and the
    number of neutral-only tokens in its support (median over seeds, found runs and not)."""
    A = pd.read_csv(OUT / "tables" / "feature_rank_by_concept_cond.csv")
    cols = [c for c in ("raw", "rawp", "ctr", "wo", "cs", "gs") if c in set(A.cond)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), gridspec_kw={"wspace": 0.3})
    out = None
    for ax, v, lab, log in ((axes[0], "z2_pct_med", "‖z‖² percentile (concept feature)", False),
                            (axes[1], "supp_N", "neutral-only tokens in its support", False)):
        W = A.pivot(index="concept", columns="cond", values=v).reset_index()
        slope(ax, W, cols, log=log, highlight="Harry Potter")
        ax.set_xticklabels([SHORT[c] for c in cols])
        ax.set_ylabel(lab)
        W = W[["concept"] + cols].rename(columns={c: f"{v}_{c}" for c in cols})
        out = W if out is None else out.merge(W, on="concept")
    axes[0].set_ylim(-3, 103)
    save(fig, "fig_feature_energy", out)


def fig_direction_rank():
    T = pd.read_csv(OUT / "tables" / "spectra_by_concept.csv")
    dirs = [("md", "concept − neutral mean"), ("ge", "generalized eigenvector"), ("lex", "lexicon tokens' top direction")]
    dirs = [d for d in dirs if f"raw_{d[0]}_pca_rank" in T]
    fig, axes = plt.subplots(1, len(dirs), figsize=(6.8, 2.3), sharey=True)
    out = T[["concept"]].copy()
    for ax, (d, lab) in zip(np.atleast_1d(axes), dirs):
        cols = [c for c in ["raw", "rawp", "ctr", "wo", "cs"] if f"{c}_{d}_pca_rank" in T]
        D = pd.DataFrame({"concept": T.concept, **{c: T[f"{c}_{d}_pca_rank"] for c in cols}})
        slope(ax, D, cols, highlight="Harry Potter")
        ax.set_xticklabels([SHORT[c] for c in cols])
        ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
        ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "200", "500"])
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        for c in cols:
            out[f"{d}_{c}_pca_rank"] = D[c]
            out[f"{d}_{c}_share"] = T[f"{c}_{d}_share"]
    np.atleast_1d(axes)[0].set_ylabel("PCA rank of the direction's energy")
    save(fig, "fig_direction_rank", out)


def fig_purity_value():
    T = pd.read_csv(OUT / "tables" / "spectra_by_concept.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    out = T[["concept"]].copy()
    for sp in ("raw", "rawp", "ctr", "wo", "cs"):
        if f"{sp}_seeded_value_pct_{sp}" not in T:
            continue
        x, y = T[f"{sp}_seeded_purity_C"], T[f"{sp}_seeded_value_pct_{sp}"]
        out[f"{sp}_purity"], out[f"{sp}_value_pct"] = x, y
        ax.scatter(x, y, s=12, color=C[sp], marker=MARK[sp], edgecolor="white", linewidth=0.3,
                   label=NAMES[sp].replace("\n", " "), zorder=2)
        ax.scatter(x.median(), y.median(), s=60, color=C[sp], marker=MARK[sp], edgecolor="black", linewidth=0.9,
                   zorder=4)
    hp = T[T.concept == "Harry Potter"]
    if len(hp):
        xs = [float(hp[f"{sp}_seeded_purity_C"].iloc[0]) for sp in ("raw", "ctr", "wo", "cs")]
        ys = [float(hp[f"{sp}_seeded_value_pct_{sp}"].iloc[0]) for sp in ("raw", "ctr", "wo", "cs")]
        ax.plot(xs, ys, color="black", lw=0.7, zorder=3)
        ax.annotate("HP", (xs[-1], ys[-1]), xytext=(4, 2), textcoords="offset points", fontsize=7)
    ax.set_xlabel("purity: concept-only share of the m tokens")
    ax.set_ylabel("value: percentile among fit features")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-3, 103)
    ax.grid(color="#EEEEEE", lw=0.5)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.2)
    save(fig, "fig_purity_value", out)


def fig_svd_rank():
    T = pd.read_csv(OUT / "tables" / "spectra_by_concept.csv")
    cols = [c for c in ("raw", "rawp", "ctr", "wo", "cs") if f"svdrank_{c}" in T]
    D = pd.DataFrame({"concept": T.concept, **{c: T[f"svdrank_{c}"] for c in cols}})
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    slope(ax, D, cols, highlight="Harry Potter")
    ax.set_xticklabels([SHORT[c] for c in cols])
    ax.set_yticks([2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_ylabel("SVD rank matching sparse-MF error")
    save(fig, "fig_svd_rank", D)


def fig_frequency():
    V = pd.read_csv(OUT / "frequency" / "vocab_sample.csv")
    P = pd.read_csv(OUT / "frequency" / "per_concept.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1), gridspec_kw={"wspace": 0.45})
    bins = np.arange(-1.5, 4.6, 0.5)
    V["b"] = pd.cut(V.lf, bins)
    g = V.groupby("b", observed=True)
    mid = np.array([iv.mid for iv in g.size().index])
    out = pd.DataFrame({"lf_bin_mid": mid, "n": g.size().values})
    for ax, k, lab in ((axes[0], "mu_proj", "projection on μ̂_B"), (axes[1], "frac10", "top-10 background-PC share")):
        q = g[k].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(mid, q[0.25], q[0.75], color="#7F7F7F", alpha=0.25, lw=0)
        ax.plot(mid, q[0.5], color="#7F7F7F", lw=1.2)
        ax.set_xlabel("log10 frequency per million")
        ax.set_ylabel(lab)
        ax.grid(color="#EEEEEE", lw=0.5)
        out[f"{k}_q25"], out[f"{k}_q50"], out[f"{k}_q75"] = q[0.25].values, q[0.5].values, q[0.75].values
    ax = axes[2]
    ax.scatter(P.lf_N, P.lf_C, s=10, color=C["cs"], edgecolor="white", linewidth=0.3)
    lim = [min(P.lf_N.min(), P.lf_C.min()) - 0.05, max(P.lf_N.max(), P.lf_C.max()) + 0.05]
    ax.plot(lim, lim, color="#999999", lw=0.6, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("neutral-only: median log10 f")
    ax.set_ylabel("concept-only: median log10 f")
    ax.grid(color="#EEEEEE", lw=0.5)
    save(fig, "fig_frequency", out)
    P[["concept", "lf_C", "lf_N", "frac10_C", "frac10_N", "nr_C", "nr_N", "mu_proj_C", "mu_proj_N"]].to_csv(FIG / "fig_frequency_concepts.csv",
                                                                               index=False)


def main():
    for f in (fig_rho_rank, fig_feature_energy, fig_direction_rank, fig_purity_value, fig_svd_rank, fig_frequency):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # keep going: report which figure failed
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
