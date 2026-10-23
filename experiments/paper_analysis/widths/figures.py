#!/usr/bin/env python
"""WP3 figures (PDF + PNG + plotted data as CSV next to each).

    .venv/bin/python -m experiments.paper_analysis.widths.figures

(a) fig_precision_ranking: windowed precision of page-enriched tokens (X20) along the dense ranking of the concept
    direction, CS RS-1% fit (blue) and CS GS fit (orange), mean over 5 seeds, for 4 example concepts; RS-1% cut
    (grey dashed), GS width (orange), natural tail of the RS-1% fit (blue tick); base rate dotted.
(b) fig_tail_vs_width: natural tail T (RS-1% fit, anchor direction) vs width per concept: GS (orange, +-1 SD over
    seeds), RS-1% (grey); identity line.
(c) fig_gs_needs_cs: neutral-only share of the concept feature per concept for the four fits; and, for GS fits,
    the neutral-only share of the head (first m_RS ranks) vs the extra ranks GS adds beyond m_RS.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.paper_analysis.widths.analyze import PRIMARY  # noqa: E402
from experiments.paper_analysis.widths.common import COLORS, OUT  # noqa: E402

EXAMPLES = ["Baseball", "Harry Potter", "Ancient Rome", "Culture of Greece"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "font.family": "DejaVu Sans"})


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Forward-window mean at position m (ranks m+1..m+w), the window of the natural-tail rule."""
    c = np.concatenate([[0], np.cumsum(x)])
    m = np.arange(len(x) - w + 1)
    return (c[m + w] - c[m]) / w


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a(curves: dict, T: pd.DataFrame, C: pd.DataFrame):
    w = PRIMARY["w"]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.85), sharey=True)
    rows = []
    xmax = 200
    for ax, c in zip(axes, EXAMPLES):
        t = T.loc[c]
        base = float(C.set_index("concept").loc[c, "base_X20"])
        for cfg, ls in (("L1", "-"), ("L12", "-")):
            y = smooth(curves[(c, cfg, "X20")], w)[:xmax]
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, color=COLORS[cfg], lw=1.2, ls=ls, label={"L1": "CS, RS-1%", "L12": "CS, GS (CINDER)"}[cfg])
            rows += [{"concept": c, "fit": cfg, "rank_start": int(i), "precision_X20_next20": float(v)} for i, v in zip(x, y)]
        ax.axhline(base, color="#999999", lw=0.6, ls=":")
        ax.axhline(PRIMARY["theta"], color="#BBBBBB", lw=0.5, ls="-", zorder=0)
        ax.axvline(t.m_rs, color=COLORS["L0"], lw=0.9, ls="--")
        ax.axvline(t.w_L12, color=COLORS["L12"], lw=0.9, ls="-.")
        ax.plot([t.T_L1], [1.02], marker="v", color=COLORS["L1"], ms=4, clip_on=False)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("dense rank")
        ax.set_title(c, fontsize=7.5, loc="left", pad=7)
    axes[0].set_ylabel("page-enriched share\n(next 20 tokens)")
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=COLORS["L0"], ls="--", lw=0.9), plt.Line2D([], [], color=COLORS["L12"], ls="-.", lw=0.9),
          plt.Line2D([], [], color=COLORS["L1"], marker="v", ls="", ms=4), plt.Line2D([], [], color="#999999", ls=":", lw=0.6)]
    l += ["RS-1% cut", "GS width", "natural tail (RS-1% fit)", "base rate"]
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12), handlelength=1.6,
               columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_precision_ranking")
    pd.DataFrame(rows).to_csv(OUT / "fig_precision_ranking.csv", index=False)
    T.loc[EXAMPLES, ["m_rs", "w_L12", "T_L1", "T_L12"]].to_csv(OUT / "fig_precision_ranking_markers.csv")


def fig_b(T: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    lim = max(T.T_L1.max(), T.w_L12.max()) * 1.08
    ax.plot([0, lim], [0, lim], color="#CCCCCC", lw=0.7, zorder=0)
    ax.scatter(T.T_L1, T.m_rs, s=12, marker="s", color=COLORS["L0"], label="RS-1% width", zorder=2, lw=0)
    ax.errorbar(T.T_L1, T.w_L12, yerr=T.w_L12_sd, fmt="o", ms=3.5, color=COLORS["L12"], ecolor=COLORS["L12"],
                elinewidth=0.6, alpha=0.9, label="GS width (±1 SD seeds)", zorder=3)
    for c in EXAMPLES:
        ax.annotate(c, (T.loc[c, "T_L1"], T.loc[c, "w_L12"]), fontsize=6, xytext=(3, 3), textcoords="offset points",
                    color="#333333")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("natural tail T (tokens; RS-1% fit, dense re-solve)")
    ax.set_ylabel("width of the concept feature (tokens)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_tail_vs_width")
    T[["T_L1", "T_L1_sd", "m_rs", "w_L12", "w_L12_sd", "T_L12", "T_RS2"]].to_csv(OUT / "fig_tail_vs_width.csv")


def fig_c(per: pd.DataFrame):
    """Left: neutral-only share of each fit's concept feature, per concept (dots) and median (bar).
    Right: GS fits only, neutral-only share of the first m_RS dense ranks of the support (head) vs the ranks GS adds
    beyond m_RS (extra), per concept with an extra part (lines pair a concept)."""
    W = per.pivot(index="concept", columns="config", values="N")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.3, 2.0), gridspec_kw={"width_ratios": [1.5, 1]})
    cfgs = [("L0", "raw\nRS-1%"), ("rnP", "raw\nGS"), ("L1", "CS\nRS-1%"), ("L12", "CS\nGS")]
    rng = np.random.default_rng(0)
    rows = []
    for i, (cfg, lab) in enumerate(cfgs):
        v = W[cfg].values
        a1.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=COLORS[cfg], lw=0, alpha=0.8, zorder=2)
        a1.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="#222222", lw=1.2, zorder=3)
        rows += [{"panel": "neutral share", "config": cfg, "concept": c, "value": float(x)} for c, x in zip(W.index, v)]
    a1.set_xticks(range(len(cfgs)))
    a1.set_xticklabels([l for _, l in cfgs], fontsize=6.5)
    a1.set_ylabel("neutral-only share of\nthe concept feature")
    a1.set_ylim(-0.01, 0.36)
    E = per.pivot(index="concept", columns="config", values="extra_N")
    H = per.pivot(index="concept", columns="config", values="head_N")
    for j, cfg in enumerate(("rnP", "L12")):
        ok = E[cfg].notna()
        x0, x1 = 2 * j, 2 * j + 1
        for c in E.index[ok]:
            a2.plot([x0, x1], [H.loc[c, cfg], E.loc[c, cfg]], color=COLORS[cfg], lw=0.4, alpha=0.5)
            rows.append({"panel": "head vs extra", "config": cfg, "concept": c, "head": float(H.loc[c, cfg]),
                         "extra": float(E.loc[c, cfg])})
        a2.plot([x0, x1], [H[cfg][ok].median(), E[cfg][ok].median()], color="#222222", lw=1.2, marker="o", ms=2.5)
    a2.set_xticks([0, 1, 2, 3])
    a2.set_xticklabels(["head", "extra", "head", "extra"], fontsize=6.5)
    a2.text(0.5, -0.30, "raw GS", ha="center", transform=a2.get_xaxis_transform(), fontsize=6.5, color=COLORS["rnP"])
    a2.text(2.5, -0.30, "CS GS", ha="center", transform=a2.get_xaxis_transform(), fontsize=6.5, color=COLORS["L12"])
    a2.set_ylim(-0.01, 0.6)
    a2.set_ylabel("neutral-only share", fontsize=7)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_gs_needs_cs")
    pd.DataFrame(rows).to_csv(OUT / "fig_gs_needs_cs.csv", index=False)


def main():
    curves = pd.read_pickle(OUT / "curves.pkl")
    T = pd.read_csv(OUT / "natural_tail_per_concept.csv", index_col=0)
    C = pd.read_csv(OUT / "concepts.csv")
    per = pd.read_csv(OUT / "gs_needs_cs_per_concept.csv")
    fig_a(curves, T, C)
    fig_b(T)
    fig_c(per)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
