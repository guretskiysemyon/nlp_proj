"""WP4 figures (PDF + PNG, data as CSV next to each).

    .venv/bin/python -m experiments.paper_analysis.robustness.figures [A] [B]

fig_A_rank:   per concept (rows; abstract S300, abstract S190, entity S300, entity S190) the rho rank of the best
              blind-labelled P feature in each run (5 seeds), EMBER / CS / CS+GS; "none" = no P feature in the top 10.
fig_A_summary: share of runs with a P feature at rank 1 / <= 3 / <= 10, abstract vs entity, per condition.
fig_B_curves: contamination: P feature presence (top 10), its rho rank and intruder share vs f, per concept;
              adjacent pages vs the unrelated-page control.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.paper_analysis.robustness.concepts import (ABSTRACT_NEW, ABSTRACT_OLD, ABSTRACT_SHORT, ADJACENT,  # noqa: E402
                                                            ENTITY, OUT)

COL = {"EMBER": "#7F7F7F", "CS": "#0072B2", "CSGS": "#D55E00", "CS_sc": "#009E73"}
NAME = {"EMBER": "EMBER (raw, RS-1%)", "CS": "CS + RS-1%", "CSGS": "CS + GS (CINDER)", "CS_sc": "CS, target = S_C"}
MARK = {"EMBER": "o", "CS": "s", "CSGS": "^", "CS_sc": "D"}
LS = {"EMBER": "-", "CS": "-", "CSGS": "--", "CS_sc": ":"}
FIG = OUT / "figures"

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def save(fig, name: str, data: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"-> {FIG / name}.pdf/.png/.csv")


def fig_A():
    R = pd.read_csv(OUT / "runs_A.csv")
    rows = ([(c, 300, "abstract") for c in ABSTRACT_NEW + ABSTRACT_OLD] + [(c, 190, "abstract") for c in ABSTRACT_SHORT]
            + [(c, 300, "entity") for c in ENTITY] + [(c, 190, "entity") for c in ENTITY])
    conds = ["EMBER", "CS", "CSGS"]
    NONE = 16
    fig, ax = plt.subplots(figsize=(3.3, 5.6))
    y, yt, yl, seps, starts = 0, [], [], [], []
    rng = np.random.default_rng(0)
    plot_rows = []
    prev = None
    for c, S, grp in rows:
        if prev is not None and (grp, S) != prev:
            seps.append(y + 0.35)
            y += 0.7
        if prev != (grp, S):
            starts.append((y, grp, S))
        prev = (grp, S)
        for k, cond in enumerate(conds):
            g = R[(R.concept == c) & (R.S == S) & (R.cond == cond)]
            yy = y + (k - 1) * 0.27
            xs = g.P_rank.fillna(NONE).to_numpy()
            # identical ranks side by side (seeds), small horizontal dodge in log space
            vals, counts = np.unique(xs, return_counts=True)
            for v, n in zip(vals, counts):
                offs = (np.arange(n) - (n - 1) / 2) * (0.03 if v == NONE else 0.045)
                ax.scatter(v * 10 ** offs, np.full(n, yy), s=9, marker=MARK[cond], color=COL[cond],
                           linewidths=0, alpha=0.9, zorder=3)
            for r in g.itertuples():
                plot_rows.append({"concept": c, "S": S, "group": grp, "cond": cond, "seed": r.seed,
                                  "P_rank": r.P_rank, "plotted_x": r.P_rank if pd.notna(r.P_rank) else "none"})
        yt.append(y)
        yl.append(f"{c}" + (" (190)" if S == 190 else ""))
        y += 1
    for s in seps:
        ax.axhline(s, color="#BBBBBB", lw=0.5, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, NONE])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "none"])
    ax.minorticks_off()
    ax.axvline(12.5, color="#BBBBBB", lw=0.5, ls=":")
    ax.set_xlim(0.8, NONE * 1.25)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\rho$ rank of the best concept-specific (P) feature")
    for xg in (1, 2, 3, 5, 10):
        ax.axvline(xg, color="#EEEEEE", lw=0.5, zorder=0)
    handles = [plt.Line2D([], [], marker=MARK[c], color=COL[c], ls="", ms=4, label=NAME[c]) for c in conds]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1.06), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    # group labels on the right
    ends = [s for s in seps] + [y - 0.5]
    for (y0, grp, S), y1 in zip(starts, ends):
        ax.text(1.02, (y0 - 0.4 + y1) / 2, f"{grp}\n" + r"$|S_C|$" + f" = {S}", transform=ax.get_yaxis_transform(),
                rotation=270, ha="left", va="center", fontsize=7, color="#555555")
    save(fig, "fig_A_rank", pd.DataFrame(plot_rows))

    # summary: share of runs with P at rank 1 / <= 3 / <= 10 per group x condition (S = 300 and 190 pooled)
    C = pd.read_csv(OUT / "concepts_A.csv")
    summ = []
    for grp in ("abstract", "entity"):
        for cond in conds:
            g = C[(C.group == grp) & (C.cond == cond)]
            for m in ("P@1", "P@3", "P@10"):
                summ.append({"group": grp, "cond": cond, "metric": m, "mean_over_concepts": g[m].mean(),
                             "n_concepts_x_S": len(g)})
    Sm = pd.DataFrame(summ)
    fig, axes = plt.subplots(1, 3, figsize=(3.3, 1.5), sharey=True)
    for ax, m in zip(axes, ("P@1", "P@3", "P@10")):
        for gi, grp in enumerate(("abstract", "entity")):
            for k, cond in enumerate(conds):
                v = Sm[(Sm.group == grp) & (Sm.cond == cond) & (Sm.metric == m)].mean_over_concepts.iloc[0]
                ax.bar(gi + (k - 1) * 0.27, v, width=0.25, color=COL[cond], linewidth=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["abstract", "entity"])
        ax.set_title({"P@1": "rank 1", "P@3": r"rank $\leq$ 3", "P@10": r"rank $\leq$ 10"}[m], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, 1.5)
    axes[0].set_ylabel("share of runs with a P feature")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c], label=NAME[c]) for c in conds]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False,
               handlelength=1, columnspacing=0.8)
    save(fig, "fig_A_summary", Sm)


def fig_B():
    """Rows: share of seeds with a P feature in the top 10; mean rho rank of the best P feature (none = 11); intruder
    share of its 20 strongest tokens. Solid: adjacent-page contamination; dashed / open: unrelated-page control."""
    K = pd.read_csv(OUT / "curves_B.csv")
    R = pd.read_csv(OUT / "runs_B.csv")
    concepts = list(ADJACENT)
    lines = [("adjacent", "EMBER"), ("adjacent", "CS"), ("adjacent", "CSGS"), ("unrelated", "EMBER"),
             ("unrelated", "CS")]
    metrics = [("P@10", "seeds with a P feature\nin the top 10", (-0.05, 1.05)),
               ("P_rank_mean_cap11", r"$\rho$ rank of best P" + "\n(mean; none = 11)", (11.5, 0.5)),
               ("P_intr", "intruder share of the\nP feature (top 20)", (-0.02, 0.45))]
    fig, axes = plt.subplots(len(metrics), len(concepts), figsize=(6.8, 4.8), sharex=True, squeeze=False)
    data = []
    for j, c in enumerate(concepts):
        for i, (m, lab, ylim) in enumerate(metrics):
            ax = axes[i][j]
            for k, (kind, cond) in enumerate(lines):
                g = K[(K.concept == c) & (K.cond == cond) & (K.kind == kind)].sort_values("f")
                if m == "P_intr":   # mean over runs that have a P feature; f = 0 is 0 by definition
                    rr = R[(R.concept == c) & (R.cond == cond) & (R.kind == kind)]
                    y = rr.groupby("f").P_intr.mean().reindex(g.f).to_numpy()
                else:
                    y = g[m].to_numpy()
                f = g.f.to_numpy()
                dx = (k - 2) * 0.006
                un = kind == "unrelated"
                ax.plot(f + dx, y, "--" if un else "-", color=COL[cond], lw=1.0 if un else 1.3, marker=MARK[cond],
                        ms=3.5, mfc="white" if un else COL[cond], alpha=0.85 if un else 1.0,
                        label=f"{NAME[cond]}, {'unrelated' if un else 'adjacent'}")
                for ff, yy in zip(f, y):
                    data.append({"concept": c, "metric": m, "kind": kind, "cond": cond, "f": ff, "value": yy})
            ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(c, fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel(r"replaced fraction $f$ of $S_C$")
                ax.set_xticks([0, 0.1, 0.25, 0.5])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, frameon=False, columnspacing=1.0)
    fig.tight_layout(h_pad=0.5, w_pad=0.8)
    save(fig, "fig_B_curves", pd.DataFrame(data))


def main():
    which = sys.argv[1:] or ["A", "B"]
    if "A" in which:
        fig_A()
    if "B" in which:
        fig_B()


if __name__ == "__main__":
    main()
