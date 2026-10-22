"""Appendix H figure: extrapolation beyond the sample (WP5), restyled from the saved plot data.

(a)-(d) precision@k among the new concept-only tokens (share labelled concept-specific, blind labels), balanced
        condition, for the four scores, one line per dictionary (mean over 17 concepts; band: 95% bootstrap CI);
        dotted: random order.
(e)     concept-specific share by occurrences on the concept's page: in-sample support tokens (solid) and new tokens
        admitted by the 1% re-solve (dashed), for EMBER and CINDER; dotted: all new page tokens. Bins with fewer
        than 5 concepts are not drawn (as in the saved figure).

Data (read only): results/paper_analysis/extrapolation/figures/fig_precision_at_k.json,
                  results/paper_analysis/extrapolation/figures/fig_precision_vs_pagefreq.json

    .venv/bin/python paper/figures/src/fig_extrapolation.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results/paper_analysis/extrapolation/figures"
GREY, BLUE, ORANGE, INK, MUTED, HAIR = "#7F7F7F", "#0072B2", "#D55E00", "#222222", "#666666", "#BBBBBB"
DICTS = [("EMBER", "EMBER (raw, RS)", GREY, "-", "o"), ("CS", "CS + RS", BLUE, "--", "s"),
         ("CINDER", "CINDER (CS + GS)", ORANGE, "-", "^")]
SCORES = [("proj", "projection"), ("cos", "cosine"), ("coef", "joint LS coefficient"), ("share", "code share")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(HAIR)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)


def main():
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    pk = json.loads((SRC / "fig_precision_at_k.json").read_text())
    pf = json.loads((SRC / "fig_precision_vs_pagefreq.json").read_text())
    fig = plt.figure(figsize=(6.8, 1.95))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.18, 1.45], wspace=0.12,
                          left=0.06, right=0.99, top=0.88, bottom=0.33)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    base = pk["balanced|base"]
    for i, (ax, (sc, name)) in enumerate(zip(axes, SCORES)):
        for D, label, color, ls, mk in DICTS:
            d = pk[f"balanced|{sc}|{D}"]
            k = np.array(d["k"])
            ax.fill_between(k, d["ci_lo"], d["ci_hi"], color=color, alpha=0.10, lw=0)
            ax.plot(k, d["mean"], color=color, ls=ls, lw=1.1, marker=mk, ms=3, markevery=[0, 4, 9, 14, 19],
                    mec="white", mew=0.4, label=label)
        ax.axhline(base, color=INK, ls=":", lw=0.8, label="random order")
        ax.set_ylim(0, 0.6)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_title(f"({'abcd'[i]}) {name}", fontsize=7.5, color=INK, loc="left", pad=2)
        ax.set_xlabel("$k$ (top-$k$ new tokens)", fontsize=7, color=INK)
        style(ax)
        if i:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("precision@$k$", fontsize=7.5, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower left", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.06, 0.0),
               handlelength=2.4, columnspacing=1.2)

    ax = fig.add_subplot(gs[0, 5])
    bins = pf["distribution"]["bins"]
    x = np.arange(len(bins))
    for D, color, mk in (("EMBER", GREY, "o"), ("CINDER", ORANGE, "^")):
        for kind, ls, filled, lab in (("in-sample support", "-", True, "in-sample"),
                                      ("new admitted (frac)", "--", False, "new, admitted")):
            d = pf[f"{D}|{kind}"]
            ok = np.array(d["plotted (>=5 concepts)"])
            m = np.array(d["mean"], float)
            ax.plot(x[ok], m[ok], color=color, ls=ls, lw=1.1, marker=mk, ms=3.4,
                    mfc=color if filled else "white", mec=color, mew=0.8, label=f"{D}, {lab}")
    d = pf["new page (all)"]
    ok = np.array(d["plotted (>=5 concepts)"])
    ax.plot(x[ok], np.array(d["mean"])[ok], color=INK, ls=":", lw=0.8, marker="x", ms=3, label="all new page tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("occurrences on the page", fontsize=7, color=INK)
    ax.set_ylabel("concept-specific share", fontsize=7.5, color=INK)
    ax.set_title("(e) precision by page frequency", fontsize=7.5, color=INK, loc="left", pad=2)
    style(ax)
    ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=2.2, borderaxespad=0.1, labelspacing=0.2)
    out = ROOT / "paper/figures/extrapolation.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print("->", out)


if __name__ == "__main__":
    main()
