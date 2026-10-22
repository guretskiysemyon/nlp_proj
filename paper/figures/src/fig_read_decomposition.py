"""Figure 4: first-layer read strength of each edit variant (paper style).

Data: results/paper_analysis/edit_direction/fig_read_decomposition.csv (WP6; one value per concept = mean over seeds).

    .venv/bin/python paper/figures/src/fig_read_decomposition.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results/paper_analysis/edit_direction/fig_read_decomposition.csv"
GREY, BLUE, ORANGE, AMBER, INK, MUTED = "#7F7F7F", "#0072B2", "#D55E00", "#E69F00", "#222222", "#666666"
PANELS = {
    "Gemma-2-2B": [("ember", "EMBER", GREY, True), ("ember_c", "EMBER,\ncentred", GREY, False),
                   ("cinder", "CINDER", ORANGE, False), ("cinder_z", "CINDER-z", AMBER, True)],
    "Llama-3.1-8B, eight languages": [("a2", "A2\n(EMBER)", GREY, True), ("a2_c", "A2,\ncentred", GREY, False),
                                      ("a3", "A3\n(CINDER)", ORANGE, False), ("a3_c", "A3, re-solved\ncentred", ORANGE, False),
                                      ("a3z", "A3z\n(CINDER-z)", AMBER, True)],
}


def main():
    data = defaultdict(list)
    with SRC.open() as f:
        for r in csv.DictReader(f):
            data[(r["model"], r["variant"])].append(float(r["read_gain"]))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), gridspec_kw={"width_ratios": [4, 5]})
    rng = np.random.default_rng(0)
    for ax, (model, variants) in zip(axes, PANELS.items()):
        for x, (key, label, color, uncentred) in enumerate(variants):
            ys = np.array(data[(model.split(",")[0], key)])
            xs = x + rng.uniform(-0.17, 0.17, len(ys))
            if uncentred:
                ax.scatter(xs, ys, s=11, color=color, lw=0, zorder=3)
            else:
                ax.scatter(xs, ys, s=11, facecolor="white", edgecolor=color, lw=0.8, zorder=3)
            ax.plot([x - 0.3, x + 0.3], [np.median(ys)] * 2, color=INK, lw=1.4, zorder=4)
        ax.axhline(1, color="#BBBBBB", lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([v[1] for v in variants], fontsize=7)
        ax.tick_params(axis="y", labelsize=7, colors=MUTED, length=2)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#BBBBBB")
        ax.set_title(model, fontsize=8.5, color=INK, loc="left", pad=3)
    axes[0].set_yticks([1, 1.5, 2, 3])
    axes[1].set_yticks([1, 2, 4, 8])
    axes[0].set_ylabel("layer-0 read strength\n(× random direction)", fontsize=7.5, color=INK)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.2, top=0.9, wspace=0.18)
    out = ROOT / "paper/figures/read_decomposition.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("->", out)


if __name__ == "__main__":
    main()
