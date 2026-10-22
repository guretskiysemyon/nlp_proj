"""Figure 1: the Harry Potter feature is quiet in EMBER's raw matrix and loud after Contrastive Sharpening.

Reads the saved seed-42 fits (summary.json candidates: rho and Gemini stage-1 descriptions) and plots the
top-12 candidate features by mass ratio rho for EMBER's raw matrix and for the CS matrix (same 1% row sparsity).
Short labels are our paraphrases of the stage-1 descriptions (printed by this script for checking).

    .venv/bin/python paper/figures/src/fig_hp_rho.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RUNS = {"raw": "fixed_refit-False_plain", "cs": "fixed_refit-False"}
LABELS = {  # feature id -> short paraphrase of its stage-1 description (seed 42)
    "raw": {22: "magic, witchcraft", 9: "literacy, writing", 67: "3rd-person verbs", 36: "depiction",
            73: "books, narrative", 12: "criticism, debate", 84: "connectives", 98: "affection, desire",
            92: "commerce, retail", 20: "presentation", 23: "agreement",
            87: "Harry Potter"},
    "cs": {85: "Harry Potter", 46: "magic, fantasy", 44: "communication, language", 98: "literary works",
           7: "visual perception", 71: "motion verbs", 68: "books, reading", 66: "contention, issues",
           86: "contrast, opposition", 22: "commerce, trade", 10: "emotions", 12: "comparison"},
}
CONCEPT = {"raw": 87, "cs": 85}
GREY, EMBER_C, CINDER_C, INK, MUTED = "#B0B0B0", "#7F7F7F", "#D55E00", "#222222", "#666666"
CLIP = 1e3


def top(tag, n=12):
    s = json.loads((ROOT / "results/varwidth/Harry_Potter" / RUNS[tag] / "seed42/summary.json").read_text())
    c = sorted(s["candidates"], key=lambda x: -x["ratio_abs"])[:n]
    for i, x in enumerate(c, 1):
        print(tag, i, x["feature"], round(x["ratio_abs"], 1), x["description"][:80])
    return c


def panel(ax, tag, title):
    c = top(tag)
    ys = range(len(c))[::-1]
    for y, x in zip(ys, c):
        hp = x["feature"] == CONCEPT[tag]
        rho = min(x["ratio_abs"], CLIP)
        ax.barh(y, rho - 1, height=0.62, color=CINDER_C if hp else GREY, left=1, lw=0)
        ax.text(0.88, y, LABELS[tag][x["feature"]], ha="right", va="center", fontsize=7.5,
                color=INK if hp else MUTED, fontweight="bold" if hp else "normal")
        rank = len(c) - y
        if hp and x["ratio_abs"] > CLIP:
            ax.text(CLIP * 0.75, y, r"rank 1;  $\rho\approx 3\times10^{5}$ (no neutral mass)",
                    ha="right", va="center", fontsize=6.5, color="white")
        elif hp:
            ax.text(rho * 1.15, y, f"rank {rank} of 100", ha="left", va="center", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, CLIP)
    ax.axvline(2, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    # the dotted line is EMBER's threshold tau = 2 (named in the caption)
    ax.set_yticks([])
    ax.set_xlabel(r"mass ratio $\rho$ (concept / neutral), log scale", fontsize=7.5, color=INK)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED, length=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=3)
    ax.set_ylim(-0.6, len(c) - 0.1)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.05))
    panel(axes[0], "raw", "(a) EMBER: raw embeddings")
    panel(axes[1], "cs", "(b) CINDER: contrastively sharpened")
    fig.subplots_adjust(left=0.175, right=0.975, wspace=0.62, top=0.9, bottom=0.2)
    out = ROOT / "paper/figures/hp_rho.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("->", out)


if __name__ == "__main__":
    main()
