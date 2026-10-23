"""Figure 2: why EMBER misses quiet concepts and what sharpening changes (data: WP1, results/paper_analysis/quiet_concept).

(a) Over Gemma's vocabulary, rarer tokens share a larger component along the background mean mu_B.
(b) A single sparse feature seeded on a concept's lexicon tokens, iterated to its fixed point, per concept:
    purity (concept-only share of its m tokens) vs value (percentile of its energy among the fit's features).

    .venv/bin/python paper/figures/src/fig_mechanism.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results/paper_analysis/quiet_concept/figures"
GREY, GREEN, ORANGE, INK, MUTED = "#7F7F7F", "#009E73", "#D55E00", "#222222", "#666666"


def rows(name):
    with (SRC / name).open() as f:
        return list(csv.DictReader(f))


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.1), gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) projection on mu_B vs frequency
    fr = rows("fig_frequency.csv")
    x = np.array([float(r["lf_bin_mid"]) for r in fr])
    q25, q50, q75 = (np.array([float(r[k]) for r in fr]) for k in ("mu_proj_q25", "mu_proj_q50", "mu_proj_q75"))
    a.fill_between(x, q25, q75, color=GREY, alpha=0.18, lw=0)
    a.plot(x, q50, color=GREY, lw=1.6)
    a.axhline(0, color="#CCCCCC", lw=0.6)
    a.set_xlabel(r"token frequency in Wikipedia ($\log_{10}$ per million)", fontsize=7.5, color=INK)
    a.set_ylabel(r"projection on $\hat\mu_B$", fontsize=7.5, color=INK)
    a.set_title("(a) Rare tokens share one direction", fontsize=8.5, color=INK, loc="left", pad=3)
    a.annotate("rare", xy=(-0.5, 41), xytext=(0.4, 48), fontsize=7, color=MUTED,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    a.annotate("frequent", xy=(4.0, -46), xytext=(2.3, -46), fontsize=7, color=MUTED, va="center")

    # (b) purity vs value
    pv = rows("fig_purity_value.csv")
    series = [("raw", "raw (EMBER)", GREY, "o"), ("wo", "whitened only", GREEN, "s"), ("cs", "sharpened (CINDER)", ORANGE, "^")]
    med = {}
    for key, label, color, marker in series:
        px = np.array([float(r[f"{key}_purity"]) for r in pv])
        py = np.array([float(r[f"{key}_value_pct"]) for r in pv])
        b.scatter(px, py, s=10, color=color, marker=marker, alpha=0.55, lw=0, label=label, zorder=2)
        med[key] = (np.median(px), np.median(py))
        b.scatter(*med[key], s=70, color=color, marker=marker, edgecolor="white", lw=1.2, zorder=4)
    for k0, k1 in (("raw", "wo"), ("wo", "cs")):
        b.annotate("", xy=med[k1], xytext=med[k0],
                   arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, shrinkA=6, shrinkB=6), zorder=3)
    b.text(0.02, 104, "valuable but impure", fontsize=7, color=MUTED, va="bottom")
    b.text(0.99, 104, "pure and valuable", fontsize=7, color=MUTED, va="bottom", ha="right")
    b.text(0.99, 30, "pure but ordinary", fontsize=7, color=MUTED, ha="right")
    b.set_xlim(-0.03, 1.03)
    b.set_ylim(20, 112)
    b.set_xlabel("purity: concept-only share of the feature's tokens", fontsize=7.5, color=INK)
    b.set_ylabel("value (percentile)", fontsize=7.5, color=INK)
    b.set_title("(b) The concept feature: raw, whitened, sharpened", fontsize=8.5, color=INK, loc="left", pad=3)
    b.legend(fontsize=6.5, frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.2)

    for ax in (a, b):
        ax.tick_params(labelsize=7, colors=MUTED, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#BBBBBB")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.2, top=0.89, wspace=0.28)
    out = ROOT / "paper/figures/mechanism.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("->", out, {k: tuple(round(v, 2) for v in m) for k, m in med.items()})


if __name__ == "__main__":
    main()
