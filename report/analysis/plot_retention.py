#!/usr/bin/env python3
"""Figure 1 of the report: concept-QA retention r_L per question language, corrected results.

Mean over reliable concepts (18; 16 for Russian) with a 95% bootstrap over concepts
(10,000 resamples, default_rng(0)), read from
``results/crosslingual/google_gemma-2-2b-it/<method>_fixed/summary.csv``.
Writes ``report/figures/retention_by_language.pdf``.

Usage (from the repo root):
    .venv/bin/python report/analysis/plot_retention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crosslingual.reparse_saved import ROOT

LANGS = ["en", "de", "es", "fr", "ru"]
METHODS = [("ember", "EMBER", "#2a78d6", "o"),
           ("snmf", "SNMF", "#eb6834", "s"),
           ("snmf_ember", "SNMF+EMBER", "#1baf7a", "^")]
OUT = Path(__file__).resolve().parents[1] / "figures" / "retention_by_language.pdf"


def mean_ci(v: np.ndarray) -> tuple:
    idx = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return v.mean(), lo, hi


def main() -> int:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.edgecolor": "#555555", "xtick.color": "#555555",
                         "ytick.color": "#555555"})
    fig, ax = plt.subplots(figsize=(3.05, 2.05))
    x = np.arange(len(LANGS))
    for k, (method, label, color, marker) in enumerate(METHODS):
        df = pd.read_csv(ROOT / f"{method}_fixed" / "summary.csv")
        rel = df[df.reliable]
        stats = np.array([mean_ci(rel[rel.language == l].retention.dropna().values) for l in LANGS])
        xs = x + (k - 1) * 0.22
        ax.vlines(xs, stats[:, 1], stats[:, 2], color=color, lw=1.2)
        ax.plot(xs, stats[:, 0], ls="none", marker=marker, ms=5, color=color,
                mec="white", mew=0.8, label=label)
        print(label, " | ".join(f"{l}: {m:.2f} [{lo:.2f}, {hi:.2f}]" for l, (m, lo, hi) in zip(LANGS, stats)))
    ax.axhline(1.0, color="#555555", ls="--", lw=0.8)
    ax.text(len(LANGS) - 0.55, 1.015, "unedited", ha="right", va="bottom", fontsize=7, color="#555555")
    ax.set_xticks(x, LANGS)
    ax.set_xlim(-0.5, len(LANGS) - 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Question language")
    ax.set_ylabel("Concept-QA retention $r_L$\n(lower = more erased)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
