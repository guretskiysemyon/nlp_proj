"""Shared constants and helpers for WP3 (widths): fit tags, paths, concept list, statistics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.varwidth.common import RESULTS as VW_RESULTS
from experiments.varwidth.common import ROOT, run_dir, slug  # noqa: F401  (re-exported)

OUT = ROOT / "results" / "paper_analysis" / "widths"
CACHE = OUT / "cache"
SEEDS = list(range(42, 47))
# config name -> (fit tag, matrix)
TAGS = {
    "L0": ("fixed_refit-False_plain", "plain"),        # EMBER: raw, RS-1%
    "L1": ("fixed_refit-False", "whitened"),           # CS, RS-1%
    "L12": ("rn_refit-False", "whitened"),             # CS, GS (CINDER's fit)
    "rnP": ("rn_refit-False_plain", "plain"),          # raw, GS
    "RS2": ("fixed_sparsity-0.02_refit-False", "whitened"),  # CS, RS-2% (co-adaptation control)
}
NAMES = {"L0": "EMBER (raw, RS-1%)", "L1": "CS, RS-1%", "L12": "CS, GS (CINDER)", "rnP": "raw, GS",
         "RS2": "CS, RS-2%"}
COLORS = {"L0": "#7F7F7F", "L1": "#0072B2", "L12": "#D55E00", "z": "#E69F00", "rnP": "#009E73", "RS2": "#56B4E9"}
RIDGE = 1e-4
TOP_KEEP = 600          # positions of the dense ranking kept per run


def concepts() -> list:
    out = []
    for f in ("concept_sentences.json", "concept_sentences_extra.json"):
        out += [e["concept"] for e in json.loads((ROOT / "data" / f).read_text())]
    return out


STUDY_CONCEPTS = set(concepts())


def boot_ci(x, stat=np.mean, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    b = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def boot_spearman(x, y, n: int = 10_000, seed: int = 0) -> dict:
    """Spearman rho, its p-value and a bootstrap 95% CI over the units (pairs resampled)."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 1 and np.unique(y[i]).size > 1:
            bs.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (np.nan, np.nan)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def wilcoxon(d) -> dict:
    from scipy.stats import wilcoxon as w
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return {"n": int(len(d)), "median": float(np.median(d)) if len(d) else float("nan"), "p": float("nan")}
    r = w(d)
    lo, hi = boot_ci(d)
    return {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)), "ci_lo": lo, "ci_hi": hi,
            "W": float(r.statistic), "p": float(r.pvalue), "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}
