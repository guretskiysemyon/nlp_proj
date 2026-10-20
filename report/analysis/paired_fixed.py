#!/usr/bin/env python3
"""Paired method differences on the repaired cross-lingual results (crosslingual_next.md §6).

Per concept: the mean leak over its reliable de/es/fr/ru cells (the paper's
convention), the leak per language, and English retention r_en. Each method pair
(``ember - snmf``, ``snmf_ember - snmf``, ``ember - snmf_ember``) is differenced per
concept, so every comparison is paired (same questions, prompts and option order).
Reported: mean difference, 95% bootstrap over concepts (10,000 resamples,
default_rng(0)), paired t-test, n, and how many concepts have A > B. Also the
per-method mean leak and r_en in the same convention. Original run (``<method>/``)
next to fixed (``<method>_fixed/``); the original column reproduces the paper's draft.

Reads ``results/crosslingual/google_gemma-2-2b-it/<method>[_fixed]/summary.csv``,
writes ``paired_fixed.csv`` next to this script. CPU only, seconds.

Usage (from the repo root):
    .venv/bin/python report/analysis/paired_fixed.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy import stats

from crosslingual.reparse_saved import ROOT

OTHER = ["de", "es", "fr", "ru"]
METHODS = ("ember", "snmf", "snmf_ember")
PAIRS = (("ember", "snmf"), ("snmf_ember", "snmf"), ("ember", "snmf_ember"))
RUNS = {"original": "", "fixed": "_fixed"}
OUT = Path(__file__).resolve().parent / "paired_fixed.csv"


def per_concept(method_dir: Path) -> pd.DataFrame:
    """One row per concept: mean non-English leak, leak per language, r_en."""
    df = pd.read_csv(method_dir / "summary.csv")
    r = df[df.reliable & (df.language != "en")]
    leak = r.pivot(index="concept", columns="language", values="leak").reindex(columns=OTHER)
    out = leak.add_prefix("leak_")
    out["leak_mean"] = leak.mean(axis=1, skipna=True)
    out["r_en"] = df[df.language == "en"].set_index("concept").retention
    return out


def describe(d: pd.Series) -> Dict[str, Any]:
    v = d.dropna().values
    idx = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    t = stats.ttest_1samp(v, 0.0)
    return {"mean": v.mean(), "lo": lo, "hi": hi, "t": t.statistic, "p": t.pvalue,
            "n": len(v), "n_pos": int((v > 0).sum())}


def main() -> int:
    rows: List[Dict[str, Any]] = []
    for run, suffix in RUNS.items():
        pc = {m: per_concept(ROOT / f"{m}{suffix}") for m in METHODS}
        for m in METHODS:
            for stat in ("leak_mean", "leak_ru", "r_en"):
                rows.append({"run": run, "comparison": m, "stat": stat, **describe(pc[m][stat])})
        for a, b in PAIRS:
            for stat in ["leak_mean"] + [f"leak_{l}" for l in OTHER] + ["r_en"]:
                rows.append({"run": run, "comparison": f"{a} - {b}", "stat": stat,
                             **describe(pc[a][stat] - pc[b][stat])})
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    def fmt(r) -> str:
        return f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] t={r['t']:.2f} p={r['p']:.3f} ({r['n_pos']}/{r['n']})"

    wide = res.set_index(["comparison", "stat", "run"]).apply(fmt, axis=1).unstack("run")
    order = list(dict.fromkeys(zip(res.comparison, res.stat)))
    print("| comparison | stat | original | fixed |\n|---|---|---|---|")
    for c, s in order:
        print(f"| {c} | {s} | {wide.loc[(c, s), 'original']} | {wide.loc[(c, s), 'fixed']} |")
    print(f"\n(k/n) = concepts with a value > 0 / concepts. -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
