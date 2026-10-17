#!/usr/bin/env python
"""Aggregate results/ancient_rome_seed_sweep/*.json (scripts/run_seed_sweep.py) into per-run and per-variant stats.

    python scripts/seed_sweep_stats.py [DIR]

Writes DIR/runs.csv (one row per run) and DIR/stats.csv (mean/std/min/median/max per variant), and prints both plus
token stability across seeds: mean pairwise Jaccard of the selected-token sets, and how many tokens recur in >= 50% /
100% of the seeds. Needs the executed notebooks for the token sets (read from the "N concept features" table cell).
"""
import itertools
import json
import re
import sys
from pathlib import Path

import nbformat
import pandas as pd

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results" / "ancient_rome_seed_sweep"
TARGET = "# One column per selected feature: every token with nonzero Y_{i,t}"


def token_set(nb_path: Path) -> set:
    """Tokens (with their [C]/[N]/[B] tag) of the wide table, read from the cell's text/plain output."""
    nb = nbformat.read(nb_path, as_version=4)
    cell = next(c for c in nb.cells if c.cell_type == "code" and c.source.startswith(TARGET))
    toks = set()
    for out in cell.outputs:
        text = out.get("data", {}).get("text/plain", "") if out.output_type in ("display_data", "execute_result") else ""
        for line in text.splitlines()[1:]:
            toks.update(m.group(1) for m in re.finditer(r"(\S+ \[[CNB]\]) -?\d+\.\d+", line))
    return toks


rows, sets = [], {}
for p in sorted(DIR.glob("*_seed*.json")):
    d = json.loads(p.read_text())
    variant = p.stem.rsplit("_seed", 1)[0]
    n = d["n_distinct_tokens"]
    rows.append({"variant": variant, "seed": d["seed"], "n_features": d["n_concept_features"], "n_tokens": n,
                 "C": d["C"], "N": d["N"], "B": d["B"], "concept_share": d["C"] / n if n else float("nan"),
                 "rel_err": d["rel_err"]})
    sets[(variant, d["seed"])] = token_set(p.with_suffix(".ipynb"))

runs = pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)
runs.to_csv(DIR / "runs.csv", index=False)
metrics = ["n_features", "n_tokens", "C", "N", "B", "concept_share", "rel_err"]
stats = runs.groupby("variant")[metrics].agg(["mean", "std", "min", "median", "max"])
stats.to_csv(DIR / "stats.csv")

with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:.3f}".format):
    print(runs.to_string(index=False), "\n")
    for v, g in runs.groupby("variant"):
        print(f"{v}: n_features counts {g['n_features'].value_counts().sort_index().to_dict()}")
    print()
    print(stats.T.to_string(), "\n")

for v in runs["variant"].unique():
    ss = [s for (vv, _), s in sets.items() if vv == v]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(ss, 2) if a | b]
    freq = pd.Series([t for s in ss for t in s]).value_counts()
    print(f"{v}: {len(ss)} seeds, union {len(freq)} tokens, mean pairwise Jaccard {sum(jac) / len(jac):.3f}, "
          f"in >=50% of seeds: {(freq >= len(ss) / 2).sum()}, in all seeds: {(freq == len(ss)).sum()}")
    print("   most frequent:", ", ".join(f"{t} ({c})" for t, c in freq.head(15).items()))
