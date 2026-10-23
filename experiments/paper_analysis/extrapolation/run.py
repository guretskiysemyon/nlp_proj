"""Score every (concept, dictionary, seed) run of the benchmark and store the raw per-token results.

    .venv/bin/python -m experiments.paper_analysis.extrapolation.run [--concepts ...] [--workers 4]

Writes results/paper_analysis/extrapolation/runs/<Slug>.pkl: {(dictionary, seed): run_scores(...)}.
"""
from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor

from experiments.paper_analysis.extrapolation.common import CONCEPTS, DICTS, OUT, SEEDS
from experiments.varwidth.common import slug

RUNS = OUT / "runs"


def one(concept: str) -> str:
    import torch
    torch.set_num_threads(2)
    from experiments.paper_analysis.extrapolation.score import run_scores
    res = {(D, s): run_scores(concept, D, s) for D in DICTS for s in SEEDS}
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / f"{slug(concept)}.pkl", "wb") as f:
        pickle.dump(res, f)
    n = sum(bool(r["F_C"]) for r in res.values())
    rec = [x for r in res.values() if r["F_C"] for x in r["support_recovery"]]
    return f"[run] {concept}: {n}/{len(res)} runs with F_C; support recovery min {min(rec):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=CONCEPTS)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(one, a.concepts):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
