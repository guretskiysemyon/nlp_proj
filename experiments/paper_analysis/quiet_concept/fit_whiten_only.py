"""Whitening-only (gamma = 0) fits (and, with --matrix ctr, centring-only fits on (E_V* - mu_B)^T): EMBER's sparse MF (fixed RS-1%, k = 100, no refit, = tag fixed_refit-False)
on A_wo = ((E_V* - mu_B) W)^T, the CS matrix without the sharpening step.

    python -m experiments.paper_analysis.quiet_concept.fit_whiten_only --concepts paper --seeds 42 43 44 [--device cpu]
    python -m experiments.paper_analysis.quiet_concept.fit_whiten_only --index 7     # SLURM array task (job list)

Output: results/paper_analysis/quiet_concept/fits/<Slug>/fixed_refit-False_{g0,ctr}/seed<N>/fit.pt (fit_varwidth dict,
minus nothing), skipped if present.
"""
from __future__ import annotations

import argparse
import itertools
import os
import time

import torch

from experiments.paper_analysis.quiet_concept.common import WO_SEEDS, concepts, fit_path, matrices
from experiments.varwidth.common import VWConfig
from experiments.varwidth.core import fit_varwidth


def jobs(which: str, seeds) -> list[tuple[str, int]]:
    return list(itertools.product(concepts(which), seeds))


def run_one(concept: str, seed: int, device: str = "cpu", matrix: str = "wo") -> None:
    p = fit_path(concept, matrix, seed)
    if p.exists():
        print(f"[skip] {concept} seed {seed}", flush=True)
        return
    A = matrices(concept)[matrix].float()
    cfg = VWConfig(method="fixed", refit=False)
    assert cfg.tag() == "fixed_refit-False"
    t0 = time.time()
    fit = fit_varwidth(A, cfg, seed, device)
    fit = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in fit.items()}
    fit["matrix"] = {"wo": "whiten_only_g0", "ctr": "centred_mu_B", "rawp": "raw_perp_mu_B"}[matrix]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"fit.pt.{os.getpid()}.tmp")
    torch.save(fit, tmp)
    tmp.replace(p)
    print(f"[fit] {matrix} {concept} seed {seed}: rel_err {fit['rel_err']:.4f}  iters {fit['n_iter']}  "
          f"{time.time() - t0:.0f}s ({device})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="paper", choices=["paper", "extra", "all"])
    ap.add_argument("--seeds", type=int, nargs="*", default=WO_SEEDS)
    ap.add_argument("--index", type=int, default=None, help="run only job #index of the (concept, seed) list")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--matrix", default="wo", choices=["wo", "ctr", "rawp"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    J = jobs(a.concepts, a.seeds)
    if a.index is not None:
        J = [J[a.index]]
    for c, s in J:
        run_one(c, s, a.device, a.matrix)


if __name__ == "__main__":
    main()
