"""Stage 0 sanity (instructions_new.txt "Run plan"): planted-width synthetic data.

    .venv/bin/python -m experiments.varwidth.synth [--methods fixed m1 m2 m3 m4] [--seed 0]

Data: d=256, n=2000, k=20 features. Planted widths are log-spaced from ~0.2% to ~5% of n and rescaled (largest
remainder) so they sum to the 1% budget K = ceil(0.01 n) k = 400; z_i random unit vectors; y_i signed, |y| ~ U(1, 2)
on a random planted support; A = Z Y + dense Gaussian noise (std = noise x the RMS entry of Z Y), run at several noise
levels (default 0.02, 0.3, 1.0). At 0.02 ~80% of the tokens are ~zero vectors, an unrealistic regime (every real
token has a sizeable residual; the real rel_err is ~0.87) in which per-row top-m by |y_dense| with refit (fixed +
refit, m3) is unstable; 0.3 and 1.0 are the informative levels. Every method is fit with debug=True, which asserts
exactly K ones and the row bounds at every iteration. Recovery: recovered features are matched to planted ones by
|cos z| (Hungarian, scipy linear_sum_assignment); we report the matched |cos| , Spearman of the widths, mean |width
error| and support F1. Output: results/varwidth/stage0_synth.json and a printed table.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from experiments.varwidth.common import RESULTS, VWConfig
from experiments.varwidth.core import fit_widths_to_total, fit_varwidth


def make_synth(d: int = 256, n: int = 2000, k: int = 20, noise: float = 0.02, seed: int = 0) -> dict:
    """Planted-width data. Returns {"A" (d,n), "Z" (d,k) unit columns, "Y" (k,n), "widths" (k,)}."""
    g = torch.Generator().manual_seed(seed)
    K = math.ceil(0.01 * n) * k
    lo, hi = math.ceil(0.002 * n), math.ceil(0.05 * n)
    raw = torch.logspace(math.log10(0.002 * n), math.log10(0.05 * n), k, dtype=torch.float64)
    widths = fit_widths_to_total(raw[torch.randperm(k, generator=g)], K, lo, hi)
    Z = torch.randn(d, k, generator=g)
    Z = Z / Z.norm(dim=0)
    Y = torch.zeros(k, n)
    for i in range(k):
        idx = torch.randperm(n, generator=g)[: int(widths[i])]
        sign = torch.randint(0, 2, (len(idx),), generator=g).float() * 2 - 1
        Y[i, idx] = sign * (1 + torch.rand(len(idx), generator=g))
    A = Z @ Y
    A = A + noise * A.norm() / math.sqrt(d * n) * torch.randn(d, n, generator=g)   # noise relative to the RMS entry
    return {"A": A, "Z": Z, "Y": Y, "widths": widths, "K": K}


def recovery(planted: dict, fit: dict) -> dict:
    Zt, Yt = planted["Z"], planted["Y"]
    Zr = fit["Z"] / fit["Z"].norm(dim=0).clamp_min(1e-12)
    cos = (Zt.T @ Zr).abs().numpy()                                     # (k planted, k recovered)
    r, c = linear_sum_assignment(-cos)
    w_true = planted["widths"].numpy()[r]
    w_rec = fit["widths"].numpy()[c]
    f1 = []
    for i, j in zip(r, c):
        t, p = Yt[i] != 0, fit["Y"][j] != 0
        tp = int((t & p).sum())
        f1.append(2 * tp / max(1, int(t.sum()) + int(p.sum())))
    return {"mean_cos": float(cos[r, c].mean()), "min_cos": float(cos[r, c].min()),
            "spearman": float(spearmanr(w_true, w_rec).statistic) if np.ptp(w_rec) > 0 else float("nan"), "mean_abs_width_err": float(np.abs(w_true - w_rec).mean()),
            "mean_f1": float(np.mean(f1)), "planted_widths": w_true.tolist(), "recovered_widths": w_rec.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["fixed", "m1", "m2", "m3", "m4"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.02, 0.3, 1.0])
    ap.add_argument("--max-iter", type=int, default=5000)
    args = ap.parse_args()
    torch.set_num_threads(8)
    rows = []
    for noise in args.noise:
        planted = make_synth(noise=noise, seed=args.data_seed)
        n = planted["A"].shape[1]
        for method in args.methods:
            for refit in ([True, False] if method == "fixed" else [True]):
                cfg = VWConfig(method=method, rank=20, refit=refit, max_iter=args.max_iter)
                t = time.perf_counter()
                fit = fit_varwidth(planted["A"], cfg, seed=args.seed, debug=True)   # asserts exact K every iteration
                rec = recovery(planted, fit)
                rows.append({"noise": noise, "method": method, "refit": refit, "rel_err": fit["rel_err"],
                             "n_iter": fit["n_iter"], "lockin_iter": fit["lockin_iter"],
                             "stop_reason": fit["stop_reason"], "seconds": time.perf_counter() - t,
                             "exact_K_every_iter": True, **rec})
    out = {"data": {"d": planted["A"].shape[0], "n": n, "k": 20, "K": planted["K"], "noise_levels": args.noise,
                    "planted_widths": planted["widths"].tolist()}, "runs": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "stage0_synth.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"planted widths (sorted): {sorted(planted['widths'].tolist())}")
    print(f"{'noise':5s} {'method':8s} {'refit':5s} {'rel_err':>8s} {'iters':>6s} {'lockin':>6s} {'cos':>6s} {'spear':>6s} "
          f"{'|dw|':>6s} {'F1':>6s} {'sec':>6s}")
    for r in rows:
        print(f"{r['noise']:<5g} {r['method']:8s} {str(r['refit']):5s} {r['rel_err']:8.4f} {r['n_iter']:6d} {r['lockin_iter']:6d} "
              f"{r['mean_cos']:6.3f} {r['spearman']:6.3f} {r['mean_abs_width_err']:6.2f} {r['mean_f1']:6.3f} "
              f"{r['seconds']:6.1f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
