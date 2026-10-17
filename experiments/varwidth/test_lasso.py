"""Standalone checks for lasso.py (method 4).  Run:  .venv/bin/python experiments/varwidth/test_lasso.py

(a) vectorized CD vs sklearn Lasso (float64 reference) at a fixed lambda
(b) the returned mask has exactly K ones and every row width in [m_min, m_max]
(c) the lambda search lands within +-0.5% of K
(d) timing of one call at realistic size (d=2304, n=4760, k=100), plus the real whitened A if it exists;
    also the support agreement of the default (loose) CD tolerance with a tight one.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.varwidth.lasso import cd_solve, lasso_step  # noqa: E402

torch.set_num_threads(8)
ROOT = Path(__file__).resolve().parents[2]


def correlated_problem(d, n, k, seed, noise=0.05, rank=300):
    """A with a decaying spectrum, Z = mixed top singular directions (correlated columns, varied norms)."""
    g = torch.Generator().manual_seed(seed)
    r = min(rank, d, n)
    U = torch.linalg.qr(torch.randn(d, r, generator=g)).Q
    s = torch.logspace(2, 0, r)
    A = (U * s) @ torch.randn(r, n, generator=g) / n ** 0.5 + noise * torch.randn(d, n, generator=g)
    Uk, S, _ = torch.linalg.svd(A, full_matrices=False)
    mix = torch.eye(k) + 0.5 * torch.randn(k, k, generator=g) / k ** 0.5
    Z = (Uk[:, :k] * S[:k] ** 0.5) @ mix
    Y = torch.linalg.lstsq(Z, A).solution
    return A, Z, Y


def perturb(A, Z, seed, eps=0.02):
    g = torch.Generator().manual_seed(seed)
    d, k = Z.shape
    Z = Z + eps * Z.norm(dim=0) * torch.randn(d, k, generator=g) / d ** 0.5
    return Z, torch.linalg.lstsq(Z, A).solution


def check_mask(mask, K, m_min, m_max):
    w = mask.sum(1)
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == K, (int(mask.sum()), K)
    assert int(w.min()) >= m_min and int(w.max()) <= m_max, (int(w.min()), int(w.max()), m_min, m_max)


def test_a_vs_sklearn():
    from sklearn.linear_model import Lasso
    print("(a) CD vs sklearn Lasso (alpha = lam / d, fit_intercept=False)")
    worst = 0.0
    for seed, (d, n, k) in enumerate([(60, 40, 15), (120, 80, 30), (200, 50, 50)]):
        A, Z, _ = correlated_problem(d, n, k, seed, noise=0.2, rank=min(d, n))
        Zu = Z / Z.norm(dim=0)
        C, G = Zu.T @ A, Zu.T @ Zu
        G.diagonal().fill_(1.0)
        lam = float(C.abs().max()) * 0.15
        Y64 = torch.zeros(k, n, dtype=torch.float64)
        Y64, _, sw64, _ = cd_solve(C.double(), G.double(), Y64, lam, tol_rel=1e-10, max_sweeps=5000)
        Y32 = torch.zeros(k, n)
        Y32, _, sw32, _ = cd_solve(C, G, Y32, lam, tol_rel=1e-6, max_sweeps=5000)
        ref = Lasso(alpha=lam / d, fit_intercept=False, tol=1e-12, max_iter=200_000)
        Yref = torch.tensor(ref.fit(Zu.double().numpy(), A.double().numpy()).coef_.T)   # (k, n)
        scale = float(Yref.abs().max())
        e64 = float((Y64 - Yref).abs().max()) / scale
        e32 = float((Y32.double() - Yref).abs().max()) / scale
        supp = float(((Y32 != 0) != (Yref != 0)).float().mean())
        worst = max(worst, e32)
        print(f"   d={d} n={n} k={k} lam={lam:.3g} nnz_ref={int((Yref != 0).sum())} | rel max err f64={e64:.1e} "
              f"({sw64} sweeps) f32={e32:.1e} ({sw32} sweeps) | support mismatch f32={supp:.1e}")
        assert e64 < 1e-6 and e32 < 1e-3, (e64, e32)
    print(f"   ok (worst f32 rel err {worst:.1e})")


def test_bc_mask_and_bisection():
    print("(b,c) exact-K mask with floor/cap; lambda search within +-0.5% of K")
    cases = [  # (d, n, k, K, m_min, m_max)
        (300, 500, 20, 100, 1, 25),        # 1% of 500 = 5 per row
        (300, 500, 20, 500, 1, 25),        # 5%
        (300, 500, 20, 100, 0, 500),       # floor/cap off
        (400, 1000, 40, 400, 2, 50),
        (400, 1000, 40, 80, 2, 50),        # K = k*m_min: floor only
    ]
    for ci, (d, n, k, K, m_min, m_max) in enumerate(cases):
        A, Z, Y = correlated_problem(d, n, k, 100 + ci)
        state = {}
        for call in range(4):
            mask, state = lasso_step(A, Z, Y, K, m_min, m_max, state)
            check_mask(mask, K, m_min, m_max)
            assert state["nnz_ok"] and abs(state["nnz_lasso"] - K) <= 0.005 * K + 1e-9 or \
                abs(state["nnz_lasso"] - K) <= 1, (state["nnz_lasso"], K, state["trace"])
            Z, Y = perturb(A, Z, 1000 * ci + call)
        print(f"   n={n} k={k} K={K} [{m_min},{m_max}]: last call nnz_lasso={state['nnz_lasso']} "
              f"trim={state['n_trim']} fill={state['n_fill']} lam_steps={state['lam_steps']} sweeps={state['sweeps']}")
    # robustness: sweep budget exhausted / absurd lambda warm start -> mask still exact
    A, Z, Y = correlated_problem(300, 500, 20, 7)
    for st in ({"max_total_sweeps": 2}, {"lam": 1e6}, {"lam": 1e-8, "max_lam_steps": 2}):
        mask, st = lasso_step(A, Z, Y, 100, 1, 25, dict(st))
        check_mask(mask, 100, 1, 25)
    print("   ok (also exact K when the search is cut short)")


def time_calls(A, Z, Y, K, m_min, m_max, label, n_calls=6):
    state = {}
    times, rows = [], []
    for c in range(n_calls):
        mask, state = lasso_step(A, Z, Y, K, m_min, m_max, state)
        check_mask(mask, K, m_min, m_max)
        times.append(state["time"])
        rows.append(f"      call {c}: {state['time']:.3f}s warm={state['warm']} lam_steps={state['lam_steps']} "
                    f"sweeps={state['sweeps']} nnz_lasso={state['nnz_lasso']} trim={state['n_trim']} "
                    f"fill={state['n_fill']} lam={state['lam']:.4g}")
        if c == 0:
            # support agreement of the default tolerance with a tight solve at the same lambda
            tight = {"lam": state["lam"], "tol_rel": 1e-4, "max_sweeps": 2000, "max_total_sweeps": 20000,
                     "tol_nnz": 1.0}
            t0 = time.perf_counter()
            mask_t, tight = lasso_step(A, Z, Y, K, m_min, m_max, tight)
            tt = time.perf_counter() - t0
            jac = float((mask & mask_t).sum() / (mask | mask_t).sum())
            lj = float(((state["y"] != 0) & (tight["y"] != 0)).sum() / ((state["y"] != 0) | (tight["y"] != 0)).sum())
            rows.append(f"      tight CD (tol_rel=1e-4) at same lam: {tight['sweeps']} sweeps {tt:.2f}s, "
                        f"lasso-support Jaccard {lj:.4f}, final-mask Jaccard {jac:.4f}")
        Z, Y = perturb(A, Z, 50 + c, eps=0.01)
    print(f"   {label}: first {times[0]:.3f}s, later mean {sum(times[1:]) / max(len(times) - 1, 1):.3f}s")
    print("\n".join(rows))


def test_d_timing():
    print(f"(d) timing, {torch.get_num_threads()} threads")
    d, n, k = 2304, 4760, 100
    K, m_min, m_max = 48 * k, 10, 238
    A, Z, Y = correlated_problem(d, n, k, 3)
    time_calls(A, Z, Y, K, m_min, m_max, f"synthetic correlated d={d} n={n} k={k} K={K}")
    p = ROOT / "results" / "varwidth" / "prep" / "Ancient_Rome_whitened_S300.pt"
    if not p.exists():
        print(f"   real A not found ({p}); skipped")
        return
    A = torch.load(p, map_location="cpu", weights_only=False)["A"].float()
    d, n = A.shape
    import math
    K, m_min, m_max = math.ceil(0.01 * n) * k, math.ceil(0.002 * n), math.ceil(0.05 * n)
    U, S, V = torch.svd_lowrank(A, q=k + 10, niter=4)
    g = torch.Generator().manual_seed(0)
    Z = U[:, :k] * S[:k] ** 0.5 + 0.1 * torch.randn(d, k, generator=g) * float(S[:k].mean() ** 0.5) / d ** 0.5
    Y = torch.linalg.lstsq(Z, A).solution
    time_calls(A, Z, Y, K, m_min, m_max, f"real {p.name} d={d} n={n} k={k} K={K} [{m_min},{m_max}] (Z: SVD + noise)")


if __name__ == "__main__":
    test_a_vs_sklearn()
    test_bc_mask_and_bisection()
    test_d_timing()
    print("all tests passed")
