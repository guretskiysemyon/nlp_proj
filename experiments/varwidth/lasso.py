"""Method 4 Y-step: L1 ALS with one lambda bisected to ~K nonzeros, then exact-K floor/cap trim/fill.

    mask, state = lasso_step(A, Z, Y_prev, K, m_min, m_max, state)

Called by core.py every cfg.m4_every iterations right after the dense Y-solve; the loop's support refit removes the
lasso shrinkage afterwards, so only the mask is returned.

Algorithm
---------
1. Unit-normalize Z's columns (L1 is not scale-invariant). In the unit scale y_unit = y * ||z||, so |y_unit| is the
   contribution score |y| * ||z|| of methods 1/4.
2. Lasso per token, y_j = argmin 1/2 ||a_j - Zu y||^2 + lam ||y||_1, by cyclic coordinate descent vectorized over all
   n tokens at once. With G = Zu^T Zu (unit diagonal) and C = Zu^T A we keep R = C - G Y; for coordinate i:
       rho = R[i] + Y[i];  Y[i] <- soft(rho, lam);  R -= G[:, i] (x) delta.
   delta is non-zero only on tokens whose y_i is (or becomes) non-zero (~1-2% of n), so the rank-1 update is applied
   only to those columns (index_add_), making one sweep ~ 100 x O(n) cheap ops instead of 100 dense k x n updates.
   R is recomputed exactly (one GEMM) at the start of every lambda solve, so float32 drift does not accumulate;
   float32 is therefore sufficient (checked against sklearn's float64 Lasso in test_lasso.py).
   Convergence: max |delta| over a sweep <= tol_rel * lam (we only need the support, which is decided by |rho| vs lam,
   so changes much smaller than lam are irrelevant), capped at max_sweeps per lambda.
3. lambda search: warm-started from state["lam"] (first call: the K-th largest |y_dense_unit|, exact for orthogonal Z).
   Each new lambda comes from a log-log secant through the last two trials (nnz ~ lam^-p locally; p = 1 for the
   first step). Until the target is bracketed the step is clipped to [1/2, 2] (at most a doubling/halving); once
   bracketed, a secant point outside the bracket is replaced by the geometric midpoint (safeguarded secant).
   Stops when |nnz - K| <= tol_nnz * K (default 0.5%) or after max_lam_steps / max_total_sweeps.
   Y is warm-started across lambda steps from the latest solution, and across calls from either state["y"] (previous
   lasso solution, unit scale) or the scaled dense Y_prev, whichever has the lower lasso objective at the start lambda.
4. Trim/fill to exactly K with pool_floor_cap on a score where every lasso non-zero ranks above every zero:
       non-zero: 2 + |y_unit| / max|y_unit|          (in (2, 3])
       zero:     |rho| / lam clipped to [0, 1]       (KKT correlation: how close the entry is to entering the lasso)
   so trimming drops the smallest-contribution lasso entries and filling adds the entries that would enter next.
"""
from __future__ import annotations

import math
import time

import torch

try:
    from .common import pool_floor_cap
except ImportError:                                            # run as a plain script from this directory
    from common import pool_floor_cap

DEFAULTS = dict(
    tol_nnz=0.005,          # |nnz - K| <= tol_nnz * K
    tol_rel=0.02,           # CD stops when max |delta y| over a sweep <= tol_rel * lam
    max_sweeps=50,          # CD sweeps per lambda
    max_lam_steps=30,       # lambda trials per call
    max_total_sweeps=600,   # hard cap on CD sweeps per call
)


def lasso_objective(C: torch.Tensor, G: torch.Tensor, Y: torch.Tensor, lam: float) -> float:
    """1/2||A - Zu Y||^2 - 1/2||A||^2 + lam ||Y||_1, from C = Zu^T A and G = Zu^T Zu."""
    return float(-(Y * C).sum() + 0.5 * (Y * (G @ Y)).sum() + lam * Y.abs().sum())


def cd_solve(C: torch.Tensor, G: torch.Tensor, Y: torch.Tensor, lam: float, tol_rel: float = 0.02,
             max_sweeps: int = 50) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Cyclic CD for all n lasso problems at once (in place on Y). G must have unit diagonal.
    Returns (Y, R = C - G Y, sweeps used, last max |delta|)."""
    k, n = C.shape
    R = C - G @ Y
    Gcols = [G[:, i:i + 1].contiguous() for i in range(k)]
    dense_thr = n // 4
    tol = tol_rel * lam
    maxd = float("inf")
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        maxd = 0.0
        for i in range(k):
            yi = Y[i]
            rho = R[i] + yi
            new = torch.sign(rho) * torch.clamp(rho.abs() - lam, min=0.0)
            delta = new - yi
            idx = delta.nonzero().squeeze(1)
            m = idx.numel()
            if m == 0:
                continue
            Y[i] = new
            if m > dense_thr:
                R.addr_(Gcols[i].squeeze(1), delta, alpha=-1.0)
                d = float(delta.abs().max())
            else:
                dv = delta.index_select(0, idx)
                R.index_add_(1, idx, Gcols[i] * dv, alpha=-1.0)
                d = float(dv.abs().max())
            if d > maxd:
                maxd = d
        if maxd <= tol:
            break
    return Y, R, sweeps, maxd


def lasso_step(A: torch.Tensor, Z: torch.Tensor, Y_prev: torch.Tensor, K: int, m_min: int, m_max: int,
               state: dict | None) -> tuple[torch.Tensor, dict]:
    """A (d,n) f32, Z (d,k), Y_prev (k,n) current Y (warm start). Returns (mask bool (k,n) with exactly K ones and
    rows within [m_min, m_max], state).

    state (dict, updated and returned; options may be preset in it, see DEFAULTS):
        lam         lambda reached (unit-norm Z scale); warm start for the next call
        y           (k,n) f32 lasso solution in the unit-Z scale; warm start for the next call
        sweeps      CD sweeps used in this call;  lam_steps: lambda trials in this call
        nnz_lasso   lasso non-zeros before trim/fill;  n_trim / n_fill: entries removed / added by floor/cap
        nnz_ok      whether |nnz_lasso - K| <= tol_nnz * K was reached
        warm        "state" or "dense": which warm start was used
        time        seconds for this call;  trace: [(lam, nnz, sweeps)] per lambda trial
        calls, total_time, total_sweeps: cumulative over calls
    """
    t0 = time.perf_counter()
    state = {} if state is None else state
    opt = {key: state.get(key, v) for key, v in DEFAULTS.items()}
    A = A.float().contiguous()
    Z = Z.float().contiguous()
    k, n = Y_prev.shape

    # 1. unit-norm columns; the scale moves into Y
    norms = Z.norm(dim=0).clamp_min(1e-12)
    Zu = Z / norms
    G = Zu.T @ Zu
    G.diagonal().fill_(1.0)
    C = Zu.T @ A
    Yd = (Y_prev.float() * norms[:, None]).contiguous()     # row-major: CD reads rows; lstsq gives col-major

    # initial lambda
    lam = state.get("lam")
    if lam is None or not math.isfinite(lam) or lam <= 0:
        lam = float(torch.topk(Yd.abs().reshape(-1), K).values[-1])

    # warm start: previous lasso solution or the scaled dense solve, whichever is better at the start lambda
    Y = Yd.clone()
    warm = "dense"
    ys = state.get("y")
    if ys is not None and tuple(ys.shape) == (k, n):
        ys = ys.float().contiguous()
        if lasso_objective(C, G, ys, lam) <= lasso_objective(C, G, Y, lam):
            Y, warm = ys.clone(), "state"

    target_lo, target_hi = K * (1 - opt["tol_nnz"]), K * (1 + opt["tol_nnz"])
    lo = hi = None                  # lo: (lam, nnz) with nnz > K (lambda too small); hi: nnz < K (too large)
    trace = []
    total = 0
    best = None                     # (|nnz-K|, lam, Y, R)
    R = None
    for step in range(opt["max_lam_steps"]):
        budget = min(opt["max_sweeps"], opt["max_total_sweeps"] - total)
        if budget <= 0:
            break
        Y, R, sw, _ = cd_solve(C, G, Y, lam, opt["tol_rel"], budget)
        total += sw
        nnz = int((Y != 0).sum())
        trace.append((lam, nnz, sw))
        if best is None or abs(nnz - K) < best[0]:
            best = (abs(nnz - K), lam, Y.clone(), R.clone())
        if target_lo <= nnz <= target_hi:
            break
        if nnz > K:
            lo = (lam, nnz)
        else:
            hi = (lam, nnz)
        # next lambda: log-log secant through the last two trials (nnz ~ lam^-p locally)
        lt, nt = math.log(lam), math.log(max(nnz, 1))
        p = 1.0                                          # default elasticity
        if len(trace) >= 2 and trace[-2][1] > 0 and nnz > 0 and trace[-2][0] != lam:
            pe = -(nt - math.log(trace[-2][1])) / (lt - math.log(trace[-2][0]))
            if pe > 0.1:
                p = pe
        x = lt + (nt - math.log(K)) / p
        if lo is not None and hi is not None:
            a, b = math.log(lo[0]), math.log(hi[0])      # a < b
            if not (a < x < b):
                x = 0.5 * (a + b)                        # safeguard: geometric bisection
        else:
            x = lt + min(max(x - lt, -math.log(2)), math.log(2))    # bracketing: at most doubling / halving
        lam = math.exp(x)

    _, lam_b, Y, R = best
    nnz_lasso = int((Y != 0).sum())
    nz = Y != 0

    # 4. exact-K floor/cap trim/fill
    yabs = Y.abs()
    ymax = float(yabs.max()) if nnz_lasso else 1.0
    score = torch.where(nz, 2.0 + yabs / ymax, ((R + Y).abs() / lam_b).clamp(0.0, 1.0))
    mask = pool_floor_cap(score.contiguous(), K, m_min, m_max)

    el = time.perf_counter() - t0
    state.update(
        lam=lam_b, y=Y, sweeps=total, lam_steps=len(trace), nnz_lasso=nnz_lasso,
        nnz_ok=bool(target_lo <= nnz_lasso <= target_hi),
        n_trim=int((nz & ~mask).sum()), n_fill=int((mask & ~nz).sum()), warm=warm, time=el, trace=trace,
        calls=state.get("calls", 0) + 1, total_time=state.get("total_time", 0.0) + el,
        total_sweeps=state.get("total_sweeps", 0) + total,
    )
    return mask, state
