"""The shared variable-width ALS loop of instructions_new.txt ("Shared loop") and the method-5 merge step.

    fit = fit_varwidth(A, VWConfig(method="m2"), seed=42)          # A: (d, n) float32; fit has the fit.pt keys
    fit5 = merge_features(A, fit, [(3, 17)])                        # method 5 on a finished fit

One iteration (lambda = cfg.ridge):
    1. Z <- A Y^T (Y Y^T + lambda I)^-1                             paper's Z-update
    2. Y_dense <- (Z^T Z + lambda I)^-1 Z^T A                       paper's dense Y-solve
    3. S <- Allocate(Y_dense, Z)                                    exactly K ones, rows in [m_min, m_max]
    4. refit: y_{S_j, j} = (Z_S^T Z_S + lambda I)^-1 Z_S^T a_j      (cfg.refit; else Y = Y_dense * S, as in the paper)
    5. rows of Y -> unit l2, scale absorbed into Z's columns

Allocate (cfg.method):
    fixed  each row its top ceil(sparsity*n) entries by |y_dense| (the paper's WTA; ignores floor/cap)
    m1     score |y_ij| * ||z_i||, then pool_floor_cap (optional continuation: K_t from cont_start to sparsity)
    m2     score |y_ij| / sigma_i, sigma_i = 1.4826 * MAD of the dense row (torch.median = lower median for even n)
    m3     per-row widths m_i from floor-subtracted entropy, refreshed every m3_every iterations; row i keeps its
           top m_i by |y_dense|
    m4     lasso_step (lasso.py) every m4_every iterations; support held fixed (refit only) in between

Cost per iteration (d=2304, n=4760, k=100): two dense GEMMs (A Y^T and Z^T A, ~1.1 GFLOP each) plus k x k / k x n
work. Everything else reuses those products: the refit only needs Z^T Z and Z^T A (gathered per token), and the
relative error is computed from the expansion ||A||^2 - 2 <Z^T A, Y> + <Z^T Z, Y Y^T> without forming A - Z Y.

Stopping: after the continuation phase (if any), stop once rel_err has not improved by >= cfg.tol (absolute, on the
relative error) over the best value for cfg.patience consecutive iterations AND churn stayed < cfg.churn_tol over
those same iterations; otherwise stop at cfg.max_iter. The FINAL iterate is returned (not the best one): with exact-K
supports the final and best errors differ by < tol by construction, and the final iterate is the one whose support
the churn/lock-in diagnostics describe.
"""
from __future__ import annotations

import math
import time

import torch

from ember.utils import set_seed
from experiments.varwidth.common import VWConfig, budgets, churn, mask_from_row_widths, pool_floor_cap


# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

def mad_sigma(Yd: torch.Tensor) -> torch.Tensor:
    """sigma_i = 1.4826 * median_j |y_ij - median_j y_ij| over the whole dense row, guarded at 1e-8. (k,)"""
    med = Yd.median(dim=1, keepdim=True).values
    mad = (Yd - med).abs().median(dim=1).values
    return (1.4826 * mad).clamp_min(1e-8)


def fit_widths_to_total(r: torch.Tensor, K: int, lo: int, hi: int) -> torch.Tensor:
    """Raw widths r (k,) >= 0 -> integers in [lo, hi] summing exactly to K (method 3, step 3).

    "Scale to K, clip to [lo, hi], redistribute the clipped excess proportionally, repeat until stable" has the fixed
    point x_i = clip(alpha * r_i, lo, hi) with alpha chosen so sum x = K; we find alpha by bisection (monotone in
    alpha), then round by largest remainder (ties by row index). Rounding last keeps the result inside [lo, hi]: x is
    in [lo, hi] with integer bounds, so floor(x) >= lo and a row receiving +1 had frac > 0, i.e. floor(x) < hi.
    """
    k = r.numel()
    assert k * lo <= K <= k * hi, (k, lo, K, hi)
    x = r.detach().double().cpu().clamp_min(0)
    if float(x.max()) <= 0:
        x = torch.ones_like(x)
    x = x.clamp_min(1e-12 * float(x.max()))            # zero rows still reach hi as alpha -> inf (feasibility)
    a_lo, a_hi = 0.0, 1.0
    while float(x.mul(a_hi).clamp(lo, hi).sum()) < K:
        a_hi *= 2.0
    for _ in range(200):
        a = 0.5 * (a_lo + a_hi)
        if float(x.mul(a).clamp(lo, hi).sum()) < K:
            a_lo = a
        else:
            a_hi = a
    x = x.mul(a_hi).clamp(lo, hi)
    m = x.floor().long()
    R = K - int(m.sum())
    if R > 0:
        frac = torch.where(m < hi, x - m.double(), torch.full_like(x, -1.0))
        order = torch.sort(frac, descending=True, stable=True).indices
        m[order[:R]] += 1
    elif R < 0:                                        # float slack only (sum x slightly > K): take from largest frac
        frac = torch.where(m > lo, x - m.double(), torch.full_like(x, 2.0))
        order = torch.sort(frac, stable=True).indices
        m[order[:-R]] -= 1
    assert int(m.sum()) == K and int(m.min()) >= lo and int(m.max()) <= hi, (int(m.sum()), K)
    return m.to(r.device)


def entropy_widths(Yd: torch.Tensor, K: int, lo: int, hi: int, window: float, c: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Method 3 widths. Per row: top W = ceil(window*n) of |y_dense|, w = max(|y| - c*sigma, 0)^2, p = w / sum w,
    r = exp(H(p)) (m_min if sum w = 0); then fit_widths_to_total. sigma is the method-2 MAD over the whole row.
    Returns (m (k,) long, r (k,) float)."""
    k, n = Yd.shape
    W = max(1, math.ceil(window * n))
    top = torch.topk(Yd.abs(), W, dim=1, sorted=False).values
    w = (top - c * mad_sigma(Yd)[:, None]).clamp_min(0).double() ** 2
    s = w.sum(1)
    p = w / s.clamp_min(1e-300)[:, None]
    H = -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum(1)
    r = torch.where(s > 0, H.exp(), torch.full_like(H, float(lo)))
    return fit_widths_to_total(r, K, lo, hi), r.float()


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------

def refit_support(ZtZ: torch.Tensor, ZtA: torch.Tensor, mask: torch.Tensor, ridge: float) -> torch.Tensor:
    """Per-token least squares on the support: y_{S_j,j} = (Z_S^T Z_S + ridge I)^-1 Z_S^T a_j, zero elsewhere.

    Vectorized by support size: tokens with |S_j| = 1 are a single division; for each larger size s the tokens are
    gathered into one batched (n_s, s, s) solve (float64). Grouping matters because a few tokens can sit in ~30 rows
    while most sit in 0-3: padding everything to the largest size was ~10x slower. Uses only the k x k Gram Z^T Z and
    the k x n Z^T A.
    """
    k, n = mask.shape
    G = ZtZ.double()
    B = ZtA
    cnt = mask.sum(0)                                                      # (n,) support size per token
    Y = torch.zeros(k, n, dtype=torch.float64, device=ZtA.device)
    maskT = mask.T
    # size 1: y = z_i^T a_j / (z_i^T z_i + ridge)
    j1 = torch.nonzero(cnt == 1).squeeze(1)
    if len(j1):
        i1 = maskT[j1].to(torch.uint8).argmax(1)
        Y[i1, j1] = B[i1, j1].double() / (G.diagonal()[i1] + ridge)
    for sz in torch.unique(cnt[cnt > 1]).tolist():
        J = torch.nonzero(cnt == sz).squeeze(1)
        idx = torch.topk(maskT[J].to(torch.uint8), sz, dim=1, sorted=False).indices    # (n_s, sz): exactly S_j
        Gs = G[idx[:, :, None], idx[:, None, :]]
        Gs.diagonal(dim1=1, dim2=2).add_(ridge)
        cols = J[:, None].expand_as(idx)
        sol = torch.linalg.solve(Gs, B[idx, cols].double().unsqueeze(-1)).squeeze(-1)
        Y[idx, cols] = sol
    return Y.to(ZtA.dtype)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def spd_inverse(G: torch.Tensor, ridge: float) -> torch.Tensor:
    """(G + ridge I)^-1 for a k x k Gram G, factorized in float64 (returned in float64).

    Cholesky; if it fails (near-duplicate or dead columns make G singular to working precision, e.g. right after a
    random init), the ridge is raised x10 (starting from 1e-12 x the mean diagonal) until it succeeds.
    """
    G = G.double()
    eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    r = ridge
    scale = float(G.diagonal().mean().clamp_min(1e-30))
    for _ in range(20):
        L, info = torch.linalg.cholesky_ex(G + r * eye)
        if int(info) == 0:
            return torch.cholesky_inverse(L)
        r = max(10 * r, 1e-12 * scale)
    raise torch.linalg.LinAlgError("spd_inverse: Gram not positive definite even with a large ridge")


def _budget_at(it: int, n: int, cfg: VWConfig) -> tuple[int, int, int]:
    """(K_t, m_min_t, m_max_t) at iteration it. Fixed: m_min = m_max = ceil(s*n). Continuation (m1): sparsity falls
    linearly from cont_start to sparsity over cont_iters; m_max_t is raised to ceil(K_t / k) if needed (feasible)."""
    if cfg.method == "fixed":
        K, _, _ = budgets(n, cfg)
        m = K // cfg.rank
        return K, m, m
    if cfg.continuation and it < cfg.cont_iters:
        s_t = cfg.cont_start + (cfg.sparsity - cfg.cont_start) * it / cfg.cont_iters
        K, lo, hi = budgets(n, cfg, sparsity=s_t)
        return K, lo, min(n, max(hi, math.ceil(K / cfg.rank)))
    return budgets(n, cfg)


def _run_loop(A: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor, cfg: VWConfig, n_iters: int, stop: bool,
              lasso_step=None, verbose: bool = False, debug: bool = False, t_start: float | None = None,
              tag: str = "fit") -> dict:
    """Runs up to n_iters iterations of the shared loop from (Z, Y) and returns the fit dict (minus config/seed).

    stop=False runs exactly n_iters iterations (merge step); stop=True applies the stopping rule after the
    continuation phase. Z is only used for its shape: step 1 recomputes it from Y.
    """
    t_start = time.perf_counter() if t_start is None else t_start
    d, n = A.shape
    k = cfg.rank
    lam = cfg.ridge
    normA2 = float(A.double().pow(2).sum())
    t_free = cfg.cont_iters if (cfg.continuation and cfg.method == "m1") else 0

    churn_hist, err_hist = [], []
    extra: dict = {}
    if cfg.method == "m3":
        extra.update(m3_update_iters=[], m3_widths=[], m3_raw=None)
    if cfg.method == "m4":
        extra.update(m4_calls=[])
        lasso_state: dict = {}
    m3_m = None
    prev_mask = None
    YYt = Y.double() @ Y.double().T
    best, no_imp, low_churn = float("inf"), 0, 0
    stop_reason = "max_iter"
    it = -1

    for it in range(n_iters):
        K, lo, hi = _budget_at(it, n, cfg)

        # 1. Z-update (dense GEMM: faster than a sparse product at 1% density on CPU)
        AYt = A @ Y.T                                                          # (d, k)
        Z = (AYt.double() @ spd_inverse(YYt, lam)).float()                     # (d, k); k x k work in float64

        # 2. dense Y-solve; Z^T A is kept for the refit and the error
        ZtA = Z.T @ A                                                          # (k, n)
        Zd = Z.double()
        ZtZ = Zd.T @ Zd                                                        # (k, k) float64
        Yd = (spd_inverse(ZtZ, lam) @ ZtA.double()).float()                    # (k, n); float64: ZtZ can be ill-conditioned

        # 3. allocation
        if cfg.method == "fixed":
            mask = mask_from_row_widths(Yd.abs(), torch.full((k,), lo, dtype=torch.long))
        elif cfg.method == "m1":
            mask = pool_floor_cap(Yd.abs() * Z.norm(dim=0)[:, None], K, lo, hi)
        elif cfg.method == "m2":
            mask = pool_floor_cap(Yd.abs() / mad_sigma(Yd)[:, None], K, lo, hi)
        elif cfg.method in ("rn", "rnn"):
            # pooling on the row-normalized dense Y (magnitude stays in Z): |y_ij| / ||y_i||_2;
            # rnn first subtracts the method-3 noise floor of the whole row: max(|y_ij| - c * sigma_i, 0)
            row = Yd.norm(dim=1, keepdim=True).clamp_min(1e-12)
            score = Yd.abs()
            if cfg.method == "rnn":
                score = (score - cfg.m3_c * mad_sigma(Yd)[:, None]).clamp_min(0)
            mask = pool_floor_cap(score / row, K, lo, hi)
        elif cfg.method == "m3":
            if it % cfg.m3_every == 0:
                m3_m, raw = entropy_widths(Yd, K, lo, hi, cfg.m3_window, cfg.m3_c)
                extra["m3_update_iters"].append(it)
                extra["m3_widths"].append(m3_m.tolist())
                extra["m3_raw"] = raw.tolist()
            mask = mask_from_row_widths(Yd.abs(), m3_m)
        elif cfg.method == "m4":
            if it % cfg.m4_every == 0 or prev_mask is None:
                if lasso_step is None:
                    from experiments.varwidth.lasso import lasso_step
                mask, lasso_state = lasso_step(A, Z, Yd, K, lo, hi, lasso_state)
                extra["m4_calls"].append({"iter": it, **{kk: v for kk, v in lasso_state.items()
                                                         if isinstance(v, (int, float, bool, str))}})
            else:
                mask = prev_mask
        else:
            raise ValueError(cfg.method)

        # 4. refit on the support (or the paper's masked dense solve)
        Y = refit_support(ZtZ, ZtA, mask, lam) if cfg.refit else Yd * mask

        # error from the expansion (Z, Y before the rescale; Z Y is invariant to it)
        Ydbl = Y.double()
        YYt = Ydbl @ Ydbl.T
        e2 = normA2 - 2.0 * float((ZtA * Y).sum(dtype=torch.float64)) + float((ZtZ * YYt).sum())
        err = math.sqrt(max(e2, 0.0) / normA2)
        if not math.isfinite(err):
            raise FloatingPointError(f"non-finite rel_err at iter {it}")

        if debug:
            nnz_rows = mask.sum(1)
            assert int(mask.sum()) == K, (it, int(mask.sum()), K)
            assert int(nnz_rows.min()) >= lo and int(nnz_rows.max()) <= hi, (it, int(nnz_rows.min()), int(nnz_rows.max()), lo, hi)
            assert not bool(((Y != 0) & ~mask).any()), it
            if it % 50 == 0:
                direct = float(torch.linalg.norm(A - Z @ Y) / math.sqrt(normA2))
                assert abs(direct - err) < 1e-4, (it, direct, err)

        # 5. unit rows of Y, scale into Z (rows that are all zero keep scale 1)
        nr = YYt.diagonal().clamp_min(0).sqrt()
        nr = torch.where(nr > 0, nr, torch.ones_like(nr))
        Y = Y / nr.to(Y.dtype)[:, None]
        Z = Z * nr.to(Z.dtype)[None, :]
        YYt = YYt / (nr[:, None] * nr[None, :])

        c = churn(prev_mask, mask, K)
        churn_hist.append(c)
        err_hist.append(err)
        prev_mask = mask

        if verbose and (it % 100 == 0):
            w = mask.sum(1)
            print(f"[{tag}] iter {it:5d}: rel_err={err:.5f} churn={c:.4f} best={best:.5f} no_imp={no_imp} "
                  f"low_churn={low_churn} widths min/med/max={int(w.min())}/{int(w.median())}/{int(w.max())}")

        if it < t_free:
            continue
        if err < best - cfg.tol:
            best, no_imp = err, 0
        else:
            no_imp += 1
        low_churn = low_churn + 1 if c < cfg.churn_tol else 0
        if stop and no_imp >= cfg.patience and low_churn >= cfg.patience:
            stop_reason = "converged"
            break

    K, lo, hi = _budget_at(it, n, cfg)
    return {"Z": Z.float().cpu(), "Y": Y.float().cpu(), "widths": prev_mask.sum(1).long().cpu(),
            "churn": churn_hist, "rel_err_hist": err_hist, "rel_err": err_hist[-1], "n_iter": len(err_hist),
            "lockin_iter": lockin_iter(churn_hist, cfg.churn_tol), "elapsed": time.perf_counter() - t_start,
            "stop_reason": stop_reason, "K": K, "m_min": lo, "m_max": hi, "extra": extra}


def lockin_iter(churn_hist: list[float], tol: float) -> int:
    """First iteration t with churn < tol at every iteration from t to the end; -1 if the last churn is >= tol."""
    t = len(churn_hist)
    while t > 0 and churn_hist[t - 1] < tol:
        t -= 1
    return t if t < len(churn_hist) else -1


def fit_varwidth(A: torch.Tensor, cfg: VWConfig, seed: int, device: str = "cpu", verbose: bool = False,
                 lasso_step=None, debug: bool = False) -> dict:
    """One fit of the shared loop. A: (d, n). Returns the fit.pt dict (see common.py's docstring).

    Init "randn": the paper's N(0,1) Y (drawn as an (n, k) tensor and transposed, the paper's G0 layout) from a
    torch.Generator seeded with `seed` (ember.utils.set_seed(seed) is also called); Z is drawn too but step 1
    overwrites it. Init "svd": Z = U_k S_k, Y = V_k^T. debug=True asserts exactly K ones and row bounds at every
    iteration and checks the error expansion against ||A - ZY|| every 50 iterations.
    """
    t0 = time.perf_counter()
    if cfg.continuation and cfg.method != "m1":
        raise ValueError("continuation is defined for method m1 only")
    set_seed(seed)
    A = A.detach().to(device=device, dtype=torch.float32).contiguous()
    d, n = A.shape
    k = cfg.rank
    if cfg.init == "randn":
        gen = torch.Generator().manual_seed(int(seed))
        Y = torch.randn(n, k, generator=gen).T.contiguous().to(device)
        Z = torch.randn(d, k, generator=gen).to(device)
    else:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        Z, Y = (U[:, :k] * S[:k]).contiguous(), Vh[:k].contiguous()
    out = _run_loop(A, Z, Y, cfg, cfg.max_iter, stop=True, lasso_step=lasso_step, verbose=verbose, debug=debug,
                    t_start=t0, tag=f"{cfg.method} seed {seed}")
    out["config"] = cfg.to_dict()
    out["seed"] = seed
    return out


# ---------------------------------------------------------------------------
# Method 5: merge
# ---------------------------------------------------------------------------

def merge_pair(z_a: torch.Tensor, y_a: torch.Tensor, z_b: torch.Tensor, y_b: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-1 approximation (u, y) of M = z_a y_a + z_b y_b (d x n, rank <= 2) without forming M.

    M = Q Y2 with Q = [z_a z_b], Y2 = [y_a; y_b]. With Q = Qo R (thin QR), M M^T = Qo (R Y2 Y2^T R^T) Qo^T, so the top
    left singular vector is u = Qo v, v the top eigenvector of the 2 x 2 matrix R (Y2 Y2^T) R^T; y = u^T M = (u^T Q) Y2.
    u has unit norm; y is supported on the union of the two supports.
    """
    Q = torch.stack([z_a, z_b], 1).double()
    Y2 = torch.stack([y_a, y_b], 0).double()
    Qo, R = torch.linalg.qr(Q)
    C = R @ (Y2 @ Y2.T) @ R.T
    evals, evecs = torch.linalg.eigh(C)
    u = Qo @ evecs[:, -1]
    y = (u @ Q) @ Y2
    return u.float(), y.float()


def merge_features(A: torch.Tensor, fit: dict, pairs: list[tuple[int, int]], scores_fn=None,
                   cfg: VWConfig | None = None, seed: int = 0, extra_iters: int = 300, device: str = "cpu",
                   lasso_step=None, verbose: bool = False) -> dict:
    """Method 5: merge each pair (a, b) into slot a, re-seed slot b, then run extra_iters more iterations.

    For each pair: merged (u, y) = merge_pair(z_a, y_a, z_b, y_b); the support (union of the two) is trimmed to m_max
    by scores_fn(y, u) (default |y| * ||u||, i.e. method 1's score; u is unit so this is |y|); slot a <- (u, y), slot b
    <- a fresh random feature: z ~ N(0, I), y ~ N(0, 1) on ceil(sparsity*n) random tokens (the mean width, so the
    state stays near the K budget; the next allocation restores exactly K). Pairs that reuse a slot already touched
    by an earlier pair are skipped (logged). The loop then runs exactly extra_iters iterations (no early stop) with
    the fit's config (continuation off) and returns a new fit dict with extra["merged_pairs"] / ["skipped_pairs"];
    n_iter / churn / rel_err_hist cover only the extra iterations and extra["rel_err_before"] is the input fit's.
    """
    t0 = time.perf_counter()
    cfg = cfg or VWConfig(**fit["config"])
    cfg = VWConfig(**{**cfg.to_dict(), "continuation": False})
    A = A.detach().to(device=device, dtype=torch.float32).contiguous()
    d, n = A.shape
    Z = fit["Z"].clone().float().to(device)
    Y = fit["Y"].clone().float().to(device)
    _, _, m_max = budgets(n, cfg)
    gen = torch.Generator().manual_seed(int(seed))
    width0 = math.ceil(cfg.sparsity * n)
    used, merged, skipped = set(), [], []
    for a, b in pairs:
        a, b = int(a), int(b)
        if a == b or a in used or b in used:
            skipped.append((a, b))
            continue
        used.update((a, b))
        u, y = merge_pair(Z[:, a], Y[a], Z[:, b], Y[b])
        support = (Y[a] != 0) | (Y[b] != 0)
        score = scores_fn(y, u) if scores_fn is not None else y.abs() * u.norm()
        score = torch.where(support, score, torch.full_like(score, -1.0))
        n_keep = min(int(support.sum()), m_max)
        keep = torch.zeros(n, dtype=torch.bool, device=y.device)
        keep[torch.sort(score, descending=True, stable=True).indices[:n_keep]] = True
        Z[:, a], Y[a] = u, y * keep
        Z[:, b] = torch.randn(d, generator=gen).to(device)
        yb = torch.zeros(n, device=device)
        yb[torch.randperm(n, generator=gen)[:width0].to(device)] = torch.randn(width0, generator=gen).to(device)
        Y[b] = yb
        merged.append({"a": a, "b": b, "union": int(support.sum()), "kept": n_keep,
                       "cos_ab": float(torch.nn.functional.cosine_similarity(fit["Z"][:, a], fit["Z"][:, b], dim=0))})
    out = _run_loop(A, Z, Y, cfg, extra_iters, stop=False, lasso_step=lasso_step, verbose=verbose, t_start=t0,
                    tag="merge")
    out["extra"].update(merged_pairs=merged, skipped_pairs=skipped, rel_err_before=fit["rel_err"])
    out["config"] = cfg.to_dict()
    out["seed"] = seed
    return out
