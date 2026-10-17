"""Variable per-feature sparsity for the Step 3.1 Sparse MF (the "varnnz" notebooks).

Same ALS fit as ``ember.utils.SparseMatrixFactorization`` (A ~ F G^T, F = Z dense, G = Y^T WTA-sparse), but the
winner-takes-all step no longer keeps the same ceil(s*n) tokens in every feature:

    cfg = SparsityConfig(mode="per_row", dist="uniform")       # knobs: see SparsityConfig
    best = fit_best_of(A, cfg, rank=100, n_restarts=3, seed=42, max_iter=20_000, patience=500, base_reg=1e-4)
    Z, Y = best["Z"], best["Y"]                                  # (d, k), (k, n) as in the notebooks

Modes (every mode keeps the same total budget M = k * ceil(s * n), so reconstruction errors are comparable):
  per_row          row i keeps m_i tokens, m drawn once from `dist` (constant | uniform | loguniform | gaussian),
                   clipped to [floor, cap] and rescaled to sum to M. `assign="matched"` re-sorts the drawn budgets onto
                   the rows by the participation ratio of each row's top 2*ceil(s*n) magnitudes during the
                   early phase.
  pooled_rownorm   one pool of M entries, scored |G_ti| / ||G_:,i||  (rows of Y unit-norm, z takes the magnitude)
  pooled_contrib   one pool of M entries, scored |G_ti| * ||F_:,i||  (z unit-norm: contribution to the reconstruction)

Guardrails, combinable with any mode:
  floor / cap      per-row min / max fraction of n
  anneal           budgets start at `anneal_start` (fraction of n per row) and shrink geometrically to the target
                   over `anneal_iters` iterations
  noise            support selection uses scores * exp(sigma_t * eps), sigma_t decaying linearly from `noise_sigma0`
                   to 0 over `noise_iters`; kept values are always the exact least-squares values

Best-loss tracking and patience start only once anneal, noise and matching are over (`t_free`), so the denser early
phase is never returned as the "best" fit. With mode="per_row", dist="constant", anneal=False, noise_sigma0=0 the fit is
identical to SparseMatrixFactorization (budgets and noise use their own generator, not the global RNG).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field

import pandas as pd
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from ember.utils import SparseMatrixFactorization, set_seed
from factorization.seminmf import fix_hoyer_scale, init_knn, init_svd, wta_features

MODES = ("per_row", "pooled_rownorm", "pooled_contrib")
DISTS = ("constant", "uniform", "loguniform", "gaussian")
ASSIGNS = ("random", "matched")


@dataclass
class SparsityConfig:
    mode: str = "per_row"
    dist: str = "uniform"
    assign: str = "random"
    sparsity: float = 0.01                      # s: mean fraction of n kept per row (sets the total budget M)
    uniform_range: tuple = (0.005, 0.015)       # U(lo, hi), fractions of n
    loguniform_range: tuple = (0.0025, 0.04)    # log-uniform over [lo, hi]: 16x spread, geometric mean 1%
    gauss_std: float = 0.003                    # N(s, std), fraction of n
    floor: float = 0.002                        # per-row min fraction of n
    cap: float = 0.05                           # per-row max fraction of n
    anneal: bool = True
    anneal_start: float = 0.10                  # per-row fraction of n at iteration 0
    anneal_iters: int = 1500
    noise_sigma0: float = 0.5                   # 0 disables the noisy top-m
    noise_iters: int | None = None              # None: anneal_iters if anneal else 1000
    match_every: int = 50
    match_iters: int | None = None              # None: the anneal/noise phase (1000 if both are off)

    def __post_init__(self):
        self.uniform_range = tuple(self.uniform_range)
        self.loguniform_range = tuple(self.loguniform_range)
        assert self.mode in MODES, self.mode
        assert self.dist in DISTS, self.dist
        assert self.assign in ASSIGNS, self.assign
        assert 0 < self.floor <= self.sparsity <= self.cap <= 1, (self.floor, self.sparsity, self.cap)
        assert not self.anneal or self.anneal_start >= self.sparsity, (self.anneal_start, self.sparsity)

    # ---- phase lengths ----
    @property
    def noise_len(self) -> int:
        if self.noise_sigma0 <= 0:
            return 0
        return self.noise_iters if self.noise_iters is not None else (self.anneal_iters if self.anneal else 1000)

    @property
    def anneal_len(self) -> int:
        return self.anneal_iters if self.anneal else 0

    @property
    def match_len(self) -> int:
        if self.mode != "per_row" or self.assign != "matched":
            return 0
        if self.match_iters is not None:
            return self.match_iters
        return max(self.anneal_len, self.noise_len) or 1000

    @property
    def t_free(self) -> int:
        """First iteration at which the support rule is final; best-loss tracking starts here."""
        return max(self.anneal_len, self.noise_len, self.match_len)

    # ---- cache keys ----
    def to_meta(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    def slug(self) -> str:
        parts = [self.mode]
        if self.mode == "per_row":
            parts.append(self.dist)
            if self.assign != "random":
                parts.append(self.assign)
        if self.anneal:
            parts.append(f"an{self.anneal_start:g}x{self.anneal_iters}")
        if self.noise_sigma0 > 0:
            parts.append(f"nz{self.noise_sigma0:g}")
        h = hashlib.sha1(json.dumps(self.to_meta(), sort_keys=True).encode()).hexdigest()[:6]
        return "-".join(parts) + f"-{h}"


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

def row_bounds(cfg: SparsityConfig, n: int) -> tuple[int, int]:
    """Per-row (floor, cap) as token counts."""
    lo = max(1, math.ceil(cfg.floor * n))
    hi = min(n, max(lo, math.floor(cfg.cap * n), math.ceil(cfg.sparsity * n)))   # cap >= target, so M stays feasible
    return lo, hi


def total_budget(cfg: SparsityConfig, n: int, k: int) -> int:
    """M = k * ceil(s * n): the number of non-zeros the fixed-budget fit keeps."""
    return k * max(1, math.ceil(cfg.sparsity * n))


def _fit_to_total(x: torch.Tensor, total: int, lo: int, hi: int, iters: int = 50) -> torch.Tensor:
    """Float budgets -> integers in [lo, hi] summing exactly to `total` (rescale/clip, then largest remainder)."""
    k = x.numel()
    assert k * lo <= total <= k * hi, f"total budget {total} infeasible for {k} rows in [{lo}, {hi}]"
    x = x.double().clamp_min(1e-12)
    for _ in range(iters):
        x = (x * total / x.sum()).clamp(lo, hi)
        if abs(float(x.sum()) - total) < 0.5:
            break
    m = x.floor().long().clamp(lo, hi)
    diff = total - int(m.sum())
    frac = x - m.double()
    if diff > 0:                                   # hand out the remainder to the largest fractional parts with room
        order = torch.argsort(torch.where(m < hi, frac, torch.full_like(frac, -1.0)), descending=True)
        for i in order.tolist():
            if diff == 0:
                break
            if m[i] < hi:
                m[i] += 1
                diff -= 1
    while diff < 0:                                # rounding overshoot (rare): take from the largest rows above lo
        i = int(torch.argmax(torch.where(m > lo, m, torch.zeros_like(m))))
        m[i] -= 1
        diff += 1
    assert int(m.sum()) == total and int(m.min()) >= lo and int(m.max()) <= hi
    return m


def draw_budgets(cfg: SparsityConfig, n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """Per-row target budgets m (k,) for mode='per_row'; sums to total_budget(cfg, n, k)."""
    lo, hi = row_bounds(cfg, n)
    total = total_budget(cfg, n, k)
    if cfg.dist == "constant":
        return torch.full((k,), total // k, dtype=torch.long)
    if cfg.dist == "uniform":
        a, b = cfg.uniform_range
        frac = a + (b - a) * torch.rand(k, generator=gen, dtype=torch.float64)
    elif cfg.dist == "loguniform":
        a, b = cfg.loguniform_range
        frac = torch.exp(math.log(a) + (math.log(b) - math.log(a)) * torch.rand(k, generator=gen, dtype=torch.float64))
    elif cfg.dist == "gaussian":
        frac = cfg.sparsity + cfg.gauss_std * torch.randn(k, generator=gen, dtype=torch.float64)
    else:
        raise ValueError(cfg.dist)
    return _fit_to_total(frac.clamp_min(cfg.floor) * n, total, lo, hi)


def participation_ratio(G: torch.Tensor) -> torch.Tensor:
    """(sum |g|)^2 / sum g^2 per column of G (n, k): roughly how many tokens carry a feature's mass.

    Returns (k,) in G's float dtype (float32 for non-float G); all-zero columns -> 0.
    """
    a = G.detach().abs().double()
    l2 = a.norm(dim=0)                                         # computed as (||g||_1 / ||g||_2)^2: no overflow
    pr = torch.where(l2 > 0, (a.sum(0) / l2.clamp_min(1e-300)) ** 2, torch.zeros_like(l2))
    return pr.to(G.dtype if G.is_floating_point() else torch.float32)


def match_budgets(m: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Re-sort the budget values m (k,) onto the columns of G: widest-looking column gets the largest budget.

    Rank matching by participation ratio: the i-th largest PR column gets the i-th largest budget. Same multiset of
    values as m (so the sum is unchanged), same dtype/device as m. Ties in PR are broken by column index (stable).
    """
    assert m.ndim == 1 and m.numel() == G.shape[1], (tuple(m.shape), tuple(G.shape))
    order = torch.argsort(participation_ratio(G).cpu(), descending=True, stable=True).to(m.device)
    out = torch.empty_like(m)
    out[order] = torch.sort(m, descending=True).values
    return out


# ---------------------------------------------------------------------------
# Support selection
# ---------------------------------------------------------------------------

def topk_per_row(scores: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """0/1 mask (n, k) keeping the top m[i] entries of column i of scores."""
    n = scores.shape[0]
    mmax = min(n, int(m.max()))
    _, idx = torch.topk(scores, mmax, dim=0, largest=True, sorted=True)              # (mmax, k)
    keep = (torch.arange(mmax, device=scores.device)[:, None] < m.to(scores.device)[None, :]).to(scores.dtype)
    return torch.zeros_like(scores).scatter_(0, idx, keep)


def topk_pooled(scores: torch.Tensor, total: int, floor: int, cap: int) -> torch.Tensor:
    """0/1 mask (n, k): top `floor` per column forced, the rest of `total` filled by a global top-k over ranks floor..cap-1.

    scores (n, k) >= 0. Every column keeps between `floor` and `cap` entries and exactly `total` are kept overall.
    Requires 0 <= floor <= cap and k * floor <= total <= k * cap (after clamping cap to n).
    """
    n, k = scores.shape
    total, floor, cap = int(total), int(floor), min(int(cap), n)
    if not 0 <= floor <= cap:
        raise ValueError(f"topk_pooled: need 0 <= floor <= cap, got floor={floor}, cap={cap} (n={n})")
    if not k * floor <= total <= k * cap:
        raise ValueError(f"topk_pooled: total={total} infeasible for k={k} columns with per-column "
                         f"floor={floor}, cap={cap} (need {k * floor} <= total <= {k * cap})")
    if cap == 0:
        return torch.zeros_like(scores)
    vals, idx = torch.topk(scores, cap, dim=0, largest=True, sorted=True)            # (cap, k), ranks 0..cap-1
    keep = torch.zeros(cap, k, dtype=torch.bool, device=scores.device)
    keep[:floor] = True                                                               # forced per-column floor
    extra = total - k * floor
    if extra > 0:                                                                     # pooled fill, ranks floor..cap-1
        _, j = torch.topk(vals[floor:].reshape(-1), extra, largest=True, sorted=False)
        keep[floor:].view(-1)[j] = True
    return torch.zeros_like(scores).scatter_(0, idx, keep.to(scores.dtype))


# ---------------------------------------------------------------------------
# The factorization
# ---------------------------------------------------------------------------

class VariableSparsityMF(SparseMatrixFactorization):
    """SparseMatrixFactorization with a per-feature / pooled, annealed, noisy WTA (see the module docstring)."""

    def __init__(self, rank: int, fitting_device: str = "cpu", cfg: SparsityConfig | None = None,
                 seed: int = 0, verbose: bool = False, progress: bool = False, desc: str = "fit"):
        cfg = cfg or SparsityConfig()
        super().__init__(rank, fitting_device=fitting_device, g_sparsity=cfg.sparsity)
        self.cfg = cfg
        self.seed = seed
        self.verbose = verbose
        self.progress = progress   # tqdm bar over iterations (phase, loss, best, patience)
        self.desc = desc

    # ---- schedule ----
    def _anneal_factor(self, it: int) -> float:
        """Multiplier on the target budgets: anneal_start/s at it=0, geometric down to 1 at anneal_iters."""
        c = self.cfg
        if not c.anneal or it >= c.anneal_iters:
            return 1.0
        return (c.anneal_start / c.sparsity) ** (1.0 - it / c.anneal_iters)

    def _sigma(self, it: int) -> float:
        L = self.cfg.noise_len
        return 0.0 if it >= L else self.cfg.noise_sigma0 * (1.0 - it / L)

    def _scores(self) -> torch.Tensor:
        G = self.G_.data
        if self.cfg.mode == "per_row":
            return G.abs()
        if self.cfg.mode == "pooled_rownorm":
            return G.abs() / G.norm(dim=0, keepdim=True).clamp_min(1e-12)
        if self.cfg.mode == "pooled_contrib":
            return G.abs() * self.F_.data.norm(dim=0, keepdim=True)
        raise ValueError(self.cfg.mode)

    def _sparsify(self, it: int) -> None:
        c, G = self.cfg, self.G_.data
        n, k = G.shape
        if c.mode == "per_row" and c.assign == "matched" and it < c.match_len and it % c.match_every == 0:
            # PR over each column's top-w magnitudes only: on the dense column the ~n small background entries
            # dominate, so narrow features look wide (rank correlation with the true width flips sign).
            w = min(n, 2 * math.ceil(c.sparsity * n))
            self.m = match_budgets(self.m, torch.topk(G.abs(), w, dim=0).values)

        scores = self._scores()
        sigma = self._sigma(it)
        if sigma > 0:
            eps = torch.randn(scores.shape, generator=self._gen, dtype=torch.float32).to(scores.device, scores.dtype)
            scores = scores * torch.exp(sigma * eps)

        f = self._anneal_factor(it)
        if c.mode == "per_row":
            m_t = self.m if f == 1.0 else torch.clamp((self.m.double() * f).ceil().long(), max=n)
            mask = topk_per_row(scores, m_t)
        else:
            total_t = min(n * k, math.ceil(self.total * f))
            cap_t = min(n, max(self.cap, math.ceil(total_t / k)))
            mask = topk_pooled(scores, total_t, self.floor, cap_t)
        G.mul_(mask)

    def fit(self, A, max_iter=500, tol=1e-4, reg=1e-6, patience=100, verbose=None, init="random"):
        verbose = self.verbose if verbose is None else verbose
        A = A.to(self.fitting_device)
        d_model, n_tokens = A.shape
        K = self.rank
        c = self.cfg

        # Budgets and noise use their own generator, so the global RNG stream (init) matches the parent class.
        self._gen = torch.Generator().manual_seed(int(self.seed))
        self.floor, self.cap = row_bounds(c, n_tokens)
        self.total = total_budget(c, n_tokens, K)
        self.m = draw_budgets(c, n_tokens, K, self._gen) if c.mode == "per_row" else None
        self.m_drawn = None if self.m is None else self.m.clone()

        if init == "knn":
            F0, G0 = init_knn(A, K)
        elif init == "svd":
            F0, G0 = init_svd(A, K)
        elif init == "random":
            G0 = torch.randn(n_tokens, K, device=self.fitting_device)
            F0 = torch.randn(d_model, K, device=self.fitting_device)
        else:
            raise ValueError(f"Unknown init '{init}'")

        self.G_ = nn.Parameter(G0.to(self.fitting_device))
        self.F_ = nn.Parameter(F0.to(self.fitting_device))

        self.loss_history = []
        best_F, best_G = self._loop(A, 0, max_iter, tol, reg, patience, verbose, float("inf"), None, None, 0)
        self._finish(best_F, best_G)
        return self

    def _loop(self, A, it0, max_iter, tol, reg, patience, verbose, best_loss, best_F, best_G, num_no_improve):
        """ALS iterations it0..max_iter-1 from the current F_, G_; returns the best post-t_free (F, G) or (None, None).

        The loop state lives in this frame's locals, so resume_interrupted() can pick it up from a traceback.
        """
        K = self.rank
        t_free = self.cfg.t_free
        eye_K = torch.eye(K, device=self.fitting_device)
        bar = tqdm(total=max_iter, initial=it0, desc=getattr(self, "desc", "fit"), mininterval=1.0) \
            if getattr(self, "progress", False) else None

        with torch.no_grad():
            for it in range(it0, max_iter):
                if bar is not None:
                    bar.update(1)
                GtG = self.G_.T @ self.G_
                inv_G = torch.linalg.inv(GtG + eye_K * reg)
                self.F_.data.copy_(A @ self.G_ @ inv_G)
                wta_features(self.F_, pct_keep=self.sparsity)  # no-op (sparsity=1.0)
                fix_hoyer_scale(self.F_, self.G_)

                FtF = self.F_.T @ self.F_
                inv_F = torch.linalg.inv(FtF + eye_K * reg)
                self.G_.data.copy_(A.T @ self.F_ @ inv_F)

                self._sparsify(it)

                loss = torch.norm(A - self.F_ @ self.G_.T, p="fro") ** 2
                self.loss_history.append(loss.item())
                if not math.isfinite(loss.item()):
                    raise FloatingPointError(f"non-finite loss at iter {it}")

                if it < t_free:   # anneal / noise / matching still running: the support rule is not final yet
                    if verbose and it % 50 == 0:
                        print(f"Iter {it:4d}: loss={loss.item():.6f}  (early phase until iter {t_free}, no best tracking yet)")
                    continue

                if loss.item() < best_loss - tol:
                    best_loss = loss.item()
                    best_F = self.F_.data.clone()
                    best_G = self.G_.data.clone()
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if verbose and (it % 50 == 0 or num_no_improve == 1):
                    print(f"Iter {it:4d}: loss={loss.item():.6f}  (best={best_loss:.6f}, no_improve={num_no_improve})")

                if num_no_improve >= patience:
                    if verbose:
                        print(f"Stopping early at iter {it} (no improvement in {patience} iters)")
                    break

        if bar is not None:
            bar.close()
        return best_F, best_G

    def _finish(self, best_F, best_G) -> None:
        if best_F is not None:
            with torch.no_grad():
                self.F_.data.copy_(best_F)
                self.G_.data.copy_(best_G)
        self.n_iter = len(self.loss_history)
        self.t_free = self.cfg.t_free
        self.nnz_per_row = (self.G_.data != 0).sum(0).cpu()     # (k,) final tokens per feature
        self.W = self.G_.detach().clone()
        self.H = self.F_.T


# ---------------------------------------------------------------------------
# Restarts
# ---------------------------------------------------------------------------

def _fit_with_ridge(nmf, A: torch.Tensor, max_iter: int, patience: int = 500, base_reg: float = 1e-4,
                    tries: int = 4) -> float:
    """ember.utils.fit_with_ridge, but retrying (ridge x10) only on numerical failures, so real errors surface at once."""
    reg = float(base_reg)
    for t in range(tries):
        try:
            nmf.fit(A, max_iter, patience=patience, reg=reg)
            return reg
        except (torch.linalg.LinAlgError, FloatingPointError) as e:
            if t == tries - 1:
                raise RuntimeError(f"fit failed after {tries} tries (last ridge {reg:g})") from e
            reg *= 10.0


def fit_best_of(A: torch.Tensor, cfg: SparsityConfig, rank: int, n_restarts: int, seed: int, max_iter: int,
                patience: int, base_reg: float, device: str = "cpu", concept_metrics_fn=None,
                verbose: bool = True, progress: bool = False, _resume: dict | None = None) -> dict:
    """Fits `n_restarts` times (seed + r) and keeps the lowest relative error ||A - ZY||_F / ||A||_F.

    concept_metrics_fn(Y) -> dict, optional: extra per-restart columns (e.g. top rho), computed on the (k, n) Y.
    Returns {"Z", "Y", "ridge_used", "budgets", "nnz_per_row", "loss_history", "t_free", "best_restart",
             "restarts" (DataFrame), "elapsed"}.
    `_resume` is internal (see resume_interrupted): finish an interrupted restart, then run the remaining ones.
    """
    if max_iter <= cfg.t_free:
        raise ValueError(f"max_iter={max_iter} <= t_free={cfg.t_free}: the fit would end inside the anneal/noise phase "
                         f"and return a non-final (denser) support; raise max_iter or shorten anneal_iters/noise_iters")
    A_cpu = A.detach().float().cpu()
    A_norm = torch.norm(A_cpu).item()
    rows, best, t0, r0 = [], None, time.perf_counter(), 0
    if _resume is not None:
        rows, best, t0, r0 = list(_resume["rows"]), _resume["best"], _resume["t0"], _resume["r"]
    for r in range(r0, n_restarts):
        s = seed + r
        if _resume is not None and r == r0:
            smf, t_r, ridge, st = _resume["smf"], _resume["t_r"], _resume["reg"], _resume["loop"]
            it0 = len(smf.loss_history)
            print(f"resuming restart {r} (seed {s}) at iter {it0} (early phase until iter {cfg.t_free})")
            best_F, best_G = VariableSparsityMF._loop(
                smf, A.to(smf.fitting_device), it0, max_iter, st["tol"], ridge, patience, verbose,
                st["best_loss"], st["best_F"], st["best_G"], st["num_no_improve"])
            VariableSparsityMF._finish(smf, best_F, best_G)
        else:
            set_seed(s)
            smf = VariableSparsityMF(rank, fitting_device=device, cfg=cfg, seed=s, verbose=verbose, progress=progress,
                                     desc=f"restart {r} (seed {s}) of {n_restarts}")
            t_r = time.perf_counter()
            ridge = _fit_with_ridge(smf, A, max_iter, patience=patience, base_reg=base_reg)
        Z = smf.F_.detach().float().cpu()       # (d, k)
        Y = smf.G_.detach().float().cpu().T     # (k, n)
        err = (torch.norm(A_cpu - Z @ Y) / A_norm).item()
        nnz = smf.nnz_per_row
        row = {"restart": r, "seed": s, "rel_err": err, "ridge": ridge, "iters": smf.n_iter,
               "nnz_total": int(nnz.sum()), "nnz_min": int(nnz.min()), "nnz_median": float(nnz.float().median()),
               "nnz_max": int(nnz.max()), "seconds": time.perf_counter() - t_r}
        if concept_metrics_fn is not None:
            row.update(concept_metrics_fn(Y))
        rows.append(row)
        print(f"restart {r} (seed {s}): rel_err={err:.4f}  iters={smf.n_iter}  "
              f"nnz/row min={row['nnz_min']} median={row['nnz_median']:.0f} max={row['nnz_max']}")
        if best is None or err < best["rel_err"]:
            best = {"rel_err": err, "Z": Z, "Y": Y, "ridge_used": ridge, "best_restart": r,
                    "budgets": None if smf.m is None else smf.m.clone(), "budgets_drawn": smf.m_drawn,
                    "nnz_per_row": nnz.clone(), "loss_history": list(smf.loss_history), "t_free": smf.t_free}
        del smf
    best["restarts"] = pd.DataFrame(rows)
    best["elapsed"] = time.perf_counter() - t0   # wall time, including any pause before a resume
    return best


def resume_interrupted(tb=None) -> dict:
    """Finish a fit_best_of call that was stopped (KeyboardInterrupt) and return what fit_best_of would have returned.

    Reads the interrupted frames from the last traceback (IPython keeps it in sys.last_traceback): the model with its
    current F_/G_, loss history and budgets, the loop state (best loss, best F/G, patience counter) and the finished
    restarts. Call it before another error replaces that traceback. Works on models built by an older version of
    this module too (same attributes).
    """
    import sys
    tb = tb or getattr(sys, "last_traceback", None)
    if tb is None:
        raise RuntimeError("no traceback found: resume_interrupted() must run right after the interrupted cell")
    outer = loop = None
    while tb is not None:
        f = tb.tb_frame
        if f.f_code.co_name == "fit_best_of":
            outer = dict(f.f_locals)
        elif "best_loss" in f.f_locals and "self" in f.f_locals and hasattr(f.f_locals["self"], "loss_history"):
            loop = dict(f.f_locals)       # VariableSparsityMF._loop (or fit, in older versions of this module)
        tb = tb.tb_next
    if outer is None or loop is None:
        raise RuntimeError("the last traceback is not an interrupted fit_best_of fit (was it stopped inside a fit?)")
    smf = loop["self"]
    kw = {k: outer[k] for k in ("A", "cfg", "rank", "n_restarts", "seed", "max_iter", "patience", "base_reg",
                                 "device", "concept_metrics_fn", "verbose")}
    state = {"smf": smf, "rows": outer["rows"], "best": outer["best"], "t0": outer["t0"], "r": outer["r"],
             "t_r": outer.get("t_r", time.perf_counter()), "reg": loop["reg"],
             "loop": {k: loop[k] for k in ("tol", "best_loss", "best_F", "best_G", "num_no_improve")}}
    return fit_best_of(**kw, _resume=state)
