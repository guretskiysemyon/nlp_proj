"""Shared pieces of the variable-width sparse MF study (instructions_new.txt).

Layout of experiments/varwidth/:
    common.py    this file: VWConfig, exact-K floor/cap allocation, small helpers, output paths
    core.py      fit_varwidth(): the shared ALS loop + Allocate for fixed / m1 / m2 / m3 (+ m4 via lasso.py), merge step
    lasso.py     method 4 Y-step (L1 ALS with lambda bisected to K nonzeros)
    prep.py      per-concept data: V*, labels, (whitened) A, cached under results/varwidth/prep/
    select.py    rho filter + 2-stage Gemini judge, identical to the notebooks' 3.2.b-3.2.e
    run.py       one (concept, config, seed) run -> results/varwidth/<concept>/<tag>/seed<N>/
    evaluate.py  aggregate metrics + report

Run output (written by run.py, read by evaluate.py), per run directory:
    fit.pt        {"Z": (d,k) f32, "Y": (k,n) f32, "widths": (k,) long, "churn": [float]*iters,
                   "rel_err_hist": [float]*iters, "rel_err": float, "n_iter": int, "lockin_iter": int,
                   "elapsed": float, "stop_reason": str, "K": int, "m_min": int, "m_max": int,
                   "config": dict (VWConfig.to_dict()), "seed": int, "extra": dict}
    summary.json  {"concept", "tag", "seed", "config", "n_concept_features", "F_C", "n_distinct_tokens",
                   "tokens_per_feature": {feat: width}, "C", "N", "B", "rel_err", "vstar", "fit_elapsed",
                   "total_elapsed", "n_iter", "lockin_iter", "widths": [k ints],
                   "candidates": [{"feature", "ratio_abs", "description", "is_member", "confidence", "selected"}],
                   "widest": [{"feature", "width", "description"}] (10 widest features, stage-1 description only),
                   "judge_reps": [[F_C of rerun 1], ...] (only with --judge-reps), "merge": {...} (only for method 5)}
    tables.md     the 3.2.f wide table (one column per F_C feature: token [C/N/B] |y|)
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "varwidth"
METHODS = ("fixed", "m1", "m2", "m3", "m4", "rn", "rnn")


def slug(concept: str) -> str:
    return re.sub(r"\W+", "_", concept).strip("_")


@dataclass
class VWConfig:
    method: str = "m2"              # fixed | m1 (contribution) | m2 (noise-normalized) | m3 (entropy) | m4 (L1 ALS)
                                    # | rn (pooling on |y_ij| / ||y_i||) | rnn (same, minus m3_c * sigma_i first)
    rank: int = 100                 # k
    sparsity: float = 0.01          # total budget K = ceil(sparsity * n) * k
    floor: float = 0.002            # m_min = ceil(floor * n)
    cap: float = 0.03               # m_max = ceil(cap * n); was 0.05 (spec default) until Oct 17 2026
    ridge: float = 1e-4             # lambda_r
    max_iter: int = 20_000          # T
    tol: float = 1e-4               # rel_err improvement threshold
    patience: int = 200             # iterations without tol improvement (and churn < churn_tol) before stopping
    churn_tol: float = 0.005        # support churn bar (fraction of K entries changed since the last iteration)
    init: str = "randn"             # randn (paper's N(0,1)) | svd (truncated-SVD init)
    refit: bool = True              # step 4 support refit; False: Y = Y_dense * S (paper)
    floorcap: bool = True           # floor/cap on per-row widths (methods 1, 2, 4 pooling; method 3 clipping)
    continuation: bool = False      # method 1 only: K_t from 5% to 1% linearly over the first cont_iters
    cont_start: float = 0.05
    cont_iters: int = 2000
    m3_every: int = 20              # method 3: R, width update period
    m3_window: float = 0.02         # method 3: W, top fraction of |y_dense| used for the entropy
    m3_c: float = 2.0               # method 3: noise multiplier c
    m4_every: int = 10              # method 4: lasso Y-step period (refit on fixed support in between)

    def __post_init__(self):
        assert self.method in METHODS, self.method
        assert self.init in ("randn", "svd"), self.init

    def to_dict(self) -> dict:
        return asdict(self)

    def tag(self) -> str:
        """Directory name: method plus every switch that differs from the default."""
        d, base = self.to_dict(), asdict(VWConfig(method=self.method))
        unused = {"floor", "cap", "floorcap"} if self.method == "fixed" else set()   # fixed ignores the bounds
        parts = [self.method]
        for k, v in d.items():
            if k != "method" and k not in unused and v != base[k]:
                parts.append(f"{k}-{v}")
        return "_".join(str(p) for p in parts)


def budgets(n: int, cfg: VWConfig, sparsity: float | None = None) -> tuple[int, int, int]:
    """(K, m_min, m_max) for n tokens. K = ceil(s*n)*k; without floorcap: m_min = 0, m_max = n."""
    s = cfg.sparsity if sparsity is None else sparsity
    K = math.ceil(s * n) * cfg.rank
    if not cfg.floorcap:
        return K, 0, n
    return K, math.ceil(cfg.floor * n), math.ceil(cfg.cap * n)


def pool_floor_cap(score: torch.Tensor, K: int, m_min: int, m_max: int) -> torch.Tensor:
    """Exact-K allocation with floor and cap (spec "Allocation with floor and cap"). score: (k, n), >= 0.

    1. each row keeps its top m_min entries; 2. the rest are sorted descending; 3. they are added in order,
    skipping rows that reached m_max, until K ones. Ties are broken by flat index (row-major). Returns bool (k, n).
    m_min = 0 and m_max = n gives the plain global top-K.

    Speed (identical result to a full sort, checked against brute force): with a cap (m_max < n) an entry can only be
    admitted if its stable rank within its row is < m_max, so the greedy pass runs on each row's stable top-m_max set
    minus the floor; every one of those is admissible, and the answer is the first K - k*m_min of them in the stable
    global order. Without a cap it runs on the candidate set {score >= t}, t = the C-th largest remaining score, which
    is exactly a prefix of the stable descending order (all entries > t, then all ties at t in index order); C grows
    if that prefix is too short.
    """
    k, n = score.shape
    assert k * m_min <= K <= k * m_max, (k, m_min, K, m_max)
    s = score.detach().float().contiguous()
    if m_min > 0:
        mask = _row_topk_mask(s, torch.full((k,), m_min, dtype=torch.long, device=s.device))
    else:
        mask = torch.zeros(k, n, dtype=torch.bool, device=s.device)
    rest = K - k * m_min
    if rest == 0:
        return mask
    flat = s.reshape(-1)
    if m_max < n:
        cand = _row_topk_mask(s, torch.full((k,), m_max, dtype=torch.long, device=s.device)) & ~mask
        idx = torch.nonzero(cand.reshape(-1)).squeeze(1)                      # ascending flat index
        chosen = idx[torch.sort(flat[idx], descending=True, stable=True).indices[:rest]]
        mask.view(-1)[chosen] = True
        return mask
    flat = flat.masked_fill(mask.reshape(-1), float("-inf"))
    total = k * n - k * m_min                                                 # admissible (non-floor) entries
    C = min(total, 2 * rest + 64)
    while True:
        if C >= total:
            order = torch.sort(flat, descending=True, stable=True).indices[:total]   # floor entries (-inf) at the end
        else:
            t = torch.topk(flat, C, sorted=False).values.min()
            if not torch.isfinite(t):                                         # prefix reaches the floor's -inf: full sort
                C = total
                continue
            cand = torch.nonzero(flat >= t).squeeze(1)                        # ascending flat index
            order = cand[torch.sort(flat[cand], descending=True, stable=True).indices]
        chosen = _greedy_cap(order, n, k, m_min, m_max, rest)
        if chosen is not None:
            mask.view(-1)[chosen] = True
            return mask
        assert C < total, "pool_floor_cap: not enough admissible entries"
        C = min(total, 4 * C)


def _greedy_cap(order: torch.Tensor, n: int, k: int, m_min: int, m_max: int, rest: int) -> torch.Tensor | None:
    """First `rest` entries of `order` (flat indices, a prefix of the sorted order) whose row is still below m_max;
    None if the prefix is too short."""
    rows = order // n
    # rank of each entry within its row, in sorted order (stable sort by row keeps the score order inside a row)
    by_row = torch.sort(rows, stable=True)
    starts = torch.zeros(k + 1, dtype=torch.long, device=order.device)
    starts[1:] = torch.cumsum(torch.bincount(rows, minlength=k), 0)
    rank_sorted = torch.arange(len(order), device=order.device) - starts[by_row.values]
    rank = torch.empty_like(rank_sorted)
    rank[by_row.indices] = rank_sorted
    chosen = order[(m_min + rank) < m_max][:rest]
    return chosen if len(chosen) == rest else None


def _row_topk_mask(s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Bool (k, n): row i's top m[i] entries by s, ties at the boundary broken by column index (as topk_rows_stable).

    Fast path: threshold each row at its m[i]-th largest value (torch.topk); exact whenever no row has a tie at its
    boundary, which is checked (counts == m). Rows that fail fall back to the stable full-row sort.
    """
    k, n = s.shape
    m = torch.as_tensor(m, device=s.device).long().clamp(0, n)
    mx = int(m.max()) if k else 0
    if mx == 0:
        return torch.zeros(k, n, dtype=torch.bool, device=s.device)
    vals = torch.topk(s, mx, dim=1, sorted=True).values                       # (k, mx), descending
    t = vals.gather(1, (m - 1).clamp_min(0)[:, None])
    mask = ((s >= t) & (m > 0)[:, None]).contiguous()
    bad = mask.sum(1) != m
    if bool(bad.any()):
        rows = torch.nonzero(bad).squeeze(1)
        order = torch.sort(s[rows], dim=1, descending=True, stable=True).indices
        sub = torch.zeros(len(rows), n, dtype=torch.bool, device=s.device)
        sub.scatter_(1, order, torch.arange(n, device=s.device)[None, :].expand_as(order) < m[rows][:, None])
        mask[rows] = sub
    return mask


def topk_rows_stable(score: torch.Tensor, m) -> torch.Tensor:
    """Indices of the top-m entries of each row (ties by column index). m: int, or a (k,) tensor of per-row widths
    (then returns a (k, max m) index tensor padded with -1)."""
    order = torch.sort(score, dim=1, descending=True, stable=True).indices
    if isinstance(m, int):
        return order[:, :m]
    m = torch.as_tensor(m, device=score.device)
    mx = int(m.max())
    idx = order[:, :mx].clone()
    idx[torch.arange(mx, device=score.device)[None, :] >= m[:, None]] = -1
    return idx


def mask_from_row_widths(score: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Bool (k, n) mask with row i's top m[i] entries by score (ties by column index)."""
    return _row_topk_mask(score.detach().float().contiguous(), torch.as_tensor(m))


def rel_err(A: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor) -> float:
    return float(torch.linalg.norm(A - Z @ Y) / torch.linalg.norm(A))


def churn(prev: torch.Tensor | None, mask: torch.Tensor, K: int) -> float:
    """Fraction of the K support entries that changed since the previous mask."""
    if prev is None:
        return 1.0
    return float((mask & ~prev).sum()) / K


def run_dir(concept: str, tag: str, seed: int) -> Path:
    return RESULTS / slug(concept) / tag / f"seed{seed}"
