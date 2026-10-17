"""Tests for experiments/variable_sparsity/varsparse.py.

Run with pytest (if installed) or directly:
    ~/.conda/envs/Ember/bin/python experiments/variable_sparsity/test_varsparse.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_HERE, _ROOT, _ROOT / "external" / "snmf"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from ember.utils import SparseMatrixFactorization, set_seed  # noqa: E402
from factorization.seminmf import wta_cols  # noqa: E402

import varsparse as vs  # noqa: E402
from varsparse import (  # noqa: E402
    DISTS,
    SparsityConfig,
    VariableSparsityMF,
    _fit_to_total,
    draw_budgets,
    fit_best_of,
    match_budgets,
    participation_ratio,
    row_bounds,
    topk_per_row,
    topk_pooled,
    total_budget,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}") from e
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


def _low_rank(d=16, n=200, r=6, seed=0, noise=0.05):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, r, generator=g) @ torch.randn(r, n, generator=g)
    return A + noise * torch.randn(d, n, generator=g)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _cfg(**kw):
    base = dict(mode="per_row", dist="uniform", sparsity=0.05, uniform_range=(0.03, 0.07),
                loguniform_range=(0.02, 0.10), gauss_std=0.01, floor=0.01, cap=0.15,
                anneal=False, noise_sigma0=0.0)
    base.update(kw)
    return SparsityConfig(**base)


def _brute_pooled(scores, total, floor, cap):
    """Reference: top `floor` per column forced, then a greedy global fill over ranks floor..cap-1."""
    n, k = scores.shape
    mask = torch.zeros_like(scores)
    cands = []
    for j in range(k):
        order = torch.argsort(scores[:, j], descending=True).tolist()
        for r, i in enumerate(order[:cap]):
            if r < floor:
                mask[i, j] = 1.0
            else:
                cands.append((float(scores[i, j]), i, j))
    cands.sort(key=lambda t: -t[0])
    for _, i, j in cands[: total - k * floor]:
        mask[i, j] = 1.0
    return mask


def _fit(A, cfg, seed=0, rank=4, max_iter=100, patience=30, reg=1e-6):
    set_seed(seed)
    smf = VariableSparsityMF(rank, cfg=cfg, seed=seed)
    smf.fit(A, max_iter=max_iter, patience=patience, reg=reg)
    return smf


# ---------------------------------------------------------------------------
# SparsityConfig
# ---------------------------------------------------------------------------

def test_config_rejects_invalid_values():
    _raises(AssertionError, SparsityConfig, mode="pooled")
    _raises(AssertionError, SparsityConfig, dist="cauchy")
    _raises(AssertionError, SparsityConfig, assign="sorted")
    _raises(AssertionError, SparsityConfig, floor=0.02, sparsity=0.01)          # floor > s
    _raises(AssertionError, SparsityConfig, cap=0.005, sparsity=0.01)           # cap < s
    _raises(AssertionError, SparsityConfig, anneal=True, anneal_start=0.005)    # start below target
    for m in vs.MODES:
        SparsityConfig(mode=m)
    for d in DISTS:
        SparsityConfig(dist=d)
    for a in vs.ASSIGNS:
        SparsityConfig(assign=a)


def test_config_slug_stable_and_sensitive():
    base = SparsityConfig()
    assert base.slug() == SparsityConfig().slug()
    assert base.slug() == SparsityConfig(uniform_range=list(base.uniform_range)).slug()   # list vs tuple
    variants = dict(mode="pooled_rownorm", dist="loguniform", assign="matched", sparsity=0.02,
                    uniform_range=(0.004, 0.016), loguniform_range=(0.002, 0.04), gauss_std=0.004,
                    floor=0.001, cap=0.06, anneal=False, anneal_start=0.2, anneal_iters=1000,
                    noise_sigma0=0.25, noise_iters=500, match_every=25, match_iters=700)
    seen = {base.slug()}
    for key, val in variants.items():
        s = SparsityConfig(**{key: val}).slug()
        assert s != base.slug(), key
        assert s not in seen, key
        seen.add(s)
    assert base.slug().startswith("per_row-uniform-")


def test_config_to_meta_json_roundtrip():
    for cfg in (SparsityConfig(), SparsityConfig(mode="pooled_contrib", noise_iters=7, anneal=False),
                SparsityConfig(dist="gaussian", assign="matched", match_iters=10)):
        meta = cfg.to_meta()
        s = json.dumps(meta, sort_keys=True)
        back = SparsityConfig(**json.loads(s))
        assert back == cfg
        assert back.slug() == cfg.slug()


def test_config_t_free():
    assert SparsityConfig(anneal=False, noise_sigma0=0).t_free == 0
    assert SparsityConfig(anneal=True, anneal_iters=123, noise_sigma0=0).t_free == 123
    assert SparsityConfig(anneal=False, noise_sigma0=0.3).t_free == 1000
    assert SparsityConfig(anneal=False, noise_sigma0=0.3, noise_iters=40).t_free == 40
    assert SparsityConfig(anneal=True, anneal_iters=50, noise_sigma0=0.3, noise_iters=80).t_free == 80
    assert SparsityConfig(anneal=False, noise_sigma0=0, assign="matched").t_free == 1000
    assert SparsityConfig(anneal=False, noise_sigma0=0, assign="matched", match_iters=9).t_free == 9
    assert SparsityConfig(mode="pooled_rownorm", anneal=False, noise_sigma0=0, assign="matched").t_free == 0


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

def test_draw_budgets_sum_bounds_determinism():
    n, k = 1000, 50
    for dist in DISTS:
        cfg = SparsityConfig(dist=dist, anneal=False)
        lo, hi = row_bounds(cfg, n)
        total = total_budget(cfg, n, k)
        for seed in range(5):
            m = draw_budgets(cfg, n, k, _gen(seed))
            assert m.shape == (k,) and m.dtype == torch.long, (dist, m.dtype)
            assert int(m.sum()) == total, (dist, int(m.sum()), total)
            assert int(m.min()) >= lo and int(m.max()) <= hi, (dist, m.min(), m.max(), lo, hi)
            assert torch.equal(m, draw_budgets(cfg, n, k, _gen(seed))), dist
        if dist != "constant":
            assert not torch.equal(draw_budgets(cfg, n, k, _gen(0)), draw_budgets(cfg, n, k, _gen(1))), dist


def test_draw_budgets_constant_equal():
    for n, k, s in ((1000, 50, 0.01), (333, 7, 0.013), (200, 4, 0.05)):
        cfg = SparsityConfig(dist="constant", sparsity=s, anneal=False, cap=max(0.05, s))
        m = draw_budgets(cfg, n, k, _gen(0))
        assert (m == math.ceil(s * n)).all()
        assert int(m.sum()) == total_budget(cfg, n, k)


def test_draw_budgets_loguniform_wider_than_uniform():
    n, k = 2000, 100
    cu, cl = SparsityConfig(dist="uniform", anneal=False), SparsityConfig(dist="loguniform", anneal=False)
    su = sum(draw_budgets(cu, n, k, _gen(s)).double().std().item() for s in range(10)) / 10
    sl = sum(draw_budgets(cl, n, k, _gen(s)).double().std().item() for s in range(10)) / 10
    assert sl > su, (sl, su)
    ru = sum((lambda m: m.max() / m.min())(draw_budgets(cu, n, k, _gen(s))).item() for s in range(10))
    rl = sum((lambda m: m.max() / m.min())(draw_budgets(cl, n, k, _gen(s))).item() for s in range(10))
    assert rl > ru, (rl, ru)


def test_fit_to_total_infeasible_and_feasible():
    x = torch.rand(5) + 0.1
    _raises(AssertionError, _fit_to_total, x, 25, 2, 4)    # > k*hi
    _raises(AssertionError, _fit_to_total, x, 9, 2, 4)     # < k*lo
    for total in (10, 13, 17, 20):
        m = _fit_to_total(x, total, 2, 4)
        assert int(m.sum()) == total and int(m.min()) >= 2 and int(m.max()) <= 4
    # extreme spreads
    k = 50
    for x in (torch.tensor([1e6] + [1.0] * (k - 1)), torch.logspace(0, 8, k), torch.full((k,), 1e-9)):
        for total in (k * 2, k * 2 + 1, k * 25, k * 50 - 1, k * 50):
            m = _fit_to_total(x, total, 2, 50)
            assert int(m.sum()) == total


# ---------------------------------------------------------------------------
# Participation ratio / matching (pending implementation)
# ---------------------------------------------------------------------------

def test_participation_ratio_contract():
    n, k = 40, 6
    g = _gen(0)
    G = torch.zeros(n, k)
    widths = [1, 3, 7, 12, 40, 0]
    for j, w in enumerate(widths):
        if w:
            idx = torch.randperm(n, generator=g)[:w]
            signs = torch.where(torch.rand(w, generator=g) < 0.5, -1.0, 1.0)
            G[idx, j] = 2.5 * signs
    pr = participation_ratio(G)
    assert pr.shape == (k,)
    assert torch.isfinite(pr).all()
    for j, w in enumerate(widths):
        assert abs(float(pr[j]) - w) < 1e-4, (j, float(pr[j]), w)

    R = torch.randn(30, 5, generator=g)
    expect = R.abs().sum(0) ** 2 / (R ** 2).sum(0)
    assert torch.allclose(participation_ratio(R).double(), expect.double(), rtol=1e-5)
    # scale-invariant per column, in [1, n] for non-zero columns
    assert torch.allclose(participation_ratio(R * torch.tensor([1e-3, 1, 7, 100, 3.0])).double(),
                          expect.double(), rtol=1e-4)
    assert (participation_ratio(R) >= 1 - 1e-5).all() and (participation_ratio(R) <= 30 + 1e-4).all()


def test_match_budgets_contract():
    g = _gen(1)
    n, k = 60, 8
    m = torch.tensor([3, 9, 1, 5, 5, 12, 2, 7], dtype=torch.long)
    for trial in range(4):
        if trial == 0:  # columns with known participation ratios (distinct widths)
            G = torch.zeros(n, k)
            widths = torch.randperm(k, generator=g) * 5 + 2
            for j, w in enumerate(widths.tolist()):
                G[torch.randperm(n, generator=g)[:w], j] = torch.randn(1, generator=g).abs().item() + 0.1
        else:
            G = torch.randn(n, k, generator=g) * torch.rand(n, k, generator=g).pow(1 + 3 * torch.rand(1, k, generator=g))
        out = match_budgets(m.clone(), G)
        assert out.dtype == m.dtype and out.shape == m.shape
        assert torch.equal(torch.sort(out).values, torch.sort(m).values)
        assert int(out.sum()) == int(m.sum())
        pr = participation_ratio(G)
        for i in range(k):
            for j in range(k):
                if pr[i] > pr[j]:
                    assert out[i] >= out[j], (trial, i, j, pr[i], pr[j], out[i], out[j])
        if trial == 0:  # strictly distinct ratios: the widest column gets the max budget
            assert out[int(pr.argmax())] == m.max() and out[int(pr.argmin())] == m.min()


# ---------------------------------------------------------------------------
# Support selection
# ---------------------------------------------------------------------------

def test_topk_per_row_keeps_top_m():
    g = _gen(2)
    n, k = 50, 6
    S = torch.rand(n, k, generator=g)
    m = torch.tensor([1, 4, 10, 2, 7, 50])
    mask = topk_per_row(S, m)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(mask.sum(0).long(), m)
    for j in range(k):
        kept, dropped = S[mask[:, j] == 1, j], S[mask[:, j] == 0, j]
        if dropped.numel():
            assert kept.min() > dropped.max()


def test_topk_per_row_equals_wta_cols_for_constant_m():
    g = _gen(3)
    for n, k, p in ((100, 5, 0.05), (257, 9, 0.013), (40, 3, 0.5)):
        G = torch.randn(n, k, generator=g)
        ref = G.clone()
        wta_cols(ref, pct_keep=p)
        m = torch.full((k,), max(1, math.ceil(p * n)), dtype=torch.long)
        mask = topk_per_row(G.abs(), m)
        assert torch.equal(mask.bool(), ref != 0)
        assert torch.equal(G * mask, ref)


def test_topk_pooled_contract_and_bruteforce():
    g = _gen(4)
    for n, k, floor, cap in ((12, 4, 2, 6), (20, 3, 1, 20), (9, 5, 3, 3), (15, 6, 0, 5)):
        for _ in range(3):
            S = torch.rand(n, k, generator=g)
            for total in sorted({k * floor, k * floor + 1, (k * floor + k * cap) // 2, k * cap - 1, k * cap}):
                if not (k * floor <= total <= k * cap):
                    continue
                mask = topk_pooled(S, total, floor, cap)
                assert mask.shape == S.shape
                assert set(mask.unique().tolist()) <= {0.0, 1.0}
                assert int(mask.sum()) == total, (n, k, floor, cap, total, int(mask.sum()))
                cs = mask.sum(0)
                assert (cs >= floor).all() and (cs <= cap).all(), (cs, floor, cap)
                if floor:
                    top = torch.topk(S, floor, dim=0).indices
                    assert (mask.gather(0, top) == 1).all()
                assert torch.equal(mask, _brute_pooled(S, total, floor, cap)), (n, k, floor, cap, total)


# ---------------------------------------------------------------------------
# The factorization
# ---------------------------------------------------------------------------

def test_constant_matches_sparse_mf_bitwise():
    for seed, (d, n, rank, s) in ((0, (16, 200, 4, 0.05)), (7, (12, 150, 6, 0.02)), (3, (20, 97, 3, 0.1))):
        A = _low_rank(d, n, seed=seed + 100)
        cfg = SparsityConfig(mode="per_row", dist="constant", sparsity=s, anneal=False, noise_sigma0=0,
                             floor=min(0.002, s), cap=max(0.05, s))
        set_seed(seed)
        ref = SparseMatrixFactorization(rank, g_sparsity=s)
        ref.fit(A, max_iter=60, patience=20, verbose=False)
        set_seed(seed)
        new = VariableSparsityMF(rank, cfg=cfg, seed=seed)
        new.fit(A, max_iter=60, patience=20)
        assert torch.equal(ref.F_.data, new.F_.data), seed
        assert torch.equal(ref.G_.data, new.G_.data), seed
        assert new.t_free == 0
        assert (new.nnz_per_row == math.ceil(s * n)).all()


def test_anneal_returns_target_budgets():
    A = _low_rank(16, 200, seed=11)
    cfg = _cfg(anneal=True, anneal_start=0.2, anneal_iters=40)
    smf = _fit(A, cfg, seed=1, max_iter=150, patience=30)
    assert smf.t_free == 40
    assert len(smf.loss_history) > smf.t_free
    assert torch.equal(smf.nnz_per_row, smf.m), (smf.nnz_per_row, smf.m)
    assert torch.equal(smf.m, smf.m_drawn)
    assert int(smf.nnz_per_row.sum()) == total_budget(cfg, 200, 4)
    assert torch.equal((smf.G_.data != 0).sum(0), smf.nnz_per_row)
    # the early phase was actually denser (lower loss) than the final sparse fit
    assert min(smf.loss_history[: smf.t_free]) < min(smf.loss_history[smf.t_free:])


def test_best_loss_ignores_early_phase():
    A = _low_rank(16, 200, seed=12)
    cfg = _cfg(anneal=True, anneal_start=0.3, anneal_iters=30)
    smf = _fit(A, cfg, seed=2, max_iter=cfg.t_free + 1, patience=1000)
    assert len(smf.loss_history) == cfg.t_free + 1
    assert torch.equal(smf.nnz_per_row, smf.m)
    smf = _fit(A, cfg, seed=2, max_iter=cfg.t_free + 3, patience=1000)
    assert torch.equal(smf.nnz_per_row, smf.m)


def test_noise_deterministic_given_seed():
    A = _low_rank(16, 200, seed=13)
    cfg = _cfg(noise_sigma0=0.5, noise_iters=25)
    a = _fit(A, cfg, seed=5, max_iter=60, patience=20)
    b = _fit(A, cfg, seed=5, max_iter=60, patience=20)
    assert a.t_free == 25
    assert a.loss_history == b.loss_history
    assert torch.equal(a.F_.data, b.F_.data) and torch.equal(a.G_.data, b.G_.data)
    assert torch.equal(a.nnz_per_row, a.m)
    # a different budget/noise seed (same global init) changes the fit
    set_seed(5)
    c = VariableSparsityMF(4, cfg=cfg, seed=6).fit(A, max_iter=60, patience=20)
    assert c.loss_history != a.loss_history


def test_no_noise_no_anneal_t_free_zero():
    A = _low_rank(16, 200, seed=14)
    for dist in DISTS:
        cfg = _cfg(dist=dist, anneal=False, noise_sigma0=0)
        assert cfg.t_free == 0
        smf = _fit(A, cfg, seed=0, max_iter=30, patience=10)
        assert smf.t_free == 0
        assert torch.equal(smf.nnz_per_row, smf.m), dist


def test_matched_fit_permutes_budgets():
    A = _low_rank(16, 200, seed=15)
    cfg = _cfg(dist="loguniform", assign="matched", match_every=5, match_iters=20)
    smf = _fit(A, cfg, seed=3, max_iter=60, patience=20)
    assert smf.t_free == 20
    assert torch.equal(torch.sort(smf.m).values, torch.sort(smf.m_drawn).values)
    assert torch.equal(smf.nnz_per_row, smf.m)


def test_pooled_fits_keep_total_budget():
    A = _low_rank(16, 200, seed=16)
    for mode in ("pooled_rownorm", "pooled_contrib"):
        for anneal in (False, True):
            cfg = _cfg(mode=mode, anneal=anneal, anneal_start=0.2, anneal_iters=20)
            smf = _fit(A, cfg, seed=4, max_iter=60, patience=20)
            lo, hi = row_bounds(cfg, 200)
            assert smf.m is None
            assert int(smf.nnz_per_row.sum()) == total_budget(cfg, 200, 4), (mode, anneal)
            assert int(smf.nnz_per_row.min()) >= lo and int(smf.nnz_per_row.max()) <= hi, (mode, anneal)


# ---------------------------------------------------------------------------
# Restarts
# ---------------------------------------------------------------------------

def test_fit_best_of_picks_min_rel_err():
    A = _low_rank(16, 200, seed=17)
    cfg = _cfg(dist="uniform")

    def metrics(Y):
        return {"y_rows": int(Y.shape[0]), "y_absmax": float(Y.abs().max())}

    best = fit_best_of(A, cfg, rank=4, n_restarts=3, seed=10, max_iter=40, patience=10, base_reg=1e-6,
                       concept_metrics_fn=metrics)
    df = best["restarts"]
    assert len(df) == 3
    assert list(df["seed"]) == [10, 11, 12]
    assert {"y_rows", "y_absmax", "rel_err", "nnz_total"} <= set(df.columns)
    assert (df["y_rows"] == 4).all()
    assert best["rel_err"] == df["rel_err"].min()
    assert best["best_restart"] == int(df["rel_err"].idxmin())
    Z, Y = best["Z"], best["Y"]
    assert Z.shape == (16, 4) and Y.shape == (4, 200)
    err = (torch.norm(A - Z @ Y) / torch.norm(A)).item()
    assert abs(err - best["rel_err"]) < 1e-6
    assert torch.equal((Y != 0).sum(1), best["nnz_per_row"])
    assert torch.equal(best["nnz_per_row"], best["budgets"])
    assert (df["nnz_total"] == total_budget(cfg, 200, 4)).all()
    for key in ("ridge_used", "loss_history", "t_free", "elapsed", "budgets_drawn"):
        assert key in best


# ---------------------------------------------------------------------------

def test_resume_interrupted_matches_uninterrupted():
    # Interrupt inside the noise phase and after it (best tracking running, 2nd restart not started yet);
    # resuming from the traceback must give the uninterrupted result, including the remaining restarts.
    A = _low_rank(d=16, n=300, seed=3)
    cfg = _cfg(mode="per_row", dist="uniform", anneal=False, noise_sigma0=0.5, noise_iters=20)
    orig = vs.VariableSparsityMF._sparsify
    for stop_at, n_restarts in [(10, 1), (35, 2)]:
        fired = []

        def boom(self, it, _orig=orig, _stop=stop_at):
            if it == _stop and not fired:
                fired.append(it)
                raise KeyboardInterrupt
            return _orig(self, it)

        vs.VariableSparsityMF._sparsify = boom
        tb = None
        try:
            fit_best_of(A, cfg, 4, n_restarts, 0, 120, 20, 1e-6, verbose=False)
        except KeyboardInterrupt:
            tb = sys.exc_info()[2]
        finally:
            vs.VariableSparsityMF._sparsify = orig
        assert tb is not None and fired == [stop_at]
        out = vs.resume_interrupted(tb)
        ref = fit_best_of(A, cfg, 4, n_restarts, 0, 120, 20, 1e-6, verbose=False)
        assert len(out["restarts"]) == n_restarts
        assert math.isclose(out["rel_err"], ref["rel_err"], rel_tol=1e-4), (out["rel_err"], ref["rel_err"])
        assert int(out["nnz_per_row"].sum()) == int(ref["nnz_per_row"].sum())


def test_resume_interrupted_without_fit_traceback_raises():
    try:
        raise ValueError("not a fit")
    except ValueError:
        tb = sys.exc_info()[2]
    _raises(RuntimeError, vs.resume_interrupted, tb)


if __name__ == "__main__":
    import time
    import traceback

    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    t0 = time.perf_counter()
    for name, fn in tests:
        t = time.perf_counter()
        try:
            fn()
            print(f"PASS  {name}  ({time.perf_counter() - t:.2f}s)")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}  {type(e).__name__}: {e}")
            if not isinstance(e, NotImplementedError):
                traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed in {time.perf_counter() - t0:.1f}s")
    sys.exit(1 if failed else 0)
