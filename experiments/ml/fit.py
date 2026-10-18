#!/usr/bin/env python
"""Factorization stage of the three approaches (experiments/ml/PLAN.md). Needs only the embedding matrix.

    python -m experiments.ml.fit --concept "Harry Potter" --approach A1|A2|A3 [--N 40] [--device cuda]

A1  per-language EMBER: for each language, EMBER's fixed-width (row-sparse, 1%) fit on that language's full set.
A2  one fixed-width fit on the multilingual sample (N concept + N neutral sentences per language), then Z fixed and a
    row-sparse Y (top ceil(s*n_full) per row of the dense least-squares solve) for the full multilingual set.
A3  A2 on the CS matrix (8-language background whitening + sharpening toward the concept sentences of all 8 languages)
    with GS-norm (rn, floor 0.2%, cap 3%) in the fit and in the extrapolation (pool_floor_cap on |y_ij| / ||y_i||).

Writes run_dir/(lang/)fit.pt: Z (d, k) in the factorized space, Y (k, n) on the token set, vstar_ids, codes, langs of
the sentences, `unmap` (d, d) for A3 (row form: c = c_x @ unmap; identity otherwise), and fit stats.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

from experiments.ml.common import (DATA, LANGS, RANK, RIDGE, SEED, SPARSITY, embeddings, run_dir, save_json, slug,
                                   special_ids, token_set, tokenizer)
from experiments.varwidth.common import VWConfig, mask_from_row_widths, pool_floor_cap
from experiments.varwidth.core import fit_varwidth

ETA, GAMMA, EPS_REL = 0.05, 1.0, 1e-4


# ---------------------------------------------------------------- data
def _flatten(sets: dict, langs=LANGS):
    sents, labels, ls = [], [], []
    for L in langs:
        for side, lab in (("concept", "concept"), ("neutral", "neutral")):
            for r in sets[L][side]:
                sents.append(r["text"] if isinstance(r, dict) else r)
                labels.append(lab)
                ls.append(L)
    return sents, labels, ls


def load_full(concept: str) -> dict:
    return json.loads((DATA / "full" / f"{slug(concept)}.json").read_text())


def load_sample(concept: str, N: int) -> dict:
    return json.loads((DATA / "sample" / f"{slug(concept)}_N{N}.json").read_text())


# ---------------------------------------------------------------- CS (8-language background)
_BG = {}


def background_stats() -> dict:
    """Mean and eigendecomposition of the embeddings of the token types of results/ml/data/background_ml.json
    (equal numbers of native sentences per language), cached at results/ml/data/bg_ml_llama.pt."""
    path = DATA / "bg_ml_llama.pt"
    if "s" in _BG:
        return _BG["s"]
    if path.exists():
        _BG["s"] = torch.load(path)
        return _BG["s"]
    import sys
    from experiments.ml.common import ROOT
    sys.path.insert(0, str(ROOT / "experiments" / "whitening"))
    from sharpen import fit_background
    bg = json.loads((DATA / "background_ml.json").read_text())
    tok, special = tokenizer(), special_ids()
    types = set()
    for L in LANGS:
        for s in bg[L]:
            types.update(tok(s if isinstance(s, str) else s["text"], add_special_tokens=False)["input_ids"])
    types = sorted(types - special)
    stats = fit_background(embeddings()[types])
    stats["n_types"] = len(types)
    torch.save(stats, path)
    _BG["s"] = stats
    return stats


def cs_transform(full: dict):
    import sys
    from experiments.ml.common import ROOT
    sys.path.insert(0, str(ROOT / "experiments" / "whitening"))
    from sharpen import BackgroundWhitener, Sharpener, Transform
    stats = {k: (v.double() if torch.is_tensor(v) else v) for k, v in background_stats().items()}
    bw = BackgroundWhitener(stats, EPS_REL)
    tok, special = tokenizer(), special_ids()
    target = set()
    for L in LANGS:
        for r in full[L]["concept"]:
            target.update(tok(r["text"] if isinstance(r, dict) else r, add_special_tokens=False)["input_ids"])
    target = sorted(target - special)
    sh = Sharpener(bw.whiten(embeddings()[target].double()), ETA, GAMMA)
    return Transform(bw, sh), len(target)


# ---------------------------------------------------------------- extrapolation
def extrapolate(Z: torch.Tensor, A: torch.Tensor, method: str, cfg: VWConfig) -> torch.Tensor:
    """Z fixed: dense ridge solve for Y on A (d, n), then the method's support rule with n's budget."""
    Zd = Z.double()
    G = Zd.T @ Zd + cfg.ridge * torch.eye(Z.shape[1], dtype=torch.float64, device=Z.device)
    Yd = torch.linalg.solve(G, Zd.T @ A.double()).float()
    k, n = Yd.shape
    if method == "fixed":
        mask = mask_from_row_widths(Yd.abs(), torch.full((k,), math.ceil(cfg.sparsity * n), dtype=torch.long))
    else:
        K = math.ceil(cfg.sparsity * n) * k
        lo, hi = math.ceil(cfg.floor * n), math.ceil(cfg.cap * n)
        mask = pool_floor_cap(Yd.abs() / Yd.norm(dim=1, keepdim=True).clamp_min(1e-12), K, lo, hi)
    return Yd * mask


# ---------------------------------------------------------------- one fit
def _fit(A: torch.Tensor, method: str, device: str) -> tuple[dict, VWConfig]:
    cfg = VWConfig(method=method, rank=RANK, sparsity=SPARSITY, ridge=RIDGE, refit=False)
    return fit_varwidth(A, cfg, seed=SEED, device=device), cfg


def _save(path, Z, Y, ts, unmap, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"Z": Z.float().cpu(), "Y": Y.to(torch.float16).cpu(), "vstar_ids": ts["vstar_ids"],
                "codes": ts["codes"], "unmap": None if unmap is None else unmap.float().cpu(), **extra}, path)
    save_json(path.with_name("fit_summary.json"), {k: v for k, v in extra.items() if not torch.is_tensor(v)})


def run(concept: str, approach: str, N: int, device: str, force: bool = False):
    E = embeddings()
    full = load_full(concept)
    t0 = time.time()
    if approach == "A1":
        for L in LANGS:
            out = run_dir(concept, "A1", L) / "fit.pt"
            if out.exists() and not force:
                continue
            s, lab, ls = _flatten(full, [L])
            ts = token_set(s, lab, ls)
            A = E[ts["vstar_ids"]].T.contiguous()
            fit, _ = _fit(A, "fixed", device)
            _save(out, fit["Z"], fit["Y"], ts, None,
                  {"concept": concept, "approach": "A1", "lang": L, "n": len(ts["vstar_ids"]),
                   "rel_err": float(fit["rel_err"]), "n_iter": int(fit["n_iter"]), "secs": time.time() - t0})
            print(f"[A1 {concept} {L}] |V*|={len(ts['vstar_ids'])} rel_err={fit['rel_err']:.4f}", flush=True)
        return
    out = run_dir(concept, approach) / "fit.pt"
    if out.exists() and not force:
        print(f"exists: {out}")
        return
    sample = load_sample(concept, N)
    s_s, lab_s, ls_s = _flatten(sample)
    s_f, lab_f, ls_f = _flatten(full)
    ts_s, ts_f = token_set(s_s, lab_s, ls_s), token_set(s_f, lab_f, ls_f)
    method = "fixed" if approach == "A2" else "rn"
    unmap, extra = None, {}
    if approach == "A3":
        T, n_target = cs_transform(full)
        fwd = lambda ids: T.forward(E[ids].double()).float()
        unmap = (T.sh.Sm @ T.bw.W_inv) if T.sh is not None else T.bw.W_inv
        extra = {"n_target": n_target, "eta": ETA, "gamma": GAMMA, "eps_rel": EPS_REL}
    else:
        fwd = lambda ids: E[ids]
    A_s = fwd(ts_s["vstar_ids"]).T.contiguous()
    fit, cfg = _fit(A_s, method, device)
    A_f = fwd(ts_f["vstar_ids"]).T.contiguous().to(device)
    Y_f = extrapolate(fit["Z"].to(device), A_f, method, cfg)
    err = float(torch.linalg.norm(A_f - fit["Z"].to(device) @ Y_f) / torch.linalg.norm(A_f))
    _save(out, fit["Z"], Y_f, ts_f, unmap,
          {"concept": concept, "approach": approach, "N": N, "n_sample": len(ts_s["vstar_ids"]),
           "n_full": len(ts_f["vstar_ids"]), "rel_err_sample": float(fit["rel_err"]), "rel_err_full": err,
           "n_iter": int(fit["n_iter"]), "widths_full": (Y_f != 0).sum(1).tolist(), "secs": time.time() - t0,
           **extra})
    print(f"[{approach} {concept}] |V*_sample|={len(ts_s['vstar_ids'])} |V*_full|={len(ts_f['vstar_ids'])} "
          f"rel_err sample={fit['rel_err']:.4f} full={err:.4f} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, nargs="+")
    ap.add_argument("--approach", required=True, nargs="+", choices=["A1", "A2", "A3"])
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for c in a.concept:
        for ap_ in a.approach:
            run(c, ap_, a.N, a.device, a.force)


if __name__ == "__main__":
    main()
