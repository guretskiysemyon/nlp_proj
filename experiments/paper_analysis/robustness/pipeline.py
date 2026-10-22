"""WP4 data + fit pipeline: any concept-sentence list -> V*, codes, A (raw or CS) -> fit (experiments/varwidth/core.py).

Reproduces experiments/varwidth/prep.py (build_dataset / build_vstar / whitening_transform) for an explicit list of
concept sentences, so contaminated and sub-sampled S_C sets need no file under data/. The neutral side is the project's
300 neutral sentences (ConceptDataset, neutral_sample_seed 42). CS uses the project's wiki background stats and
defaults (eta 0.05, gamma 1, eps_rel 1e-4) and a sharpening target given as token ids.

Nothing large is cached: per variant only V*, codes and tokens (meta/<variant>.pt); A is rebuilt in memory (seconds).
Fits are saved compactly (Z float32, Y as sparse indices/values) to fits/<variant>/<cond>/seed<N>.pt.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.paper_analysis.robustness.concepts import DATA, FITS
from experiments.varwidth.common import ROOT, VWConfig, slug

META = DATA / "meta"
VARIANTS = DATA / "variants"
PROJECT_TARGETS = ROOT / "experiments" / "whitening" / "data" / "gemma" / "targets"

_E = None
_BW = None


def get_E() -> torch.Tensor:
    global _E
    if _E is None:
        from experiments.varwidth.prep import load_input_embeddings
        _E = load_input_embeddings()
    return _E


def get_bw():
    """BackgroundWhitener of prep.whitening_transform (wiki background, eps_rel from defaults.json)."""
    global _BW
    if _BW is None:
        import sys
        sys.path.insert(0, str(ROOT / "experiments" / "whitening"))
        from sharpen import BackgroundWhitener
        from experiments.varwidth.prep import BACKGROUND, WH_DIR
        defaults = json.loads((WH_DIR / "defaults.json").read_text())
        stats = torch.load(WH_DIR / f"bg_{BACKGROUND}_stats.pt", map_location="cpu")
        stats = {k: (v.double() if torch.is_tensor(v) else v) for k, v in stats.items()}
        _BW = (BackgroundWhitener(stats, defaults["eps_rel"]), defaults)
    return _BW


def write_variant(variant: str, concept: str, sentences: list[str], info: dict | None = None) -> Path:
    """The concept-sentence file ConceptDataset reads for this variant (schema of data/concept_sentences.json)."""
    VARIANTS.mkdir(parents=True, exist_ok=True)
    p = VARIANTS / f"{variant}.json"
    p.write_text(json.dumps([{"concept": concept, "source": "wp4", "sentences": sentences, **(info or {})}],
                            ensure_ascii=False, indent=1))
    return p


def variant_meta(variant: str, concept: str, force: bool = False) -> dict:
    """V*, codes (1 C, 0 N, 2 B), tokens of the variant: prep.build_dataset + prep.build_vstar on VARIANTS/<variant>.json."""
    path = META / f"{variant}.pt"
    if path.exists() and not force:
        return torch.load(path, weights_only=False)
    from ember.local_datasets import ConceptDataset
    from experiments.varwidth.prep import NEUTRAL_SAMPLE_SEED, build_vstar, get_tokenizer
    tok = get_tokenizer()
    ds = ConceptDataset(concept_name=concept, concept_path=VARIANTS / f"{variant}.json",
                        neutral_sample_seed=NEUTRAL_SAMPLE_SEED)
    vstar_ids, token_to_code, special = build_vstar(concept, ds, tok)
    codes = np.array([token_to_code.get(t, 0) for t in vstar_ids])
    meta = {"variant": variant, "concept": concept, "vstar_ids": vstar_ids, "codes": codes,
            "token_to_code": token_to_code, "tokens": tok.convert_ids_to_tokens(vstar_ids), "special": sorted(special),
            "sentences": [s for s, lbl in ds.data if lbl == concept],
            "n_neutral_sents": sum(lbl == "Neutral" for _, lbl in ds.data)}
    META.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(meta, tmp)
    tmp.replace(path)
    c = meta["codes"]
    print(f"[meta] {variant}: |V*|={len(vstar_ids)} C/N/B={(c == 1).sum()}/{(c == 0).sum()}/{(c == 2).sum()}",
          flush=True)
    return meta


def target_ids(concept: str, kind: str, meta: dict | None = None) -> list[int]:
    """Sharpening target: 'page' = the concept's own page (project target file, else ours); 'sc' = token ids of the
    variant's own concept sentences (built like a page target)."""
    if kind == "page":
        for d in (PROJECT_TARGETS, DATA / "targets"):
            p = d / f"{slug(concept)}.json"
            if p.exists():
                return json.loads(p.read_text())["page"]
        raise FileNotFoundError(f"no page target for {concept}")
    if kind == "sc":
        from experiments.paper_analysis.robustness.fetch import page_target_ids
        return page_target_ids(meta["sentences"])
    raise ValueError(kind)


def build_A(meta: dict, matrix: str, target: str | None = None) -> tuple[torch.Tensor, dict]:
    """A (d, n): raw E_V*^T, or the CS matrix X^T (prep.whitening_transform with the given target)."""
    E = get_E()
    E_v = E.index_select(0, torch.tensor(meta["vstar_ids"])).float()
    if matrix == "plain":
        return E_v.T.contiguous(), {}
    bw, defaults = get_bw()                                    # puts experiments/whitening on sys.path
    from sharpen import Sharpener, Transform
    from experiments.varwidth.prep import GAMMA
    tids = target_ids(meta["concept"], target, meta)
    tids = sorted({int(t) for t in tids if 0 <= int(t) < E.shape[0]} - set(meta["special"]))
    U_t = bw.whiten(E.index_select(0, torch.tensor(tids)).double())
    T = Transform(bw, Sharpener(U_t, defaults["eta"], GAMMA))
    X = T.forward(E_v).float()
    return X.T.contiguous(), {"n_target": len(tids), "T_inv": T.sh.Sm @ bw.W_inv}


def cfg_for(method: str) -> VWConfig:
    """EMBER / CS: fixed RS-1% (no refit); GS: rn (global K = 1%, floor 0.2%, cap 3%), no refit — project tags
    fixed_refit-False[_plain] and rn_refit-False."""
    return VWConfig(method=method, refit=False)


def fit_path(variant: str, cond: str, seed: int) -> Path:
    return FITS / variant / cond / f"seed{seed}.pt"


def save_fit(path: Path, fit: dict, extra: dict):
    Y = fit["Y"].detach().float().cpu()
    nz = torch.nonzero(Y != 0)
    out = {"Z": fit["Z"].detach().float().cpu(), "Y_idx": nz.to(torch.int32), "Y_val": Y[nz[:, 0], nz[:, 1]],
           "Y_shape": tuple(Y.shape), "widths": (Y != 0).sum(1).long(), "rel_err": float(fit["rel_err"]),
           "n_iter": fit["n_iter"], "stop_reason": fit["stop_reason"], "elapsed": fit["elapsed"],
           "config": fit["config"], "seed": fit["seed"], **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(out, tmp)
    tmp.replace(path)


def load_Y(fit: dict) -> torch.Tensor:
    Y = torch.zeros(fit["Y_shape"], dtype=torch.float32)
    idx = fit["Y_idx"].long()
    Y[idx[:, 0], idx[:, 1]] = fit["Y_val"]
    return Y


def run_fit(variant: str, concept: str, cond: str, seed: int, device: str, A_cache: dict) -> Path:
    from experiments.paper_analysis.robustness.concepts import CONDITIONS
    from experiments.varwidth.core import fit_varwidth
    path = fit_path(variant, cond, seed)
    if path.exists():
        return path
    matrix, method, target = CONDITIONS[cond]
    key = (variant, matrix, target)
    if key not in A_cache:
        A_cache.clear()
        meta = variant_meta(variant, concept)
        A_cache[key] = build_A(meta, matrix, target)
    A, info = A_cache[key]
    t0 = time.time()
    fit = fit_varwidth(A, cfg_for(method), seed, device)
    save_fit(path, fit, {"variant": variant, "concept": concept, "cond": cond, "matrix": matrix, "target": target,
                         "n_target": info.get("n_target")})
    print(f"[fit] {variant} {cond} seed {seed}: rel_err={fit['rel_err']:.4f} iters={fit['n_iter']} "
          f"({time.time() - t0:.1f}s, {device})", flush=True)
    return path
