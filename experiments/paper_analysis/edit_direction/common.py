"""WP6 edit_direction: shared loaders (Gemma-2-2B), the layer-0 read gain, the background spectrum, and the edit
variants of one (concept, seed) run. Read-only on every existing artifact.

Edit variants (rows c_t, TL-scaled embedding space; tokens = concept-only V* columns touched by F_C, as in
experiments/varwidth/collateral.py sparse_set):
    ember      L0 fit (plain, Fixed-1%), TJ, the fit's own Z                         (= EMBER as published)
    ember_c    EMBER's Y and F_C, Z re-solved against the CENTERED embeddings A - mu_B 1^T
    cinder     L12 fit (CS + GS), HJ = TJ or EJ@5, Z mapped back with T^-1            (= CINDER)
    cinder_c   CINDER's Y and F_C, Z re-solved against A - mu_B 1^T                  (identity check, Prop. 1)
    cinder_z   CINDER's Y and F_C, Z re-solved against the raw A                     (= CINDER-z, collateral.zo_set)
All re-solves: Z = B Y^T (Y Y^T + lam I)^-1, lam = 1e-4 (core.py step 1, collateral.zo_set).

Read gain (D16 / collateral_diag.a3z_offline): |[W_q; W_k; W_v] (g * c_hat)| for layer 0, g = 1 + input_layernorm
weight, relative to the mean over 500 random Gaussian unit directions (seed 0); median over edited tokens. Secondary
(Llama-style, diagnose_a3.read_gain): the MLP read |[W_gate; W_up] (g_ff * c_hat)| with g_ff = 1 + pre_feedforward
norm weight, and the concatenation [qkv; gate; up] of the two.
"""
from __future__ import annotations

import functools
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/morg/NLP_2526b/oripetel/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # gated repos resolve from the local cache only

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "paper_analysis" / "edit_direction"
VW = ROOT / "results" / "varwidth"
BG = ROOT / "experiments" / "whitening" / "data" / "gemma" / "bg_wiki_stats.pt"
SNAP = "/home/morg/NLP_2526b/oripetel/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/*/"
LAM = 1e-4
SEEDS = range(42, 47)
VARIANTS = ("ember", "ember_c", "cinder", "cinder_z")
N_DECILES = 10

COLORS = {"ember": "#7F7F7F", "ember_c": "#009E73", "cinder": "#D55E00", "cinder_z": "#E69F00", "cs": "#0072B2"}
LABELS = {"ember": "EMBER", "ember_c": "EMBER, centered Z", "cinder": "CINDER", "cinder_z": "CINDER-z"}


# ----------------------------------------------------------------------------------------------- layer-0 reads
@functools.lru_cache(maxsize=1)
def layer0():
    from safetensors import safe_open
    snap = glob.glob(SNAP)[0]
    W = {}
    for f in glob.glob(snap + "*.safetensors"):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.startswith("model.layers.0.") and k.endswith("weight"):
                    W[k.replace("model.layers.0.", "")] = h.get_tensor(k).double()
    g = 1 + W["input_layernorm.weight"]
    gf = 1 + W["pre_feedforward_layernorm.weight"]
    Wqkv = torch.cat([W[f"self_attn.{x}_proj.weight"] for x in "qkv"])
    Wmlp = torch.cat([W[f"mlp.{x}_proj.weight"] for x in ("gate", "up")])
    return {"g": g.float(), "gf": gf.float(), "qkv": Wqkv.float(), "mlp": Wmlp.float()}


def _raw_reads(U: torch.Tensor) -> dict:
    """Unnormalized layer-0 read norms of the rows of U (n, d) after unit-normalizing them."""
    L = layer0()
    U = U.double()
    U = (U / U.norm(dim=1, keepdim=True).clamp_min(1e-30)).float()
    q = (L["qkv"] @ (L["g"] * U).T).norm(dim=0)
    m = (L["mlp"] @ (L["gf"] * U).T).norm(dim=0)
    return {"qkv": q.double(), "mlp": m.double(), "all": (q.double() ** 2 + m.double() ** 2).sqrt()}


@functools.lru_cache(maxsize=1)
def random_base() -> dict:
    """Mean read norm of 500 random Gaussian unit directions (torch seed 0), as in collateral_diag.a3z_offline."""
    torch.manual_seed(0)
    R = torch.randn(500, layer0()["qkv"].shape[1], dtype=torch.float64)
    return {k: float(v.mean()) for k, v in _raw_reads(R).items()}


def read_gain(U: torch.Tensor) -> dict:
    """Per-row read gain (x random direction) for the qkv (primary), mlp and all reads. numpy arrays."""
    base = random_base()
    return {k: (v / base[k]).numpy() for k, v in _raw_reads(U).items()}


# ----------------------------------------------------------------------------------------------- background spectrum
@functools.lru_cache(maxsize=1)
def background():
    """mu_B, eigenvalues (descending) and eigenvectors of Sigma_B, decile index of every eigenvector."""
    st = torch.load(BG, weights_only=False)
    lam, Q = st["lam"].double(), st["Q"].double()
    o = torch.argsort(lam, descending=True)
    lam, Q = lam[o], Q[:, o]
    d = lam.numel()
    dec = torch.minimum(torch.arange(d) * N_DECILES // d, torch.tensor(N_DECILES - 1))
    return {"mu": st["mu"].double(), "lam": lam, "Q": Q, "dec": dec, "lam_mean": float(lam.mean())}


def spectral(U: torch.Tensor) -> dict:
    """Per row: energy share per Sigma_B eigen-decile (decile 0 = largest eigenvalues), share on the top eigenvector,
    Rayleigh ratio u^T Sigma_B u / (|u|^2 mean eigenvalue), and the energy share along mu_B."""
    bg = background()
    U = U.double()
    P = (U @ bg["Q"]) ** 2
    tot = P.sum(1, keepdim=True)
    P = P / tot
    dec = torch.zeros(U.shape[0], N_DECILES, dtype=torch.float64)
    dec.index_add_(1, bg["dec"], P)
    ray = (P * bg["lam"]).sum(1) / bg["lam_mean"]
    mh = bg["mu"] / bg["mu"].norm()
    mu_share = (U @ mh) ** 2 / (U ** 2).sum(1)
    return {"dec": dec.numpy(), "top1": P[:, 0].numpy(), "ray": ray.numpy(), "mu_share": mu_share.numpy()}


# ----------------------------------------------------------------------------------------------- runs
def concepts() -> list[tuple[str, str]]:
    """(concept, slug) for every concept with both an EMBER (L0) and a CS + GS (L12) fit."""
    from experiments.varwidth.layer_eval import CONFIGS
    out = []
    for p in sorted(VW.glob(f"*/{CONFIGS['L0']}")):
        sl = p.parent.name
        if any((VW / sl / CONFIGS["L12"]).glob("seed*/fit.pt")):
            prep = VW / "prep" / f"{sl}_plain_S300.pt"
            name = torch.load(prep, weights_only=False)["concept"]
            out.append((name, sl))
    return out


def fc_features(concept: str, rung: str, seed: int) -> list[int] | None:
    """F_C of a rung from the saved layer_cache.json (no rebuild, no judge call). Ancient Rome's EJ count uses the
    v1r word list, as collateral.fc_rows does. None if the run has no cache."""
    from experiments.varwidth.collateral import V1R
    from experiments.varwidth.j3_eval import j3_count
    from experiments.varwidth.j3_words import concept_lists
    from experiments.varwidth.layer_eval import RUNGS, fc, load
    cfg, judge = RUNGS[rung]
    cache = load(concept, cfg, seed)
    if cache is None:
        return None
    if concept in V1R:
        cw = concept_lists("v1r")[concept]
        cache = {**cache, "candidates": [{**c, "lo": j3_count(cw, c["fw20"])} for c in cache["candidates"]]}
    return [int(r["feature"]) for r in fc(cache, judge)]


@functools.lru_cache(maxsize=4)
def preps(slug: str):
    pp = torch.load(VW / "prep" / f"{slug}_plain_S300.pt", weights_only=False)
    pw = torch.load(VW / "prep" / f"{slug}_whitened_S300.pt", weights_only=False)
    assert list(pp["vstar_ids"]) == list(pw["vstar_ids"]), slug
    return pp, pw


def resolve(B: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Z = B Y^T (Y Y^T + lam I)^-1 (ridge least squares with Y fixed)."""
    k = Y.shape[0]
    return torch.linalg.solve(Y @ Y.T + LAM * torch.eye(k, dtype=Y.dtype), Y @ B.T).T


def run_edits(concept: str, slug: str, seed: int) -> dict | None:
    """Edit variants of one run plus the matrices needed for the reconstruction checks."""
    from experiments.varwidth.common import run_dir
    from experiments.varwidth.layer_eval import CONFIGS
    pp, pw = preps(slug)
    A = pp["A"].double()                                  # (d, n) raw TL-scaled E_V*^T
    mu = pw["mu_B"].double()
    assert torch.allclose(mu, background()["mu"]), "prep mu_B differs from bg_wiki_stats mu"
    Ac = A - mu[:, None]
    Ti = pw["T_inv"].double()
    codes = np.asarray(pp["codes"])
    out = {"A": A, "Ac": Ac, "Aw": pw["A"].double(), "T": pw["T_matrix"].double(), "mu": mu, "codes": codes,
           "vstar": [int(t) for t in pp["vstar_ids"]], "tokens": pp["tokens"], "sets": {}}
    for meth, rung, cfg in (("ember", "L0", "L0"), ("cinder", "L123", "L12")):
        p = run_dir(concept, CONFIGS[cfg], seed) / "fit.pt"
        if not p.exists():
            return None
        fit = torch.load(p, map_location="cpu", weights_only=False)
        Z, Y = fit["Z"].double(), fit["Y"].double()
        F = fc_features(concept, rung, seed)
        out[f"{meth}_fit"] = {"Z": Z, "Y": Y, "F": F, "rel_err": float(fit["rel_err"]), "n_iter": int(fit["n_iter"])}
        if not F:
            continue
        cols = np.where((Y[F].abs() > 0).any(0).numpy() & (codes == 1))[0].tolist()
        Fy = Y[F][:, cols]

        def edit(Zm):
            return (Zm[:, F] @ Fy).T

        if meth == "ember":
            out["sets"]["ember"] = {"C": edit(Z), "cols": cols, "F": F}
            out["sets"]["ember_c"] = {"C": edit(resolve(Ac, Y)), "cols": cols, "F": F}
        else:
            out["sets"]["cinder"] = {"C": edit(Ti.T @ Z), "cols": cols, "F": F}
            out["sets"]["cinder_c"] = {"C": edit(resolve(Ac, Y)), "cols": cols, "F": F}
            out["sets"]["cinder_z"] = {"C": edit(resolve(A, Y)), "cols": cols, "F": F}
    return out


def q(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"p10": float(np.percentile(x, 10)), "p50": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "mean": float(x.mean())}


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
