#!/usr/bin/env python
"""WP6 edit_direction, Llama-3.1-8B-Instruct (multilingual study, worktree ../nlp_proj_ml, read only): the same
decomposition as gemma.py on the saved ML-EMBER (A2) and ML-CINDER (A3) fits. CPU only.

    .venv/bin/python -m experiments.paper_analysis.edit_direction.llama     # -> llama_runs.json, llama_cloud.json

Edits (c_t rows, raw Llama embedding space; tokens = concept-only V* columns touched by F_C, as evaluate.run_edit):
    a2       A2's own Z (fit on the sample, Y extrapolated to the full V*)             = ML-EMBER
    a2_o     A2's Y, Z re-solved on the raw full A
    a2_c     A2's Y, Z re-solved on the centered full A - mu_B
    a3       A3's Z mapped back with fit["unmap"] (= S^-1/2 W^-1, row form)              = ML-CINDER
    a3_c     A3's Y, Z re-solved on A - mu_B (the stationary mapped-back dictionary, Prop. 1)
    a3z      A3's Y, Z re-solved on the raw A                                            = A3z (a3z.py)
Read gain as diagnose_a3.read_gain: |W_in (g * c_hat)|, W_in = [q; k; v; gate; up] of layer 0, g = input RMSNorm
weight, over the median for random Gaussian directions (here 2000, seed 0); also the qkv-only read (Gemma's
definition).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch

from experiments.paper_analysis.edit_direction.common import OUT, q, save_json

ML = Path("/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml/results/ml")
SNAP = "/home/morg/NLP_2526b/oripetel/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/"
LAM = 1e-4


def load_weights():
    from safetensors import safe_open
    snap = Path(glob.glob(SNAP)[0])
    wm = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    names = ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight"] + \
        [f"model.layers.0.self_attn.{p}_proj.weight" for p in "qkv"] + \
        [f"model.layers.0.mlp.{p}_proj.weight" for p in ("gate", "up")]
    W = {}
    for n in names:
        with safe_open(str(snap / wm[n]), framework="pt") as f:
            W[n] = f.get_tensor(n).float()
    E = W.pop("model.embed_tokens.weight")
    g = W.pop("model.layers.0.input_layernorm.weight")
    Wqkv = torch.cat([W[f"model.layers.0.self_attn.{p}_proj.weight"] for p in "qkv"])
    Wmlp = torch.cat([W[f"model.layers.0.mlp.{p}_proj.weight"] for p in ("gate", "up")])
    return E, g, Wqkv, Wmlp


def main():
    torch.set_num_threads(4)
    E, g, Wqkv, Wmlp = load_weights()
    d = E.shape[1]

    def raw(U):
        U = U.double()
        U = (U / U.norm(dim=1, keepdim=True)).float() * g
        a = (U @ Wqkv.T).norm(dim=1).double()
        m = (U @ Wmlp.T).norm(dim=1).double()
        return {"all": (a ** 2 + m ** 2).sqrt(), "qkv": a}

    torch.manual_seed(0)
    R = torch.randn(2000, d)
    base = {k: float(v.median()) for k, v in raw(R).items()}

    def read(U):
        return {k: (v / base[k]).numpy() for k, v in raw(U).items()}

    st = torch.load(ML / "data" / "bg_ml_llama.pt", weights_only=False)
    mu, lam, Q = st["mu"].double(), st["lam"].double(), st["Q"].double()
    o = torch.argsort(lam, descending=True)
    lam, Q = lam[o], Q[:, o]
    dec_idx = torch.minimum(torch.arange(d) * 10 // d, torch.tensor(9))
    mh = mu / mu.norm()

    def spec(U):
        U = U.double()
        P = (U @ Q) ** 2
        P = P / P.sum(1, keepdim=True)
        dec = torch.zeros(U.shape[0], 10, dtype=torch.float64)
        dec.index_add_(1, dec_idx, P)
        return {"dec": dec.mean(0).tolist(), "ray": (P * lam).sum(1).numpy() / float(lam.mean()),
                "mu_share": ((U @ mh) ** 2 / (U ** 2).sum(1)).numpy()}

    def cosr(a, b):
        return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=1).numpy()

    def resolve(B, Y):
        return torch.linalg.solve(Y @ Y.T + LAM * torch.eye(Y.shape[0], dtype=Y.dtype), Y @ B.T).T

    def metrics(C, Et):
        rg, sp = read(C), spec(C)
        return {"n": int(C.shape[0]), "read_all": float(np.median(rg["all"])), "read_qkv": float(np.median(rg["qkv"])),
                "rayleigh": float(np.median(sp["ray"])), "mu_share": float(np.median(sp["mu_share"])),
                "dec_share": sp["dec"], "cos_ce": float(np.median(cosr(C, Et))),
                "removed": float(np.median((1 - (Et - C).norm(dim=1) / Et.norm(dim=1)).numpy())),
                "beta_mu": float(np.median((C @ mu / mu.dot(mu)).numpy()))}

    out = {"base_random_median": base, "mu_read": {k: float(v[0]) for k, v in read(mh[None]).items()},
           "mu_rayleigh": float(spec(mh[None])["ray"][0]),
           "mu_top1_energy": float((Q[:, 0] @ mh) ** 2), "lam1_over_mean": float(lam[0] / lam.mean()),
           "mu_norm_over_median_e": float(mu.norm() / E.double().norm(dim=1).median())}
    eig = read(Q.T)
    out["eig_read_by_decile"] = [float(np.mean(eig["all"][dec_idx.numpy() == i])) for i in range(10)]
    out["eig_read_qkv_by_decile"] = [float(np.mean(eig["qkv"][dec_idx.numpy() == i])) for i in range(10)]
    out["eig_read_top5"] = eig["all"][:5].tolist()
    print({k: v for k, v in out.items()}, flush=True)
    rows = []
    for d3 in sorted((ML / "runs").iterdir()):
        sel3 = d3 / "A3" / "selection.json"
        if not sel3.exists():
            continue
        F3 = json.loads(sel3.read_text())["F_C"]
        if not F3:
            continue
        row = {"concept": d3.name, "variants": {}}
        for ap in ("A2", "A3"):
            p = d3 / ap
            sel = p / "selection.json"
            if not sel.exists():
                continue
            F = json.loads(sel.read_text())["F_C"]
            if not F:
                continue
            fit = torch.load(p / "fit.pt", weights_only=False)
            Y, Z = fit["Y"].double(), fit["Z"].double()
            ids = list(fit["vstar_ids"])
            A = E[ids].double().T
            Ac = A - mu[:, None]
            codes = torch.as_tensor(fit["codes"])
            cols = torch.nonzero((Y[F] != 0).any(0) & (codes == 1)).flatten()
            Fy = Y[F][:, cols]
            Et = A.T[cols]

            def edit(Zm):
                return (Zm[:, F] @ Fy).T

            if ap == "A2":
                sets = {"a2": edit(Z), "a2_o": edit(resolve(A, Y)), "a2_c": edit(resolve(Ac, Y))}
            else:
                Cm = edit(Z) @ fit["unmap"].double()
                Cc, Co = edit(resolve(Ac, Y)), edit(resolve(A, Y))
                sets = {"a3": Cm, "a3_c": Cc, "a3z": Co}
                D = Co - Cc
                row["a3_pair"] = {"cos_a3z_vs_a3": q(cosr(Co, Cm)), "cos_a3c_vs_a3": q(cosr(Cc, Cm)),
                                  "cos_a3z_vs_a3c": q(cosr(Co, Cc)),
                                  "diff_zc_mu_share": q(((D @ mh) ** 2 / (D ** 2).sum(1)).numpy()),
                                  "e_read_all": float(np.median(read(Et)["all"])),
                                  "e_beta_mu": float(np.median((Et @ mu / mu.dot(mu)).numpy())),
                                  "rel_err_full_map_with_mu": float((Ac - (fit["unmap"].double().T @ Z) @ Y).norm()
                                                                    / A.norm()),
                                  "rel_err_full_c_with_mu": float((Ac - resolve(Ac, Y) @ Y).norm() / A.norm()),
                                  "rel_err_full_o": float((A - resolve(A, Y) @ Y).norm() / A.norm())}
            for k, C in sets.items():
                row["variants"][k] = metrics(C, Et)
            del A, Ac
        rows.append(row)
        print(d3.name, {k: (round(v["read_all"], 2), round(v["mu_share"], 2)) for k, v in row["variants"].items()},
              flush=True)
    out["runs"] = rows
    save_json(OUT / "llama_runs.json", out)


if __name__ == "__main__":
    main()
