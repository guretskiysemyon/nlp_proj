#!/usr/bin/env python
"""How much of a CS feature's mapped-back direction comes from the eta floor (the null space of M when m < d)?

For an X-space feature z~ (column of Z) with coordinates c_j = p_j . z~ in the eigenbasis of M, the whitened-space
direction is S_eta^{-gamma/2} z~ = sum_j c_j g_j^{-gamma/2} p_j. floor_share = sum_{j: g_j < 1.1 eta} c_j^2 g_j^-gamma
/ sum_j c_j^2 g_j^-gamma (share of the squared norm before W^-1), and floor_share_x = the same share of z~ itself
(before the map). Reported for the lexicon-identified concept feature and the mean over the 20 highest-rho features.

    python -m experiments.paper_analysis.llama_cs.mapback [--device cuda:0]
Output: results/paper_analysis/llama_cs/mapback.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from experiments.paper_analysis.llama_cs.common import FITS, FIT_CONCEPTS, OUT, page_target, slug, write_json
from experiments.paper_analysis.llama_cs.spectra import mom, whiten

CONFIGS = ["cs_e0.05_g1", "cs_e0.05_g0.5", "cs_e0.5_g1", "cs_e0.5_g0.5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="llama")
    a = ap.parse_args()
    out = []
    for c in FIT_CONCEPTS:
        U = whiten(a.model, page_target(a.model, c), a.device)
        lam, P = torch.linalg.eigh(mom(U))
        lam, P = lam.flip(0).clamp_min(0), P.flip(1)
        for cfg in CONFIGS:
            eta, gamma = float(cfg.split("_")[1][1:]), float(cfg.split("_")[2][1:])
            g = (1 - eta) * lam + eta
            floor = g < 1.1 * eta
            for seed in (42, 43, 44):
                d = FITS / a.model / slug(c) / cfg / f"seed{seed}"
                if not (d / "fit.pt").exists():
                    continue
                f = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)
                met = json.loads((d / "metrics.json").read_text())
                Z = f["Z"].double().to(a.device)                       # (d, k)
                C2 = (P.T @ Z) ** 2                                    # (d, k)
                back = C2 * g.pow(-gamma).unsqueeze(1)
                fs_back = (back[floor].sum(0) / back.sum(0)).cpu().numpy()
                fs_x = (C2[floor].sum(0) / C2.sum(0)).cpu().numpy()
                top = np.argsort(-np.asarray(met["rho_all"]), kind="stable")[:20]
                out.append({"concept": c, "config": cfg, "seed": seed, "floor_dims": int(floor.sum()),
                            "floor_frac": float(floor.float().mean()),
                            "cf_floor_share_back": float(fs_back[met["cf"]]), "cf_floor_share_x": float(fs_x[met["cf"]]),
                            "top20_floor_share_back": float(fs_back[top].mean()),
                            "top20_floor_share_x": float(fs_x[top].mean()),
                            "all_floor_share_back": float(fs_back.mean())})
                print(out[-1], flush=True)
        torch.cuda.empty_cache()
    write_json(OUT / "mapback.json", out)


if __name__ == "__main__":
    main()
