#!/usr/bin/env python
"""Gemma reference: the same judge-free metrics (llama_cs.metrics) on the EXISTING Gemma fits
results/varwidth/<slug>/<tag>/seed<N>/fit.pt, tags EMBER = fixed_refit-False_plain (L0) and
CS + RS-1% = fixed_refit-False (L1, eta = 0.05, gamma = 1), 18 EMBER concepts, seeds 42-46 where present.

    python -m experiments.paper_analysis.llama_cs.gemma_ref
Output: results/paper_analysis/llama_cs/gemma_ref.json  {"<concept>|<cfg>|<seed>": metrics}
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.paper_analysis.llama_cs.common import CONCEPTS18, OUT, ROOT, read_json, slug, write_json
from experiments.paper_analysis.llama_cs.metrics import fit_metrics

TAGS = {"raw": "fixed_refit-False_plain", "cs_e0.05_g1": "fixed_refit-False"}


def main():
    torch.set_num_threads(4)
    path = OUT / "gemma_ref.json"
    res = read_json(path) if path.exists() else {}
    for c in CONCEPTS18:
        prep = torch.load(ROOT / "results" / "varwidth" / "prep" / f"{slug(c)}_plain_S300.pt", map_location="cpu",
                          weights_only=False)
        ids, codes = list(prep["vstar_ids"]), np.asarray(prep["codes"])
        for cfg, tag in TAGS.items():
            for seed in range(42, 47):
                key = f"{c}|{cfg}|{seed}"
                p = ROOT / "results" / "varwidth" / slug(c) / tag / f"seed{seed}" / "fit.pt"
                if key in res or not p.exists():
                    continue
                f = torch.load(p, map_location="cpu", weights_only=False)
                m = fit_metrics(f["Y"].float(), ids, codes, "gemma", c)
                m.update({"concept": c, "config": cfg, "seed": seed, "model": "gemma", "rel_err": float(f["rel_err"])})
                res[key] = m
                print(f"[gemma {c} {cfg} s{seed}] n_tau={m['n_tau']} cf lo={m['cf_lo']} rank={m['cf_rank']} "
                      f"rho={m['cf_rho']:.1f} pur={m['cf_purity']:.2f} auc={m['auc']:.3f}", flush=True)
        write_json(path, res, indent=None)


if __name__ == "__main__":
    main()
