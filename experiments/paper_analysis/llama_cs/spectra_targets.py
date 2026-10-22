#!/usr/bin/env python
"""Target variants on Llama (and Gemma page for reference): page, leak-free (page minus V*), 8-language pooled (ml8).
Per (concept, target): m, n_signal (eigenvalues of M above the largest eigenvalue of a same-size random background-type
null), lam_1, null lam_1, mean self-share of V* concept-only tokens that are in the target (gamma = 1, eta = 0.05),
on/off-target energy boost, and the concept-only energy share E_C of X (eta = 0.05, gamma = 1).

    python -m experiments.paper_analysis.llama_cs.spectra_targets [--device cuda:0]
Output: results/paper_analysis/llama_cs/spectra/targets.json
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from experiments.paper_analysis.llama_cs.common import (FIT_CONCEPTS, OUT, ml_pooled_target, page_target, vstar,
                                                        write_json)
from experiments.paper_analysis.llama_cs.spectra import bg_types, mom, whiten

ETA = 0.05


def one(model, concept, kind, dev):
    v = vstar(model, concept)
    ids, codes = v["vstar_ids"], np.asarray(v["codes"])
    if kind == "page":
        tgt = page_target(model, concept)
    elif kind == "lf":
        vs = set(ids)
        tgt = [t for t in page_target(model, concept) if t not in vs]
    else:
        tgt = ml_pooled_target(concept)
    m = len(tgt)
    U = whiten(model, tgt, dev)
    lam, P = torch.linalg.eigh(mom(U))
    lam, P = lam.flip(0).clamp_min(0), P.flip(1)
    tset = set(tgt)
    pool = [t for t in bg_types(model) if t not in tset]
    if len(pool) < m:   # ml8 targets exceed the English background pool: null from random vocabulary rows instead
        from experiments.paper_analysis.llama_cs.common import embeddings, excluded_ids
        ex = excluded_ids(model)
        pool = [t for t in range(embeddings(model).shape[0]) if t not in tset and t not in ex]
    nul = random.Random(1).sample(pool, m)
    lam0 = torch.linalg.eigvalsh(mom(whiten(model, nul, dev))).flip(0)
    on = np.array([t in tset for t in ids])
    Uv = whiten(model, ids, dev)
    c2 = (Uv @ P) ** 2
    nu2 = c2.sum(1)
    uMu = (c2 * lam).sum(1)
    sel = torch.as_tensor(on & (codes == 1), device=dev)
    g = (1 - ETA) * lam + ETA
    ex = (c2 * g).sum(1)
    ratio = (ex / nu2).cpu().numpy()
    self_share = ((1 - ETA) * nu2 ** 2 / m) / ((1 - ETA) * uMu + ETA * nu2)
    return {"model": model, "concept": concept, "target": kind, "m": m, "m_over_d": m / lam.numel(),
            "n_signal": int((lam > lam0[0]).sum()), "lam1": float(lam[0]), "null_lam1": float(lam0[0]),
            "frac_C_in_target": float(on[codes == 1].mean()),
            "self_share": float(self_share[sel].mean()) if sel.any() else 0.0,
            "boost": float(ratio[on].mean() / ratio[~on].mean()) if on.any() and (~on).any() else float("nan"),
            "E_C": float(ex[torch.as_tensor(codes == 1, device=dev)].sum() / ex.sum()),
            "count_C": float((codes == 1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = []
    for c in FIT_CONCEPTS:
        for kind in ("page", "lf", "ml8"):
            r = one("llama", c, kind, a.device)
            out.append(r)
            print(r, flush=True)
            torch.cuda.empty_cache()
    write_json(OUT / "spectra" / "targets.json", out)


if __name__ == "__main__":
    main()
