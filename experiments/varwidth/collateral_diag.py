#!/usr/bin/env python
"""Saved provenance for numbers quoted in CINDER_collateral_evaluation.md (CPU only).

    python -m experiments.varwidth.collateral_diag mu      # median mu = |c|/|e| and cos(c, e), EMBER vs CS+GS, anchor seed 42
    python -m experiments.varwidth.collateral_diag pilot   # pilot grid with collateral on the variant's own edited tokens

Outputs results/varwidth/layers/collateral/diag/{mu_cos.json, pilot_own_tokens.json}.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch

from experiments.varwidth.collateral import OUT

DIAG = OUT / "diag"


def mu_cos(concepts=("Harry Potter", "Baseball", "Gun")):
    from experiments.varwidth.erase import edit_vectors
    from experiments.varwidth.prep import prepare
    out = {}
    for c in concepts:
        for rung in ("L0", "L12"):
            ids, C, F, info = edit_vectors(c, rung, 42)
            whit = info.get("whitened")
            prep = prepare(c, "whitened" if whit else "plain", verbose=False)
            vid = [int(t) for t in prep["vstar_ids"]]
            idx = [vid.index(i) for i in ids]
            A = prep["A"][:, idx].T.double()
            E = A @ prep["T_inv"].double() + prep["mu_B"].double() if whit else A
            C = C.double()
            r = (C.norm(dim=1) / E.norm(dim=1)).numpy()
            cos = torch.nn.functional.cosine_similarity(C, E, dim=1).numpy()
            out[f"{c}|{rung}"] = {"n_edit": len(ids), "mu_median": float(np.median(r)), "mu_min": float(r.min()),
                                  "mu_max": float(r.max()), "cos_median": float(np.median(cos))}
            print(c, rung, out[f"{c}|{rung}"], flush=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / "mu_cos.json").write_text(json.dumps(out, indent=1))


def pilot():
    from experiments.varwidth.erase_v2 import coll_from_tok
    sets = torch.load(OUT / "Harry_Potter" / "edits_s42.pt", weights_only=False)
    rows = []
    for f in sorted(glob.glob(str(OUT / "Harry_Potter" / "_pilot_s42_v0metric" / "*" / "grid.json"))):
        v = f.split("/")[-2]
        ids = sets[v.split("_", 1)[1]]["ids"]
        for r in json.load(open(f)):
            own, union, n = coll_from_tok(r["coll_tok"], ids)
            rows.append({"variant": v, "r": r["r"], "efficacy": r["efficacy"], "specificity": r["specificity"],
                         "collateral_own": own, "collateral_union": union, "n_own_tokens": n})
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / "pilot_own_tokens.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} rows -> {DIAG / 'pilot_own_tokens.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("mu", "pilot"))
    a = ap.parse_args()
    {"mu": mu_cos, "pilot": pilot}[a.what]()


if __name__ == "__main__":
    main()
