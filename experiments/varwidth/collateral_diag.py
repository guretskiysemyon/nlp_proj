#!/usr/bin/env python
"""Saved provenance for numbers quoted in CINDER_collateral_evaluation.md (CPU only).

    python -m experiments.varwidth.collateral_diag mu      # median mu = |c|/|e| and cos(c, e), EMBER vs CS+GS, anchor seed 42
    python -m experiments.varwidth.collateral_diag pilot   # pilot grid with collateral on the variant's own edited tokens
    python -m experiments.varwidth.collateral_diag a3z     # amendment 7 offline check (alignment, layer-0 read strength)

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


def a3z_offline():
    """Amendment 7 offline check: per edit set, median cos(c, e), median mu = |c|/|e|, and layer-0 attention read
    strength (|[W_q; W_k; W_v] (g * c_hat)|, g = 1 + input_layernorm weight) relative to 500 random directions; for
    the A3z sets also the cosine to the parent's mapped-back edit. -> diag/a3z_offline.json"""
    import glob as _glob

    from safetensors import safe_open

    from experiments.varwidth.collateral import POC, edit_sets, load_fit, seeds
    snap = _glob.glob("/home/morg/NLP_2526b/oripetel/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/*/")[0]
    W = {}
    for f in _glob.glob(snap + "*.safetensors"):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.startswith("model.layers.0.") and any(x in k for x in ("input_layernorm", "q_proj", "k_proj", "v_proj")):
                    W[k] = h.get_tensor(k).double()
    g = 1 + W["model.layers.0.input_layernorm.weight"]
    Wcat = torch.cat([W[f"model.layers.0.self_attn.{x}_proj.weight"] for x in ("q", "k", "v")])

    def read(u):
        u = u / u.norm(dim=1, keepdim=True)
        return (Wcat @ (g * u).T).norm(dim=0)
    torch.manual_seed(0)
    rnd = read(torch.randn(500, Wcat.shape[1], dtype=torch.float64)).mean().item()
    out = {}
    for c in POC:
        for s in seeds(c):
            st = edit_sets(c, s)
            E = load_fit(c, "L0", s)[0]["A"].double().T
            row = {}
            for v in ("ember", "cinder", "cinder_z", "cinder_rs", "cinder_rs_z"):
                x = st[v]
                if not x["ids"]:
                    continue
                C, e = x["C"].double(), E[x["cols"]]
                row[v] = [float(np.median(torch.nn.functional.cosine_similarity(C, e, dim=1).numpy())),
                          float(np.median((C.norm(dim=1) / e.norm(dim=1)).numpy())), read(C).median().item() / rnd]
            for a, b in (("cinder_z", "cinder"), ("cinder_rs_z", "cinder_rs")):
                if a in row:
                    row[a].append(torch.nn.functional.cosine_similarity(st[a]["C"].double(), st[b]["C"].double(),
                                                                        dim=1).median().item())
            out[f"{c} {s}"] = row
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / "a3z_offline.json").write_text(json.dumps(out, indent=1))
    print(f"-> {DIAG / 'a3z_offline.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("mu", "pilot", "a3z"))
    a = ap.parse_args()
    {"mu": mu_cos, "pilot": pilot, "a3z": a3z_offline}[a.what]()


if __name__ == "__main__":
    main()
