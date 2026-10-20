#!/usr/bin/env python
"""How big is each method's edit at the same delta? Sets CINDER's delta grid (CPU, no model).

Both methods edit a concept token t as e_t <- e_t - delta c_t, with c_t = F' G'_t (selected features only).
delta only scales c_t, so what the grid really spans is the relative edit size delta * ||c_t|| / ||e_t||
(e_t in the scaled space the edit works in, x sqrt(d) for Gemma). For every edited token of the 6 concepts:
    s_t = ||c_t|| / ||e_t||   (the relative edit at delta = 1),   cos(c_t, e_t)
for EMBER (Part 1 features, mf_outputs/) and CINDER (mf_outputs/cinder/).

Rule (decided 2026-10-18): r = median s_t (EMBER) / median s_t (CINDER), pooled over the edited tokens of the
concepts that both methods edit.
    0.5 <= r <= 2   CINDER uses EMBER's grid as is: 0.5, 1, 2, 5, 10, 50, 100, 200
    otherwise       CINDER's grid = EMBER's grid x r (2 significant digits), so each grid point makes the same
                    median relative edit as the EMBER point it replaces.
Output: judge_rerun/Cinder/delta_grid.json (read by cinder.grid) and edit_size.csv next to it.

    python -m cinder.edit_size [--concepts ...]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch

from cinder import common as C


def edit_vectors(concept: str, root) -> tuple[list[int], torch.Tensor]:
    """(token ids, c_t rows) exactly as embed_edit._erase_factored builds them, from the features under root."""
    from ember.erasure import embed_edit, features
    features.MF_OUTPUTS_ROOT = root
    pkl, pot = features._embedding_paths(C.MODEL_NAME, concept, C.RANK, C.SEED)
    F, G, vprime = features.load_embedding_payload(pkl)
    pot_df = pd.read_csv(pot)
    if pot_df.empty:
        return [], torch.zeros(0, F.shape[0])
    feat = features.select_embed_feature_ids(pot_df, ratio_thresh=2.0)
    ctok = embed_edit._build_concept_token_set(C.MODEL_NAME, concept, C.RANK, C.SEED, vprime)
    Gp = torch.as_tensor(G[:, feat])
    tids, rows = embed_edit._eligible_tids_rows(vprime, Gp, ctok or set())
    if not tids:
        return [], torch.zeros(0, F.shape[0])
    Cmat = (torch.as_tensor(F[:, feat]).double() @ Gp[rows].double().T).T      # (n_tok, d)
    return tids, Cmat


def round_sig(x: float, n: int = 2) -> float:
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concepts", nargs="+", default=C.CONCEPTS)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    from experiments.varwidth.prep import load_input_embeddings
    E = load_input_embeddings().double()                                         # scaled, as the edit sees it

    rows = []
    for concept in a.concepts:
        for method, root in (("EMBER", C.EMBER_MF_ROOT), ("CINDER", C.MF_ROOT)):
            tids, Cm = edit_vectors(concept, root)
            if not tids:
                rows.append({"concept": concept, "method": method, "token_id": None})
                continue
            e = E[torch.tensor(tids)]
            s = (Cm.norm(dim=1) / e.norm(dim=1)).numpy()
            cos = torch.nn.functional.cosine_similarity(Cm, e, dim=1).numpy()
            rows += [{"concept": concept, "method": method, "token_id": int(t), "s": float(si), "cos": float(ci)}
                     for t, si, ci in zip(tids, s, cos)]
    df = pd.DataFrame(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OUT / "edit_size.csv", index=False)

    tok = df.dropna(subset=["token_id"])
    per = (tok.groupby(["concept", "method"])
              .agg(n_tokens=("s", "size"), s_median=("s", "median"), s_q10=("s", lambda x: x.quantile(.1)),
                   s_q90=("s", lambda x: x.quantile(.9)), cos_median=("cos", "median"))
              .reset_index())
    paired = [c for c, g in tok.groupby("concept") if set(g["method"]) == {"EMBER", "CINDER"}]
    med = tok[tok["concept"].isin(paired)].groupby("method")["s"].median()   # concepts both methods edit
    if "CINDER" not in med or "EMBER" not in med:
        raise SystemExit("no edited tokens for one of the methods: build the CINDER features first")
    r = float(med["EMBER"] / med["CINDER"])
    if 0.5 <= r <= 2.0:
        grid, rule = list(C.EMBER_GRID), "0.5 <= r <= 2: EMBER's grid unchanged"
    else:
        grid, rule = [round_sig(d * r) for d in C.EMBER_GRID], "r outside [0.5, 2]: EMBER's grid x r"
    per_concept_r = {c: float(g.set_index("method").loc["EMBER", "s_median"] / g.set_index("method").loc["CINDER", "s_median"])
                     for c, g in per.groupby("concept") if set(g["method"]) == {"EMBER", "CINDER"}}
    out = {
        "grid": grid, "rule": rule, "r": r,
        "median_s": {"EMBER": float(med["EMBER"]), "CINDER": float(med["CINDER"])},
        "per_concept_r": per_concept_r,
        "median_relative_edit_at_grid": {
            "EMBER": {str(d): float(med["EMBER"] * d) for d in C.EMBER_GRID},
            "CINDER": {str(d): float(med["CINDER"] * d) for d in grid}},
        "paired_concepts": paired,
        "concepts_without_cinder_edit": sorted(set(a.concepts) - set(tok.loc[tok.method == "CINDER", "concept"])),
        "per_concept": per.to_dict(orient="records"),
    }
    C.write_json(C.GRID_JSON, out)
    print(per.to_string(index=False))
    print(f"\nmedian s: EMBER {med['EMBER']:.4f}  CINDER {med['CINDER']:.4f}  r = {r:.3f}  ({rule})")
    print(f"per-concept r: { {k: round(v, 3) for k, v in per_concept_r.items()} }")
    print(f"CINDER grid: {grid} -> {C.GRID_JSON}")


if __name__ == "__main__":
    main()
