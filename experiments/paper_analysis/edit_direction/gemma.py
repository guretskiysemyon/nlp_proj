#!/usr/bin/env python
"""WP6 edit_direction, Gemma-2-2B: per-run direction metrics of EMBER, EMBER (centered Z), CINDER, CINDER-z, the
reconstruction checks of the proposition, the eigenvector read profile and the random-direction clouds. CPU only.

    .venv/bin/python -m experiments.paper_analysis.edit_direction.gemma runs     # -> gemma_runs.json, gemma_runs.csv
    .venv/bin/python -m experiments.paper_analysis.edit_direction.gemma cloud    # -> gemma_cloud.json
    .venv/bin/python -m experiments.paper_analysis.edit_direction.gemma check    # -> check_collateral.json

Definitions in common.py. Every number is a median over the edited tokens of a run unless named otherwise.
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch

from experiments.paper_analysis.edit_direction.common import (OUT, SEEDS, background, concepts, q, read_gain,
                                                              run_edits, save_json, spectral)


def cosr(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.cosine_similarity(a, b, dim=1).numpy()


def perp(U: torch.Tensor, mh: torch.Tensor) -> torch.Tensor:
    return U - (U @ mh)[:, None] * mh[None]


def variant_metrics(C: torch.Tensor, E: torch.Tensor, mu: torch.Tensor) -> dict:
    mh = mu / mu.norm()
    rg = read_gain(C)
    sp = spectral(C)
    en = E.norm(dim=1)
    Cp = perp(C, mh)
    rgp, spp = read_gain(Cp), spectral(Cp)
    m = {"n": int(C.shape[0]),
         "read_qkv_perp_mu": float(np.median(rgp["qkv"])), "rayleigh_perp_mu": float(np.median(spp["ray"])),
         "dec_share_perp_mu": spp["dec"].mean(0).tolist(),
         "read_qkv": float(np.median(rg["qkv"])), "read_mlp": float(np.median(rg["mlp"])),
         "read_all": float(np.median(rg["all"])),
         "rayleigh": float(np.median(sp["ray"])), "rayleigh_gmean": float(np.exp(np.log(sp["ray"]).mean())),
         "dec_share": sp["dec"].mean(0).tolist(), "top1_share": float(sp["top1"].mean()),
         "mu_share": float(np.median(sp["mu_share"])), "mu_share_mean": float(sp["mu_share"].mean()),
         "cos_ce": float(np.median(cosr(C, E))),
         "cos_ce_perp_mu": float(np.median(cosr(perp(C, mh), perp(E, mh)))),
         "ratio_ce": float(np.median((C.norm(dim=1) / en).numpy())),
         "removed": float(np.median((1 - (E - C).norm(dim=1) / en).numpy())),
         "proj_e": float(np.median(((C * E).sum(1) / en ** 2).numpy())),
         "beta_mu": float(np.median((C @ mu / mu.dot(mu)).numpy())),
         # share of the alignment c.e that runs through mu: (c.mh)(e.mh) / (c.e)
         "align_via_mu": float(np.median(((C @ mh) * (E @ mh) / (C * E).sum(1)).numpy())),
         "read_qkv_q": q(rg["qkv"]), "rayleigh_q": q(sp["ray"]), "mu_share_q": q(sp["mu_share"]),
         "cos_ce_q": q(cosr(C, E))}
    return m


def rel(X: torch.Tensor, R: torch.Tensor) -> float:
    return float(R.norm() / X.norm())


def recon(run: dict) -> dict:
    """Reconstruction errors of the CINDER fit (all V* columns and the edited columns) under the three dictionaries:
    mapped back (Z_map), centered re-solve (Z_c), raw re-solve (Z_o); with and without the intercept mu_B."""
    from experiments.paper_analysis.edit_direction.common import resolve
    f = run["cinder_fit"]
    Y, Zt = f["Y"], f["Z"]
    A, Ac, Aw, T = run["A"], run["Ac"], run["Aw"], run["T"]
    Ti = torch.linalg.inv(T)
    Zmap, Zc, Zo = Ti.T @ Zt, resolve(Ac, Y), resolve(A, Y)
    out = {}
    cols = run["sets"]["cinder"]["cols"] if "cinder" in run["sets"] else None
    for name, idx in (("all", slice(None)), ("edited", cols)):
        if idx is None:
            continue
        a, ac, aw, y = A[:, idx], Ac[:, idx], Aw[:, idx], Y[:, idx]
        out[name] = {
            "cs_objective_map": rel(aw, aw - Zt @ y),                  # ||T^T(Ac - Z_map Y)|| / ||T^T Ac|| (whitened)
            "cs_objective_c": rel(aw, aw - T.T @ Zc @ y),
            "cs_objective_o_centered": rel(aw, aw - T.T @ (Zo @ y - run["mu"][:, None])),
            "eucl_centered_map": rel(ac, ac - Zmap @ y),               # raw metric, centered target
            "eucl_centered_c": rel(ac, ac - Zc @ y),
            "eucl_raw_map_with_mu": rel(a, ac - Zmap @ y),             # mu_B + Z_map y_t vs e_t, relative to |A|
            "eucl_raw_o": rel(a, a - Zo @ y),                          # CINDER-z's dictionary, no intercept
            "eucl_raw_map_no_mu": rel(a, a - Zmap @ y),                # mapped-back dictionary used without mu_B
            "eucl_raw_c_no_mu": rel(a, a - Zc @ y),
        }
    out["Z_map_vs_Z_c"] = float((Zmap - Zc).norm() / Zc.norm())
    out["fit_rel_err"] = f["rel_err"]
    if run["ember_fit"]["F"] is not None:
        e = run["ember_fit"]
        out["ember_eucl_raw"] = rel(A, A - e["Z"] @ e["Y"])
        out["ember_Z_vs_resolve"] = float((e["Z"] - resolve(A, e["Y"])).norm() / e["Z"].norm())
    return out


def one_run(concept: str, slug: str, seed: int) -> dict | None:
    run = run_edits(concept, slug, seed)
    if run is None:
        return None
    A, mu = run["A"], run["mu"]
    row = {"concept": concept, "seed": seed, "variants": {},
           "F": {k: run[f"{k}_fit"]["F"] for k in ("ember", "cinder")}}
    for v, st in run["sets"].items():
        if v == "cinder_c":
            continue
        E = A.T[st["cols"]]
        row["variants"][v] = variant_metrics(st["C"], E, mu)
    S = run["sets"]
    if "cinder" in S:
        Cm, Cc, Co = S["cinder"]["C"], S["cinder_c"]["C"], S["cinder_z"]["C"]
        E = A.T[S["cinder"]["cols"]]
        D = Co - Cm
        mh = mu / mu.norm()
        row["cinder_pair"] = {
            "cos_map_vs_centered_resolve": q(cosr(Cm, Cc)),
            "cos_z_vs_map": q(cosr(Co, Cm)),
            # how much of the difference CINDER-z - CINDER lies along mu_B (1 = exactly rank-one mu term)
            "diff_mu_share": q(((D @ mh) ** 2 / (D ** 2).sum(1)).numpy()),
            "diff_norm_over_c": q((D.norm(dim=1) / Cm.norm(dim=1)).numpy()),
            "e_read_qkv": float(np.median(read_gain(E)["qkv"])),
            "e_rayleigh": float(np.median(spectral(E)["ray"])),
            "e_mu_share": float(np.median(spectral(E)["mu_share"])),
            "e_beta_mu": float(np.median((E @ mu / mu.dot(mu)).numpy())),
        }
        row["recon"] = recon(run)
    if "ember" in S:
        row["ember_pair"] = {"cos_c_vs_native": q(cosr(S["ember_c"]["C"], S["ember"]["C"]))}
    if "ember" in S and "cinder" in S:
        se, sc = S["ember"]["cols"], S["cinder"]["cols"]
        row["overlap"] = {"n_ember": len(se), "n_cinder": len(sc), "n_shared": len(set(se) & set(sc))}
    return row


def runs():
    torch.set_num_threads(4)
    t0 = time.time()
    rows = []
    for concept, slug in concepts():
        for s in SEEDS:
            r = one_run(concept, slug, s)
            if r is None:
                print(f"[skip] {concept} {s}: no fit", flush=True)
                continue
            rows.append(r)
            v = r["variants"]
            msg = " ".join(f"{k}={v[k]['read_qkv']:.2f}/{v[k]['mu_share']:.2f}" for k in v)
            print(f"{concept:28s} {s} {msg} ({time.time() - t0:.0f}s)", flush=True)
    save_json(OUT / "gemma_runs.json", rows)
    keys = ["n", "read_qkv", "read_mlp", "read_all", "read_qkv_perp_mu", "rayleigh", "rayleigh_gmean", "rayleigh_perp_mu", "top1_share", "mu_share",
            "cos_ce", "cos_ce_perp_mu", "ratio_ce", "removed", "proj_e", "beta_mu", "align_via_mu"]
    with open(OUT / "gemma_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["concept", "seed", "variant"] + keys + [f"dec{i + 1}" for i in range(10)])
        for r in rows:
            for v, m in r["variants"].items():
                w.writerow([r["concept"], r["seed"], v] + [m[k] for k in keys] + m["dec_share"])
    print(f"{len(rows)} runs -> {OUT / 'gemma_runs.json'}")


def cloud():
    """Read gain vs spectral position for (i) every eigenvector of Sigma_B, (ii) 3000 spectrally tilted Gaussian
    directions Q diag(lam^(alpha/2)) g, alpha ~ U[-3, 3], (iii) 1000 mixtures sqrt(1-s) r_perp + sqrt(s) mu_hat,
    s ~ U[0, 1], (iv) 2000 isotropic Gaussian directions, (v) 5000 random vocabulary embeddings."""
    torch.set_num_threads(4)
    bg = background()
    lam, Q, mu = bg["lam"], bg["Q"], bg["mu"]
    gen = torch.Generator().manual_seed(1)
    d = lam.numel()
    out = {"lam_over_mean": (lam / bg["lam_mean"]).tolist()}
    rg = read_gain(Q.T)
    out["eig"] = {k: v.tolist() for k, v in rg.items()}
    mh = mu / mu.norm()
    out["eig_cos_mu"] = (Q.T @ mh).tolist()
    alpha = torch.rand(3000, generator=gen, dtype=torch.float64) * 6 - 3
    G = torch.randn(3000, d, generator=gen, dtype=torch.float64)
    V = (G * lam.clamp_min(1e-12)[None].pow(alpha[:, None] / 2)) @ Q.T
    sets = {"tilt": V}
    s = torch.rand(1000, generator=gen, dtype=torch.float64)
    R = torch.randn(1000, d, generator=gen, dtype=torch.float64)
    R = R - (R @ mh)[:, None] * mh[None]
    R = R / R.norm(dim=1, keepdim=True)
    sets["mu_mix"] = (1 - s).sqrt()[:, None] * R + s.sqrt()[:, None] * mh[None]
    sets["iso"] = torch.randn(2000, d, generator=gen, dtype=torch.float64)
    from experiments.varwidth.prep import load_input_embeddings
    Evoc = load_input_embeddings().double()
    idx = torch.randperm(Evoc.shape[0], generator=gen)[:5000]
    sets["vocab"] = Evoc[idx]
    sets["mu"] = mh[None]
    for k, U in sets.items():
        rg, sp = read_gain(U), spectral(U)
        out[k] = {"read_qkv": rg["qkv"].tolist(), "read_mlp": rg["mlp"].tolist(), "read_all": rg["all"].tolist(),
                  "rayleigh": sp["ray"].tolist(), "mu_share": sp["mu_share"].tolist()}
        if k == "tilt":
            out[k]["alpha"] = alpha.tolist()
        print(k, "read median", float(np.median(rg["qkv"])), "ray median", float(np.median(sp["ray"])), flush=True)
    save_json(OUT / "gemma_cloud.json", out)


def check():
    """My edit sets vs the cached collateral edit sets (edits_s<seed>.pt) of the 15 collateral concepts."""
    from experiments.varwidth.collateral import OUT as COL
    from experiments.varwidth.common import slug as slugf
    res = {}
    for concept, sl in concepts():
        for s in SEEDS:
            p = COL / slugf(concept) / f"edits_s{s}.pt"
            if not p.exists():
                continue
            ref = torch.load(p, weights_only=False)
            run = run_edits(concept, sl, s)
            r = {}
            for v in ("ember", "cinder", "cinder_z"):
                if v not in ref:
                    continue
                mine = run["sets"].get(v)
                ids_ref = ref[v]["ids"]
                ids_mine = [run["vstar"][j] for j in mine["cols"]] if mine else []
                ok_ids = ids_ref == ids_mine
                cmin = float(cosr(mine["C"], ref[v]["C"].double()).min()) if (mine and ok_ids and ids_ref) else None
                r[v] = {"same_ids": ok_ids, "n": len(ids_ref), "min_cos": cmin}
            res[f"{concept} {s}"] = r
            print(concept, s, r, flush=True)
    save_json(OUT / "check_collateral.json", res)


def tokens():
    """Mean coordinate beta = e.mu_B / |mu_B|^2 of V* tokens by class (concept-only, neutral-only, both) per concept,
    and of 20,000 random vocabulary rows (the background average is 1 by construction)."""
    from experiments.paper_analysis.edit_direction.common import preps
    from experiments.varwidth.prep import load_input_embeddings
    mu = background()["mu"]
    out = {"per_concept": {}}
    for concept, sl in concepts():
        pp, _ = preps(sl)
        A, codes = pp["A"].double(), np.asarray(pp["codes"])
        b = (A.T @ mu / mu.dot(mu)).numpy()
        cs = torch.nn.functional.cosine_similarity(A.T, mu[None], dim=1).numpy()
        out["per_concept"][concept] = {name: {"beta_median": float(np.median(b[codes == c])),
                                              "cos_median": float(np.median(cs[codes == c])),
                                              "norm_median": float(np.median(A.norm(dim=0).numpy()[codes == c])),
                                              "n": int((codes == c).sum())}
                                       for name, c in (("concept_only", 1), ("neutral_only", 0), ("both", 2))}
    E = load_input_embeddings().double()
    idx = torch.randperm(E.shape[0], generator=torch.Generator().manual_seed(2))[:20000]
    b = (E[idx] @ mu / mu.dot(mu)).numpy()
    out["vocab_random"] = {"beta_median": float(np.median(b)), "beta_p10": float(np.percentile(b, 10)),
                           "beta_p90": float(np.percentile(b, 90))}
    save_json(OUT / "token_mean_coordinate.json", out)
    print(json.dumps(out["vocab_random"]), flush=True)


if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("runs", "cloud", "check", "tokens"))
    a = ap.parse_args()
    {"runs": runs, "cloud": cloud, "check": check, "tokens": tokens}[a.what]()
