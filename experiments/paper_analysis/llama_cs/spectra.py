#!/usr/bin/env python
"""Spectrum of the whitened page second moment M = U_C^T U_C / m and of S_eta = (1 - eta) M + eta I, for the 18 EMBER
concepts on Gemma-2-2B-it (d = 2304) and Llama-3.1-8B-Instruct (d = 4096). No fits.

Per (model, concept):
  m, m/d, numerical rank of M, tr(M)/d, lambda_max
  per eta: share of the d directions at the floor (g_j < 1.1 eta), condition number g_max / g_min of S_eta
  noise null: M_0 from m random background (English Wikipedia) types NOT on the page -> same statistics;
              n_signal = # eigenvalues of M above the null's largest eigenvalue (directions resolvable from noise)
  Ledoit-Wolf shrinkage intensity toward I (the whitened background):
              eta_LW = beta^2 / ||M - I||_F^2, beta^2 = (sum_j ||u_j||^4 / m - ||M||_F^2) / m   (clipped to [0, 1])
  split-half reproducibility: page types split in two random halves, mean principal-angle sine between the
              top-r eigenvector subspaces of the two halves' M (r = 10, 50); same for the null
  V* diagnostics (tokens of the 300 + 300 sentences):
      self_share   (gamma = 1) share of a page token's sharpened energy u^T S u that comes from its own term
                   (1 - eta) ||u||^4 / m  (V* concept-only tokens that are on the page; mean)
      boost        mean ||x_t||^2 / ||u_t||^2 of V* tokens on the page over the same for V* tokens off the page
      span_share   share of an off-page V* token's whitened energy inside span(U_C) (1 when m >= d)
      E_C          concept-only share of ||X||_F^2 vs their count share
Output: results/paper_analysis/llama_cs/spectra/spectra.json, spectra_<model>_<slug>.pt (lam, null lam) for plots.

    python -m experiments.paper_analysis.llama_cs.spectra [--device cuda:0]
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from experiments.paper_analysis.llama_cs.common import (CONCEPTS18, DATA, MODELS, OUT, ROOT, embeddings, page_target,
                                                        read_json, slug, vstar, whitener, write_json)

ETAS = [0.05, 0.1, 0.2, 0.5]
GAMMAS = [0.5, 1.0]
SDIR = OUT / "spectra"


def gemma_vstar(concept):
    p = torch.load(ROOT / "results" / "varwidth" / "prep" / f"{slug(concept)}_plain_S300.pt", map_location="cpu", weights_only=False)
    return {"vstar_ids": list(p["vstar_ids"]), "codes": np.asarray(p["codes"])}


def bg_types(model):
    return read_json(DATA / f"bg_wiki_{model}_types.json")


def whiten(model, ids, dev):
    bw = whitener(model)
    E = embeddings(model)
    X = E.index_select(0, torch.as_tensor(list(ids), dtype=torch.long)).to(dev).double()
    return (X - bw.mu.to(dev)) @ bw.W.to(dev)


def mom(U):
    return U.T @ U / U.shape[0]


def top_sines(Ma, Mb, r):
    _, Pa = torch.linalg.eigh(Ma)
    _, Pb = torch.linalg.eigh(Mb)
    Qa, Qb = Pa[:, -r:], Pb[:, -r:]
    cos = torch.linalg.svdvals(Qa.T @ Qb).clamp(0, 1)
    return float((1 - cos ** 2).clamp_min(0).sqrt().mean())


def split_half(U, seed=0):
    m = U.shape[0]
    perm = torch.as_tensor(random.Random(seed).sample(range(m), m), device=U.device)
    h = m // 2
    Ma, Mb = mom(U[perm[:h]]), mom(U[perm[h:2 * h]])
    return {f"sine_top{r}": top_sines(Ma, Mb, r) for r in (10, 50)}


def lw_eta(U, M):
    m, d = U.shape
    n4 = (U.pow(2).sum(1) ** 2).mean()
    M2 = (M * M).sum()
    beta2 = (n4 - M2) / m
    delta2 = ((M - torch.eye(d, dtype=M.dtype, device=M.device)) ** 2).sum()
    return float(min(1.0, max(0.0, float(beta2 / delta2))))


def analyze(model, concept, dev):
    d = MODELS[model]["d"]
    tgt = page_target(model, concept)
    m = len(tgt)
    U = whiten(model, tgt, dev)
    M = mom(U)
    lam, P = torch.linalg.eigh(M)
    lam, P = lam.flip(0).clamp_min(0), P.flip(1)
    tol = float(lam[0]) * max(m, d) * torch.finfo(torch.float64).eps
    rank = int((lam > tol).sum())
    # noise null: m random background types off the page
    tset = set(tgt)
    pool = [t for t in bg_types(model) if t not in tset]
    nul_ids = random.Random(1).sample(pool, m)
    U0 = whiten(model, nul_ids, dev)
    M0 = mom(U0)
    lam0 = torch.linalg.eigvalsh(M0).flip(0).clamp_min(0)
    out = {"model": model, "concept": concept, "d": d, "m": m, "m_over_d": m / d, "rank": rank,
           "trace_over_d": float(lam.sum() / d), "lam_max": float(lam[0]), "null_trace_over_d": float(lam0.sum() / d),
           "null_lam_max": float(lam0[0]), "n_signal": int((lam > lam0[0]).sum()),
           "eta_LW": lw_eta(U, M), "eta_LW_null": lw_eta(U0, M0),
           "split_half": split_half(U), "split_half_null": split_half(U0)}
    per_eta = {}
    for eta in ETAS:
        g = (1 - eta) * lam + eta
        g0 = (1 - eta) * lam0 + eta
        per_eta[f"{eta:g}"] = {"frac_floor": float((g < 1.1 * eta).float().mean()),
                               "cond": float(g[0] / g[-1]), "null_frac_floor": float((g0 < 1.1 * eta).float().mean()),
                               "null_cond": float(g0[0] / g0[-1])}
    out["per_eta"] = per_eta
    # V* diagnostics
    v = vstar(model, concept) if model == "llama" else gemma_vstar(concept)
    ids, codes = v["vstar_ids"], np.asarray(v["codes"])
    on = np.array([t in tset for t in ids])
    Uv = whiten(model, ids, dev)
    c2 = (Uv @ P) ** 2                                         # (n, d) squared coordinates in the eigenbasis
    nu2 = c2.sum(1)
    uMu = (c2 * lam).sum(1)
    out["n_vstar"] = len(ids)
    out["frac_C_on_page"] = float(on[codes == 1].mean())
    out["frac_N_on_page"] = float(on[codes == 0].mean())
    sel = torch.as_tensor(on & (codes == 1), device=dev)
    diag = {}
    for eta in ETAS:
        g = (1 - eta) * lam + eta
        selfterm = (1 - eta) * nu2 ** 2 / m
        energy1 = (1 - eta) * uMu + eta * nu2
        row = {"self_share_g1": float((selfterm[sel] / energy1[sel]).mean())}
        for gamma in GAMMAS:
            ex = (c2 * g.pow(gamma)).sum(1)
            ratio = (ex / nu2).cpu().numpy()
            row[f"boost_g{gamma:g}"] = float(ratio[on].mean() / ratio[~on].mean())
            row[f"E_C_g{gamma:g}"] = float(ex[torch.as_tensor(codes == 1, device=dev)].sum() / ex.sum())
        diag[f"{eta:g}"] = row
    out["vstar_diag"] = diag
    out["count_share_C"] = float((codes == 1).mean())
    e_w = nu2.cpu().numpy()
    out["E_C_white"] = float(e_w[codes == 1].sum() / e_w.sum())
    if rank < d:
        inspan = c2[:, :rank].sum(1) / nu2
        out["span_share_offpage"] = float(inspan[torch.as_tensor(~on, device=dev)].mean())
        out["span_share_offpage_N"] = float(inspan[torch.as_tensor(~on & (codes == 0), device=dev)].mean())
    else:
        out["span_share_offpage"] = out["span_share_offpage_N"] = 1.0
    SDIR.mkdir(parents=True, exist_ok=True)
    torch.save({"lam": lam.cpu(), "lam_null": lam0.cpu(), "m": m, "d": d}, SDIR / f"spectra_{model}_{slug(concept)}.pt")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["gemma", "llama"])
    ap.add_argument("--concepts", nargs="+", default=CONCEPTS18)
    a = ap.parse_args()
    torch.set_num_threads(4)
    path = SDIR / "spectra.json"
    res = read_json(path) if path.exists() else {}
    for model in a.models:
        for c in a.concepts:
            key = f"{model}|{c}"
            if key in res:
                continue
            t0 = time.time()
            res[key] = analyze(model, c, a.device)
            r = res[key]
            print(f"[{model} {c}] m/d={r['m_over_d']:.2f} rank={r['rank']} n_signal={r['n_signal']} "
                  f"floor@.05={r['per_eta']['0.05']['frac_floor']:.2f} cond={r['per_eta']['0.05']['cond']:.0f} "
                  f"etaLW={r['eta_LW']:.3f} self={r['vstar_diag']['0.05']['self_share_g1']:.2f} "
                  f"boost={r['vstar_diag']['0.05']['boost_g1']:.2f} ({time.time() - t0:.0f}s)", flush=True)
            write_json(path, res)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
