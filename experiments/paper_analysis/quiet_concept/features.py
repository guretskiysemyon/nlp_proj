"""Fit-level analysis: the judge-free concept feature of every saved fit and where it ranks.

    python -m experiments.paper_analysis.quiet_concept.features [--concepts all] [--threads 4]

For every (concept, cond, seed) with a fit (cond: raw = EMBER L0, wo = whiten only (this WP), cs = CS + RS-1% L1,
gs = CS + GS L12, gs_raw = plain GS):
    rho_i (EMBER ratio_abs), rho rank, lexicon hits h_i = # of the feature's first 20 distinct source words that are in
    the concept's 20-word lexicon (concept_words_v1.json; the EJ@5 rule accepts h_i >= 5);
    best concept feature b = argmax h (ties -> best rho rank); its rho, rho rank, width, C/N/B counts, the share of
    ||z_b||^2 in sum_i ||z_i||^2 and its percentile, the energy share and PCA rank of z_b in its own matrix, the mean
    squared norm of its support tokens relative to the matrix's mean token, and the judge's F_C (summary.json; no new
    Gemini calls) with the best rho rank among F_C.
Output: results/paper_analysis/quiet_concept/features/<Slug>.json (list of rows) and features_all.csv.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.quiet_concept.common import (OUT, SEEDS, best_concept_feature, concepts, dump,
                                                              feature_words_all, lexicon, lexicon_hits, load_fit,
                                                              load_summary, matrices, rho, rho_rank, whitener,
                                                              word_counts)
from experiments.varwidth.common import slug

CONDS = {"raw": "raw", "gs_raw": "raw", "rawp": "rawp", "ctr": "ctr", "wo": "wo", "cs": "cs", "gs": "cs"}


def analyse(concept: str) -> list[dict]:
    M = matrices(concept)
    codes = M["codes"]
    counts = word_counts(concept)
    lex = lexicon(concept)
    spec = {}
    for sp in ("raw", "rawp", "ctr", "wo", "cs"):
        A = M[sp].float()
        lam = torch.linalg.eigvalsh(A.double() @ A.double().T).flip(0)
        tok_e = A.pow(2).sum(0).double().numpy()
        spec[sp] = (A, lam, float(lam.sum()), tok_e)
    wh = whitener()
    Sig = (wh["Q"] * wh["lam"]) @ wh["Q"].T                                  # background covariance Sigma_B
    lam_mean = float(wh["lam"].mean())
    lam_B = wh["lam"].numpy()
    back = {"raw": None, "rawp": None, "ctr": None, "wo": wh["W_inv"], "cs": M["T_inv"].T}  # column atoms back to raw space
    rows = []
    for cond, sp in CONDS.items():
        A, lam, tot, tok_e = spec[sp]
        for s in SEEDS:
            f = load_fit(concept, cond, s)
            if f is None:
                continue
            Y, Z = f["Y"].float(), f["Z"].float()
            r = rho(Y, codes)
            rk = rho_rank(r)
            fw = feature_words_all(Y, M["vstar_ids"], counts, n=20)
            h = lexicon_hits(fw, lex)
            b = best_concept_feature(h, rk)
            zn2 = Z.pow(2).sum(0).double().numpy()
            Zn = Z / Z.norm(dim=0, keepdim=True)
            Edir = (Zn.T @ A).double().pow(2).sum(1).numpy()                             # energy along each z_i
            eb = float(Edir[b])
            supp = np.flatnonzero(Y[b].abs().numpy() > 1e-12)
            summ = load_summary(concept, cond, s)
            FC = summ["F_C"] if summ else None
            Zr = Z.double() if back[sp] is None else back[sp] @ Z.double()
            Zr = Zr / Zr.norm(dim=0, keepdim=True)
            bgv = ((Sig @ Zr) * Zr).sum(0).numpy()                                      # v^T Sigma_B v per atom
            Ym = (Y.abs() > 1e-12)
            supp_e = (Ym.double() * torch.as_tensor(tok_e)[None, :]).sum(1).numpy()     # tr K_SS per feature
            coh = zn2 / supp_e                                                           # ~ sigma_1^2 / tr K_SS
            others = np.arange(len(zn2)) != b
            row = {"concept": concept, "cond": cond, "seed": s, "rel_err": float(f["rel_err"]),
                   "feat_coh_median": float(np.median(coh[others])), "best_coh": float(coh[b]),
                   "coh_pct": float((coh[others] < coh[b]).mean() * 100),
                   "feat_supp_energy_rel_median": float(np.median(supp_e[others] / Ym.sum(1).numpy()[others])
                                                        / tok_e.mean()),
                   "feat_rho_median": float(np.median(r)),
                   "best": b, "hits": int(h[b]), "rho": float(r[b]), "rho_rank": int(rk[b]),
                   "rho_max": float(r.max()), "n_ej5": int((h >= 5).sum()),
                   "n_ej5_top20rho": int(((h >= 5) & (rk <= 20)).sum()),
                   "best_rank_among_ej5": int(rk[h >= 5].min()) if (h >= 5).any() else None,
                   "width": int(len(supp)), "supp_C": int((codes[supp] == 1).sum()),
                   "supp_N": int((codes[supp] == 0).sum()), "supp_B": int((codes[supp] == 2).sum()),
                   "z2_share": float(zn2[b] / zn2.sum()), "z2_pct": float((zn2 < zn2[b]).mean() * 100),
                   "z2_rank": int((zn2 > zn2[b]).sum()) + 1,
                   "dir_share": eb / tot, "dir_pca_rank": int((lam > eb).sum()) + 1,
                   "dir_pct": float((Edir < eb).mean() * 100),
                   "bgvar_rel": float(bgv[b] / lam_mean), "bgvar_pct_feat": float((bgv < bgv[b]).mean() * 100),
                   "bgvar_rank_in_bg_spectrum": int((lam_B > bgv[b]).sum()) + 1,
                   "bgvar_rel_feat_median": float(np.median(bgv) / lam_mean),
                   "supp_tok_energy_rel": float(tok_e[supp].mean() / tok_e.mean()),
                   "words": fw[b][:20],
                   "F_C": FC, "FC_best_rank": (int(min(rk[i] for i in FC)) if FC else None),
                   "b_in_FC": (b in FC) if FC is not None else None,
                   "rho_top20": np.sort(r)[::-1][:20].round(3).tolist(),
                   "hits_by_rank": h[np.argsort(-r, kind="stable")][:30].tolist()}
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--part", default="0/1", help="i/n: process every n-th concept starting at i")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    (OUT / "features").mkdir(parents=True, exist_ok=True)
    i, n = map(int, a.part.split("/"))
    for c in (a.only or concepts(a.concepts)[i::n]):
        p = OUT / "features" / f"{slug(c)}.json"
        if p.exists() and not a.force:
            continue
        rows = analyse(c)
        dump(rows, p)
        df = pd.DataFrame(rows)
        g = df.groupby("cond")[["hits", "rho_rank", "rho", "z2_pct", "dir_pca_rank"]].median()
        print(c, "\n", g.round(2).to_string(), flush=True)
    import json
    allrows = []
    for c in concepts("all"):
        p = OUT / "features" / f"{slug(c)}.json"
        if p.exists():
            allrows += json.loads(p.read_text())
    df = pd.DataFrame(allrows).drop(columns=["words", "rho_top20", "hits_by_rank", "F_C"])
    df.to_csv(OUT / "features_all.csv", index=False)


if __name__ == "__main__":
    main()
