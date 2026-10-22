"""Matrix-level analysis per concept: PCA spectra of raw / whiten-only / CS matrices, selection-free concept
directions (their energy share, PCA rank, best-m-subset value and purity), the cPCA gains g_j, and truncated-SVD
rank matching of the saved sparse-MF errors.

    python -m experiments.paper_analysis.quiet_concept.spectra [--concepts all] [--threads 4]

Outputs (results/paper_analysis/quiet_concept/):
    spectra/<Slug>.json      per concept: spectrum summaries, direction metrics, SVD-matched ranks per seed
    spectra/<Slug>_sv.npz    eigenvalues of A A^T (descending) for raw / wo / cs (for later plots)
Definitions (A: d x n, columns a_t; G = A A^T, eigenvalues lam_1 >= ... >= lam_d; ||A||^2 = tr G):
    energy share of unit u     s(u) = u^T G u / tr G = sum_t (u^T a_t)^2 / ||A||_F^2
    PCA rank of u              1 + #{j : lam_j > u^T G u}  (where u's energy would sit in the PCA spectrum)
    m-subset value             V_m(u) = sum of the m largest (u^T a_t)^2, m = ceil(0.01 n) (= one EMBER feature's
                               support); its share V_m(u)/||A||^2, and purity = share of concept-only tokens in that
                               top-m set (what a feature along u would contain)
    concept directions         md: mean of concept-only columns - mean of neutral-only columns (equivariant: the CS
                               md is T^T times the raw md, mu_B cancels); ge: top generalized eigenvector of
                               (M_C, M_N + alpha tr(M_N)/d I), M = uncentred second moments of the columns, alpha 0.1
    SVD-matched rank           smallest r with sqrt(1 - sum_{j<=r} lam_j / tr G) <= rel_err of the saved sparse fit
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from experiments.paper_analysis.quiet_concept.common import (OUT, SEEDS, concepts, dump, lexicon, load_fit,
                                                              matrices, whitener, word_counts)
from experiments.varwidth.common import slug

ALPHA = 0.1
SPACES = ("raw", "rawp", "ctr", "wo", "cs")
FIT_CONDS = {"raw": ["raw", "gs_raw"], "rawp": ["rawp"], "ctr": ["ctr"], "wo": ["wo"], "cs": ["cs", "gs"]}


def spectrum(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    G = A @ A.T
    lam, V = torch.linalg.eigh(G)
    return lam.flip(0).clamp_min(0), V.flip(1)


def svd_matched_rank(lam: torch.Tensor, rel_err: float) -> int:
    tot = float(lam.sum())
    resid = (1 - torch.cumsum(lam, 0) / tot).clamp_min(0).sqrt()       # error of rank r = 1..d
    ok = torch.nonzero(resid <= rel_err).flatten()
    return int(ok[0]) + 1 if len(ok) else len(lam)


def lexicon_tokens(concept: str, vstar_ids: list, codes: np.ndarray) -> np.ndarray:
    """Columns of concept-only tokens whose most frequent source word (normalized) is in the concept's lexicon."""
    from experiments.varwidth import overlap_judge as oj
    from experiments.varwidth.prep import get_tokenizer
    tok = get_tokenizer()
    counts = word_counts(concept)
    lex = set(lexicon(concept))
    keep = []
    for j, t in enumerate(vstar_ids):
        c = counts.get(int(t))
        if codes[j] != 1 or not c:
            continue
        merged = {}
        for (_, w), k in c.items():
            merged[w] = merged.get(w, 0) + k
        w = oj.normalize(oj._piece(max(merged, key=merged.get), tok.decode([int(t)])))
        if w in lex:
            keep.append(j)
    return np.array(keep, dtype=int)


def direction_metrics(u: torch.Tensor, A: torch.Tensor, lam: torch.Tensor, codes: np.ndarray, m: int,
                      V: torch.Tensor, lex_idx: np.ndarray | None = None, feat_e: dict | None = None) -> dict:
    u = u / u.norm()
    p = (u @ A)                                                         # projections (n,)
    e = p.pow(2)
    tot = float(lam.sum())
    eu = float(e.sum())
    top = torch.topk(e, m).indices.numpy()
    vm = float(e[top].sum())
    extra = {}
    if lex_idx is not None:
        extra["Vm_purity_lex"] = float(np.isin(top, lex_idx).mean())
    for cond, fe in (feat_e or {}).items():                             # V_m(u) vs the fit's feature energies
        extra[f"Vm_pct_{cond}"] = float(np.mean([(f < vm).mean() * 100 for f in fe]))
    return {"share": eu / tot, "pca_rank": int((lam > eu).sum()) + 1, "x_avg_dir": eu / (tot / A.shape[0]),
            "Vm_share": vm / tot, "Vm_purity_C": float((codes[top] == 1).mean()),
            "Vm_purity_N": float((codes[top] == 0).mean()), **extra,
            "cos2_pc1": float((V[:, 0] @ u) ** 2), "cos2_top10": float(((V[:, :10].T @ u) ** 2).sum()),
            "C_share_of_dir_energy": float(e[torch.as_tensor(codes == 1)].sum()) / eu}


def seeded_rank1(A: torch.Tensor, L: np.ndarray, m: int, codes: np.ndarray, lam_tot: float, fe: dict,
                 max_iter: int = 100) -> dict:
    """Rank-1 hard-thresholded ALS (one EMBER feature, alone) started on the lexicon tokens L:
    u <- top left singular vector of A_S;  S <- the m tokens with the largest |u^T a_t|;  until S repeats.
    Reports the fixed point: purity, lexicon share, overlap with L, value sigma_1(A_S)^2 (share of ||A||^2 and
    percentile among the saved fits' feature energies ||z_i||^2)."""
    S = np.sort(L)
    seen = []
    for it in range(max_iter):
        U, sv, _ = torch.linalg.svd(A[:, torch.as_tensor(S)], full_matrices=False)
        u = U[:, 0]
        S_new = np.sort(torch.topk((u @ A).abs(), m).indices.numpy())
        if any(np.array_equal(S_new, x) for x in seen):
            S = S_new
            break
        seen.append(S_new)
        S = S_new
    v = float(torch.linalg.svdvals(A[:, torch.as_tensor(S)])[0] ** 2)
    out = {"iters": it + 1, "purity_C": float((codes[S] == 1).mean()), "purity_N": float((codes[S] == 0).mean()),
           "lex_share": float(np.isin(S, L).mean()), "frac_L_kept": float(np.isin(L, S).mean()),
           "value_share": v / lam_tot}
    for cond, zs in fe.items():
        out[f"value_pct_{cond}"] = float(np.mean([(z < v).mean() * 100 for z in zs]))
    return out


def concept_directions(A: torch.Tensor, codes: np.ndarray) -> dict:
    C, N = A[:, torch.as_tensor(codes == 1)], A[:, torch.as_tensor(codes == 0)]
    md = C.mean(1) - N.mean(1)
    MC, MN = C @ C.T / C.shape[1], N @ N.T / N.shape[1]
    d = A.shape[0]
    L = torch.linalg.cholesky(MN + ALPHA * float(torch.trace(MN)) / d * torch.eye(d, dtype=A.dtype))
    Li = torch.linalg.inv(L)
    w, Vg = torch.linalg.eigh(Li @ MC @ Li.T)
    ge = Li.T @ Vg[:, -1]
    if float(ge @ md) < 0:
        ge = -ge
    return {"md": md / md.norm(), "ge": ge / ge.norm(), "ge_ratio": float(w[-1])}


def spectrum_summary(lam: torch.Tensor) -> dict:
    s = lam / lam.sum()
    cs = torch.cumsum(s, 0)
    return {"top1": float(s[0]), "top5": float(s[:5].sum()), "top10": float(s[:10].sum()),
            "top100": float(s[:100].sum()), "pr": float(1 / (s ** 2).sum()),
            "n50": int((cs < 0.5).sum()) + 1, "n90": int((cs < 0.9).sum()) + 1}


def token_energy(A: torch.Tensor, codes: np.ndarray) -> dict:
    e = A.pow(2).sum(0).numpy()
    tot = e.sum()
    c, nn, b = codes == 1, codes == 0, codes == 2
    return {"C_share": float(e[c].sum() / tot), "N_share": float(e[nn].sum() / tot), "B_share": float(e[b].sum() / tot),
            "C_frac_tokens": float(c.mean()), "N_frac_tokens": float(nn.mean()),
            "mean_C": float(e[c].mean()), "mean_N": float(e[nn].mean()), "median_C": float(np.median(e[c])),
            "median_N": float(np.median(e[nn])), "C_over_N_mean": float(e[c].mean() / e[nn].mean()),
            "top1pct_token_share": float(np.sort(e)[::-1][:math.ceil(0.01 * len(e))].sum() / tot)}


def cpca(M: dict, Acs: torch.Tensor, Awo: torch.Tensor, lam_cs, V_cs, codes) -> dict:
    """g_j of S_eta from the CS transform (S^{1/2} = W^{-1} T), checks E_X(p_j) = g_j E_U(p_j), and how p_1 relates
    to the concept tokens."""
    wh = whitener()
    Sp = wh["W_inv"] @ M["T"]
    Sp = (Sp + Sp.T) / 2
    h, P = torch.linalg.eigh(Sp)
    g = (h.flip(0) ** 2)                                                 # gamma = 1: S^{1/2} eigenvalues squared
    P = P.flip(1)
    eta = M["eta"]
    EX = (P.T @ Acs).pow(2).sum(1)
    EU = (P.T @ Awo).pow(2).sum(1)
    rel = float(((EX - g * EU).abs() / EX.clamp_min(1e-30)).max())
    p1 = P[:, 0]
    c, nn = torch.as_tensor(codes == 1), torch.as_tensor(codes == 0)
    pu = p1 @ Awo
    # Rayleigh ratios in whitened space = raw-space generalized Rayleigh vs (Sigma_B + eps I) along W p_1
    out = {"g1": float(g[0]), "g2": float(g[1]), "g10": float(g[9]), "n_g_gt1": int((g > 1).sum()),
           "n_g_lt_2eta": int((g < 2 * eta).sum()), "lam1_page": float((g[0] - eta) / (1 - eta)),
           "check_EX_eq_gEU_maxrel": rel,
           "p1_whitened_energy_per_C_token": float(pu[c].pow(2).mean()),
           "p1_whitened_energy_per_N_token": float(pu[nn].pow(2).mean()),
           "neutral_whitened_mean_energy_per_dir": float(Awo[:, nn].pow(2).sum(0).mean() / Awo.shape[0]),
           "concept_whitened_mean_energy_per_dir": float(Awo[:, c].pow(2).sum(0).mean() / Awo.shape[0]),
           "p1_share_cs": float(EX[0] / lam_cs.sum()), "p1_pca_rank_cs": int((lam_cs > EX[0]).sum()) + 1,
           "p1_cos2_pc1_cs": float((V_cs[:, 0] @ p1) ** 2),
           "p1_share_wo": float(EU[0] / (Awo.pow(2).sum())),
           "g_top20": g[:20].tolist()}
    return out, P, g


def run(concept: str) -> dict:
    t0 = time.time()
    M = matrices(concept)
    codes = M["codes"]
    n = len(codes)
    m = math.ceil(0.01 * n)
    res = {"concept": concept, "n": n, "m": m, "n_C": int((codes == 1).sum()), "n_N": int((codes == 0).sum()),
           "n_B": int((codes == 2).sum())}
    lex_idx = lexicon_tokens(concept, M["vstar_ids"], codes)
    res["n_lex_tokens"] = int(len(lex_idx))
    lams, eig, feat = {}, {}, {}
    for sp in SPACES:
        A = M[sp]
        lam, V = spectrum(A)
        lams[sp], eig[sp] = lam, V
        r = {"spectrum": spectrum_summary(lam), "tokens": token_energy(A, codes), "svd_rank": {}, "rel_err": {}}
        fe = {}
        for cond in FIT_CONDS[sp]:
            errs, zs = {}, []
            for s in SEEDS:
                f = load_fit(concept, cond, s)
                if f is not None:
                    errs[s] = float(f["rel_err"])
                    zs.append(f["Z"].double().pow(2).sum(0).numpy())
            r["rel_err"][cond] = errs
            r["svd_rank"][cond] = {s: svd_matched_rank(lam, e) for s, e in errs.items()}
            if zs:
                fe[cond] = zs
                r[f"feat_energy_{cond}"] = {"median_share": float(np.median(np.concatenate(zs)) / float(lam.sum())),
                                            "sum_share": float(np.mean([z.sum() for z in zs]) / float(lam.sum()))}
        feat[sp] = fe
        dirs = concept_directions(A, codes)
        if len(lex_idx) >= 2:
            AL = A[:, torch.as_tensor(lex_idx)]
            UL, SL, _ = torch.linalg.svd(AL, full_matrices=False)
            dirs["lex"] = UL[:, 0]
            lm = AL.mean(1) - A[:, torch.as_tensor(codes == 0)].mean(1)
            dirs["lexmd"] = lm / lm.norm()
            eL = AL.pow(2).sum(0)
            r["lex"] = {"tok_energy_rel": float(eL.mean() / A.pow(2).sum(0).mean()),
                        "coherence": float(SL[0] ** 2 / eL.sum()), "value_share": float(SL[0] ** 2 / lam.sum()),
                        "coherence_ctr_lex": float(torch.linalg.svdvals(AL - AL.mean(1, keepdim=True))[0] ** 2
                                                   / (AL - AL.mean(1, keepdim=True)).pow(2).sum())}
            r["seeded"] = seeded_rank1(A, lex_idx, m, codes, float(lam.sum()), fe)
        r["ge_ratio"] = dirs.pop("ge_ratio")
        r["dirs"] = {k: direction_metrics(u, A, lam, codes, m, V, lex_idx, fe) for k, u in dirs.items()}
        if sp in ("raw", "rawp", "ctr"):                       # raw coordinates: background variance along each direction
            wh = whitener()
            for k, u in dirs.items():
                u = u / u.norm()
                bv = float(((wh["Q"].T @ u) ** 2 * wh["lam"]).sum())
                r["dirs"][k]["bgvar_rel"] = bv / float(wh["lam"].mean())
                r["dirs"][k]["bgvar_rank"] = int((wh["lam"] > bv).sum()) + 1
        r["cos_md_ge"] = float(dirs["md"] @ dirs["ge"])
        res[sp] = r
    res["cpca"], P, g = cpca(M, M["cs"], M["wo"], lams["cs"], eig["cs"], codes)
    # p_1 (the top sharpening direction) measured in each space: whitened-space p_1, raw-space W p_1 direction
    wh = whitener()
    p1 = P[:, 0]
    raw_dir = wh["W"] @ p1                                              # raw direction with Rayleigh ratio lam_1
    res["p1_dirs"] = {"raw": direction_metrics(raw_dir, M["raw"], lams["raw"], codes, m, eig["raw"], lex_idx),
                      "wo": direction_metrics(p1, M["wo"], lams["wo"], codes, m, eig["wo"], lex_idx),
                      "cs": direction_metrics(p1, M["cs"], lams["cs"], codes, m, eig["cs"], lex_idx)}
    np.savez_compressed(OUT / "spectra" / f"{slug(concept)}_sv.npz", **{sp: lams[sp].numpy() for sp in SPACES},
                        g=g.numpy())
    res["elapsed"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--part", default="0/1", help="i/n: process every n-th concept starting at i")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    (OUT / "spectra").mkdir(parents=True, exist_ok=True)
    i, n = map(int, a.part.split("/"))
    for c in (a.only or concepts(a.concepts)[i::n]):
        p = OUT / "spectra" / f"{slug(c)}.json"
        if p.exists() and not a.force:
            continue
        r = run(c)
        dump(r, p)
        print(f"{c:28s} md rank raw/wo/cs {r['raw']['dirs']['md']['pca_rank']:5d} {r['wo']['dirs']['md']['pca_rank']:5d} "
              f"{r['cs']['dirs']['md']['pca_rank']:5d}  share {r['raw']['dirs']['md']['share']:.4f} "
              f"{r['wo']['dirs']['md']['share']:.4f} {r['cs']['dirs']['md']['share']:.4f}  svd-rank raw/cs "
              f"{r['raw']['svd_rank'].get('raw')} {r['cs']['svd_rank'].get('cs')}  ({r['elapsed']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
