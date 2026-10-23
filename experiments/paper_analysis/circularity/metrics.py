"""Judge-free metrics of one fit (Y: k x n sparse codes, Z: d x k, A: d x n the factorized matrix).

Concept feature f* = argmax over features of (lexicon hits among its top-20 source words, rho); found@5 = hits >= 5
(EJ@5), found@2 = hits >= 2. All metrics below are computed on f* whether or not it is "found".
    rho_i       mean_C |Y_it| / (mean_N |Y_it| + 1e-8)        (ember compute_embedding_stats ratio_abs)
    rho_rank    1 + #{i : rho_i > rho_f*}  (1 = top)
    log_dom     log(rho_f* / max_{i != f*} rho_i)
    purity_C    share of concept-only tokens in supp(f*);  purity_strict / _strict_any / _lexonly likewise
    lex_top20   hits / 20  (lexicon precision of the 20 source words the judge would see)
    lex_tok     share of supp(f*) tokens whose source word is a lexicon word
    auc_C       ROC AUC concept-only vs neutral-only, score = |dense code of f*|, dense = (Z^T Z + 1e-4 I)^-1 Z^T A
                (the fit's own dense Y-solve); auc_S: strict tokens vs neutral-only
    auc_tau     the eta-sweep AUC: concept-only vs neutral-only, score = max |Y| over features with rho >= 2
    out_<T>     held-out tokens: the same auc_C / auc_S restricted to tokens that are NOT token types of target T,
                and purity_strict of supp(f*) restricted to those tokens
"""
from __future__ import annotations

import math

import numpy as np
import torch

TAU, RIDGE, N_WORDS = 2.0, 1e-4, 20


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted 1/2."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def row_order(y: torch.Tensor) -> list[int]:
    mags = y.abs()
    nz = torch.nonzero(mags > 1e-12).flatten()
    return nz[torch.sort(mags[nz], descending=True, stable=True).indices].tolist()


def feature_words(order, words) -> list[str]:
    out, seen = [], set()
    for j in order:
        w = words[j]
        if w and w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) == N_WORDS:
                break
    return out


def compute(Y: torch.Tensor, Z: torch.Tensor | None, A: torch.Tensor | None, lab: dict, targets) -> dict:
    Y = Y.detach().float().cpu()
    codes = np.asarray(lab["codes"])
    is_c, is_n = codes == 1, codes == 0
    Ya = Y.abs().numpy()
    rho = Ya[:, is_c].mean(1) / (Ya[:, is_n].mean(1) + 1e-8)
    lexset = set(lab["lexicon"])
    words = lab["words"]
    orders = [row_order(Y[i]) for i in range(Y.shape[0])]
    fws = [feature_words(o, words) for o in orders]
    hits = np.array([sum(w in lexset for w in fw) for fw in fws])
    f = int(max(range(len(rho)), key=lambda i: (hits[i], rho[i])))
    supp = Ya[f] > 1e-12
    others = np.delete(rho, f)
    m = {"f_star": f, "hits": int(hits[f]), "found5": bool(hits[f] >= 5), "found2": bool(hits[f] >= 2),
         "rho_rank": int(1 + (rho > rho[f]).sum()), "rho": float(rho[f]), "log_rho": float(math.log(max(rho[f], 1e-12))),
         "log_dom": float(math.log(max(rho[f], 1e-12) / max(others.max(), 1e-12))), "width": int(supp.sum()),
         "n_tau": int((rho >= TAU).sum()), "lex_top20": float(hits[f] / N_WORDS),
         "lex_tok": float(lab["lex"][supp].mean()) if supp.any() else 0.0}
    for key, mask in (("C", is_c), ("strict", lab["strict"]), ("strict_any", lab["strict_any"]),
                      ("lexonly", lab["lexonly"])):
        m[f"purity_{key}"] = float(np.asarray(mask)[supp].mean()) if supp.any() else 0.0
    sel = rho >= TAU
    score_tau = Ya[sel].max(0) if sel.any() else np.zeros(Ya.shape[1])
    m["auc_tau"] = auc(score_tau[is_c], score_tau[is_n])
    strict = np.asarray(lab["strict"])
    if Z is not None and A is not None:
        Zc, Ac = Z.detach().double().cpu(), A.detach().double().cpu()
        G = Zc.T @ Zc + RIDGE * torch.eye(Zc.shape[1], dtype=torch.float64)
        dense = torch.linalg.solve(G, Zc.T @ Ac)[f].abs().numpy()
        m["auc_C"] = auc(dense[is_c], dense[is_n])
        m["auc_S"] = auc(dense[strict], dense[is_n])
        for T in targets:
            out = ~np.asarray(lab[f"in_{T}"])
            m[f"out_{T}_auc_C"] = auc(dense[is_c & out], dense[is_n & out])
            m[f"out_{T}_auc_S"] = auc(dense[strict & out], dense[is_n & out])
            so = supp & out
            m[f"out_{T}_purity_strict"] = float(strict[so].mean()) if so.any() else float("nan")
            m[f"out_{T}_n_supp"] = int(so.sum())
    m["rho_all"] = [round(float(x), 4) for x in rho]
    m["hits_all"] = [int(x) for x in hits]
    m["top20_tokens"] = [lab["tokens"][j] for j in orders[f][:20]]
    m["top20_words"] = fws[f]
    m["top20_codes"] = [int(codes[j]) for j in orders[f][:20]]
    m["top20_strict"] = [bool(strict[j]) for j in orders[f][:20]]
    return m
