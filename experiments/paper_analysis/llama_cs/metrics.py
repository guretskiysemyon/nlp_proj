"""Judge-free metrics of one sparse-MF fit (Y: k x n on V*), identical for Llama and Gemma.

    rho_i      mean_{t in C}|Y_it| / (mean_{t in N}|Y_it| + 1e-8)       (ember.utils.compute_embedding_stats, ratio_abs;
               C = concept-only tokens, N = neutral-only, "both" excluded)
    lo_i       lexicon hits: # of the concept's 20 words (j3_data/concept_words_v1.json) among the first 20 distinct
               normalized source words of feature i's tokens in |y| order (experiments/varwidth/layer_eval.py: fw20 + j3_count)
    concept feature (cf): argmax lo over ALL k features (ties: higher rho); "found" if lo >= 5 (EJ@5)
    cf_rank    1-based rank of cf by rho among the k features;  cf_rho, cf_width, cf_purity (concept-only share of its support)
    n_tau      # features with rho >= 2 (tau);  n_ej5 = # features with lo >= 5
    auc        concept-only (1) vs neutral-only (0), score = max |Y| over rho >= 2 features (experiments/whitening/eta_sweep.py)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

TAU, LO_T = 2.0, 5


@lru_cache(maxsize=64)
def _word_ctx(model: str, concept: str):
    from ember.judge_words import token_word_counts
    from experiments.paper_analysis.llama_cs.common import lexicon, tokenizer, vstar
    tok = tokenizer(model)
    sents = vstar(model, concept)["all_sents"] if model == "llama" else None
    if sents is None:
        from experiments.varwidth.prep import build_dataset
        sents = [s for s, _ in build_dataset(concept).data]
    return token_word_counts(tok, sents), lexicon(concept), tok


def rho(Y: torch.Tensor, codes: np.ndarray) -> np.ndarray:
    A = Y.abs().double()
    c = torch.as_tensor(codes == 1)
    nn = torch.as_tensor(codes == 0)
    mc = A[:, c].mean(1)
    mn = A[:, nn].mean(1)
    return (mc / (mn + 1e-8)).numpy()


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos, neg = labels == 1, labels == 0
    r = rankdata(scores)
    n1, n0 = pos.sum(), neg.sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def fit_metrics(Y: torch.Tensor, vstar_ids, codes, model: str, concept: str) -> dict:
    from experiments.varwidth import overlap_judge as oj
    from experiments.varwidth.j3_eval import j3_count
    Y = Y.detach().float().cpu()
    codes = np.asarray(codes)
    counts, cw, tok = _word_ctx(model, concept)
    k, n = Y.shape
    r = rho(Y, codes)
    order_rho = np.argsort(-r, kind="stable")
    rank_of = np.empty(k, dtype=int)
    rank_of[order_rho] = np.arange(1, k + 1)
    lo, widths, purity = np.zeros(k, dtype=int), np.zeros(k, dtype=int), np.zeros(k)
    mags = Y.abs()
    for f in range(k):
        nz = torch.nonzero(mags[f] > 1e-12).flatten()
        order = nz[torch.sort(mags[f][nz], descending=True, stable=True).indices].tolist()
        widths[f] = len(order)
        purity[f] = float((codes[order] == 1).mean()) if order else 0.0
        fw = oj.feature_words([vstar_ids[j] for j in order], counts, n=20, tok_str=lambda t: tok.decode([t]))
        lo[f] = j3_count(cw, fw) if cw else 0
    best = int(max(range(k), key=lambda f: (lo[f], r[f])))
    tau_feats = np.nonzero(r >= TAU)[0]
    lab = codes[np.isin(codes, [0, 1])]
    if len(tau_feats):
        score = mags[tau_feats].max(0).values.numpy()[np.isin(codes, [0, 1])]
    else:
        score = np.zeros(len(lab))
    return {
        "k": k, "n": n, "n_tau": int((r >= TAU).sum()), "frac_tau": float((r >= TAU).mean()),
        "n_ej5": int((lo >= LO_T).sum()), "n_ej5_top20": int(sum(lo[f] >= LO_T for f in order_rho[:20])),
        "cf": best, "cf_lo": int(lo[best]), "cf_found": bool(lo[best] >= LO_T), "cf_rank": int(rank_of[best]),
        "cf_rho": float(r[best]), "cf_width": int(widths[best]), "cf_purity": float(purity[best]),
        "cf_in_top20": bool(rank_of[best] <= 20), "top1_is_cf": bool(rank_of[best] == 1),
        "rho_top1": float(r[order_rho[0]]), "rho_top2": float(r[order_rho[1]]),
        "purity_tau_mean": float(purity[tau_feats].mean()) if len(tau_feats) else float("nan"),
        "auc": auc_score(score, lab),
        "lo_all": lo.tolist(), "rho_all": r.tolist(),
    }
