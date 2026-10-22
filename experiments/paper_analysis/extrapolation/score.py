"""Extrapolation rules: score and admit the new tokens of a concept on a saved dictionary's concept features.

    from experiments.paper_analysis.extrapolation.score import run_scores
    res = run_scores("Harry Potter", "CINDER", 42)      # dict of per-token scores and admission masks

Setting (one run = concept x dictionary x seed). A = the fit's matrix on V* (raw E^T for EMBER, the CS matrix X^T
for CS / CINDER; prep cache), A_new = the new tokens' rows in the same space (X_new = (E_new - mu_B) T, the run's
own transform, nothing re-estimated). Z (d x k), Y (k x n) the saved fit, F_C its concept features under the
dictionary's judge (common.DICTS). Every concept feature f is oriented so that its saved support has sum Y >= 0.

Scores of a token t on f (a token's score is its max over f in F_C):
    proj    <a_t, z_f> / ||z_f||                  naive projection on the concept feature (motivation doc)
    cos     <a_t, z_f> / (||a_t|| ||z_f||)        the same, token-normalized
    coef    y_ft of the joint dense solve Y_d = (Z^T Z + lam I)^-1 Z^T A_ext (lam = the fit's ridge): what the
            fixed-Z sparse re-solve ranks by (EMBER's rule ranks |y|; we rank signed, oriented y)
    share   y_ft |y_ft| / ||y_{.t}||^2           fraction of the token's dense code energy on f, signed
            ("channelling": a token whose code lives on another feature, e.g. magic, gets a small share)

Admission rules (which new tokens join f's support; new token admitted if admitted to some f in F_C):
    frac    the fit's own sparsity rule re-applied over V*_ext: RS-1% (top ceil(0.01 n_ext) by |y_d| per row) for the
            RS fits (EMBER, CS); GS over V*_ext (budget ceil(0.01 n_ext) k, floor/cap of the fit) for CINDER
    width   absolute width held: f keeps its saved width w_f over V*_ext (top w_f by |y_d|)
    cutoff  |y_ft| >= the smallest |y_d| among f's saved support tokens on V* (the in-sample cut-off, sign-free)
    argmax  y_ft > 0 and f = argmax_i |y_it| (the token's dominant feature on the joint solve)
    share50 y_ft > 0 and y_ft^2 >= 0.5 ||y_{.t}||^2
Every rule also reports which saved support tokens on V* it keeps (support retention).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import torch

from experiments.paper_analysis.extrapolation.common import DICTS, TOKDIR, concept_features, load_fit
from experiments.varwidth.common import budgets, pool_floor_cap, slug, VWConfig

RULES = ("frac", "width", "cutoff", "argmax", "share50")
SCORES = ("proj", "cos", "coef", "share")


@lru_cache(maxsize=4)
def token_data(concept: str) -> dict:
    d = json.loads((TOKDIR / f"{slug(concept)}.json").read_text())
    e = torch.load(TOKDIR / f"{slug(concept)}_Enew.pt", weights_only=False)
    assert e["ids"].tolist() == d["new"]
    d["E_new"] = e["E"]
    return d


@lru_cache(maxsize=4)
def prep_of(concept: str, matrix: str) -> dict:
    from experiments.varwidth.prep import prepare
    return prepare(concept, matrix, verbose=False)


@lru_cache(maxsize=4)
def new_matrix(concept: str, matrix: str) -> torch.Tensor:
    """(d, m) rows of the new tokens in the fit's space."""
    td = token_data(concept)
    E_new = td["E_new"].double()
    if matrix == "plain":
        return E_new.T.float().contiguous()
    p = prep_of(concept, "whitened")
    return ((E_new - p["mu_B"].double()) @ p["T_matrix"].double()).T.float().contiguous()


def run_scores(concept: str, dictionary: str, seed: int) -> dict | None:
    cfg, judge = DICTS[dictionary]
    F = concept_features(concept, cfg, seed, judge)
    fit = load_fit(concept, cfg, seed)
    matrix = "plain" if cfg == "L0" else "whitened"
    p = prep_of(concept, matrix)
    A = p["A"].double()                                   # (d, n)
    A_new = new_matrix(concept, matrix).double()          # (d, m)
    Z, Y = fit["Z"].double(), fit["Y"].double()
    k, n = Y.shape
    m = A_new.shape[1]
    conf = fit["config"]
    lam = conf["ridge"]
    A_ext = torch.cat([A, A_new], 1)
    G = Z.T @ Z + lam * torch.eye(k, dtype=torch.float64)
    Yd = torch.linalg.solve(G, Z.T @ A_ext)               # (k, n + m)
    n_ext = n + m
    out = {"concept": concept, "dictionary": dictionary, "seed": seed, "F_C": F, "n": n, "m": m,
           "widths": [int((Y[f] != 0).sum()) for f in F]}
    if not F:
        return out
    # support recovery check: saved support vs top-w_f of the in-sample dense solve
    rec = []
    for f in F:
        w = int((Y[f] != 0).sum())
        top = set(torch.topk(Yd[f, :n].abs(), w).indices.tolist())
        rec.append(len(top & set(torch.nonzero(Y[f]).flatten().tolist())) / max(w, 1))
    out["support_recovery"] = rec
    sign = torch.tensor([1.0 if float(Y[f].sum()) >= 0 else -1.0 for f in F], dtype=torch.float64)
    zf = Z[:, F] * sign                                   # (d, |F|) oriented
    zn = zf.norm(dim=0)
    proj = (A_new.T @ zf) / zn                            # (m, |F|)
    cos = proj / A_new.norm(dim=0)[:, None].clamp_min(1e-12)
    coef = (Yd[F] * sign[:, None]).T                      # (n+m, |F|) oriented
    colnorm2 = (Yd ** 2).sum(0)                           # (n+m,)
    share = coef * coef.abs() / colnorm2[:, None].clamp_min(1e-30)
    sc = {"proj": proj.max(1).values, "cos": cos.max(1).values, "coef": coef[n:].max(1).values,
          "share": share[n:].max(1).values}
    out["scores"] = {k_: v.float().numpy() for k_, v in sc.items()}      # columns: token_data(concept)["new"]
    # in-sample versions for the saved support / V* comparisons
    projV = ((A.T @ zf) / zn).max(1).values
    out["scores_vstar"] = {"proj": projV.float().numpy(),
                           "cos": (projV / A.norm(dim=0).clamp_min(1e-12)).float().numpy(),
                           "coef": coef[:n].max(1).values.float().numpy(),
                           "share": share[:n].max(1).values.float().numpy()}
    # admission rules
    absY = Yd.abs()
    adm = {}
    cut = torch.stack([absY[f, :n][Y[f] != 0].min() for f in F])
    dom = absY.argmax(0)
    sup = (Y[F] != 0)
    out["support_vstar"] = sup.any(0).numpy()
    td = token_data(concept)
    pos = {t: n + i for i, t in enumerate(td["new"])}
    out["cond"] = {}
    for cond, cd in td["conds"].items():
        # the extended set of this neutral-ratio condition: V* + its new tokens (columns of A_ext)
        cols = torch.tensor(list(range(n)) + sorted(pos[int(t)] for t in cd["groups"]), dtype=torch.long)
        aY = absY[:, cols]
        n_c = len(cols)
        adm = {}
        if conf["method"] == "fixed":                 # RS-1% re-applied over V*_ext
            w_ext = math.ceil(conf["sparsity"] * n_c)
            rows = torch.zeros(len(F), n_c, dtype=torch.bool)
            rows.scatter_(1, torch.topk(aY[F], w_ext, dim=1).indices, True)
        else:   # GS over V*_ext, as the fit's Allocate step ("rn"): |y_ij| / ||y_i||, exact K, floor/cap
            vc = VWConfig(**{kk: conf[kk] for kk in ("method", "rank", "sparsity", "floor", "cap", "floorcap")})
            K, lo, hi = budgets(n_c, vc)
            score = (aY / aY.norm(dim=1, keepdim=True).clamp_min(1e-12)).float()
            rows = pool_floor_cap(score, K, lo, hi)[F]
        adm["frac"] = rows
        rows = torch.zeros(len(F), n_c, dtype=torch.bool)
        for r, f in enumerate(F):
            rows[r, torch.topk(aY[f], int((Y[f] != 0).sum())).indices] = True
        adm["width"] = rows
        adm["cutoff"] = aY[F] >= cut[:, None]
        adm["argmax"] = torch.stack([(dom[cols] == f) & (coef[cols, r] > 0) for r, f in enumerate(F)])
        adm["share50"] = share[cols].T > 0.5
        new_ids = [int(t) for t in sorted(cd["groups"], key=lambda t: pos[int(t)])]
        out["cond"][cond] = {
            "new_ids": new_ids,                                            # order of the admit_new columns
            "admit_new": {r: v[:, n:].any(0).numpy() for r, v in adm.items()},
            "admit_vstar": {r: v[:, :n].any(0).numpy() for r, v in adm.items()},
            "retain": {r: float((v[:, :n] & sup).sum() / sup.sum()) for r, v in adm.items()},
            "width_frac": [int(v) for v in adm["frac"].sum(1)], "n_ext": n_c}
    return out
