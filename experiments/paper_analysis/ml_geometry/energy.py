"""How loud is language identity in the multilingual matrix, raw vs CS?  (judge-free, no fits)

    python -m experiments.paper_analysis.ml_geometry.energy [--workers 2]

Per concept, on the matrix the A2/A3 factorization sees (the N = 30 + 30 per-language sample, rows = V* token types):
  raw  E_V                          (A2)
  cs   X = (E_V - mu_B) W S^(1/2)   (A3: 8-language background whitening, sharpening toward the 8 concept pages;
                                     same construction and settings as experiments/ml/fit.cs_transform)
Token labels: language = argmax_L p(L|t) when p >= 0.8 (tokens.py), else "shared"; concept = code 1, neutral = code 0.
  lang_between   between-language share of the (row-centred) energy, over language-labelled tokens:
                 sum_L n_L ||m_L - m||^2 / sum_t ||x_t - m||^2
  conc_between   the same for concept-only vs neutral-only tokens
  lang_proj      energy share in the 7-dim span of the centred language centroids (random 7-dim: 7/4096)
  pc_eta_lang    mean over the top 10 principal directions of the between-language share of that direction's
                 variance (eta^2); pc_eta_conc the same for concept vs neutral
English-only reference (A1 en, English full set): conc_between of the raw English matrix.
cPCA enrichment (the earlier check, docs/Part 2/multilingual_whitening.md: HP, g = 6.82 English bg vs 2.82 8-language
bg): top eigenvalues g of S_eta = (1-eta) U_C^T U_C / m + eta I for the target (types of the 8 concept pages) whitened
with (a) the 8-language background results/ml/data/bg_ml_llama.pt, (b) the English Wikipedia background built by
WP8 (results/paper_analysis/llama_cs/data/bg_wiki_llama_stats.pt, 300k sentences); and the language purity
(mean maxshare of p(L|t)) of the 50 full-set V* tokens with the largest projection on each of the top 5 directions.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool

import numpy as np
import torch

from experiments.paper_analysis.ml_geometry import tokens as TK
from experiments.paper_analysis.ml_geometry.common import (CACHE, LANGS, MLRES, OUT, ROOT, concepts, embeddings,
                                                           flatten, load_full, save_json, slug, special_ids, tokenizer)

ETA, GAMMA, EPS_REL = 0.05, 1.0, 1e-4
BG_ML = MLRES / "data" / "bg_ml_llama.pt"
BG_EN = ROOT / "results" / "paper_analysis" / "llama_cs" / "data" / "bg_wiki_llama_stats.pt"
_BG = {}


def bg(name):
    if name not in _BG:
        st = torch.load(BG_ML if name == "ml" else BG_EN, weights_only=False, map_location="cpu")
        lam, Q = st["lam"].double().clamp_min(0), st["Q"].double()
        eps = EPS_REL * lam.mean()
        _BG[name] = {"mu": st["mu"].double(), "W": (Q * (lam + eps).pow(-0.5)) @ Q.T}
    return _BG[name]


def token_set(sentences, labels):
    tok, special = tokenizer(), special_ids()
    in_c, in_n = set(), set()
    for s, lab in zip(sentences, labels):
        ids = set(tok(s, add_special_tokens=False)["input_ids"]) - special
        (in_c if lab == "concept" else in_n).update(ids)
    v = sorted(in_c | in_n)
    codes = np.array([2 if (t in in_c and t in in_n) else 1 if t in in_c else 0 for t in v])
    return v, codes


def between(Xc: torch.Tensor, labels: np.ndarray) -> float:
    """Between-class share of the energy of rows Xc (already restricted to labelled rows), centred at their mean."""
    X = Xc - Xc.mean(0)
    tot = float((X ** 2).sum())
    b = 0.0
    for c in np.unique(labels):
        m = labels == c
        b += m.sum() * float((X[torch.from_numpy(m)].mean(0) ** 2).sum())
    return b / tot


def stats(X: torch.Tensor, lang_lab: np.ndarray, codes: np.ndarray) -> dict:
    X = X.double()
    lm = lang_lab >= 0
    cm = codes != 2
    out = {"lang_between": between(X[torch.from_numpy(lm)], lang_lab[lm]),
           "conc_between": between(X[torch.from_numpy(cm)], codes[cm])}
    Xl = X[torch.from_numpy(lm)]
    Xl = Xl - Xl.mean(0)
    cents = torch.stack([Xl[torch.from_numpy(lang_lab[lm] == i)].mean(0) for i in np.unique(lang_lab[lm])])
    B, _ = torch.linalg.qr(cents.T)                                        # (d, <=8) span of the centroids
    out["lang_proj"] = float(((Xl @ B) ** 2).sum() / (Xl ** 2).sum())
    out["lang_proj_dim"] = int(torch.linalg.matrix_rank(cents))
    Xc = X - X.mean(0)
    U, Sv, V = torch.svd_lowrank(Xc, q=30, niter=6)
    eta_l, eta_c = [], []
    for j in range(10):
        p = (Xc @ V[:, j]).numpy()
        for lab, mask, acc in ((lang_lab, lm, eta_l), (codes, cm, eta_c)):
            pv, lv = p[mask], lab[mask]
            tot = ((pv - pv.mean()) ** 2).sum()
            acc.append(sum((lv == c).sum() * (pv[lv == c].mean() - pv.mean()) ** 2 for c in np.unique(lv)) / tot)
    out["pc_eta_lang"] = float(np.mean(eta_l))
    out["pc_eta_conc"] = float(np.mean(eta_c))
    out["pc_eta_lang_list"] = [round(float(x), 3) for x in eta_l]
    out["top10_var_share"] = float((Sv[:10] ** 2).sum() / (Xc ** 2).sum())
    return out


def hard_lang(tk, ids, thr=0.8):
    P = TK.lang_probs(tk, ids)
    lab = P.argmax(1)
    lab[P.max(1) < thr] = -1
    return lab, P


def cpca(name, target_ids, v_full, P_full, E):
    b = bg(name)
    U_t = (E[target_ids].double() - b["mu"]) @ b["W"]
    m = U_t.shape[0]
    _, Sv, V = torch.svd_lowrank(U_t, q=25, niter=6)
    g = (1 - ETA) * Sv ** 2 / m + ETA
    Uv = (E[v_full].double() - b["mu"]) @ b["W"]
    sign = torch.sign(U_t.mean(0) @ V[:, :5])
    sign[sign == 0] = 1
    proj = Uv @ (V[:, :5] * sign)
    pur, top_lang = [], []
    for j in range(5):
        top = torch.argsort(proj[:, j], descending=True)[:50].numpy()
        pur.append(float(P_full[top].max(1).mean()))
        top_lang.append(LANGS[int(np.bincount(P_full[top].argmax(1), minlength=8).argmax())])
    return {"g_top": [round(float(x), 3) for x in g[:10]], "purity_top5": [round(x, 3) for x in pur],
            "purity_mean": float(np.mean(pur)), "top_lang_top5": top_lang}


def run_concept(concept: str) -> dict:
    path = CACHE / f"energy_{slug(concept)}.json"
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(2)
    E = embeddings()
    tk = TK.load(concept)
    sample = json.loads((MLRES / "data" / "sample" / f"{slug(concept)}_N30.json").read_text())
    s, lab, _ = flatten(sample)
    v, codes = token_set(s, lab)
    lang_lab, _ = hard_lang(tk, v)
    full = load_full(concept)
    tok, special = tokenizer(), special_ids()
    target = set()
    for L in LANGS:
        for r in full[L]["concept"]:
            target.update(tok(r["text"] if isinstance(r, dict) else r, add_special_tokens=False)["input_ids"])
    target = sorted(target - special)
    # CS transform, as experiments/ml/fit.cs_transform (8-language background, target = 8 concept pages)
    b = bg("ml")
    U_t = (E[target].double() - b["mu"]) @ b["W"]
    S = (1 - ETA) * (U_t.T @ U_t / U_t.shape[0]) + ETA * torch.eye(U_t.shape[1], dtype=torch.float64)
    g, Pm = torch.linalg.eigh(S)
    Sp = (Pm * g.pow(GAMMA / 2)) @ Pm.T
    Ev = E[v].double()
    X = ((Ev - b["mu"]) @ b["W"]) @ Sp
    out = {"concept": concept, "n": len(v), "shared_frac": float((lang_lab < 0).mean()),
           "lang_counts": np.bincount(lang_lab[lang_lab >= 0], minlength=8).tolist(),
           "raw": stats(Ev, lang_lab, codes), "cs": stats(X, lang_lab, codes),
           "white_only": stats((Ev - b["mu"]) @ b["W"], lang_lab, codes)}
    # English-only reference (A1 en: English full set)
    se, le, _ = flatten(full, ["en"])
    ve, ce = token_set(se, le)
    Ee = E[ve].double()
    cm = ce != 2
    out["en_raw_conc_between"] = between(Ee[torch.from_numpy(cm)], ce[cm])
    # cPCA under the two backgrounds; V* of the full multilingual set for the purity
    sf, lf, _ = flatten(full)
    vf, _ = token_set(sf, lf)
    _, Pf = hard_lang(tk, vf)
    out["cpca_ml"] = cpca("ml", target, vf, Pf, E)
    out["cpca_en"] = cpca("en", target, vf, Pf, E)
    save_json(path, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--concepts", nargs="*")
    a = ap.parse_args()
    cs = a.concepts or concepts()
    res = []
    with Pool(a.workers) as p:
        for r in p.imap_unordered(run_concept, cs):
            print(r["concept"], "lang_between raw/cs %.3f/%.3f conc_between raw/cs %.4f/%.4f g_en %.2f g_ml %.2f" % (
                r["raw"]["lang_between"], r["cs"]["lang_between"], r["raw"]["conc_between"], r["cs"]["conc_between"],
                r["cpca_en"]["g_top"][0], r["cpca_ml"]["g_top"][0]), flush=True)
            res.append(r)
    if not a.concepts:
        save_json(OUT / "energy.json", sorted(res, key=lambda r: concepts().index(r["concept"])))


if __name__ == "__main__":
    main()
