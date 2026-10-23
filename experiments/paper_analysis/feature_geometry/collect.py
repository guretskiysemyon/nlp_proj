#!/usr/bin/env python
"""WP2 step 0: per-feature and per-pair geometry of every saved fit (L0 = EMBER, L1 = CS + RS-1%, L12 = CS + GS).

    .venv/bin/python -m experiments.paper_analysis.feature_geometry.collect [--concepts ...] [--workers 4]

No new fits, no Gemini. For every (concept, config, seed 42-46) with a fit.pt:

features.parquet, one row per feature i (k = 100):
    width, nC/nN/nB (support counts by token code), share (concept-only share of the support), rho (EMBER's
    mean|Y| over C / mean|Y| over N), sign (sign of sum_t y_it; directions are sign-aligned with it),
    znorm_fit = ||z_i|| in the space the fit ran (raw for L0, CS space for L1/L12), znorm_raw = ||z_i T^-1|| (raw
    space; = znorm_fit for L0), znorm_cs = ||z_i^raw T|| (CS metric; = znorm_fit for CS fits),
    support properties: e_norm / e_norm_w (mean and |y|-weighted mean raw token norm ||e_t||), x_norm (mean CS-space
    token norm ||x_t||), u_norm / u_norm_w (background-whitened token norm ||(e_t - mu_B) W||), logf / logf_w (mean and |y|-weighted mean log10 frequency per million in a 30k-article
    Wikipedia sample, +1 smoothing), f0 (share of support tokens never seen in the sample),
    projection decomposition in the fit space (z*_i = A y_i^T, exact when supports are disjoint): l1 = sum_t |y_it|,
    nbar = sum |y| ||a_t|| / l1 (|y|-weighted fit-space token norm), kappa = ||A y_i^T|| / sum |y| ||a_t|| (coherence),
    pnorm = ||A y_i^T||,
    cos_mu = cos(z_i^raw, mu_B) (background mean embedding direction),
    lo = lexicon hits (concept_words_v1) among the feature's first 20 source words, cand_rank (rank among the top-20
    by rho = the judge set, else -1), tj (EMBER judge accepted, from layer_cache), hj = tj or (cand and lo >= 5),
    in_fc (F_C of summary.json), label (hand label P/B/N via j3 group key, if any), desc (stage-1 Gemini description
    from judge.json, if any).
words.json: {"concept|cfg|seed|f": first 20 normalized source words}.
tokens/<slug>.npz: per V* token e_norm, u_norm, x_norm (and ids, codes).
pairs/<slug>__<cfg>__s<seed>.npz: upper triangles (k(k-1)/2) of the sign-aligned cosine matrices
    cos_fit (fit space), cos_raw (raw space), cos_cs (CS metric), cos_w (background-whitening metric W only),
    cos_rawc (raw space with the mu_B direction projected out), and jac (Jaccard of the two supports).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.feature_geometry.common import CONFIGS, OUT, SEEDS

PAIRS = OUT / "pairs"


def _unit_cols(M: torch.Tensor) -> torch.Tensor:
    return M / M.norm(dim=0, keepdim=True).clamp_min(1e-30)


def _triu(C: torch.Tensor) -> np.ndarray:
    k = C.shape[0]
    iu = torch.triu_indices(k, k, offset=1)
    return C[iu[0], iu[1]].numpy().astype(np.float32)


def _bg_W(eps: float) -> torch.Tensor:
    from experiments.varwidth.prep import WH_DIR
    s = torch.load(WH_DIR / "bg_wiki_stats.pt", map_location="cpu")
    lam, Q = s["lam"].double(), s["Q"].double()
    return (Q * (lam + eps).rsqrt()) @ Q.T


def concept_rows(concept: str) -> tuple[list, dict]:
    torch.set_num_threads(1)
    from ember.judge_words import token_word_counts
    from experiments.varwidth import overlap_judge as oj
    from experiments.varwidth.common import run_dir, slug
    from experiments.varwidth.j3_eval import group_key, j3_count
    from experiments.varwidth.j3_words import concept_lists, DATA
    from experiments.varwidth.judge_compare import row_order
    from experiments.varwidth.prep import build_dataset, get_tokenizer, prepare

    tok = get_tokenizer()
    pp = prepare(concept, "plain", verbose=False)
    pw = prepare(concept, "whitened", verbose=False)
    assert list(pp["vstar_ids"]) == list(pw["vstar_ids"])
    vstar = list(pp["vstar_ids"])
    codes = np.asarray(pp["codes"])
    E = pp["A"].double()                                     # (d, n) raw embeddings (x sqrt d)
    X = pw["A"].double()                                     # (d, n) CS-space token vectors
    T = pw["T_matrix"].double()                              # row convention: x = (e - mu_B) T
    Tinv = pw["T_inv"].double()
    mu = pw["mu_B"].double()
    W = _bg_W(float(pw["eps"]))
    mu_u = mu / mu.norm()
    e_norm, x_norm = E.norm(dim=0).numpy(), X.norm(dim=0).numpy()
    u_norm = ((E - mu[:, None]).T @ W).norm(dim=1).numpy()   # background-whitened token norm (Mahalanobis)
    wc = np.load(OUT / "wiki_token_counts.npz")
    cnt, ntok = wc["counts"], float(wc["n_tokens"])
    (OUT / "tokens").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "tokens" / f"{slug(concept)}.npz", vstar=np.asarray(vstar), codes=codes,
                        e_norm=e_norm, u_norm=u_norm, x_norm=x_norm, count=cnt[np.asarray(vstar)])
    vcnt = cnt[np.asarray(vstar)]
    logf = np.log10((vcnt + 1) / ntok * 1e6)
    counts = token_word_counts(tok, [s for s, _ in build_dataset(concept).data])
    cw = concept_lists("v1").get(concept)
    labels = {}
    for p in sorted(DATA.glob("labels*.json")):
        labels.update(json.loads(p.read_text()))
    isC, isN = codes == 1, codes == 0
    rows, words = [], {}
    PAIRS.mkdir(parents=True, exist_ok=True)
    for cfg, tag in CONFIGS.items():
        for seed in SEEDS:
            d = run_dir(concept, tag, seed)
            if not (d / "fit.pt").exists():
                continue
            fit = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)
            Z, Y = fit["Z"].double(), fit["Y"].double()
            k = Z.shape[1]
            A = E if cfg == "L0" else X
            sgn = torch.sign(Y.sum(1))
            sgn[sgn == 0] = 1
            Zs = Z * sgn                                     # sign-aligned columns
            if cfg == "L0":
                Zraw, Zcs = Zs, (Zs.T @ T).T
            else:
                Zraw, Zcs = (Zs.T @ Tinv).T, Zs
            Zw = (Zraw.T @ W).T
            Zrc = Zraw - mu_u[:, None] * (mu_u @ Zraw)[None, :]
            cos = {n: _unit_cols(M).T @ _unit_cols(M) for n, M in
                   (("cos_fit", Zs), ("cos_raw", Zraw), ("cos_cs", Zcs), ("cos_w", Zw), ("cos_rawc", Zrc))}
            S = (Y.abs() > 1e-12)
            Sf = S.double()
            inter = Sf @ Sf.T
            w = Sf.sum(1)
            jac = inter / (w[:, None] + w[None, :] - inter).clamp_min(1)
            np.savez_compressed(PAIRS / f"{slug(concept)}__{cfg}__s{seed}.npz",
                                **{n: _triu(C) for n, C in cos.items()}, jac=_triu(jac))
            Yabs = Y.abs().numpy()
            rho = Yabs[:, isC].mean(1) / (Yabs[:, isN].mean(1) + 1e-8)
            AY = A @ Y.T                                     # (d, k) projections A y_i^T
            anorm = A.norm(dim=0).numpy()
            cache = json.loads((d / "layer_cache.json").read_text()) if (d / "layer_cache.json").exists() else None
            cand = {c["feature"]: c for c in cache["candidates"]} if cache else {}
            summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
            fcset = set(summ.get("F_C", []))
            descs = json.loads((d / "judge.json").read_text()).get("descriptions", {}) if (d / "judge.json").exists() else {}
            znf, znr, znc = Z.norm(dim=0).numpy(), Zraw.norm(dim=0).numpy(), Zcs.norm(dim=0).numpy()
            cmu = (mu_u @ _unit_cols(Zraw)).numpy()
            for i in range(k):
                sup = np.nonzero(S[i].numpy())[0]
                order = row_order(Y.float(), i)
                fw = oj.feature_words([vstar[j] for j in order], counts, n=20, tok_str=lambda t: tok.decode([t]))
                words[f"{concept}|{cfg}|{seed}|{i}"] = fw
                c = cand.get(i)
                if c is not None and c["fw20"] != fw:
                    print(f"WARN fw20 mismatch {concept} {cfg} {seed} {i}", flush=True)
                lo = j3_count(cw, fw) if cw else 0
                ya = Yabs[i, sup]
                l1 = float(ya.sum())
                wsum = float((ya * anorm[sup]).sum())
                rows.append({
                    "concept": concept, "cfg": cfg, "seed": seed, "feature": i, "width": len(sup),
                    "nC": int(isC[sup].sum()), "nN": int(isN[sup].sum()), "nB": int((codes[sup] == 2).sum()),
                    "share": float(isC[sup].mean()) if len(sup) else 0.0, "rho": float(rho[i]),
                    "sign": int(sgn[i]), "znorm_fit": float(znf[i]), "znorm_raw": float(znr[i]),
                    "znorm_cs": float(znc[i]),
                    "e_norm": float(e_norm[sup].mean()), "e_norm_w": float((ya * e_norm[sup]).sum() / l1),
                    "x_norm": float(x_norm[sup].mean()), "x_norm_w": float((ya * x_norm[sup]).sum() / l1),
                    "u_norm": float(u_norm[sup].mean()), "u_norm_w": float((ya * u_norm[sup]).sum() / l1),
                    "logf": float(logf[sup].mean()), "logf_w": float((ya * logf[sup]).sum() / l1),
                    "f0": float((vcnt[sup] == 0).mean()),
                    "l1": l1, "nbar": wsum / l1, "kappa": float(AY[:, i].norm()) / wsum, "pnorm": float(AY[:, i].norm()),
                    "cos_mu": float(cmu[i]), "lo": int(lo),
                    "cand_rank": int(c["rank"]) if c else -1, "tj": bool(c and c["tj"]),
                    "hj": bool(c and (c["tj"] or (c["lo"] or 0) >= 5)), "in_fc": i in fcset,
                    "label": labels.get(group_key(concept, fw)), "desc": descs.get(str(i)),
                    "rel_err": float(fit["rel_err"])})
    return rows, words


def main():
    from experiments.varwidth.judge_report import concepts
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cs = a.concepts or concepts()
    rows, words = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for c, (r, w) in zip(cs, ex.map(concept_rows, cs)):
            rows += r
            words.update(w)
            print(f"{c}: {len(r)} feature rows", flush=True)
    df = pd.DataFrame(rows)
    suffix = "" if a.concepts is None else "_partial"
    df.to_parquet(OUT / f"features{suffix}.parquet")
    (OUT / f"words{suffix}.json").write_text(json.dumps(words, ensure_ascii=False))
    print(f"{len(df)} rows, {df.groupby(['concept', 'cfg', 'seed']).ngroups} runs -> {OUT}")


if __name__ == "__main__":
    main()
