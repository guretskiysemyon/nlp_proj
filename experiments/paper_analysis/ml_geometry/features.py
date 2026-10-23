"""Per-feature language composition of every saved multilingual fit (judge-free).

    python -m experiments.paper_analysis.ml_geometry.features [--workers 4]

For each concept (38), fit seed (42, 1, 2) and approach (A2, A3 on the full multilingual set; A1 per language), and
for each of the k = 200 features i (row Y_i of the saved fit, support = nonzero entries):
  rho        EMBER's ratio_abs (mean |Y| over concept-only tokens / mean over neutral-only tokens), as select.py
  rho_rank   1 = highest rho in the fit; candidate = rho >= 2 and rank <= 10 (the judged set)
  q_L        language mass: sum_t |Y_it| p(L|t) / sum_t |Y_it| (tokens.py); maxshare = max_L q_L, H = entropy (bits)
  s_*        script mass (Latin / Devanagari / Thai / other), same weights
  ej_hits    EJ: exact matches between the first 20 normalized source words (|Y| order, select.py's feature_words)
             and the concept's lexicon (union over the 8 languages, results/ml/lexicons.json); EJ5 = ej_hits >= 5
  lex_langs  languages whose own lexicon list has a match among the first 50 source words
  selected   in the study's F_C (selection.json: TJ for A1/A2, TJ or EJ@5 for A3)
Null: per fit, 200 random supports of the observed widths drawn uniformly from V*, weights = the observed |Y| values.
Writes cache/feat_<slug>.pkl per concept; `--collect` merges them into features.csv.gz and fits.csv.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.ml_geometry import tokens as TK
from experiments.paper_analysis.ml_geometry.common import (CACHE, DOM, EJ_T, EJ_WORDS, LANGS, MLRES, OUT, SCRIPTS, SEEDS,
                                                           TAU, TOP_N, concepts, entropy_bits, fit_dir, script_of, slug,
                                                           tok_str)

LEX = json.loads((MLRES / "lexicons.json").read_text())
DESC_LANG = re.compile(r"\b(english|german|french|italian|portuguese|spanish|hindi|thai|romance|devanagari|language|"
                       r"multilingual|translation|grammatic\w*|function word\w*|preposition\w*|article\w*|pronoun\w*|"
                       r"conjunction\w*|suffix\w*|prefix\w*|particle\w*|verb form\w*|inflect\w*)\b", re.I)


@lru_cache(maxsize=None)
def _tstr(t):
    return tok_str(t)


@lru_cache(maxsize=None)
def _script(t):
    return SCRIPTS.index(script_of(t))


def _rho(Yabs: np.ndarray, codes: np.ndarray) -> np.ndarray:
    c, n = codes == 1, codes == 0
    mc = Yabs[:, c].mean(1) if c.any() else np.zeros(len(Yabs))
    mn = Yabs[:, n].mean(1) if n.any() else np.zeros(len(Yabs))
    return mc / (mn + 1e-8)


def analyse_fit(concept, approach, seed, lang, tk, counts, lex):
    from experiments.varwidth.overlap_judge import feature_words, normalize
    d = fit_dir(concept, approach, seed, lang)
    fit = torch.load(d / "fit.pt", weights_only=False, map_location="cpu")
    sel = json.loads((d / "selection.json").read_text())
    judge = json.loads((d / "judge.json").read_text()) if (d / "judge.json").exists() else {}
    Y = fit["Y"].float().numpy()
    vstar = [int(t) for t in fit["vstar_ids"]]
    codes = np.asarray(fit["codes"]).astype(int)
    k, n = Y.shape
    Yabs = np.abs(Y)
    P = TK.lang_probs(tk, vstar)                                     # (n, 8)
    S = np.zeros((n, len(SCRIPTS)))
    S[np.arange(n), [_script(t) for t in vstar]] = 1
    rho = _rho(Yabs, codes)
    order_rho = np.argsort(-rho, kind="stable")
    rank = np.empty(k, int)
    rank[order_rho] = np.arange(1, k + 1)
    cand = set(int(f) for f in order_rho[:TOP_N] if rho[f] >= TAU)
    F_C = set(int(f) for f in sel["F_C"])
    union = list(dict.fromkeys(w for L in LANGS for w in lex.get(L, [])))
    word_lang = {}
    for L in LANGS:
        for w in lex.get(L, []):
            word_lang.setdefault(w, set()).add(L)
    mass = Yabs.sum(1, keepdims=True).clip(1e-12)
    Q = (Yabs @ P) / mass                                            # (k, 8)
    Sm = (Yabs @ S) / mass
    rows = []
    Yt = torch.from_numpy(Y)
    for f in range(k):
        nz = np.nonzero(Y[f])[0]
        nzt = torch.from_numpy(nz)
        o = nzt[torch.argsort(Yt[f, nzt].abs(), descending=True)].numpy()   # select.py's order (ties included)
        tids = [vstar[int(i)] for i in o]
        fw50 = feature_words(tids, counts, n=50, tok_str=_tstr)
        fw = fw50[:EJ_WORDS]
        fset = {normalize(x) for x in fw}
        hits = [w for w in union if w in fset]
        f50 = {normalize(x) for x in fw50}
        lex_langs = sorted({L for w in union if w in f50 for L in word_lang[w]})
        q = Q[f]
        jd = judge.get(str(f), {})
        row = {"concept": concept, "approach": approach, "seed": seed, "lang": lang or "", "feature": f,
               "width": len(nz), "rho": float(rho[f]), "rho_rank": int(rank[f]), "candidate": f in cand,
               "selected": f in F_C, "maxshare": float(q.max()), "maxlang": LANGS[int(q.argmax())],
               "H": entropy_bits(q), "n_lang10": int((q >= 0.10).sum()),
               "n_script5": int((Sm[f, :3] >= 0.05).sum()), "ej_hits": len(hits), "ej5": len(hits) >= EJ_T,
               "lex_langs": len(lex_langs), "lex_langs_list": ",".join(lex_langs), "matches": "|".join(hits),
               "conc_only_mass": float(Yabs[f, codes == 1].sum() / mass[f, 0]),
               **{f"q_{L}": float(q[i]) for i, L in enumerate(LANGS)},
               **{f"s_{s}": float(Sm[f, i]) for i, s in enumerate(SCRIPTS)}}
        if f in cand or f in F_C or len(hits) >= 3:
            row["top_tokens"] = " ".join(repr(_tstr(t)) for t in tids[:15])
            row["words"] = ", ".join(fw)
        if jd:
            row["desc"] = jd.get("description", "")
            row["desc_lang"] = bool(DESC_LANG.search(jd.get("description", "")))
            v = jd.get("verdict")
            row["tj"] = bool(v and v[0] and float(v[1] or 0) >= 0.85)
        rows.append(row)
    # null: random supports of the same widths
    rng = np.random.default_rng(seed * 1000 + (LANGS.index(lang) if lang else 9))
    widths = (Y != 0).sum(1)
    null = []
    for f in range(k):
        w = int(widths[f])
        if w == 0:
            continue
        idx = rng.choice(n, size=w, replace=False)
        wt = rng.permutation(Yabs[f, Yabs[f] > 0])
        q = (wt @ P[idx]) / wt.sum()
        null.append(q.max())
    fit_row = {"concept": concept, "approach": approach, "seed": seed, "lang": lang or "", "n": n,
               "null_frac_dom": float(np.mean(np.array(null) >= DOM)), "null_med_maxshare": float(np.median(null)),
               "vstar_lang_mass": P.sum(0).round(1).tolist(), "n_FC": len(F_C), "n_tau": int((rho >= TAU).sum())}
    return rows, fit_row


def run_concept(concept: str) -> str:
    path = CACHE / f"feat_{slug(concept)}.pkl"
    if path.exists():
        return f"{concept}: cached"
    torch.set_num_threads(1)
    tk = TK.load(concept)
    counts = tk["counts"]
    lex = LEX[concept]["words"]
    rows, fits = [], []
    for seed in SEEDS:
        for a in ("A2", "A3"):
            r, f = analyse_fit(concept, a, seed, None, tk, counts, lex)
            rows += r
            fits.append(f)
        for L in LANGS:
            cL = {t: Counter({kw: n for kw, n in c.items() if kw[0] == L}) for t, c in counts.items()}
            cL = {t: c for t, c in cL.items() if c}
            r, f = analyse_fit(concept, "A1", seed, L, tk, cL, lex)
            rows += r
            fits.append(f)
    with open(path, "wb") as fh:
        pickle.dump({"rows": rows, "fits": fits}, fh)
    return f"{concept}: {len(rows)} features"


def collect():
    rows, fits = [], []
    for c in concepts():
        with open(CACHE / f"feat_{slug(c)}.pkl", "rb") as fh:
            d = pickle.load(fh)
        rows += d["rows"]
        fits += d["fits"]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "features.csv.gz", index=False)
    pd.DataFrame(fits).to_csv(OUT / "fits.csv", index=False)
    print(f"{len(df)} feature rows, {len(fits)} fits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--concepts", nargs="*")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        return collect()
    cs = a.concepts or concepts()
    with Pool(a.workers) as p:
        for msg in p.imap_unordered(run_concept, cs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
