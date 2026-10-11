#!/usr/bin/env python3
"""Proof of concept: do the same concept's words in other languages share a
direction in Gemma's INPUT embedding space with the English EMBER concept feature?

No generation, no judge, no model forward pass -- only the embedding matrix
(read straight from the cached safetensors) and the already-fit English features.

For each (concept C, language L):

  Measure A  (English EMBER subspace  ->  foreign words)
      Q_C = orthonormal basis of the concept's 1-3 embedding-track features.
      score(token) = ||Q_C^T e||^2 / ||e||^2   ("how much of the token's arrow lies in Q_C")
      AUC = P(random concept word in L scores higher than a random ordinary word in L).

  Measure B  (foreign concept direction  <->  English concept direction)
      a_L^C = mean(concept words in L) - mean(ordinary words in L), unit length.
      cos(a_en^C, a_L^C), compared with (i) cosines against OTHER concepts' directions
      (controls for a shared "content word" offset) and (ii) random word sets.
      Also: forward / reverse AUCs under these axes, and a within-language held-out ceiling.

Usage:
    .venv/bin/python experiments/crosslingual_concept_axis.py
    .venv/bin/python experiments/crosslingual_concept_axis.py --concepts Gun Heroin --langs ru de
"""
import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from ember.erasure import features

ROOT = Path(__file__).resolve().parents[1]
WORDS_JSON = Path(__file__).with_name("crosslingual_words.json")
OUT_DIR = Path(__file__).with_name("crosslingual_results")
WORD_RE = re.compile(r"[^\W\d_]{4,}")  # letters only, >=4 chars


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def load_embedding_matrix(model_name: str) -> torch.Tensor:
    idx = json.load(open(hf_hub_download(model_name, "model.safetensors.index.json", local_files_only=True)))
    key = "model.embed_tokens.weight"
    shard = hf_hub_download(model_name, idx["weight_map"][key], local_files_only=True)
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(key)  # (|V|, d), bf16


def qa_texts_by_concept(lang: str) -> dict:
    """{concept: [text, ...]} from the QA files (questions + options + open answers)."""
    mc = ROOT / ("data/mc_questions.json" if lang == "en" else f"data/qa_lang/mc_questions_{lang}.json")
    op = ROOT / ("data/open_questions.json" if lang == "en" else f"data/qa_lang/open_questions_{lang}.json")
    out = {}
    for path, kind in ((mc, "mc"), (op, "open")):
        for concept, splits in json.load(open(path)).items():
            texts = out.setdefault(concept, [])
            for items in splits.values():
                for it in items:
                    texts.append(it["q"])
                    texts.extend(it["options"] if kind == "mc" else [it["a"]])
    return out


def words_from_text(texts) -> set:
    return {w for t in texts for w in WORD_RE.findall(t)}


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #

def auc_parts(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Per positive: fraction of negatives it beats (ties = 0.5). mean() is the AUC."""
    s = np.sort(neg)
    lo, hi = np.searchsorted(s, pos, "left"), np.searchsorted(s, pos, "right")
    return (lo + 0.5 * (hi - lo)) / len(neg)


def auc_ci(parts: np.ndarray, rng, n_boot=2000):
    boots = rng.choice(parts, size=(n_boot, len(parts)), replace=True).mean(1)
    return float(parts.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def energy(X: torch.Tensor, Q: torch.Tensor) -> np.ndarray:
    return ((X @ Q).pow(2).sum(1) / X.pow(2).sum(1).clamp_min(1e-12)).numpy()


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="google/gemma-2-2b-it")
    ap.add_argument("--concepts", nargs="*", default=None, help="default: all concepts in the words JSON")
    ap.add_argument("--langs", nargs="*", default=["de", "es", "fr", "ru"])
    ap.add_argument("--rank", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio-thresh", type=float, default=2.0)
    ap.add_argument("--n-null", type=int, default=500)
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--pool", choices=["single", "mean"], default="single",
                    help="single: only words that are ONE Gemma token ('▁word'); "
                         "mean: every word = mean of its sub-token embeddings (concept and background alike)")
    ap.add_argument("--only-distinct", action="store_true",
                    help="drop foreign concept words that are identical/similar to the English word (kills trivial surface-form overlap)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    words = {k: v for k, v in json.load(open(WORDS_JSON)).items() if not k.startswith("_")}
    concepts = args.concepts or list(words)
    langs = ["en"] + args.langs

    tok = AutoTokenizer.from_pretrained(args.model_name)
    vocab = tok.get_vocab()
    E = load_embedding_matrix(args.model_name)
    print(f"embedding matrix {tuple(E.shape)} {E.dtype}; concepts={concepts}; langs={langs}")

    cache = {}  # word -> vector (or None); one entry per surface word

    def wvec(w):
        if w not in cache:
            if args.pool == "single":
                tid = vocab.get("▁" + w)
                cache[w] = None if tid is None else E[tid].float()
            else:
                ids = tok.encode(" " + w, add_special_tokens=False)
                cache[w] = E[torch.as_tensor(ids)].float().mean(0) if ids else None
        return cache[w]

    def mat(ws):
        vs = [wvec(w) for w in ws]
        return torch.stack([v for v in vs if v is not None])

    # ---- word lists, with a similarity-to-English flag ----------------------------------
    # W[c][L] = list of dicts {word, kind}; kind in {same, similar, distinct}
    W = {c: {} for c in concepts}
    for c in concepts:
        en_words = words[c]["en"]
        for L in langs:
            seen, rows = set(), []
            for i, w in enumerate(words[c][L]):
                if w is None or w in seen or wvec(w) is None:
                    continue
                seen.add(w)
                r = difflib.SequenceMatcher(None, w.lower(), en_words[i].lower()).ratio()
                kind = "same" if r >= 0.99 else "similar" if r >= 0.7 else "distinct"
                if args.only_distinct and L != "en" and kind != "distinct":
                    continue
                rows.append({"word": w, "kind": kind})
            W[c][L] = rows
    wlist = lambda c, L: [r["word"] for r in W[c][L]]

    # ---- backgrounds: ordinary words of language L ----------------------------------------
    bg_nat = {}      # (L, c) -> words from OTHER concepts' QA texts, in L (concept c's own text excluded)
    bg_nat_all = {}  # L -> union (used for the language offset mu)
    for L in langs:
        per = {k: words_from_text(v) for k, v in qa_texts_by_concept(L).items()}
        bg_nat_all[L] = sorted(set().union(*per.values()))
        for c in concepts:
            others = set().union(*[v for k, v in per.items() if k != c])
            bg_nat[(L, c)] = sorted(w for w in others - set(wlist(c, L)) if wvec(w) is not None)
    bg_oth = {(L, c): sorted({w for c2 in concepts if c2 != c for w in wlist(c2, L)} - set(wlist(c, L)))
              for L in langs for c in concepts}

    # mean arrow of ordinary words in L = the language's common offset; subtracted for centering / axes
    mu = {L: mat(bg_nat_all[L]).mean(0) for L in langs}
    base = mu
    print({L: len(bg_nat_all[L]) for L in langs}, "ordinary background words per language")

    # ---- English EMBER subspaces ------------------------------------------------------
    Q, kdims = {}, {}
    for c in concepts:
        pkl, pot = features._embedding_paths(args.model_name, c, args.rank, args.seed)
        F_dir, _, _ = features.load_embedding_payload(pkl)
        feat = features.select_embed_feature_ids(pd.read_csv(pot), ratio_thresh=args.ratio_thresh)
        Q[c], _ = torch.linalg.qr(F_dir[:, feat].float())
        kdims[c] = len(feat)
    d = E.shape[1]
    print({c: f"k'={kdims[c]}" for c in concepts}, f" (chance energy fraction ~ k'/{d})")

    # ---- axes (Measure B) -------------------------------------------------------------
    axis = {(c, L): unit(mat(wlist(c, L)).mean(0) - base[L])
            for c in concepts for L in langs if W[c][L]}

    results, A_rows, B_rows = [], [], []
    for c in concepts:
        for L in langs:
            wl = W[c][L]
            if len(wl) < 5:
                print(f"[skip] {c}/{L}: only {len(wl)} usable words")
                continue
            X = mat(wlist(c, L)); Xc = X - mu[L]
            N_nat, N_oth = mat(bg_nat[(L, c)]), mat(bg_oth[(L, c)])
            q = Q[c]

            # ---- Measure A
            row = {"concept": c, "lang": L, "n": len(wl),
                   "n_distinct": sum(r["kind"] == "distinct" for r in wl), "k": kdims[c]}
            for tag, neg in (("nat", N_nat), ("oth", N_oth)):
                for ctag, f_pos, f_neg in (("raw", X, neg), ("cen", Xc, neg - mu[L])):
                    parts = auc_parts(energy(f_pos, q), energy(f_neg, q))
                    row[f"A_{tag}_{ctag}"] = float(parts.mean())
                    if tag == "nat" and ctag == "raw":
                        row["A_nat_raw"], row["A_lo"], row["A_hi"] = auc_ci(parts, rng)
                        dist = np.array([r["kind"] == "distinct" for r in wl])
                        row["A_nat_raw_distinct"] = float(parts[dist].mean()) if dist.sum() >= 5 else np.nan
            A_rows.append(row)

            # ---- Measure B: forward axis AUCs (English axis applied to L words)
            a_en = axis.get((c, "en"))
            if L != "en" and a_en is not None:
                fwd = {"concept": c, "lang": L}
                fwd["fwd_nat"] = float(auc_parts((X @ a_en).numpy(), (N_nat @ a_en).numpy()).mean())
                fwd["fwd_oth"] = float(auc_parts((X @ a_en).numpy(), (N_oth @ a_en).numpy()).mean())
                # reverse: L axis applied to the English words
                a_L = axis[(c, L)]
                Xe = mat(wlist(c, "en"))
                fwd["rev_nat"] = float(auc_parts((Xe @ a_L).numpy(), (mat(bg_nat[("en", c)]) @ a_L).numpy()).mean())
                fwd["rev_oth"] = float(auc_parts((Xe @ a_L).numpy(), (mat(bg_oth[("en", c)]) @ a_L).numpy()).mean())
                # within-language held-out ceiling (fit axis on half, test on the other half)
                ceil = []
                for _ in range(args.n_splits):
                    perm = rng.permutation(len(X)); h = len(X) // 2
                    fit, test = perm[:h], perm[h:]
                    ax = unit(X[fit].mean(0) - base[L])
                    ceil.append(auc_parts((X[test] @ ax).numpy(), (N_nat @ ax).numpy()).mean())
                fwd["ceil_nat"] = float(np.mean(ceil))
                # random-set null for cos(a_en, a_L)
                cos = float(a_en @ a_L)
                Pn = N_nat
                null = []
                for _ in range(args.n_null):
                    pick = rng.choice(len(Pn), size=len(X), replace=False)
                    null.append(float(a_en @ unit(Pn[pick].mean(0) - base[L])))
                null = np.array(null)
                fwd["cos"] = cos
                fwd["cos_null_mean"], fwd["cos_null_sd"] = float(null.mean()), float(null.std())
                fwd["cos_p_rand"] = float((null >= cos).mean())
                # random-axis nulls for the controlled (vs other concepts' words) AUCs:
                # same statistic, but the axis is built from a random set of ordinary words instead
                Xoe, Pe = mat(bg_oth[("en", c)]), mat(bg_nat[("en", c)])
                rev_null, fwd_null = [], []
                for _ in range(args.n_null):
                    ax_L = unit(Pn[rng.choice(len(Pn), size=len(X), replace=False)].mean(0) - base[L])
                    rev_null.append(auc_parts((Xe @ ax_L).numpy(), (Xoe @ ax_L).numpy()).mean())
                    ax_e = unit(Pe[rng.choice(len(Pe), size=len(Xe), replace=False)].mean(0) - base["en"])
                    fwd_null.append(auc_parts((X @ ax_e).numpy(), (N_oth @ ax_e).numpy()).mean())
                rev_null, fwd_null = np.array(rev_null), np.array(fwd_null)
                fwd["rev_oth_null"], fwd["rev_oth_p"] = float(rev_null.mean()), float((rev_null >= fwd["rev_oth"]).mean())
                fwd["fwd_oth_null"], fwd["fwd_oth_p"] = float(fwd_null.mean()), float((fwd_null >= fwd["fwd_oth"]).mean())
                fwd["qfrac"] = float((Q[c].T @ a_L).pow(2).sum())
                B_rows.append(fwd)

    # ---- cross-concept cosine matrices (control for generic content-word offset) -------
    mats = {}
    for L in args.langs:
        cs = [c for c in concepts if (c, "en") in axis and (c, L) in axis]
        M = np.array([[float(axis[(ce, "en")] @ axis[(cl, L)]) for cl in cs] for ce in cs])
        Qm = np.array([[float((Q[ce].T @ axis[(cl, L)]).pow(2).sum()) for cl in cs] for ce in cs])
        mats[L] = {"concepts": cs, "cos": M.tolist(), "qfrac": Qm.tolist()}

    # ---- report ---------------------------------------------------------------------
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    dfA, dfB = pd.DataFrame(A_rows), pd.DataFrame(B_rows)
    print("\n=== MEASURE A: English EMBER subspace -> concept words in L (AUC; 0.5 = chance) ===")
    print("    A_nat = vs ordinary words from QA texts; A_oth = vs other concepts' words in L; raw/cen = without/with language-mean centering")
    print(dfA.round(3).to_string(index=False))
    print("\n=== MEASURE B: English concept axis <-> L concept axis ===")
    print("    cos = cos(a_en, a_L);  cos_p_rand = P(random word set gives cos >= observed)")
    print("    fwd_* = AUC of L concept words under the English axis; rev_* = AUC of English words under the L axis;")
    print("    ceil_nat = same-language held-out ceiling;  qfrac = share of a_L inside the EMBER subspace (chance ~ k'/d)")
    print("    *_oth_null / *_oth_p = mean and upper-tail p of the same AUC when the axis comes from random ordinary words")
    print(dfB.round(3).to_string(index=False))
    for L, m in mats.items():
        cs = m["concepts"]
        M = np.array(m["cos"])
        print(f"\n--- cos(English axis of row concept, {L} axis of column concept) ---")
        print(pd.DataFrame(M, index=cs, columns=cs).round(3).to_string())
        diag, off = np.diag(M).mean(), M[~np.eye(len(cs), dtype=bool)].mean()
        top1 = np.mean([np.argmax(M[:, j]) == j for j in range(len(cs))])
        print(f"diag mean={diag:.3f}  off-diag mean={off:.3f}  top-1 match (chance {1/len(cs):.2f}) = {top1:.2f}")

    OUT_DIR.mkdir(exist_ok=True)
    tag = args.pool + ("_distinct" if args.only_distinct else "")
    dfA.to_csv(OUT_DIR / f"measure_A_{tag}.csv", index=False)
    dfB.to_csv(OUT_DIR / f"measure_B_{tag}.csv", index=False)
    json.dump(mats, open(OUT_DIR / f"cos_matrices_{tag}.json", "w"), indent=1)
    json.dump({c: {L: W[c][L] for L in langs} for c in concepts}, open(OUT_DIR / f"words_used_{tag}.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\nsaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
