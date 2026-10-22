"""Per-concept token statistics of the multilingual full set (8 languages x 300 concept + 300 neutral sentences).

    python -m experiments.paper_analysis.ml_geometry.tokens [--workers 4]

For every token id of the full set: occurrences per language (both sides), per language on the concept side, and
the source-word counts that experiments/ml/select.py feeds to feature_words (ember.judge_words.token_word_counts,
pythainlp words for Thai). Cached as cache/tokens_<slug>.pkl. Every V* token of A2/A3 (full set) and of each A1
language fit is a token of these sentences, so each gets a language label:
    p(L | t) = (c_L(t) / N_L) / sum_L' (c_L'(t) / N_L'),   N_L = token occurrences in language L,
i.e. the language distribution of the token's occurrences corrected for the languages' different corpus sizes.
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from multiprocessing import Pool

import numpy as np

from experiments.paper_analysis.ml_geometry.common import CACHE, LANGS, concepts, flatten, load_full, slug, tokenizer


def build(concept: str) -> str:
    path = CACHE / f"tokens_{slug(concept)}.pkl"
    if path.exists():
        return f"{concept}: cached"
    from ember.judge_words import token_word_counts
    tok = tokenizer()
    sents, labels, langs = flatten(load_full(concept))
    counts = token_word_counts(tok, sents, langs)              # tid -> Counter{(lang, word): n}, as select.py
    occ = {}                                                   # tid -> (8,) occurrences per language (both sides)
    occ_c = {}                                                 # tid -> (8,) concept side only
    N = np.zeros(len(LANGS))
    for s, lab, L in zip(sents, labels, langs):
        li = LANGS.index(L)
        ids = tok(s, add_special_tokens=False)["input_ids"]
        N[li] += len(ids)
        for t, n in Counter(ids).items():
            occ.setdefault(t, np.zeros(len(LANGS)))[li] += n
            if lab == "concept":
                occ_c.setdefault(t, np.zeros(len(LANGS)))[li] += n
    with open(path, "wb") as f:
        pickle.dump({"counts": counts, "occ": occ, "occ_c": occ_c, "N": N}, f)
    return f"{concept}: {len(occ)} types"


def load(concept: str) -> dict:
    with open(CACHE / f"tokens_{slug(concept)}.pkl", "rb") as f:
        return pickle.load(f)


def lang_probs(tk: dict, ids) -> np.ndarray:
    """(len(ids), 8) p(L | t) as in the module docstring."""
    out = np.zeros((len(ids), len(LANGS)))
    for i, t in enumerate(ids):
        c = tk["occ"].get(int(t))
        if c is not None:
            r = c / tk["N"]
            out[i] = r / r.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as p:
        for msg in p.imap_unordered(build, concepts()):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
