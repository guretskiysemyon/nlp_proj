"""Per-concept token labels for the judge-free metrics (decided before any WP7 fit was looked at).

    python -m experiments.paper_analysis.circularity.labels

For every V* token t of a concept (S300 prep, the same V* for every condition):
    code      1 concept-only (in S_C, never in S_N), 0 neutral-only, 2 both (prep codes)
    word      its normalized source word, as experiments/varwidth/overlap_judge.feature_words derives it (most common
              whole word it came from in S_C ∪ S_N, _piece, normalize); "" if not a word (_is_word fails)
    lex       word is in the concept's lexicon (j3_data/concept_words_v1.json, normalized)
    n_sc      occurrences of t in the 300 S_C sentences (add_special_tokens=False); N_SC their total
    lr        log( (n_sc / N_SC) / ((n_W + 1) / (N_W + V)) ), n_W from wikifreq_gemma.pt (40k Wikipedia articles)
    strict    code == 1 and (lex or (n_sc >= 2 and lr >= log 10))     <- primary strict label
    strict_any  code == 1 and (lex or lr >= log 10)                    <- sensitivity: no minimum count
    lexonly   code == 1 and lex                                         <- sensitivity: lexicon only
    in_<T>    t is a token type of target T (texts.py), for the held-out-token metrics
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

from experiments.paper_analysis.circularity.common import (OUT, PAPER_CONCEPTS, TARGETS, TEXTS, load_prep, read_json,
                                                         slug, write_json)

LR_MIN, NSC_MIN = math.log(10.0), 2


def token_words(tok, vstar, sents) -> list[str]:
    from ember.judge_words import token_word_counts
    from experiments.varwidth import overlap_judge as oj
    counts = token_word_counts(tok, sents)
    out = []
    for t in vstar:
        c = counts.get(int(t))
        ts = tok.decode([int(t)])
        if c:
            merged = Counter()
            for (_, word), k in c.items():
                merged[word] += k
            word = oj._piece(merged.most_common(1)[0][0], ts)
        else:
            word = ts
        w = oj.normalize(word)
        out.append(w if oj._is_word(w) else "")
    return out


def build(concept: str, tok, wf, lexicons) -> dict:
    path = OUT / "labels" / f"{slug(concept)}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    from experiments.varwidth.prep import build_dataset
    prep = load_prep(concept, "plain")
    vstar = [int(t) for t in prep["vstar_ids"]]
    codes = np.asarray(prep["codes"])
    ds = build_dataset(concept)
    s_c = [s for s, lbl in ds.data if lbl == concept]
    words = token_words(tok, vstar, [s for s, _ in ds.data])
    lexset = set(lexicons[concept])
    lex = np.array([w in lexset and w != "" for w in words])
    cnt = Counter()
    n_sc_tot = 0
    for seq in tok(s_c, add_special_tokens=False)["input_ids"]:
        cnt.update(seq)
        n_sc_tot += len(seq)
    n_sc = np.array([cnt.get(t, 0) for t in vstar])
    V, N_W = wf["counts"].numel(), wf["n_tokens"]
    n_w = wf["counts"][torch.tensor(vstar)].numpy()
    with np.errstate(divide="ignore"):
        lr = np.log(n_sc / n_sc_tot) - np.log((n_w + 1) / (N_W + V))
    is_c = codes == 1
    strict = is_c & (lex | ((n_sc >= NSC_MIN) & (lr >= LR_MIN)))
    strict_any = is_c & (lex | (lr >= LR_MIN))
    tg = read_json(TEXTS / "targets" / f"{slug(concept)}.json")["ids"]
    lab = {"concept": concept, "vstar_ids": vstar, "codes": codes, "words": words, "lex": lex, "n_sc": n_sc,
           "N_SC": n_sc_tot, "lr": lr, "strict": strict, "strict_any": strict_any, "lexonly": is_c & lex,
           "tokens": [tok.decode([t]) for t in vstar], "lexicon": sorted(lexset)}
    for T in TARGETS:
        s = set(tg[T])
        lab[f"in_{T}"] = np.array([t in s for t in vstar])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lab, path)
    return lab


def main():
    from experiments.paper_analysis.circularity.common import whitening_module
    from experiments.varwidth.j3_words import concept_lists
    tok = whitening_module().tokenizer("gemma")
    wf = torch.load(OUT / "wikifreq_gemma.pt", weights_only=False)
    lexicons = concept_lists("v1")
    rows = []
    for c in PAPER_CONCEPTS:
        lab = build(c, tok, wf, lexicons)
        is_c = lab["codes"] == 1
        r = {"concept": c, "n": len(lab["codes"]), "C": int(is_c.sum()), "N": int((lab["codes"] == 0).sum()),
             "B": int((lab["codes"] == 2).sum()), "strict": int(lab["strict"].sum()),
             "strict_any": int(lab["strict_any"].sum()), "lexonly": int(lab["lexonly"].sum()),
             "lex_all_codes": int(lab["lex"].sum())}
        for T in TARGETS:
            r[f"C_in_{T}"] = round(float(lab[f"in_{T}"][is_c].mean()), 3)
            r[f"strict_in_{T}"] = round(float(lab[f"in_{T}"][lab["strict"]].mean()), 3) if lab["strict"].any() else None
            r[f"N_in_{T}"] = round(float(lab[f"in_{T}"][lab["codes"] == 0].mean()), 3)
        rows.append(r)
        print(r, flush=True)
    write_json(OUT / "labels_summary.json", rows)


if __name__ == "__main__":
    main()
