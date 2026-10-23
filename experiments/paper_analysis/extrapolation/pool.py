"""Blind labelling pool: which tokens get a hand label, written without any rule / dictionary / sample information.

    .venv/bin/python -m experiments.paper_analysis.extrapolation.pool [--batches 12]

Tokens labelled per concept (decided here, before any label exists):
    FULL concepts (FULL_LABEL, the 6 with the fewest new page tokens + Harry Potter): every new page token (C*/B*)
    other concepts: the union over runs (3 dictionaries x 5 seeds, F_C non-empty) of
        - the top POOL_K new page tokens by each score (proj, cos, coef, share; ranked among new page tokens only,
          so precision@k for k <= POOL_K is exact in both neutral-ratio conditions), and
        - every new page token admitted by any admission rule in either condition, plus
        - N_RANDOM new page tokens drawn uniformly (base rate)
    all concepts: IN_SAMPLE random tokens of the union of the saved concept-feature supports (concept-only V* tokens,
        i.e. EMBER's edit sets) over the 3 dictionaries x 5 seeds (the in-sample reference)
New neutral tokens (N*) need no label: they come from random Wikipedia sentences and are labelled U.

Each item shows only: concept, token string, its 3 most frequent source words in the concept text, one context
snippet (the first concept sentence containing it, +-8 words). Items are shuffled within concept (random ids).
Outputs (results/paper_analysis/extrapolation/labels/):
    batch_XX.json   [{"concept", "items": [{"id", "token", "words", "context"}]}]   for the labellers
    key.json        {id: {"concept", "tid", "sets": [...]}}                         kept away from the labellers
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from collections import Counter

import numpy as np

from experiments.paper_analysis.extrapolation.common import CONCEPTS, OUT, TOKDIR, page_sentences, write_json
from experiments.paper_analysis.extrapolation.score import SCORES
from experiments.varwidth.common import slug

FULL_LABEL = ["Valentine's Day", "Halloween", "Tuberculosis", "Surfing", "Baseball", "Heroin", "Harry Potter"]
POOL_K = 20
N_RANDOM = 50
IN_SAMPLE = 80
SEED = 2026
LABDIR = OUT / "labels"


def concept_text(concept: str) -> list:
    from experiments.varwidth.prep import build_dataset
    ds = build_dataset(concept)
    s_c = list(dict.fromkeys(s for s, lbl in ds.data if lbl == concept))
    page, _ = page_sentences(concept)
    return s_c + [s for s in dict.fromkeys(page) if s not in set(s_c)]


def contexts(tok, sents: list, tids: set) -> tuple[dict, dict]:
    """tid -> 3 most frequent source words (concept text only), tid -> first context snippet."""
    from ember.judge_words import token_word_counts
    wc = token_word_counts(tok, sents)
    words = {}
    for t in tids:
        c = Counter()
        for (_, w), k in wc.get(t, {}).items():
            c[w] += k
        words[t] = [w for w, _ in c.most_common(3)]
    ctx = {}
    todo = set(tids)
    for s in sents:
        if not todo:
            break
        enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
        for t, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
            if t in todo:
                left = s[:a].split()[-8:]
                right = s[b:].split()[:8]
                ctx[t] = " ".join(left) + " [[" + s[a:b].strip() + "]] " + " ".join(right)
                todo.discard(t)
    return words, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=12)
    a = ap.parse_args()
    from experiments.varwidth.prep import get_tokenizer
    tok = get_tokenizer()
    rng = random.Random(SEED)
    key, per_concept = {}, {}
    for ci, c in enumerate(CONCEPTS):
        td = json.loads((TOKDIR / f"{slug(c)}.json").read_text())
        toks, new, vstar = td["tokens"], td["new"], td["vstar"]
        page_new = [t for t in new if toks[str(t)]["c_page"] > 0]
        R = pickle.load(open(OUT / "runs" / f"{slug(c)}.pkl", "rb"))
        sets: dict[int, set] = {}

        def add(t, s):
            sets.setdefault(int(t), set()).add(s)
        is_page = np.array([toks[str(t)]["c_page"] > 0 for t in new])
        ins = set()
        for (D, s), r in R.items():
            if not r["F_C"]:
                continue
            for sc in SCORES:
                v = np.where(is_page, r["scores"][sc], -np.inf)
                for i in np.argsort(-v, kind="stable")[:POOL_K]:
                    add(new[i], "pool")
            for cond, cd in r["cond"].items():
                for rule, mask in cd["admit_new"].items():
                    for i in np.nonzero(mask)[0]:
                        t = cd["new_ids"][i]
                        if toks[str(t)]["c_page"] > 0:
                            add(t, "pool")
            ins.update(vstar[i] for i in np.nonzero(r["support_vstar"])[0] if toks[str(vstar[i])]["vcode"] == 1)
        if c in FULL_LABEL:
            for t in page_new:
                add(t, "full")
        for t in rng.sample(page_new, min(N_RANDOM, len(page_new))):
            add(t, "random")
        for t in rng.sample(sorted(ins), min(IN_SAMPLE, len(ins))):
            add(t, "insample")
        words, ctx = contexts(tok, concept_text(c), set(sets))
        items = []
        order = sorted(sets)
        rng.shuffle(order)
        for t in order:
            iid = f"{ci:02d}{rng.randrange(16**5):05x}"
            while iid in key:
                iid = f"{ci:02d}{rng.randrange(16**5):05x}"
            key[iid] = {"concept": c, "tid": t, "sets": sorted(sets[t])}
            items.append({"id": iid, "token": toks[str(t)]["tok"], "words": words.get(t, []),
                          "context": ctx.get(t, "")})
        per_concept[c] = items
        cnt = Counter(s for t in sets for s in sets[t])
        print(f"[pool] {c}: {len(items)} items {dict(cnt)}", flush=True)
    # batches: whole concepts, greedy balance by item count
    loads = [[0, i, []] for i in range(a.batches)]
    for c in sorted(per_concept, key=lambda c: -len(per_concept[c])):
        loads.sort()
        loads[0][0] += len(per_concept[c])
        loads[0][2].append(c)
    LABDIR.mkdir(parents=True, exist_ok=True)
    for n_items, i, cs in sorted(loads, key=lambda x: x[1]):
        write_json(LABDIR / f"batch_{i:02d}.json", [{"concept": c, "items": per_concept[c]} for c in cs])
        print(f"[pool] batch_{i:02d}: {n_items} items, {cs}")
    write_json(LABDIR / "key.json", key, indent=None)
    print(f"-> {LABDIR} ({len(key)} items)")


if __name__ == "__main__":
    main()
