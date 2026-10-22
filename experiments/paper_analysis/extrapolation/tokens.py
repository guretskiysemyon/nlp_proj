"""Extended token sets per concept: the page tokens EMBER never sampled, plus matched new neutral tokens.

    .venv/bin/python -m experiments.paper_analysis.extrapolation.tokens [--concepts ...]

Per concept (gemma-2-2b-it tokenizer, special tokens removed, as V* in experiments/varwidth/prep.py):
    S_C        the 300 sampled concept sentences, S_N the 300 sampled neutral sentences (ConceptDataset, seed 42)
    page       every sentence of the page revision used for the CS target (experiments/whitening/data/pages)
    S_C^ext    S_C u page (deduplicated);  new page sentences = page \\ S_C
    pool       one sentence per page of experiments/resample_sentences/data/wiki_dump_pages.jsonl (1000 random
               English Wikipedia pages; fetch_en_sentences.neutral_sentences, random.Random(7)), S_N and page excluded
    S_N^ext    S_N + the first m pool sentences, with m set by the neutral-ratio condition:
                 balanced  smallest m with #N* >= #C*  (the notebook's C* = N* rule)
                 natural   m = number of new page sentences (EMBER's 1:1 concept:neutral sentence ratio)
    new tokens (types(S_C^ext) u types(S_N^ext)) \\ V*, grouped C* (concept text only), B* (both), N* (neutral only)

Per token (V* and new): page count (occurrences in S_C^ext), sample count (in S_C), Wikipedia count
(wiki_token_counts.pt), its source words in the concept text (most frequent first), lexicon hit (any normalized
source word in concept_words_v1), and the V* code. Output: results/paper_analysis/extrapolation/tokens/<Slug>.json.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

import numpy as np
import torch

from experiments.paper_analysis.extrapolation.common import DUMP, OUT, TOKDIR, all_concepts, page_sentences, \
    write_json
from experiments.varwidth.common import ROOT, slug

POOL_SEED = 7
CONDS = ("balanced", "natural")


def neutral_pool() -> list:
    """One random prose sentence per page of the dump sample, pages in random.Random(POOL_SEED) order (the body of
    fetch_en_sentences.neutral_sentences, without its fixed-n stop)."""
    import json
    sys.path.insert(0, str(ROOT / "experiments" / "resample_sentences"))
    from fetch_en_sentences import PROSE_END, split_sentences
    pages = [json.loads(line) for line in DUMP.open(encoding="utf-8")]
    rng = random.Random(POOL_SEED)
    out = []
    for page in rng.sample(pages, len(pages)):
        paras = [ln.strip() for ln in page["text"].splitlines() if PROSE_END.search(ln.strip())]
        sents = split_sentences(paras)
        if sents:
            out.append(rng.choice(sents))
    return out


def build(concept: str, wiki: np.ndarray, n_wiki: int, lex: dict, pool_all: list) -> dict:
    from ember.judge_words import token_word_counts
    from experiments.varwidth.overlap_judge import normalize
    from experiments.varwidth.prep import build_dataset, get_tokenizer, prepare
    tok = get_tokenizer()
    special = set(tok.all_special_ids)
    prep = prepare(concept, "plain", verbose=False)
    vstar = [int(t) for t in prep["vstar_ids"]]
    vset = set(vstar)
    vcode = dict(zip(vstar, np.asarray(prep["codes"]).tolist()))
    ds = build_dataset(concept)
    s_c = list(dict.fromkeys(s for s, lbl in ds.data if lbl == concept))
    s_n = list(dict.fromkeys(s for s, lbl in ds.data if lbl != concept))
    page, meta = page_sentences(concept)
    sc_set = set(s_c)
    new_page = [s for s in dict.fromkeys(page) if s not in sc_set]
    s_c_ext = s_c + new_page

    def ids_of(sents):
        return [[int(t) for t in seq if int(t) not in special]
                for seq in tok(sents, add_special_tokens=False)["input_ids"]] if sents else []

    c_seqs = ids_of(s_c_ext)
    c_count = Counter(t for seq in c_seqs for t in seq)                 # page frequency (S_C^ext)
    s_count = Counter(t for seq in c_seqs[:len(s_c)] for t in seq)      # sample frequency (S_C)
    c_ids = set(c_count)
    n_ids0 = {t for seq in ids_of(s_n) for t in seq}
    excl = set(s_n) | set(s_c_ext)
    pool = [s for s in pool_all if s not in excl]
    pool_seqs = ids_of(pool)

    def groups(n_ids):
        new = (c_ids | n_ids) - vset
        g = {}
        for t in new:
            g[t] = "B*" if (t in c_ids and t in n_ids) else ("C*" if t in c_ids else "N*")
        return g

    # balanced: smallest m with #N* >= #C*;  natural: m = #new page sentences
    n_ids, m_bal = set(n_ids0), None
    cnt = Counter(groups(n_ids).values())
    trace = []
    for m in range(len(pool_seqs) + 1):
        if m > 0:
            n_ids |= set(pool_seqs[m - 1])
            cnt = Counter(groups(n_ids).values())
        if m % 10 == 0:
            trace.append((m, cnt["C*"], cnt["B*"], cnt["N*"]))
        if m_bal is None and cnt["N*"] >= cnt["C*"]:
            m_bal = m
        if m_bal is not None and m >= len(new_page):
            break
    m_nat = min(len(new_page), len(pool_seqs))
    conds = {}
    for name, m in (("balanced", m_bal), ("natural", m_nat)):
        if m is None:
            m = len(pool_seqs)
        n_ids = set(n_ids0).union(*[set(s) for s in pool_seqs[:m]])
        g = groups(n_ids)
        conds[name] = {"m_neutral": m, "groups": {str(t): v for t, v in g.items()},
                       "counts": dict(Counter(g.values())), "neutral_ids": sorted(n_ids)}
    # every token we ever score: V* and the union of the new tokens of both conditions
    all_new = sorted(set().union(*[set(map(int, c["groups"])) for c in conds.values()]))
    pool_used = pool[:max(c["m_neutral"] for c in conds.values())]
    words = token_word_counts(tok, s_c_ext + pool_used)
    lexset = set(lex.get(concept, []))
    rows = {}
    for t in vstar + all_new:
        wc = Counter()
        for (_, w), k in words.get(t, {}).items():
            wc[w] += k
        top = [w for w, _ in wc.most_common(8)]
        nw = {normalize(w) for w in wc}
        rows[str(t)] = {"tok": tok.convert_ids_to_tokens(t), "words": top, "c_page": c_count.get(t, 0),
                        "c_sample": s_count.get(t, 0), "c_wiki": int(wiki[t]),
                        "lex": bool(nw & lexset), "vcode": vcode.get(t)}
    n_page_tokens = sum(c_count.values())
    out = {"concept": concept, "page": meta, "n_page_sents": len(page), "n_sampled": len(s_c),
           "n_sampled_on_page": len(sc_set & set(page)), "n_new_page_sents": len(new_page),
           "n_neutral_sampled": len(s_n), "n_page_tokens": n_page_tokens, "n_wiki_tokens": n_wiki,
           "vstar": vstar, "new": all_new, "conds": conds, "trace": trace, "tokens": rows,
           "lexicon": sorted(lexset), "pool_seed": POOL_SEED, "pool_size": len(pool)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=None)
    a = ap.parse_args()
    from experiments.varwidth.j3_words import concept_lists
    w = torch.load(OUT / "wiki_token_counts.pt", weights_only=False)
    wiki, n_wiki = w["counts"].numpy(), w["n_tokens"]
    lex = concept_lists("v1")
    pool_all = neutral_pool()
    for c in a.concepts or all_concepts():
        out = build(c, wiki, n_wiki, lex, pool_all)
        write_json(TOKDIR / f"{slug(c)}.json", out, indent=None)
        cc = {k: v["counts"] | {"m": v["m_neutral"]} for k, v in out["conds"].items()}
        print(f"[tokens] {c}: page {out['n_page_sents']} sents ({out['n_sampled_on_page']}/300 sampled on page), "
              f"{out['n_new_page_sents']} new; |V*|={len(out['vstar'])} new={len(out['new'])} {cc}", flush=True)


if __name__ == "__main__":
    main()
