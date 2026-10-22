#!/usr/bin/env python
"""General-Wikipedia token frequencies (Gemma tokenizer) for the exclusivity label of WP3.

    .venv/bin/python -m experiments.paper_analysis.widths.wikifreq [--n-articles 25000]

Source: the one cached shard of wikimedia/wikipedia 20231101.en (train-00040-of-00041, 156k articles). A seeded random
sample of articles (titles equal to one of the 38 study concepts are dropped) is tokenized with add_special_tokens=False;
counts are token occurrences (cf) and article document frequency (df).
Output: results/paper_analysis/widths/wiki_counts.npz {cf: (V,) int64, df: (V,) int64, n_tokens, n_docs}.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.paper_analysis.widths.common import OUT, STUDY_CONCEPTS

SHARD = Path("/home/morg/NLP_2526b/oripetel/.cache/huggingface/hub/datasets--wikimedia--wikipedia/snapshots/"
             "b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en/train-00040-of-00041.parquet")
PATH = OUT / "wiki_counts.npz"


def build(n_articles: int = 25_000, seed: int = 0, batch: int = 256) -> Path:
    import pyarrow.parquet as pq

    from experiments.varwidth.prep import get_tokenizer
    t0 = time.time()
    tb = pq.read_table(SHARD, columns=["title", "text"])
    titles = tb.column("title").to_pylist()
    rng = np.random.default_rng(seed)
    keep = [i for i in rng.permutation(len(titles)) if titles[i] not in STUDY_CONCEPTS][:n_articles]
    texts = tb.column("text").take(np.sort(keep)).to_pylist()
    del tb
    tok = get_tokenizer()
    V = len(tok)
    cf = np.zeros(V, dtype=np.int64)
    df = np.zeros(V, dtype=np.int64)
    n_tok = 0
    for b in range(0, len(texts), batch):
        enc = tok(texts[b:b + batch], add_special_tokens=False)["input_ids"]
        for ids in enc:
            a = np.asarray(ids, dtype=np.int64)
            cf += np.bincount(a, minlength=V)
            df[np.unique(a)] += 1
            n_tok += len(a)
        if b % (batch * 20) == 0:
            print(f"[wikifreq] {b + len(enc)}/{len(texts)} articles, {n_tok / 1e6:.1f}M tokens ({time.time() - t0:.0f}s)",
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PATH, cf=cf, df=df, n_tokens=n_tok, n_docs=len(texts), seed=seed, shard=str(SHARD))
    print(f"[wikifreq] {len(texts)} articles, {n_tok} tokens -> {PATH} ({time.time() - t0:.0f}s)")
    return PATH


def load() -> dict:
    z = np.load(PATH)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-articles", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.n_articles, a.seed)
