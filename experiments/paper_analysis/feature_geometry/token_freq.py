#!/usr/bin/env python
"""Gemma token frequencies on a Wikipedia sample (judge-free support property for the norm regression).

    .venv/bin/python -m experiments.paper_analysis.feature_geometry.token_freq [--n-articles 30000]

Reads the one cached shard of wikimedia/wikipedia 20231101.en (HF_HOME), takes a seeded random sample of articles,
tokenizes the text with gemma-2-2b-it's tokenizer (no special tokens) and counts token ids.
Writes results/paper_analysis/feature_geometry/wiki_token_counts.npz {counts (vocab,), n_tokens, n_articles, shard}.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from experiments.paper_analysis.feature_geometry.common import OUT

HF_HOME = Path(os.environ.get("HF_HOME", "/home/morg/NLP_2526b/oripetel/.cache/huggingface"))


def shard_path() -> Path:
    base = HF_HOME / "hub" / "datasets--wikimedia--wikipedia" / "snapshots"
    files = sorted(base.glob("*/20231101.en/*.parquet"))
    assert files, base
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-articles", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    from experiments.varwidth.prep import get_tokenizer
    p = shard_path()
    t = pq.read_table(p, columns=["text"])
    n = t.num_rows
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(n, size=min(a.n_articles, n), replace=False))
    texts = t.column("text").take(idx).to_pylist()
    tok = get_tokenizer()
    counts = np.zeros(len(tok), dtype=np.int64)
    B = 256
    for i in range(0, len(texts), B):
        for ids in tok(texts[i:i + B], add_special_tokens=False)["input_ids"]:
            np.add.at(counts, np.asarray(ids, dtype=np.int64), 1)
        if (i // B) % 20 == 0:
            print(f"{i + B}/{len(texts)} articles, {counts.sum():,} tokens", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "wiki_token_counts.npz", counts=counts, n_tokens=int(counts.sum()),
                        n_articles=len(texts), shard=str(p.name), seed=a.seed)
    print(f"{len(texts)} articles of {n}, {counts.sum():,} tokens, {(counts > 0).sum():,} distinct ids -> "
          f"{OUT / 'wiki_token_counts.npz'}")


if __name__ == "__main__":
    main()
