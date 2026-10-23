"""Gemma token frequencies in ordinary English Wikipedia, for the strict (exclusivity) concept label.

    python -m experiments.paper_analysis.circularity.wikifreq

Source: the cached wikimedia/wikipedia 20231101.en shard train-00040-of-00041 (the only shard in the HF cache);
40,000 articles drawn with random.Random(0), minus any article titled like one of the 18 paper concepts' main pages;
full article text, tokenizer(add_special_tokens=False). Output: OUT/wikifreq_gemma.pt {"counts": (V,) int64,
"n_tokens", "n_articles", "shard"}.
"""
from __future__ import annotations

import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments.paper_analysis.circularity.common import OUT, PAPER_CONCEPTS, whitening_module

N_ARTICLES = 40_000
SHARD = ("/home/morg/NLP_2526b/oripetel/.cache/huggingface/hub/datasets--wikimedia--wikipedia/snapshots/"
         "b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en/train-00040-of-00041.parquet")


def main():
    import pyarrow.parquet as pq
    path = OUT / "wikifreq_gemma.pt"
    if path.exists():
        print("exists", path)
        return
    t0 = time.time()
    tok = whitening_module().tokenizer("gemma")
    tab = pq.read_table(SHARD, columns=["title", "text"])
    titles, texts = tab.column("title").to_pylist(), tab.column("text").to_pylist()
    skip = set(PAPER_CONCEPTS) | {"Firearm"}
    rows = [i for i in random.Random(0).sample(range(len(titles)), len(titles)) if titles[i] not in skip][:N_ARTICLES]
    counts = np.zeros(len(tok), dtype=np.int64)
    n = 0
    for b in range(0, len(rows), 500):
        batch = [texts[i] for i in rows[b:b + 500]]
        flat = np.fromiter(itertools.chain.from_iterable(tok(batch, add_special_tokens=False)["input_ids"]),
                           dtype=np.int64)
        counts += np.bincount(flat, minlength=len(tok))
        n += len(flat)
        if b % 10000 == 0:
            print(f"{b + len(batch)} articles, {n} tokens ({time.time() - t0:.0f}s)", flush=True)
    counts = torch.from_numpy(counts)
    torch.save({"counts": counts, "n_tokens": n, "n_articles": len(rows), "shard": Path(SHARD).name}, path)
    print(f"{len(rows)} articles, {n} tokens, {(counts > 0).sum().item()} types ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
