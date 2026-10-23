"""Background token frequencies of English Wikipedia for the exclusivity label (WP5, extrapolation).

    .venv/bin/python -m experiments.paper_analysis.extrapolation.wikifreq [--stride 4]

Reads every `stride`-th row group of the cached parquet shard wikimedia/wikipedia 20231101.en train-00040-of-00041
(the only shard in HF_HOME), tokenizes the article text with the gemma-2-2b-it tokenizer (no special tokens) and
counts token occurrences over the whole vocabulary. Articles whose title is one of the benchmark concepts are
skipped. Output: results/paper_analysis/extrapolation/wiki_token_counts.pt {"counts": (V,) int64, "n_tokens",
"n_articles", "row_groups", "shard"}.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from experiments.paper_analysis.extrapolation.common import OUT, PARQUET, all_concepts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    import pyarrow.parquet as pq

    from experiments.varwidth.prep import get_tokenizer
    tok = get_tokenizer()
    V = len(tok)
    counts = np.zeros(V, dtype=np.int64)
    skip = set(all_concepts())
    f = pq.ParquetFile(PARQUET)
    rgs = list(range(0, f.metadata.num_row_groups, a.stride))
    n_art, t0 = 0, time.time()
    for i, rg in enumerate(rgs):
        t = f.read_row_group(rg, columns=["title", "text"])
        texts = [x for ti, x in zip(t.column("title").to_pylist(), t.column("text").to_pylist()) if ti not in skip]
        for j in range(0, len(texts), 64):
            ids = np.concatenate([np.asarray(x, dtype=np.int64)
                                  for x in tok(texts[j:j + 64], add_special_tokens=False)["input_ids"]])
            counts += np.bincount(ids, minlength=V)[:V]
        n_art += len(texts)
        print(f"[wikifreq] rg {rg} ({i + 1}/{len(rgs)}): {n_art} articles, {counts.sum() / 1e6:.1f}M tokens "
              f"({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"counts": torch.from_numpy(counts), "n_tokens": int(counts.sum()), "n_articles": n_art,
                "row_groups": rgs, "shard": str(PARQUET)}, OUT / "wiki_token_counts.pt")
    print(f"-> {OUT / 'wiki_token_counts.pt'}")


if __name__ == "__main__":
    main()
