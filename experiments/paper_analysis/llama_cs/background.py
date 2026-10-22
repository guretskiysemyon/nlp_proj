#!/usr/bin/env python
"""English Wikipedia background statistics for Llama-3.1-8B-Instruct, on the SAME sentences as Gemma's existing
background (experiments/whitening/data/gemma/bg_wiki_stats.pt, n = 107,472 Gemma token types).

The Gemma background came from experiments/whitening/background.py: a deterministic sentence stream
(wikimedia/wikipedia 20231101.en shards in a seeded order -- shard 40 first, the one in the HF cache -- articles in a
seeded permutation, <= 4 prose sentences each) cut into chunks of 20k sentences; sampling stopped at the first
chunk where both models' split-half stability criteria held. Its sentence file is not in this checkout, so this
script regenerates the stream with the same code (background.SentenceSource), adds chunks until the cumulative Gemma
type count equals 107,472 (identifying the stopping chunk exactly), and then takes the Llama token types of the same
sentences (each type once, special/reserved ids removed) -> mean and eigendecomposition (float64, chunked).
Also records the Llama split-half stability (top-100 eigvec principal-angle sines) at that point.

    python -m experiments.paper_analysis.llama_cs.background
Output: results/paper_analysis/llama_cs/data/bg_wiki_llama_stats.pt, bg_wiki_llama_meta.json, bg_wiki_llama_types.json
"""
from __future__ import annotations

import json
import random
import time

import torch

from experiments.paper_analysis.llama_cs.common import DATA, excluded_ids, tokenizer, write_json

GEMMA_N = 107_472


def types_of(model, sents):
    tok, ex = tokenizer(model), excluded_ids(model)
    ids = set()
    for i in range(0, len(sents), 2000):
        for seq in tok(sents[i:i + 2000], add_special_tokens=False)["input_ids"]:
            ids.update(seq)
    return {int(i) for i in ids if i not in ex}


def main():
    torch.set_num_threads(4)
    import background as wbg  # experiments/whitening/background.py (on sys.path via common)
    import common as wc       # experiments/whitening/common.py
    t0 = time.time()
    src = wbg.SentenceSource()
    g_types, l_types, n_sent, curve = set(), set(), 0, []
    sent_f = (DATA / "wiki_sentences.jsonl").open("w", encoding="utf-8")
    for c in range(wbg.MAX_SENTS // wbg.CHUNK):
        rows = src.take(wbg.CHUNK)
        if not rows:
            raise RuntimeError("sentence source exhausted before matching the Gemma background")
        sents = [r["sentence"] for r in rows]
        for r in rows:
            sent_f.write(json.dumps({"chunk": c, **r}, ensure_ascii=False) + "\n")
        n_sent += len(sents)
        g_types |= types_of("gemma", sents)
        l_types |= types_of("llama", sents)
        curve.append({"chunk": c, "n_sentences": n_sent, "gemma_types": len(g_types), "llama_types": len(l_types),
                      "shards_used": src.n_shards_used})
        print(f"[bg] chunk {c}: {n_sent} sentences, gemma {len(g_types)}, llama {len(l_types)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if len(g_types) >= GEMMA_N:
            break
    sent_f.close()
    write_json(DATA / "bg_wiki_gemma_types.json", sorted(g_types), indent=None)
    matched = len(g_types) == GEMMA_N
    print(f"[bg] gemma types {len(g_types)} vs stored {GEMMA_N}: {'MATCH' if matched else 'MISMATCH'}", flush=True)
    ids = sorted(l_types)
    emb = wc.Embeddings("llama")
    try:
        t1 = time.time()
        st = wc.chunked_stats(emb, ids)
        st["n_tokens"] = len(ids)
        torch.save(st, DATA / "bg_wiki_llama_stats.pt")
        print(f"[bg] llama stats n={len(ids)} ({time.time() - t1:.0f}s)", flush=True)
        perm = random.Random(42).sample(ids, len(ids))
        h = len(perm) // 2
        sines = wc.principal_angle_sines_top(wc.chunked_stats(emb, perm[:h]), wc.chunked_stats(emb, perm[h:2 * h]), 100)
    finally:
        emb.free()
    write_json(DATA / "bg_wiki_llama_types.json", ids, indent=None)
    write_json(DATA / "bg_wiki_llama_meta.json", {
        "n_sentences": n_sent, "n_chunks": len(curve), "gemma_types": len(g_types), "gemma_types_stored": GEMMA_N,
        "gemma_match": matched, "llama_types": len(ids), "eps_rel": 1e-4, "curve": curve,
        "llama_split_half_top100_mean_sine": float(sines.mean()), "llama_split_half_top100_max_sine": float(sines.max()),
        "source": "experiments/whitening/background.py SentenceSource (wikimedia/wikipedia 20231101.en, seed 42)",
        "secs": time.time() - t0})
    print(f"[bg] done; split-half mean sine {float(sines.mean()):.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
