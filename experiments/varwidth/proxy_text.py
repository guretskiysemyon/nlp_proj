#!/usr/bin/env python
"""Token-count tables for the ladder's text proxies (PREREG): held-out concept text and general text.

    python -m experiments.varwidth.proxy_text [--wiki-articles 3000]

-> results/varwidth/layers/text/<name>.json = {"n_tokens": N, "n_units": U, "counts": {tid: occurrences}} (gemma
   tokenizer, add_special_tokens=False):
   heldout_<slug>: paper concepts (data/mc_questions.json): every concept MC question (QA_train + QA_test) + its correct answer (the
       erasure eval's own text, never seen by the fit); other concepts: the page sentences not among the 300 sampled
       (experiments/whitening/data/pages/<slug>.json minus data/concept_sentences_extra.json). n_units < 50 is flagged.
   general_eval: MMLU (the 1050 cached questions + choices) and the 805 AlpacaEval instructions (the specificity and
       coherence evals).
   general_wiki: --wiki-articles random articles (seed 0) of wikimedia/wikipedia 20231101.en shard 40 (the last;
       whitening drew shards in a seeded order from the start), skipping any article whose title is one of the
       38 concepts.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from experiments.varwidth.common import RESULTS, ROOT, slug

OUT = RESULTS / "layers" / "text"


def count(tok, texts) -> dict:
    c, n = Counter(), 0
    for ids in tok(list(texts), add_special_tokens=False)["input_ids"]:
        c.update(ids)
        n += len(ids)
    return {"n_tokens": n, "n_units": len(texts), "counts": {str(k): v for k, v in c.items()}}


def heldout_texts(concept: str) -> list:
    mc = json.loads((ROOT / "data" / "mc_questions.json").read_text())
    if concept in mc:
        return [f"{o['q']} {o['correct_answer']}" for key, items in mc[concept].items() if key.startswith("QA_") for o in items
                if isinstance(o, dict) and o.get("q")]
    ex = {e["concept"]: set(e["sentences"]) for e in json.loads((ROOT / "data" / "concept_sentences_extra.json").read_text())}
    page = json.loads((ROOT / "experiments" / "whitening" / "data" / "pages" / f"{slug(concept)}.json").read_text())
    return [s for s in page["sentences"] if s not in ex[concept]]


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from experiments.varwidth.judge_report import concepts
    from experiments.varwidth.prep import get_tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki-articles", type=int, default=3000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer()
    cs = concepts()
    for c in cs:
        p = OUT / f"heldout_{slug(c)}.json"
        if not p.exists():
            t = heldout_texts(c)
            p.write_text(json.dumps(count(tok, t)))
            print(f"{c}: {len(t)} held-out units{'  (< 50: flagged)' if len(t) < 50 else ''}")
    p = OUT / "general_eval.json"
    if not p.exists():
        from ember.evals.alpaca import load_alpaca_eval_local
        mm = json.loads((ROOT / "data" / "mmlu_cache.json").read_text())
        texts = [" ".join([m["question"], *m["choices"]]) for m in mm]
        texts += [ex["instruction"] for ex in load_alpaca_eval_local() if "instruction" in ex]
        p.write_text(json.dumps(count(tok, texts)))
        print(f"general_eval: {len(texts)} units")
    p = OUT / "general_wiki.json"
    if not p.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        f = hf_hub_download("wikimedia/wikipedia", "20231101.en/train-00040-of-00041.parquet", repo_type="dataset")
        t = pq.read_table(f, columns=["title", "text"])
        titles, texts = t.column("title").to_pylist(), t.column("text").to_pylist()
        skip = {c.casefold() for c in cs}
        idx = [i for i in range(len(titles)) if titles[i].casefold() not in skip]
        pick = random.Random(0).sample(idx, a.wiki_articles)
        p.write_text(json.dumps(count(tok, [texts[i] for i in pick])))
        print(f"general_wiki: {a.wiki_articles} articles of {len(titles)}")


if __name__ == "__main__":
    main()
