#!/usr/bin/env python3
"""Multilingual whitening background ("wiki_ml") for the multilingual notebooks (8 languages, splitting.LANGS).

Generalises background.py (Instructions.txt, "Sampling recipe") the way V* itself is built for the multilingual flow:
the same number of sentences per language, token types pooled over languages and each counted once.

1. Sentences per language: wikimedia/wikipedia 20231101.<lang> parquet shards (hf_hub_download, HF cache), shards in a
   seeded order; articles of a shard in a seeded permutation, at most PER_ARTICLE random prose sentences each
   (native_neutral.prose_paragraphs / looks_like_prose and splitting.split_sentences, the filters of the native neutral
   sets). Thai: at most 3 random prose paragraphs per article are split (pythainlp crfcut is slow).
   English reuses data/wiki_sentences.jsonl (background.py, same rule).
   Fallback without network: multilingual_hp/data/dump_ext pages (1,500 per language).
   Cached in data/ml/wiki_sentences_<lang>.jsonl.
2. Rounds: round r adds CHUNK more sentences in every language. After each round the pooled types are split in half
   at random and the two halves' top-TOP_K eigenvector subspaces compared (mean principal-angle sine). Sampling stops
   when mean sine <= SINE_TOL and n_types >= MIN_TYPES, or at MAX_PER_LANG sentences per language.
3. Output (per model): data/<model>/bg_wiki_ml_stats.pt (mu, lam, Q, n), bg_wiki_ml_types.json,
   bg_wiki_ml_report.json (stability curve, types per language, shares). "wiki_en" is the existing bg_wiki_stats.pt.

Usage: python experiments/whitening/background_ml.py [--model llama] [--max-per-lang 40000] [--chunk 5000]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from common import (DATA, ROOT, SEED, Embeddings, chunked_stats, excluded_ids, load_env, log, model_dir,
                    principal_angle_sines_top, read_json, tokenizer, write_json)

ML_DIR = DATA / "ml"
EN_SENTS = DATA / "wiki_sentences.jsonl"
DATASET = "wikimedia/wikipedia"
PER_ARTICLE = 4
TOP_K = 100
SINE_TOL = 0.1
MIN_TYPES = 60_000
TH_MAX_PARAS = 3
DUMP_EXT = ROOT / "experiments/multilingual_hp/data/dump_ext"


def langs() -> list[str]:
    from splitting import LANGS
    return list(LANGS)


# ---------------------------------------------------------------- per-language sentence streams
def article_sentences(text: str, lang: str, key: str) -> list[str]:
    from native_neutral import looks_like_prose, prose_paragraphs
    from splitting import split_sentences
    rng = random.Random(key)
    paras = prose_paragraphs(text or "", lang)
    if lang == "th" and len(paras) > TH_MAX_PARAS:
        paras = rng.sample(paras, TH_MAX_PARAS)
    sents = [s for s in split_sentences(paras, lang) if looks_like_prose(s, lang)]
    return rng.sample(sents, PER_ARTICLE) if len(sents) > PER_ARTICLE else sents


def shard_files(lang: str) -> list[str]:
    from huggingface_hub import HfApi
    load_env()
    files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(DATASET, repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"20231101.{lang}/") and f.endswith(".parquet"))


def download(fname: str) -> Path:
    from huggingface_hub import hf_hub_download
    load_env()
    for attempt in range(8):
        try:
            return Path(hf_hub_download(DATASET, fname, repo_type="dataset", token=os.environ.get("HF_TOKEN")))
        except Exception as e:  # noqa: BLE001
            wait = min(300, 10 * 2 ** attempt)
            log(f"[wiki_ml] {type(e).__name__} on {fname}: {str(e)[:120]}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download failed: {fname}")


class LangStream:
    """Deterministic sentence stream for one language, cached on disk (append-only, prefix-stable)."""

    def __init__(self, lang: str):
        self.lang = lang
        self.path = ML_DIR / f"wiki_sentences_{lang}.jsonl"
        self.sents: list[str] = []
        if lang == "en":
            with EN_SENTS.open(encoding="utf-8") as f:
                self.sents = [json.loads(line)["sentence"] for line in f]
            return
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                self.sents = [json.loads(line)["sentence"] for line in f]
        self._articles = None  # iterator over (article_id, title, text) in the seeded order

    def _article_iter(self):
        import pyarrow.parquet as pq
        try:
            files = shard_files(self.lang)
        except Exception as e:  # noqa: BLE001
            log(f"[wiki_ml/{self.lang}] cannot list shards ({e}); using the local dump_ext pages")
            files = []
        order = random.Random(f"{SEED}-{self.lang}").sample(files, len(files))
        if not order:
            pages = [json.loads(l) for l in (DUMP_EXT / f"wiki_dump_pages_{self.lang}.jsonl").open(encoding="utf-8")]
            random.Random(SEED).shuffle(pages)
            for p in pages:
                yield p["id"], p["title"], p["text"], "dump_ext"
            return
        for fname in order:
            pf = pq.ParquetFile(download(fname))
            t = pf.read(columns=["id", "title", "text"])
            ids, titles, texts = (t.column(c).to_pylist() for c in ("id", "title", "text"))
            del t
            perm = random.Random(f"{SEED}-{fname}").sample(range(len(ids)), len(ids))
            log(f"[wiki_ml/{self.lang}] shard {fname}: {len(ids)} articles")
            for j in perm:
                yield ids[j], titles[j], texts[j], fname

    def ensure(self, n: int) -> bool:
        """Make the first n sentences available; False if the source ran out first."""
        if len(self.sents) >= n:
            return True
        if self.lang == "en":
            return False
        if self._articles is None:
            self._articles = self._article_iter()
            seen = len(self.sents)
            if seen:  # replay the stream to the stored position (articles are processed in the same order)
                skipped = 0
                while skipped < seen:
                    aid, title, text, src = next(self._articles)
                    skipped += len(article_sentences(text, self.lang, f"{SEED}-{self.lang}-{aid}"))
        ML_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with self.path.open("a", encoding="utf-8") as f:
            while len(self.sents) < n:
                try:
                    aid, title, text, src = next(self._articles)
                except StopIteration:
                    return False
                for s in article_sentences(text, self.lang, f"{SEED}-{self.lang}-{aid}"):
                    f.write(json.dumps({"article_id": aid, "title": title, "source": src, "sentence": s},
                                       ensure_ascii=False) + "\n")
                    self.sents.append(s)
        log(f"[wiki_ml/{self.lang}] {len(self.sents)} sentences ({time.time() - t0:.0f}s)")
        return True


# ---------------------------------------------------------------- types and stability
def types_of(model: str, sents: list[str]) -> set[int]:
    tok, ex = tokenizer(model), excluded_ids(model)
    ids: set[int] = set()
    for i in range(0, len(sents), 2000):
        for seq in tok(sents[i:i + 2000], add_special_tokens=False)["input_ids"]:
            ids.update(seq)
    return {int(i) for i in ids if i not in ex}


def run(model: str, max_per_lang: int, chunk: int):
    md = model_dir(model)
    report_path = md / "bg_wiki_ml_report.json"
    report = read_json(report_path) if report_path.exists() else {"curve": []}
    L = langs()
    streams = {l: LangStream(l) for l in L}
    emb = Embeddings(model)
    per_lang: dict[str, set[int]] = {l: set() for l in L}
    n, stop = 0, None
    try:
        while n < max_per_lang:
            n = min(max_per_lang, n + chunk)
            short = [l for l in L if not streams[l].ensure(n)]
            if short:
                log(f"[wiki_ml] sources exhausted for {short} at {n} sentences per language")
            for l in L:
                per_lang[l] = types_of(model, streams[l].sents[:n])
            pooled = sorted(set().union(*per_lang.values()))
            perm = random.Random(SEED).sample(pooled, len(pooled))
            h = len(perm) // 2
            t0 = time.time()
            sines = principal_angle_sines_top(chunked_stats(emb, perm[:h]), chunked_stats(emb, perm[h:2 * h]), TOP_K)
            p = {"n_per_lang": n, "n_types": len(pooled), "mean_sine": float(sines.mean()),
                 "max_sine": float(sines.max()), "types_per_lang": {l: len(per_lang[l]) for l in L}}
            report["curve"] = [q for q in report["curve"] if q["n_per_lang"] != n] + [p]
            write_json(report_path, report)
            log(f"[wiki_ml/{model}] {n}/lang: types={len(pooled)} mean_sine={p['mean_sine']:.3f} "
                f"({time.time() - t0:.0f}s)")
            if p["mean_sine"] <= SINE_TOL and len(pooled) >= MIN_TYPES:
                stop = "stability"
                break
            if short:
                stop = "exhausted"
                break
        stop = stop or "cap"
        pooled = sorted(set().union(*per_lang.values()))
        stats = chunked_stats(emb, pooled)
        torch.save({**stats, "n_tokens": len(pooled), "ids_file": "bg_wiki_ml_types.json", "langs": L,
                    "n_per_lang": n}, md / "bg_wiki_ml_stats.pt")
        write_json(md / "bg_wiki_ml_types.json", pooled, indent=None)
        only = {l: len(per_lang[l] - set().union(*(per_lang[o] for o in L if o != l))) for l in L}
        report.update({"stop": stop, "n_per_lang": n, "n_types": len(pooled), "langs": L,
                       "types_per_lang": {l: len(per_lang[l]) for l in L}, "types_only_in_lang": only,
                       "chunk": chunk, "max_per_lang": max_per_lang, "per_article": PER_ARTICLE, "seed": SEED})
        write_json(report_path, report)
        log(f"[wiki_ml/{model}] done ({stop}): {n}/lang, {len(pooled)} types -> {md / 'bg_wiki_ml_stats.pt'}")
    finally:
        emb.free()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama", choices=["gemma", "llama"])
    ap.add_argument("--max-per-lang", type=int, default=40_000)
    ap.add_argument("--chunk", type=int, default=5_000)
    args = ap.parse_args()
    run(args.model, args.max_per_lang, args.chunk)


if __name__ == "__main__":
    main()
