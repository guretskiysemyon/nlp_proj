"""NLLB-200 top-up translations for concept pages shorter than N_FULL usable sentences.

    tasks   (CPU) per (concept, lang != en) the deficit N_FULL - min(N_FULL, native pool) (N_FULL if no page);
            sources = sentences of the English concept page (pools/<slug>/en.json) that are NOT among EMBER's 300
            English concept sentences (the English set), in a seeded order (random.Random(SEED), one order per
            concept, shared by all languages); EMBER's own sentences are appended last as a fallback.
            deficit * (1 + SURPLUS) + 5 sources are queued, so QA failures can be skipped.
            -> translations/tasks.json  {lang: {slug: [english sentence, ...]}}
    run     (GPU) facebook/nllb-200-3.3B, fp16, beam 4; resumable cache translations/nllb_<lang>.jsonl, one line
            {"src","tgt"} per unique English sentence (already cached sentences are skipped).

    python -m experiments.ml.data.translate_nllb --stage tasks
    python -m experiments.ml.data.translate_nllb --stage run     (SLURM: experiments/ml/data/translate.sbatch)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

from experiments.ml.data.common import LANGS, N_FULL, POOLS, REPO, SEED, TRANS, concepts, dump_json, load_json, slug

MODEL = "facebook/nllb-200-3.3B"
NLLB_CODE = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "it": "ita_Latn", "pt": "por_Latn",
             "hi": "hin_Deva", "es": "spa_Latn", "th": "tha_Thai"}
SURPLUS = 0.15
TASKS_PATH = TRANS / "tasks.json"


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s).strip().casefold()


def ember_en():
    return {e["concept"]: e["sentences"] for e in load_json(REPO / "data" / "concept_sentences.json")}


def english_sources(concept: str, ember):
    used = {norm(s) for s in ember[concept]}
    pool = load_json(POOLS / slug(concept) / "en.json")["sentences"]
    fresh = [s for s in pool if norm(s) not in used]
    random.Random(SEED).shuffle(fresh)
    rest = list(ember[concept])
    random.Random(SEED).shuffle(rest)
    return fresh, rest


def native_count(concept: str, lang: str) -> int:
    p = POOLS / slug(concept) / f"{lang}.json"
    return len(load_json(p)["sentences"]) if p.exists() else 0


def make_tasks():
    ember = ember_en()
    tasks = {l: {} for l in LANGS if l != "en"}
    summary = {}
    for c in concepts():
        fresh, rest = english_sources(c, ember)
        srcs = fresh + rest
        for lang in tasks:
            deficit = N_FULL - min(N_FULL, native_count(c, lang))
            if deficit == 0:
                continue
            k = min(len(srcs), int(deficit * (1 + SURPLUS)) + 5)
            tasks[lang][slug(c)] = srcs[:k]
            summary.setdefault(c, {})[lang] = {"deficit": deficit, "queued": k, "fresh_en": len(fresh)}
    dump_json(TASKS_PATH, tasks)
    n_unique = {l: len({s for v in d.values() for s in v}) for l, d in tasks.items()}
    print(json.dumps(summary, indent=1))
    print("unique sources per language:", n_unique, "total", sum(n_unique.values()))


def cache_path(lang: str, worker: str = ""):
    return TRANS / (f"nllb_{lang}.{worker}.jsonl" if worker else f"nllb_{lang}.jsonl")


def load_cache(lang: str):
    """All cache files of a language (nllb_<lang>.jsonl and per-worker nllb_<lang>.<worker>.jsonl)."""
    out = {}
    for p in sorted(TRANS.glob(f"nllb_{lang}.*jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["src"]] = r["tgt"]
    return out


def run(batch_tokens: int, beams: int, first=(), model_path=MODEL, only_first=False, langs=None,
        skip_first=False, worker="", max_batch=16):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tasks = load_json(TASKS_PATH)
    first_slugs = [slug(c) for c in first]
    phases = [first_slugs] if only_first else ([None] if skip_first else [first_slugs, None])
    todo = []  # (lang, sources): the --first concepts in every language, then everything else
    tasks = {l: d for l, d in tasks.items() if langs is None or l in langs}
    seen = {lang: set(load_cache(lang)) for lang in tasks}
    if skip_first:  # another worker translates the --first concepts
        for lang, d in tasks.items():
            seen[lang].update(s for k in first_slugs for s in d.get(k, []))
    for phase in phases:
        for lang, d in tasks.items():
            srcs = sorted({s for k, v in d.items() if phase is None or k in phase for s in v} - seen[lang], key=len)
            seen[lang].update(srcs)
            todo.append((lang, srcs))
    print("to translate:", [(l, len(v)) for l, v in todo], flush=True)
    if not any(v for _, v in todo):
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path, src_lang=NLLB_CODE["en"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = model.cuda().eval()
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    for lang, srcs in todo:
        if not srcs:
            continue
        bos = tok.convert_tokens_to_ids(NLLB_CODE[lang])
        t1, done = time.time(), 0
        with cache_path(lang, worker).open("a", encoding="utf-8") as fh:
            i = 0
            while i < len(srcs):
                n_tok = len(tok(srcs[i])["input_ids"])
                bs = max(1, min(max_batch, batch_tokens // max(n_tok, 1)))
                while True:
                    batch = srcs[i:i + bs]
                    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=400).to("cuda")
                    try:
                        with torch.no_grad():
                            out = model.generate(**enc, forced_bos_token_id=bos, num_beams=beams,
                                                 max_new_tokens=int(enc["input_ids"].shape[1] * 2 + 20))
                        break
                    except torch.OutOfMemoryError:
                        del enc
                        torch.cuda.empty_cache()
                        if bs == 1:
                            raise
                        bs = max(1, bs // 2)
                        print(f"[{lang}] OOM, batch -> {bs}", flush=True)
                for s, t in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
                    fh.write(json.dumps({"src": s, "tgt": t.strip(), "beams": beams}, ensure_ascii=False) + "\n")
                fh.flush()
                i += len(batch)
                done += len(batch)
                if done % 200 < len(batch):
                    print(f"[{lang}] {done}/{len(srcs)} ({time.time() - t1:.0f}s)", flush=True)
        print(f"[{lang}] done {len(srcs)} in {time.time() - t1:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["tasks", "run"])
    ap.add_argument("--batch-tokens", type=int, default=1024)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--first", nargs="*", default=[], help="concepts translated first (all languages)")
    ap.add_argument("--only-first", action="store_true")
    ap.add_argument("--model-path", default=MODEL, help="local snapshot dir (faster than the NFS HF cache)")
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--skip-first", action="store_true", help="leave the --first concepts to another worker")
    ap.add_argument("--worker", default="", help="cache file suffix (one per concurrent worker)")
    ap.add_argument("--max-batch", type=int, default=16)
    args = ap.parse_args()
    if args.stage == "tasks":
        make_tasks()
    else:
        run(args.batch_tokens, args.beams, args.first, args.model_path, args.only_first, args.langs,
            args.skip_first, args.worker, args.max_batch)


if __name__ == "__main__":
    main()
