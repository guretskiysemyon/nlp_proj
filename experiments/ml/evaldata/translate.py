#!/usr/bin/env python
"""Translate English segments with NLLB-200 3.3B (all) or MADLAD-400 3B (only the low ones).

Resumable: every finished batch is appended to results/ml/data/qa/work/<engine>_<lang>.jsonl
({"src", "tr"}); a restart skips segments already there. The model is loaded once and
all languages are processed in the same job.

    python -m experiments.ml.evaldata.translate --engine nllb   [--langs it pt hi th] [--phases prio mmlu rest ctx bt]
    python -m experiments.ml.evaldata.translate --engine madlad [--langs ...]

nllb:   facebook/nllb-200-3.3B, fp16 on one GPU, beam 4. Order: the priority concepts
        (tasks.PRIORITY_CONCEPTS) in all languages, then the Thai MMLU items, then the rest;
        then every short option list of an MC item as one line "a; b; c; d" (ctx, gives single
        words their context; work/ctx_<lang>.jsonl); optionally the round-trip back-translation of every NLLB output into English
        (work/bt_nllb_<lang>.jsonl {"src", "tr", "bt"}), used for the chrF quality column.
madlad: google/madlad400-3b-mt, fp32 split over >= 2 visible GPUs, else fp16 on one; beam 4; segments flagged by common.needs_retranslation (lost number, or low
        Kiwi / LaBSE).
Set NLLB_PATH / MADLAD_PATH to a local snapshot directory to load from fast local disk.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Callable, List

import torch

from experiments.ml.evaldata.common import (MT_LANGS, candidates, NLLB_CODE, WORK, append_jsonl, cache_map, load_jsonl,
                                            needs_retranslation, normalize_digits)
from experiments.ml.evaldata.tasks import (PRIORITY_CONCEPTS, mc_strings, mmlu_strings, option_lists, segments_for,
                                           segments_of)

NLLB = os.environ.get("NLLB_PATH", "facebook/nllb-200-3.3B")
MADLAD = os.environ.get("MADLAD_PATH", "google/madlad400-3b-mt")


def madlad_todo(lang: str) -> List[str]:
    nllb = cache_map(WORK / f"nllb_{lang}.jsonl")
    ctx = candidates("ctx", lang)
    todo = []
    for s in segments_for(lang):
        if s not in nllb:
            raise SystemExit(f"{lang}: NLLB translation missing for {s!r}; run NLLB first")
        flag = needs_retranslation(lang, s, nllb[s], ctx.get(s, ()))
        if flag is None:
            raise SystemExit(f"{lang}: no quality score for {s!r}; run score.py --engine nllb first")
        if flag:
            todo.append(s)
    return todo


class Engine:
    def __init__(self, name: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.name = name
        t0 = time.time()
        if name == "nllb":
            self.tok = AutoTokenizer.from_pretrained(NLLB, src_lang="eng_Latn")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(NLLB, torch_dtype=torch.float16,
                                                               low_cpu_mem_usage=True).cuda().eval()
        else:
            n = torch.cuda.device_count()
            self.tok = AutoTokenizer.from_pretrained(MADLAD)
            if n >= 2:  # fp32 split over the GPUs
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    MADLAD, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="auto",
                    max_memory={i: "10GiB" for i in range(n)}).eval()
            else:  # one 12 GB GPU: fp16 (HF T5 clamps fp16 overflows; bad outputs lose on LaBSE anyway)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    MADLAD, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda().eval()
        print(f"[{name}] loaded from {NLLB if name == 'nllb' else MADLAD} in {time.time() - t0:.0f}s", flush=True)

    def n_tokens(self, s: str) -> int:
        return len(self.tok(s)["input_ids"])

    @torch.no_grad()
    def translate(self, texts: List[str], lang: str, to_english: bool = False) -> List[str]:
        if self.name == "nllb":
            self.tok.src_lang = NLLB_CODE[lang] if to_english else "eng_Latn"
            tgt = "eng_Latn" if to_english else NLLB_CODE[lang]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            kw = {"forced_bos_token_id": self.tok.convert_tokens_to_ids(tgt)}
        else:
            assert not to_english
            enc = self.tok([f"<2{lang}> {t}" for t in texts], return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
            kw = {}
        dev = next(self.model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        n_in = int(enc["attention_mask"].sum(1).max())
        out = self.model.generate(**enc, num_beams=4, max_new_tokens=min(512, 2 * n_in + 16), **kw)
        return [normalize_digits(t.strip()) for t in self.tok.batch_decode(out, skip_special_tokens=True)]


def run_batches(eng: Engine, texts: List[str], fn: Callable[[List[str]], List[str]],
                write: Callable[[List[str], List[str]], None], tag: str, tok_budget: int, max_bs: int) -> None:
    """Length-sorted batches of ``texts`` through ``fn``; each batch is written before the next."""
    print(f"[{tag}] {len(texts)} to go", flush=True)
    if not texts:
        return
    lens = {s: eng.n_tokens(s) for s in texts}
    texts = sorted(texts, key=lambda s: lens[s])
    t0, n, i, last = time.time(), 0, 0, 0
    while i < len(texts):
        j = i + 1
        while j < len(texts) and j - i < max_bs and (j - i + 1) * lens[texts[j]] <= tok_budget:
            j += 1
        batch = texts[i:j]
        try:
            outs = fn(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            max_bs = max(1, len(batch) // 2)
            print(f"  OOM at bs={len(batch)}, max_bs -> {max_bs}", flush=True)
            continue
        write(batch, outs)
        n += len(batch)
        i = j
        if n - last >= 500 or i == len(texts):
            last = n
            print(f"  [{tag}] {n}/{len(texts)}  {n / (time.time() - t0):.1f} seg/s  len={lens[batch[-1]]}",
                  flush=True)


def forward(eng: Engine, lang: str, todo: List[str], args) -> None:
    path = WORK / f"{eng.name}_{lang}.jsonl"
    done = cache_map(path)
    run_batches(eng, [s for s in dict.fromkeys(todo) if s not in done],
                lambda b: eng.translate(b, lang),
                lambda b, o: append_jsonl(path, [{"src": s, "tr": t} for s, t in zip(b, o)]),
                f"{eng.name}/{lang}", args.tok_budget, args.max_bs)


def back(eng: Engine, lang: str, args) -> None:
    """NLLB round trip lang -> English of every cached NLLB translation."""
    fwd = load_jsonl(WORK / f"nllb_{lang}.jsonl")
    path = WORK / f"bt_nllb_{lang}.jsonl"
    done = {(r["src"], r["tr"]) for r in load_jsonl(path)}
    by_tr = {}
    for r in fwd:
        if (r["src"], r["tr"]) not in done:
            by_tr.setdefault(r["tr"], []).append(r["src"])

    def write(batch, outs):
        append_jsonl(path, [{"src": s, "tr": t, "bt": b} for t, b in zip(batch, outs) for s in by_tr[t]])

    run_batches(eng, list(by_tr), lambda b: eng.translate(b, lang, to_english=True), write,
                f"bt/{lang}", args.tok_budget, args.max_bs)


def rebuild() -> None:
    """Write whatever is complete into the final files (build.py --partial --no-report)."""
    import sys
    from experiments.ml.evaldata import build
    argv, sys.argv = sys.argv, ["build", "--partial", "--no-report"]
    try:
        build.main()
    except Exception as e:  # never lose the translation job over a build problem
        print(f"rebuild failed: {e!r}", flush=True)
    finally:
        sys.argv = argv


def context(eng: Engine, lang: str, args) -> None:
    """Translate each short option list as one line; work/ctx_<lang>.jsonl {"src", "tr"} (joined)."""
    path = WORK / f"ctx_{lang}.jsonl"
    done = cache_map(path)
    run_batches(eng, [s for s in option_lists(lang) if s not in done],
                lambda b: eng.translate(b, lang),
                lambda b, o: append_jsonl(path, [{"src": s, "tr": t} for s, t in zip(b, o)]),
                f"ctx/{lang}", args.tok_budget, args.max_bs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["nllb", "madlad"], required=True)
    ap.add_argument("--langs", nargs="+", default=list(MT_LANGS))
    ap.add_argument("--tok-budget", type=int, default=1536, help="max batch_size * input length")
    ap.add_argument("--max-bs", type=int, default=64)
    ap.add_argument("--phases", nargs="+", default=["prio", "mmlu", "rest", "ctx"],
                    choices=["prio", "mmlu", "rest", "ctx", "bt"],
                    help="nllb: priority concepts, Thai MMLU, remaining segments, option lists in context "
                         "(ctx), round-trip back-translation (bt, off by default: as slow as the forward pass)")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.engine == "nllb":
        # priority concepts (all langs) -> Thai MMLU -> everything else; the cache skips repeats
        prio = segments_of(mc_strings(PRIORITY_CONCEPTS))
        mmlu = segments_of(mmlu_strings())
        phases = []
        if "prio" in args.phases:
            phases += [(l, prio) for l in args.langs]
        if "mmlu" in args.phases and "th" in args.langs:
            phases.append(("th", mmlu))
        if "rest" in args.phases:
            for l in args.langs:  # what the earlier phases of THIS language already covered
                skip = set(prio) if "prio" in args.phases else set()
                skip |= set(mmlu) if (l == "th" and "mmlu" in args.phases) else set()
                phases.append((l, [s for s in segments_for(l) if s not in skip]))
    else:
        prio = mmlu = None
        phases = [(l, madlad_todo(l)) for l in args.langs]
    for l, t in phases:
        print(f"phase {l}: {len(t)} segments", flush=True)
    eng = Engine(args.engine)
    n_early = sum(1 for l, t in phases if t is prio or (args.engine == "nllb" and t is mmlu))
    for k, (l, t) in enumerate(phases):
        forward(eng, l, list(t), args)
        if args.engine == "nllb" and k == n_early - 1:
            rebuild()  # priority concepts + Thai MMLU available before the long tail
    print("FORWARD DONE", flush=True)
    if args.engine == "nllb":
        rebuild()
    if args.engine == "nllb" and "ctx" in args.phases:
        for l in args.langs:
            context(eng, l, args)
    if args.engine == "nllb" and "bt" in args.phases:
        for l in args.langs:
            back(eng, l, args)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
