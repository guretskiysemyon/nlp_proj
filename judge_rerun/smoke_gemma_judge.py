#!/usr/bin/env python3
"""Smoke test for a non-Qwen local judge (built for google/gemma-4-12B-it).

Loads the judge once through judge_rerun.qwen_judge.QwenJudge (the loader is
generic -- AutoModelForCausalLM + chat template + NF4 -- despite the name),
then runs a few of Gemini's already-scored Alpaca records through the
project's own prompts (ember.evals.gemini.GeminiEvaluator) and reports:
  * does the model load / generate sensible text (no NaN/garbage on this GPU),
  * does it follow the "Rating: [[n]]" format (strict vs tolerant parse),
  * response length / seconds per call / peak GPU memory,
  * item-level comparison with Gemini on a handful of items chosen to
    include the rare non-2 scores.

Usage:
    python -m judge_rerun.smoke_gemma_judge [--judge-model google/gemma-4-12B-it]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from ember.evals.gemini import GeminiEvaluator
from judge_rerun.qwen_judge import QwenJudge, parse_rating

STRICT_RE = re.compile(r"Rating:\s*\[\[(\d+)\]\]")


def pick_items(items: list[dict], n_per_cell: int) -> list[dict]:
    """A few items per (gemini_instruct, gemini_fluency) cell, rare cells first."""
    cells: dict[tuple[int, int], list[dict]] = {}
    for it in items:
        cells.setdefault((it["instruct_score"], it["fluency_score"]), []).append(it)
    picked: list[dict] = []
    for key in sorted(cells, key=lambda k: len(cells[k])):
        picked.extend(cells[key][:n_per_cell])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge-model", default="google/gemma-4-12B-it")
    ap.add_argument("--samples-path", default=str(
        ROOT_DIR / "archive_gemini" / "data" / "baseline_samples"
        / "samples_alpaca_train_google_gemma-2-2b-it.json"))
    ap.add_argument("--n-per-cell", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=350)
    ap.add_argument("--show", type=int, default=3, help="how many full responses to print")
    args = ap.parse_args()

    n_gpu = torch.cuda.device_count()
    print(f"torch={torch.__version__} gpus={n_gpu} "
          f"{[torch.cuda.get_device_name(i) for i in range(n_gpu)]}", flush=True)

    items = pick_items(json.loads(Path(args.samples_path).read_text()), args.n_per_cell)
    print(f"{len(items)} items: " + ", ".join(
        f"(i{it['instruct_score']},f{it['fluency_score']})" for it in items), flush=True)

    t0 = time.time()
    judge = QwenJudge(model_id=args.judge_model)
    device_map = getattr(judge.model, "hf_device_map", None)
    device_map_desc = sorted(set(map(str, device_map.values()))) if device_map else "none (single device)"
    print(f"loaded {args.judge_model} in {time.time() - t0:.0f}s; "
          f"device_map spans {device_map_desc}", flush=True)
    print("weights GB per GPU:", [round(torch.cuda.memory_allocated(i) / 1e9, 2) for i in range(n_gpu)],
          flush=True)

    prompts = {
        "instruct": [GeminiEvaluator.ALPACA_INSTRUCT_PROMPT.format(
            instructions=it["instruction"], completion=it["completion"]) for it in items],
        "fluency": [GeminiEvaluator.ALPACA_FLUENCY_PROMPT.format(completion=it["completion"])
                    for it in items],
    }
    results: dict[str, list[str]] = {}
    for axis, ps in prompts.items():
        t0 = time.time()
        results[axis] = judge.generate_batch(ps, max_new_tokens=args.max_new_tokens,
                                             batch_size=args.batch_size)
        dt = time.time() - t0
        print(f"[{axis}] {len(ps)} calls in {dt:.0f}s = {dt / len(ps):.1f}s/call", flush=True)

    print("peak GB per GPU:", [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(n_gpu)],
          flush=True)

    for axis in ("instruct", "fluency"):
        rs = results[axis]
        strict = sum(1 for r in rs if STRICT_RE.search(r))
        tolerant = 0
        pairs = []
        for it, r in zip(items, rs):
            try:
                s = parse_rating(r)
                tolerant += 1
            except ValueError:
                s = None
            pairs.append((it[f"{axis}_score"], s))
        words = [len(r.split()) for r in rs]
        print(f"[{axis}] strict '[[n]]' format {strict}/{len(rs)}; tolerant parse ok {tolerant}/{len(rs)}; "
              f"words/response mean={sum(words) / len(words):.0f} max={max(words)}", flush=True)
        print(f"[{axis}] (gemini, judge) pairs: {pairs}", flush=True)

    print("\n================ sample responses ================", flush=True)
    for axis in ("instruct", "fluency"):
        for i in range(min(args.show, len(items))):
            print(f"\n--- {axis} item {i} (gemini={items[i][f'{axis}_score']}) ---\n{results[axis][i]}", flush=True)


if __name__ == "__main__":
    main()
