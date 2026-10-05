#!/usr/bin/env python3
"""Negative-control sensitivity check for a local judge.

The pristine baseline is nearly all-2 under Gemini (148/150 instruct, 94/150
fluency), so agreeing with Gemini there cannot show that a judge can tell good
text from damaged text -- and the erasure grid is exactly about damaged text.
This script builds known-bad variants of the SAME Gemini-scored items and checks
that the judge's score drops:

  instruct axis : completion paired with a DIFFERENT item's instruction
                  ("mismatch") -> should score 0 (unrelated).
  fluency axis  : words of the completion shuffled ("scramble") and the
                  completion replaced by its first sentence looped
                  ("repeat") -> should score 0 (incomprehensible / repetitive).

Each corrupted item is paired with its own clean version judged in the same run,
so the report is a paired comparison: mean clean vs mean corrupted score and
the fraction of items whose score strictly dropped. Prompts and parser are the
project's own, same as agreement_study.py (original fluency prompt by default).

Usage:
    python -m judge_rerun.judge_sensitivity --judge-model google/gemma-4-12B-it
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ember.evals.gemini import GeminiEvaluator
from judge_rerun.pass2_judge_baseline import _judge_id
from judge_rerun.qwen_judge import ALPACA_FLUENCY_PROMPT_QWEN, QwenJudge, parse_rating


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--judge-model", default="google/gemma-4-12B-it")
    p.add_argument("--model-name", default="google_gemma-2-2b-it")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n-items", type=int, default=30,
                    help="items per control (the same first-N Gemini-scored items are used for all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fluency-prompt", choices=("original", "qwen_strengthened"), default="original")
    p.add_argument("--judge-batch-size", type=int, default=4)
    p.add_argument("--out-path", default=None)
    return p.parse_args()


def scramble(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def repeat_first_sentence(text: str, target_chars: int) -> str:
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0].strip() or text[:80]
    out = first
    while len(out) < target_chars:
        out += " " + first
    return out[:target_chars]


def safe_parse(resp: str) -> Optional[int]:
    try:
        return parse_rating(resp)
    except ValueError:
        return None


def paired_summary(clean: List[Optional[int]], bad: List[Optional[int]]) -> Dict[str, Any]:
    pairs = [(c, b) for c, b in zip(clean, bad) if c is not None and b is not None]
    n = len(pairs)
    return {
        "n_pairs": n,
        "n_parse_failed_clean": sum(c is None for c in clean),
        "n_parse_failed_corrupted": sum(b is None for b in bad),
        "mean_clean": sum(c for c, _ in pairs) / n if n else None,
        "mean_corrupted": sum(b for _, b in pairs) / n if n else None,
        "frac_dropped": sum(b < c for c, b in pairs) / n if n else None,
        "frac_corrupted_scored_0": sum(b == 0 for _, b in pairs) / n if n else None,
        "frac_corrupted_scored_2": sum(b == 2 for _, b in pairs) / n if n else None,
    }


def main() -> None:
    args = parse_args()
    samples_path = (ROOT_DIR / "archive_gemini" / "data" / "baseline_samples"
                    / f"samples_alpaca_{args.split}_{args.model_name}.json")
    items = json.loads(samples_path.read_text())[:args.n_items]
    n = len(items)
    rng = random.Random(args.seed)

    # mismatch: shift instructions by a fixed offset coprime-ish to n so no item keeps its own
    shift = n // 2 + 1
    mismatched_instr = [items[(i + shift) % n]["instruction"] for i in range(n)]
    scrambled = [scramble(it["completion"], rng) for it in items]
    repeated = [repeat_first_sentence(it["completion"], len(it["completion"])) for it in items]

    fluency_template = (GeminiEvaluator.ALPACA_FLUENCY_PROMPT if args.fluency_prompt == "original"
                        else ALPACA_FLUENCY_PROMPT_QWEN)
    instruct_fmt = GeminiEvaluator.ALPACA_INSTRUCT_PROMPT

    conditions: Dict[str, List[str]] = {
        "instruct_clean": [instruct_fmt.format(instructions=it["instruction"], completion=it["completion"])
                           for it in items],
        "instruct_mismatch": [instruct_fmt.format(instructions=mismatched_instr[i],
                                                   completion=items[i]["completion"])
                              for i in range(n)],
        "fluency_clean": [fluency_template.format(completion=it["completion"]) for it in items],
        "fluency_scramble": [fluency_template.format(completion=c) for c in scrambled],
        "fluency_repeat": [fluency_template.format(completion=c) for c in repeated],
    }

    judge_id = _judge_id(args.judge_model)
    ckpt_dir = ROOT_DIR / "judge_rerun" / "agreement_ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    t0 = time.time()
    judge = QwenJudge(model_id=args.judge_model)
    print(f"judge loaded: {args.judge_model} ({time.time() - t0:.0f}s)", flush=True)

    responses: Dict[str, List[str]] = {}
    for name, prompts in conditions.items():
        ckpt = ckpt_dir / f"sensitivity__{judge_id}__{args.fluency_prompt}__{args.split}_n{n}_s{args.seed}__{name}.json"
        responses[name] = judge.generate_batch(prompts, batch_size=args.judge_batch_size,
                                               checkpoint_path=ckpt)
    scores = {k: [safe_parse(r) for r in v] for k, v in responses.items()}

    summary = {
        "judge_model": args.judge_model, "fluency_prompt": args.fluency_prompt, "n_items": n,
        "instruct_mismatch": paired_summary(scores["instruct_clean"], scores["instruct_mismatch"]),
        "fluency_scramble": paired_summary(scores["fluency_clean"], scores["fluency_scramble"]),
        "fluency_repeat": paired_summary(scores["fluency_clean"], scores["fluency_repeat"]),
    }
    per_item = []
    for i, it in enumerate(items):
        row = {"idx": i, "instruction": it["instruction"], "completion": it["completion"],
               "gemini_instruct_score": it["instruct_score"], "gemini_fluency_score": it["fluency_score"],
               "mismatched_instruction": mismatched_instr[i], "scrambled": scrambled[i],
               "repeated": repeated[i]}
        for k in conditions:
            row[f"{k}_score"] = scores[k][i]
            row[f"{k}_response"] = responses[k][i]
        per_item.append(row)

    out_path = Path(args.out_path) if args.out_path else (
        ROOT_DIR / "judge_rerun" / f"judge_sensitivity_{args.split}__{judge_id}.json")
    out_path.write_text(json.dumps({"summary": summary, "per_item": per_item},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    for k in ("instruct_mismatch", "fluency_scramble", "fluency_repeat"):
        s = summary[k]
        print(f"[{k}] n={s['n_pairs']} clean={s['mean_clean']:.2f} corrupted={s['mean_corrupted']:.2f} "
              f"dropped={s['frac_dropped']:.0%} scored0={s['frac_corrupted_scored_0']:.0%} "
              f"scored2={s['frac_corrupted_scored_2']:.0%} "
              f"(parse fails clean/corrupted: {s['n_parse_failed_clean']}/{s['n_parse_failed_corrupted']})",
              flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
