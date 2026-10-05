#!/usr/bin/env python3
"""Qwen-vs-Gemini agreement study on Alpaca coherence (judge_swap_plan.md SS6).

Re-judges Gemini's EXACT already-scored completions
(archive_gemini/data/baseline_samples/samples_alpaca_{split}_*.json) with the
local Qwen judge, same prompts, then compares item-by-item against Gemini's
recorded instruct_score/fluency_score. This is the rigorous version of the
earlier mean-score comparison: same text, two judges, no confound from
batched-generation nondeterminism producing slightly different completions.

Gate (per the plan): mean score delta within +-0.15 and a healthy exact-match
rate, per axis, before trusting this judge on the real 8-delta grid.

Usage:
    python -m judge_rerun.agreement_study --split train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from ember.erasure import log
from ember.evals.gemini import GeminiEvaluator
from judge_rerun.pass2_judge_baseline import _judge_id
from judge_rerun.qwen_judge import ALPACA_FLUENCY_PROMPT_QWEN, MODEL_ID, QwenJudge, parse_rating


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--model-name", default="google_gemma-2-2b-it")
    p.add_argument("--judge-model", default=MODEL_ID,
                    help="the local judge model to load (default: the 14B used in the first study)")
    p.add_argument("--fluency-prompt", choices=("original", "qwen_strengthened"),
                    default="qwen_strengthened",
                    help="'original' = verbatim GeminiEvaluator prompt; 'qwen_strengthened' = "
                         "ALPACA_FLUENCY_PROMPT_QWEN (what the 14B study used, so it stays the default)")
    p.add_argument("--judge-batch-size", type=int, default=8)
    p.add_argument("--samples-path", default=None,
                    help="override the archived samples_alpaca_<split>_<model>.json path")
    p.add_argument("--out-path", default=None,
                    help="default: agreement_study_<split>.json for the default 14B judge, "
                         "agreement_study_<split>__<judge_id>.json for any other judge")
    return p.parse_args()


def confusion(gold: List[int], pred: List[int]) -> Dict[str, int]:
    return {f"gold={g}_pred={p}": sum(1 for gg, pp in zip(gold, pred) if gg == g and pp == p)
            for g in (0, 1, 2) for p in (0, 1, 2)}


def summarize(axis: str, gold: List[int], pred: List[int]) -> Dict[str, Any]:
    n = len(gold)
    exact = sum(1 for g, p in zip(gold, pred) if g == p) / n
    mean_gold = sum(gold) / n
    mean_pred = sum(pred) / n
    mae = sum(abs(g - p) for g, p in zip(gold, pred)) / n
    return {
        "axis": axis, "n": n,
        "mean_gold_gemini": mean_gold, "mean_pred_qwen": mean_pred,
        "mean_delta": mean_pred - mean_gold,
        "exact_match_rate": exact, "mean_abs_error": mae,
        "confusion": confusion(gold, pred),
    }


def main() -> None:
    args = parse_args()
    log.configure()
    log.set_context(method="ember", concept="_agreement_study", stage="agreement_study")

    samples_path = Path(args.samples_path) if args.samples_path else (
        ROOT_DIR / "archive_gemini" / "data" / "baseline_samples"
        / f"samples_alpaca_{args.split}_{args.model_name}.json"
    )
    items: List[Dict[str, Any]] = json.loads(samples_path.read_text())
    log.info("loaded %d Gemini-scored records from %s", len(items), samples_path)

    judge_id = _judge_id(args.judge_model)
    fluency_template = (GeminiEvaluator.ALPACA_FLUENCY_PROMPT if args.fluency_prompt == "original"
                        else ALPACA_FLUENCY_PROMPT_QWEN)
    # Per-chunk checkpoints: a slow judge (32B) takes hours per pass and the
    # partition can preempt/requeue, so a kill must not lose finished calls.
    # The fluency checkpoint is keyed by prompt too, so a stale one can't be
    # resumed under a different prompt.
    ckpt_dir = ROOT_DIR / "judge_rerun" / "agreement_ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    stem = f"{samples_path.stem}__{judge_id}"
    instruct_ckpt = ckpt_dir / f"instruct__{stem}.json"
    fluency_ckpt = ckpt_dir / f"fluency__{stem}__{args.fluency_prompt}.json"
    log.info("judge_id=%s fluency_prompt=%s batch_size=%d", judge_id, args.fluency_prompt,
              args.judge_batch_size)

    t_load = time.time()
    judge = QwenJudge(model_id=args.judge_model)
    log.info("judge loaded: %s (%.1fs)", args.judge_model, time.time() - t_load)

    instruct_prompts = [
        GeminiEvaluator.ALPACA_INSTRUCT_PROMPT.format(
            instructions=it["instruction"], completion=it["completion"])
        for it in items
    ]
    t0 = time.time()
    instruct_responses = judge.generate_batch(
        instruct_prompts, batch_size=args.judge_batch_size, checkpoint_path=instruct_ckpt)
    log.info("instruct-relevance pass done: %d calls (%.1fs)", len(items), time.time() - t0)

    fluency_prompts = [fluency_template.format(completion=it["completion"]) for it in items]
    t0 = time.time()
    fluency_responses = judge.generate_batch(
        fluency_prompts, batch_size=args.judge_batch_size, checkpoint_path=fluency_ckpt)
    log.info("fluency pass done: %d calls (%.1fs)", len(items), time.time() - t0)

    per_item: List[Dict[str, Any]] = []
    for it, i_resp, f_resp in zip(items, instruct_responses, fluency_responses):
        try:
            i_score = parse_rating(i_resp)
        except ValueError:
            i_score = None
        try:
            f_score = parse_rating(f_resp)
        except ValueError:
            f_score = None
        per_item.append({
            "instruction": it["instruction"], "completion": it["completion"],
            "gemini_instruct_score": it["instruct_score"], "qwen_instruct_score": i_score,
            "gemini_fluency_score": it["fluency_score"], "qwen_fluency_score": f_score,
            "qwen_instruct_response": i_resp, "qwen_fluency_response": f_resp,
        })

    clean_instr = [(r["gemini_instruct_score"], r["qwen_instruct_score"]) for r in per_item
                   if r["qwen_instruct_score"] is not None]
    clean_flu = [(r["gemini_fluency_score"], r["qwen_fluency_score"]) for r in per_item
                 if r["qwen_fluency_score"] is not None]

    results = {
        "n_items": len(items),
        "n_instruct_parse_failed": len(items) - len(clean_instr),
        "n_fluency_parse_failed": len(items) - len(clean_flu),
        "instruct": summarize("instruct", [g for g, _ in clean_instr], [p for _, p in clean_instr]),
        "fluency": summarize("fluency", [g for g, _ in clean_flu], [p for _, p in clean_flu]),
    }

    suffix = "" if args.judge_model == MODEL_ID else f"__{judge_id}"
    out_path = Path(args.out_path) if args.out_path else (
        ROOT_DIR / "judge_rerun" / f"agreement_study_{args.split}{suffix}.json")
    results["judge_model"] = args.judge_model
    results["fluency_prompt"] = args.fluency_prompt
    out_path.write_text(json.dumps({"summary": results, "per_item": per_item},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    instruct_ckpt.unlink(missing_ok=True)
    fluency_ckpt.unlink(missing_ok=True)

    for axis in ("instruct", "fluency"):
        r = results[axis]
        log.info("[%s] mean_gold(Gemini)=%.3f mean_pred(Qwen)=%.3f delta=%+.3f "
                  "exact_match=%.1f%% mae=%.3f",
                  axis, r["mean_gold_gemini"], r["mean_pred_qwen"], r["mean_delta"],
                  r["exact_match_rate"] * 100, r["mean_abs_error"])
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
