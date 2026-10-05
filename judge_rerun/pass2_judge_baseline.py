#!/usr/bin/env python3
"""Judge the pristine-model baseline's Alpaca completions with Qwen.

Reads judge_rerun/pass1_baseline.py's output (the 150 pristine completions,
unscored) and scores each one with the local Qwen judge, using the project's
own prompts/parser verbatim (ember.evals.gemini.GeminiEvaluator) so this is
directly comparable to the archived Gemini-era numbers.

This produces alp_instr_base / alp_flu_base -- the denominators every delta's
harmonic_alpaca needs -- judged by the SAME instrument (Qwen) that will judge
the delta grid in Pass 2 proper. Reusing the archived Gemini-scored baseline
here would silently mix instruments (docs/judge_swap_plan.md SS2.3).

Usage:
    python -m judge_rerun.pass2_judge_baseline
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
from ember.utils import _safe_model_name
from judge_rerun.qwen_judge import ALPACA_FLUENCY_PROMPT_QWEN, QwenJudge, parse_rating


def _judge_id(model_id: str) -> str:
    """e.g. 'Qwen/Qwen2.5-32B-Instruct' -> 'qwen2.5-32b-instruct'."""
    name = model_id.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9.]+", "-", name).strip("-")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="google/gemma-2-2b-it",
                    help="the model UNDER ERASURE (picks which pass1_outputs dir to read)")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-14B-Instruct",
                    help="the local judge model to load (any size that fits the GPU budget)")
    p.add_argument("--judge-batch-size", type=int, default=8)
    p.add_argument("--fluency-prompt", choices=("original", "qwen_strengthened"),
                    default="original",
                    help="'original' = verbatim GeminiEvaluator prompt (test whether this "
                         "judge model follows the cutoff instruction unaided); "
                         "'qwen_strengthened' = the Qwen2.5-14B-tuned restatement "
                         "(ALPACA_FLUENCY_PROMPT_QWEN)")
    p.add_argument("--out-dir", default=str(ROOT_DIR / "judge_rerun" / "pass1_outputs"))
    p.add_argument("--overwrite", action="store_true",
                    help="redo everything, including the instruct-relevance pass")
    p.add_argument("--redo-fluency-only", action="store_true",
                    help="reuse this judge's previous instruct_response per item from "
                         "alpaca_judged__<judge_id>.json and only re-run the fluency pass")
    p.add_argument("--rescore-only", action="store_true",
                    help="no GPU: re-parse the raw responses already saved in "
                         "alpaca_judged__<judge_id>.json with the current parser and rewrite "
                         "the scores/summary files")
    return p.parse_args()


def _score_and_write(items: List[Dict[str, Any]], instruct_responses: List[str],
                     fluency_responses: List[str], args: argparse.Namespace, judge_id: str,
                     judged_path: Path, scores_path: Path) -> None:
    """Parse the judge's raw responses and write the per-item + summary files."""
    per_item: List[Dict[str, Any]] = []
    instruct_scores: List[int] = []
    fluency_scores: List[int] = []
    n_instruct_failed = 0
    n_fluency_failed = 0

    for it, i_resp, f_resp in zip(items, instruct_responses, fluency_responses):
        try:
            i_score = parse_rating(i_resp)
        except ValueError:
            i_score = None
            n_instruct_failed += 1
        try:
            f_score = parse_rating(f_resp)
        except ValueError:
            f_score = None
            n_fluency_failed += 1

        if i_score is not None:
            instruct_scores.append(i_score)
        if f_score is not None:
            fluency_scores.append(f_score)

        per_item.append({
            "idx": it["idx"],
            "instruct_score": i_score,
            "fluency_score": f_score,
            "instruct_response": i_resp,
            "fluency_response": f_resp,
        })

    mean_instruct = sum(instruct_scores) / len(instruct_scores) if instruct_scores else float("nan")
    mean_fluency = sum(fluency_scores) / len(fluency_scores) if fluency_scores else float("nan")

    log.info("mean_instruct_score=%.4f mean_fluency_score=%.4f "
              "(parse failures: instruct=%d/%d fluency=%d/%d)",
              mean_instruct, mean_fluency, n_instruct_failed, len(items),
              n_fluency_failed, len(items))

    judged_path.write_text(json.dumps(per_item, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "judge_id": judge_id,
        "judge_model": args.judge_model,
        "fluency_prompt": args.fluency_prompt,
        "n_items": len(items),
        "mean_instruct_score": mean_instruct,
        "mean_fluency_score": mean_fluency,
        "n_instruct_parse_failed": n_instruct_failed,
        "n_fluency_parse_failed": n_fluency_failed,
    }
    scores_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    log.configure()
    log.set_context(method="ember", concept="_baseline", stage="pass2_judge_baseline")

    judge_id = _judge_id(args.judge_model)
    fluency_prompt_template = (GeminiEvaluator.ALPACA_FLUENCY_PROMPT if args.fluency_prompt == "original"
                               else ALPACA_FLUENCY_PROMPT_QWEN)

    base_dir = (Path(args.out_dir) / _safe_model_name(args.model_name)
                / f"rank{args.rank}" / f"seed{args.seed}" / "_baseline")
    completions_path = base_dir / "alpaca_completions" / "pristine.json"
    if not completions_path.exists():
        raise FileNotFoundError(
            f"{completions_path} missing -- run judge_rerun.pass1_baseline first")

    scores_path = base_dir / f"baseline_alpaca_scores__{judge_id}.json"
    judged_path = base_dir / f"alpaca_judged__{judge_id}.json"
    log.info("judge_id=%s fluency_prompt=%s -> %s", judge_id, args.fluency_prompt, scores_path)
    if (scores_path.exists() and not args.overwrite and not args.redo_fluency_only
            and not args.rescore_only):
        log.info("skip: %s already exists (pass --overwrite or --redo-fluency-only to redo)",
                  scores_path)
        return

    items: List[Dict[str, Any]] = json.loads(completions_path.read_text())

    if args.rescore_only:
        prior = {row["idx"]: row for row in json.loads(judged_path.read_text())}
        log.info("rescore-only: re-parsing %d saved responses from %s (no judge loaded)",
                  len(items), judged_path)
        _score_and_write(items, [prior[it["idx"]]["instruct_response"] for it in items],
                         [prior[it["idx"]]["fluency_response"] for it in items],
                         args, judge_id, judged_path, scores_path)
        return
    log.info("judging %d pristine completions (%d instruct + %d fluency calls)",
              len(items), len(items), len(items))

    # ------------------------------------------------------------------ #
    # Instruct relevance: reuse a previous run's responses if asked --   #
    # this axis was already close to the archived Gemini baseline        #
    # (1.88 vs 1.987), unlike fluency, so there's no need to re-spend     #
    # ~30 min of GPU time re-confirming it.                               #
    # ------------------------------------------------------------------ #
    instruct_ckpt = base_dir / f"_instruct_responses__{judge_id}.json"
    fluency_ckpt = base_dir / f"_fluency_responses__{judge_id}.json"
    instruct_responses: List[str] | None = None

    if args.overwrite:
        instruct_ckpt.unlink(missing_ok=True)
        fluency_ckpt.unlink(missing_ok=True)

    if args.redo_fluency_only and judged_path.exists():
        prior = {row["idx"]: row["instruct_response"] for row in json.loads(judged_path.read_text())}
        instruct_responses = [prior[it["idx"]] for it in items]
        log.info("reusing %d instruct responses from a previous %s", len(instruct_responses), judged_path)
        fluency_ckpt.unlink(missing_ok=True)  # force a fresh fluency pass under the new prompt

    judge = QwenJudge(model_id=args.judge_model)
    log.info("judge loaded: %s", args.judge_model)

    if instruct_responses is None:
        # generate_batch resumes from instruct_ckpt itself (any prefix already
        # done, including a full prefix -> no GPU work at all) -- this pass
        # can take well over an hour on a bigger judge, so per-chunk
        # checkpointing is what survives a SLURM time-limit kill mid-pass.
        t0 = time.time()
        instruct_prompts = [
            GeminiEvaluator.ALPACA_INSTRUCT_PROMPT.format(
                instructions=it["instruction"], completion=it["completion"])
            for it in items
        ]
        instruct_responses = judge.generate_batch(
            instruct_prompts, batch_size=args.judge_batch_size, checkpoint_path=instruct_ckpt)
        log.info("instruct-relevance pass: %d calls (%.1fs)", len(items), time.time() - t0)

    t0 = time.time()
    fluency_prompts = [
        fluency_prompt_template.format(completion=it["completion"])
        for it in items
    ]
    fluency_responses = judge.generate_batch(
        fluency_prompts, batch_size=args.judge_batch_size, checkpoint_path=fluency_ckpt)
    log.info("fluency pass: %d calls (%.1fs)", len(items), time.time() - t0)

    _score_and_write(items, instruct_responses, fluency_responses, args, judge_id,
                     judged_path, scores_path)
    instruct_ckpt.unlink(missing_ok=True)
    fluency_ckpt.unlink(missing_ok=True)
    log.set_context(concept="_baseline", stage="pass2_judge_baseline")
    log.info("done: wrote %s and %s", scores_path, judged_path)


if __name__ == "__main__":
    main()
