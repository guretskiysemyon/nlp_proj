#!/usr/bin/env python3
"""Pass 2 of the decoupled SNMF HP search: judge, score, pick the best HP cell.

For ONE concept, takes judge_rerun/pass1_snmf_top3.py's saved output (the
top-3 HP cells by the judge-free `harmonic`, each with its first 50 Alpaca
completions) and finishes the job:
  - judges each cell's completions with the local Qwen judge (instruct-
    relevance + fluency, the project's own prompts verbatim), via the same
    _judge_one_delta helper EMBER's Pass 2 uses
  - combines that with the cell's already-known efficacy/specificity into
    coherence and harmonic_alpaca
  - picks the winning cell

This replaces SNMF's original Step 3 (pipeline._run_validate_topk, which
generated+judged with the retired GeminiEvaluator) and the pick in
pipeline._pick_best_hp_for_test, and mirrors both:
  - efficacy/specificity are taken from the grid row as-is (validate passes
    them in as precomputed_metrics and never re-runs MMLU/QA/SimDom);
  - validate calls evaluate_model with min_alpaca unset, so SNMF has NO
    Alpaca gate (unlike EMBER's 0.9) -- --min-alpaca defaults to None;
  - the pick is io.topk_per_concept(k=1, primary="harmonic_alpaca"), the
    same call (and HP tie-breakers) as _pick_best_hp_for_test.

The Alpaca denominators are the SAME pristine-model Qwen baseline EMBER's
Pass 2 used (judge_rerun/Ember/pass1_outputs/.../_baseline/), so SNMF and
EMBER scores are directly comparable.

Usage:
    python -m judge_rerun.pass2_snmf_judge --concept "Golf" "Heroin" "Nazism"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import pandas as pd

from ember.erasure import io, log
from ember.evals.gemini import GeminiEvaluator
from ember.utils import _safe_concept, _safe_model_name
from judge_rerun.pass2_delta_search import (
    _combine_alpaca_metrics, _judge_one_delta, _load_baselines)
from judge_rerun.pass2_judge_baseline import _judge_id
from judge_rerun.qwen_judge import ALPACA_FLUENCY_PROMPT_QWEN, QwenJudge

# Grid-row columns carried into the output table (HPs + judge-free metrics).
_HP_COLS = ("w_mode", "feature_source", "ratio_thresh", "coverage_thresh",
            "delta_in", "layer_lo_in", "layer_hi_in", "k_features_mlp_in",
            "delta_out", "layer_lo_out", "layer_hi_out", "k_features_mlp_out")
_FAST_COLS = ("mmlu_acc", "qa_acc", "simdom_acc", "efficacy", "specificity", "harmonic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept", required=True, nargs="+",
                    help="one or more concepts, run back-to-back sharing one judge load")
    p.add_argument("--model-name", default="google/gemma-2-2b-it",
                    help="the model UNDER ERASURE (selects the pass1 dir)")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--judge-batch-size", type=int, default=4)
    p.add_argument("--fluency-prompt", choices=("original", "qwen_strengthened"),
                    default="original",
                    help="MUST match what the baseline was scored with")
    p.add_argument("--min-alpaca", type=float, default=None,
                    help="optional Alpaca gate; default None = SNMF's original validate "
                         "behavior (no gate). EMBER uses 0.9.")
    p.add_argument("--pass1-dir", default=str(ROOT_DIR / "judge_rerun" / "pass1_outputs_snmf"))
    p.add_argument("--baseline-dir", default=str(ROOT_DIR / "judge_rerun" / "Ember" / "pass1_outputs"),
                    help="root holding <model>/rank*/seed*/_baseline/ (shared with EMBER)")
    p.add_argument("--out-dir", default=str(ROOT_DIR / "judge_rerun" / "pass2_outputs_snmf"))
    p.add_argument("--overwrite", action="store_true",
                    help="re-judge every cell, ignoring the checkpoint")
    return p.parse_args()


def run_concept(args: argparse.Namespace, concept: str,
                judge: Optional[QwenJudge]) -> Optional[QwenJudge]:
    """Run Pass 2 for ONE concept. Returns the (possibly newly-loaded) judge,
    so a caller looping over several concepts in one process shares it."""
    log.set_context(method="snmf", concept=concept, stage="pass2_snmf_judge")

    judge_id = _judge_id(args.judge_model)
    fluency_template = (GeminiEvaluator.ALPACA_FLUENCY_PROMPT if args.fluency_prompt == "original"
                        else ALPACA_FLUENCY_PROMPT_QWEN)

    run_rel = Path(_safe_model_name(args.model_name)) / f"rank{args.rank}" / f"seed{args.seed}"
    pass1_concept_dir = Path(args.pass1_dir) / run_rel / _safe_concept(concept)
    baseline_dir = Path(args.baseline_dir) / run_rel / "_baseline"
    out_dir = Path(args.out_dir) / run_rel / _safe_concept(concept)

    best_path = out_dir / f"best_hp__{judge_id}.json"
    if best_path.exists() and not args.overwrite:
        log.info("%s already finished (%s exists); skipping", concept, best_path)
        return judge

    top3_path = pass1_concept_dir / "top3_hps.json"
    if not top3_path.exists():
        log.warning("%s missing -- SNMF Pass 1 not finished for this concept; skipping", top3_path)
        return judge
    pass1 = json.loads(top3_path.read_text())
    cells: List[Dict[str, Any]] = pass1["cells"]
    alpaca_n = int(pass1["alpaca_n"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bases = _load_baselines(baseline_dir, concept, judge_id, args.fluency_prompt)
    log.info("judge=%s fluency_prompt=%s min_alpaca=%s cells=%d alpaca_n=%d",
             args.judge_model, args.fluency_prompt, args.min_alpaca, len(cells), alpaca_n)
    log.info("alpaca baselines: instr=%.4f flu=%.4f",
             bases["alp_instr_base"], bases["alp_flu_base"])

    ckpt_path = out_dir / f"_rows__{judge_id}.json"
    judged_path = out_dir / f"alpaca_judged__{judge_id}.json"
    rows: Dict[str, Dict[str, Any]] = {}
    per_item_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    if ckpt_path.exists() and not args.overwrite:
        rows = json.loads(ckpt_path.read_text())
        log.info("resume: %d cells already scored in %s", len(rows), ckpt_path)
        if judged_path.exists():
            per_item_by_cell = json.loads(judged_path.read_text())

    for i, cell in enumerate(cells):
        key = f"cell_{i}"
        log.set_context(stage=key)
        if key in rows and not args.overwrite:
            log.info("%s already scored (harmonic_alpaca=%.4f); skipping",
                     key, rows[key].get("harmonic_alpaca", float("nan")))
            continue

        completions_path = pass1_concept_dir / "alpaca_completions" / f"{key}.json"
        items = json.loads(completions_path.read_text())[:alpaca_n]

        if judge is None:  # load lazily: a fully-resumed run needs no GPU at all
            judge = QwenJudge(model_id=args.judge_model)
            log.info("judge loaded: %s", args.judge_model)

        instruct_ckpt = out_dir / f"_ckpt__{judge_id}__{key}_instruct.json"
        fluency_ckpt = out_dir / f"_ckpt__{judge_id}__{key}_fluency.json"
        alpaca = _judge_one_delta(judge, items, fluency_template, args.judge_batch_size,
                                   instruct_ckpt, fluency_ckpt)
        per_item_by_cell[key] = alpaca.pop("_per_item")

        efficacy, specificity = float(cell["efficacy"]), float(cell["specificity"])
        # _combine_alpaca_metrics gates on min_alpaca; -inf disables the gate,
        # which is what SNMF's original validate does (min_alpaca=None).
        min_alpaca = args.min_alpaca if args.min_alpaca is not None else float("-inf")
        row: Dict[str, Any] = {"cell": i}
        row.update({c: cell[c] for c in _HP_COLS + _FAST_COLS if c in cell})
        row.update(_combine_alpaca_metrics(efficacy, specificity, alpaca["alpaca_instr"],
                                           alpaca["alpaca_flu"], bases, min_alpaca))
        row.update({k: alpaca[k] for k in ("n_instruct_parse_failed",
                                           "n_fluency_parse_failed", "judge_wall_time_s")})
        row["_alpaca_n"] = len(items)
        rows[key] = row

        log.info("%s instr=%.3f flu=%.3f coherence=%.4f harmonic=%.4f harmonic_alpaca=%.4f%s "
                 "(%.0fs, parse fails %d/%d, n=%d)",
                 key, row["alpaca_instr"], row["alpaca_flu"], row["coherence"],
                 row["harmonic"], row["harmonic_alpaca"],
                 f" GATED[{row['gate']}]" if row.get("gate") else "",
                 row["judge_wall_time_s"], row["n_instruct_parse_failed"],
                 row["n_fluency_parse_failed"], len(items))

        ckpt_path.write_text(json.dumps(rows, indent=2))
        judged_path.write_text(json.dumps(per_item_by_cell, ensure_ascii=False, indent=2))
        instruct_ckpt.unlink(missing_ok=True)  # this cell is recorded; drop its batch checkpoints
        fluency_ckpt.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Pick the winner -- the same call as pipeline._pick_best_hp_for_test #
    # ------------------------------------------------------------------ #
    log.set_context(stage="pass2_snmf_judge")
    df = pd.DataFrame(list(rows.values())).sort_values("cell").reset_index(drop=True)
    metrics_path = out_dir / f"cell_metrics__{judge_id}.csv"
    df.to_csv(metrics_path, index=False)

    best = io.topk_per_concept(df, k=1, primary="harmonic_alpaca").iloc[0]
    log.info("BEST cell=%d harmonic_alpaca=%.4f (harmonic=%.4f, coherence=%.4f)",
             int(best["cell"]), float(best["harmonic_alpaca"]),
             float(best["harmonic"]), float(best["coherence"]))

    summary: Dict[str, Any] = {
        "concept": concept,
        "judge_id": judge_id,
        "judge_model": args.judge_model,
        "fluency_prompt": args.fluency_prompt,
        "min_alpaca": args.min_alpaca,
        "alpaca_n": alpaca_n,
        "n_cells_scored": len(df),
        "n_cells_total": len(cells),
        "complete": len(df) == len(cells),
        "baselines": bases,
        "best_cell": int(best["cell"]),
        "best_harmonic_alpaca": float(best["harmonic_alpaca"]),
        "best_hp": {c: cells[int(best["cell"])][c] for c in _HP_COLS if c in cells[int(best["cell"])]},
        "best_metrics": {c: float(best[c]) for c in
                         ("efficacy", "specificity", "harmonic", "alp_instr_frac",
                          "alp_flu_frac", "coherence", "harmonic_alpaca")},
    }
    best_path.write_text(json.dumps(summary, indent=2))
    log.info("done: wrote %s and %s", metrics_path, best_path)
    return judge


def main() -> None:
    args = parse_args()
    log.configure()
    judge: Optional[QwenJudge] = None
    for concept in args.concept:
        judge = run_concept(args, concept, judge)


if __name__ == "__main__":
    main()
