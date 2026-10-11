#!/usr/bin/env python3
"""Pass 1's pristine-model counterpart: generate the baseline, don't judge it.

Mirrors pass1_ember_delta.py but applies NO embedding edit -- the model stays
pristine throughout. Needed because the project's own harmonic_alpaca formula
(ember/evals/orchestrator.py, methods/ember.py) divides each delta's Alpaca
score by the pristine model's Alpaca score, and that denominator must be
judged by the same instrument (Qwen) as the numerator -- the archived
Gemini-scored baseline in archive_gemini/data/baselines/ cannot be reused for
this (see docs/judge_swap_plan.md SS2.3: mixing judges silently corrupts the
fraction).

Computes, once, with no edit applied:
  - MMLU accuracy (global, 50 items -- letter-parsed, no judge)
  - the 150 Alpaca-train completions (global, unscored -- Pass 2 judges these)
Then loops over all 18 concepts for:
  - concept-QA and SimDom-QA accuracy (per-concept item sets, letter-parsed)

Usage:
    python -m judge_rerun.pass1_baseline
    python -m judge_rerun.pass1_baseline --concepts "Harry Potter" "Golf"
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

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # must be set before torch imports (gotcha #14)

import pandas as pd

from ember.erasure import log
from ember.erasure.model_loader import load_hf_model
from ember.evals.alpaca import load_alpaca_eval_local
from ember.evals.mc import evaluate_mc_generation, prepare_mc_items
from ember.evals.mmlu import evaluate_mmlu_generation, load_mmlu_items
from ember.evals.model_wrap import ensure_wrapped_model
from ember.local_datasets import load_alpaca_indices, load_mc_qa_items, load_mmlu_indices
from ember.utils import _safe_model_name

DEFAULT_CONCEPTS = [
    "Ancient Rome", "Artificial intelligence", "Baseball", "COVID-19 pandemic",
    "Cannabis", "Culture of Greece", "Gambling", "Golf", "Gun", "Halloween",
    "Harry Potter", "Heroin", "Nazism", "Pornography", "Republic of Ireland",
    "Uranium", "Valentine's Day", "World War II",
]  # matches judge_rerun/submit_pass1_all.sh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS)
    p.add_argument("--model-name", default="google/gemma-2-2b-it")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpaca-batch-size", type=int, default=8)   # matches configs/ember_gemma.yaml
    p.add_argument("--mmlu-batch-size", type=int, default=8)
    p.add_argument("--out-dir", default=str(ROOT_DIR / "judge_rerun" / "Ember" / "pass1_outputs"))
    p.add_argument("--overwrite", action="store_true",
                    help="redo even if baseline_raw.csv already exists")
    return p.parse_args()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = parse_args()
    log.configure()
    log.set_context(method="ember", concept="_baseline", stage="pass1_baseline")

    out_dir = (Path(args.out_dir) / _safe_model_name(args.model_name)
               / f"rank{args.rank}" / f"seed{args.seed}" / "_baseline")
    completions_dir = out_dir / "alpaca_completions"
    out_dir.mkdir(parents=True, exist_ok=True)
    completions_dir.mkdir(parents=True, exist_ok=True)
    log.info("writing outputs to %s", out_dir)

    csv_path = out_dir / "baseline_raw.csv"
    if csv_path.exists() and not args.overwrite:
        log.info("skip: %s already exists (pass --overwrite to redo)", csv_path)
        return

    # ------------------------------------------------------------------ #
    # Load once. No embed_edit import, no snapshot/restore -- the model  #
    # is never mutated.                                                  #
    # ------------------------------------------------------------------ #
    model, tokenizer = load_hf_model(args.model_name)
    tm = ensure_wrapped_model(model, tokenizer)

    mmlu_items = load_mmlu_items(load_mmlu_indices("train"))

    alpaca_indices = load_alpaca_indices("train")
    alpaca_all = load_alpaca_eval_local()
    alpaca_instructions_full = [ex["instruction"] for ex in alpaca_all if "instruction" in ex]
    alpaca_instructions = [alpaca_instructions_full[i] for i in alpaca_indices]

    log.info("global eval sets ready: mmlu=%d alpaca=%d", len(mmlu_items), len(alpaca_instructions))

    # ------------------------------------------------------------------ #
    # Global (non-concept) pristine metrics: MMLU + Alpaca completions   #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    mmlu_correct, mmlu_invalid, _ = evaluate_mmlu_generation(
        tm, mmlu_items, batch_size=args.mmlu_batch_size)
    n_mmlu = len(mmlu_items)
    mmlu_acc = mmlu_correct / n_mmlu if n_mmlu else float("nan")
    mmlu_invalid_rate = mmlu_invalid / n_mmlu if n_mmlu else float("nan")
    log.info("pristine mmlu_acc=%.3f (%.1fs)", mmlu_acc, time.time() - t0)

    t0 = time.time()
    completions = tm.generate_multiple(
        alpaca_instructions, max_new_tokens=200, do_sample=False,
        batch_size=args.alpaca_batch_size, verbose=False,
    )
    log.info("pristine alpaca completions=%d (%.1fs)", len(completions), time.time() - t0)

    completions_path = completions_dir / "pristine.json"
    completions_path.write_text(
        json.dumps(
            [{"idx": i, "instruction": instr, "completion": comp}
             for i, (instr, comp) in enumerate(zip(alpaca_instructions, completions))],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # Per-concept pristine metrics: QA + SimDom (concept-specific items)  #
    # ------------------------------------------------------------------ #
    rows: List[Dict[str, Any]] = []
    for concept in args.concepts:
        t0 = time.time()
        log.set_context(concept=concept, stage="pass1_baseline")

        qa_items = prepare_mc_items(load_mc_qa_items("qa_train", concept=concept), seed=args.seed)
        simdom_items = prepare_mc_items(load_mc_qa_items("simdom_train", concept=concept), seed=args.seed)
        if not qa_items:
            raise ValueError(f"No qa_train MC items found for concept={concept!r}")
        if not simdom_items:
            raise ValueError(f"No simdom_train MC items found for concept={concept!r}")

        qa_correct, qa_invalid, _ = evaluate_mc_generation(tm, qa_items)
        sim_correct, sim_invalid, _ = evaluate_mc_generation(tm, simdom_items)
        n_qa, n_sim = len(qa_items), len(simdom_items)

        row = {
            "concept": concept,
            "mmlu_acc": mmlu_acc,
            "mmlu_invalid_rate": mmlu_invalid_rate,
            "qa_acc": qa_correct / n_qa if n_qa else float("nan"),
            "qa_invalid_rate": qa_invalid / n_qa if n_qa else float("nan"),
            "simdom_acc": sim_correct / n_sim if n_sim else float("nan"),
            "simdom_invalid_rate": sim_invalid / n_sim if n_sim else float("nan"),
            "n_alpaca_completions": len(completions),
            "wall_time_s": time.time() - t0,
        }
        rows.append(row)
        log.info("concept=%r qa_acc=%.3f simdom_acc=%.3f (%.1fs)",
                 concept, row["qa_acc"], row["simdom_acc"], row["wall_time_s"])

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.set_context(concept="_baseline", stage="pass1_baseline")
    log.info("done: %d concepts written to %s (alpaca completions at %s)",
              len(rows), csv_path, completions_path)


if __name__ == "__main__":
    main()
