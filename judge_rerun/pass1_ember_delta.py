#!/usr/bin/env python3
"""Pass 1 of the decoupled EMBER delta search: generate, don't judge.

For ONE concept, loops over EMBER's candidate deltas and, for each:
  - applies the embedding edit (reusing ember.erasure.embed_edit, unmodified)
  - scores MMLU / concept-QA / SimDom-QA (letter-parsed, no judge needed)
  - generates the Alpaca-train completions, but does NOT score them

Nothing here talks to any judge (Gemini or otherwise) and the model is never
asked to share memory with one. Output per delta is saved to disk so a
separate Pass 2 script (not written yet) can load a judge afterward, score
the saved completions, and pick the best delta.

This lives in judge_rerun/ (new code written for this project), not inside
ember/ (the original paper's reference implementation) -- it only *imports*
from ember, never edits it.

Usage:
    python -m judge_rerun.pass1_ember_delta --concept "Harry Potter"
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

from ember.erasure import embed_edit, log
from ember.erasure.model_loader import load_hf_model
from ember.evals.alpaca import load_alpaca_eval_local
from ember.evals.mc import evaluate_mc_generation, prepare_mc_items
from ember.evals.mmlu import evaluate_mmlu_generation, load_mmlu_items
from ember.evals.model_wrap import ensure_wrapped_model
from ember.local_datasets import load_alpaca_indices, load_mc_qa_items, load_mmlu_indices
from ember.utils import _safe_concept, _safe_model_name

DEFAULT_DELTAS = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]  # matches configs/ember_gemma.yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept", required=True, help='e.g. "Harry Potter"')
    p.add_argument("--model-name", default="google/gemma-2-2b-it")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratio-thresh", type=float, default=2.0)
    p.add_argument("--deltas", type=float, nargs="+", default=DEFAULT_DELTAS)
    p.add_argument("--alpaca-batch-size", type=int, default=8)   # matches configs/ember_gemma.yaml
    p.add_argument("--mmlu-batch-size", type=int, default=8)
    p.add_argument("--out-dir", default=str(ROOT_DIR / "judge_rerun" / "Ember" / "pass1_outputs"))
    p.add_argument("--overwrite", action="store_true",
                    help="redo this concept even if grid_raw.csv already exists")
    return p.parse_args()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = parse_args()
    log.configure()
    log.set_context(method="ember", concept=args.concept, stage="pass1")

    out_dir = (Path(args.out_dir) / _safe_model_name(args.model_name)
               / f"rank{args.rank}" / f"seed{args.seed}" / _safe_concept(args.concept))
    completions_dir = out_dir / "alpaca_completions"
    out_dir.mkdir(parents=True, exist_ok=True)
    completions_dir.mkdir(parents=True, exist_ok=True)
    log.info("writing outputs to %s", out_dir)

    grid_path = out_dir / "grid_raw.csv"
    if grid_path.exists() and not args.overwrite:
        log.info("skip: %s already exists (pass --overwrite to redo)", grid_path)
        return

    # ------------------------------------------------------------------ #
    # Load once: model, snapshot, and every eval set this concept needs  #
    # ------------------------------------------------------------------ #
    model, tokenizer = load_hf_model(args.model_name)
    tm = ensure_wrapped_model(model, tokenizer)
    pristine_embed = embed_edit.snapshot(model)

    qa_items = prepare_mc_items(load_mc_qa_items("qa_train", concept=args.concept), seed=args.seed)
    simdom_items = prepare_mc_items(load_mc_qa_items("simdom_train", concept=args.concept), seed=args.seed)
    if not qa_items:
        raise ValueError(f"No qa_train MC items found for concept={args.concept!r}")
    if not simdom_items:
        raise ValueError(f"No simdom_train MC items found for concept={args.concept!r}")

    mmlu_items = load_mmlu_items(load_mmlu_indices("train"))

    alpaca_indices = load_alpaca_indices("train")
    alpaca_all = load_alpaca_eval_local()
    alpaca_instructions_full = [ex["instruction"] for ex in alpaca_all if "instruction" in ex]
    alpaca_instructions = [alpaca_instructions_full[i] for i in alpaca_indices]

    log.info("eval sets ready: qa=%d simdom=%d mmlu=%d alpaca=%d",
             len(qa_items), len(simdom_items), len(mmlu_items), len(alpaca_instructions))

    # ------------------------------------------------------------------ #
    # One pass per delta: edit, score (no judge), save completions      #
    # ------------------------------------------------------------------ #
    rows: List[Dict[str, Any]] = []
    for delta in args.deltas:
        t0 = time.time()
        log.set_context(stage=f"delta={delta}")

        embed_edit.restore(model, pristine_embed)
        edit_info = embed_edit.apply_concept_embed_edit_factored(
            hf_model=model, model_name=args.model_name, concept_name=args.concept,
            delta_embed=delta, rank=args.rank, seed=args.seed,
            ratio_thresh=args.ratio_thresh,
        )

        mmlu_correct, mmlu_invalid, _ = evaluate_mmlu_generation(
            tm, mmlu_items, batch_size=args.mmlu_batch_size)
        qa_correct, qa_invalid, _ = evaluate_mc_generation(tm, qa_items)
        sim_correct, sim_invalid, _ = evaluate_mc_generation(tm, simdom_items)

        completions = tm.generate_multiple(
            alpaca_instructions, max_new_tokens=200, do_sample=False,
            batch_size=args.alpaca_batch_size, verbose=False,
        )

        n_mmlu, n_qa, n_sim = len(mmlu_items), len(qa_items), len(simdom_items)
        row = {
            "concept": args.concept,
            "delta_embed": delta,
            "k_features_embed": edit_info["k_features_embed"],
            "n_tokens_edited": edit_info["n_tokens_edited"],
            "mmlu_acc": mmlu_correct / n_mmlu if n_mmlu else float("nan"),
            "mmlu_invalid_rate": mmlu_invalid / n_mmlu if n_mmlu else float("nan"),
            "qa_acc": qa_correct / n_qa if n_qa else float("nan"),
            "qa_invalid_rate": qa_invalid / n_qa if n_qa else float("nan"),
            "simdom_acc": sim_correct / n_sim if n_sim else float("nan"),
            "simdom_invalid_rate": sim_invalid / n_sim if n_sim else float("nan"),
            "n_alpaca_completions": len(completions),
            "wall_time_s": time.time() - t0,
        }
        rows.append(row)
        log.info("delta=%s n_tokens_edited=%d mmlu_acc=%.3f qa_acc=%.3f simdom_acc=%.3f (%.1fs)",
                  delta, edit_info["n_tokens_edited"], row["mmlu_acc"], row["qa_acc"],
                  row["simdom_acc"], row["wall_time_s"])

        completions_path = completions_dir / f"delta_{delta}.json"
        completions_path.write_text(
            json.dumps(
                [{"idx": i, "instruction": instr, "completion": comp}
                 for i, (instr, comp) in enumerate(zip(alpaca_instructions, completions))],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    embed_edit.restore(model, pristine_embed)

    pd.DataFrame(rows).to_csv(grid_path, index=False)
    log.set_context(stage="pass1")
    log.info("done: %d deltas written to %s (completions under %s)",
              len(rows), grid_path, completions_dir)


if __name__ == "__main__":
    main()
