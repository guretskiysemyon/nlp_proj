#!/usr/bin/env python3
"""Post-hoc re-pick of the best delta for an ALREADY fully-judged concept,
applying the new top-k / first-N-questions policy -- no GPU, no judge calls.

Pass 2 originally judged all 8 (non-mmlu-gated) deltas on their full 150
saved Alpaca completions. The updated policy (see docs/Part 1/ember_search.md and
pass2_delta_search.py's module docstring) only judges the top --top-k deltas
by the fast (judge-free) harmonic score, and only their first --alpaca-n
completions. For a concept Pass 2 already finished, every number that policy
needs is already on disk:
  - _rows__{judge}.json has the fast `harmonic` for every delta (computed
    before any judge call was made, so it's unaffected by the new policy).
  - alpaca_judged__{judge}.json has every per-item instruct/fluency score for
    all 150 questions x 8 deltas.
This script re-ranks by that fast harmonic, keeps the top-k, averages only
the first alpaca-n items' cached scores, and re-derives harmonic_alpaca via
the same _combine_alpaca_metrics formula pass2_delta_search.py uses, then
picks the winner with the same io.pick_best_embed_delta.

It never touches _rows__/alpaca_judged__/best_delta__, and never writes into
the original concept's output directory at all (many of those are owned by a
teammate with no group-write bit -- writing there would need their chmod, not
ours to take). Instead it writes best_delta_topk__{judge}.json and
delta_metrics_topk__{judge}.csv into a parallel tree (same model/rank/seed/
concept layout), which this script's invoker owns.

Legacy tool (2026-10-09): all 12 concepts that had been judged on the old 8x150
grid were recomputed with this script and promoted into Ember/pass2_outputs
(see judge_rerun/Ember/README.md). The defaults below therefore point at the
archived full-grid originals and the archived recompute tree; only use it
again for a concept that is later judged with --top-k 8 --alpaca-n 150.

Usage:
    python -m judge_rerun.pass2_recompute_topk --concept "Harry Potter"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from ember.erasure import io, log
from ember.utils import _safe_concept, _safe_model_name
from judge_rerun.pass2_delta_search import _combine_alpaca_metrics, _gemini_reference, _load_baselines
from judge_rerun.pass2_judge_baseline import _judge_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept", required=True, help='e.g. "Harry Potter"')
    p.add_argument("--model-name", default="google/gemma-2-2b-it")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--alpaca-n", type=int, default=50)
    p.add_argument("--pass1-dir", default=str(ROOT_DIR / "judge_rerun" / "Ember" / "pass1_outputs"))
    p.add_argument("--pass2-dir", default=str(ROOT_DIR / "judge_rerun" / "Ember" / "archive" / "pass2_outputs_original"),
                    help="where to READ the original full-grid _rows__/alpaca_judged__/best_delta__")
    p.add_argument("--out-dir", default=str(ROOT_DIR / "judge_rerun" / "Ember" / "archive" / "pass2_outputs_topk_recompute"),
                    help="where to WRITE the recomputed best_delta_topk__/delta_metrics_topk__ "
                         "(kept separate from --pass2-dir: that tree is often owned by a teammate "
                         "without a group-write bit, and this never needs to modify it)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.configure()
    log.set_context(method="ember", concept=args.concept, stage="pass2_recompute_topk")

    judge_id = _judge_id(args.judge_model)
    run_rel = Path(_safe_model_name(args.model_name)) / f"rank{args.rank}" / f"seed{args.seed}"
    src_dir = Path(args.pass2_dir) / run_rel / _safe_concept(args.concept)
    out_dir = Path(args.out_dir) / run_rel / _safe_concept(args.concept)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = src_dir / f"_rows__{judge_id}.json"
    judged_path = src_dir / f"alpaca_judged__{judge_id}.json"
    best_path = src_dir / f"best_delta__{judge_id}.json"
    for p in (rows_path, judged_path, best_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- this concept hasn't been fully judged yet, "
                                     f"run pass2_delta_search for it first")

    rows: Dict[str, Dict[str, Any]] = json.loads(rows_path.read_text())
    per_item_by_delta: Dict[str, List[Dict[str, Any]]] = json.loads(judged_path.read_text())
    orig_summary = json.loads(best_path.read_text())

    fluency_prompt = orig_summary["fluency_prompt"]
    min_alpaca = orig_summary.get("min_alpaca", 0.9)
    baseline_dir = Path(args.pass1_dir) / run_rel / "_baseline"
    bases = _load_baselines(baseline_dir, args.concept, judge_id, fluency_prompt)

    non_gated_keys = [k for k, r in rows.items() if r.get("gate") != "min_mmlu"]
    non_gated_keys.sort(key=lambda k: rows[k]["harmonic"], reverse=True)
    top_keys = set(non_gated_keys[:args.top_k])
    log.info("fast screen: %d/%d deltas pass min_mmlu; re-picking top %d by harmonic: %s",
             len(non_gated_keys), len(rows), args.top_k,
             [rows[k]["delta_embed"] for k in non_gated_keys[:args.top_k]])

    new_rows: Dict[str, Dict[str, Any]] = dict(rows)
    for key in top_keys:
        items = per_item_by_delta.get(key, [])
        subset = [it for it in items if it["idx"] < args.alpaca_n]
        instr_scores = [it["instruct_score"] for it in subset if it["instruct_score"] is not None]
        flu_scores = [it["fluency_score"] for it in subset if it["fluency_score"] is not None]
        alpaca_instr = sum(instr_scores) / len(instr_scores) if instr_scores else float("nan")
        alpaca_flu = sum(flu_scores) / len(flu_scores) if flu_scores else float("nan")

        row = dict(rows[key])
        combined = _combine_alpaca_metrics(row["efficacy"], row["specificity"],
                                           alpaca_instr, alpaca_flu, bases, min_alpaca)
        row.update(combined)
        row["n_instruct_parse_failed"] = sum(1 for it in subset if it["instruct_score"] is None)
        row["n_fluency_parse_failed"] = sum(1 for it in subset if it["fluency_score"] is None)
        row["_alpaca_n"] = args.alpaca_n
        new_rows[key] = row
        log.info("delta=%s (n=%d) instr=%.3f flu=%.3f coherence=%.4f harmonic_alpaca=%.4f%s",
                 row["delta_embed"], len(subset), alpaca_instr, alpaca_flu, row["coherence"],
                 row["harmonic_alpaca"], f" GATED[{row['gate']}]" if row.get("gate") else "")

    for key, r in rows.items():
        if key in top_keys or r.get("gate") == "min_mmlu":
            continue
        row = dict(r)
        row["gate"] = "not_top_k"
        for f in ("harmonic_alpaca", "alpaca_instr", "alpaca_flu",
                  "alp_instr_frac", "alp_flu_frac", "coherence"):
            row.pop(f, None)
        new_rows[key] = row

    df = pd.DataFrame(list(new_rows.values())).sort_values("delta_embed").reset_index(drop=True)
    metrics_path = out_dir / f"delta_metrics_topk__{judge_id}.csv"
    df.to_csv(metrics_path, index=False)

    best = io.pick_best_embed_delta(df)
    orig_best_delta = orig_summary.get("best_delta_embed")
    same_pick = (orig_best_delta is not None
                 and float(best["delta_embed"]) == float(orig_best_delta))
    log.info("BEST (top-%d/n=%d) delta_embed=%s harmonic_alpaca=%.4f -- %s original full-grid pick "
             "(delta=%s harmonic_alpaca=%.4f)",
             args.top_k, args.alpaca_n, best["delta_embed"], float(best.get("harmonic_alpaca") or 0.0),
             "MATCHES" if same_pick else "DIFFERS from",
             orig_best_delta, float(orig_summary.get("best_harmonic_alpaca") or 0.0))

    summary: Dict[str, Any] = {
        "concept": args.concept,
        "judge_id": judge_id,
        "judge_model": args.judge_model,
        "fluency_prompt": fluency_prompt,
        "min_alpaca": min_alpaca,
        "top_k": args.top_k,
        "alpaca_n": args.alpaca_n,
        "recomputed_from": rows_path.name,
        "best_delta_embed": float(best["delta_embed"]),
        "best_harmonic_alpaca": float(best.get("harmonic_alpaca") or 0.0),
        "original_best_delta_embed": orig_best_delta,
        "original_best_harmonic_alpaca": orig_summary.get("best_harmonic_alpaca"),
        "same_pick_as_original": same_pick,
    }

    ref = _gemini_reference(args.concept, args.model_name, args.rank, args.seed)
    if ref is not None:
        ref_best = io.pick_best_embed_delta(ref)
        summary["gemini_best_delta_embed"] = float(ref_best["delta_embed"])
        summary["agrees_with_gemini"] = (
            float(best["delta_embed"]) == float(ref_best["delta_embed"]))

    best_path_out = out_dir / f"best_delta_topk__{judge_id}.json"
    best_path_out.write_text(json.dumps(summary, indent=2))
    log.info("done: wrote %s and %s", metrics_path, best_path_out)


if __name__ == "__main__":
    main()
