#!/usr/bin/env python3
"""Re-score the EMBER delta grid's MC/MMLU with the fixed evaluator (judge_rerun/mc_fixed.py).

The original grid (pass1_ember_delta.py) scored MMLU / QA_train / SimDom_train
with a 4-token (MMLU: 6) budget, so "The answer is **C**" was cut before the
letter and counted wrong. This script re-applies each delta's edit exactly as
Pass 1 did, re-scores the three sets with a 32-token budget and the fixed
parser, and saves every per-question record (the original grid kept only totals).

It never writes into pass1_outputs/ or pass2_outputs/. Output, per concept:
    judge_rerun/Ember/rescore_mc/<model>/rank<r>/seed<s>/<Concept>/
        records/delta_<d>.json   per-question records (raw output, letters, old-equivalent)
        grid_rescored.csv        one row per delta: new metrics in grid_raw.csv's
                                 column names, plus grid_* (the old grid row) and
                                 old_* (old-equivalent recomputed from this run)
        compare.json             fast harmonic / top-k / pick, old vs new (CPU)

Determinism control: greedy decoding makes the first 4 (6) generated tokens
identical to the old run, so old_* must equal grid_* for every delta
(``control_ok``). If it does not, the rescore is not comparable and the job says so.

Deltas: by default only those with any invalid answer in grid_raw.csv (a delta
with none cannot change); --all-deltas re-scores all 8 (use it for the pilot,
the zero-invalid deltas then serve as extra controls).

Usage:
    python -m judge_rerun.rescore_ember_grid --concept Uranium --all-deltas
    python -m judge_rerun.rescore_ember_grid --concept Uranium --compare-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # must be set before torch imports (gotcha #14)

import pandas as pd

from ember.erasure import io, log
from ember.utils import _safe_concept, _safe_model_name
from judge_rerun.mc_fixed import MC_MAX_NEW_TOKENS, MMLU_MAX_NEW_TOKENS
from judge_rerun.pass2_delta_search import _combine_alpaca_metrics, _load_baselines, _metrics_for_delta
from judge_rerun.pass2_judge_baseline import _judge_id

SETS = ("mmlu", "qa", "simdom")
EMBER_DIR = ROOT_DIR / "judge_rerun" / "Ember"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept", required=True, nargs="+")
    p.add_argument("--model-name", default="google/gemma-2-2b-it")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratio-thresh", type=float, default=2.0)       # pass1_ember_delta.py default
    p.add_argument("--all-deltas", action="store_true",
                   help="re-score every delta, not only those with an invalid answer")
    p.add_argument("--reparse", action="store_true",
                   help="no GPU: re-apply the current parse_letter_fixed to the saved records, "
                        "rewrite grid_rescored.csv, then compare")
    p.add_argument("--compare-only", action="store_true",
                   help="skip the GPU part, recompute compare.json from grid_rescored.csv")
    p.add_argument("--overwrite", action="store_true", help="re-score deltas already on disk")
    # selection policy, must match pass2_delta_search.py
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--fluency-prompt", default="original")
    p.add_argument("--min-mmlu", type=float, default=0.90)
    p.add_argument("--min-alpaca", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--pass1-dir", default=str(EMBER_DIR / "pass1_outputs"))
    p.add_argument("--pass2-dir", default=str(EMBER_DIR / "pass2_outputs"))
    p.add_argument("--out-dir", default=str(EMBER_DIR / "rescore_mc"))
    return p.parse_args()


def _rel(args: argparse.Namespace) -> Path:
    return Path(_safe_model_name(args.model_name)) / f"rank{args.rank}" / f"seed{args.seed}"


# ---------------------------------------------------------------------------- #
# GPU part                                                                      #
# ---------------------------------------------------------------------------- #

class _Scorer:
    """Loads Gemma and the shared eval sets once; re-scores (concept, delta) cells."""

    def __init__(self, args: argparse.Namespace) -> None:
        from ember.erasure import embed_edit
        from ember.erasure.model_loader import load_hf_model
        from ember.evals.mmlu import load_mmlu_items
        from ember.evals.model_wrap import ensure_wrapped_model
        from ember.local_datasets import load_mmlu_indices

        self.args = args
        self.embed_edit = embed_edit
        model, tokenizer = load_hf_model(args.model_name)
        self.model = model
        self.tm = ensure_wrapped_model(model, tokenizer)
        self.pristine_embed = embed_edit.snapshot(model)
        self.mmlu_items = load_mmlu_items(load_mmlu_indices("train"))

    def concept_items(self, concept: str):
        from ember.evals.mc import prepare_mc_items
        from ember.local_datasets import load_mc_qa_items
        qa = prepare_mc_items(load_mc_qa_items("qa_train", concept=concept), seed=self.args.seed)
        sim = prepare_mc_items(load_mc_qa_items("simdom_train", concept=concept), seed=self.args.seed)
        if not qa or not sim:
            raise ValueError(f"missing qa_train / simdom_train items for {concept!r}")
        return qa, sim

    def score_delta(self, concept: str, delta: float, qa_items, sim_items) -> Dict[str, Any]:
        from judge_rerun.mc_fixed import evaluate_mc_fixed, evaluate_mmlu_fixed
        a = self.args
        self.embed_edit.restore(self.model, self.pristine_embed)
        info = self.embed_edit.apply_concept_embed_edit_factored(
            hf_model=self.model, model_name=a.model_name, concept_name=concept,
            delta_embed=delta, rank=a.rank, seed=a.seed, ratio_thresh=a.ratio_thresh,
        )
        out = {
            "mmlu": evaluate_mmlu_fixed(self.tm, self.mmlu_items),
            "qa": evaluate_mc_fixed(self.tm, qa_items),
            "simdom": evaluate_mc_fixed(self.tm, sim_items),
        }
        self.embed_edit.restore(self.model, self.pristine_embed)
        return {"edit_info": {"k_features_embed": int(info["k_features_embed"]),
                              "n_tokens_edited": int(info["n_tokens_edited"])},
                "sets": out}


def _row(concept: str, grid_row: pd.Series, res: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {"concept": concept, "delta_embed": float(grid_row["delta_embed"]),
                           **res["edit_info"]}
    ok = (res["edit_info"]["k_features_embed"] == int(grid_row["k_features_embed"])
          and res["edit_info"]["n_tokens_edited"] == int(grid_row["n_tokens_edited"]))
    row["features_match"] = ok
    control = ok
    for s in SETS:
        r = res["sets"][s]
        row[f"{s}_acc"] = r["acc"]
        row[f"{s}_invalid_rate"] = r["invalid_rate"]
        row[f"old_{s}_acc"] = r["old_acc"]
        row[f"old_{s}_invalid_rate"] = r["old_invalid_rate"]
        row[f"grid_{s}_acc"] = float(grid_row[f"{s}_acc"])
        row[f"grid_{s}_invalid_rate"] = float(grid_row[f"{s}_invalid_rate"])
        control &= (abs(r["old_acc"] - row[f"grid_{s}_acc"]) < 1e-9
                    and abs(r["old_invalid_rate"] - row[f"grid_{s}_invalid_rate"]) < 1e-9)
    row["control_ok"] = control
    row["mc_max_new_tokens"] = MC_MAX_NEW_TOKENS
    row["mmlu_max_new_tokens"] = MMLU_MAX_NEW_TOKENS
    row["wall_time_s"] = round(seconds, 1)
    return row


def rescore_concept(scorer: _Scorer, args: argparse.Namespace, concept: str) -> None:
    grid = pd.read_csv(Path(args.pass1_dir) / _rel(args) / _safe_concept(concept) / "grid_raw.csv")
    out_dir = Path(args.out_dir) / _rel(args) / _safe_concept(concept)
    rec_dir = out_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "grid_rescored.csv"
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    any_invalid = (grid[[f"{s}_invalid_rate" for s in SETS]] > 0).any(axis=1)
    todo = grid if args.all_deltas else grid[any_invalid]
    log.info("%s: re-scoring %d/%d deltas: %s", concept, len(todo), len(grid),
             todo["delta_embed"].tolist())
    qa_items, sim_items = scorer.concept_items(concept)

    rows = {float(r["delta_embed"]): r.to_dict() for _, r in done.iterrows()}
    for _, g in todo.iterrows():
        delta = float(g["delta_embed"])
        if delta in rows and not args.overwrite:
            log.info("delta=%s already re-scored; skipping", delta)
            continue
        log.set_context(stage=f"rescore delta={delta}")
        t0 = time.time()
        res = scorer.score_delta(concept, delta, qa_items, sim_items)
        row = _row(concept, g, res, time.time() - t0)
        (rec_dir / f"delta_{delta}.json").write_text(json.dumps(
            {"concept": concept, "delta_embed": delta, **res["edit_info"],
             "mc_max_new_tokens": MC_MAX_NEW_TOKENS, "mmlu_max_new_tokens": MMLU_MAX_NEW_TOKENS,
             **{s: res["sets"][s] for s in SETS}},
            ensure_ascii=False, indent=1), encoding="utf-8")
        rows[delta] = row
        pd.DataFrame(sorted(rows.values(), key=lambda r: r["delta_embed"])).to_csv(csv_path, index=False)
        log.info("delta=%s qa %.2f->%.2f (inv %.2f->%.2f) simdom %.2f->%.2f mmlu %.2f->%.2f "
                 "control_ok=%s (%.0fs)", delta,
                 row["grid_qa_acc"], row["qa_acc"], row["grid_qa_invalid_rate"], row["qa_invalid_rate"],
                 row["grid_simdom_acc"], row["simdom_acc"], row["grid_mmlu_acc"], row["mmlu_acc"],
                 row["control_ok"], row["wall_time_s"])
        if not row["control_ok"]:
            log.warning("CONTROL FAILED for delta=%s: the old-equivalent scores do not reproduce "
                        "grid_raw.csv, so this rescore is not comparable", delta)


# ---------------------------------------------------------------------------- #
# CPU part: old vs new selection                                                #
# ---------------------------------------------------------------------------- #

def _select(grid: pd.DataFrame, bases: Dict[str, float], judged: pd.DataFrame,
            args: argparse.Namespace) -> Dict[str, Any]:
    """Pass 2's policy on one grid: min_mmlu gate, top-k by fast harmonic, then
    harmonic_alpaca from the cached Qwen scores where they exist."""
    fast = {float(r["delta_embed"]): _metrics_for_delta(r, bases, None, args.min_mmlu, args.min_alpaca)
            for _, r in grid.iterrows()}
    passing = [d for d, m in fast.items() if m.get("gate") != "min_mmlu"]
    passing.sort(key=lambda d: fast[d]["harmonic"], reverse=True)
    top = passing[:args.top_k]
    judged_by_delta = {float(r["delta_embed"]): r for _, r in judged.iterrows()
                       if pd.notna(r.get("alpaca_instr"))}
    rows, missing = [], []
    for d in top:
        m = dict(fast[d])
        if d in judged_by_delta:
            j = judged_by_delta[d]
            m.update(_combine_alpaca_metrics(m["efficacy"], m["specificity"],
                                             float(j["alpaca_instr"]), float(j["alpaca_flu"]),
                                             bases, args.min_alpaca))
        else:
            missing.append(d)
            m["harmonic_alpaca"] = float("nan")
        rows.append(m)
    best = None
    if not missing and rows:
        best = io.pick_best_embed_delta(pd.DataFrame(rows))
    return {
        "fast": {repr(d): {k: fast[d].get(k) for k in ("qa_acc", "efficacy", "specificity", "harmonic", "gate")}
                 for d in sorted(fast)},
        "top_k": top,
        "top_k_harmonic_alpaca": {repr(r["delta_embed"]): r["harmonic_alpaca"] for r in rows},
        "needs_judging": missing,
        "best_delta_embed": None if best is None else float(best["delta_embed"]),
        "best_harmonic_alpaca": None if best is None else float(best["harmonic_alpaca"]),
    }


def reparse_concept(args: argparse.Namespace, concept: str) -> None:
    """Re-parse every saved records/delta_*.json with the current parser (no generation)."""
    from judge_rerun.mc_fixed import reparse
    rel = _rel(args)
    grid = pd.read_csv(Path(args.pass1_dir) / rel / _safe_concept(concept) / "grid_raw.csv")
    out_dir = Path(args.out_dir) / rel / _safe_concept(concept)
    csv_path = out_dir / "grid_rescored.csv"
    old_rows = {float(r["delta_embed"]): r for _, r in pd.read_csv(csv_path).iterrows()}
    rows = []
    for delta, old in sorted(old_rows.items()):
        rec_path = out_dir / "records" / f"delta_{delta}.json"
        saved = json.loads(rec_path.read_text(encoding="utf-8"))
        sets = {s: reparse(saved[s]) for s in SETS}
        saved.update(sets)
        rec_path.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        g = grid[grid["delta_embed"].astype(float) == delta].iloc[0]
        res = {"edit_info": {"k_features_embed": int(saved["k_features_embed"]),
                             "n_tokens_edited": int(saved["n_tokens_edited"])}, "sets": sets}
        row = _row(concept, g, res, float(old["wall_time_s"]))
        changed = [s for s in SETS if abs(row[f"{s}_acc"] - float(old[f"{s}_acc"])) > 1e-9
                   or abs(row[f"{s}_invalid_rate"] - float(old[f"{s}_invalid_rate"])) > 1e-9]
        if changed:
            log.info("%s delta=%s re-parse changed %s", concept, delta,
                     {s: (float(old[f"{s}_acc"]), row[f"{s}_acc"]) for s in changed})
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def compare_concept(args: argparse.Namespace, concept: str) -> Dict[str, Any]:
    rel = _rel(args)
    grid = pd.read_csv(Path(args.pass1_dir) / rel / _safe_concept(concept) / "grid_raw.csv")
    out_dir = Path(args.out_dir) / rel / _safe_concept(concept)
    resc = pd.read_csv(out_dir / "grid_rescored.csv")
    judge_id = _judge_id(args.judge_model)
    bases = _load_baselines(Path(args.pass1_dir) / rel / "_baseline", concept, judge_id, args.fluency_prompt)
    judged = pd.read_csv(Path(args.pass2_dir) / rel / _safe_concept(concept)
                         / f"delta_metrics__{judge_id}.csv")

    # New grid: re-scored deltas replaced, the rest (no invalid answer) kept as is.
    new_grid = grid.copy()
    for _, r in resc.iterrows():
        i = new_grid.index[new_grid["delta_embed"].astype(float) == float(r["delta_embed"])][0]
        for s in SETS:
            new_grid.loc[i, f"{s}_acc"] = r[f"{s}_acc"]
            new_grid.loc[i, f"{s}_invalid_rate"] = r[f"{s}_invalid_rate"]

    old, new = _select(grid, bases, judged, args), _select(new_grid, bases, judged, args)
    result = {
        "concept": concept,
        "control_ok_all": bool(resc["control_ok"].all()),
        "features_match_all": bool(resc["features_match"].all()),
        "rescored_deltas": resc["delta_embed"].tolist(),
        "old": old, "new": new,
        "top_k_changed": sorted(old["top_k"]) != sorted(new["top_k"]),
        "pick_changed": old["best_delta_embed"] != new["best_delta_embed"],
    }
    (out_dir / "compare.json").write_text(json.dumps(result, indent=2, default=float))
    log.info("%s: control_ok=%s top-%d old=%s new=%s; pick old=%s new=%s; needs judging: %s",
             concept, result["control_ok_all"], args.top_k, old["top_k"], new["top_k"],
             old["best_delta_embed"], new["best_delta_embed"], new["needs_judging"])
    return result


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = parse_args()
    log.configure()
    log.set_context(method="ember", stage="rescore_mc")
    if args.reparse:
        for c in args.concept:
            reparse_concept(args, c)
    elif not args.compare_only:
        scorer = _Scorer(args)
        for c in args.concept:
            log.set_context(method="ember", concept=c, stage="rescore_mc")
            rescore_concept(scorer, args, c)
    for c in args.concept:
        compare_concept(args, c)


if __name__ == "__main__":
    main()
