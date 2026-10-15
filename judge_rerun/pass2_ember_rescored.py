#!/usr/bin/env python3
"""EMBER Pass 2 on the re-scored grid: final best delta per concept after the MC fix.

judge_rerun/rescore_ember_grid.py re-scored every delta's MMLU / QA_train /
SimDom_train with the fixed evaluator (judge_rerun/Ember/rescore_mc/). This
script applies Pass 2's selection to those scores, exactly as
pass2_delta_search.py does (min_mmlu gate, top-k by the fast harmonic, Qwen
judge on the first --alpaca-n saved Alpaca completions, harmonic_alpaca,
io.pick_best_embed_delta), with one difference: Qwen scores are reused wherever
they already exist.

  - A top-k delta that the old Pass 2 already judged on the first 50 completions
    reuses those means (alpaca_instr / alpaca_flu from the old
    delta_metrics__<judge>.csv). Alpaca is open-ended text, unaffected by the fix.
  - A top-k delta never judged before is judged now. Its completions already
    exist: Pass 1 saved 150 for all 8 deltas of every concept.
  - harmonic_alpaca is always recomputed from the NEW efficacy / specificity.

It never writes into pass1_outputs/ or pass2_outputs/. Output, per concept:
    judge_rerun/Ember/pass2_rescored/<model>/rank<r>/seed<s>/<Concept>/
        delta_metrics__<judge>.csv, best_delta__<judge>.json,
        alpaca_judged__<judge>.json (only deltas judged by this script), _rows__/_ckpt__ (resume)
and best_deltas_summary.csv for all concepts processed.

Without a delta to judge no GPU is needed, so the same command writes the final
picks for the concepts whose top 3 did not change.

Usage:
    python -m judge_rerun.pass2_ember_rescored --concept Uranium Gun ...
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
from judge_rerun.pass2_delta_search import _judge_one_delta, _load_baselines, _metrics_for_delta
from judge_rerun.pass2_judge_baseline import _judge_id
from judge_rerun.qwen_judge import ALPACA_FLUENCY_PROMPT_QWEN

EMBER_DIR = ROOT_DIR / "judge_rerun" / "Ember"
SETS = ("mmlu", "qa", "simdom")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept", required=True, nargs="+")
    p.add_argument("--model-name", default="google/gemma-2-2b-it")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--judge-batch-size", type=int, default=4)
    p.add_argument("--fluency-prompt", choices=("original", "qwen_strengthened"), default="original")
    p.add_argument("--min-mmlu", type=float, default=0.90)    # same as pass2_delta_search.py
    p.add_argument("--min-alpaca", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--alpaca-n", type=int, default=50)
    p.add_argument("--pass1-dir", default=str(EMBER_DIR / "pass1_outputs"))
    p.add_argument("--old-pass2-dir", default=str(EMBER_DIR / "pass2_outputs"))
    p.add_argument("--rescore-dir", default=str(EMBER_DIR / "rescore_mc"))
    p.add_argument("--out-dir", default=str(EMBER_DIR / "pass2_rescored"))
    return p.parse_args()


def _rescored_grid(path: Path, n_expected: int = 8) -> pd.DataFrame:
    """grid_rescored.csv -> grid_raw.csv's columns (new scores), all deltas required."""
    df = pd.read_csv(path)
    if len(df) != n_expected:
        raise ValueError(f"{path} has {len(df)} deltas, expected {n_expected}: "
                         f"run rescore_ember_grid.py --all-deltas first")
    cols = ["concept", "delta_embed", "k_features_embed", "n_tokens_edited"]
    cols += [f"{s}_{m}" for s in SETS for m in ("acc", "invalid_rate")]
    return df[cols].sort_values("delta_embed").reset_index(drop=True)


def _old_alpaca(path: Path, alpaca_n: int) -> Dict[float, Dict[str, Any]]:
    """Qwen means the old Pass 2 computed, for deltas judged on exactly the first alpaca_n."""
    if not path.exists():
        return {}
    out = {}
    for _, r in pd.read_csv(path).iterrows():
        if pd.notna(r.get("alpaca_instr")) and int(r.get("_alpaca_n", -1)) == alpaca_n:
            out[float(r["delta_embed"])] = {
                "alpaca_instr": float(r["alpaca_instr"]), "alpaca_flu": float(r["alpaca_flu"]),
                "n_instruct_parse_failed": int(r["n_instruct_parse_failed"]),
                "n_fluency_parse_failed": int(r["n_fluency_parse_failed"]),
                "judge_wall_time_s": float(r["judge_wall_time_s"]),
                "alpaca_source": "old_pass2"}
    return out


def run_concept(args: argparse.Namespace, concept: str, judge: Optional[Any]) -> tuple:
    log.set_context(method="ember", concept=concept, stage="pass2_rescored")
    rel = Path(_safe_model_name(args.model_name)) / f"rank{args.rank}" / f"seed{args.seed}"
    judge_id = _judge_id(args.judge_model)
    fluency_template = (GeminiEvaluator.ALPACA_FLUENCY_PROMPT if args.fluency_prompt == "original"
                        else ALPACA_FLUENCY_PROMPT_QWEN)
    pass1_dir = Path(args.pass1_dir) / rel / _safe_concept(concept)
    out_dir = Path(args.out_dir) / rel / _safe_concept(concept)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = _rescored_grid(Path(args.rescore_dir) / rel / _safe_concept(concept) / "grid_rescored.csv")
    bases = _load_baselines(Path(args.pass1_dir) / rel / "_baseline", concept, judge_id, args.fluency_prompt)
    old = _old_alpaca(Path(args.old_pass2_dir) / rel / _safe_concept(concept)
                      / f"delta_metrics__{judge_id}.csv", args.alpaca_n)

    # Fast screen, same ordering as pass2_delta_search.py (stable sort over grid order).
    fast = {float(r["delta_embed"]): _metrics_for_delta(r, bases, None, args.min_mmlu, args.min_alpaca)
            for _, r in grid.iterrows()}
    passing = [d for d, m in fast.items() if m.get("gate") != "min_mmlu"]
    passing.sort(key=lambda d: fast[d]["harmonic"], reverse=True)
    top = passing[:args.top_k]
    log.info("fast screen on re-scored grid: top %d = %s (old Qwen scores for %s)",
             args.top_k, top, sorted(d for d in top if d in old))

    ckpt_path = out_dir / f"_rows__{judge_id}.json"
    judged_path = out_dir / f"alpaca_judged__{judge_id}.json"
    new_alpaca = {float(k): v for k, v in json.loads(ckpt_path.read_text()).items()} if ckpt_path.exists() else {}
    per_item = json.loads(judged_path.read_text()) if judged_path.exists() else {}

    rows = []
    for _, raw in grid.iterrows():
        d = float(raw["delta_embed"])
        if d not in top:
            row = dict(fast[d])
            if row.get("gate") != "min_mmlu":
                row["gate"] = "not_top_k"
            rows.append(row)
            continue
        alp = old.get(d) or new_alpaca.get(d)
        if alp is None:
            items = json.loads((pass1_dir / "alpaca_completions" / f"delta_{d}.json").read_text())[:args.alpaca_n]
            if judge is None:
                from judge_rerun.qwen_judge import QwenJudge
                judge = QwenJudge(model_id=args.judge_model)
                log.info("judge loaded: %s", args.judge_model)
            log.set_context(stage=f"delta={d}")
            res = _judge_one_delta(judge, items, fluency_template, args.judge_batch_size,
                                   out_dir / f"_ckpt__{judge_id}__delta_{d}_instruct.json",
                                   out_dir / f"_ckpt__{judge_id}__delta_{d}_fluency.json")
            per_item[repr(d)] = res.pop("_per_item")
            alp = {**res, "alpaca_source": "pass2_rescored"}
            new_alpaca[d] = alp
            ckpt_path.write_text(json.dumps({repr(k): v for k, v in new_alpaca.items()}, indent=2))
            judged_path.write_text(json.dumps(per_item, ensure_ascii=False, indent=2))
            for axis in ("instruct", "fluency"):
                (out_dir / f"_ckpt__{judge_id}__delta_{d}_{axis}.json").unlink(missing_ok=True)
            log.info("delta=%s judged: instr=%.3f flu=%.3f (%.0fs)", d, alp["alpaca_instr"],
                     alp["alpaca_flu"], alp["judge_wall_time_s"])
        row = _metrics_for_delta(raw, bases, alp, args.min_mmlu, args.min_alpaca)
        row["alpaca_source"] = alp["alpaca_source"]
        row["_alpaca_n"] = args.alpaca_n
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("delta_embed").reset_index(drop=True)
    df.to_csv(out_dir / f"delta_metrics__{judge_id}.csv", index=False)
    best = io.pick_best_embed_delta(df)
    old_best_path = Path(args.old_pass2_dir) / rel / _safe_concept(concept) / f"best_delta__{judge_id}.json"
    old_best = json.loads(old_best_path.read_text()) if old_best_path.exists() else {}
    summary = {
        "concept": concept, "judge_id": judge_id, "judge_model": args.judge_model,
        "fluency_prompt": args.fluency_prompt, "min_mmlu": args.min_mmlu,
        "min_alpaca": args.min_alpaca, "top_k": args.top_k, "alpaca_n": args.alpaca_n,
        "scoring": "mc_fixed (judge_rerun/mc_fixed.py), grid judge_rerun/Ember/rescore_mc",
        "top_k_deltas": top,
        "judged_now": sorted(d for d in top if d not in old),
        "baselines": bases,
        "best_delta_embed": float(best["delta_embed"]),
        "best_harmonic_alpaca": float(best.get("harmonic_alpaca") or 0.0),
        "old_best_delta_embed": old_best.get("best_delta_embed"),
        "old_best_harmonic_alpaca": old_best.get("best_harmonic_alpaca"),
        "pick_changed": old_best.get("best_delta_embed") != float(best["delta_embed"]),
    }
    (out_dir / f"best_delta__{judge_id}.json").write_text(json.dumps(summary, indent=2))
    log.info("BEST delta=%s harmonic_alpaca=%.4f (old: %s, %.4f)", summary["best_delta_embed"],
             summary["best_harmonic_alpaca"], summary["old_best_delta_embed"],
             summary["old_best_harmonic_alpaca"] or 0.0)
    return judge, summary


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = parse_args()
    log.configure()
    judge, summaries = None, []
    for c in args.concept:
        judge, s = run_concept(args, c, judge)
        summaries.append(s)
    rel = Path(_safe_model_name(args.model_name)) / f"rank{args.rank}" / f"seed{args.seed}"
    path = Path(args.out_dir) / rel / "best_deltas_summary.csv"
    keep = ["concept", "best_delta_embed", "best_harmonic_alpaca", "old_best_delta_embed",
            "old_best_harmonic_alpaca", "pick_changed", "top_k_deltas", "judged_now"]
    new = pd.DataFrame([{k: s[k] for k in keep} for s in summaries])
    if path.exists():
        prev = pd.read_csv(path)
        new = pd.concat([prev[~prev["concept"].isin(new["concept"])], new])
    new.sort_values("concept").to_csv(path, index=False)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
