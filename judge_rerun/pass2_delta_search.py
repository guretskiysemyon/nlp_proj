#!/usr/bin/env python3
"""Pass 2 of the decoupled EMBER delta search: judge, score, pick the best delta.

For ONE concept, takes Pass 1's saved output and finishes the job:
  - reads the raw MC/MMLU accuracies per delta (grid_raw.csv, judge-free)
  - judges each delta's 150 saved Alpaca completions with the local Qwen judge
    (instruct-relevance + fluency, the project's own prompts verbatim)
  - computes the full metric set and picks the winning delta

The metric formulas and the selection rule are NOT reinvented here: the
fraction/harmonic definitions mirror ember/evals/orchestrator.py, the gates
come from ember/erasure/methods/ember.py::grid_eval_kwargs (min_mmlu=0.90,
min_alpaca=0.9), and the final pick calls ember.erasure.io.pick_best_embed_delta
directly so the tie-breaking and the "every row gated -> delta=0.0" fallback
are byte-identical to the original pipeline.

Usage:
    # quick end-to-end check on a few deltas first (~1h/delta on a 32B judge)
    python -m judge_rerun.pass2_delta_search --concept "Harry Potter" --deltas 0.5 10.0 200.0
    # then the rest; already-judged deltas are skipped via the checkpoint
    python -m judge_rerun.pass2_delta_search --concept "Harry Potter"
    # several concepts in ONE job/GPU allocation, loading the judge only once
    # (the model-load itself is only ~20-30s on the rtx_2080 nodes -- the real
    # win is paying SLURM's queue wait for that node just once, which has been
    # observed to run 15+ hours when both pinned nodes are busy):
    python -m judge_rerun.pass2_delta_search --concept "Golf" "Heroin" "Nazism"

Delta-selection policy (default --top-k 3 --alpaca-n 50, see docs/Part 1/ember_search.md):
    Before any judge call, every delta's judge-free fast metrics (mmlu/qa/
    simdom -> efficacy/specificity/harmonic) are computed from grid_raw.csv.
    Only the --top-k deltas (by that fast `harmonic`, among those passing
    min_mmlu) get judged at all, and only on their first --alpaca-n saved
    completions (idx < alpaca-n; the saved order is fixed across deltas/
    concepts, so "first N" means the same N questions everywhere). Deltas
    outside the top-k are recorded with gate="not_top_k", harmonic_alpaca=0,
    no judge calls spent. Pass --top-k 8 --alpaca-n 150 to reproduce the
    original full-grid, full-question behavior exactly.
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

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import pandas as pd

from ember.erasure import io, log
from ember.evals.gemini import GeminiEvaluator
from ember.evals.harmonic import harmonic_mean
from ember.utils import _safe_concept, _safe_model_name
from judge_rerun.pass2_judge_baseline import _judge_id
from judge_rerun.qwen_judge import ALPACA_FLUENCY_PROMPT_QWEN, QwenJudge, parse_rating

CHANCE = 0.25  # 4-choice MC floor, as in orchestrator.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept", required=True, nargs="+",
                    help='one or more concepts, e.g. --concept "Harry Potter" "Golf"; '
                         "run back-to-back in this process, sharing one judge load")
    p.add_argument("--model-name", default="google/gemma-2-2b-it",
                    help="the model UNDER ERASURE (selects the pass1_outputs dir)")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--judge-batch-size", type=int, default=4)
    p.add_argument("--fluency-prompt", choices=("original", "qwen_strengthened"),
                    default="original",
                    help="MUST match what the baseline was scored with, or the "
                         "coherence fraction mixes two instruments")
    p.add_argument("--deltas", type=float, nargs="+", default=None,
                    help="judge only these deltas (default: every delta in grid_raw.csv)")
    p.add_argument("--min-mmlu", type=float, default=0.90)    # methods/ember.py grid_eval_kwargs
    p.add_argument("--min-alpaca", type=float, default=0.9)   # methods/ember.py grid_eval_kwargs
    p.add_argument("--top-k", type=int, default=3,
                    help="of the deltas passing min_mmlu, judge only the top-k by the fast "
                         "(judge-free) harmonic score; pass a number >= the grid size to judge "
                         "every non-gated delta (original behavior)")
    p.add_argument("--alpaca-n", type=int, default=50,
                    help="judge only the first N saved Alpaca completions (idx < N) per kept "
                         "delta; pass a number >= 150 to judge all of them (original behavior)")
    p.add_argument("--pass1-dir", default=str(ROOT_DIR / "judge_rerun" / "Ember" / "pass1_outputs"))
    p.add_argument("--out-dir", default=str(ROOT_DIR / "judge_rerun" / "Ember" / "pass2_outputs"))
    p.add_argument("--overwrite", action="store_true",
                    help="re-judge every delta, ignoring the checkpoint")
    return p.parse_args()


def _mc_frac(acc: float, base: float) -> float:
    """clip((acc - 0.25) / (base - 0.25), 0, 1) -- orchestrator.py::_eval_mc_set."""
    if (base - CHANCE) == 0:
        return 0.0
    return min(1.0, max((acc - CHANCE) / (base - CHANCE), 0.0))


def _load_baselines(base_dir: Path, concept: str, judge_id: str,
                    fluency_prompt: str) -> Dict[str, float]:
    """Pristine-model denominators: MC/MMLU from the baseline pass, Alpaca from
    the SAME judge + prompt this run uses (see judge_swap_plan.md SS2.3)."""
    mc_path = base_dir / "baseline_raw.csv"
    alpaca_path = base_dir / f"baseline_alpaca_scores__{judge_id}.json"
    if not mc_path.exists():
        raise FileNotFoundError(f"{mc_path} missing -- run judge_rerun.pass1_baseline first")
    if not alpaca_path.exists():
        raise FileNotFoundError(
            f"{alpaca_path} missing -- run judge_rerun.pass2_judge_baseline "
            f"--judge-model <this judge> first")

    mc = pd.read_csv(mc_path)
    row = mc[mc["concept"].astype(str) == str(concept)]
    if row.empty:
        raise KeyError(f"no baseline row for concept={concept!r} in {mc_path}")
    row = row.iloc[0]

    alp = json.loads(alpaca_path.read_text())
    # The footgun this guards: a baseline scored with one fluency prompt and a
    # grid scored with another gives coherence a numerator and denominator from
    # two different instruments -- silent, plausible-looking, wrong.
    if alp.get("fluency_prompt") != fluency_prompt:
        raise ValueError(
            f"baseline was scored with fluency_prompt={alp.get('fluency_prompt')!r} "
            f"but this run uses {fluency_prompt!r}. Re-run the baseline with the "
            f"same prompt, or pass --fluency-prompt {alp.get('fluency_prompt')!r}.")

    return {
        "mmlu_base": float(row["mmlu_acc"]),
        "qa_base": float(row["qa_acc"]),
        "simdom_base": float(row["simdom_acc"]),
        "alp_instr_base": float(alp["mean_instruct_score"]),
        "alp_flu_base": float(alp["mean_fluency_score"]),
    }


def _judge_one_delta(judge: QwenJudge, items: List[Dict[str, Any]],
                     fluency_template: str, batch_size: int,
                     instruct_ckpt: Path, fluency_ckpt: Path) -> Dict[str, Any]:
    """Score one delta's 150 completions on both axes. Returns means + per-item."""
    instruct_prompts = [
        GeminiEvaluator.ALPACA_INSTRUCT_PROMPT.format(
            instructions=it["instruction"], completion=it["completion"])
        for it in items
    ]
    fluency_prompts = [fluency_template.format(completion=it["completion"]) for it in items]

    t0 = time.time()
    instruct_responses = judge.generate_batch(
        instruct_prompts, batch_size=batch_size, checkpoint_path=instruct_ckpt)
    fluency_responses = judge.generate_batch(
        fluency_prompts, batch_size=batch_size, checkpoint_path=fluency_ckpt)
    elapsed = time.time() - t0

    per_item: List[Dict[str, Any]] = []
    instruct_scores: List[int] = []
    fluency_scores: List[int] = []
    n_instruct_failed = n_fluency_failed = 0

    for it, i_resp, f_resp in zip(items, instruct_responses, fluency_responses):
        try:
            i_score: Optional[int] = parse_rating(i_resp)
        except ValueError:
            i_score, n_instruct_failed = None, n_instruct_failed + 1
        try:
            f_score: Optional[int] = parse_rating(f_resp)
        except ValueError:
            f_score, n_fluency_failed = None, n_fluency_failed + 1

        if i_score is not None:
            instruct_scores.append(i_score)
        if f_score is not None:
            fluency_scores.append(f_score)
        per_item.append({"idx": it["idx"], "instruct_score": i_score,
                         "fluency_score": f_score,
                         "instruct_response": i_resp, "fluency_response": f_resp})

    # Unparseable items are EXCLUDED from the mean, matching how the pristine
    # baseline's denominators were computed (pass2_judge_baseline.py). Note this
    # differs from the original Gemini pipeline, which defaulted a bad finish to
    # 0 after 3 retries; with 0 failures observed so far the two agree, but the
    # choice is deliberate rather than accidental.
    return {
        "alpaca_instr": sum(instruct_scores) / len(instruct_scores) if instruct_scores else float("nan"),
        "alpaca_flu": sum(fluency_scores) / len(fluency_scores) if fluency_scores else float("nan"),
        "n_instruct_parse_failed": n_instruct_failed,
        "n_fluency_parse_failed": n_fluency_failed,
        "judge_wall_time_s": elapsed,
        "_per_item": per_item,
    }


def _combine_alpaca_metrics(efficacy: float, specificity: float,
                            alpaca_instr: float, alpaca_flu: float,
                            bases: Dict[str, float], min_alpaca: float) -> Dict[str, Any]:
    """The alpaca-dependent half of orchestrator.py's metric block, factored out
    so a post-hoc re-pick (pass2_recompute_topk.py) can recompute it from a
    cached judge score without re-deriving the formula."""
    instr_base, flu_base = bases["alp_instr_base"], bases["alp_flu_base"]
    alp_instr_frac = min(1.0, alpaca_instr / instr_base) if instr_base > 0 else float("nan")
    alp_flu_frac = min(1.0, alpaca_flu / flu_base) if flu_base > 0 else float("nan")
    coherence = harmonic_mean([alp_instr_frac, alp_flu_frac])
    harmonic_alpaca = harmonic_mean([efficacy, specificity, coherence])

    gate = ""
    if alp_instr_frac < min_alpaca or alp_flu_frac < min_alpaca:
        harmonic_alpaca = 0.0
        gate = "min_alpaca"

    return {
        "alpaca_instr": alpaca_instr, "alpaca_flu": alpaca_flu,
        "alp_instr_frac": alp_instr_frac, "alp_flu_frac": alp_flu_frac,
        "coherence": coherence, "harmonic_alpaca": harmonic_alpaca, "gate": gate,
    }


def _metrics_for_delta(raw: pd.Series, bases: Dict[str, float],
                       alpaca: Optional[Dict[str, Any]],
                       min_mmlu: float, min_alpaca: float) -> Dict[str, Any]:
    """Replicates orchestrator.py's metric block for one grid row."""
    mmlu_acc = float(raw["mmlu_acc"])
    mmlu_frac = _mc_frac(mmlu_acc, bases["mmlu_base"])

    out: Dict[str, Any] = {
        "delta_embed": float(raw["delta_embed"]),
        "n_tokens_edited": int(raw["n_tokens_edited"]),
        "mmlu_acc": mmlu_acc,
        "mmlu_frac": mmlu_frac,
    }

    # orchestrator.py early-stop: mmlu_frac < min_mmlu zeroes the rest and never
    # reaches QA/SimDom or Alpaca at all (so we also skip the judge calls).
    if mmlu_frac < min_mmlu:
        out.update({
            "qa_acc": 0.0, "qa_frac": 0.0, "simdom_acc": 0.0, "simdom_frac": 0.0,
            "efficacy": 0.0, "specificity": 0.0, "harmonic": 0.0,
            "alpaca_instr": float("nan"), "alpaca_flu": float("nan"),
            "alp_instr_frac": float("nan"), "alp_flu_frac": float("nan"),
            "coherence": 0.0, "harmonic_alpaca": 0.0,
            "gate": "min_mmlu",
        })
        return out

    qa_frac = _mc_frac(float(raw["qa_acc"]), bases["qa_base"])
    sim_frac = _mc_frac(float(raw["simdom_acc"]), bases["simdom_base"])
    efficacy = 1.0 - qa_frac
    specificity = harmonic_mean([mmlu_frac, sim_frac])

    out.update({
        "qa_acc": float(raw["qa_acc"]), "qa_frac": qa_frac,
        "simdom_acc": float(raw["simdom_acc"]), "simdom_frac": sim_frac,
        "efficacy": efficacy, "specificity": specificity,
        "harmonic": harmonic_mean([efficacy, specificity]),
    })

    if alpaca is None:
        return out

    combined = _combine_alpaca_metrics(efficacy, specificity,
                                       alpaca["alpaca_instr"], alpaca["alpaca_flu"],
                                       bases, min_alpaca)
    out.update(combined)
    out.update({
        "n_instruct_parse_failed": alpaca["n_instruct_parse_failed"],
        "n_fluency_parse_failed": alpaca["n_fluency_parse_failed"],
        "judge_wall_time_s": alpaca["judge_wall_time_s"],
    })
    return out


def _gemini_reference(concept: str, model_name: str, rank: int, seed: int) -> Optional[pd.DataFrame]:
    """The archived Gemini-era grid for this concept, if it was one of the 8 run."""
    path = (ROOT_DIR / "archive_gemini" / "results" / "ember" / _safe_model_name(model_name)
            / f"rank{rank}" / f"seed{seed}" / "train_mc" / _safe_concept(concept) / "hps.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


def run_concept(args: argparse.Namespace, concept: str,
                judge: Optional[QwenJudge]) -> Optional[QwenJudge]:
    """Run Pass 2 for ONE concept. Returns the (possibly newly-loaded) judge,
    so a caller looping over several concepts in one process shares it."""
    log.set_context(method="ember", concept=concept, stage="pass2_delta_search")

    judge_id = _judge_id(args.judge_model)
    fluency_template = (GeminiEvaluator.ALPACA_FLUENCY_PROMPT if args.fluency_prompt == "original"
                        else ALPACA_FLUENCY_PROMPT_QWEN)

    run_rel = Path(_safe_model_name(args.model_name)) / f"rank{args.rank}" / f"seed{args.seed}"
    pass1_concept_dir = Path(args.pass1_dir) / run_rel / _safe_concept(concept)
    baseline_dir = Path(args.pass1_dir) / run_rel / "_baseline"
    out_dir = Path(args.out_dir) / run_rel / _safe_concept(concept)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / f"best_delta__{judge_id}.json"
    if best_path.exists() and not args.overwrite:
        log.info("%s already finished (%s exists); skipping", concept, best_path)
        return judge

    grid_path = pass1_concept_dir / "grid_raw.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"{grid_path} missing -- run judge_rerun.pass1_ember_delta first")
    grid = pd.read_csv(grid_path)

    bases = _load_baselines(baseline_dir, concept, judge_id, args.fluency_prompt)
    log.info("judge=%s fluency_prompt=%s", args.judge_model, args.fluency_prompt)
    log.info("baselines: mmlu=%.4f qa=%.4f simdom=%.4f alp_instr=%.4f alp_flu=%.4f",
             bases["mmlu_base"], bases["qa_base"], bases["simdom_base"],
             bases["alp_instr_base"], bases["alp_flu_base"])

    wanted = grid if args.deltas is None else grid[grid["delta_embed"].isin(args.deltas)]
    if wanted.empty:
        raise ValueError(f"no grid rows match --deltas {args.deltas}")

    ckpt_path = out_dir / f"_rows__{judge_id}.json"
    judged_path = out_dir / f"alpaca_judged__{judge_id}.json"
    rows: Dict[str, Dict[str, Any]] = {}
    per_item_by_delta: Dict[str, List[Dict[str, Any]]] = {}
    if ckpt_path.exists() and not args.overwrite:
        rows = json.loads(ckpt_path.read_text())
        log.info("resume: %d delta rows already scored in %s", len(rows), ckpt_path)
        if judged_path.exists():
            per_item_by_delta = json.loads(judged_path.read_text())

    # ------------------------------------------------------------------ #
    # Fast, judge-free screen: rank by the already-known mmlu/qa/simdom   #
    # harmonic and keep only the top --top-k for the expensive judge step #
    # ------------------------------------------------------------------ #
    fast_metrics: Dict[str, Dict[str, Any]] = {
        repr(float(raw["delta_embed"])): _metrics_for_delta(raw, bases, None, args.min_mmlu, args.min_alpaca)
        for _, raw in wanted.iterrows()
    }
    non_gated_keys = [k for k, m in fast_metrics.items() if m.get("gate") != "min_mmlu"]
    non_gated_keys.sort(key=lambda k: fast_metrics[k]["harmonic"], reverse=True)
    judge_keys = set(non_gated_keys[:args.top_k])
    log.info("fast screen: %d/%d deltas pass min_mmlu=%.2f; judging top %d by harmonic=%s",
             len(non_gated_keys), len(fast_metrics), args.min_mmlu, args.top_k,
             [fast_metrics[k]["delta_embed"] for k in non_gated_keys[:args.top_k]])

    for _, raw in wanted.iterrows():
        delta = float(raw["delta_embed"])
        key = repr(delta)

        if key not in judge_keys:
            # mmlu-gated, or passed the gate but didn't make the fast top-k:
            # no judge calls spent on it under the current policy.
            row = dict(fast_metrics[key])
            if row.get("gate") != "min_mmlu":
                row["gate"] = "not_top_k"
            rows[key] = row
            log.set_context(stage=f"delta={delta}")
            log.info("delta=%s %s -> no judge calls", delta,
                     "mmlu-gated" if row["gate"] == "min_mmlu" else "not in fast top-k")
            ckpt_path.write_text(json.dumps(rows, indent=2))
            continue

        cached = rows.get(key)
        stale = cached is not None and cached.get("_alpaca_n") != args.alpaca_n
        if cached is not None and not args.overwrite and not stale:
            log.info("delta=%s already scored (harmonic_alpaca=%.4f); skipping",
                     delta, cached.get("harmonic_alpaca", float("nan")))
            continue

        log.set_context(stage=f"delta={delta}")
        completions_path = pass1_concept_dir / "alpaca_completions" / f"delta_{delta}.json"
        if not completions_path.exists():
            raise FileNotFoundError(f"{completions_path} missing")
        items = json.loads(completions_path.read_text())
        if args.alpaca_n < len(items):
            items = items[:args.alpaca_n]

        if judge is None:  # load lazily: a fully-resumed run needs no GPU at all
            judge = QwenJudge(model_id=args.judge_model)
            log.info("judge loaded: %s", args.judge_model)

        instruct_ckpt = out_dir / f"_ckpt__{judge_id}__delta_{delta}_instruct.json"
        fluency_ckpt = out_dir / f"_ckpt__{judge_id}__delta_{delta}_fluency.json"
        alpaca = _judge_one_delta(judge, items, fluency_template, args.judge_batch_size,
                                   instruct_ckpt, fluency_ckpt)
        per_item_by_delta[key] = alpaca.pop("_per_item")
        rows[key] = _metrics_for_delta(raw, bases, alpaca, args.min_mmlu, args.min_alpaca)
        rows[key]["_alpaca_n"] = args.alpaca_n

        r = rows[key]
        log.info("delta=%s instr=%.3f flu=%.3f coherence=%.4f harmonic=%.4f "
                 "harmonic_alpaca=%.4f%s (%.0fs, parse fails %d/%d, n=%d)",
                 delta, r["alpaca_instr"], r["alpaca_flu"], r["coherence"],
                 r["harmonic"], r["harmonic_alpaca"],
                 f" GATED[{r['gate']}]" if r.get("gate") else "",
                 r["judge_wall_time_s"], r["n_instruct_parse_failed"],
                 r["n_fluency_parse_failed"], len(items))

        ckpt_path.write_text(json.dumps(rows, indent=2))
        judged_path.write_text(json.dumps(per_item_by_delta, ensure_ascii=False, indent=2))
        instruct_ckpt.unlink(missing_ok=True)  # this delta is recorded; drop its batch checkpoints
        fluency_ckpt.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Pick the winner -- io.pick_best_embed_delta, not a reimplementation #
    # ------------------------------------------------------------------ #
    log.set_context(stage="pass2_delta_search")
    df = pd.DataFrame(list(rows.values())).sort_values("delta_embed").reset_index(drop=True)
    metrics_path = out_dir / f"delta_metrics__{judge_id}.csv"
    df.to_csv(metrics_path, index=False)

    n_expected = len(grid)
    if len(df) < n_expected:
        log.warning("only %d/%d deltas scored -- best-delta pick is PROVISIONAL "
                    "until the rest are judged", len(df), n_expected)

    best = io.pick_best_embed_delta(df)
    log.info("BEST delta_embed=%s harmonic_alpaca=%.4f (of %d/%d deltas scored)",
             best["delta_embed"], float(best.get("harmonic_alpaca") or 0.0), len(df), n_expected)

    summary: Dict[str, Any] = {
        "concept": concept,
        "judge_id": judge_id,
        "judge_model": args.judge_model,
        "fluency_prompt": args.fluency_prompt,
        "min_mmlu": args.min_mmlu,
        "min_alpaca": args.min_alpaca,
        "top_k": args.top_k,
        "alpaca_n": args.alpaca_n,
        "n_deltas_scored": len(df),
        "n_deltas_total": n_expected,
        "complete": len(df) == n_expected,
        "baselines": bases,
        "best_delta_embed": float(best["delta_embed"]),
        "best_harmonic_alpaca": float(best.get("harmonic_alpaca") or 0.0),
    }

    # ------------------------------------------------------------------ #
    # Side-by-side against the archived Gemini-era grid, when it exists   #
    # ------------------------------------------------------------------ #
    ref = _gemini_reference(concept, args.model_name, args.rank, args.seed)
    if ref is not None:
        ref_best = io.pick_best_embed_delta(ref)
        summary["gemini_best_delta_embed"] = float(ref_best["delta_embed"])
        summary["gemini_best_harmonic_alpaca"] = float(ref_best.get("harmonic_alpaca") or 0.0)
        summary["agrees_with_gemini"] = (
            float(best["delta_embed"]) == float(ref_best["delta_embed"]))
        log.info("Gemini-era pick for this concept: delta=%s harmonic_alpaca=%.4f -> %s",
                 ref_best["delta_embed"], float(ref_best.get("harmonic_alpaca") or 0.0),
                 "AGREES" if summary["agrees_with_gemini"] else "DIFFERS")
        log.info("per-delta comparison (qwen vs gemini harmonic_alpaca):")
        ref_by_delta = {float(r["delta_embed"]): r for _, r in ref.iterrows()}
        for _, r in df.iterrows():
            g = ref_by_delta.get(float(r["delta_embed"]))
            log.info("  delta=%-6s qwen=%.4f  gemini=%s  (coherence %.4f vs %s)",
                     r["delta_embed"], r["harmonic_alpaca"],
                     f"{g['harmonic_alpaca']:.4f}" if g is not None else "   n/a",
                     r["coherence"],
                     f"{g['coherence']:.4f}" if g is not None else "n/a")

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
