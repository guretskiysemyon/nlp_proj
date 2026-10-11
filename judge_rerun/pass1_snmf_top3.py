#!/usr/bin/env python3
"""Pass 1 of the decoupled SNMF HP search: grid (if needed), narrow to the
top-3 HP cells by fast (judge-free) harmonic, generate -- but do NOT judge --
their first 50 Alpaca completions.

For each concept:
  - If an archived 144-cell grid already exists (judge-free, so Gemini vs
    Qwen doesn't matter -- see docs/ember_search.md), reuse it as-is: no
    grid is re-run. Its top_hps.csv has 15 rows; we take the top 3 of those
    by `harmonic` ourselves (never generating completions for the other 12).
  - Otherwise, run SNMF's 144-cell grid fresh via the SAME pipeline code
    `ember/run_erasure.py` uses (ember.erasure.pipeline._run_method_grid),
    just with `topk=3` so only 3 rows are ever written to top_hps.csv in the
    first place -- and Step 3 (validate, which inline-generates+judges all
    150 Alpaca completions with the retired GeminiEvaluator) is never
    called at all.

For the resulting 3 HP cells (either path), drive SNMFMethod.apply/snapshot/
restore (its real MLP-weight edit, not EMBER's simpler embed edit) and
generate completions for only the first --alpaca-n (default 50) Alpaca
questions -- the same fixed-order subset judge_rerun/pass2_delta_search.py
uses for EMBER, via the same load_alpaca_indices("train") -- since nothing
downstream will ever judge the other 100.

The grid stage's own per-cell checkpointing (ember/erasure/io.py's
load_done_set/append_csv_row, used by _run_method_grid already) means a
timed-out multi-concept job just needs an identical resubmit to continue.

--with-ember runs SNMF+EMBER (the paper's `snmf_ember`) instead of plain
SNMF: each concept's best EMBER delta (the Qwen-judged pick in
judge_rerun/Ember/) is applied under every SNMF cell, and the SNMF grid is
searched again on top of it -- the same thing ember/erasure/pipeline.py does
with `ember_step.enabled`. It switches the defaults to
configs/snmf_ember_gemma.yaml (grids under results/snmf_ef/, archived ones
under archive_gemini/results/snmf_ef/) and judge_rerun/SNMF_EMBER/pass1_outputs.
A grid is only used if every row carries exactly this concept's EMBER delta
(0.0 for plain SNMF).

Usage:
    python -m judge_rerun.pass1_snmf_top3 --concept "Culture of Greece" "Golf" ...
    python -m judge_rerun.pass1_snmf_top3 --with-ember --concept "Harry Potter" ...
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import pandas as pd

from ember.erasure import eval as eval_mod
from ember.erasure import io, log
from ember.erasure import pipeline as pipeline_mod
from ember.erasure.config import load_yaml
from ember.erasure.methods import base as methods_base
from ember.erasure.model_loader import load_hf_model
from ember.evals.alpaca import load_alpaca_eval_local
from ember.evals.model_wrap import ensure_wrapped_model
from ember.local_datasets import load_alpaca_indices
from ember.utils import _safe_concept, _safe_model_name

# The judge-free grid stage never touches this baseline value (grid_eval_kwargs
# sets eval_alpaca=False for snmf), so reading the archived, Gemini-scored
# copy here is safe -- it is loaded only to satisfy get_baselines_for_mode's
# unconditional (but here unused) alpaca-baseline read. The MC baselines
# (mmlu/qa/simdom) it reads alongside are judge-free and already cover all
# 18 concepts, confirmed against archive_gemini/data/baselines/.
ARCHIVE_BASELINE_DIR = ROOT_DIR / "archive_gemini" / "data" / "baselines"

DEFAULT_CONFIG = ROOT_DIR / "configs" / "snmf_gemma.yaml"
EMBER_CONFIG = ROOT_DIR / "configs" / "snmf_ember_gemma.yaml"
# Pass 1 output root for plain SNMF (False) and SNMF+EMBER (True).
DEFAULT_OUT_DIRS = {False: ROOT_DIR / "judge_rerun" / "SNMF" / "pass1_outputs",
                    True: ROOT_DIR / "judge_rerun" / "SNMF_EMBER" / "pass1_outputs"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept", required=True, nargs="+",
                    help='one or more concepts, run back-to-back sharing one model load')
    p.add_argument("--with-ember", action="store_true",
                    help="SNMF+EMBER: apply each concept's best EMBER delta under every "
                         "SNMF cell (switches the --config and --out-dir defaults)")
    p.add_argument("--config", default=None,
                    help="base SNMF config (grid HP ranges come from here); default "
                         "configs/snmf_gemma.yaml, or configs/snmf_ember_gemma.yaml "
                         "with --with-ember")
    p.add_argument("--model-name", default=None, help="override config's model_name")
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--top-k", type=int, default=3,
                    help="HP cells to keep (and generate completions for) after the "
                         "judge-free grid/archived top_hps.csv")
    p.add_argument("--alpaca-n", type=int, default=50,
                    help="generate only the first N saved Alpaca completions (idx < N) "
                         "per kept cell -- nothing downstream judges the rest")
    p.add_argument("--alpaca-batch-size", type=int, default=8)
    p.add_argument("--out-dir", default=None,
                    help="default judge_rerun/SNMF/pass1_outputs, or "
                         "judge_rerun/SNMF_EMBER/pass1_outputs with --with-ember")
    p.add_argument("--overwrite", action="store_true",
                    help="redo this concept's completion generation even if already saved "
                         "(does NOT force-rerun an already-complete grid; use the grid's own "
                         "--overwrite path for that if ever needed)")
    return p.parse_args()


def resolve_variant_defaults(args: argparse.Namespace) -> None:
    """Fill --config / --out-dir for plain SNMF or SNMF+EMBER (--with-ember)."""
    if args.config is None:
        args.config = str(EMBER_CONFIG if args.with_ember else DEFAULT_CONFIG)
    if args.out_dir is None:
        args.out_dir = str(DEFAULT_OUT_DIRS[args.with_ember])


def load_run_config(args: argparse.Namespace):
    """Load --config, apply the CLI overrides, and check it matches --with-ember."""
    cfg = load_yaml(Path(args.config))
    cfg.train_eval = "mc"
    cfg.run_tests_after_train = False
    if args.model_name:
        cfg.model_name = args.model_name
    if args.rank:
        cfg.rank = args.rank
    if args.seed:
        cfg.seed = args.seed
    if bool(cfg.ember_step.enabled) != args.with_ember:
        raise ValueError(f"{args.config} has ember_step.enabled={cfg.ember_step.enabled}, "
                         f"but --with-ember is {args.with_ember}")
    return cfg


def ember_delta(cfg, concept: str) -> float:
    """The EMBER delta applied under every SNMF cell: 0.0 for plain SNMF, else the
    concept's best delta from the Qwen-judged EMBER search (judge_rerun/Ember/)."""
    if not cfg.ember_step.enabled:
        return 0.0
    summary = (ROOT_DIR / "judge_rerun" / "Ember" / "pass2_outputs"
               / _safe_model_name(cfg.model_name) / f"rank{cfg.rank}" / f"seed{cfg.seed}"
               / "best_deltas_summary.csv")
    df = pd.read_csv(summary)
    match = df[df["concept"] == concept]
    if match.empty:
        raise KeyError(f"no EMBER delta for {concept!r} in {summary}")
    return float(match["best_delta_embed"].iloc[0])


def check_grid_delta(df: pd.DataFrame, expected: float, source: Path) -> None:
    """Refuse a grid that was not run with exactly this concept's EMBER delta."""
    found = sorted(set(pd.to_numeric(df["delta_embed"]).round(6)))
    if found != [round(expected, 6)]:
        raise ValueError(f"{source} was run with delta_embed {found}, "
                         f"but this concept's EMBER delta is {expected}")


def _grid_dir(cfg) -> Path:
    """results/snmf/... for plain SNMF, results/snmf_ef/... with the EMBER step."""
    return io.train_dir("snmf", cfg.model_name, cfg.rank, cfg.seed, cfg.train_eval,
                        embed_step_enabled=bool(cfg.ember_step.enabled))


def _archived_top_hps_path(cfg, concept: str) -> Path:
    live = _grid_dir(cfg)
    return (ROOT_DIR / "archive_gemini" / live.relative_to(ROOT_DIR)
            / _safe_concept(concept) / "top_hps.csv")


def _live_top_hps_path(cfg, concept: str) -> Path:
    return _grid_dir(cfg) / _safe_concept(concept) / "top_hps.csv"


def _out_concept_dir(args: argparse.Namespace, cfg, concept: str) -> Path:
    run_rel = Path(_safe_model_name(cfg.model_name)) / f"rank{cfg.rank}" / f"seed{cfg.seed}"
    return Path(args.out_dir) / run_rel / _safe_concept(concept)


def _already_done(out_dir: Path, top_k: int) -> bool:
    if not (out_dir / "top3_hps.json").exists():
        return False
    return all((out_dir / "alpaca_completions" / f"cell_{i}.json").exists()
               for i in range(top_k))


def _build_hp(row: pd.Series, method: methods_base.Method) -> Dict[str, Any]:
    hp: Dict[str, Any] = {c: row[c] for c in method.hp_columns() if c in row.index}
    if "delta_embed" not in hp and "delta_embed" in row.index:
        hp["delta_embed"] = float(row["delta_embed"])
    return hp


def run_concept(args: argparse.Namespace, cfg, concept: str,
                loaded: Dict[str, Any]) -> Dict[str, Any]:
    """Process one concept. ``loaded`` carries the lazily-created model/
    tokenizer/baselines across concepts in this process; returned (possibly
    updated) for the next call."""
    log.set_context(method=io.method_dir_name("snmf", bool(cfg.ember_step.enabled)),
                    concept=concept, stage="pass1_snmf_top3")

    out_dir = _out_concept_dir(args, cfg, concept)
    if _already_done(out_dir, args.top_k) and not args.overwrite:
        log.info("%s already has top-%d cells + completions; skipping", concept, args.top_k)
        return loaded

    delta_embed = ember_delta(cfg, concept)
    if cfg.ember_step.enabled:
        log.info("%s: EMBER delta %g applied under every SNMF cell", concept, delta_embed)
    archived = _archived_top_hps_path(cfg, concept)
    live = _live_top_hps_path(cfg, concept)

    if archived.exists():
        log.info("%s: reusing archived judge-free grid -> %s", concept, archived)
        top_df = io.read_csv_safe(archived)
        check_grid_delta(top_df, delta_embed, archived)
        top3 = io.topk_per_concept(top_df, k=args.top_k, primary="harmonic")
    elif live.exists() and not args.overwrite:
        log.info("%s: live grid already has a top_hps.csv -> %s", concept, live)
        top3 = io.read_csv_safe(live)
        check_grid_delta(top3, delta_embed, live)
        if len(top3) > args.top_k:
            top3 = io.topk_per_concept(top3, k=args.top_k, primary="harmonic")
    else:
        log.info("%s: no grid yet -- running the 144-cell judge-free grid "
                 "(top_k=%d so only %d rows are ever written)", concept, args.top_k, args.top_k)
        if loaded.get("hf_model") is None:
            _load_model_and_baselines(args, cfg, loaded)
        sub_cfg = copy.copy(cfg)
        sub_cfg.concepts = [concept]
        sub_cfg.topk = args.top_k
        sub_cfg.overwrite = args.overwrite
        method = methods_base.get("snmf")
        pipeline_mod._run_method_grid(
            method, sub_cfg, loaded["hf_model"], loaded["tokenizer"],
            loaded["baselines"], {concept: delta_embed}, loaded["alpaca_bs"])
        if not live.exists():
            log.warning("%s: grid finished but no top_hps.csv written (empty grid?); skipping", concept)
            return loaded
        top3 = io.read_csv_safe(live)
        check_grid_delta(top3, delta_embed, live)

    if top3.empty:
        log.warning("%s: no HP cells to generate completions for; skipping", concept)
        return loaded

    if loaded.get("hf_model") is None:
        _load_model_and_baselines(args, cfg, loaded)

    method = methods_base.get("snmf")
    hf_model, tokenizer = loaded["hf_model"], loaded["tokenizer"]

    alpaca_indices = load_alpaca_indices("train")[:args.alpaca_n]
    alpaca_all = load_alpaca_eval_local()
    alpaca_instructions_full = [ex["instruction"] for ex in alpaca_all if "instruction" in ex]
    alpaca_instructions = [alpaca_instructions_full[i] for i in alpaca_indices]

    completions_dir = out_dir / "alpaca_completions"
    completions_dir.mkdir(parents=True, exist_ok=True)

    method.on_concept_start(hf_model, concept, cfg)
    snap = method.snapshot(hf_model)

    cells: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(top3.reset_index(drop=True).iterrows()):
        method.restore(hf_model, snap)
        hp = _build_hp(row, method)
        info = method.apply(hf_model, tokenizer, hp, concept, cfg)

        tm = ensure_wrapped_model(hf_model, tokenizer)
        completions = tm.generate_multiple(
            alpaca_instructions, max_new_tokens=200, do_sample=False,
            batch_size=args.alpaca_batch_size, verbose=False,
        )

        cell_path = completions_dir / f"cell_{i}.json"
        cell_path.write_text(json.dumps(
            # "idx" is the LOCAL position (0..alpaca_n-1) within the fixed
            # 150-question order, matching pass2_delta_search.py's convention
            # for EMBER -- not the raw alpaca_eval_local() dataset index.
            [{"idx": pos, "instruction": instr, "completion": comp}
             for pos, (instr, comp) in enumerate(zip(alpaca_instructions, completions))],
            ensure_ascii=False, indent=2))

        cell_row = {**hp, **info, **{k: row[k] for k in
                   ("mmlu_acc", "qa_acc", "simdom_acc", "efficacy", "specificity", "harmonic")
                   if k in row.index}}
        cells.append(cell_row)
        log.info("%s cell=%d harmonic=%.4f -> wrote %s", concept, i,
                 cell_row.get("harmonic", float("nan")), cell_path)

    method.restore(hf_model, snap)
    method.on_concept_end(hf_model, concept, cfg)

    (out_dir / "top3_hps.json").write_text(json.dumps({
        "concept": concept, "top_k": args.top_k, "alpaca_n": args.alpaca_n,
        **({"ember_delta_embed": delta_embed} if cfg.ember_step.enabled else {}),
        "cells": cells,
    }, indent=2))
    log.info("%s: done, wrote %d cells to %s", concept, len(cells), out_dir)
    return loaded


def _load_model_and_baselines(args: argparse.Namespace, cfg, loaded: Dict[str, Any]) -> None:
    log.info("loading model %s", cfg.model_name)
    hf_model, tokenizer = load_hf_model(cfg.model_name)
    alpaca_bs = cfg.eval.alpaca_batch_size or 50
    baselines = {
        "train_mc": eval_mod.get_baselines(
            hf_model, "train_mc", model_name=cfg.model_name, tokenizer=tokenizer,
            alpaca_batch_size=alpaca_bs, required_concepts=args.concept,
            baseline_out_dir=ARCHIVE_BASELINE_DIR,
        ),
    }
    loaded["hf_model"] = hf_model
    loaded["tokenizer"] = tokenizer
    loaded["baselines"] = baselines
    loaded["alpaca_bs"] = alpaca_bs


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = parse_args()
    resolve_variant_defaults(args)
    log.configure()
    cfg = load_run_config(args)

    loaded: Dict[str, Any] = {"hf_model": None}
    for concept in args.concept:
        loaded = run_concept(args, cfg, concept, loaded)


if __name__ == "__main__":
    main()
