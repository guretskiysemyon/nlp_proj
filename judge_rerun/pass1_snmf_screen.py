#!/usr/bin/env python3
"""SNMF Pass 1 with a loop screen: a wider candidate pool for concepts whose
plain top-3 cells all break free-text generation (so far: Ancient Rome).

pass1_snmf_top3.py keeps the top 3 grid cells by the judge-free `harmonic`.
For Ancient Rome all three collapse into "Roman Roman Roman ..." (45-49 of 50
Alpaca answers), because the judge-free score only sees multiple-choice tests.
This script instead:

  1. ranks the concept's FULL 144-cell grid (hps.csv) exactly like Pass 1
     does (io.topk_per_concept, primary="harmonic", same tie-breakers, no
     extra filter);
  2. walks down that ranking generating the first --alpaca-n Alpaca answers
     per cell (Gemma, greedy -- the same generation call as Pass 1), and
     treats a cell whose answers are word-for-word identical to a
     higher-ranked cell's as a DUPLICATE (same edit: e.g. `layer_hi_in` 8 vs
     12 when layers 9-12 hold no concept features);
  3. counts each distinct cell's LOOPING answers: an answer loops if one
     word containing a letter or digit is repeated --loop-run (8) or more
     times in a row (punctuation-only tokens, e.g. ASCII-art "|||", are
     ignored). The pristine model has 0/150 such answers. A cell with more
     than --max-loop-answers (5, i.e. >10%) looping answers is dropped;
  4. stops as soon as --keep (3) distinct, non-looping cells are found (or
     after --max-rows grid rows). Every row below the third survivor ranks
     lower, so it could never replace one: generating it would change nothing.

The survivors are written in Pass 1's own format (top3_hps.json +
alpaca_completions/cell_0..2.json), so judge_rerun/pass2_snmf_judge.py judges
them unchanged. Everything the screen saw goes to screen/: one completions
file per walked grid row and screen_report.{json,csv}.

Before running, move the concept's previous (unscreened) top3_hps.json and
alpaca_completions/ aside -- this script refuses to overwrite them.

--with-ember screens an SNMF+EMBER grid instead (results/snmf_ef/, outputs in
judge_rerun/SNMF_EMBER/pass1_outputs), as in pass1_snmf_top3.py.

Usage:
    python -m judge_rerun.pass1_snmf_screen --concept "Ancient Rome"
    python -m judge_rerun.pass1_snmf_screen --with-ember --concept "<Concept>"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import pandas as pd

from ember.erasure import io, log
from ember.erasure.methods import base as methods_base
from ember.evals.alpaca import load_alpaca_eval_local
from ember.evals.model_wrap import ensure_wrapped_model
from ember.local_datasets import load_alpaca_indices
from ember.utils import _safe_concept
from judge_rerun.pass1_snmf_top3 import (
    _build_hp, _grid_dir, _load_model_and_baselines, _out_concept_dir, check_grid_delta,
    ember_delta, load_run_config, resolve_variant_defaults)

_FAST_COLS = ("mmlu_acc", "qa_acc", "simdom_acc", "efficacy", "specificity", "harmonic")
_HAS_ALNUM = re.compile(r"[^\W_]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept", required=True, nargs="+",
                    help="one or more concepts, run back-to-back sharing one model load")
    p.add_argument("--with-ember", action="store_true",
                    help="screen the SNMF+EMBER grid (switches --config / --out-dir defaults)")
    p.add_argument("--config", default=None,
                    help="default configs/snmf_gemma.yaml, or configs/snmf_ember_gemma.yaml "
                         "with --with-ember")
    p.add_argument("--model-name", default=None, help="override config's model_name")
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--keep", type=int, default=3,
                    help="stop once this many distinct, non-looping cells are found; "
                         "they are handed to Pass 2")
    p.add_argument("--max-loop-answers", type=int, default=5,
                    help="a cell with MORE looping answers than this is dropped")
    p.add_argument("--loop-run", type=int, default=8,
                    help="consecutive repeats of one word that make an answer 'looping'")
    p.add_argument("--max-rows", type=int, default=40,
                    help="safety limit: stop walking the ranking after this many grid "
                         "rows even if fewer than --keep cells were found")
    p.add_argument("--alpaca-n", type=int, default=50)
    p.add_argument("--alpaca-batch-size", type=int, default=8)
    p.add_argument("--out-dir", default=None,
                    help="default judge_rerun/SNMF/pass1_outputs, or "
                         "judge_rerun/SNMF_EMBER/pass1_outputs with --with-ember")
    return p.parse_args()


def is_looping(text: str, run_len: int) -> bool:
    """True if one word containing a letter/digit repeats ``run_len``+ times in a row."""
    words = text.split()
    run = 1
    for prev, cur in zip(words, words[1:]):
        run = run + 1 if (cur == prev and _HAS_ALNUM.search(cur)) else 1
        if run >= run_len:
            return True
    return False


def _grid_hps_path(cfg, concept: str) -> Path:
    """The concept's full grid: the live run's hps.csv, else the archived one."""
    live = _grid_dir(cfg)
    for root in (live, ROOT_DIR / "archive_gemini" / live.relative_to(ROOT_DIR)):
        path = root / _safe_concept(concept) / "hps.csv"
        if path.exists():
            return path
    raise FileNotFoundError(f"no hps.csv for {concept!r} under {live} or its archive_gemini copy")


def _hp_desc(row: pd.Series) -> str:
    return (f"in {row['delta_in']:g}@{int(row['layer_lo_in'])}-{int(row['layer_hi_in'])} / "
            f"out {row['delta_out']:g}@{int(row['layer_lo_out'])}-{int(row['layer_hi_out'])}")


def run_concept(args: argparse.Namespace, cfg, concept: str,
                loaded: Dict[str, Any]) -> Dict[str, Any]:
    log.set_context(method=io.method_dir_name("snmf", bool(cfg.ember_step.enabled)),
                    concept=concept, stage="pass1_snmf_screen")

    out_dir = _out_concept_dir(args, cfg, concept)
    final_path = out_dir / "top3_hps.json"
    if final_path.exists():
        prev = json.loads(final_path.read_text())
        if prev.get("selection") == "loop_screen":
            log.info("%s already screened (%s); skipping", concept, final_path)
            return loaded
        raise RuntimeError(
            f"{final_path} holds an unscreened top-3 selection. Move it and "
            f"{out_dir / 'alpaca_completions'} aside (e.g. into {out_dir / '_unscreened_top3'}) "
            f"before screening.")
    if (out_dir / "alpaca_completions").exists():
        raise RuntimeError(f"{out_dir / 'alpaca_completions'} exists without a top3_hps.json; "
                           f"move it aside before screening.")

    hps_path = _grid_hps_path(cfg, concept)
    grid = io.read_csv_safe(hps_path)
    check_grid_delta(grid, ember_delta(cfg, concept), hps_path)
    ranked = io.topk_per_concept(grid, k=len(grid), primary="harmonic").reset_index(drop=True)
    log.info("%s: %d grid rows from %s", concept, len(ranked), hps_path)

    screen_dir = out_dir / "screen"
    rows_dir = screen_dir / "alpaca_completions"
    rows_dir.mkdir(parents=True, exist_ok=True)

    alpaca_indices = load_alpaca_indices("train")[:args.alpaca_n]
    alpaca_all = load_alpaca_eval_local()
    instructions_full = [ex["instruction"] for ex in alpaca_all if "instruction" in ex]
    instructions = [instructions_full[i] for i in alpaca_indices]

    method = methods_base.get("snmf")
    snap = None
    report: List[Dict[str, Any]] = []
    distinct: List[Dict[str, Any]] = []   # report entries of distinct cells, in rank order
    kept: List[Dict[str, Any]] = []       # distinct AND non-looping, in rank order
    answers_by_rank: Dict[int, List[str]] = {}

    for pos, row in ranked.iterrows():
        if len(kept) >= args.keep or pos >= args.max_rows:
            break
        grid_rank = pos + 1
        row_path = rows_dir / f"row_{grid_rank:03d}.json"
        hp = _build_hp(row, method)

        if row_path.exists():  # resume: generation is deterministic, reuse it
            saved = json.loads(row_path.read_text())
            items, info = saved["items"], saved["info"]
        else:
            if loaded.get("hf_model") is None:
                _load_model_and_baselines(args, cfg, loaded)
            hf_model, tokenizer = loaded["hf_model"], loaded["tokenizer"]
            if snap is None:
                method.on_concept_start(hf_model, concept, cfg)
                snap = method.snapshot(hf_model)
            method.restore(hf_model, snap)
            info = method.apply(hf_model, tokenizer, hp, concept, cfg)
            tm = ensure_wrapped_model(hf_model, tokenizer)
            completions = tm.generate_multiple(
                instructions, max_new_tokens=200, do_sample=False,
                batch_size=args.alpaca_batch_size, verbose=False,
            )
            items = [{"idx": i, "instruction": instr, "completion": comp}
                     for i, (instr, comp) in enumerate(zip(instructions, completions))]
            row_path.write_text(json.dumps({"grid_rank": grid_rank, "hp": hp, "info": info,
                                            "items": items}, ensure_ascii=False, indent=2,
                                           default=str))

        answers = [it["completion"] for it in items]
        duplicate_of: Optional[int] = next(
            (d["grid_rank"] for d in distinct if answers_by_rank[d["grid_rank"]] == answers), None)
        n_loop = sum(is_looping(a, args.loop_run) for a in answers)
        entry: Dict[str, Any] = {
            "grid_rank": grid_rank, "hp_desc": _hp_desc(row),
            **{c: row[c] for c in _FAST_COLS if c in row.index},
            "duplicate_of": duplicate_of,
            "n_loop_answers": n_loop,
            "looping": n_loop > args.max_loop_answers,
            "selected": False,
            "_hp": hp, "_info": info,
        }
        report.append(entry)
        if duplicate_of is None:
            answers_by_rank[grid_rank] = answers
            distinct.append(entry)
            if not entry["looping"]:
                entry["selected"] = True
                kept.append(entry)
        log.info("%s rank %d (%s) harmonic=%.4f loops=%d/%d%s", concept, grid_rank,
                 entry["hp_desc"], float(row["harmonic"]), n_loop, len(answers),
                 f" DUPLICATE of rank {duplicate_of}" if duplicate_of else
                 (" LOOPING" if entry["looping"] else f" KEPT ({len(kept)}/{args.keep})"))

    if snap is not None:
        method.restore(loaded["hf_model"], snap)
        method.on_concept_end(loaded["hf_model"], concept, cfg)

    if len(kept) < args.keep:
        log.warning("%s: only %d distinct non-looping cells in the first %d grid rows (wanted %d)",
                    concept, len(kept), len(report), args.keep)
    if not kept:
        raise RuntimeError(f"{concept}: no cell passed the loop screen; nothing to hand to Pass 2")

    criteria = {
        "ranking": "io.topk_per_concept(hps.csv, primary='harmonic'), no extra filter",
        "duplicate": "word-for-word identical Alpaca answers to a higher-ranked cell",
        "looping_answer": f"one word containing a letter/digit repeated >= {args.loop_run} times in a row",
        "looping_cell": f"more than {args.max_loop_answers} of {args.alpaca_n} answers looping",
        "stop": f"first {args.keep} distinct non-looping cells in ranking order "
                f"(safety limit {args.max_rows} rows)",
        "rows_walked": len(report),
    }
    public = [{k: v for k, v in e.items() if not k.startswith("_")} for e in report]
    (screen_dir / "screen_report.json").write_text(json.dumps(
        {"concept": concept, "grid": str(hps_path.relative_to(ROOT_DIR)), "criteria": criteria,
         "rows": public}, indent=2, default=str))
    pd.DataFrame(public).to_csv(screen_dir / "screen_report.csv", index=False)

    completions_dir = out_dir / "alpaca_completions"
    completions_dir.mkdir(parents=True, exist_ok=True)
    cells: List[Dict[str, Any]] = []
    for i, d in enumerate(kept):
        items = json.loads((rows_dir / f"row_{d['grid_rank']:03d}.json").read_text())["items"]
        (completions_dir / f"cell_{i}.json").write_text(json.dumps(items, ensure_ascii=False, indent=2))
        cells.append({**d["_hp"], **d["_info"],
                      **{c: d[c] for c in _FAST_COLS if c in d}, "grid_rank": d["grid_rank"]})
    final_path.write_text(json.dumps({
        "concept": concept, "top_k": len(cells), "alpaca_n": args.alpaca_n,
        "selection": "loop_screen", "criteria": criteria, "cells": cells,
    }, indent=2, default=str))
    log.info("%s: kept grid ranks %s -> %s", concept, [d["grid_rank"] for d in kept], final_path)
    return loaded


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
