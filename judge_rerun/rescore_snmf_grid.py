#!/usr/bin/env python3
"""Re-score the SNMF grid's invalid cells with the fixed MC evaluator (judge_rerun/mc_fixed.py).

The SNMF grid (ember/erasure/pipeline.py, 144 cells per concept) scored MMLU /
QA_train / SimDom_train with a 4-token (MMLU: 6) budget, so a cut-off answer
counted as wrong. This script re-applies each affected cell with the same edit
code Pass 1 used (SNMFMethod + pass1_snmf_top3._build_hp), re-scores all three
sets with a 32-token budget and the fixed parser, and saves every per-question
record (the grid kept only totals).

Which cells: by default every cell with any invalid answer in a set the grid
actually measured (``--cells all`` re-scores all 144). A cell with no invalid
answer keeps its grid scores.

Direction of the fix: a recovered QA answer can only raise QA accuracy, so the
cell's efficacy drops; a recovered MMLU / SimDom answer can only raise its
specificity. The new grid (re-scored cells replaced, the rest as is) is then
re-ranked exactly as Pass 1 ranked it: the grid's min_mmlu gate on mmlu_frac
(harmonic 0; max_qa_acc is never applied in MC mode), top 3 by `harmonic` via
io.topk_per_concept. Denominators are the ones the grid used
(archive_gemini/data/baselines, MMLU 0.52), so only the scoring changes.

Output, per concept (never writes into results/, archive_gemini/ or judge_rerun/SNMF/pass*_outputs):
    judge_rerun/SNMF/rescore_mc/<model>/rank<r>/seed<s>/<Concept>/
        records/cell_<row>.json   per-question records (raw output, letters, old-equivalent)
        grid_rescored.csv         one row per re-scored cell: HP, new metrics in the
                                  grid's column names, grid_* (old) and old_* (old-equivalent
                                  from this run), features_match, control_ok
        compare.json              old vs new top 3 / pick, cells needing Alpaca + judging,
                                  near-tie warning

Determinism control: the first 4 (6) generated tokens equal the old run, so
old_* must reproduce the grid row for every set the grid measured (a gated cell
never measured QA / SimDom). Mismatches are logged as CONTROL FAILED.

--with-ember (SNMF+EMBER): the same on results/snmf_ef grids with
configs/snmf_ember_gemma.yaml, under judge_rerun/SNMF_EMBER/. The EMBER delta
comes from the repaired EMBER picks (judge_rerun/Ember/pass2_rescored). A concept
whose delta changed gets a FRESH grid: all 144 cells at the new delta, MMLU first
and QA / SimDom only for cells that pass the MMLU gate, as the grid did; there is
no old run at that delta, so no control (control_applicable = False).

Usage:
    python -m judge_rerun.rescore_snmf_grid --concept Uranium "Culture of Greece"
    python -m judge_rerun.rescore_snmf_grid --with-ember --concept Heroin Gun
    python -m judge_rerun.rescore_snmf_grid --concept Uranium --compare-only
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

import numpy as np
import pandas as pd

from ember.erasure import io, log
from ember.erasure.config import load_yaml
from ember.evals.harmonic import harmonic_mean
from ember.utils import _safe_concept, _safe_model_name
from judge_rerun.mc_fixed import MC_MAX_NEW_TOKENS, MMLU_MAX_NEW_TOKENS
from judge_rerun.pass1_snmf_top3 import _build_hp
from judge_rerun.pass2_delta_search import _combine_alpaca_metrics, _load_baselines
from judge_rerun.pass2_judge_baseline import _judge_id

SETS = ("mmlu", "qa", "simdom")
CHANCE = 0.25
CONFIGS = {False: ROOT_DIR / "configs" / "snmf_gemma.yaml",
           True: ROOT_DIR / "configs" / "snmf_ember_gemma.yaml"}
ARCHIVE_BASELINE_DIR = ROOT_DIR / "archive_gemini" / "data" / "baselines"
SNMF_DIR = ROOT_DIR / "judge_rerun" / "SNMF"
SNMF_EMBER_DIR = ROOT_DIR / "judge_rerun" / "SNMF_EMBER"
VARIANT_DIRS = {False: SNMF_DIR, True: SNMF_EMBER_DIR}
# Final EMBER deltas after the MC scoring repair: SNMF+EMBER applies these under every cell.
EMBER_PICKS = (ROOT_DIR / "judge_rerun" / "Ember" / "pass2_rescored" / "google_gemma-2-2b-it"
               / "rank100" / "seed42" / "best_deltas_summary.csv")
EMBER_BASELINE_PARENT = ROOT_DIR / "judge_rerun" / "Ember" / "pass1_outputs"  # Qwen Alpaca baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept", required=True, nargs="+")
    p.add_argument("--with-ember", action="store_true",
                   help="SNMF+EMBER: configs/snmf_ember_gemma.yaml, results/snmf_ef grids, "
                        "judge_rerun/SNMF_EMBER, EMBER delta from the repaired EMBER picks")
    p.add_argument("--cells", choices=("invalid", "all"), default="invalid")
    p.add_argument("--compare-only", action="store_true",
                   help="skip the GPU part, recompute compare.json from grid_rescored.csv")
    p.add_argument("--overwrite", action="store_true", help="re-score cells already on disk")
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--fluency-prompt", default="original")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--pass1-dir", default=None)
    p.add_argument("--pass2-dir", default=None)
    p.add_argument("--out-dir", default=None)
    return resolve_dirs(p.parse_args())


def resolve_dirs(args: argparse.Namespace) -> argparse.Namespace:
    base = VARIANT_DIRS[bool(getattr(args, "with_ember", False))]
    for attr, sub in (("pass1_dir", "pass1_outputs"), ("pass2_dir", "pass2_outputs"), ("out_dir", "rescore_mc")):
        if getattr(args, attr, None) is None:
            setattr(args, attr, str(base / sub))
    return args


def load_cfg(with_ember: bool = False):
    cfg = load_yaml(CONFIGS[with_ember])
    cfg.train_eval = "mc"
    cfg.run_tests_after_train = False
    if bool(cfg.ember_step.enabled) != with_ember:
        raise ValueError(f"{CONFIGS[with_ember]} has ember_step.enabled={cfg.ember_step.enabled}, "
                         f"expected {with_ember}")
    return cfg


def target_delta(cfg, concept: str) -> float:
    """EMBER delta under every cell: 0 for plain SNMF, else the repaired EMBER pick."""
    if not cfg.ember_step.enabled:
        return 0.0
    df = pd.read_csv(EMBER_PICKS)
    row = df[df["concept"] == concept]
    if row.empty:
        raise KeyError(f"no repaired EMBER delta for {concept!r} in {EMBER_PICKS}")
    return float(row["best_delta_embed"].iloc[0])


def grid_delta(grid: pd.DataFrame) -> float:
    found = sorted(set(pd.to_numeric(grid["delta_embed"]).round(6)))
    if len(found) != 1:
        raise ValueError(f"grid mixes delta_embed values {found}")
    return float(found[0])


def _rel(cfg) -> Path:
    return Path(_safe_model_name(cfg.model_name)) / f"rank{cfg.rank}" / f"seed{cfg.seed}"


def grid_path(cfg, concept: str) -> Path:
    """The grid Pass 1 used: results/snmf/... if it ran fresh, else the archived one."""
    live = io.train_dir("snmf", cfg.model_name, cfg.rank, cfg.seed, cfg.train_eval,
                        embed_step_enabled=bool(cfg.ember_step.enabled))
    for root in (live, ROOT_DIR / "archive_gemini" / live.relative_to(ROOT_DIR)):
        path = root / _safe_concept(concept) / "hps.csv"
        if path.exists():
            return path
    raise FileNotFoundError(f"no SNMF grid for {concept!r}")


def load_grid(cfg, concept: str):
    """(grid, old_delta, new_delta, fresh). For SNMF+EMBER, a concept whose repaired
    EMBER delta differs from the one its grid was run with is 'fresh': the same 144
    SNMF cells must be scored again at the new delta, so delta_embed is replaced."""
    grid = pd.read_csv(grid_path(cfg, concept))
    old, new = grid_delta(grid), target_delta(cfg, concept)
    fresh = abs(old - new) > 1e-9
    if fresh:
        grid = grid.copy()
        grid["delta_embed"] = new
    return grid, old, new, fresh


def mc_bases(concept: str, model_name: str) -> Dict[str, float]:
    """Pristine denominators the grid used (archive_gemini/data/baselines)."""
    d = ARCHIVE_BASELINE_DIR / _safe_model_name(model_name)
    per = lambda f: json.loads((d / f).read_text())["meta"]["per_concept"][concept]["accuracy_generation"]
    mmlu = json.loads((d / "baseline_mmlu_train.json").read_text())["metrics"]["accuracy_generation"]
    return {"mmlu": float(mmlu), "qa": float(per("baseline_qa_train_mc.json")),
            "simdom": float(per("baseline_simdom_train_mc.json"))}


def _frac(acc: float, base: float) -> float:
    if base - CHANCE == 0:
        return 0.0
    return min(1.0, max((acc - CHANCE) / (base - CHANCE), 0.0))


def grid_metrics(mmlu: float, qa: float, sim: float, bases: Dict[str, float],
                 min_mmlu: float, max_qa: float) -> Dict[str, Any]:
    """orchestrator.py's metric block with the grid's early-stop gate.

    Only min_mmlu applies: orchestrator.py checks max_qa_acc in the open-QA branch
    only, never for MC, so the train_mc grids were not gated on QA (checked: this
    reproduces every grid row's harmonic)."""
    mf, qf, sf = _frac(mmlu, bases["mmlu"]), _frac(qa, bases["qa"]), _frac(sim, bases["simdom"])
    if mf < min_mmlu:
        return {"efficacy": 0.0, "specificity": 0.0, "harmonic": 0.0, "gate": "min_mmlu"}
    eff, spec = 1.0 - qf, harmonic_mean([mf, sf])
    return {"efficacy": eff, "specificity": spec, "harmonic": harmonic_mean([eff, spec]), "gate": ""}


def measured_sets(row: pd.Series, bases: Dict[str, float], min_mmlu: float, max_qa: float) -> List[str]:
    """Sets the grid actually evaluated for this cell (gates stop early)."""
    if _frac(float(row["mmlu_acc"]), bases["mmlu"]) < min_mmlu:
        return ["mmlu"]
    return list(SETS)


# ---------------------------------------------------------------------------- #
# GPU part                                                                      #
# ---------------------------------------------------------------------------- #

class _Scorer:
    def __init__(self, cfg) -> None:
        from ember.erasure.methods import base as methods_base
        from ember.erasure.model_loader import load_hf_model
        from ember.evals.mmlu import load_mmlu_items
        from ember.evals.model_wrap import ensure_wrapped_model
        from ember.local_datasets import load_mmlu_indices

        self.cfg = cfg
        self.method = methods_base.get("snmf")
        self.model, self.tokenizer = load_hf_model(cfg.model_name)
        self.tm = ensure_wrapped_model(self.model, self.tokenizer)
        self.snap = self.method.snapshot(self.model)
        self.mmlu_items = load_mmlu_items(load_mmlu_indices("train"))

    def concept_items(self, concept: str):
        from ember.evals.mc import prepare_mc_items
        from ember.local_datasets import load_mc_qa_items
        qa = prepare_mc_items(load_mc_qa_items("qa_train", concept=concept), seed=self.cfg.seed)
        sim = prepare_mc_items(load_mc_qa_items("simdom_train", concept=concept), seed=self.cfg.seed)
        if not qa or not sim:
            raise ValueError(f"missing qa_train / simdom_train items for {concept!r}")
        return qa, sim

    def score_cell(self, concept: str, row: pd.Series, qa_items, sim_items,
                   measured: List[str], bases: Dict[str, float], min_mmlu: float) -> Dict[str, Any]:
        """Score MMLU, then QA + SimDom -- except, like the grid, for a cell the grid
        gated on MMLU that is still gated with the fixed MMLU score: its harmonic is 0
        whatever QA / SimDom are, and these broken edits take ~3 min per set to score."""
        from judge_rerun.mc_fixed import evaluate_mc_fixed, evaluate_mmlu_fixed
        self.method.restore(self.model, self.snap)
        hp = _build_hp(row, self.method)
        info = self.method.apply(self.model, self.tokenizer, hp, concept, self.cfg)
        sets = {"mmlu": evaluate_mmlu_fixed(self.tm, self.mmlu_items)}
        still_gated = (measured == ["mmlu"]
                       and _frac(sets["mmlu"]["acc"], bases["mmlu"]) < min_mmlu)
        if still_gated:
            skipped = {"acc": 0.0, "invalid_rate": 0.0, "old_acc": 0.0, "old_invalid_rate": 0.0,
                       "n": 0, "records": [], "skipped": "still min_mmlu-gated"}
            sets["qa"], sets["simdom"] = dict(skipped), dict(skipped)
        else:
            sets["qa"] = evaluate_mc_fixed(self.tm, qa_items)
            sets["simdom"] = evaluate_mc_fixed(self.tm, sim_items)
        self.method.restore(self.model, self.snap)
        return {"hp": {k: (v.item() if hasattr(v, "item") else v) for k, v in hp.items()},
                "edit_info": {k: int(info[k]) for k in ("k_features_mlp_in", "k_features_mlp_out")},
                "sets": sets}


def _row(idx: int, grid_row: pd.Series, res: Dict[str, Any], measured: List[str],
         seconds: float, fresh: bool = False) -> Dict[str, Any]:
    row: Dict[str, Any] = {"row": idx}
    row.update({c: grid_row[c] for c in io.SNMF_HP_COLUMNS + ["delta_embed"] if c in grid_row.index})
    row.update(res["edit_info"])
    row["features_match"] = all(int(grid_row[k]) == v for k, v in res["edit_info"].items())
    control = row["features_match"]
    for s in SETS:
        r = res["sets"][s]
        row[f"{s}_acc"], row[f"{s}_invalid"] = r["acc"], r["invalid_rate"]
        row[f"old_{s}_acc"], row[f"old_{s}_invalid"] = r["old_acc"], r["old_invalid_rate"]
        row[f"grid_{s}_acc"], row[f"grid_{s}_invalid"] = float(grid_row[f"{s}_acc"]), float(grid_row[f"{s}_invalid"])
        if s in measured and not fresh:
            control &= (abs(r["old_acc"] - row[f"grid_{s}_acc"]) < 1e-9
                        and abs(r["old_invalid_rate"] - row[f"grid_{s}_invalid"]) < 1e-9)
    row["measured_sets"] = ",".join(measured)
    # Fresh delta: grid_* are the old run at another delta, nothing to reproduce;
    # only the SNMF feature counts (independent of delta) are checked.
    row["control_applicable"] = not fresh
    row["control_ok"] = control
    row["mc_max_new_tokens"], row["mmlu_max_new_tokens"] = MC_MAX_NEW_TOKENS, MMLU_MAX_NEW_TOKENS
    row["wall_time_s"] = round(seconds, 1)
    return row


def rescore_concept(scorer: _Scorer, args: argparse.Namespace, concept: str) -> None:
    cfg = scorer.cfg
    grid, old_delta, new_delta, fresh = load_grid(cfg, concept)
    bases = mc_bases(concept, cfg.model_name)
    min_mmlu, max_qa = float(cfg.eval.min_mmlu), float(cfg.eval.max_qa_acc)
    out_dir = Path(args.out_dir) / _rel(cfg) / _safe_concept(concept)
    rec_dir = out_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "grid_rescored.csv"
    rows = ({int(r["row"]): r.to_dict() for _, r in pd.read_csv(csv_path).iterrows()}
            if csv_path.exists() else {})

    todo = []
    for idx, g in grid.iterrows():
        if fresh:
            # every cell, MMLU first; QA / SimDom skipped while MMLU-gated (as the grid did)
            todo.append((idx, g, ["mmlu"]))
            continue
        measured = measured_sets(g, bases, min_mmlu, max_qa)
        if args.cells == "all" or any(float(g[f"{s}_invalid"]) > 0 for s in measured):
            todo.append((idx, g, measured))
    log.info("%s: re-scoring %d/%d cells%s", concept, len(todo), len(grid),
             f" (FRESH grid: delta_embed {old_delta:g} -> {new_delta:g})" if fresh
             else (f" (delta_embed {new_delta:g})" if cfg.ember_step.enabled else ""))

    scorer.method.on_concept_start(scorer.model, concept, cfg)
    qa_items, sim_items = scorer.concept_items(concept)
    n_fail = 0
    for n, (idx, g, measured) in enumerate(todo):
        if idx in rows and not args.overwrite:
            continue
        log.set_context(stage=f"rescore cell={idx}")
        t0 = time.time()
        res = scorer.score_cell(concept, g, qa_items, sim_items, measured, bases, min_mmlu)
        row = _row(idx, g, res, measured, time.time() - t0, fresh=fresh)
        (rec_dir / f"cell_{idx}.json").write_text(json.dumps(
            {"concept": concept, "row": idx, "hp": res["hp"], **res["edit_info"],
             "mc_max_new_tokens": MC_MAX_NEW_TOKENS, "mmlu_max_new_tokens": MMLU_MAX_NEW_TOKENS,
             **{s: res["sets"][s] for s in SETS}},
            ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        rows[idx] = row
        pd.DataFrame([rows[k] for k in sorted(rows)]).to_csv(csv_path, index=False)
        log.info("[%d/%d] cell=%d qa %.2f->%.2f (inv %.2f->%.2f) simdom %.2f->%.2f mmlu %.2f->%.2f "
                 "control_ok=%s (%.0fs)", n + 1, len(todo), idx,
                 row["grid_qa_acc"], row["qa_acc"], row["grid_qa_invalid"], row["qa_invalid"],
                 row["grid_simdom_acc"], row["simdom_acc"], row["grid_mmlu_acc"], row["mmlu_acc"],
                 row["control_ok"], row["wall_time_s"])
        if not row["control_ok"]:
            n_fail += 1
            log.warning("CONTROL FAILED for %s cell=%d (measured sets %s)", concept, idx, measured)
    scorer.method.on_concept_end(scorer.model, concept, cfg)
    log.info("%s: done, %d control failures this run", concept, n_fail)


# ---------------------------------------------------------------------------- #
# CPU part: old vs new selection                                                #
# ---------------------------------------------------------------------------- #

KEY = ["delta_embed", "delta_in", "layer_lo_in", "layer_hi_in", "delta_out", "layer_lo_out", "layer_hi_out"]


def _key(r) -> tuple:
    return tuple(round(float(r[k]), 6) for k in KEY)


def _rank(grid: pd.DataFrame, bases, min_mmlu, max_qa, top_k: int) -> pd.DataFrame:
    df = grid.copy()
    m = [grid_metrics(float(r.mmlu_acc), float(r.qa_acc), float(r.simdom_acc), bases, min_mmlu, max_qa)
         for r in df.itertuples()]
    for k in ("efficacy", "specificity", "harmonic", "gate"):
        df[k] = [x[k] for x in m]
    return io.topk_per_concept(df, k=len(df), primary="harmonic")


def compare_concept(args: argparse.Namespace, cfg, concept: str) -> Dict[str, Any]:
    rel = _rel(cfg)
    grid = pd.read_csv(grid_path(cfg, concept))
    bases = mc_bases(concept, cfg.model_name)
    min_mmlu, max_qa = float(cfg.eval.min_mmlu), float(cfg.eval.max_qa_acc)
    out_dir = Path(args.out_dir) / rel / _safe_concept(concept)
    resc = pd.read_csv(out_dir / "grid_rescored.csv")
    fresh_grid, old_delta, new_delta, fresh = load_grid(cfg, concept)
    if fresh and len(resc) < len(grid):
        log.warning("%s: fresh grid only %d/%d cells scored; comparison is partial",
                    concept, len(resc), len(grid))

    # Sanity: our formula reproduces the grid's own harmonic on the old scores.
    old_ranked = _rank(grid, bases, min_mmlu, max_qa, args.top_k)
    formula_err = float(np.max(np.abs(old_ranked["harmonic"].values
                                      - grid.loc[old_ranked.index, "harmonic"].values)))

    new_grid = fresh_grid.copy()
    new_grid["rescored"] = False
    for _, r in resc.iterrows():
        i = int(r["row"])
        for s in SETS:
            new_grid.loc[i, f"{s}_acc"] = r[f"{s}_acc"]
            new_grid.loc[i, f"{s}_invalid"] = r[f"{s}_invalid"]
        new_grid.loc[i, "rescored"] = True
    new_ranked = _rank(new_grid, bases, min_mmlu, max_qa, args.top_k)

    # Old top 3 = what Pass 1 actually generated Alpaca for (Ancient Rome: loop screen).
    pass1 = json.loads((Path(args.pass1_dir) / rel / _safe_concept(concept) / "top3_hps.json").read_text())
    old_top = [_key(c) for c in pass1["cells"]]
    judge_id = _judge_id(args.judge_model)
    cm = pd.read_csv(Path(args.pass2_dir) / rel / _safe_concept(concept) / f"cell_metrics__{judge_id}.csv")
    alpaca_by_key = {old_top[int(r["cell"])]: (float(r["alpaca_instr"]), float(r["alpaca_flu"]))
                     for _, r in cm.iterrows()}
    alp_bases = _load_baselines(EMBER_BASELINE_PARENT / rel / "_baseline", concept, judge_id, args.fluency_prompt)

    top = new_ranked.head(args.top_k)
    top_rows, needs = [], []
    for i, r in top.iterrows():
        k = _key(r)
        entry = {"row": int(i), "hp": dict(zip(KEY, k)), "harmonic": float(r["harmonic"]),
                 "efficacy": float(r["efficacy"]), "specificity": float(r["specificity"]),
                 "rescored": bool(r["rescored"]), "in_old_top3": k in old_top}
        if k in alpaca_by_key:
            instr, flu = alpaca_by_key[k]
            entry.update(_combine_alpaca_metrics(entry["efficacy"], entry["specificity"],
                                                 instr, flu, alp_bases, float("-inf")))
        else:
            entry["harmonic_alpaca"] = float("nan")
            needs.append(int(i))
        top_rows.append(entry)

    best = None
    if not needs:
        best = max(top_rows, key=lambda e: e["harmonic_alpaca"])  # ties: first in rank order
    # Near tie at the top-3 boundary between a re-scored and an old (grid-run) cell:
    # one QA question moves efficacy by 0.02 / (qa_base - 0.25).
    third, fourth = new_ranked.iloc[args.top_k - 1], new_ranked.iloc[args.top_k]
    one_q = 0.02 / (bases["qa"] - CHANCE)
    near_tie = bool(abs(float(third["harmonic"]) - float(fourth["harmonic"])) < one_q
                    and bool(third["rescored"]) != bool(fourth["rescored"]))
    result = {
        "concept": concept,
        "grid": str(grid_path(cfg, concept).relative_to(ROOT_DIR)),
        "loop_screen_concept": (Path(args.pass1_dir) / rel / _safe_concept(concept) / "screen").exists(),
        "formula_max_abs_err_vs_grid": formula_err,
        "n_rescored": int(len(resc)),
        "control_ok_all": bool(resc["control_ok"].all()),
        "n_control_failed": int((~resc["control_ok"].astype(bool)).sum()),
        "delta_embed_old": old_delta, "delta_embed_new": new_delta, "fresh_grid": fresh,
        "features_match_all": bool(resc["features_match"].all()),
        "old_top3_rows": [int(i) for i in grid.index if _key(grid.loc[i]) in old_top],
        "new_top3": top_rows,
        "top3_changed": sorted(_key(new_ranked.loc[e["row"]]) for e in top_rows) != sorted(old_top),
        "needs_alpaca_and_judging_rows": needs,
        "best_row": None if best is None else best["row"],
        "best_harmonic_alpaca": None if best is None else best["harmonic_alpaca"],
        "boundary": {"third_row": int(third.name), "third_harmonic": float(third["harmonic"]),
                     "fourth_row": int(fourth.name), "fourth_harmonic": float(fourth["harmonic"]),
                     "one_question_efficacy": one_q, "near_tie_mixed_runs": near_tie},
    }
    (out_dir / "compare.json").write_text(json.dumps(result, indent=2, default=float))
    log.info("%s: control_ok=%s (%d failed) top3_changed=%s needs judging rows=%s near_tie=%s%s",
             concept, result["control_ok_all"], result["n_control_failed"], result["top3_changed"],
             needs, near_tie, " [loop-screen concept: check by hand]" if result["loop_screen_concept"] else "")
    return result


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = parse_args()
    log.configure()
    method = "snmf_ef" if args.with_ember else "snmf"
    log.set_context(method=method, stage="rescore_mc")
    cfg = load_cfg(args.with_ember)
    if not args.compare_only:
        scorer = _Scorer(cfg)
        for c in args.concept:
            log.set_context(method=method, concept=c, stage="rescore_mc")
            rescore_concept(scorer, args, c)
    for c in args.concept:
        log.set_context(method=method, concept=c, stage="compare")
        compare_concept(args, cfg, c)


if __name__ == "__main__":
    main()
