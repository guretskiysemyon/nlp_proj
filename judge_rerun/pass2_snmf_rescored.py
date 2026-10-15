#!/usr/bin/env python3
"""SNMF Pass 1 + Pass 2 on the re-scored grid: final best cell per concept after the MC fix.

judge_rerun/rescore_snmf_grid.py re-scored the SNMF grid's cells with the fixed
evaluator (judge_rerun/SNMF/rescore_mc/). This script finishes the selection:

  stage "generate" (Gemma, 1 GPU): re-rank the corrected grid (min_mmlu gate, top 3
    by `harmonic`, io.topk_per_concept order). Cells already in the old top 3 keep
    their saved Alpaca answers. A cell NEW to the top 3 gets its first --alpaca-n
    Alpaca answers generated (same instructions, order and settings as
    pass1_snmf_top3.py) and the loop check of pass1_snmf_screen.py: more than
    --max-loop-answers looping answers rejects the cell and the next one in the
    ranking moves up (checked the same way). Ancient Rome keeps its loop screen:
    cells its screen_report marked looping / duplicate are skipped.

  stage "judge" (Qwen, 3 GPUs; no GPU if nothing is new): Qwen-judge the new cells'
    answers, reuse the old Qwen means for the others (Alpaca is open-ended text,
    unaffected by the fix), recompute harmonic_alpaca from the NEW efficacy /
    specificity, pick with io.topk_per_concept(k=1, primary="harmonic_alpaca") as
    pass2_snmf_judge.py does (no min_alpaca gate, as SNMF's original validate).

Never writes into results/, archive_gemini/ or judge_rerun/SNMF/pass*_outputs.
Output: judge_rerun/SNMF/pass2_rescored/<model>/rank<r>/seed<s>/<Concept>/
    new_cells.json, alpaca_completions/row_<i>.json (generate);
    cell_metrics__<judge>.csv, best_hp__<judge>.json, alpaca_judged__<judge>.json (judge)
and best_hps_summary.csv.

--with-ember: the same for SNMF+EMBER (judge_rerun/SNMF_EMBER/). For a concept
whose EMBER delta changed, every top-3 cell is new (different edit), so all
three get answers generated and judged.

Usage:
    python -m judge_rerun.pass2_snmf_rescored --stage generate --concept "COVID-19 pandemic"
    python -m judge_rerun.pass2_snmf_rescored --with-ember --stage generate --concept Gun
    python -m judge_rerun.pass2_snmf_rescored --stage judge --concept <all 18>
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
from ember.utils import _safe_concept
from judge_rerun.pass1_snmf_screen import _hp_desc, is_looping
from judge_rerun.pass2_delta_search import _combine_alpaca_metrics, _judge_one_delta, _load_baselines
from judge_rerun.pass2_judge_baseline import _judge_id
from judge_rerun.rescore_snmf_grid import (EMBER_BASELINE_PARENT, KEY, SETS, VARIANT_DIRS, _key, _rank,
                                           _rel, load_cfg, load_grid, mc_bases)

HP_COLS = ("delta_embed", "w_mode", "feature_source", "ratio_thresh", "coverage_thresh",
           "delta_in", "layer_lo_in", "layer_hi_in", "k_features_mlp_in",
           "delta_out", "layer_lo_out", "layer_hi_out", "k_features_mlp_out")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("generate", "judge"), required=True)
    p.add_argument("--concept", required=True, nargs="+")
    p.add_argument("--with-ember", action="store_true",
                   help="SNMF+EMBER: judge_rerun/SNMF_EMBER/, results/snmf_ef grids, repaired EMBER delta")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--alpaca-n", type=int, default=50)
    p.add_argument("--alpaca-batch-size", type=int, default=8)   # pass1_snmf_top3.py default
    p.add_argument("--max-loop-answers", type=int, default=5)    # pass1_snmf_screen.py defaults
    p.add_argument("--loop-run", type=int, default=8)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--judge-batch-size", type=int, default=4)
    p.add_argument("--fluency-prompt", default="original")
    p.add_argument("--pass1-dir", default=None)
    p.add_argument("--pass2-dir", default=None)
    p.add_argument("--rescore-dir", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    base = VARIANT_DIRS[args.with_ember]
    for attr, sub in (("pass1_dir", "pass1_outputs"), ("pass2_dir", "pass2_outputs"),
                      ("rescore_dir", "rescore_mc"), ("out_dir", "pass2_rescored")):
        if getattr(args, attr) is None:
            setattr(args, attr, str(base / sub))
    return args


class Concept:
    """Corrected ranking + what is known about each cell's Alpaca answers."""

    def __init__(self, args: argparse.Namespace, cfg, concept: str) -> None:
        self.name, self.args, self.cfg = concept, args, cfg
        rel = _rel(cfg)
        self.out_dir = Path(args.out_dir) / rel / _safe_concept(concept)
        (self.out_dir / "alpaca_completions").mkdir(parents=True, exist_ok=True)
        self.pass1_dir = Path(args.pass1_dir) / rel / _safe_concept(concept)

        # SNMF+EMBER with a changed EMBER delta: delta_embed is the new one, and every
        # cell is re-scored, so no old Alpaca answers match (KEY includes delta_embed).
        grid, _, _, self.fresh = load_grid(cfg, concept)
        resc = pd.read_csv(Path(args.rescore_dir) / rel / _safe_concept(concept) / "grid_rescored.csv")
        for _, r in resc.iterrows():
            for s in SETS:
                grid.loc[int(r["row"]), f"{s}_acc"] = r[f"{s}_acc"]
                grid.loc[int(r["row"]), f"{s}_invalid"] = r[f"{s}_invalid"]
        self.ranked = _rank(grid, mc_bases(concept, cfg.model_name),
                            float(cfg.eval.min_mmlu), float(cfg.eval.max_qa_acc), args.top_k)

        pass1 = json.loads((self.pass1_dir / "top3_hps.json").read_text())
        self.old_cells = {_key(c): i for i, c in enumerate(pass1["cells"])}  # key -> old cell index
        screen = self.pass1_dir / "screen" / "screen_report.csv"
        self.screen = {}
        if screen.exists():
            for r in pd.read_csv(screen).itertuples():
                self.screen[r.hp_desc] = "loop" if r.looping else ("dup" if pd.notna(r.duplicate_of) else "ok")
        nc = self.out_dir / "new_cells.json"
        self.new_cells: Dict[str, Any] = json.loads(nc.read_text()) if nc.exists() else {}

    def save_new_cells(self) -> None:
        (self.out_dir / "new_cells.json").write_text(json.dumps(self.new_cells, indent=2, default=float))

    def walk(self) -> tuple:
        """Final top 3 in rank order, and the first new cell still lacking answers (or None)."""
        top: List[int] = []
        for i, r in self.ranked.iterrows():
            if r["harmonic"] <= 0:
                break
            status = self.screen.get(_hp_desc(r))
            if status in ("loop", "dup"):
                continue
            if _key(r) not in self.old_cells:
                info = self.new_cells.get(str(i))
                if info is None:
                    return top, int(i)
                if info["looping"]:
                    continue
            top.append(int(i))
            if len(top) == self.args.top_k:
                break
        return top, None


# ---------------------------------------------------------------------------- #
# stage generate                                                                #
# ---------------------------------------------------------------------------- #

def stage_generate(args: argparse.Namespace, cfg, concepts: List[str]) -> None:
    from ember.erasure.methods import base as methods_base
    from ember.erasure.model_loader import load_hf_model
    from ember.evals.alpaca import load_alpaca_eval_local
    from ember.evals.model_wrap import ensure_wrapped_model
    from ember.local_datasets import load_alpaca_indices
    from judge_rerun.pass1_snmf_top3 import _build_hp

    alpaca_all = load_alpaca_eval_local()
    full = [ex["instruction"] for ex in alpaca_all if "instruction" in ex]
    instructions = [full[i] for i in load_alpaca_indices("train")[:args.alpaca_n]]
    model = tokenizer = tm = snap = None
    method = methods_base.get("snmf")

    for concept in concepts:
        log.set_context(method="snmf_ef" if cfg.ember_step.enabled else "snmf", concept=concept, stage="pass2_rescored:generate")
        c = Concept(args, cfg, concept)
        started = False
        while True:
            top, need = c.walk()
            if need is None:
                break
            if model is None:
                model, tokenizer = load_hf_model(cfg.model_name)
                tm = ensure_wrapped_model(model, tokenizer)
                snap = method.snapshot(model)
            if not started:
                method.on_concept_start(model, concept, cfg)
                started = True
            row = c.ranked.loc[need]
            method.restore(model, snap)
            hp = _build_hp(row, method)
            info = method.apply(model, tokenizer, hp, concept, cfg)
            comps = tm.generate_multiple(instructions, max_new_tokens=200, do_sample=False,
                                         batch_size=args.alpaca_batch_size, verbose=False)
            method.restore(model, snap)
            items = [{"idx": k, "instruction": ins, "completion": comp}
                     for k, (ins, comp) in enumerate(zip(instructions, comps))]
            (c.out_dir / "alpaca_completions" / f"row_{need}.json").write_text(
                json.dumps(items, ensure_ascii=False, indent=2))
            n_loop = sum(is_looping(it["completion"], args.loop_run) for it in items)
            c.new_cells[str(need)] = {
                "row": need, "hp_desc": _hp_desc(row),
                "hp": {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in HP_COLS if k in row.index},
                "edit_info": {k: int(v) for k, v in info.items() if k.startswith("k_features_mlp")},
                **{k: float(row[k]) for k in ("mmlu_acc", "qa_acc", "simdom_acc", "efficacy", "specificity", "harmonic")},
                "n_loop_answers": n_loop, "looping": n_loop > args.max_loop_answers,
            }
            c.save_new_cells()
            log.info("row %d (%s) harmonic=%.4f loops=%d/%d%s", need, _hp_desc(row),
                     float(row["harmonic"]), n_loop, len(items),
                     " LOOPING -> rejected, next cell moves up" if n_loop > args.max_loop_answers else " ok")
        if started:
            method.on_concept_end(model, concept, cfg)
        top, _ = c.walk()
        log.info("%s: final top %d rows %s (new: %s)", concept, args.top_k, top,
                 [i for i in top if _key(c.ranked.loc[i]) not in c.old_cells])


# ---------------------------------------------------------------------------- #
# stage judge                                                                   #
# ---------------------------------------------------------------------------- #

def stage_judge(args: argparse.Namespace, cfg, concepts: List[str]) -> None:
    judge_id = _judge_id(args.judge_model)
    fluency_template = GeminiEvaluator.ALPACA_FLUENCY_PROMPT if args.fluency_prompt == "original" else None
    if fluency_template is None:
        from judge_rerun.qwen_judge import ALPACA_FLUENCY_PROMPT_QWEN as fluency_template
    rel = _rel(cfg)
    judge = None
    summaries = []
    for concept in concepts:
        log.set_context(method="snmf_ef" if cfg.ember_step.enabled else "snmf", concept=concept, stage="pass2_rescored:judge")
        c = Concept(args, cfg, concept)
        top, need = c.walk()
        if need is not None:
            raise RuntimeError(f"{concept}: row {need} has no Alpaca answers yet; run --stage generate first")
        bases = _load_baselines(EMBER_BASELINE_PARENT / rel / "_baseline", concept, judge_id, args.fluency_prompt)
        old_pass2 = Path(args.pass2_dir) / rel / _safe_concept(concept)
        old = {int(r["cell"]): r for _, r in pd.read_csv(old_pass2 / f"cell_metrics__{judge_id}.csv").iterrows()}
        ckpt = c.out_dir / f"_rows__{judge_id}.json"
        judged_path = c.out_dir / f"alpaca_judged__{judge_id}.json"
        new_alp = json.loads(ckpt.read_text()) if ckpt.exists() else {}
        per_item = json.loads(judged_path.read_text()) if judged_path.exists() else {}

        rows = []
        for rank_pos, i in enumerate(top):
            r = c.ranked.loc[i]
            k = _key(r)
            if k in c.old_cells:
                o = old[c.old_cells[k]]
                alp = {"alpaca_instr": float(o["alpaca_instr"]), "alpaca_flu": float(o["alpaca_flu"]),
                       "source": f"old_pass2 cell_{c.old_cells[k]}"}
            elif str(i) in new_alp:
                alp = new_alp[str(i)]
            else:
                items = json.loads((c.out_dir / "alpaca_completions" / f"row_{i}.json").read_text())[:args.alpaca_n]
                if judge is None:
                    from judge_rerun.qwen_judge import QwenJudge
                    judge = QwenJudge(model_id=args.judge_model)
                    log.info("judge loaded: %s", args.judge_model)
                res = _judge_one_delta(judge, items, fluency_template, args.judge_batch_size,
                                       c.out_dir / f"_ckpt__{judge_id}__row_{i}_instruct.json",
                                       c.out_dir / f"_ckpt__{judge_id}__row_{i}_fluency.json")
                per_item[f"row_{i}"] = res.pop("_per_item")
                alp = {**res, "source": "pass2_rescored"}
                new_alp[str(i)] = alp
                ckpt.write_text(json.dumps(new_alp, indent=2))
                judged_path.write_text(json.dumps(per_item, ensure_ascii=False, indent=2))
                log.info("row %d judged: instr=%.3f flu=%.3f (%.0fs)", i, alp["alpaca_instr"],
                         alp["alpaca_flu"], alp["judge_wall_time_s"])
            row = {"cell": rank_pos, "row": int(i), **{h: r[h] for h in HP_COLS if h in r.index},
                   **{f: float(r[f]) for f in ("mmlu_acc", "qa_acc", "simdom_acc", "efficacy", "specificity", "harmonic")}}
            row.update(_combine_alpaca_metrics(row["efficacy"], row["specificity"], alp["alpaca_instr"],
                                               alp["alpaca_flu"], bases, float("-inf")))
            row["alpaca_source"] = alp["source"]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(c.out_dir / f"cell_metrics__{judge_id}.csv", index=False)
        best = io.topk_per_concept(df, k=1, primary="harmonic_alpaca").iloc[0]
        old_best = json.loads((old_pass2 / f"best_hp__{judge_id}.json").read_text())
        best_hp = {h: (best[h].item() if hasattr(best[h], "item") else best[h]) for h in HP_COLS if h in best.index}
        same = all(abs(float(best_hp[k]) - float(old_best["best_hp"][k])) < 1e-9 for k in KEY if k in old_best["best_hp"])
        summary = {
            "concept": concept, "judge_id": judge_id, "judge_model": args.judge_model,
            "scoring": "mc_fixed (judge_rerun/mc_fixed.py), grid judge_rerun/SNMF/rescore_mc",
            "top_rows": top, "new_rows": [i for i in top if _key(c.ranked.loc[i]) not in c.old_cells],
            "best_row": int(best["row"]), "best_hp": best_hp,
            "best_harmonic_alpaca": float(best["harmonic_alpaca"]),
            "best_metrics": {m: float(best[m]) for m in ("efficacy", "specificity", "harmonic", "alp_instr_frac",
                                                         "alp_flu_frac", "coherence", "harmonic_alpaca")},
            "old_best_harmonic_alpaca": float(old_best["best_harmonic_alpaca"]),
            "pick_changed": not same,
        }
        (c.out_dir / f"best_hp__{judge_id}.json").write_text(json.dumps(summary, indent=2, default=float))
        log.info("BEST row %d (%s) harmonic_alpaca=%.4f (old %.4f)%s", summary["best_row"], _hp_desc(best),
                 summary["best_harmonic_alpaca"], summary["old_best_harmonic_alpaca"],
                 " PICK CHANGED" if not same else "")
        summaries.append({"concept": concept, **{h: best_hp.get(h) for h in HP_COLS},
                          "best_harmonic_alpaca": summary["best_harmonic_alpaca"],
                          "old_best_harmonic_alpaca": summary["old_best_harmonic_alpaca"],
                          "pick_changed": not same, "new_rows": summary["new_rows"]})

    path = Path(args.out_dir) / rel / "best_hps_summary.csv"
    new = pd.DataFrame(summaries)
    if path.exists():
        prev = pd.read_csv(path)
        new = pd.concat([prev[~prev["concept"].isin(new["concept"])], new])
    new.sort_values("concept").to_csv(path, index=False)
    log.info("wrote %s", path)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = parse_args()
    log.configure()
    cfg = load_cfg(args.with_ember)
    (stage_generate if args.stage == "generate" else stage_judge)(args, cfg, args.concept)


if __name__ == "__main__":
    main()
