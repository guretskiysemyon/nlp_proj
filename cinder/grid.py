#!/usr/bin/env python
"""CINDER delta grid, Part 1's protocol (GPU: Gemma only, no judge).

For each concept and each delta of the grid (judge_rerun/Cinder/delta_grid.json, from cinder.edit_size):
apply the CINDER edit (EMBER's edit code on mf_outputs/cinder/), score MMLU (50, train), QA_train and
SimdomQA_train with the fixed MC scoring (judge_rerun/mc_fixed.py, as Part 1's repaired EMBER grid), and save
every per-question record. Then Pass 2's fast screen (min_mmlu gate, top 3 by harmonic of efficacy and
specificity; judge_rerun/pass2_delta_search.py) picks the 3 deltas to judge, and their Alpaca completions are
generated (the first 56 train instructions, greedy, 200 new tokens, batch 8: the first 50 are batched exactly as
in Part 1, and cinder.pick judges those 50).

Edge rule: if the fast screen's best delta is the largest (smallest) grid value, one more delta is scored,
2x the largest (half the smallest), and the top 3 is recomputed. At most one step per side.

Output, per concept: judge_rerun/Cinder/grid/<Concept>/
    records/delta_<d>.json, grid_rescored.csv (Part 1's column names), top3.json,
    alpaca_completions/delta_<d>.json (top 3 only)

    python -m cinder.grid --concepts Cannabis Gun ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # before torch (gotcha #14)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from cinder import common as C

SETS = ("mmlu", "qa", "simdom")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def fast_top(grid: pd.DataFrame, bases: dict, k: int) -> tuple[list[float], dict]:
    """Pass 2's fast screen (same ordering as pass2_ember_rescored.py: stable sort over grid order)."""
    from judge_rerun.pass2_delta_search import _metrics_for_delta
    grid = grid.sort_values("delta_embed").reset_index(drop=True)
    fast = {float(r["delta_embed"]): _metrics_for_delta(r, bases, None, C.MIN_MMLU, C.MIN_ALPACA)
            for _, r in grid.iterrows()}
    passing = [d for d, m in fast.items() if m.get("gate") != "min_mmlu"]
    passing.sort(key=lambda d: fast[d]["harmonic"], reverse=True)
    return passing[:k], fast


class Runner:
    def __init__(self):
        from ember.erasure import embed_edit
        from ember.erasure.model_loader import load_hf_model
        from ember.evals.alpaca import load_alpaca_eval_local
        from ember.evals.mmlu import load_mmlu_items
        from ember.evals.model_wrap import ensure_wrapped_model
        from ember.local_datasets import load_alpaca_indices, load_mmlu_indices
        C.use_cinder_features()
        self.embed_edit = embed_edit
        self.model, tok = load_hf_model(C.MODEL_NAME)
        self.tm = ensure_wrapped_model(self.model, tok)
        self.pristine = embed_edit.snapshot(self.model)
        self.mmlu_items = load_mmlu_items(load_mmlu_indices("train"))
        alpaca = [ex["instruction"] for ex in load_alpaca_eval_local() if "instruction" in ex]
        self.alpaca = [alpaca[i] for i in load_alpaca_indices("train")][:C.ALPACA_GEN_N]

    def edit(self, concept: str, delta: float) -> dict:
        self.embed_edit.restore(self.model, self.pristine)
        return self.embed_edit.apply_concept_embed_edit_factored(
            hf_model=self.model, model_name=C.MODEL_NAME, concept_name=concept, delta_embed=delta,
            rank=C.RANK, seed=C.SEED, ratio_thresh=2.0)

    def score(self, concept: str, delta: float, qa, sim) -> dict:
        from judge_rerun.mc_fixed import evaluate_mc_fixed, evaluate_mmlu_fixed
        info = self.edit(concept, delta)
        sets = {"mmlu": evaluate_mmlu_fixed(self.tm, self.mmlu_items), "qa": evaluate_mc_fixed(self.tm, qa),
                "simdom": evaluate_mc_fixed(self.tm, sim)}
        self.embed_edit.restore(self.model, self.pristine)
        return {"edit_info": {"k_features_embed": int(info["k_features_embed"]),
                              "n_tokens_edited": int(info["n_tokens_edited"])}, "sets": sets}

    def completions(self, concept: str, delta: float) -> list:
        self.edit(concept, delta)
        out = self.tm.generate_multiple(self.alpaca, max_new_tokens=200, do_sample=False, batch_size=8, verbose=False)
        self.embed_edit.restore(self.model, self.pristine)
        return [{"idx": i, "instruction": a, "completion": c} for i, (a, c) in enumerate(zip(self.alpaca, out))]


def run_concept(rn: Runner, concept: str, deltas: list[float], extend: bool, overwrite: bool) -> dict:
    from ember.evals.mc import prepare_mc_items
    from ember.local_datasets import load_mc_qa_items
    from judge_rerun.mc_fixed import MC_MAX_NEW_TOKENS, MMLU_MAX_NEW_TOKENS
    from judge_rerun.pass2_delta_search import _load_baselines

    out = C.GRID_OUT / C.safe(concept)
    (out / "records").mkdir(parents=True, exist_ok=True)
    pot = C.potential_csv(concept)
    if not pot.exists():
        raise FileNotFoundError(f"{pot} missing: run python -m cinder.build_features --concepts {concept!r}")
    if pd.read_csv(pot).empty:
        log(f"{concept}: the two-way judge accepted no feature, nothing to edit; skipping")
        C.write_json(out / "top3.json", {"concept": concept, "skipped": "no TWJ feature"})
        return {"concept": concept, "skipped": True}

    qa = prepare_mc_items(load_mc_qa_items("qa_train", concept=concept), seed=C.SEED)
    sim = prepare_mc_items(load_mc_qa_items("simdom_train", concept=concept), seed=C.SEED)
    bases = _load_baselines(C.BASELINE_DIR, concept, C.JUDGE_ID, "original")
    csv_path = out / "grid_rescored.csv"
    rows = {} if overwrite or not csv_path.exists() else \
        {float(r["delta_embed"]): r.to_dict() for _, r in pd.read_csv(csv_path).iterrows()}

    def do(delta: float):
        if delta in rows:
            return
        t0 = time.time()
        res = rn.score(concept, delta, qa, sim)
        row = {"concept": concept, "delta_embed": delta, **res["edit_info"]}
        for s in SETS:
            row[f"{s}_acc"] = res["sets"][s]["acc"]
            row[f"{s}_invalid_rate"] = res["sets"][s]["invalid_rate"]
        row.update(mc_max_new_tokens=MC_MAX_NEW_TOKENS, mmlu_max_new_tokens=MMLU_MAX_NEW_TOKENS,
                   wall_time_s=round(time.time() - t0, 1))
        C.write_json(out / "records" / f"delta_{delta}.json",
                     {"concept": concept, "delta_embed": delta, **res["edit_info"],
                      "mc_max_new_tokens": MC_MAX_NEW_TOKENS, "mmlu_max_new_tokens": MMLU_MAX_NEW_TOKENS,
                      **{s: res["sets"][s] for s in SETS}})
        rows[delta] = row
        pd.DataFrame(sorted(rows.values(), key=lambda r: r["delta_embed"])).to_csv(csv_path, index=False)
        log(f"{concept} delta={delta}: k={row['k_features_embed']} tokens={row['n_tokens_edited']} "
            f"mmlu {row['mmlu_acc']:.2f} qa {row['qa_acc']:.2f} (inv {row['qa_invalid_rate']:.2f}) "
            f"simdom {row['simdom_acc']:.2f} ({row['wall_time_s']:.0f}s)")

    for d in deltas:
        do(float(d))
    grid = pd.DataFrame(rows.values())
    top, fast = fast_top(grid, bases, C.TOP_K)
    extended = []
    if extend and top:
        lo, hi = min(deltas), max(deltas)
        for edge, new in ((hi, hi * 2), (lo, lo / 2)):
            if top[0] == edge:
                do(float(new))
                extended.append(float(new))
        if extended:
            grid = pd.DataFrame(rows.values())
            top, fast = fast_top(grid, bases, C.TOP_K)

    comp_dir = out / "alpaca_completions"
    comp_dir.mkdir(exist_ok=True)
    for d in top:
        p = comp_dir / f"delta_{d}.json"
        if p.exists() and not overwrite:
            continue
        t0 = time.time()
        p.write_text(json.dumps(rn.completions(concept, d), ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"{concept} delta={d}: {C.ALPACA_GEN_N} Alpaca completions ({time.time() - t0:.0f}s)")

    summary = {"concept": concept, "grid": [float(d) for d in deltas], "extended": extended, "top_k": top,
               "fast": {repr(d): {k: m.get(k) for k in ("qa_acc", "efficacy", "specificity", "harmonic", "gate")}
                        for d, m in sorted(fast.items())},
               "baselines": bases, "alpaca_generated": C.ALPACA_GEN_N, "alpaca_judged": C.ALPACA_N}
    C.write_json(out / "top3.json", summary)
    log(f"{concept}: top {C.TOP_K} = {top}" + (f" (grid extended by {extended})" if extended else ""))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concepts", nargs="+", default=C.CONCEPTS)
    ap.add_argument("--deltas", type=float, nargs="+", default=None, help="default: judge_rerun/Cinder/delta_grid.json")
    ap.add_argument("--no-extend", action="store_true", help="no extra delta when the best one is at a grid edge")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    from ember.erasure import log as elog
    elog.configure()
    deltas = a.deltas or C.load_grid()
    log(f"grid: {deltas}")
    rn = Runner()
    for c in a.concepts:
        run_concept(rn, c, deltas, not a.no_extend, a.overwrite)


if __name__ == "__main__":
    main()
