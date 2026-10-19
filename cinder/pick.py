#!/usr/bin/env python
"""Best CINDER delta per concept: Part 1's Pass 2 on the CINDER grid (GPU: Qwen2.5-32B judge).

Same selection as judge_rerun/pass2_ember_rescored.py (the EMBER numbers we compare with): min_mmlu 0.90 gate,
top 3 by the fast harmonic (cinder.grid's top3.json), Qwen2.5-32B-Instruct on the first 50 Alpaca completions
with the original fluency prompt, harmonic_alpaca with the same pristine baselines
(judge_rerun/Ember/pass1_outputs/.../_baseline: the unedited model is the same for both methods),
min_alpaca 0.9, io.pick_best_embed_delta.

Output, per concept: judge_rerun/Cinder/pick/<Concept>/delta_metrics__<judge>.csv, best_delta__<judge>.json,
alpaca_judged__<judge>.json; and judge_rerun/Cinder/pick/best_deltas_summary.csv.

    python -m cinder.pick --concepts Cannabis Gun ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from cinder import common as C
from cinder.grid import fast_top


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def run_concept(concept: str, judge, batch_size: int):
    from ember.erasure import io
    from ember.evals.gemini import GeminiEvaluator
    from judge_rerun.pass2_delta_search import _judge_one_delta, _load_baselines, _metrics_for_delta

    gdir = C.GRID_OUT / C.safe(concept)
    top3 = json.loads((gdir / "top3.json").read_text())
    if top3.get("skipped"):
        log(f"{concept}: {top3['skipped']}; no pick")
        return judge, None
    out = C.PICK_OUT / C.safe(concept)
    out.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(gdir / "grid_rescored.csv").sort_values("delta_embed").reset_index(drop=True)
    bases = _load_baselines(C.BASELINE_DIR, concept, C.JUDGE_ID, "original")
    top, fast = fast_top(grid, bases, C.TOP_K)
    if [float(d) for d in top] != [float(d) for d in top3["top_k"]]:
        raise ValueError(f"{concept}: top 3 {top} differs from grid's {top3['top_k']}")

    ckpt_path = out / f"_rows__{C.JUDGE_ID}.json"
    judged_path = out / f"alpaca_judged__{C.JUDGE_ID}.json"
    alp_by = {float(k): v for k, v in json.loads(ckpt_path.read_text()).items()} if ckpt_path.exists() else {}
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
        if d not in alp_by:
            items = json.loads((gdir / "alpaca_completions" / f"delta_{d}.json").read_text())[:C.ALPACA_N]
            if judge is None:
                from judge_rerun.qwen_judge import QwenJudge
                judge = QwenJudge(model_id=C.JUDGE_MODEL)
                log(f"judge loaded: {C.JUDGE_MODEL}")
            res = _judge_one_delta(judge, items, GeminiEvaluator.ALPACA_FLUENCY_PROMPT, batch_size,
                                   out / f"_ckpt__{C.JUDGE_ID}__delta_{d}_instruct.json",
                                   out / f"_ckpt__{C.JUDGE_ID}__delta_{d}_fluency.json")
            per_item[repr(d)] = res.pop("_per_item")
            alp_by[d] = res
            ckpt_path.write_text(json.dumps({repr(k): v for k, v in alp_by.items()}, indent=2))
            judged_path.write_text(json.dumps(per_item, ensure_ascii=False, indent=2))
            for axis in ("instruct", "fluency"):
                (out / f"_ckpt__{C.JUDGE_ID}__delta_{d}_{axis}.json").unlink(missing_ok=True)
            log(f"{concept} delta={d}: instr={res['alpaca_instr']:.3f} flu={res['alpaca_flu']:.3f} "
                f"({res['judge_wall_time_s']:.0f}s)")
        row = _metrics_for_delta(raw, bases, alp_by[d], C.MIN_MMLU, C.MIN_ALPACA)
        row["_alpaca_n"] = C.ALPACA_N
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("delta_embed").reset_index(drop=True)
    df.to_csv(out / f"delta_metrics__{C.JUDGE_ID}.csv", index=False)
    best = io.pick_best_embed_delta(df)
    summary = {"concept": concept, "method": "cinder", "judge_model": C.JUDGE_MODEL, "fluency_prompt": "original",
               "min_mmlu": C.MIN_MMLU, "min_alpaca": C.MIN_ALPACA, "top_k": C.TOP_K, "alpaca_n": C.ALPACA_N,
               "grid": top3["grid"], "extended": top3.get("extended", []), "top_k_deltas": top,
               "baselines": bases, "best_delta_embed": float(best["delta_embed"]),
               "best_harmonic_alpaca": float(best.get("harmonic_alpaca") or 0.0),
               "at_grid_edge": float(best["delta_embed"]) in (min(grid["delta_embed"]), max(grid["delta_embed"]))}
    C.write_json(out / f"best_delta__{C.JUDGE_ID}.json", summary, indent=2)
    log(f"{concept}: BEST delta={summary['best_delta_embed']} harmonic_alpaca={summary['best_harmonic_alpaca']:.4f}")
    return judge, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concepts", nargs="+", default=C.CONCEPTS)
    ap.add_argument("--judge-batch-size", type=int, default=4)   # pass2_ember_rescored.py default
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    from ember.erasure import log as elog
    elog.configure()
    judge, summaries = None, []
    for c in a.concepts:
        judge, s = run_concept(c, judge, a.judge_batch_size)
        if s:
            summaries.append(s)
    write_pick_summary()


def write_pick_summary() -> None:
    """best_deltas_summary.csv rebuilt from every concept's best_delta file (safe with parallel jobs)."""
    keep = ["concept", "best_delta_embed", "best_harmonic_alpaca", "top_k_deltas", "extended", "at_grid_edge"]
    rows = [{k: json.loads(p.read_text()).get(k) for k in keep}
            for p in sorted(C.PICK_OUT.glob(f"*/best_delta__{C.JUDGE_ID}.json"))]
    if rows:
        path = C.PICK_OUT / "best_deltas_summary.csv"
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        pd.DataFrame(rows).sort_values("concept").to_csv(tmp, index=False)
        tmp.replace(path)
        log(f"wrote {path}")


if __name__ == "__main__":
    main()
