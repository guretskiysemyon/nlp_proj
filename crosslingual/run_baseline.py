#!/usr/bin/env python
"""Pristine-model baseline for the cross-lingual experiment.

    python -m crosslingual.run_baseline --concepts "Harry Potter" Golf --languages en de
    python -m crosslingual.run_baseline                 # all 18 concepts x 5 languages

Loads the unedited model once, then for each (concept, language) asks the 50
``QA_test`` and 50 ``SimdomQA_test`` MC questions with the native-language
scaffold, and writes one JSON per cell::

    <out-dir>/<model_safe>/baseline/<Concept_safe>/<lang>.json

holding accuracy, invalid rate, timing and every per-question record. A cell
whose file exists is skipped (so an interrupted job resumes), unless
``--overwrite``. ``summary.csv`` next to the cells is rebuilt from all cell
files at the end of every run.

``reliable`` is False when the pristine ``QA_test`` accuracy is below 0.40:
the model never knew the concept in that language, so there is nothing to
erase and the cell is left out of the leakage averages.
"""
from __future__ import annotations

import os

# Must be set before torch is imported (gotcha #14): transformers auto-compiles
# Gemma-2 generation on GPU, and the cluster nodes lack Python.h for inductor.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):  # allow `python crosslingual/run_baseline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosslingual.data import CONCEPTS, LANGUAGES, REPO_ROOT, SETS

MIN_RELIABLE_QA_ACC = 0.40
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "crosslingual"


def cell_path(out_dir: Path, model_name: str, concept: str, lang: str) -> Path:
    from ember.utils import _safe_concept, _safe_model_name
    return (out_dir / _safe_model_name(model_name) / "baseline"
            / _safe_concept(concept) / f"{lang}.json")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def write_summary(out_dir: Path, model_name: str) -> Path:
    """Rebuild ``summary.csv`` (one row per cell) from every cell JSON on disk."""
    import pandas as pd
    from ember.utils import _safe_model_name

    root = out_dir / _safe_model_name(model_name) / "baseline"
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.glob("*/*.json")):
        cell = json.loads(p.read_text(encoding="utf-8"))
        row = {"concept": cell["concept"], "language": cell["language"],
               "reliable": cell["reliable"], "limit": cell["limit"]}
        for s in SETS:
            r = cell["results"][s]
            row.update({f"{s}_acc": r["accuracy"], f"{s}_invalid": r["invalid_rate"],
                        f"{s}_n": r["n"], f"{s}_sec": r["seconds"]})
        rows.append(row)
    path = root / "summary.csv"
    if rows:
        pd.DataFrame(rows).sort_values(["concept", "language"]).to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--concepts", nargs="+", default=list(CONCEPTS),
                   help="Concept names, exactly as in the QA JSON (quote multi-word ones).")
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--seed", type=int, default=42, help="Option-shuffle seed.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N questions per set (smoke tests). Cells "
                        "written with a limit are marked and are not real baselines.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--overwrite", action="store_true", help="Recompute existing cells.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = [c for c in args.concepts if c not in CONCEPTS]
    if unknown:
        print(f"unknown concept(s) {unknown}; expected some of {list(CONCEPTS)}")
        return 2

    def done(concept: str, lang: str) -> bool:
        # A cell counts as done only if it was written with the same --limit,
        # so a smoke-test cell never stands in for a real baseline.
        path = cell_path(args.out_dir, args.model, concept, lang)
        if not path.exists():
            return False
        return json.loads(path.read_text(encoding="utf-8")).get("limit") == args.limit

    cells = [(c, l) for c in args.concepts for l in args.languages]
    todo = [(c, l) for c, l in cells if args.overwrite or not done(c, l)]
    print(f"[baseline] {len(cells)} cells requested, {len(todo)} to run "
          f"({len(cells) - len(todo)} already on disk)")
    if not todo:
        print(f"[baseline] summary -> {write_summary(args.out_dir, args.model)}")
        return 0

    import torch
    from ember.erasure import model_loader
    from ember.evals.model_wrap import ensure_wrapped_model
    from crosslingual.mc_eval import evaluate_cell

    torch.manual_seed(args.seed)
    t0 = time.time()
    hf_model, tokenizer = model_loader.load_hf_model(
        args.model, "", dtype=model_loader.pick_dtype("bf16"))
    hf_model.eval()
    tm = ensure_wrapped_model(hf_model, tokenizer)
    device = str(next(hf_model.parameters()).device)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[baseline] model loaded in {time.time() - t0:.1f}s on {device} ({gpu})")

    commit = _git_commit()
    for k, (concept, lang) in enumerate(todo, 1):
        t_cell = time.time()
        with torch.no_grad():
            results = evaluate_cell(tm, concept, lang, seed=args.seed, limit=args.limit)
        qa_acc = results["QA_test"]["accuracy"]
        cell = {
            "concept": concept, "language": lang, "model": args.model,
            "config": "pristine", "prompt_lang": lang, "seed": args.seed,
            "limit": args.limit, "dtype": "bf16", "gpu": gpu, "git_commit": commit,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reliable": qa_acc >= MIN_RELIABLE_QA_ACC,
            "min_reliable_qa_acc": MIN_RELIABLE_QA_ACC,
            "results": results,
        }
        path = cell_path(args.out_dir, args.model, concept, lang)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cell, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

        sd = results["SimdomQA_test"]
        flag = "" if cell["reliable"] else "  [UNRELIABLE: QA_test < 0.40]"
        print(f"[baseline] {k}/{len(todo)} {concept!r} {lang}: "
              f"QA_test={qa_acc:.2f} (invalid {results['QA_test']['invalid_rate']:.2f}) "
              f"Simdom={sd['accuracy']:.2f} (invalid {sd['invalid_rate']:.2f}) "
              f"{time.time() - t_cell:.1f}s{flag}", flush=True)

    print(f"[baseline] done in {(time.time() - t0) / 60:.1f} min; "
          f"summary -> {write_summary(args.out_dir, args.model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
