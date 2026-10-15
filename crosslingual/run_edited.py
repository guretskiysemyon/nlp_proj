#!/usr/bin/env python
"""Edited-model runs for the cross-lingual experiment.

    python -m crosslingual.run_edited --methods ember snmf --concepts "Harry Potter"
    python -m crosslingual.run_edited --methods snmf_ember
    python -m crosslingual.run_edited                    # all methods, all 18 concepts

For each (method, concept): apply the edit picked by the English HP search
(EMBER: best δ, SNMF: best HP cell, SNMF+EMBER: best SNMF cell searched on top
of the concept's EMBER δ; all Qwen-judged), then ask the same
``QA_test`` / ``SimdomQA_test`` MC questions in every language with exactly
the baseline's items, shuffle, prompts and decoding (``mc_eval.evaluate_cell``).
Writes::

    <out-dir>/<model_safe>/<method>/<Concept_safe>/<lang>.json   # same format as baseline/
    <out-dir>/<model_safe>/<method>/<Concept_safe>/edit.json     # HP, edit check, MMLU
    <out-dir>/<model_safe>/<method>/summary.csv                  # + pristine, retention, leak

``edit.json`` holds the **edit check**: the edited model is re-scored on the
English sets the HP search used (``QA_train``, ``SimdomQA_train``, the 50
MMLU-train questions, with the search's own eval code), next to the search's
numbers for the same HP. Matching numbers (and feature counts) mean this is
the edit that was tuned. It also holds the English MMLU on the 1000-question
test split, computed per (concept, method) -- the Gemini-era runner reused the
first concept's MMLU for every concept. The pristine MMLU (same code) is
computed once into ``baseline/_mmlu.json``.

A (method, concept) whose ``edit.json`` and language files all exist is
skipped, so a preempted job resumes. No judge is used anywhere.
"""
from __future__ import annotations

import os

# Must be set before torch is imported (gotcha #14).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # allow `python crosslingual/run_edited.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosslingual.data import CONCEPTS, LANGUAGES, REPO_ROOT, SETS
from crosslingual.run_baseline import (DEFAULT_OUT_DIR, MIN_RELIABLE_QA_ACC,
                                       _git_commit, cell_path)

METHODS: tuple[str, ...] = ("ember", "snmf", "snmf_ember")
RUN_REL = Path("google_gemma-2-2b-it") / "rank100" / "seed42"
EMBER_SUMMARY = (REPO_ROOT / "judge_rerun" / "Ember" / "pass2_outputs" / RUN_REL
                 / "best_deltas_summary.csv")
EMBER_PASS1 = REPO_ROOT / "judge_rerun" / "Ember" / "pass1_outputs" / RUN_REL
SNMF_PASS1 = REPO_ROOT / "judge_rerun" / "SNMF" / "pass1_outputs" / RUN_REL
SNMF_PASS2 = REPO_ROOT / "judge_rerun" / "SNMF" / "pass2_outputs" / RUN_REL
SNMF_EMBER_PASS1 = REPO_ROOT / "judge_rerun" / "SNMF_EMBER" / "pass1_outputs" / RUN_REL
SNMF_EMBER_PASS2 = REPO_ROOT / "judge_rerun" / "SNMF_EMBER" / "pass2_outputs" / RUN_REL
JUDGE_ID = "qwen2.5-32b-instruct"
CONFIGS = {"ember": REPO_ROOT / "configs" / "ember_gemma.yaml",
           "snmf": REPO_ROOT / "configs" / "snmf_gemma.yaml",
           "snmf_ember": REPO_ROOT / "configs" / "snmf_ember_gemma.yaml"}
# snmf_ember is the snmf method with the EMBER step: SNMFMethod.apply edits the
# embeddings when the HP has delta_embed != 0, and snapshot/restore covers both.
REGISTRY_NAME = {"ember": "ember", "snmf": "snmf", "snmf_ember": "snmf"}
MMLU_BATCH_SIZE = 8  # as in the HP search


def _safe(concept: str) -> str:
    from ember.utils import _safe_concept
    return _safe_concept(concept)


def method_dir(out_dir: Path, model_name: str, method: str, concept: str) -> Path:
    from ember.utils import _safe_model_name
    return out_dir / _safe_model_name(model_name) / method / _safe(concept)


# --------------------------------------------------------------------- #
# The picks, and the HP search's own numbers for them                   #
# --------------------------------------------------------------------- #

def resolve_pick(method: str, concept: str) -> Dict[str, Any]:
    """``{"hp": <dict for method.apply>, "search": <the search's row for it>}``."""
    if method == "ember":
        with EMBER_SUMMARY.open(encoding="utf-8") as f:
            row = next((r for r in csv.DictReader(f) if r["concept"] == concept), None)
        if row is None:
            raise KeyError(f"{concept!r} not in {EMBER_SUMMARY}")
        delta = float(row["best_delta_embed"])
        with (EMBER_PASS1 / _safe(concept) / "grid_raw.csv").open(encoding="utf-8") as f:
            grid = next(r for r in csv.DictReader(f) if float(r["delta_embed"]) == delta)
        search = {k: float(grid[k]) for k in ("qa_acc", "simdom_acc", "mmlu_acc",
                                              "k_features_embed", "n_tokens_edited")}
        search["harmonic_alpaca"] = float(row["best_harmonic_alpaca"])
        return {"hp": {"delta_embed": delta}, "search": search}

    if method == "snmf":
        best = json.loads((SNMF_PASS2 / _safe(concept) / f"best_hp__{JUDGE_ID}.json")
                          .read_text(encoding="utf-8"))
        top3 = json.loads((SNMF_PASS1 / _safe(concept) / "top3_hps.json")
                          .read_text(encoding="utf-8"))
        cell = top3["cells"][best["best_cell"]]
        search = {k: float(cell[k]) for k in ("qa_acc", "simdom_acc", "mmlu_acc",
                                              "k_features_mlp_in", "k_features_mlp_out")}
        search["harmonic_alpaca"] = float(best["best_harmonic_alpaca"])
        # delta_embed = 0: plain SNMF, no EMBER step (as pass1_snmf_top3._build_hp).
        return {"hp": {**best["best_hp"], "delta_embed": 0.0}, "search": search}

    if method == "snmf_ember":
        best = json.loads((SNMF_EMBER_PASS2 / _safe(concept) / f"best_hp__{JUDGE_ID}.json")
                          .read_text(encoding="utf-8"))
        top3 = json.loads((SNMF_EMBER_PASS1 / _safe(concept) / "top3_hps.json")
                          .read_text(encoding="utf-8"))
        cell = top3["cells"][best["best_cell"]]
        hp = dict(best["best_hp"])
        delta = float(hp.get("delta_embed", 0.0))
        if delta == 0.0 or delta != float(top3["ember_delta_embed"]):
            raise ValueError(f"{concept!r}: snmf_ember pick has delta_embed {delta}, "
                             f"Pass 1 ran with {top3['ember_delta_embed']}")
        search = {k: float(cell[k]) for k in ("qa_acc", "simdom_acc", "mmlu_acc",
                                              "k_features_embed", "n_tokens_edited",
                                              "k_features_mlp_in", "k_features_mlp_out")}
        search["harmonic_alpaca"] = float(best["best_harmonic_alpaca"])
        return {"hp": hp, "search": search}

    raise ValueError(f"unknown method {method!r}")


def load_config(method: str, model_name: str, hp: Dict[str, Any]):
    from ember.erasure.config import load_yaml
    cfg = load_yaml(CONFIGS[method])
    cfg.train_eval = "mc"
    cfg.run_tests_after_train = False
    if cfg.model_name != model_name:
        raise ValueError(f"{CONFIGS[method]} is for {cfg.model_name}, not {model_name}")
    if bool(cfg.ember_step.enabled) != (method == "snmf_ember"):
        raise ValueError(f"{CONFIGS[method]} has ember_step.enabled={cfg.ember_step.enabled}")
    if method in ("snmf", "snmf_ember"):
        # SNMFMethod.apply selects features from the CONFIG, not from the HP dict,
        # so the config must match what the grid ran with.
        for key, have in (("feature_source", cfg.snmf.feature_source),
                          ("ratio_thresh", cfg.selection.ratio_thresh),
                          ("coverage_thresh", cfg.selection.coverage_thresh),
                          ("w_mode", cfg.snmf.w_mode)):
            if key in hp and str(hp[key]) != str(have):
                raise ValueError(f"snmf pick has {key}={hp[key]!r} but config has {have!r}")
    return cfg


# --------------------------------------------------------------------- #
# English checks with the HP search's own eval code                     #
# --------------------------------------------------------------------- #

class EnglishEval:
    """The search's English sets (train split) plus MMLU test, loaded once."""

    def __init__(self, seed: int) -> None:
        from ember.evals.mmlu import load_mmlu_items
        from ember.local_datasets import load_mmlu_indices
        self.seed = seed
        self.mmlu_train = load_mmlu_items(load_mmlu_indices("train"))
        self.mmlu_test = load_mmlu_items(load_mmlu_indices("test"))

    @staticmethod
    def _mmlu(tm: Any, items: List[Any]) -> Dict[str, Any]:
        from ember.evals.mmlu import evaluate_mmlu_generation
        correct, invalid, _ = evaluate_mmlu_generation(tm, items, batch_size=MMLU_BATCH_SIZE)
        return {"acc": correct / len(items), "invalid": invalid / len(items), "n": len(items)}

    def mmlu_test_acc(self, tm: Any) -> Dict[str, Any]:
        return self._mmlu(tm, self.mmlu_test)

    def search_sets(self, tm: Any, concept: str) -> Dict[str, Any]:
        from ember.evals.mc import evaluate_mc_generation, prepare_mc_items
        from ember.local_datasets import load_mc_qa_items
        out: Dict[str, Any] = {"mmlu_acc": self._mmlu(tm, self.mmlu_train)["acc"]}
        for key, set_name in (("qa_acc", "qa_train"), ("simdom_acc", "simdom_train")):
            items = prepare_mc_items(load_mc_qa_items(set_name, concept=concept), seed=self.seed)
            correct, _, _ = evaluate_mc_generation(tm, items)
            out[key] = correct / len(items)
        return out


# --------------------------------------------------------------------- #
# Output                                                                #
# --------------------------------------------------------------------- #

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def is_done(out_dir: Path, model_name: str, method: str, concept: str,
            languages: List[str]) -> bool:
    d = method_dir(out_dir, model_name, method, concept)
    return (d / "edit.json").exists() and all((d / f"{l}.json").exists() for l in languages)


def _retention(acc: float, pristine: float) -> Optional[float]:
    if pristine <= 0.25:
        return None
    return min(1.0, max(0.0, (acc - 0.25) / (pristine - 0.25)))


def write_summary(out_dir: Path, model_name: str, method: str) -> Path:
    """One row per (concept, language): edited and pristine accuracy, retention, leak.

    ``retention = clip((acc − 0.25) / (pristine − 0.25), 0, 1)`` on ``QA_test``;
    ``leak = retention_L − retention_en`` (> 0: the concept survives better in L).
    ``reliable`` is the pristine cell's flag (QA_test ≥ 0.40).
    """
    import pandas as pd
    from ember.utils import _safe_model_name

    root = out_dir / _safe_model_name(model_name) / method
    rows: List[Dict[str, Any]] = []
    for edit_path in sorted(root.glob("*/edit.json")):
        edit = json.loads(edit_path.read_text(encoding="utf-8"))
        concept = edit["concept"]
        per_lang: Dict[str, Dict[str, Any]] = {}
        for lang_path in sorted(edit_path.parent.glob("*.json")):
            if lang_path.name == "edit.json":
                continue
            cell = json.loads(lang_path.read_text(encoding="utf-8"))
            base = json.loads(cell_path(out_dir, model_name, concept, cell["language"])
                              .read_text(encoding="utf-8"))
            row = {"concept": concept, "language": cell["language"],
                   "reliable": base["reliable"]}
            for s in SETS:
                row[f"{s}_acc"] = cell["results"][s]["accuracy"]
                row[f"{s}_invalid"] = cell["results"][s]["invalid_rate"]
                row[f"{s}_pristine"] = base["results"][s]["accuracy"]
            row["retention"] = _retention(row["QA_test_acc"], row["QA_test_pristine"])
            row["mmlu_test_acc_en"] = edit["mmlu_test"]["acc"]
            row["hp"] = json.dumps(edit["hp"], sort_keys=True)
            per_lang[cell["language"]] = row
        r_en = per_lang.get("en", {}).get("retention")
        for row in per_lang.values():
            row["leak"] = (None if r_en is None or row["retention"] is None
                           else row["retention"] - r_en)
            rows.append(row)
    path = root / "summary.csv"
    if rows:
        cols = ["concept", "language", "reliable", "QA_test_acc", "QA_test_pristine",
                "retention", "leak", "QA_test_invalid", "SimdomQA_test_acc",
                "SimdomQA_test_pristine", "SimdomQA_test_invalid", "mmlu_test_acc_en", "hp"]
        pd.DataFrame(rows)[cols].sort_values(["concept", "language"]).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--concepts", nargs="+", default=list(CONCEPTS),
                   help="Concept names, exactly as in the QA JSON (quote multi-word ones).")
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--seed", type=int, default=42,
                   help="Option-shuffle seed; must match the baseline's.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--overwrite", action="store_true", help="Recompute existing cells.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve and print every pick, load nothing, run nothing.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = [c for c in args.concepts if c not in CONCEPTS]
    if unknown:
        print(f"unknown concept(s) {unknown}; expected some of {list(CONCEPTS)}")
        return 2

    # Every edited cell is compared against the pristine cell; require them up front.
    missing = [(c, l) for c in args.concepts for l in args.languages
               if not cell_path(args.out_dir, args.model, c, l).exists()]
    if missing:
        print(f"[edited] missing pristine baseline cells (run run_baseline first): {missing}")
        return 2

    jobs = [(m, c) for m in args.methods for c in args.concepts]
    picks = {(m, c): resolve_pick(m, c) for m, c in jobs}
    for (m, c), pick in picks.items():
        load_config(m, args.model, pick["hp"])  # fail early on a config mismatch
        print(f"[edited] {m:10s} {c!r}: {pick['hp']}")
    if args.dry_run:
        return 0

    todo = [(m, c) for m, c in jobs
            if args.overwrite or not is_done(args.out_dir, args.model, m, c, args.languages)]
    print(f"[edited] {len(jobs)} (method, concept) requested, {len(todo)} to run")

    if todo:
        import torch
        from ember.erasure import model_loader
        from ember.erasure.methods import base as methods_base
        from ember.evals.model_wrap import ensure_wrapped_model
        from crosslingual.mc_eval import evaluate_cell

        torch.manual_seed(args.seed)
        t0 = time.time()
        hf_model, tokenizer = model_loader.load_hf_model(
            args.model, "", dtype=model_loader.pick_dtype("bf16"))
        hf_model.eval()
        tm = ensure_wrapped_model(hf_model, tokenizer)
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print(f"[edited] model loaded in {time.time() - t0:.1f}s ({gpu})", flush=True)
        english = EnglishEval(args.seed)
        commit = _git_commit()

        # Pristine MMLU, once, with the same code as the edited models.
        from ember.utils import _safe_model_name
        mmlu_base_path = args.out_dir / _safe_model_name(args.model) / "baseline" / "_mmlu.json"
        if not mmlu_base_path.exists():
            with torch.no_grad():
                t = time.time()
                base = {"mmlu_test": english.mmlu_test_acc(tm),
                        "mmlu_train_acc": english._mmlu(tm, english.mmlu_train)["acc"],
                        "model": args.model, "git_commit": commit}
            _write_json(mmlu_base_path, base)
            print(f"[edited] pristine MMLU test={base['mmlu_test']['acc']:.3f} "
                  f"train={base['mmlu_train_acc']:.2f} ({time.time() - t:.0f}s)", flush=True)

        for k, (method_name, concept) in enumerate(todo, 1):
            t_job = time.time()
            pick = picks[(method_name, concept)]
            cfg = load_config(method_name, args.model, pick["hp"])
            method = methods_base.get(REGISTRY_NAME[method_name])
            snap = method.snapshot(hf_model)
            try:
                method.on_concept_start(hf_model, concept, cfg)
                info = method.apply(hf_model, tokenizer, pick["hp"], concept, cfg)
                with torch.no_grad():
                    check = english.search_sets(tm, concept)
                    mmlu_test = english.mmlu_test_acc(tm)
                    out_dir = method_dir(args.out_dir, args.model, method_name, concept)
                    for lang in args.languages:
                        results = evaluate_cell(tm, concept, lang, seed=args.seed)
                        _write_json(out_dir / f"{lang}.json", {
                            "concept": concept, "language": lang, "model": args.model,
                            "config": method_name, "prompt_lang": lang, "seed": args.seed,
                            "hp": pick["hp"], "dtype": "bf16", "gpu": gpu,
                            "git_commit": commit,
                            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "results": results,
                        })
                        print(f"[edited]   {method_name} {concept!r} {lang}: "
                              f"QA_test={results['QA_test']['accuracy']:.2f} "
                              f"(invalid {results['QA_test']['invalid_rate']:.2f}) "
                              f"Simdom={results['SimdomQA_test']['accuracy']:.2f}", flush=True)
            finally:
                method.restore(hf_model, snap)
                method.on_concept_end(hf_model, concept, cfg)

            diffs = {key: round(check[key] - pick["search"][key], 4)
                     for key in ("qa_acc", "simdom_acc", "mmlu_acc")}
            k_keys = [key for key in pick["search"] if key.startswith(("k_features", "n_tokens"))]
            k_match = all(int(info.get(key, -1)) == int(pick["search"][key]) for key in k_keys)
            _write_json(out_dir / "edit.json", {
                "concept": concept, "method": method_name, "hp": pick["hp"],
                "edit_info": info, "search": pick["search"],
                "check": {"english_train": check, "diff_vs_search": diffs,
                          "features_match": k_match,
                          "reproduces_search": k_match and all(v == 0 for v in diffs.values())},
                "mmlu_test": mmlu_test, "seconds": round(time.time() - t_job, 1),
                "git_commit": commit,
            })
            flag = "OK" if k_match and all(v == 0 for v in diffs.values()) else "DIFFERS"
            print(f"[edited] {k}/{len(todo)} {method_name} {concept!r}: edit check {flag} "
                  f"(features {'match' if k_match else 'DIFFER'}; Δ vs search {diffs}) "
                  f"MMLU test={mmlu_test['acc']:.3f} {time.time() - t_job:.0f}s", flush=True)

    for m in args.methods:
        print(f"[edited] summary -> {write_summary(args.out_dir, args.model, m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
