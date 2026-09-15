#!/usr/bin/env python
"""Cross-lingual erasure evaluation -- CLI entrypoint.

    python crosslingual/run.py --concept "Harry Potter" [options]

One concept per invocation (deliberately): concepts are parallelized as
separate jobs -- e.g. a SLURM array, the pattern ``ember/evals/gemini.py``
already assumes via ``SLURM_ARRAY_TASK_ID`` -- rather than looped inside one
long-running process. ``scripts/run_crosslingual_all.sh`` drives the agreed
concept set.

Execution flow (see §9 of the spec)::

    for config in configs:                  # one model load per config
        load hf_model, tokenizer fresh
        if config != "pristine": apply the config's best HP exactly once
        english MMLU once for this config (never swept across languages)
        for language in languages:
            ensure this language's pristine baseline exists
            for wrapper in (["native"] if language == "en" else wrappers):
                for qa_kind in ("mc", "open"):
                    evaluate QA_test + SimdomQA_test, write the cell, append summary
        free hf_model

The edit is language-independent, so it is applied once per config and every
language/wrapper/qa_kind cell is evaluated against the same resident weights.
``min_mmlu`` / ``max_qa_acc`` early-stopping is deliberately *not* applied
anywhere here, matching what the pipeline does for its own final-test stage --
a cell that early-stopped would produce an ``r_en`` that is not comparable to
``r_L``.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # allow `python crosslingual/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosslingual import baseline as cl_baseline
from crosslingual import best_hp as cl_hp
from crosslingual import data as cl_data
from crosslingual import prompts as cl_prompts
from crosslingual import results_io as cl_io

QA_KINDS: tuple[str, ...] = ("mc", "open")
WRAPPERS: tuple[str, ...] = ("english", "native")


# ========================================================================== #
# CLI                                                                         #
# ========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crosslingual",
        description="Evaluate whether a concept erasure survives translation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--concept", required=True,
                   help="Exactly one concept name (quote it: several contain "
                        "spaces or an apostrophe).")
    p.add_argument("--configs", nargs="+", default=list(cl_hp.CONFIGS),
                   choices=list(cl_hp.CONFIGS),
                   help="Which edits to evaluate.")
    p.add_argument("--languages", nargs="+", default=list(cl_data.LANGUAGES),
                   choices=list(cl_data.LANGUAGES),
                   help="Which languages the QA content is drawn from.")
    p.add_argument("--wrappers", nargs="+", default=list(WRAPPERS),
                   choices=list(WRAPPERS),
                   help="Instruction-scaffold language. Ignored for --languages en, "
                        "where the English wrapper IS the native wrapper.")
    p.add_argument("--qa-kinds", nargs="+", default=list(QA_KINDS),
                   choices=list(QA_KINDS),
                   help="Question formats to evaluate.")
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-eval", choices=("mc", "open"), default="mc",
                   help="Which grid search's HP to load. Must match how that "
                        "grid was actually run, per method.")
    p.add_argument("--results-root", type=Path, default=Path("results"),
                   help="Root of the tree holding the English grid outputs "
                        "(hps.csv / top_hps_valid.csv / best_embed.csv).")
    p.add_argument("--out-dir", type=Path, default=cl_io.DEFAULT_OUT_DIR)
    p.add_argument("--configs-dir", type=Path, default=None,
                   help="Directory holding the grid YAMLs (default: <repo>/configs).")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate data + HP-cache presence and exit. Loads "
                        "nothing onto the GPU.")
    p.add_argument("--force", action="store_true",
                   help="Recompute cells that already have output files. Does "
                        "not invalidate the per-language pristine baselines; "
                        "delete data/crosslingual_baselines/ for that.")
    return p


# ========================================================================== #
# Dry run                                                                     #
# ========================================================================== #

def dry_run(args: argparse.Namespace) -> int:
    """Validate data + HP caches without touching the GPU. Returns an exit code."""
    concept = args.concept
    failures: List[str] = []
    print("=" * 74)
    print(f"DRY RUN  concept={concept!r}  model={args.model}  "
          f"rank={args.rank}  seed={args.seed}  train_eval={args.train_eval}")
    print("=" * 74)

    print("\n--- QA data ---")
    for lang in args.languages:
        for qa_kind in args.qa_kinds:
            path = cl_data.qa_path(lang, qa_kind)
            try:
                keys = cl_data.concept_keys(lang, qa_kind)
            except FileNotFoundError as e:
                print(f"  FAIL {lang}/{qa_kind}: {e}")
                failures.append(f"data {lang}/{qa_kind}")
                continue
            if concept not in keys:
                near = [k for k in keys if k.lower().startswith(concept[:4].lower())]
                print(f"  FAIL {lang}/{qa_kind}: concept not a key in {path} "
                      f"({len(keys)} concepts; similar: {near or 'none'})")
                failures.append(f"data {lang}/{qa_kind}")
                continue
            counts = cl_data.describe_concept(concept, lang, qa_kind)
            bad = [k for k, v in counts.items() if v == 0]
            status = "FAIL" if bad else "PASS"
            if bad:
                failures.append(f"data {lang}/{qa_kind} empty {bad}")
            print(f"  {status} {lang}/{qa_kind}: QA_test={counts['qa_test']} "
                  f"SimdomQA_test={counts['simdom_test']}")

    print("\n--- prompt wrappers ---")
    needed = {"en"}
    for lang in args.languages:
        for wrapper in (["native"] if lang == "en" else args.wrappers):
            needed.add(cl_prompts.wrapper_lang_for(lang, wrapper))
    for wl in sorted(needed):
        try:
            cl_prompts.check_wrapper_lang(wl)
            print(f"  PASS wrapper_lang={wl}: "
                  f"{cl_prompts.WRAPPER_STRINGS[wl]['question_label']!r} / "
                  f"{cl_prompts.WRAPPER_STRINGS[wl]['answer_label']!r}")
        except ValueError as e:
            print(f"  FAIL wrapper_lang={wl}: {e}")
            failures.append(f"wrapper {wl}")

    print("\n--- best-HP caches (--results-root "
          f"{Path(args.results_root).resolve()}) ---")
    for config in cl_hp.order_configs(args.configs):
        if config == "pristine":
            print("  PASS pristine: no HP needed")
            continue
        loc = cl_hp.locate(config, concept, model_name=args.model, rank=args.rank,
                           seed=args.seed, train_eval=args.train_eval,
                           results_root=args.results_root,
                           configs_dir=args.configs_dir)
        if loc.ok:
            src = (loc.valid_csv if loc.valid_csv and loc.valid_csv.exists()
                   else loc.hp_csv)
            print(f"  PASS {config}: yaml={loc.yaml_path.name} hp={src}")
            if loc.best_embed_csv is not None:
                print(f"       delta_embed <- {loc.best_embed_csv}")
        else:
            print(f"  FAIL {config}:")
            for problem in loc.problems:
                print(f"       - {problem}")
            failures.append(f"hp {config}")

    print("\n--- pristine baselines ---")
    mmlu_path = (cl_baseline.ember_io.baseline_dir(args.model)
                 / "baseline_mmlu_test.json")
    if mmlu_path.exists():
        print(f"  PASS English MMLU baseline: {mmlu_path}")
    else:
        print(f"  FAIL English MMLU baseline missing: {mmlu_path} "
              f"(denominator of mmlu_frac)")
        failures.append("mmlu baseline")
    for lang in args.languages:
        entry = cl_baseline.read_baseline(args.model, concept, lang)
        path = cl_baseline.baseline_path(args.model, concept, lang)
        if cl_baseline.baseline_is_complete(entry, args.qa_kinds):
            # Show the flag as it will be re-derived for the qa_kinds in scope,
            # not whatever an earlier wider/narrower run stored.
            reliable = cl_baseline._is_reliable(entry, tuple(args.qa_kinds))
            flag = "" if reliable else "  [reliable=false]"
            print(f"  PASS {lang}: cached {path.name}{flag}")
        elif "pristine" in args.configs:
            print(f"  PASS {lang}: not cached; will be computed on the pristine pass")
        else:
            print(f"  FAIL {lang}: no cached baseline at {path} and 'pristine' "
                  f"is not in --configs, so no unedited model will be loaded")
            failures.append(f"baseline {lang}")

    print("\n--- environment ---")
    has_key = any(os.getenv(k) for k in
                  ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_TOKEN"))
    if "open" in args.qa_kinds and not has_key:
        print("  FAIL GEMINI_API_KEY / GOOGLE_API_KEY / GEMINI_API_TOKEN unset "
              "(required by the open-QA judge)")
        failures.append("gemini key")
    else:
        print(f"  PASS gemini key: {'set' if has_key else 'not needed (mc only)'}")

    print("\n" + "=" * 74)
    if failures:
        print(f"DRY RUN FAILED ({len(failures)} problem(s)): {failures}")
        return 1
    n_cells = sum(len(["native"] if lang == "en" else args.wrappers)
                  for lang in args.languages) * len(args.configs) * len(args.qa_kinds)
    print(f"DRY RUN PASSED -- {n_cells} cells to evaluate")
    return 0


# ========================================================================== #
# Real run                                                                    #
# ========================================================================== #

def _seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _LazyEvaluator:
    """Create the Gemini judge on first use, so MC-only runs need no API key."""

    def __init__(self) -> None:
        self._ev: Any = None

    def get(self) -> Any:
        if self._ev is None:
            from ember.evals.gemini import GeminiEvaluator
            self._ev = GeminiEvaluator()
        return self._ev


def _evaluate_subset(tm: Any, *, concept: str, lang: str, subset: str,
                     qa_kind: str, wrapper_lang: str, base: float,
                     evaluator: _LazyEvaluator, seed: int) -> Dict[str, Any]:
    """Evaluate one (subset, qa_kind) and return a result block."""
    from ember.evals.mc import prepare_mc_items

    from crosslingual.evaluate import evaluate_mc, evaluate_open, qa_frac_for

    items = cl_data.load_qa_items(f"{subset}_test", concept, lang, qa_kind)
    total = len(items)
    if total == 0:
        raise ValueError(f"No {qa_kind} items for {concept!r}/{lang}/{subset}_test")

    if qa_kind == "mc":
        prepared = prepare_mc_items(items, seed=seed)
        correct, invalid, records = evaluate_mc(tm, prepared, wrapper_lang=wrapper_lang)
        invalid_rate: Optional[float] = invalid / total
    else:
        correct, records = evaluate_open(tm, evaluator.get(), items,
                                         wrapper_lang=wrapper_lang)
        invalid_rate = None

    acc = correct / total
    return cl_io.build_subset_block(
        accuracy=acc, base=base, frac=qa_frac_for(qa_kind, acc, base),
        n=total, records=records, invalid_rate=invalid_rate,
    )


def _block_from_baseline(entry: Dict[str, Any], qa_kind: str,
                         subset: str) -> Dict[str, Any]:
    """Reuse the pristine baseline pass as the pristine native-wrapper result.

    The pristine native cell *is* the baseline measurement -- same model, same
    content, same wrapper -- so re-generating it would burn a full pass to
    reproduce numbers that are already on disk, and its frac is 1.0 by
    construction.
    """
    from crosslingual.evaluate import qa_frac_for

    src = entry[qa_kind][subset]
    base = float(src["accuracy"])
    return cl_io.build_subset_block(
        accuracy=base, base=base, frac=qa_frac_for(qa_kind, base, base),
        n=int(src["n"]), records=src.get("records", []),
        invalid_rate=src.get("invalid_rate"),
    )


def _run_cell(tm: Any, *, args: argparse.Namespace, config: str, language: str,
              wrapper: str, qa_kind: str, base_entry: Dict[str, Any],
              mmlu: Dict[str, float], evaluator: _LazyEvaluator) -> None:
    """Evaluate one (config, language, wrapper, qa_kind) cell and persist it."""
    from ember.evals.harmonic import harmonic_mean

    concept = args.concept
    wrapper_lang = cl_prompts.wrapper_lang_for(language, wrapper)
    path = cl_io.cell_json_path(args.out_dir, language, concept, config, wrapper)

    if cl_io.cell_has_qa_kind(path, qa_kind) and not args.force:
        print(f"[cl-run] skip (done): {config}/{language}/{wrapper}/{qa_kind}")
        return

    print(f"[cl-run] eval: config={config} lang={language} wrapper={wrapper} "
          f"(scaffold={wrapper_lang}) qa_kind={qa_kind}")

    reuse_baseline = (config == "pristine" and wrapper == "native"
                      and cl_baseline.baseline_is_complete(base_entry, (qa_kind,)))
    blocks: Dict[str, Dict[str, Any]] = {}
    for subset in ("qa", "simdom"):
        base = cl_baseline.get_base_accuracy(base_entry, qa_kind, subset)
        if reuse_baseline:
            blocks[subset] = _block_from_baseline(base_entry, qa_kind, subset)
        else:
            blocks[subset] = _evaluate_subset(
                tm, concept=concept, lang=language, subset=subset, qa_kind=qa_kind,
                wrapper_lang=wrapper_lang, base=base, evaluator=evaluator,
                seed=args.seed,
            )

    qa_frac = blocks["qa"]["frac"]
    simdom_frac = blocks["simdom"]["frac"]
    efficacy = 1.0 - qa_frac
    specificity = harmonic_mean([mmlu["mmlu_frac"], simdom_frac])
    metrics = {
        "qa_frac": qa_frac,
        "simdom_frac": simdom_frac,
        "mmlu_acc_en": mmlu["mmlu_acc"],
        "mmlu_frac_en": mmlu["mmlu_frac"],
        "efficacy": efficacy,
        "specificity": specificity,
        "harmonic": harmonic_mean([efficacy, specificity]),
    }

    identity = {
        "concept": concept, "language": language, "config": config,
        "wrapper": wrapper, "wrapper_lang": wrapper_lang,
        "model": args.model, "rank": args.rank, "seed": args.seed,
        "train_eval": args.train_eval,
        "reliable": bool(base_entry.get("reliable", True)),
    }
    cl_io.write_cell_qa_kind(
        path, identity=identity, qa_kind=qa_kind,
        block={"qa": blocks["qa"], "simdom": blocks["simdom"], "metrics": metrics},
    )

    row = {
        **identity, "qa_kind": qa_kind,
        "qa_acc": blocks["qa"]["accuracy"], "qa_base": blocks["qa"]["base_accuracy"],
        "qa_frac": qa_frac,
        "simdom_acc": blocks["simdom"]["accuracy"],
        "simdom_base": blocks["simdom"]["base_accuracy"],
        "simdom_frac": simdom_frac,
        "qa_invalid": blocks["qa"].get("invalid_rate"),
        "simdom_invalid": blocks["simdom"].get("invalid_rate"),
        "n_qa": blocks["qa"]["n"], "n_simdom": blocks["simdom"]["n"],
        **{k: metrics[k] for k in ("mmlu_acc_en", "mmlu_frac_en",
                                   "efficacy", "specificity", "harmonic")},
    }
    key = {k: row[k] for k in cl_io.SUMMARY_KEY}
    if args.force:
        cl_io.drop_summary_rows(args.out_dir, key)
    cl_io.append_summary_row(args.out_dir, row)
    print(f"[cl-run] wrote {path}  qa_frac={qa_frac:.4f} "
          f"efficacy={efficacy:.4f} specificity={specificity:.4f}")


def _load_model(args: argparse.Namespace):
    from ember.erasure import model_loader
    return model_loader.load_hf_model(
        args.model, "", dtype=model_loader.pick_dtype("bf16"),
    )


def _cuda_cleanup() -> None:
    """Reclaim GPU memory once a model has gone out of scope.

    ``del`` on a *parameter* only drops that local name, so freeing has to
    happen where the model's references actually live: each config's model is
    scoped inside :func:`_run_config`, and this runs in the caller once that
    frame (and every ``WrappedHFModel`` wrapping the same weights) is gone.
    Skipping this is what made the second config's load OOM: with the previous
    model still resident, ``device_map="auto"`` offloaded layers to CPU and the
    Gemma-2 lm_head untie clone then had no room.
    """
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_all_baselines(hf_model: Any, tokenizer: Any,
                          args: argparse.Namespace,
                          evaluator: _LazyEvaluator) -> Dict[str, Dict[str, Any]]:
    """Compute/read every requested language's pristine baseline on ``hf_model``."""
    out: Dict[str, Dict[str, Any]] = {}
    for language in args.languages:
        out[language] = cl_baseline.ensure_language_baseline(
            hf_model, concept=args.concept, lang=language, model_name=args.model,
            tokenizer=tokenizer, evaluator=evaluator.get() if "open" in args.qa_kinds
            else None,
            qa_kinds=tuple(args.qa_kinds),
        )
    return out


def _config_cells(args: argparse.Namespace, config: str):
    """Yield every ``(language, wrapper, qa_kind)`` cell for one config."""
    for language in args.languages:
        wrappers = ["native"] if language == "en" else args.wrappers
        for wrapper in wrappers:
            for qa_kind in args.qa_kinds:
                yield language, wrapper, qa_kind


def _config_fully_done(args: argparse.Namespace, config: str,
                       baselines: Dict[str, Dict[str, Any]]) -> bool:
    """True when every cell for ``config`` already has output (resume shortcut).

    Lets a resumed run skip a finished config's model load entirely. For
    ``pristine`` the per-language baselines must also be cached, since a later
    edited config reads them as its denominators.
    """
    if args.force:
        return False
    for language, wrapper, qa_kind in _config_cells(args, config):
        path = cl_io.cell_json_path(args.out_dir, language, args.concept,
                                    config, wrapper)
        if not cl_io.cell_has_qa_kind(path, qa_kind):
            return False
    if config == "pristine":
        return all(
            cl_baseline.baseline_is_complete(baselines.get(lang), tuple(args.qa_kinds))
            for lang in args.languages
        )
    return True


def _run_config(args: argparse.Namespace, *, config: str,
                baselines: Dict[str, Dict[str, Any]],
                evaluator: _LazyEvaluator) -> None:
    """Load a fresh model, apply ``config`` once, evaluate every language cell.

    Every model reference (the ``hf_model`` and any ``WrappedHFModel`` around
    the same weights) is a local of this frame, so returning from it drops them
    all; the caller then calls :func:`_cuda_cleanup` to reclaim the GPU before
    the next config loads. Reload-per-config with a single applied HP is the
    intended design (spec §7) -- no snapshot/restore cycling.
    """
    from ember.evals.model_wrap import ensure_wrapped_model

    print("\n" + "=" * 74)
    print(f"[cl-run] CONFIG {config}")
    print("=" * 74)

    if _config_fully_done(args, config, baselines):
        print(f"[cl-run] config {config} already complete; skipping model load")
        return

    hf_model, tokenizer = _load_model(args)
    if config != "pristine":
        cl_hp.apply_config(
            hf_model, tokenizer, config, args.concept,
            model_name=args.model, rank=args.rank, seed=args.seed,
            train_eval=args.train_eval, results_root=args.results_root,
            configs_dir=args.configs_dir,
        )
    tm = ensure_wrapped_model(hf_model, tokenizer)

    if config == "pristine":
        baselines.update(_ensure_all_baselines(hf_model, tokenizer, args, evaluator))

    mmlu = cl_baseline.english_mmlu_frac(
        tm, config=config, model_name=args.model, tokenizer=tokenizer,
    )

    for language in args.languages:
        base_entry = baselines.get(language)
        if base_entry is None:
            raise RuntimeError(
                f"no pristine baseline for language {language}; "
                f"run with 'pristine' in --configs first"
            )
        wrappers = ["native"] if language == "en" else args.wrappers
        for wrapper in wrappers:
            for qa_kind in args.qa_kinds:
                _run_cell(tm, args=args, config=config, language=language,
                          wrapper=wrapper, qa_kind=qa_kind,
                          base_entry=base_entry, mmlu=mmlu, evaluator=evaluator)

    # Explicit for intent; the frame teardown on return does the real work.
    del tm, hf_model, tokenizer


def real_run(args: argparse.Namespace) -> int:
    _seed_everything(args.seed)
    evaluator = _LazyEvaluator()
    configs = cl_hp.order_configs(args.configs)
    print(f"[cl-run] configs={configs} languages={args.languages} "
          f"wrappers={args.wrappers} qa_kinds={args.qa_kinds}")

    baselines: Dict[str, Dict[str, Any]] = {}
    for language in args.languages:
        cached = cl_baseline.read_baseline(args.model, args.concept, language)
        if cl_baseline.baseline_is_complete(cached, tuple(args.qa_kinds)):
            baselines[language] = cached

    missing = [lang for lang in args.languages if lang not in baselines]
    if missing and "pristine" not in configs:
        # The denominators must come from an unedited model, so one is loaded
        # just for them before any edit is applied.
        print(f"[cl-run] {len(missing)} baseline(s) missing {missing} and "
              f"'pristine' is not in --configs; loading a pristine model for them")
        _bl_model, _bl_tok = _load_model(args)
        baselines.update(_ensure_all_baselines(_bl_model, _bl_tok, args, evaluator))
        del _bl_model, _bl_tok
        _cuda_cleanup()

    for config in configs:
        try:
            _run_config(args, config=config, baselines=baselines,
                        evaluator=evaluator)
        finally:
            _cuda_cleanup()

    _print_delta_l(args)
    return 0


def _print_delta_l(args: argparse.Namespace) -> None:
    """Print Δ_L = r_L - r_en per (config, language, wrapper, qa_kind)."""
    summary = cl_io.load_summary(args.out_dir)
    if summary.empty:
        return
    summary = summary[summary["concept"].astype(str) == str(args.concept)]
    if summary.empty:
        return
    table = cl_io.compute_delta_l(summary)
    cols = ["config", "language", "wrapper", "qa_kind", "qa_frac",
            "qa_frac_en", "delta_l", "reliable"]
    cols = [c for c in cols if c in table.columns]
    print("\n" + "=" * 74)
    print(f"Δ_L (retention vs. the same config's English retention) -- "
          f"{args.concept!r}")
    print("=" * 74)
    print(table[cols].to_string(index=False))
    print(f"\nsummary.csv -> {cl_io.summary_csv_path(args.out_dir)}")


# ========================================================================== #
# Entry point                                                                 #
# ========================================================================== #

def main(argv: Optional[List[str]] = None) -> int:
    # Load .env before anything reads GEMINI_API_KEY, matching what
    # ember/run_erasure.py and ember/run_eval.py do.
    from dotenv import load_dotenv
    load_dotenv()

    args = build_parser().parse_args(argv)
    if args.concept not in cl_data.CONCEPTS:
        print(f"[cl-run] WARN: concept {args.concept!r} is not one of the 18 "
              f"known concepts; it must still be an exact key in every QA file.")
    args.out_dir = Path(args.out_dir)
    if args.dry_run:
        return dry_run(args)
    rc = dry_run(args)
    if rc != 0:
        print("\n[cl-run] aborting: the pre-flight checks above failed. "
              "Fix them (or narrow --configs / --languages) and re-run.")
        return rc
    return real_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
