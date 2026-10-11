"""Per-language, per-concept pristine QA/SimDom baselines (with an on-disk cache).

Every retention ratio needs a denominator, and for a non-English cell that
denominator must be *that language's own* pristine accuracy. Reusing the
English baseline would conflate the model's cross-lingual competence with the
erasure's efficacy: a French qa_frac of 0.5 would look like erasure when it may
just be that the pristine model answers half as many French questions.

Baselines are computed once on the pristine model, per (concept, language), for
both ``qa_kind`` and both subsets (``QA_test`` and ``SimdomQA_test``), always
under the *native* wrapper -- the "English wrapper on translated content"
ablation is a property of the edited-config cells, and comparing an edited cell
against a differently-wrapped baseline would mix the two effects.

Cache layout (separate from ``data/baselines/``, which is English-only and
keyed by set name rather than by language)::

    data/crosslingual_baselines/<model_safe>/<concept_safe>_<lang>.json

The per-question records are cached alongside the accuracies: they are what
the pristine/native result cell reports, so keeping them here means that cell
costs nothing to produce beyond the baseline pass it already is.

Screening rule (§6): a (concept, language) whose pristine ``QA_test`` accuracy
is below 0.40 (MC) or 0.30 (open) is still written, but flagged
``"reliable": false``. The flag propagates into every downstream result row so
the analysis step can exclude the cell from the headline Δ_L numbers instead of
the cell being silently dropped here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ember.erasure import io as ember_io
from ember.evals.baselines import _model_safe_name, compute_or_read_baseline
from ember.evals.gemini import GeminiEvaluator
from ember.evals.mc import prepare_mc_items
from ember.evals.model_wrap import ensure_wrapped_model
from ember.utils import _safe_concept, _safe_model_name

from crosslingual import data as cl_data
from crosslingual.evaluate import evaluate_mc, evaluate_open

#: Pristine QA_test accuracy below this marks the (concept, language) unreliable.
MIN_RELIABLE_MC_ACC = 0.40
MIN_RELIABLE_OPEN_ACC = 0.30

QA_KINDS: tuple[str, ...] = ("mc", "open")
SUBSETS: tuple[str, ...] = ("qa", "simdom")

BASELINE_ROOT = ember_io.DATA_DIR / "crosslingual_baselines"

#: Per-config English MMLU numbers live here so the pristine ``data/baselines/``
#: cache is never written to by this script.
MMLU_ROOT = BASELINE_ROOT / "_mmlu_by_config"


# ========================================================================== #
# Paths                                                                       #
# ========================================================================== #

def baseline_path(model_name: str, concept: str, lang: str) -> Path:
    """``data/crosslingual_baselines/<model_safe>/<concept_safe>_<lang>.json``."""
    return (BASELINE_ROOT / _safe_model_name(model_name)
            / f"{_safe_concept(concept)}_{cl_data.check_language(lang)}.json")


def read_baseline(model_name: str, concept: str, lang: str) -> Optional[Dict[str, Any]]:
    """Return the cached baseline entry, or ``None`` when absent/unreadable."""
    path = baseline_path(model_name, concept, lang)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # a truncated cache should not kill a long run
        print(f"[cl-baseline] WARN: unreadable cache {path}: {type(e).__name__}: {e}")
        return None


def baseline_is_complete(entry: Optional[Dict[str, Any]],
                         qa_kinds: List[str] | tuple[str, ...] = QA_KINDS) -> bool:
    """True when ``entry`` already holds every requested qa_kind x subset cell."""
    if not entry:
        return False
    for qa_kind in qa_kinds:
        kind_entry = entry.get(qa_kind)
        if not isinstance(kind_entry, dict):
            return False
        for subset in SUBSETS:
            if not isinstance(kind_entry.get(subset), dict):
                return False
            if kind_entry[subset].get("accuracy") is None:
                return False
    return True


# ========================================================================== #
# Screening rule                                                              #
# ========================================================================== #

_RELIABILITY_FLOOR = {"mc": MIN_RELIABLE_MC_ACC, "open": MIN_RELIABLE_OPEN_ACC}


def _is_reliable(entry: Dict[str, Any],
                 qa_kinds: List[str] | tuple[str, ...] = QA_KINDS) -> bool:
    """Apply the §6 screening rule, restricted to the qa_kinds actually in use.

    A (concept, language) is unreliable when its pristine ``QA_test`` accuracy
    for any qa_kind being evaluated falls below that kind's floor (MC < 0.40,
    open < 0.30). Only the kinds in ``qa_kinds`` are checked, so an MC-only run
    is not dragged to ``reliable=false`` by a weak open-ended baseline it never
    reports on -- which is common for a small model in a non-English language.
    """
    for qa_kind in qa_kinds:
        acc = ((entry.get(qa_kind) or {}).get("qa") or {}).get("accuracy")
        if acc is not None and float(acc) < _RELIABILITY_FLOOR[qa_kind]:
            return False
    return True


# ========================================================================== #
# Computation                                                                 #
# ========================================================================== #

def compute_language_baseline(
        pristine_model: Any,
        *,
        concept: str,
        lang: str,
        model_name: str,
        tokenizer: Any = None,
        evaluator: Optional[GeminiEvaluator] = None,
        qa_kinds: List[str] | tuple[str, ...] = QA_KINDS,
        mc_seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate the *pristine* model on one (concept, language). No caching."""
    cl_data.check_language(lang)
    tm = ensure_wrapped_model(pristine_model, tokenizer)
    wrapper_lang = lang  # native wrapper, always -- see module docstring

    entry: Dict[str, Any] = {
        "concept": concept,
        "language": lang,
        "model": model_name,
        "wrapper": "native",
        "wrapper_lang": wrapper_lang,
    }

    for qa_kind in qa_kinds:
        kind_entry: Dict[str, Any] = {}
        for subset in SUBSETS:
            set_name = f"{subset}_test"
            items = cl_data.load_qa_items(set_name, concept, lang, qa_kind)
            total = len(items)
            if total == 0:
                raise ValueError(
                    f"No {qa_kind} items for concept={concept!r} lang={lang} "
                    f"set={set_name}; check data/qa_lang/."
                )

            if qa_kind == "mc":
                prepared = prepare_mc_items(items, seed=mc_seed)
                correct, invalid, records = evaluate_mc(
                    tm, prepared, wrapper_lang=wrapper_lang)
                kind_entry[subset] = {
                    "accuracy": correct / total,
                    "n": total,
                    "invalid_rate": invalid / total,
                    "records": records,
                }
            else:
                ev = evaluator or GeminiEvaluator()
                correct, records = evaluate_open(
                    tm, ev, items, wrapper_lang=wrapper_lang)
                kind_entry[subset] = {
                    "accuracy": correct / total,
                    "n": total,
                    "records": records,
                }

            print(f"[cl-baseline] {concept!r} lang={lang} {qa_kind}/{subset}_test "
                  f"acc={kind_entry[subset]['accuracy']:.4f} (n={total})")
        entry[qa_kind] = kind_entry

    entry["reliable"] = _is_reliable(entry, qa_kinds)
    entry["screened_on"] = sorted(qa_kinds)
    if not entry["reliable"]:
        floors = ", ".join(f"{k}<{_RELIABILITY_FLOOR[k]}" for k in qa_kinds)
        print(f"[cl-baseline] SCREENED OUT: {concept!r} lang={lang} is below the "
              f"reliability floor ({floors}); rows will carry reliable=false")
    return entry


def ensure_language_baseline(
        pristine_model: Any,
        *,
        concept: str,
        lang: str,
        model_name: str,
        tokenizer: Any = None,
        evaluator: Optional[GeminiEvaluator] = None,
        qa_kinds: List[str] | tuple[str, ...] = QA_KINDS,
        force: bool = False,
) -> Dict[str, Any]:
    """Return the (concept, language) baseline, computing + caching on a miss.

    ``pristine_model`` must be an unedited model: this is the denominator for
    every edited config in that language.
    """
    cached = None if force else read_baseline(model_name, concept, lang)
    if baseline_is_complete(cached, qa_kinds):
        print(f"[cl-baseline] cache hit: {concept!r} lang={lang}")
        # The stored reliable flag reflects whatever qa_kinds the run that
        # wrote it used; re-derive it for the qa_kinds in scope now so an
        # MC-only run is not stuck with an MC+open verdict (or vice versa).
        flag = _is_reliable(cached, qa_kinds)
        if cached.get("reliable") != flag or cached.get("screened_on") != sorted(qa_kinds):
            cached["reliable"] = flag
            cached["screened_on"] = sorted(qa_kinds)
            ember_io.save_json_atomic(baseline_path(model_name, concept, lang), cached)
            print(f"[cl-baseline]   reliability re-evaluated for "
                  f"qa_kinds={sorted(qa_kinds)}: reliable={flag}")
        return cached

    if pristine_model is None:
        raise RuntimeError(
            f"No cached cross-lingual baseline for concept={concept!r} lang={lang} "
            f"and no pristine model available to compute one. Run this concept "
            f"with 'pristine' in --configs at least once."
        )

    entry = compute_language_baseline(
        pristine_model, concept=concept, lang=lang, model_name=model_name,
        tokenizer=tokenizer, evaluator=evaluator, qa_kinds=qa_kinds,
    )
    # Merge over any partial cache so a qa_kind computed by an earlier,
    # narrower run is not thrown away.
    merged = dict(cached or {})
    merged.update({k: v for k, v in entry.items() if k != "reliable"})
    merged["reliable"] = _is_reliable(merged, qa_kinds)
    merged["screened_on"] = sorted(qa_kinds)

    path = baseline_path(model_name, concept, lang)
    ember_io.save_json_atomic(path, merged)
    print(f"[cl-baseline] wrote {path}")
    return merged


def get_base_accuracy(entry: Dict[str, Any], qa_kind: str, subset: str) -> float:
    """Pristine accuracy for one (qa_kind, subset) out of a baseline entry."""
    try:
        return float(entry[qa_kind][subset]["accuracy"])
    except (KeyError, TypeError) as e:
        raise KeyError(
            f"Baseline entry has no {qa_kind}/{subset} accuracy "
            f"(concept={entry.get('concept')!r} lang={entry.get('language')})"
        ) from e


# ========================================================================== #
# English MMLU, once per config                                               #
# ========================================================================== #

def english_mmlu_frac(
        model: Any,
        *,
        config: str,
        model_name: str,
        tokenizer: Any = None,
        split: str = "test",
        force: bool = False,
) -> Dict[str, float]:
    """Return ``{"mmlu_acc", "mmlu_base", "mmlu_frac"}`` for the resident model.

    MMLU is general-domain: it is neither concept- nor language-specific, so it
    is evaluated **once per config, in English only**, and the resulting
    ``mmlu_frac`` is reused as the first term of ``specificity`` in every
    language. Sweeping it across languages would be pure waste.

    The per-config MMLU number is cached under
    ``data/crosslingual_baselines/_mmlu_by_config/<config>/`` via
    :func:`ember.evals.baselines.compute_or_read_baseline`, so the pristine
    ``data/baselines/`` cache is read but never written by this script. For
    ``config == "pristine"`` the number is by definition the pristine baseline,
    so that cache is read directly and no GPU work happens at all.
    """
    set_name = f"mmlu_{split}"
    pristine_path = (ember_io.baseline_dir(model_name) / f"baseline_{set_name}.json")
    if not pristine_path.exists():
        raise FileNotFoundError(
            f"Pristine English MMLU baseline missing: {pristine_path}. It is the "
            f"denominator of mmlu_frac; run the English pipeline (or "
            f"ember.evals.baselines) for this model first."
        )
    mmlu_base = float(json.loads(pristine_path.read_text(encoding="utf-8"))
                      ["metrics"]["accuracy_generation"])

    if config == "pristine" and not force:
        mmlu_acc = mmlu_base
    else:
        out_dir = MMLU_ROOT / config
        out_dir.mkdir(parents=True, exist_ok=True)
        if force:
            stale = out_dir / _model_safe_name(model_name) / f"baseline_{set_name}.json"
            if stale.exists():
                stale.unlink()
        result = compute_or_read_baseline(
            ensure_wrapped_model(model, tokenizer), set_name, out_dir=str(out_dir),
        )
        mmlu_acc = float(result.metrics.get("accuracy_generation", 0.0))

    frac = (0.0 if (mmlu_base - 0.25) == 0
            else min(1.0, max((mmlu_acc - 0.25) / (mmlu_base - 0.25), 0.0)))
    print(f"[cl-baseline] config={config} mmlu_acc={mmlu_acc:.4f} "
          f"base={mmlu_base:.4f} frac={frac:.4f}")
    return {"mmlu_acc": mmlu_acc, "mmlu_base": mmlu_base, "mmlu_frac": frac}


__all__ = [
    "MIN_RELIABLE_MC_ACC", "MIN_RELIABLE_OPEN_ACC", "QA_KINDS", "SUBSETS",
    "BASELINE_ROOT", "MMLU_ROOT",
    "baseline_path", "read_baseline", "baseline_is_complete",
    "compute_language_baseline", "ensure_language_baseline",
    "get_base_accuracy", "english_mmlu_frac",
]
