"""Loaders for the multilingual evaluation database (results/ml/data/{qa,mmlu}).

Items are ember's dataclasses, so they plug straight into
``judge_rerun.mc_fixed.evaluate_mc_fixed`` / ``evaluate_mmlu_fixed`` (and ember/evals):

    load_mc(lang, concept, split)  -> List[PreparedMCItem]
    load_mmlu(lang, split)         -> List[MMLUItem]

Prompts: use ``experiments.ml.prompts.build_mc_prompt(it.question, it.options_by_letter, lang)``
for the native scaffold (``build_mmlu`` style: ``dict(zip(LETTER_CHOICES, it.choices))``).
The ember evaluators build the English scaffold.

MC option order is **language-aligned**: the permutation is seeded by
``(concept, split, index, seed)`` only (same rule as crosslingual/data.py), so item *i*
has the same correct letter in every language. Note this differs from ember's own
``prepare_mc_items`` shuffle (hashed on the English text), so English letters here are
not those of the single-language EMBER runs.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List

from crosslingual.data import _permutation
from ember.evals.schema import LETTER_CHOICES, MMLUItem, PreparedMCItem
from experiments.ml.evaldata.common import LANGS, MMLU_DIR, QA_DIR, SPLITS

MMLU_SPLITS = ("validation", "test")
#: the δ grid runs on "train"; for MMLU that is the 50-item validation split
_MMLU_ALIAS = {"train": "validation", "val": "validation", "dev": "validation"}


def _check_lang(lang: str) -> None:
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}; expected one of {LANGS}")


@lru_cache(maxsize=None)
def _mc(lang: str) -> Dict[str, Any]:
    _check_lang(lang)
    return json.loads((QA_DIR / f"mc_questions_{lang}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _mmlu(lang: str) -> Dict[str, Any]:
    _check_lang(lang)
    return json.loads((MMLU_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def load_mc(lang: str, concept: str, split: str, seed: int = 42) -> List[PreparedMCItem]:
    """The 50 MC items of (lang, concept, split), options shuffled language-aligned."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    data, en = _mc(lang), _mc("en")
    if concept not in data:
        raise KeyError(f"concept {concept!r} not in mc_questions_{lang}.json")
    out: List[PreparedMCItem] = []
    for i, (obj, eobj) in enumerate(zip(data[concept][split], en[concept][split], strict=True)):
        options = [o.strip() for o in obj["options"]]
        correct = obj["correct_answer"].strip()
        if len(options) != 4 or correct not in options:
            raise ValueError(f"malformed item {lang}/{concept}/{split}[{i}]")
        # correct position from the English item (translations may collide on a distractor)
        ci = [o.strip() for o in eobj["options"]].index(eobj["correct_answer"].strip())
        if options[ci] != correct:
            raise ValueError(f"correct answer position mismatch {lang}/{concept}/{split}[{i}]")
        perm = _permutation(concept, split, i, seed)
        by_letter = {L: options[p] for L, p in zip(LETTER_CHOICES, perm)}
        letter = LETTER_CHOICES[perm.index(ci)]  # position-based: safe even if two options read alike
        out.append(PreparedMCItem(concept=concept, subset=split.split("_")[0], split=split.split("_")[1],
                                  question=obj["q"].strip(), options_by_letter=by_letter,
                                  correct_letter=letter, correct_text=correct))
    return out


def load_mmlu(lang: str, split: str) -> List[MMLUItem]:
    """Global-MMLU items (50 validation / 200 test), same ids and order in every language."""
    split = _MMLU_ALIAS.get(split, split)
    if split not in MMLU_SPLITS:
        raise ValueError(f"split must be one of {MMLU_SPLITS} (or train), got {split!r}")
    return [MMLUItem(question=it["question"], choices=[it["options"][L] for L in LETTER_CHOICES],
                     answer_letter=it["answer"]) for it in _mmlu(lang)[split]]


def mmlu_ids(split: str) -> List[str]:
    split = _MMLU_ALIAS.get(split, split)
    return [it["id"] for it in _mmlu("en")[split]]
