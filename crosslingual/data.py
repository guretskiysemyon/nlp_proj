"""Translated MC items for the cross-lingual experiment.

Source: ``data/qa_lang/mc_questions_<lang>.json`` (the ``en`` file is
byte-identical to ``data/mc_questions.json``). Each file is keyed by the exact
concept string and holds ``QA_train`` / ``QA_test`` / ``SimdomQA_train`` /
``SimdomQA_test`` lists. Item *i* is the same question in every language, and
its options are listed in the same order (checked for all 7,200 non-English
test items on 2026-10-10).

Option shuffling is **language-aligned**: the permutation is seeded by
``(concept, set, index, seed)`` only, never by the (translated) text, so
question *i* has the same correct letter in every language. That keeps
letter-position bias identical across languages, so it cannot show up as a
cross-lingual difference.
"""
from __future__ import annotations

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from ember.evals.schema import LETTER_CHOICES

REPO_ROOT = Path(__file__).resolve().parents[1]
QA_LANG_DIR = REPO_ROOT / "data" / "qa_lang"

LANGUAGES: tuple[str, ...] = ("en", "de", "es", "fr", "ru")

#: The 18 concept keys, spelled exactly as in every QA JSON.
CONCEPTS: tuple[str, ...] = (
    "Ancient Rome", "Artificial intelligence", "Baseball", "COVID-19 pandemic",
    "Cannabis", "Culture of Greece", "Gambling", "Golf", "Gun", "Halloween",
    "Harry Potter", "Heroin", "Nazism", "Pornography", "Republic of Ireland",
    "Uranium", "Valentine's Day", "World War II",
)

#: Test sets evaluated: the concept's own questions, then the similar-domain ones.
SETS: tuple[str, ...] = ("QA_test", "SimdomQA_test")


def mc_path(lang: str) -> Path:
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language {lang!r}; expected one of {LANGUAGES}")
    return QA_LANG_DIR / f"mc_questions_{lang}.json"


@lru_cache(maxsize=None)
def _load(lang: str) -> Dict[str, Any]:
    with mc_path(lang).open(encoding="utf-8") as f:
        return json.load(f)


def _permutation(concept: str, set_name: str, index: int, seed: int) -> List[int]:
    """Option-order permutation for one question; independent of language."""
    key = f"{concept}|{set_name}|{index}|{seed}".encode("utf-8")
    perm = list(range(len(LETTER_CHOICES)))
    random.Random(int(hashlib.sha1(key).hexdigest()[:8], 16)).shuffle(perm)
    return perm


def load_mc_items(concept: str, lang: str, set_name: str,
                  seed: int = 42) -> List[Dict[str, Any]]:
    """Return the prepared (lettered, shuffled) MC items of one set.

    Each item: ``index``, ``question``, ``options`` ({letter: text}),
    ``correct_letter``, ``correct_text``.
    """
    data = _load(lang)
    if concept not in data:
        raise KeyError(f"concept {concept!r} not in {mc_path(lang)}")
    if set_name not in SETS:
        raise ValueError(f"set_name must be one of {SETS}, got {set_name!r}")

    items: List[Dict[str, Any]] = []
    for i, obj in enumerate(data[concept][set_name]):
        options = [o.strip() for o in obj["options"]]
        correct = obj["correct_answer"].strip()
        if len(options) != 4 or correct not in options:
            raise ValueError(f"malformed item {concept!r}/{lang}/{set_name}[{i}]")
        perm = _permutation(concept, set_name, i, seed)
        by_letter = {L: options[p] for L, p in zip(LETTER_CHOICES, perm)}
        correct_letter = next(L for L, o in by_letter.items() if o == correct)
        items.append({
            "index": i,
            "question": obj["q"].strip(),
            "options": by_letter,
            "correct_letter": correct_letter,
            "correct_text": correct,
        })
    return items
