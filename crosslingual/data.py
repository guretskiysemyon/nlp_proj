"""Language-aware QA loaders.

A parameterized copy of :func:`ember.local_datasets.load_mc_qa_items` /
:func:`ember.local_datasets.load_open_qa_examples`. The only difference is the
``lang`` argument, which selects the file to open; the item dataclasses
(:class:`~ember.local_datasets.MCQAItem`, :class:`~ember.local_datasets.OpenQAItem`)
and the parsing rules are reused verbatim so a ``lang="en"`` call returns
byte-identical items to the English pipeline.

Per-language aggregates live at ``data/qa_lang/{mc,open}_questions_<lang>.json``
and are structurally identical to the canonical English files: a dict keyed by
the exact concept string, each holding ``QA_train`` / ``QA_test`` /
``SimdomQA_train`` / ``SimdomQA_test`` lists. The per-concept files under
``data/qa_lang/translated/<lang>/`` are a legacy intermediate from the
translation step (their filenames do not follow ``ember.utils._safe_concept``)
and are deliberately never read here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from ember.local_datasets import (
    DATA_DIR,
    MCQAItem,
    OpenQAItem,
    _load_json,
    _parse_set_name,
)

LANGUAGES: tuple[str, ...] = ("en", "de", "es", "fr", "ru")

#: The 18 concept keys, spelled exactly as they appear in every QA JSON.
CONCEPTS: tuple[str, ...] = (
    "Ancient Rome", "Artificial intelligence", "Baseball", "COVID-19 pandemic",
    "Cannabis", "Culture of Greece", "Gambling", "Golf", "Gun", "Halloween",
    "Harry Potter", "Heroin", "Nazism", "Pornography", "Republic of Ireland",
    "Uranium", "Valentine's Day", "World War II",
)

QA_LANG_DIR = DATA_DIR / "qa_lang"


def check_language(lang: str) -> str:
    if lang not in LANGUAGES:
        raise ValueError(f"Unknown language {lang!r}; expected one of {LANGUAGES}")
    return lang


def mc_path(lang: str) -> Path:
    """Path to the MC questions file for ``lang`` (canonical file for ``en``)."""
    check_language(lang)
    return (DATA_DIR / "mc_questions.json" if lang == "en"
            else QA_LANG_DIR / f"mc_questions_{lang}.json")


def open_path(lang: str) -> Path:
    """Path to the open questions file for ``lang`` (canonical file for ``en``)."""
    check_language(lang)
    return (DATA_DIR / "open_questions.json" if lang == "en"
            else QA_LANG_DIR / f"open_questions_{lang}.json")


def qa_path(lang: str, qa_kind: str) -> Path:
    if qa_kind == "mc":
        return mc_path(lang)
    if qa_kind == "open":
        return open_path(lang)
    raise ValueError(f"qa_kind must be 'mc' or 'open', got {qa_kind!r}")


# ========================================================================== #
# Loaders (parameterized copies of ember.local_datasets)                      #
# ========================================================================== #

def load_mc_qa_items(set_name: str, concept: str | None = None,
                     lang: str = "en") -> List[MCQAItem]:
    """Load multiple-choice QA items in ``lang``.

    ``set_name``: ``"qa_train"``, ``"qa_test"``, ``"simdom_train"``,
    ``"simdom_test"``.
    """
    subset, split = _parse_set_name(set_name)
    json_data = _load_json(mc_path(lang))
    key = f"{subset}_{split}"
    items: List[MCQAItem] = []
    for concept_name, concept_data in json_data.items():
        if concept is not None and concept_name != concept:
            continue
        for obj in concept_data.get(key, []):
            q = obj.get("q", "").strip()
            a = obj.get("correct_answer", "").strip()
            options = obj.get("options", []) or []
            if q and a and len(options) >= 4:
                items.append(MCQAItem(concept=concept_name, subset=subset, split=split,
                                      question=q, options=list(options),
                                      correct_answer=a))
    return items


def load_open_qa_examples(set_name: str, concept: str | None = None,
                          lang: str = "en") -> List[OpenQAItem]:
    """Load open-ended QA items in ``lang``.

    ``set_name``: ``"qa_train"``, ``"qa_test"``, ``"simdom_train"``,
    ``"simdom_test"``.
    """
    subset, split = _parse_set_name(set_name)
    json_data = _load_json(open_path(lang))
    key = f"{subset}_{split}"
    items: List[OpenQAItem] = []
    for concept_name, concept_data in json_data.items():
        if concept is not None and concept_name != concept:
            continue
        for obj in concept_data.get(key, []):
            q, a = obj.get("q", "").strip(), obj.get("a", "").strip()
            if q and a:
                items.append(OpenQAItem(concept=concept_name, subset=subset,
                                        split=split, question=q, answer=a))
    return items


def load_qa_items(set_name: str, concept: str, lang: str, qa_kind: str) -> List[Any]:
    """Dispatch to the MC or open loader by ``qa_kind``."""
    if qa_kind == "mc":
        return load_mc_qa_items(set_name, concept=concept, lang=lang)
    if qa_kind == "open":
        return load_open_qa_examples(set_name, concept=concept, lang=lang)
    raise ValueError(f"qa_kind must be 'mc' or 'open', got {qa_kind!r}")


# ========================================================================== #
# Validation helpers (used by --dry-run)                                      #
# ========================================================================== #

def concept_keys(lang: str, qa_kind: str) -> List[str]:
    """Top-level concept keys present in one language/kind file."""
    path = qa_path(lang, qa_kind)
    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return list(json.load(f).keys())


def describe_concept(concept: str, lang: str, qa_kind: str) -> dict:
    """Per-subset item counts for one (concept, lang, qa_kind), for dry-run output."""
    counts = {}
    for set_name in ("qa_test", "simdom_test"):
        counts[set_name] = len(load_qa_items(set_name, concept, lang, qa_kind))
    return counts


__all__ = [
    "LANGUAGES", "CONCEPTS", "QA_LANG_DIR",
    "check_language", "mc_path", "open_path", "qa_path",
    "load_mc_qa_items", "load_open_qa_examples", "load_qa_items",
    "concept_keys", "describe_concept",
]
