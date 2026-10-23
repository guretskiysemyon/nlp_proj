"""Unique English segments to translate per language (MC for it/pt/hi/th, MMLU for th)."""
from __future__ import annotations

import json
from typing import Dict, List

from experiments.ml.evaldata.common import MMLU_DIR, MT_LANGS, en_mc, needs_translation, segment

#: translated first (coordinator priority, 2026-10-18)
PRIORITY_CONCEPTS = ("Harry Potter", "Ancient Rome", "Valentine's Day")


def mc_strings(concepts=None) -> List[str]:
    """Every question and option string of data/mc_questions.json (or of ``concepts``), unique."""
    seen: Dict[str, None] = {}
    for name, concept in en_mc().items():
        if concepts is not None and name not in concepts:
            continue
        for items in concept.values():
            for it in items:
                for s in [it["q"], *it["options"]]:
                    seen.setdefault(s.strip(), None)
    return list(seen)


def mmlu_strings() -> List[str]:
    """Question and option strings of the selected English MMLU items (for Thai)."""
    d = json.loads((MMLU_DIR / "en.json").read_text(encoding="utf-8"))
    seen: Dict[str, None] = {}
    for items in d.values():
        for it in items:
            for s in [it["question"], *it["options"].values()]:
                seen.setdefault(s, None)
    return list(seen)


def strings_for(lang: str) -> List[str]:
    assert lang in MT_LANGS
    return mc_strings() + (mmlu_strings() if lang == "th" else [])


def segments_of(strings: List[str]) -> List[str]:
    """Unique segments of ``strings`` that need a model translation."""
    seen: Dict[str, None] = {}
    for s in strings:
        for seg, _ in segment(s):
            if seg.strip() and needs_translation(seg):
                seen.setdefault(seg.strip(), None)
    return list(seen)


def segments_for(lang: str) -> List[str]:
    return segments_of(strings_for(lang))


#: options of one MC item are also translated together, as one "a; b; c; d" line, when every
#: option is at most this many words: gives single words / names their context
CTX_MAX_WORDS = 5
CTX_SEP = "; "


def option_lists(lang: str) -> List[str]:
    """Joined option lists ("a; b; c; d") of the MC items with short options (+ Thai MMLU)."""
    lists: List[List[str]] = []
    for concept in en_mc().values():
        for items in concept.values():
            lists += [[o.strip() for o in it["options"]] for it in items]
    if lang == "th":
        d = json.loads((MMLU_DIR / "en.json").read_text(encoding="utf-8"))
        lists += [list(it["options"].values()) for items in d.values() for it in items]
    seen: Dict[str, None] = {}
    for opts in lists:
        if all(len(o.split()) <= CTX_MAX_WORDS and ";" not in o and "\n" not in o for o in opts) \
                and any(needs_translation(o) for o in opts):
            seen.setdefault(CTX_SEP.join(opts), None)
    return list(seen)
