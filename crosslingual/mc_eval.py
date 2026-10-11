"""Evaluate one (concept, language) cell: QA_test + SimdomQA_test, MC.

Shared by the pristine baseline and (later) the edited-model runs, so both
see the same items, shuffle, prompts and decoding. Decoding matches
``ember.evals.mc.evaluate_mc_generation``: greedy, one question at a time,
``max_new_tokens=4``, first standalone A-D letter parsed.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ember.evals.mc import parse_letter
from ember.evals.schema import LETTER_CHOICES

from crosslingual.data import SETS, load_mc_items
from crosslingual.prompts import build_mc_prompt

MAX_NEW_TOKENS = 4


def evaluate_set(tm: Any, concept: str, lang: str, set_name: str, *,
                 seed: int = 42, limit: Optional[int] = None) -> Dict[str, Any]:
    """Run one set through a ``WrappedHFModel``; return accuracy + records."""
    items = load_mc_items(concept, lang, set_name, seed=seed)
    if limit is not None:
        items = items[:limit]

    t0 = time.time()
    records = []
    for it in items:
        prompt = build_mc_prompt(it["question"], it["options"], lang)
        raw = tm.generate(tm.wrap_prompt(prompt), max_new_tokens=MAX_NEW_TOKENS,
                          temperature=0.0, do_sample=False)
        letter = parse_letter(raw)
        records.append({
            **it,
            "model_raw": raw,
            "model_letter": letter,
            "is_valid": letter in LETTER_CHOICES,
            "is_correct": letter == it["correct_letter"],
        })

    n = len(records)
    return {
        "n": n,
        "accuracy": sum(r["is_correct"] for r in records) / n,
        "invalid_rate": sum(not r["is_valid"] for r in records) / n,
        "seconds": round(time.time() - t0, 2),
        "records": records,
    }


def evaluate_cell(tm: Any, concept: str, lang: str, *, seed: int = 42,
                  limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """``{set_name: evaluate_set(...)}`` for every set in :data:`SETS`."""
    return {s: evaluate_set(tm, concept, lang, s, seed=seed, limit=limit)
            for s in SETS}
