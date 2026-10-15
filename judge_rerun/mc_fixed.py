"""Fixed multiple-choice scoring: longer generation budget + a robust letter parser.

Why this exists (2026-10-15). The paper's evaluators (ember/evals/mc.py,
ember/evals/mmlu.py) generate only 4 (MC) / 6 (MMLU) new tokens and take the
first standalone A-D letter. Under the EMBER embedding edit Gemma often answers
"The answer is **C**" instead of a bare "C", so the 4-token cut ends at
"The answer is **" and the answer counts as wrong ("invalid"). That inflated
EMBER / SNMF+EMBER efficacy. Two further problems:

  - the parser rejects a lowercase "c";
  - WrappedHFModel._extract_response cuts the decoded text at the first
    "model\\n", so a prompt whose option ends with the word "model" returns
    the rest of the prompt as the "answer" (seen as 'B. Bellman equation\\n
    C. ...model\\nD' -> parsed B, real answer D).

This module fixes all three without touching ember/:
  - generation budget MC_MAX_NEW_TOKENS / MMLU_MAX_NEW_TOKENS (32);
  - only the newly generated tokens are decoded (no prompt-header search);
  - parse_letter_fixed (below) reads bare letters in any case, "C. text",
    "**B**", and "<answer phrase> ... X" in en/de/es/fr/ru.

Greedy decoding means the first 4 (6) generated tokens are exactly what the old
evaluator produced, so every record also carries ``old_raw`` / ``old_letter``:
the first OLD_* tokens decoded and parsed with the paper's parse_letter. Their
totals must reproduce the old grid numbers; that is the determinism control.

Prompts, option shuffle, one-question-at-a-time MC and batch-8 MMLU are kept
exactly as in ember/evals so the generated tokens are identical.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import torch

from ember.evals.mc import build_mc_prompt, parse_letter
from ember.evals.mmlu import build_mmlu_prompt
from ember.evals.schema import LETTER_CHOICES

MC_MAX_NEW_TOKENS = 32
MMLU_MAX_NEW_TOKENS = 32
OLD_MC_MAX_NEW_TOKENS = 4     # ember/evals/mc.py default
OLD_MMLU_MAX_NEW_TOKENS = 6   # ember/evals/mmlu.py default
MMLU_BATCH_SIZE = 8           # what every grid used; keep it so padding is identical

# 1. The whole output is one letter, any case, any wrapping: "c", "C.", "**B**", "(D)".
_LONE = re.compile(r"[\W_]*([A-Da-d])[\W_]*")
# 2. Starts with a capital letter plus a delimiter: "C. Hadrian", "D) x", "**B** x", "A:".
#    Needs the delimiter, so "A fire lance ..." (article) does not match.
_LEAD = re.compile(r"[\s*_#>(\[]*([A-D])(?:[.):\]]|\*\*|\s*$)")
# 3. An answer phrase, then the letter within the same line:
#    "The answer is **C**", "Die richtige Antwort ist B", "La réponse est : **D**",
#    "Ответ: A". Keywords are case-insensitive, the letter must be a capital.
_ANSWER = re.compile(
    r"(?i:answer|antwort|respuesta|r[ée]ponse|ответ|opci[óo]n|option)"
    r"[^\n]{0,30}?(?<![A-Za-z])\(?\*{0,2}([A-D])(?![A-Za-z])"
)


def parse_letter_fixed(raw: str) -> str:
    """First answer letter in ``raw`` by the rules above; ``""`` if none.

    No "first capital A-D anywhere" fallback: with a 32-token budget that would
    read the article in "A fire lance is ..." as option A.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    m = _LONE.fullmatch(s) or _LEAD.match(s) or _ANSWER.search(s)
    return m.group(1).upper() if m else ""


# ---------------------------------------------------------------------------- #
# Generation: same tokenization / kwargs as WrappedHFModel, new tokens only     #
# ---------------------------------------------------------------------------- #

@torch.no_grad()
def _generate_ids(tm: Any, wrapped_prompts: Sequence[str], max_new_tokens: int) -> List[List[int]]:
    """Greedy generation; returns the generated token ids per prompt (prompt removed)."""
    inputs = {k: v.to(tm._device()) for k, v in tm._tokenize(list(wrapped_prompts)).items()}
    out = tm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tm.tokenizer.pad_token_id)
    n_in = inputs["input_ids"].shape[1]
    return [row[n_in:].tolist() for row in out]


def _decode(tm: Any, ids: Sequence[int]) -> str:
    return tm.tokenizer.decode(ids, skip_special_tokens=True).strip()


def _record(tm: Any, ids: List[int], n_old: int) -> Dict[str, Any]:
    raw = _decode(tm, ids)
    old_raw = _decode(tm, ids[:n_old])
    return {
        "model_raw": raw,
        "model_letter": parse_letter_fixed(raw),
        "old_raw": old_raw,
        "old_letter": parse_letter(old_raw),
        "n_new_tokens": len(ids),
    }


def _tally(records: List[Dict[str, Any]], correct_key: str) -> Dict[str, Any]:
    n = len(records)
    for r in records:
        r["is_valid"] = r["model_letter"] in LETTER_CHOICES
        r["is_correct"] = r["is_valid"] and r["model_letter"] == r[correct_key]
        r["old_is_valid"] = r["old_letter"] in LETTER_CHOICES
        r["old_is_correct"] = r["old_is_valid"] and r["old_letter"] == r[correct_key]
    frac = lambda k: (sum(bool(r[k]) for r in records) / n) if n else float("nan")
    return {
        "acc": frac("is_correct"),
        "invalid_rate": 1.0 - frac("is_valid"),
        "old_acc": frac("old_is_correct"),
        "old_invalid_rate": 1.0 - frac("old_is_valid"),
        "n": n,
        "records": records,
    }


def evaluate_mc_fixed(tm: Any, items: List[Any],
                      max_new_tokens: int = MC_MAX_NEW_TOKENS) -> Dict[str, Any]:
    """MC eval, one question at a time (as ember/evals/mc.py). Items are PreparedMCItem."""
    records = []
    for it in items:
        wrapped = tm.wrap_prompt(build_mc_prompt(it.question, it.options_by_letter))
        ids = _generate_ids(tm, [wrapped], max_new_tokens)[0]
        records.append({
            "question": it.question,
            "options": it.options_by_letter,
            "correct_letter": it.correct_letter,
            "correct_text": it.correct_text,
            **_record(tm, ids, OLD_MC_MAX_NEW_TOKENS),
        })
    return _tally(records, "correct_letter")


def evaluate_mmlu_fixed(tm: Any, items: List[Any],
                        max_new_tokens: int = MMLU_MAX_NEW_TOKENS,
                        batch_size: int = MMLU_BATCH_SIZE) -> Dict[str, Any]:
    """MMLU eval, batched as ember/evals/mmlu.py. Items are MMLUItem."""
    wrapped = [tm.wrap_prompt(build_mmlu_prompt(it.question, it.choices)) for it in items]
    records = []
    for start in range(0, len(items), batch_size):
        batch_ids = _generate_ids(tm, wrapped[start:start + batch_size], max_new_tokens)
        for it, ids in zip(items[start:start + batch_size], batch_ids):
            records.append({
                "question": it.question,
                "choices": dict(zip(LETTER_CHOICES, it.choices)),
                "correct_letter": it.answer_letter,
                **_record(tm, ids, OLD_MMLU_MAX_NEW_TOKENS),
            })
    return _tally(records, "correct_letter")


__all__ = [
    "MC_MAX_NEW_TOKENS", "MMLU_MAX_NEW_TOKENS", "parse_letter_fixed",
    "evaluate_mc_fixed", "evaluate_mmlu_fixed",
]
