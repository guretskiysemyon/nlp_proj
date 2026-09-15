"""Language/wrapper-aware MC + open-QA eval loops.

These are copies of :func:`ember.evals.mc.evaluate_mc_generation` and
:func:`ember.evals.open_qa.evaluate_open_qa` with exactly one change: the
prompt comes from :mod:`crosslingual.prompts` so the instruction scaffold can
be rendered in a chosen ``wrapper_lang``. They are duplicated rather than
imported because the originals build their prompts inline.

Everything else is reused unmodified:

- :func:`ember.evals.mc.parse_letter`, :func:`ember.evals.mc.prepare_mc_items`
  and :data:`ember.evals.schema.LETTER_CHOICES` -- none of them are
  language-sensitive (the letters stay Latin in every language).
- :class:`ember.evals.model_wrap.WrappedHFModel` -- its chat-template wrapping
  and response extraction are language-agnostic; ``_extract_response``'s
  ``"model\\n"`` marker is a Gemma control token, not natural-language text.
- :class:`ember.evals.gemini.GeminiEvaluator` -- the judge rubric is English
  while the question / gold answer / attempted answer it grades are passed
  through verbatim in whatever language they arrive in. That "English rubric,
  any-language content" pattern is intentional.
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Tuple

from ember.evals.gemini import GeminiBadFinishError, GeminiEvaluator
from ember.evals.mc import parse_letter
from ember.evals.schema import LETTER_CHOICES, PreparedMCItem

from crosslingual.prompts import build_mc_prompt, build_open_prompt


def evaluate_mc(
        model: Any,
        items: List[PreparedMCItem],
        wrapper_lang: str = "en",
        max_new_tokens: int = 4,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """One-question-at-a-time MC generation evaluation.

    ``model`` must already be an ``ensure_wrapped_model``-wrapped
    :class:`~ember.evals.model_wrap.WrappedHFModel`.

    Returns ``(correct, invalid, per_question_records)``.
    """
    records: List[Dict[str, Any]] = []
    correct = 0
    invalid = 0

    for item in items:
        prompt = build_mc_prompt(item.question, item.options_by_letter,
                                 wrapper_lang=wrapper_lang)
        wrapped = model.wrap_prompt(prompt)
        raw = model.generate(wrapped, max_new_tokens=max_new_tokens,
                             temperature=0.0, do_sample=False)
        letter = parse_letter(raw)
        is_valid = letter in LETTER_CHOICES
        is_correct = is_valid and (letter == item.correct_letter)

        if not is_valid:
            invalid += 1
        if is_correct:
            correct += 1

        records.append({
            "concept": item.concept,
            "subset": item.subset,
            "split": item.split,
            "question": item.question,
            "options": item.options_by_letter,
            "correct_letter": item.correct_letter,
            "correct_text": item.correct_text,
            "model_raw": raw,
            "model_letter": letter,
            "is_valid": is_valid,
            "is_correct": is_correct,
            "wrapper_lang": wrapper_lang,
        })
    return correct, invalid, records


def evaluate_open(
        model: Any,
        evaluator: GeminiEvaluator,
        items: List[Any],
        wrapper_lang: str = "en",
        batch_size: int = 64,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Open-QA eval with Gemini scoring.

    Returns ``(correct_count, per_question_records)``. Gemini refusals are
    retried up to 3x; persistent failures default the item to incorrect.
    """
    n = len(items)
    prompts = [build_open_prompt(ex.question, wrapper_lang=wrapper_lang)
               for ex in items]

    answers: List[str] = []
    for i in range(0, n, batch_size):
        batch = prompts[i:i + batch_size]
        answers.extend(model.generate_multiple(
            batch, max_new_tokens=200, do_sample=False,
            batch_size=len(batch), verbose=False,
        ))
    if len(answers) != n:
        raise RuntimeError(
            f"Open-QA generation count mismatch: {n} questions, {len(answers)} answers"
        )

    correct = 0
    records: List[Any] = [None] * n

    def grade(idx: int, ex: Any, attempted: str) -> Tuple[int, Any, str, bool]:
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                is_correct = evaluator.judge_open_qa(
                    question=ex.question, answer=ex.answer, attempted=attempted,
                )
                return idx, ex, attempted, is_correct
            except RuntimeError as e:
                if not isinstance(e, GeminiBadFinishError) and type(e) is not RuntimeError:
                    raise
                last_exc = e
        extra = (f" finish_reason={last_exc.finish_reason}"
                 if isinstance(last_exc, GeminiBadFinishError) else "")
        print(
            f"[Gemini refused/failed{extra}] open-QA item {idx} default to incorrect\n"
            f"  Question:     {ex.question!r}\n"
            f"  Model answer: {attempted!r}\n"
            f"  Error:        {last_exc}"
        )
        return idx, ex, attempted, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as pool:
        futures = [pool.submit(grade, i, ex, answers[i])
                   for i, ex in enumerate(items)]
        for fut in concurrent.futures.as_completed(futures):
            idx, ex, attempted, is_correct = fut.result()
            if is_correct:
                correct += 1
            records[idx] = {
                "concept": ex.concept,
                "subset": ex.subset,
                "split": ex.split,
                "question": ex.question,
                "gold_answer": ex.answer,
                "model_answer": attempted,
                "is_correct": is_correct,
                "wrapper_lang": wrapper_lang,
            }
    return correct, records


# ========================================================================== #
# Metric fractions (copied verbatim from ember.evals.orchestrator)            #
# ========================================================================== #

def mc_frac(acc: float, base: float) -> float:
    """Chance-corrected MC retention, as in ``orchestrator._eval_mc_set``.

    ``base`` is the *pristine accuracy in the same language* -- never the
    English baseline for a non-English cell, which would conflate the model's
    cross-lingual competence with the erasure's efficacy.
    """
    if (base - 0.25) == 0:
        return 0.0
    return min(1.0, max((acc - 0.25) / (base - 0.25), 0.0))


def open_frac(acc: float, base: float) -> float:
    """Open-QA retention, as in ``orchestrator._eval_open_set`` (NaN if base == 0)."""
    return (min(1.0, acc / base) if base and base > 0 else float("nan"))


def qa_frac_for(qa_kind: str, acc: float, base: float) -> float:
    if qa_kind == "mc":
        return mc_frac(acc, base)
    if qa_kind == "open":
        return open_frac(acc, base)
    raise ValueError(f"qa_kind must be 'mc' or 'open', got {qa_kind!r}")


__all__ = [
    "evaluate_mc", "evaluate_open",
    "mc_frac", "open_frac", "qa_frac_for",
]
