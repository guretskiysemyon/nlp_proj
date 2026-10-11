"""Wrapper-language templates + prompt builders.

:func:`ember.evals.mc.build_mc_prompt` and the ``f"Question: {q}\\nAnswer:"``
line inside :func:`ember.evals.open_qa.evaluate_open_qa` hardcode English
scaffold text and are called from inside the generation loops, so they cannot
be parameterized without editing ``ember/``. This module re-implements only the
prompt *string assembly*, adding a ``wrapper_lang`` argument.

With ``wrapper_lang="en"`` the output is byte-identical to the English
pipeline's prompt, which is what makes the two ablation arms comparable:

- ``wrapper_lang == language``  -> native-wrapper condition.
- ``wrapper_lang == "en"`` with ``language != "en"`` -> English-wrapper (mixed)
  condition: translated question/options inside an English instruction frame.

The letter labels themselves (A/B/C/D) stay Latin in every language, because
the options are lettered with :data:`ember.evals.schema.LETTER_CHOICES` and
parsed back with :func:`ember.evals.mc.parse_letter`, neither of which is
translated.

The four non-English string sets are human-readable translations of the five
English scaffold lines and should be reviewed by a speaker of each language
before being treated as final (see §12.5 of the spec).
"""
from __future__ import annotations

from typing import Dict, Mapping

from ember.evals.schema import LETTER_CHOICES

WRAPPER_LANGS: tuple[str, ...] = ("en", "de", "es", "fr", "ru")

#: The five scaffold strings per wrapper language. ``en`` must stay exactly as
#: spelled in ``ember/evals/mc.py`` and ``ember/evals/open_qa.py``.
WRAPPER_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "choose": "Choose the single best answer.",
        "choices": "Choices:",
        "answer_fmt": "Answer only with one letter (A, B, C, or D) and no formatting.",
        "question_label": "Question:",
        "answer_label": "Answer:",
    },
    "de": {
        "choose": "Wählen Sie die einzige beste Antwort.",
        "choices": "Auswahlmöglichkeiten:",
        "answer_fmt": "Antworten Sie nur mit einem Buchstaben (A, B, C oder D) und ohne Formatierung.",
        "question_label": "Frage:",
        "answer_label": "Antwort:",
    },
    "es": {
        "choose": "Elige la única mejor respuesta.",
        "choices": "Opciones:",
        "answer_fmt": "Responde solo con una letra (A, B, C o D) y sin ningún formato.",
        "question_label": "Pregunta:",
        "answer_label": "Respuesta:",
    },
    "fr": {
        "choose": "Choisissez la seule meilleure réponse.",
        "choices": "Choix :",
        "answer_fmt": "Répondez uniquement par une lettre (A, B, C ou D), sans aucune mise en forme.",
        "question_label": "Question :",
        "answer_label": "Réponse :",
    },
    "ru": {
        "choose": "Выберите единственный лучший ответ.",
        "choices": "Варианты:",
        "answer_fmt": "Отвечайте только одной буквой (A, B, C или D), без какого-либо форматирования.",
        "question_label": "Вопрос:",
        "answer_label": "Ответ:",
    },
}

_REQUIRED_KEYS = ("choose", "choices", "answer_fmt", "question_label", "answer_label")


def check_wrapper_lang(wrapper_lang: str) -> str:
    if wrapper_lang not in WRAPPER_STRINGS:
        raise ValueError(f"Unknown wrapper_lang {wrapper_lang!r}; "
                         f"expected one of {tuple(WRAPPER_STRINGS)}")
    missing = [k for k in _REQUIRED_KEYS if not WRAPPER_STRINGS[wrapper_lang].get(k)]
    if missing:
        raise ValueError(f"WRAPPER_STRINGS[{wrapper_lang!r}] missing {missing}")
    return wrapper_lang


def wrapper_lang_for(language: str, wrapper: str) -> str:
    """Map a (language, wrapper) cell to the wrapper language to render in.

    ``wrapper="english"`` always renders the English scaffold; ``wrapper="native"``
    renders the scaffold in the content language. For ``language == "en"`` both
    collapse to ``"en"``.
    """
    if wrapper == "english":
        return "en"
    if wrapper == "native":
        return check_wrapper_lang(language)
    raise ValueError(f"wrapper must be 'english' or 'native', got {wrapper!r}")


# ========================================================================== #
# Prompt builders                                                             #
# ========================================================================== #

def build_mc_prompt(question: str, options_by_letter: Mapping[str, str],
                    wrapper_lang: str = "en") -> str:
    """Format an MC question + 4 options as a single prompt string.

    Identical to :func:`ember.evals.mc.build_mc_prompt` when
    ``wrapper_lang == "en"``, including the embedded blank lines.
    """
    s = WRAPPER_STRINGS[check_wrapper_lang(wrapper_lang)]
    lines = [
        s["choose"],
        f"{s['question_label']} {question}\n",
        s["choices"],
    ]
    for L in LETTER_CHOICES:
        lines.append(f"{L}. {options_by_letter[L]}")
    lines.append(f"\n{s['answer_fmt']}")
    return "\n".join(lines)


def build_open_prompt(question: str, wrapper_lang: str = "en") -> str:
    """Format an open-ended question as a single prompt string.

    Identical to the ``f"Question: {q}\\nAnswer:"`` line in
    :func:`ember.evals.open_qa.evaluate_open_qa` when ``wrapper_lang == "en"``.
    """
    s = WRAPPER_STRINGS[check_wrapper_lang(wrapper_lang)]
    return f"{s['question_label']} {question}\n{s['answer_label']}"


__all__ = [
    "WRAPPER_LANGS", "WRAPPER_STRINGS",
    "check_wrapper_lang", "wrapper_lang_for",
    "build_mc_prompt", "build_open_prompt",
]
