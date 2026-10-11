"""MC prompt scaffold per language (native wrapper).

The ``en`` strings are exactly those of :func:`ember.evals.mc.build_mc_prompt`,
so English prompts are byte-identical to the HP search. The other four are
translations of the same three lines. Letters stay Latin (A-D) in every
language because :func:`ember.evals.mc.parse_letter` parses them back.
"""
from __future__ import annotations

from typing import Mapping

from ember.evals.schema import LETTER_CHOICES

SCAFFOLD: dict[str, dict[str, str]] = {
    "en": {
        "choose": "Choose the single best answer.",
        "question": "Question:",
        "choices": "Choices:",
        "answer_fmt": "Answer only with one letter (A, B, C, or D) and no formatting.",
    },
    "de": {
        "choose": "Wählen Sie die einzige beste Antwort.",
        "question": "Frage:",
        "choices": "Auswahlmöglichkeiten:",
        "answer_fmt": "Antworten Sie nur mit einem Buchstaben (A, B, C oder D) und ohne Formatierung.",
    },
    "es": {
        "choose": "Elige la única mejor respuesta.",
        "question": "Pregunta:",
        "choices": "Opciones:",
        "answer_fmt": "Responde solo con una letra (A, B, C o D) y sin ningún formato.",
    },
    "fr": {
        "choose": "Choisissez la seule meilleure réponse.",
        "question": "Question :",
        "choices": "Choix :",
        "answer_fmt": "Répondez uniquement par une lettre (A, B, C ou D), sans aucune mise en forme.",
    },
    "ru": {
        "choose": "Выберите единственный лучший ответ.",
        "question": "Вопрос:",
        "choices": "Варианты:",
        "answer_fmt": "Отвечайте только одной буквой (A, B, C или D), без какого-либо форматирования.",
    },
}


def build_mc_prompt(question: str, options: Mapping[str, str], lang: str) -> str:
    """Same layout as ``ember.evals.mc.build_mc_prompt``, scaffold in ``lang``."""
    s = SCAFFOLD[lang]
    lines = [s["choose"], f"{s['question']} {question}\n", s["choices"]]
    lines += [f"{L}. {options[L]}" for L in LETTER_CHOICES]
    lines.append(f"\n{s['answer_fmt']}")
    return "\n".join(lines)
