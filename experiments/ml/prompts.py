"""MC prompt scaffold for the 8 Llama-3.1 languages (native wrapper).

en / de / es / fr are taken verbatim from :mod:`crosslingual.prompts` (so English is
byte-identical to ``ember.evals.mc.build_mc_prompt``). it / pt / hi / th are translations
of the same four lines. Letters stay Latin (A-D) in every language because the letter
parsers read them back.
"""
from __future__ import annotations

from typing import Mapping

from crosslingual.prompts import SCAFFOLD as _XL_SCAFFOLD
from ember.evals.schema import LETTER_CHOICES

LANGS = ("en", "de", "fr", "it", "pt", "hi", "es", "th")

SCAFFOLD: dict[str, dict[str, str]] = {
    **{l: dict(_XL_SCAFFOLD[l]) for l in ("en", "de", "es", "fr")},
    "it": {
        "choose": "Scegli l'unica risposta migliore.",
        "question": "Domanda:",
        "choices": "Opzioni:",
        "answer_fmt": "Rispondi solo con una lettera (A, B, C o D) e senza alcuna formattazione.",
    },
    "pt": {
        "choose": "Escolha a única melhor resposta.",
        "question": "Pergunta:",
        "choices": "Opções:",
        "answer_fmt": "Responda apenas com uma letra (A, B, C ou D) e sem qualquer formatação.",
    },
    "hi": {
        "choose": "केवल एक सबसे सही उत्तर चुनें।",
        "question": "प्रश्न:",
        "choices": "विकल्प:",
        "answer_fmt": "केवल एक अक्षर (A, B, C या D) में उत्तर दें, बिना किसी फ़ॉर्मैटिंग के।",
    },
    "th": {
        "choose": "เลือกคำตอบที่ดีที่สุดเพียงข้อเดียว",
        "question": "คำถาม:",
        "choices": "ตัวเลือก:",
        "answer_fmt": "ตอบด้วยตัวอักษรเพียงตัวเดียว (A, B, C หรือ D) เท่านั้น โดยไม่ต้องจัดรูปแบบใดๆ",
    },
}
assert set(SCAFFOLD) == set(LANGS)


def build_mc_prompt(question: str, options: Mapping[str, str], lang: str) -> str:
    """Same layout as ``ember.evals.mc.build_mc_prompt``, scaffold in ``lang``."""
    s = SCAFFOLD[lang]
    lines = [s["choose"], f"{s['question']} {question}\n", s["choices"]]
    lines += [f"{L}. {options[L]}" for L in LETTER_CHOICES]
    lines.append(f"\n{s['answer_fmt']}")
    return "\n".join(lines)
