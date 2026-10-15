"""Sentence splitting for Wikipedia plain text in en/ru/de/es/fr.

Replaces the splitter in fetch_wiki_sentences.py, which split after every
". " + capital, so German ordinal dates ("* 1. Juni 1954") and abbreviations
("род. 28 июля", "лат. errare") produced fragments. A period does NOT end a
sentence when the word before it is:
  - a 1-2 digit number, German only (ordinals: "1. Juni", "20. Jahrhundert"),
  - a single letter (initials: "A. Hitler", "И. В. Сталин"),
  - a Roman numeral, German only ("Pius XII. verurteilte"),
  - a known abbreviation of that language.
"!" and "?" always end a sentence.

Zero-width spaces (U+200B) are removed first: the es extracts keep one where each
footnote marker was ("santos.<U+200B> Algunas"), which hid the sentence boundary.

usable_sentences() also drops bibliography / reference lines (Literatur and
reference sections are part of the plain-text extracts): ISBN, page and volume
numbers, Russian imprints ("— М.: Наука, 1981. — 304 с."), access dates
("Consultado el …"), publisher + year.
"""
import re

ABBREV = {  # lower-case, without the final period
    "en": {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e", "no", "vol",
           "approx", "ca", "c", "u.s", "u.k"},
    "de": {"z", "b", "bzw", "ca", "vgl", "usw", "etc", "nr", "st", "dr", "prof", "geb", "gest", "jh",
           "jhd", "u", "a", "d", "h", "v", "chr", "n", "s", "abs", "mio", "mrd", "sog", "evtl", "inkl",
           "hrsg", "bd", "ggf", "insb", "z.b", "u.a", "d.h", "v.chr", "n.chr", "o", "ä", "dt", "engl",
           "lat", "griech", "franz", "span", "russ", "frz", "ital", "ehem", "max", "min", "mind", "gegr"},
    "ru": {"г", "гг", "т", "е", "д", "н", "э", "до", "род", "ок", "см", "им", "др", "пр", "т.е", "т.к",
           "т.п", "т.д", "с", "стр", "руб", "тыс", "млн", "млрд", "в", "вв", "ул", "проф", "акад", "св",
           "лат", "греч", "англ", "нем", "франц", "фр", "исп", "итал", "н.э", "до н.э", "р", "ум", "гл",
           "т.н", "ср", "напр", "англ", "кв", "км", "м", "чел", "ч"},
    "es": {"sr", "sra", "srta", "dr", "dra", "etc", "p", "pp", "núm", "n", "art", "aprox", "a", "c",
           "d", "j", "s", "av", "ej", "vol", "ed", "cap", "col", "sig", "ss", "a.c", "d.c", "lat", "gr",
           "ing", "fr", "al", "ca", "aprox", "op", "cit", "págs", "pág", "mill", "km"},
    "fr": {"m", "mme", "mlle", "mm", "dr", "st", "ste", "av", "apr", "j.-c", "c", "p", "pp", "etc",
           "env", "cf", "n°", "vol", "éd", "chap", "ex", "op", "cit", "lat", "gr", "angl", "all", "esp",
           "ca", "vers", "km", "hab", "mgr", "pr", "s"},
}
_BOUNDARY = re.compile(r"[.!?…]+[\"»”)\]]*(?=\s+[\"«„“(\[]?[A-ZÀ-ÖØ-ÞА-ЯЁ0-9])")
_PREV_WORD = re.compile(r"(\S+)$")
_ROMAN = re.compile(r"^[IVXLCDM]+$")
_ZERO_WIDTH = re.compile("[​⁠﻿]+")
_CITATION = re.compile(r"\[\d+\]|\[citation needed\]|\[править[^\]]*\]")
_REFERENCE = re.compile(r"""
    \bISBN\b | \bISSN\b | \bdoi\b | https?:// | www\.
  | \b(?:pp?|S|Vol|vol|Bd|Hrsg|éd|ed|Ed)\.\s?\d                 # p. 4, S. 10-15, Bd. 26
  | —\s*(?:[ТтСс]\.\s?\d|\d+\s?с\.)                           # — Т. 3, — С. 17, — 269 с.
  | \s—\s(?:М|СПб|Л|Киев)\s?\.?\s?:                             # — М.: (imprint)
  | \b(?:Consultado|Recuperado|Abgerufen|Consulté|Retrieved|Archived|Проверено|Архивировано)\b
  | \((?:англ|нем|фр|исп)\.\) | ^\((?:en|de|es|fr|it|ru)\)\s | \(Hrsg\.\) | ^In:\s
  | (?:[Vv]erlag|University\s+Press|Press|Éditions|Editorial|Издательство|изд\.)[^.]{0,40}?\b(?:1[5-9]\d\d|20\d\d)\b
""", re.X)


def _no_break(prev: str, lang: str) -> bool:
    word = prev.lstrip("([«„\"“'").rstrip(".")
    if lang == "de" and re.fullmatch(r"\d{1,2}", word):
        return True
    if re.fullmatch(r"[^\W\d_]", word):  # single letter: an initial
        return True
    if lang == "de" and _ROMAN.match(word) and len(word) <= 6:
        return True
    low = word.lower()
    abbr = ABBREV.get(lang, set())
    return low in abbr or low.split("-")[-1] in abbr or low.split(".")[-1] in abbr


def split_sentences(text: str, lang: str) -> list:
    """Split one paragraph-structured plain-text article into sentences."""
    # a space where the footnote stood between words ("expertos<ZW>en", "fin.<ZW>Según");
    # word joiners (U+2060, e.g. in IPA "[⁠i⁠]") are just dropped
    text = _ZERO_WIDTH.sub(lambda m: " " if "​" in m.group() and m.end() < len(text)
                           and text[m.end()].isalnum() else "", text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        start = 0
        for m in _BOUNDARY.finditer(line):
            punct = m.group(0)
            if punct.startswith(".") and not punct.startswith("..."):
                prev = _PREV_WORD.search(line[start:m.start()])
                if prev and _no_break(prev.group(1), lang):
                    continue
            out.append(line[start:m.end()].strip())
            start = m.end()
        if line[start:].strip():
            out.append(line[start:].strip())
    return out


def usable_sentences(text: str, lang: str) -> list:
    """Sentences kept for the concept / neutral sets: 6-60 words, ending in . ! ? » or "."""
    keep = []
    for s in split_sentences(text, lang):
        s = re.sub(r"\s+", " ", _CITATION.sub("", s)).strip()
        if 6 <= len(s.split()) <= 60 and s[-1] in ".!?»\"" and not _REFERENCE.search(s):
            keep.append(s)
    return keep
