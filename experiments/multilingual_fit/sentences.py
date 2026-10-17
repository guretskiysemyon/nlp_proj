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

In the extracts one bibliography entry is one line ("Uwe Walter: Politische Ordnung
in der Römischen Republik. Oldenbourg, Berlin/Boston 2017, ISBN …"). The split
pieces without a marker ("Uwe Walter: Politische Ordnung in der Römischen
Republik.") would pass the sentence check, so reference_line() drops the whole
line when it is at most MAX_REF_LINE_WORDS long and has a clear marker (ISBN, doi,
"abgerufen am", "Дата обращения", "— М.:" …) or starts like a citation and has a
year (not one inside a prose date) / "In:" / "(Hrsg.)" / "//". Longer lines are prose paragraphs with an inline
citation; there only the sentence with the marker goes. Everything after the first
back-matter heading (BACK_MATTER: "Weblinks", "Примечания", "Referencias", "Notes et
références" …) is dropped too: external-link titles and the pieces of split references
the line rules cannot recognise. Pass removed=[] to
usable_sentences() to get every dropped sentence with its line and the rule.
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
  | \((?:англ|нем|фр|исп)\.\)(?!\)) | ^\((?:en|de|es|fr|it|ru)\)\s | \(Hrsg\.\) | ^In:\s   # not "(Putting green (англ.))"
  | (?:[Vv]erlag|University\s+Press|Press|Éditions|Editorial|Издательство|изд\.)[^.]{0,40}?\b(?:1[5-9]\d\d|20\d\d)\b
""", re.X)
# whole-line rules (reference_line); not "Auflage", bare "abgerufen" / "consulté": also prose
MAX_REF_LINE_WORDS = 65
_LINE_MARKER = re.compile(r"""
    \bISBN\b | \bISSN\b | \bdoi\b | \bOCLC\b | \bPMID\b | \bLCCN\b | https?:// | www\.
  | \babgerufen\ am\b | \bConsultado\ el\b | \bconsulté\ le\b | \bRetrieved\b | \bArchivado\ desde\b
  | \barchiviert\ vom\b | Дата\ обращения | Архивировано | —\s*(?:М|СПб|Л|Киев|Москва)\s?\.?\s?:
""", re.X | re.I)
_U, _W = "A-ZÀ-ÖØ-ÞА-ЯЁ", r"[\w'’.\-]"
_NAME = rf"(?:(?:von|van|de|der|du|la|le|di)\s+|[{_U}]{_W}*\s+){{1,4}}[{_U}][\w'’\-]+"
_CITE_START = re.compile(rf"""
    {_NAME}(?:,\s*(?:und\s+)?{_NAME})*(?:\s+(?:u\.\s?a\.|et\ al\.))?(?:\s*\((?:Hrsg|Hg)\.\))?:\s  # Name Name (Hrsg.): Title
  | [А-ЯЁ][а-яё\-]+\s+[А-ЯЁ]\.\s?(?:[А-ЯЁ]\.)?                                    # Фамилия И. О.
  | [{_U}][\w\-]+(?:\s[{_U}][\w\-]+)?,\s[{_U}][\w.]*(?:\s[{_U}][\w.]*)*(?:;|\s\()     # Surname, First (1999) / ;
  | \((?:en|de|es|fr|it|ru)(?:\s\+\s\w+)?\)\s | —{{2,}} | —\s\(                          # (en) …, —————— (1998)
""", re.X)
# Headings that only ever open an article's back matter (notes, references, links, see
# also); everything from the first one on is dropped. Not "Literatur" / "Литература" /
# "Siehe auch" / "См. также" / "Fuentes" / "Источники": also content sections in some
# articles (Greek or pornographic literature, a section on the ancient sources).
BACK_MATTER = {
    "de": {"Weblinks", "Einzelnachweise", "Anmerkungen", "Belege", "Fußnoten"},
    "ru": {"Примечания", "Ссылки", "Комментарии"},
    "es": {"Referencias", "Enlaces externos", "Bibliografía", "Notas", "Notas y referencias", "Véase también"},
    "fr": {"Notes et références", "Bibliographie", "Liens externes", "Articles connexes", "Voir aussi",
           "Notes", "Références"},
}
# "Literatur" / "Литература" is the bibliography only when it is the last such heading before
# the back matter and at least half of its lines are reference lines (a content section on
# Greek or pornographic literature has almost none); then the back matter starts there
# (de: "Literatur", then "Dokumentationen", "Filme", "Hörspiele", then "Weblinks").
BIBLIOGRAPHY = {"de": {"Literatur"}, "ru": {"Литература"}}
_CITE_SIGN = re.compile(r"\bIn:|\((?:Hrsg|Hg)\.\)|//")
_YEAR = re.compile(r"\b(?:1[5-9]\d\d|20\d\d)\b")
_MONTHS = ("Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|"
           "janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|"
           "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|"
           "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря")
_DATE_BEFORE = re.compile(rf"(?:{_MONTHS})\s+$", re.I)          # "am 22. Juni |1941": a date in prose
# "NZZ, 18. Dezember |2020", "Audible Studios, Dezember |2025" (fr extracts put U+200E after the comma)
_CITED_DATE = re.compile(rf",[\s‎]*(?:\d{{1,2}}\.?\s*)?(?:{_MONTHS})\s+$", re.I)


def _cite_sign(line: str) -> bool:
    if _CITE_SIGN.search(line):
        return True
    return any(not _DATE_BEFORE.search(line[:m.start()]) or _CITED_DATE.search(line[:m.start()])
               for m in _YEAR.finditer(line))


def reference_line(line: str):
    """The rule by which a whole line is a bibliography entry, or None."""
    line = line.strip()
    if len(line.split()) > MAX_REF_LINE_WORDS:
        return None
    if _LINE_MARKER.search(line):
        return "line: marker"
    # German nouns are capitalised, so "Zweite Phase: … am 22. Juni 1941." looks like
    # "Name Name: Title … 2018"; a year that is part of a prose date does not count
    if _CITE_START.match(line) and _cite_sign(line):
        return "line: citation start"
    return None


def back_matter_start(lines: list, lang: str):
    """(index, heading) of the first line of the article's back matter, or (None, None)."""
    def heading(i, names):
        return lines[i].strip() in names and (i == 0 or not lines[i - 1].strip())
    end = next((i for i in range(len(lines)) if heading(i, BACK_MATTER.get(lang, ()))), None)
    stop = len(lines) if end is None else end
    bib = [i for i in range(stop) if heading(i, BIBLIOGRAPHY.get(lang, ()))]
    if bib:
        body = [l for l in lines[bib[-1] + 1:stop] if len(l.split()) >= 4]
        if body and sum(reference_line(l) is not None for l in body) >= len(body) / 2:
            return bib[-1], lines[bib[-1]].strip()
    return (end, lines[end].strip()) if end is not None else (None, None)


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


def _strip_zero_width(text: str) -> str:
    # a space where the footnote stood between words ("expertos<ZW>en", "fin.<ZW>Según");
    # word joiners (U+2060, e.g. in IPA "[⁠i⁠]") are just dropped
    return _ZERO_WIDTH.sub(lambda m: " " if "​" in m.group() and m.end() < len(text)
                           and text[m.end()].isalnum() else "", text)


def split_sentences(text: str, lang: str) -> list:
    """Split one paragraph-structured plain-text article into sentences."""
    text = _strip_zero_width(text)
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


def usable_sentences(text: str, lang: str, removed: list = None) -> list:
    """Sentences kept for the concept / neutral sets: 6-60 words, ending in . ! ? » or ".

    Sentences of that shape dropped as references are appended to `removed` (if given)
    as {"sentence", "line", "rule"}."""
    keep, lines = [], _strip_zero_width(text).splitlines()
    start, head = back_matter_start(lines, lang)
    for i, line in enumerate(lines):
        line_rule = (f"section: {head}" if start is not None and i >= start else None) or reference_line(line)
        for s in split_sentences(line, lang):
            s = re.sub(r"\s+", " ", _CITATION.sub("", s)).strip()
            if not (6 <= len(s.split()) <= 60 and s[-1] in ".!?»\""):
                continue
            rule = line_rule or ("sentence: marker" if _REFERENCE.search(s) else None)
            if rule is None:
                keep.append(s)
            elif removed is not None:
                removed.append({"sentence": s, "line": line.strip(), "rule": rule})
    return keep
