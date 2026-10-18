"""Shared language list, paths and per-language sentence splitting for the multilingual Harry Potter data.

Every stage (build_sentences.py, translate_neutral.py, native_neutral.py) imports from here so all
sentence sets are split the same way.

    split_sentences(paragraphs, lang) -> list of sentences, deduplicated, order kept

Spaced languages (en, es, fr, de, it, pt, hi): split after . ! ? । ॥ (plus closing quotes/brackets) and
whitespace when the next character is not lowercase; keep sentences with >= MIN_WORDS words.
Thai has no sentence punctuation: pythainlp's crfcut segments are merged greedily until a chunk has
>= MIN_WORDS_TH words (crfcut alone splits on initials such as "เจ. เค." and on every clause space).
"""
import re as _re
from pathlib import Path

import regex

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DATA = HERE / "data"
RAW, POOLS, TRANSLATIONS, DUMP = DATA / "raw", DATA / "pools", DATA / "translations", DATA / "dump"
CONCEPT_PATH = DATA / "concept_sentences.json"
NEUTRAL_TRANSLATED_PATH = DATA / "neutral_sentences_translated.json"
NEUTRAL_NATIVE_PATH = DATA / "neutral_sentences_native.json"
EN_DUMP_PAGES = ROOT / "experiments" / "resample_sentences" / "data" / "wiki_dump_pages.jsonl"

CONCEPT = "Harry Potter"
SEED = 42

# English Wikipedia titles to try, in order, per concept (the first with interlanguage links wins)
CONCEPT_TITLES = {
    "Harry Potter": ["Harry Potter"],
    "Gun": ["Gun", "Firearm"],
    "Halloween": ["Halloween"],
    "Pornography": ["Pornography"],
    "Valentine's Day": ["Valentine's Day"],
    "Gambling": ["Gambling"],
}


def concept_slug(concept: str) -> str:
    return _re.sub(r"\W+", "_", concept).strip("_")


def concept_paths(concept: str = CONCEPT) -> dict:
    """Data paths of one concept. Harry Potter keeps the original root layout; others live in data/concepts/<slug>/."""
    base = DATA if concept == CONCEPT else DATA / "concepts" / concept_slug(concept)
    return {"dir": base, "raw": base / "raw", "pools": base / "pools",
            "concept": base / "concept_sentences.json", "extended": base / "extended" if concept != CONCEPT else DATA / "extended"}
N_PER_LANG = 40

# code -> (English name, script name for the translation prompt)
LANGS = {
    "en": ("English", "Latin"),
    "es": ("Spanish", "Latin"),
    "fr": ("French", "Latin"),
    "de": ("German", "Latin"),
    "it": ("Italian", "Latin"),
    "pt": ("Portuguese", "Latin"),
    "hi": ("Hindi", "Devanagari"),
    "th": ("Thai", "Thai"),
}

MIN_WORDS = 5
MIN_WORDS_TH = 12

SENT_SPLIT = regex.compile(r"(?<=[.!?।॥][\"'”’»)\]]*)\s+(?=[^\p{Ll}])")
CITATION = _re.compile(r"\s*(?:\[\d+\]|\[[a-z]\]|\[note \d+\]|\[citation needed\]|\[[^\]]{1,30}\])")
BOILERPLATE = _re.compile(r"You can help Wikipedia|This article .* is a stub|is a stub\.", _re.I)
INVISIBLE = _re.compile("[\u200b\u200e\u200f\u2060\ufeff\u00ad]")  # eswiki puts ZWSP after refs
ABBREV_END = regex.compile(r"(?:^|\s)\S{1,3}\.$")
PROSE_END = regex.compile(r"[.!?।॥][\"'”’»)]*$")


def _split_th(text: str):
    from pythainlp.tokenize import sent_tokenize, word_tokenize

    out, buf = [], ""
    for seg in sent_tokenize(text, engine="crfcut"):
        buf += seg
        if buf.rstrip().endswith("."):  # Thai has no sentence periods: "ค.ศ." / "พ.ศ." must stay with the year after it
            continue
        if len([w for w in word_tokenize(buf) if w.strip()]) >= MIN_WORDS_TH:
            out.append(buf.strip())
            buf = ""
    if buf.strip() and len([w for w in word_tokenize(buf) if w.strip()]) >= MIN_WORDS_TH:
        out.append(buf.strip())
    return out


def split_sentences(paragraphs, lang: str):
    out = []
    for para in paragraphs:
        para = _re.sub(r"\s+", " ", CITATION.sub("", INVISIBLE.sub("", para))).strip()
        if not para:
            continue
        if lang == "th":
            out += [s for s in _split_th(para) if not BOILERPLATE.search(s)]
            continue
        pieces = []
        for piece in SENT_SPLIT.split(para):  # glue back splits after initials/abbreviations ("J. K.", "जे. के.", "Dr.")
            if pieces and ABBREV_END.search(pieces[-1]):
                pieces[-1] += " " + piece
            else:
                pieces.append(piece)
        for s in pieces:
            s = s.strip()
            if len(s.split()) >= MIN_WORDS and not BOILERPLATE.search(s):
                out.append(s)
    return list(dict.fromkeys(out))


def neutral_paths(concept: str = CONCEPT, rep: int = 0) -> dict:
    """Translated neutral files for one concept and replicate.

    base: the shared 320-sentence set (40 per language); rep 0 is the original file.
    ext:  that concept's extended neutral set (per language as many as its extended concept sentences).
    """
    base = NEUTRAL_TRANSLATED_PATH if rep == 0 else DATA / f"neutral_rep{rep}" / "neutral_sentences_translated.json"
    return {"base": base, "ext": concept_paths(concept)["extended"] / f"neutral_ext_translated_rep{rep}.json"}
