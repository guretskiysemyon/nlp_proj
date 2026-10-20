"""Shared constants and helpers for the multilingual sentence database (results/ml/data/).

Run every script of this package as a module from the repo root, e.g.
    python -m experiments.ml.data.fetch_pages
never as a file path: this directory holds io.py, which would shadow the stdlib ``io`` if the directory
itself were put on sys.path.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "ml" / "data"
RAW = OUT / "raw"            # raw/<slug>/<lang>.json  (parse HTML of the concept page)
POOLS = OUT / "pools"        # pools/<slug>/<lang>.json (split sentences of the concept page)
DUMP = OUT / "dump"          # extra wikimedia/wikipedia dump samples (beyond multilingual_hp's)
TRANS = OUT / "translations"  # translations/tasks.json, translations/nllb_<lang>.jsonl
NEUTRAL = OUT / "neutral"    # neutral/partition_<lang>.json
FULL = OUT / "full"
SAMPLE = OUT / "sample"
PAGES_PATH = OUT / "pages.json"
BACKGROUND_PATH = OUT / "background_ml.json"

LANGS = ("en", "de", "fr", "it", "pt", "hi", "es", "th")
SEED = 0
N_FULL = 300
SAMPLE_NS = (20, 30, 40, 50)
N_HELDOUT = 100
N_BACKGROUND = 5000

# pythainlp is not in the project venv; it was installed with --no-deps into this private directory.
_PYLIB = Path("/home/morg/NLP_2526b/oripetel/.cache/pylib_ml")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.append(str(_PYLIB))
for _p in (REPO / "experiments" / "multilingual_hp", REPO / "experiments" / "resample_sentences",
           REPO / "experiments" / "multilingual_fit"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

# English Wikipedia title per concept (as chosen by multilingual_fit/fetch_wiki_articles.py: best overlap with
# EMBER's English sentences).
EN_TITLES = {
    "Ancient Rome": "Ancient Rome", "Artificial intelligence": "Artificial intelligence", "Baseball": "Baseball",
    "COVID-19 pandemic": "COVID-19 pandemic", "Cannabis": "Cannabis (drug)", "Culture of Greece": "Culture of Greece",
    "Gambling": "Gambling", "Golf": "Golf", "Gun": "Gun", "Halloween": "Halloween", "Harry Potter": "Harry Potter",
    "Heroin": "Heroin", "Nazism": "Nazism", "Pornography": "Pornography", "Republic of Ireland": "Republic of Ireland",
    "Uranium": "Uranium", "Valentine's Day": "Valentine's Day", "World War II": "World War II",
}

# Obvious concept terms (lower-cased stems) in addition to the content words of the page titles; a neutral sentence
# containing any of them (stem at a word start; Thai/Hindi: substring) is not used as a neutral for that concept.
EXTRA_TERMS = {
    "Ancient Rome": ["roma", "rome", "römisch", "romain", "romano", "romana", "roman empire", "रोम", "โรม"],
    "Artificial intelligence": ["artificial intelligence", "künstliche intelligenz", "intelligence artificielle",
                                "intelligenza artificiale", "inteligência artificial", "inteligencia artificial",
                                "कृत्रिम बुद्धि", "ปัญญาประดิษฐ์", "machine learning"],
    "Baseball": ["baseball", "béisbol", "beisebol", "beisbol", "बेसबॉल", "เบสบอล"],
    "COVID-19 pandemic": ["covid", "coronavirus", "sars-cov", "कोविड", "कोरोना", "โควิด", "โคโรนา"],
    "Cannabis": ["cannabis", "marijuana", "marihuana", "maconha", "haschisch", "hashish", "hachís",
                 "भांग", "गांजा", "กัญชา"],
    "Culture of Greece": ["greece", "greek", "griech", "grèce", "grec", "grecia", "grego", "grécia", "griego",
                          "greco", "hellen", "ग्रीस", "यूनान", "กรีก"],
    "Gambling": ["gambl", "glücksspiel", "casino", "kasino", "cassino", "lotter", "lotto", "jeu d'argent",
                 "azzardo", "apuesta", "aposta", "जुआ", "การพนัน", "คาสิโน"],
    "Golf": ["golf", "गोल्फ़", "गोल्फ", "กอล์ฟ"],
    "Gun": ["firearm", "feuerwaffe", "arme à feu", "arma da fuoco", "arma de fogo", "arma de fuego", "pistol",
            "rifle", "gewehr", "fusil", "fucile", "revolver", "बंदूक", "आग्नेयास्त्र", "ปืน"],
    "Halloween": ["halloween", "हैलोवीन", "ฮาโลวีน", "ฮัลโลวีน"],
    "Harry Potter": ["potter", "hogwarts", "rowling", "हैरी पॉटर", "แฮร์รี่ พอตเตอร์", "พอตเตอร์"],
    "Heroin": ["heroin", "héroïne", "eroina", "heroína", "opioid", "opiat", "opiac", "हेरोइन", "เฮโรอีน"],
    "Nazism": ["nazi", "hitler", "nationalsozialis", "national socialis", "national-socialis", "nazismo", "nazisme",
               "नाज़ी", "नाजी", "हिटलर", "นาซี", "ฮิตเลอร์"],
    "Pornography": ["porn", "pornogra", "पोर्न", "อนาจาร", "ลามก", "โป๊"],
    "Republic of Ireland": ["ireland", "irish", "irland", "irlande", "irlandais", "irlanda", "irlandese", "irlandês",
                            "irlandés", "dublin", "आयरलैंड", "ไอร์แลนด์"],
    "Uranium": ["uran", "urânio", "यूरेनियम", "ยูเรเนียม"],
    "Valentine's Day": ["valentin", "valentine", "san valentino", "वैलेंटाइन", "วาเลนไทน์"],
    "World War II": ["world war", "weltkrieg", "guerre mondiale", "guerra mondiale", "guerra mundial", "wehrmacht",
                     "विश्व युद्ध", "विश्वयुद्ध", "สงครามโลก"],
}
NOSPACE = {"th"}            # substring matching (no word boundaries)
_TITLE_STOP = {"of", "the", "de", "la", "le", "les", "del", "da", "do", "di", "du", "des", "dos", "e", "et", "und",
               "y", "als", "and", "del", "della", "der", "die", "das", "il", "lo", "usage", "pays", "drug", "sport",
               "elemento", "élément", "element", "(sport)", "cultura", "culture", "kultur", "día", "dia", "jeu",
               "seconda", "segunda", "seconde", "zweiter", "repubblica", "república", "république", "republik"}


def slug(concept: str) -> str:
    return re.sub(r"\W+", "_", concept).strip("_")


def concepts():
    """The concept set chosen by $ML_CONCEPTS: "orig" (default, EMBER's 18, data/mc_questions.json), "new" (the 20
    of data/concept_sentences_extra.json, PREREG_NEW.md) or "all" (orig then new)."""
    import os
    orig = list(json.loads((REPO / "data" / "mc_questions.json").read_text(encoding="utf-8")))
    new = [e["concept"] for e in json.loads((REPO / "data" / "concept_sentences_extra.json").read_text(encoding="utf-8"))]
    return {"orig": orig, "new": new, "all": orig + new}[os.environ.get("ML_CONCEPTS", "orig")]


def dump_json(path: Path, obj, indent=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def word_count(text: str, lang: str) -> int:
    if lang == "th":
        from pythainlp.tokenize import word_tokenize
        return sum(1 for w in word_tokenize(text, engine="newmm") if w.strip())
    return len(text.split())


def concept_terms(concept: str, titles: dict) -> list:
    """Lower-cased exclusion terms for a concept: EXTRA_TERMS + content words of its titles in all languages."""
    terms = set(EXTRA_TERMS.get(concept, []))
    for lang, t in titles.items():
        if not t:
            continue
        t = re.sub(r"\(.*?\)", " ", t).strip().lower()
        if lang in NOSPACE or lang == "hi":
            terms.add(t)
        for w in re.split(r"[\s\-–]+", t):
            w = w.strip("'’,.")
            if len(w) >= 4 and w not in _TITLE_STOP:
                terms.add(w)
    return sorted(terms)


def term_matcher(terms):
    """Function sentence -> bool (True if any term occurs). Latin/Devanagari: term at a word start; Thai: substring."""
    spaced = [t for t in terms if not re.search(r"[฀-๿]", t)]
    thai = [t for t in terms if re.search(r"[฀-๿]", t)]
    pat = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t.strip()) for t in spaced if t.strip()) + ")", re.I) \
        if spaced else None

    def has(s: str) -> bool:
        if pat is not None and pat.search(s):
            return True
        return any(t in s for t in thai)
    return has
