#!/usr/bin/env python3
"""Native neutral sentences: N_PER_LANG sentences per language, each from a different random Wikipedia
article written in that language (no translation, no Wikipedia API).

Pages come from the Hugging Face dump wikimedia/wikipedia, config 20231101.<lang>, sampled with
sample_wiki_dump.sample_dump (HTTP range reads of a few parquet row groups):
    en      reuses EN_DUMP_PAGES (the 1k-page English sample, no download)
    others  DUMP/wiki_dump_pages_<lang>.jsonl (40 row groups x 10 pages, seed SEED), reused unless --force

Selection per language (random.Random(SEED)): shuffle the pages, skip pages with < MIN_PAGE_CHARS of
text, keep prose lines (ending in sentence punctuation; for Thai, which has none, long non-heading
lines containing Thai script), split with splitting.split_sentences, drop bibliography/list-like
sentences and (hi/th) sentences mostly not in the native script, drop sentences whose template
signature (first 3 + last 3 words, digits -> 0) was already used in this language, and draw one
sentence per page until N_PER_LANG.

Output: NEUTRAL_NATIVE_PATH, a list of {"sentence", "lang", "url", "title", "source": "native"} in
LANGS order (loads with ember.local_datasets.ConceptDataset(neutral_path=...)).

Usage:
    python experiments/multilingual_hp/native_neutral.py [--force]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import regex

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "resample_sentences"))

from splitting import (DUMP, EN_DUMP_PAGES, LANGS, N_PER_LANG, NEUTRAL_NATIVE_PATH, PROSE_END, SEED,  # noqa: E402
                       split_sentences)
from sample_wiki_dump import sample_dump  # noqa: E402

DUMP_N_GROUPS, DUMP_PER_GROUP = 40, 10  # 3 per group left pt short of 40 pages >= MIN_PAGE_CHARS
MIN_PAGE_CHARS = 1500
MIN_TH_LINE_CHARS = 80  # shorter Thai lines are mostly list items / captions (Thai has no sentence punctuation)
THAI_CHAR = re.compile(r"[\u0e00-\u0e7f]")
NATIVE_SCRIPT = {"hi": re.compile(r"[\u0900-\u097f]"), "th": THAI_CHAR}
MIN_NATIVE_SCRIPT_FRAC = 0.6  # of letters; drops English bibliography lines on hi/th pages
# Bibliography / reference-list lines that still end in "." ("Bd. 22 = ...", "... 1999, ISBN ...", "pp. 3-7").
CITATION_LIKE = re.compile(r"\bISBN\b|\bISSN\b|\bdoi:|https?://|\b(?:Bd|Vol|vol|pp?|S|Hrsg|ed|eds)\.\s*\d"
                           r"|\(\s*(?:Hrsg|ed|eds)\.\s*\)")
NUMBERED_LIST = re.compile(r"(?:^|\s)[0-9\u0966-\u096f]+\.\s")
CITATION_TAIL = re.compile(r",\s*(?:\d{1,2}\.?\s+\w+\s+)?\d{4}\.$")  # "Title, Publisher, 30 March 2018."
LABEL_PREFIX = re.compile(r"^[^\s:]+(?:\s+[^\s:]+){0,2}:\s")  # "Last updated: ...", "Valle: ...", "2021-01-01: ..."
# A template stripped from the dump leaves a hole: "né le à Paris", "Au mois de , le", "diffusé le .".
STRIPPED_TEMPLATE = re.compile(r"\s[,.](?:\s|$)|\b(\w+) \1\b|\(\s*[–-]?\s*\)|[–-]\s*\)|\bà \)|\ble à\b"
                               r"|\b(?:des|de|du) (?:est|dont|et)\b")
LOWER_AFTER_STOP = regex.compile(r"\p{L}{3,}[.!?] \p{Ll}")
TIMELINE = re.compile(r"^\d{4}(?:\s*[–-]\s*\d{4})?\s+[–-]\s")  # "2002 – Poesía ...", "1976–1977 – ..."
AUTHOR_TITLE = re.compile(r"^[^:]*,[^:]*:\s")  # "Marianne Eigenheer, Hans Ulrich Obrist: Gespräch ..."
# Text after these headings is references / links / lists, not prose (matched against whole lines, casefolded).
END_SECTIONS = {
    "references", "external links", "further reading", "see also", "notes", "bibliography", "sources",
    "referencias", "enlaces externos", "bibliografía", "véase también", "notas",
    "références", "notes et références", "liens externes", "bibliographie", "voir aussi", "articles connexes",
    "einzelnachweise", "weblinks", "literatur", "siehe auch", "anmerkungen", "quellen",
    "note", "bibliografia", "collegamenti esterni", "voci correlate", "altri progetti",
    "referências", "ligações externas", "ver também", "notas e referências", "bibliografia",
    "सन्दर्भ", "संदर्भ", "बाहरी कड़ियाँ", "इन्हें भी देखें", "टिप्पणी", "सन्दर्भ ग्रन्थ",
    "อ้างอิง", "แหล่งข้อมูลอื่น", "ดูเพิ่ม", "เชิงอรรถ", "บรรณานุกรม",
}


def dump_path(lang: str) -> Path:
    return EN_DUMP_PAGES if lang == "en" else DUMP / f"wiki_dump_pages_{lang}.jsonl"


def load_pages(lang: str, force: bool = False):
    path = dump_path(lang)
    if lang != "en" and (force or not path.exists()):
        sample_dump(f"20231101.{lang}", path, n_groups=DUMP_N_GROUPS, per_group=DUMP_PER_GROUP, seed=SEED)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prose_paragraphs(text: str, lang: str):
    lines = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln.casefold().rstrip(":") in END_SECTIONS:
            break
        lines.append(ln)
    if lang == "th":
        return [ln for ln in lines if len(ln) >= MIN_TH_LINE_CHARS and THAI_CHAR.search(ln)]
    return [ln for ln in lines if ln and PROSE_END.search(ln)]


def looks_like_prose(sentence: str, lang: str) -> bool:
    if CITATION_LIKE.search(sentence) or len(NUMBERED_LIST.findall(sentence)) >= 2:
        return False
    if (CITATION_TAIL.search(sentence) or LABEL_PREFIX.match(sentence) or STRIPPED_TEMPLATE.search(sentence)
            or TIMELINE.match(sentence) or AUTHOR_TITLE.match(sentence)):
        return False
    first = sentence[0]
    if not (first.isalnum() or first in "\"'“«„(") or (lang not in NATIVE_SCRIPT and first.islower()):
        return False  # fragment ("§ 940.15\", enacted ...", "em produções ...")
    if lang not in NATIVE_SCRIPT and LOWER_AFTER_STOP.search(sentence):
        return False  # stripped-template hole mid-sentence ("... en Irak. trouvèrent refuge ...")
    if lang in NATIVE_SCRIPT:
        letters = [c for c in sentence if c.isalpha()]
        native = sum(1 for c in letters if NATIVE_SCRIPT[lang].match(c))
        if not letters or native / len(letters) < MIN_NATIVE_SCRIPT_FRAC:
            return False
    if lang == "th" and sentence.count(",") >= 3:  # Thai prose rarely uses commas; these are name/spec lists
        return False
    return True


def signature(sentence: str):
    words = re.sub(r"\d", "0", sentence).split()
    return tuple(words[:3]), tuple(words[-3:])


def pick_lang(lang: str, pages, stats=None):
    rng = random.Random(SEED)
    pages = list(pages)
    rng.shuffle(pages)
    used_sigs, used_pages, out = set(), set(), []
    stats = stats if stats is not None else {}
    stats.update(short=0, no_sentence=0)
    for page in pages:
        if len(out) >= N_PER_LANG:
            break
        if page["url"] in used_pages:
            continue
        text = page.get("text") or ""
        if len(text) < MIN_PAGE_CHARS:
            stats["short"] += 1
            continue
        sents = [s for s in split_sentences(prose_paragraphs(text, lang), lang)
                 if looks_like_prose(s, lang) and signature(s) not in used_sigs]
        if not sents:
            stats["no_sentence"] += 1
            continue
        s = rng.choice(sents)
        used_sigs.add(signature(s))
        used_pages.add(page["url"])
        out.append({"sentence": s, "lang": lang, "url": page["url"], "title": page["title"], "source": "native"})
    if len(out) < N_PER_LANG:
        raise RuntimeError(f"[{lang}] only {len(out)}/{N_PER_LANG} sentences from {len(pages)} pages ({stats})")
    return out


def run(force: bool = False):
    DUMP.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang in LANGS:
        pages = load_pages(lang, force)
        stats = {}
        picked = pick_lang(lang, pages, stats)
        print(f"[{lang}] {len(pages)} pages, skipped {stats['short']} short (<{MIN_PAGE_CHARS} chars) and "
              f"{stats['no_sentence']} without usable prose -> {len(picked)} sentences", flush=True)
        rows += picked
    NEUTRAL_NATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEUTRAL_NATIVE_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {len(rows)} sentences -> {NEUTRAL_NATIVE_PATH}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download the non-English dump samples")
    run(ap.parse_args().force)
