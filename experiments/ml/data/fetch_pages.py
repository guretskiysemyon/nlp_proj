"""Concept pages for the concepts ($ML_CONCEPTS, common.concepts) x 8 languages (Wikipedia API, cached).

    fetch   one langlinks call per concept on its English page (common.EN_TITLES), one parse call per language
            -> raw/<slug>/<lang>.json  {"concept","lang","title","revid","url","html"}  (never re-fetched unless
            --force); a missing langlink is recorded in raw/<slug>/_langlinks.json
    split   raw HTML -> top-level <p> paragraphs -> multilingual_hp.splitting.split_sentences -> usable filter
            -> pools/<slug>/<lang>.json, and pages.json (title, revid, n_words, n_sentences, exists per page; merged:
            other concepts' entries are kept)

Usable filter: <= MAX_CHARS characters; hi/th: >= 60% of letters in the native script (drops English citations).
n_words counts the words of all <p> paragraphs (whitespace words; Thai: pythainlp newmm words).

    python -m experiments.ml.data.fetch_pages [--force]
"""
from __future__ import annotations

import argparse
import re
import urllib.parse

from experiments.ml.data.common import (EN_TITLES, LANGS, PAGES_PATH, POOLS, RAW, concepts, dump_json, load_json,
                                        slug, word_count)
from fetch_en_sentences import _ParagraphText  # noqa: E402  (experiments/resample_sentences)
from fetch_wiki_sentences import api  # noqa: E402            (experiments/multilingual_fit)
from splitting import split_sentences  # noqa: E402           (experiments/multilingual_hp)

MAX_CHARS = 1000
# English pages whose langlinks fill languages the main English page does not link (Gun has almost none)
ALT_EN_TITLES = {"Gun": ["Firearm"]}
NATIVE_SCRIPT = {"hi": re.compile(r"[ऀ-ॿ]"), "th": re.compile(r"[฀-๿]")}


def usable(s: str, lang: str) -> bool:
    if len(s) > MAX_CHARS:
        return False
    if lang in NATIVE_SCRIPT:
        letters = [c for c in s if c.isalpha()]
        if not letters or sum(1 for c in letters if NATIVE_SCRIPT[lang].match(c)) / len(letters) < 0.6:
            return False
    return True


def fetch(force: bool):
    for concept in concepts():
        d = RAW / slug(concept)
        todo = [l for l in LANGS if force or not (d / f"{l}.json").exists()]
        if not todo:
            continue
        ll_path = d / "_langlinks.json"
        if ll_path.exists() and not force:
            titles = load_json(ll_path)
        else:
            q = api("en", action="query", titles=EN_TITLES[concept], prop="langlinks", lllimit="max", redirects=1)
            page = q["query"]["pages"][0]
            titles = {"en": page["title"]}
            titles.update({x["lang"]: x["title"] for x in page.get("langlinks", [])})
            titles = {l: titles.get(l) for l in LANGS}
            for alt in ALT_EN_TITLES.get(concept, []):  # languages the English page does not link: a broader page
                if all(titles.values()):
                    break
                q = api("en", action="query", titles=alt, prop="langlinks", lllimit="max", redirects=1)
                links = {x["lang"]: x["title"] for x in q["query"]["pages"][0].get("langlinks", [])}
                for l in LANGS:
                    if not titles[l] and links.get(l):
                        titles[l] = links[l]
                        titles.setdefault("_via", {})[l] = alt
            dump_json(ll_path, titles)
        for lang in todo:
            if not titles.get(lang) or not isinstance(titles[lang], str):
                print(f"[{concept}] no {lang} langlink", flush=True)
                continue
            p = api(lang, action="parse", page=titles[lang], prop="text|revid", redirects=1)["parse"]
            dump_json(d / f"{lang}.json", {
                "concept": concept, "lang": lang, "title": p["title"], "revid": p["revid"],
                "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(p['title'].replace(' ', '_'))}",
                "html": p["text"]})
            print(f"[{concept}] {lang} {p['title']!r} rev {p['revid']}", flush=True)


def split():
    # merge: entries of concepts outside the current set ($ML_CONCEPTS) are kept as they are
    pages = load_json(PAGES_PATH) if PAGES_PATH.exists() else {}
    for concept in concepts():
        d = RAW / slug(concept)
        pages[concept] = {}
        for lang in LANGS:
            if not (d / f"{lang}.json").exists():
                pages[concept][lang] = {"exists": False, "title": None, "revid": None, "n_words": 0,
                                        "n_sentences": 0}
                continue
            raw = load_json(d / f"{lang}.json")
            parser = _ParagraphText()
            parser.feed(raw["html"])
            sents = [s for s in split_sentences(parser.paragraphs, lang) if usable(s, lang)]
            n_words = sum(word_count(p, lang) for p in parser.paragraphs)
            dump_json(POOLS / slug(concept) / f"{lang}.json",
                      {"concept": concept, "lang": lang, "title": raw["title"], "revid": raw["revid"],
                       "sentences": sents})
            ll = d / "_langlinks.json"
            via = (load_json(ll).get("_via", {}) if ll.exists() else {}).get(lang, EN_TITLES[concept])
            pages[concept][lang] = {"exists": True, "langlink_from_en": via, "title": raw["title"], "revid": raw["revid"], "url": raw["url"],
                                    "n_words": n_words, "n_sentences": len(sents)}
            print(f"[{concept}] {lang}: {n_words} words, {len(sents)} sentences", flush=True)
    dump_json(PAGES_PATH, pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["fetch", "split"], choices=["fetch", "split"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if "fetch" in args.stage:
        fetch(args.force)
    if "split" in args.stage:
        split()


if __name__ == "__main__":
    main()
