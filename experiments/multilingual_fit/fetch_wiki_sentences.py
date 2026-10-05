#!/usr/bin/env python3
"""Build foreign-language concept / neutral sentence sets from Wikipedia.

Mirrors how the English EMBER inputs were made (data/concept_sentences.json,
data/neutral_sentences.json): concept sentences come from the concept's own
Wikipedia article, neutral sentences from random articles.

For each concept: find the same article in de/es/fr/ru through the English
article's interlanguage links, take its plain text, split into sentences, keep
up to N (fixed-seed sample). For each language: first sentence of N random
articles as the neutral set.

Output (experiments/multilingual_fit/data/):
    concept_sentences_<lang>.json   [{"concept": ..., "sentences": [...], "title": ...}]
    neutral_sentences_<lang>.json   [{"sentence": ..., "title": ...}]
    fetch_report.json               counts per concept/lang

Usage:
    .venv/bin/python experiments/multilingual_fit/fetch_wiki_sentences.py
"""
import argparse
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "data"
UA = "EMBER-Embedding-Erasure-course-project/0.1 (academic coursework; low-volume)"
CONCEPTS = {  # concept name in data/concept_sentences.json -> English Wikipedia titles to try, in order
    # en:"Gun" has no interlanguage link to ru; en:"Firearm" is the same topic (-> ru "Огнестрельное оружие").
    "Gun": ["Gun", "Firearm"], "Heroin": ["Heroin"], "Gambling": ["Gambling"],
    "Ancient Rome": ["Ancient Rome"], "World War II": ["World War II"],
}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"«„(\[]?[A-ZÀ-ÖØ-ÞА-ЯЁ0-9])")


def api(lang: str, **params):
    params.update(format="json", formatversion="2")
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    err = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            time.sleep(1.0)
            return data
        except urllib.error.HTTPError as e:  # 429: honour Retry-After, else back off
            err = e
            wait = int(e.headers.get("Retry-After") or 0) or min(120, 10 * 2 ** attempt)
            print(f"  HTTP {e.code}; sleeping {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:  # network hiccup
            err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"API failed for {url}: {err}")


def clean_sentences(text: str):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        for s in SENT_SPLIT.split(line):
            s = re.sub(r"\[\d+\]|\[citation needed\]", "", s).strip()
            n = len(s.split())
            if 6 <= n <= 60 and s[-1] in ".!?»\"":
                out.append(s)
    return out


def langlink_title(en_title: str, lang: str):
    d = api("en", action="query", titles=en_title, prop="langlinks", lllang=lang, redirects=1)
    for p in d["query"]["pages"]:
        for ll in p.get("langlinks", []):
            return ll["title"]
    return None


def article_text(lang: str, title: str) -> str:
    d = api(lang, action="query", titles=title, prop="extracts", explaintext=1,
            exsectionformat="plain", redirects=1)
    return d["query"]["pages"][0].get("extract", "") or ""


def random_first_sentences(lang: str, n: int, rng: random.Random):
    got, seen = [], set()
    while len(got) < n:
        d = api(lang, action="query", generator="random", grnnamespace=0, grnlimit=20,
                prop="extracts", exintro=1, explaintext=1, exlimit=20)
        for p in d.get("query", {}).get("pages", []):
            if p["title"] in seen:
                continue
            seen.add(p["title"])
            sents = clean_sentences(p.get("extract", "") or "")
            if sents:
                got.append({"sentence": sents[0], "title": p["title"]})
    return got[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="*", default=["ru", "de", "es", "fr"])
    ap.add_argument("--n-concept", type=int, default=300)
    ap.add_argument("--n-neutral", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    report = {}
    for lang in args.langs:
        rng = random.Random(args.seed)
        entries = []
        for concept, en_titles in CONCEPTS.items():
            title = next((t for t in map(lambda x: langlink_title(x, lang), en_titles) if t), None)
            if not title:
                print(f"[{lang}] {concept}: no {lang} article"); continue
            sents = clean_sentences(article_text(lang, title))
            n_all = len(sents)
            if len(sents) > args.n_concept:
                sents = rng.sample(sents, args.n_concept)
            entries.append({"concept": concept, "title": title, "sentences": sents})
            report.setdefault(concept, {})[lang] = {"title": title, "n_valid_in_article": n_all, "n_used": len(sents)}
            print(f"[{lang}] {concept}: '{title}' -> {n_all} valid sentences, using {len(sents)}")
        (OUT / f"concept_sentences_{lang}.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1))
        neutral = random_first_sentences(lang, args.n_neutral, rng)
        (OUT / f"neutral_sentences_{lang}.json").write_text(json.dumps(neutral, ensure_ascii=False, indent=1))
        report.setdefault("_neutral", {})[lang] = len(neutral)
        print(f"[{lang}] neutral: {len(neutral)} sentences")
    (OUT / "fetch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
