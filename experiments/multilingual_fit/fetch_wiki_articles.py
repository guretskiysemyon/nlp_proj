#!/usr/bin/env python3
"""Download the raw Wikipedia article for every concept in en/ru/de/es/fr.

Unlike fetch_wiki_sentences.py this does NOT split sentences or sample: it stores
the full plain text of each article (plus page and revision ids), so sentence
extraction can be redone later without hitting Wikipedia again. It never touches
experiments/multilingual_fit/data/.

For each concept: try the English title candidates, keep the one whose text
contains the most of the concept's English sentences (data/concept_sentences.json),
follow its interlanguage links to ru/de/es/fr, and download every article.

Output (<out-dir>, default experiments/multilingual_fit/wiki_raw/):
    <lang>/<Concept>.json   {"concept","lang","title","en_title","pageid","revid","url","fetched_at","text"}
    report.json             per concept: chosen English title, English-sentence overlap,
                            and per language the title (or null if no article exists)

Usage:
    .venv/bin/python -I experiments/multilingual_fit/fetch_wiki_articles.py
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
UA = "EMBER-Embedding-Erasure-course-project/0.1 (academic coursework; low-volume)"
LANGS = ["ru", "de", "es", "fr"]
# concept name in data/concept_sentences.json -> English Wikipedia titles to try
TITLES = {
    "Ancient Rome": ["Ancient Rome"],
    "Artificial intelligence": ["Artificial intelligence"],
    "Baseball": ["Baseball"],
    "COVID-19 pandemic": ["COVID-19 pandemic"],
    "Cannabis": ["Cannabis", "Cannabis (drug)"],
    "Culture of Greece": ["Culture of Greece"],
    "Gambling": ["Gambling"],
    "Golf": ["Golf"],
    "Gun": ["Gun", "Firearm"],
    "Halloween": ["Halloween"],
    "Harry Potter": ["Harry Potter"],
    "Heroin": ["Heroin"],
    "Nazism": ["Nazism"],
    "Pornography": ["Pornography"],
    "Republic of Ireland": ["Republic of Ireland"],
    "Uranium": ["Uranium"],
    "Valentine's Day": ["Valentine's Day"],
    "World War II": ["World War II"],
}


def api(lang: str, **params):
    params.update(format="json", formatversion="2")
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    err = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
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


def fetch_page(lang: str, title: str):
    """Full plain text + ids of one article (redirects followed), or None if missing."""
    d = api(lang, action="query", titles=title, redirects=1,
            prop="extracts|info", explaintext=1, exsectionformat="plain")
    page = d["query"]["pages"][0]
    if page.get("missing") or page.get("invalid"):
        return None
    return {"title": page["title"], "pageid": page["pageid"], "revid": page.get("lastrevid"),
            "url": f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_")),
            "text": page.get("extract", "") or ""}


def langlinks(title: str):
    d = api("en", action="query", titles=title, redirects=1, prop="langlinks", lllimit="max")
    page = d["query"]["pages"][0]
    return {ll["lang"]: ll["title"] for ll in page.get("langlinks", [])}


def _norm(s: str) -> str:
    s = re.sub(r"\[\d+\]|\[citation needed\]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def overlap(sentences, text: str) -> float:
    """Share of the English concept sentences found verbatim in the article text."""
    t = _norm(text)
    found = sum(1 for s in sentences if len(_norm(s)) > 20 and _norm(s) in t)
    return found / max(1, len(sentences))


def safe(concept: str) -> str:
    return re.sub(r"[^\w\-]+", "_", concept).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=HERE / "wiki_raw")
    ap.add_argument("--concepts", nargs="*", default=list(TITLES))
    args = ap.parse_args()

    en_sents = {e["concept"]: e["sentences"]
                for e in json.loads((REPO / "data" / "concept_sentences.json").read_text())}
    report_path = args.out_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for concept in args.concepts:
        # English: pick the candidate title whose text holds most of our English sentences
        best = None
        for cand in TITLES[concept]:
            page = fetch_page("en", cand)
            if page is None:
                print(f"[en] {concept}: '{cand}' does not exist")
                continue
            ov = overlap(en_sents.get(concept, []), page["text"])
            print(f"[en] {concept}: '{page['title']}' {len(page['text'])} chars, "
                  f"English-sentence overlap {ov:.2f}", flush=True)
            if best is None or ov > best[1]:
                best = (page, ov)
        if best is None:
            report[concept] = {"en": None}
            continue
        en_page, ov = best
        entry = {"en_title": en_page["title"], "en_sentence_overlap": round(ov, 3),
                 "langs": {"en": {"title": en_page["title"], "chars": len(en_page["text"])}}}
        pages = {"en": en_page}

        # Interlanguage links: from the chosen English article first, then from the
        # other candidates (en:"Gun" links to no ru/de/es/fr article, en:"Firearm" does).
        cands = [en_page["title"]] + [t for t in TITLES[concept] if t != en_page["title"]]
        links = {t: langlinks(t) for t in cands}
        for lang in LANGS:
            via = next((t for t in cands if lang in links[t]), None)
            title = links[via][lang] if via else None
            page = fetch_page(lang, title) if title else None
            if page:
                page["linked_from_en"] = via
            entry["langs"][lang] = ({"title": page["title"], "chars": len(page["text"]),
                                     "linked_from_en": via} if page else None)
            print(f"[{lang}] {concept}: " + (f"'{page['title']}' {len(page['text'])} chars"
                                             if page else "NO ARTICLE"), flush=True)
            if page:
                pages[lang] = page

        for lang, page in pages.items():
            out = args.out_dir / lang / f"{safe(concept)}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"concept": concept, "lang": lang, "en_title": en_page["title"],
                                       "fetched_at": now, **page}, ensure_ascii=False, indent=1))
        report[concept] = entry
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
