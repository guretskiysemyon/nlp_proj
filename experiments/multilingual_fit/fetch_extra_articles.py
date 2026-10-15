#!/usr/bin/env python3
"""Download extra candidate articles per (concept, language) for ru/de/es/fr.

The related articles from fetch_related_articles.py (Wikipedia `morelike` search)
ran out of on-topic candidates for several pairs (holidays, drugs). This collects a
pool from two other sources for the pairs that are still short (by default the
pairs with fewer than --n sentences in data_v2/concept_report.json), keeping
everything; the sentences are filtered later (LLM judge), not here.

  1. category: pages in the concept's own category, up to --cat-depth levels of
     subcategories (default 0: the category itself). The category is found by
       a) the interlanguage link of the English category (EN_CATS), and
       b) the main article's own (non-hidden) categories whose name matches the
          concept stem (build_concept_sets.STEMS);
  2. search: the top --max-search results of each full-text search in QUERIES
     (exact phrases and alternative names; CirrusSearch syntax, main namespace),
     or of the quoted main-article title if the pair has no entry.
Category pages come first, capped at --max-cat; then search results. The main
article, articles already in related/, and disambiguation pages are skipped.

A first, broader run (categories to depth 1 and the unquoted title, all concepts)
drifted into namesakes and film pages; its partial output is in wiki_raw/_extra_broad/.

Needs wiki_raw/ from fetch_wiki_articles.py. Resumable: an article already on disk
is not downloaded again. Writes, without touching data/, main or related articles:
    wiki_raw/<lang>/extra/<Concept>/<title>.json   same fields as the main articles,
                                                   plus "sources" (how it was found)
    wiki_raw/extra_report.json   per (concept, lang): categories used, counts

Usage:
    .venv/bin/python -I experiments/multilingual_fit/fetch_extra_articles.py
"""
import argparse
import json
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_concept_sets import STEMS  # noqa: E402
import fetch_wiki_articles  # noqa: E402
from fetch_wiki_articles import api, safe  # noqa: E402

HERE = Path(__file__).resolve().parent
LANGS = ["ru", "de", "es", "fr"]
# concept -> English categories to follow (first that links to the language wins)
EN_CATS = {
    "Ancient Rome": ["Ancient Rome"],
    "Artificial intelligence": ["Artificial intelligence"],
    "Baseball": ["Baseball"],
    "COVID-19 pandemic": ["COVID-19 pandemic"],
    "Cannabis": ["Cannabis", "Cannabis (drug)"],
    "Culture of Greece": ["Culture of Greece", "Greek culture"],
    "Gambling": ["Gambling"],
    "Golf": ["Golf"],
    "Gun": ["Firearms", "Guns"],
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
# (concept, lang) -> full-text searches. Quoted phrases avoid namesakes ("Valentín",
# "Valentino"); -intitle drops the film/series pages; the drug senses of heroína /
# héroïne (also "heroine") need a drug term next to them.
NOFILM = {"ru": "-intitle:фильм", "de": "-intitle:Film", "es": "-intitle:película", "fr": "-intitle:film"}
QUERIES = {
    ("Baseball", "ru"): ['"бейсбол"'],
    ("Baseball", "es"): ['"béisbol"'],
    ("COVID-19 pandemic", "es"): ['"COVID-19"', '"pandemia de COVID-19"'],
    ("Culture of Greece", "de"): ['"griechische Kultur"', '"Kultur Griechenlands"'],
    ("Culture of Greece", "fr"): ['"culture grecque"', '"culture de la Grèce"'],
    ("Gambling", "ru"): ['"азартные игры"', '"азартная игра"'],
    ("Gambling", "fr"): ['"jeu d\'argent"', '"jeux d\'argent"', '"jeux de hasard"'],
    ("Gun", "de"): ['"Schusswaffe"', '"Feuerwaffe"'],
    ("Gun", "fr"): ['"arme à feu"'],
    ("Harry Potter", "ru"): ['"Гарри Поттер"', '"Гарри Поттера"', '"Хогвартс"'],  # main title is the novel series
    ("Halloween", "ru"): ['"Хэллоуин" ' + NOFILM["ru"]],
    ("Halloween", "de"): ['"Halloween" ' + NOFILM["de"]],
    ("Halloween", "es"): ['"Halloween" ' + NOFILM["es"], '"Noche de Brujas"', '"truco o trato"'],
    ("Halloween", "fr"): ['"Halloween" ' + NOFILM["fr"]],
    ("Heroin", "es"): ['"heroína" (droga OR opioide OR opiáceo)'],
    ("Heroin", "fr"): ['"héroïne" (drogue OR opioïde OR opiacé)'],
    ("Valentine's Day", "ru"): ['"День святого Валентина" ' + NOFILM["ru"], '"День всех влюблённых"'],
    ("Valentine's Day", "es"): ['"Día de San Valentín" ' + NOFILM["es"], '"Día de los enamorados"',
                                '"Día del amor y la amistad"'],
    ("Valentine's Day", "fr"): ['"la Saint-Valentin" ' + NOFILM["fr"], '"fête des amoureux"'],
}


def cat_langlink(en_cat: str, lang: str):
    d = api("en", action="query", titles=f"Category:{en_cat}", redirects=1,
            prop="langlinks", lllang=lang)
    page = d["query"]["pages"][0]
    lls = page.get("langlinks", [])
    return lls[0]["title"] if lls else None


def article_categories(lang: str, title: str) -> list:
    d = api(lang, action="query", titles=title, redirects=1, prop="categories",
            clshow="!hidden", cllimit="max")
    return [c["title"] for c in d["query"]["pages"][0].get("categories", [])]


def category_pages(lang: str, cats: list, depth: int, cap: int) -> list:
    """(title, "category:<cat> d<k>") for main-namespace pages, breadth first."""
    out, seen_pages, seen_cats = [], set(), set(cats)
    level = list(cats)
    for d in range(depth + 1):
        nxt = []
        for cat in level:
            cont = {}
            while True:
                r = api(lang, action="query", list="categorymembers", cmtitle=cat,
                        cmtype="page|subcat", cmlimit="max", **cont)
                for m in r.get("query", {}).get("categorymembers", []):
                    if m["ns"] == 0 and m["title"] not in seen_pages:
                        seen_pages.add(m["title"])
                        out.append((m["title"], f"category:{cat} d{d}"))
                    elif m["ns"] == 14 and m["title"] not in seen_cats:
                        seen_cats.add(m["title"])
                        nxt.append(m["title"])
                if "continue" not in r:
                    break
                cont = dict(r["continue"])
            if len(out) >= cap:
                return out[:cap]
        level = nxt
    return out[:cap]


def search_pages(lang: str, query: str, limit: int) -> list:
    r = api(lang, action="query", list="search", srsearch=query, srnamespace=0,
            srlimit=limit, srprop="")
    return [(p["title"], f"search:{query} r{i + 1}")
            for i, p in enumerate(r.get("query", {}).get("search", []))]


def fetch(lang: str, title: str):
    """Plain text + ids, or None if missing or a disambiguation page."""
    d = api(lang, action="query", titles=title, redirects=1, prop="extracts|info|pageprops",
            explaintext=1, exsectionformat="plain", ppprop="disambiguation")
    page = d["query"]["pages"][0]
    if page.get("missing") or page.get("invalid") or "disambiguation" in page.get("pageprops", {}):
        return None
    return {"title": page["title"], "pageid": page["pageid"], "revid": page.get("lastrevid"),
            "url": f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_")),
            "text": page.get("extract", "") or ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=HERE / "wiki_raw")
    ap.add_argument("--concepts", nargs="*", default=list(EN_CATS))
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--all-pairs", action="store_true",
                    help="every (concept, lang), not only the pairs short in --short-from")
    ap.add_argument("--short-from", type=Path, default=HERE / "data_v2" / "concept_report.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cat-depth", type=int, default=0)
    ap.add_argument("--max-cat", type=int, default=150)
    ap.add_argument("--max-search", type=int, default=100, help="results per query")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between API requests")
    args = ap.parse_args()
    fetch_wiki_articles.SLEEP = args.sleep

    pairs = [(c, l) for l in args.langs for c in args.concepts]
    if not args.all_pairs:
        short = json.loads(args.short_from.read_text())
        pairs = [(c, l) for c, l in pairs if short[c][l]["used"] < args.n]
    print(f"{len(pairs)} pairs: " + ", ".join(f"{c}/{l}" for c, l in pairs), flush=True)

    rep_path = args.raw_dir / "extra_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    for concept, lang in pairs:
        main_doc = json.loads((args.raw_dir / lang / f"{safe(concept)}.json").read_text())
        main_title = main_doc["title"]
        rel_dir = args.raw_dir / lang / "related" / safe(concept)
        skip = {main_title} | {json.loads(f.read_text())["title"] for f in rel_dir.glob("*.json")}

        cats = []
        for en_cat in EN_CATS[concept]:
            c = cat_langlink(en_cat, lang)
            if c:
                cats.append(c)
                break
        rx = re.compile(STEMS[concept], re.I)
        cats += [c for c in article_categories(lang, main_title)
                 if c not in cats and rx.search(c.split(":", 1)[1])]

        cands = category_pages(lang, cats, args.cat_depth, args.max_cat) if cats else []
        queries = QUERIES.get((concept, lang)) or ['"' + re.sub(r"\s*\(.*?\)", "", main_title) + '"']
        for query in queries:
            cands += search_pages(lang, query, args.max_search)
        sources = {}
        for t, src in cands:
            if t not in skip:
                sources.setdefault(t, []).append(src)

        out_dir = args.raw_dir / lang / "extra" / safe(concept)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_new = n_have = n_none = 0
        for i, (t, srcs) in enumerate(sources.items()):
            out = out_dir / f"{safe(t)[:120]}.json"
            if out.exists():
                n_have += 1
                continue
            page = fetch(lang, t)
            if page is None or page["title"] in skip:
                n_none += 1
                continue
            out.write_text(json.dumps(
                {"concept": concept, "lang": lang, "sources": srcs,
                 "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **page},
                ensure_ascii=False, indent=1))
            n_new += 1
            if (i + 1) % 25 == 0:
                print(f"  [{lang}] {concept}: {i + 1}/{len(sources)}", flush=True)
        n_cat = sum(any(s.startswith("category:") for s in v) for v in sources.values())
        report.setdefault(concept, {})[lang] = {
            "categories": cats, "search_queries": queries, "candidates": len(sources),
            "from_category": n_cat, "from_search_only": len(sources) - n_cat,
            "downloaded_now": n_new, "already_on_disk": n_have, "skipped_missing_or_disambig": n_none,
            "cat_depth": args.cat_depth, "max_cat": args.max_cat, "max_search": args.max_search}
        rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"[{lang}] {concept}: cats={cats} candidates={len(sources)} "
              f"(cat {n_cat}) new={n_new} have={n_have} skipped={n_none}", flush=True)


if __name__ == "__main__":
    main()
