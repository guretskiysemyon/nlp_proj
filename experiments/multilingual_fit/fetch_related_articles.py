#!/usr/bin/env python3
"""Download related Wikipedia articles where one article per language is too short.

The English concept sentences were sampled from "concept-related Wikipedia pages"
(paper), not from one article. Many ru/de/es/fr main articles give fewer than
300 usable sentences, so this adds related articles in the same language.

A related article, for (concept, language):
  1. is among the 50 most similar articles to the main article by Wikipedia's own
     search (CirrusSearch `morelike:<main title>`), in that language, and
  2. is linked from the main article (redirects resolved).
They are taken in `morelike` order until the main article plus the related ones hold
`--target` usable sentences (main article first; capped at `--max-related` pages).
(concept, language) pairs whose main article already has `--target` are skipped.

Needs wiki_raw/ from fetch_wiki_articles.py. Writes, without touching data/ or the
main articles:
    wiki_raw/<lang>/related/<Concept>/NN_<title>.json   same fields as the main articles
    wiki_raw/related_report.json   per (concept, lang): main count, need, the
                                   candidates considered and the pages kept

Usage:
    .venv/bin/python -I experiments/multilingual_fit/fetch_related_articles.py
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_wiki_articles import api, fetch_page, safe  # noqa: E402

HERE = Path(__file__).resolve().parent
LANGS = ["ru", "de", "es", "fr"]

# Usable-sentence estimate, same rule as fetch_wiki_sentences.py (the splitter is
# known to break on "1. Juni"-style dates; the final sets will use a fixed one).
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"«„(\[]?[A-ZÀ-ÖØ-ÞА-ЯЁ0-9])")


def n_sentences(text: str) -> int:
    n = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        for s in SPLIT.split(line):
            s = re.sub(r"\[\d+\]", "", s).strip()
            if 6 <= len(s.split()) <= 60 and s[-1] in ".!?»\"":
                n += 1
    return n


def linked_titles(lang: str, title: str) -> set:
    """Titles of main-namespace articles linked from `title`, redirects resolved."""
    out, cont = set(), {}
    while True:
        d = api(lang, action="query", generator="links", titles=title, gplnamespace=0,
                gpllimit="max", redirects=1, **cont)
        out.update(p["title"] for p in d.get("query", {}).get("pages", []) if not p.get("missing"))
        if "continue" not in d:
            return out
        cont = {k: v for k, v in d["continue"].items()}


def morelike(lang: str, title: str, limit: int) -> list:
    d = api(lang, action="query", list="search", srsearch=f"morelike:{title}",
            srnamespace=0, srlimit=limit)
    return [r["title"] for r in d.get("query", {}).get("search", [])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=HERE / "wiki_raw")
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--target", type=int, default=450,
                    help="usable sentences to reach (main + related); 300 are sampled later")
    ap.add_argument("--max-related", type=int, default=15)
    ap.add_argument("--morelike-limit", type=int, default=50)
    ap.add_argument("--exclude", type=Path, default=HERE / "related_exclude.json",
                    help="hand-made {concept: {lang: {title: reason}}} of related articles to skip")
    args = ap.parse_args()

    exclude = json.loads(args.exclude.read_text()) if args.exclude.exists() else {}
    main_report = json.loads((args.raw_dir / "report.json").read_text())
    rep_path = args.raw_dir / "related_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for concept in args.concepts or list(main_report):
        for lang in args.langs:
            info = main_report[concept]["langs"].get(lang)
            if not info:
                print(f"[{lang}] {concept}: no main article, skipped")
                continue
            main_title = info["title"]
            main_doc = json.loads((args.raw_dir / lang / f"{safe(concept)}.json").read_text())
            n_main = n_sentences(main_doc["text"])
            entry = {"main_title": main_title, "main_sentences": n_main, "target": args.target}
            if n_main >= args.target:
                entry.update(need=0, kept=[])
                print(f"[{lang}] {concept}: main has {n_main}, nothing to add", flush=True)
                report.setdefault(concept, {})[lang] = entry
                rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
                continue

            links = linked_titles(lang, main_title)
            similar = morelike(lang, main_title, args.morelike_limit)
            skip = exclude.get(concept, {}).get(lang, {})
            cands = [t for t in similar if t in links and t != main_title and t not in skip]
            entry.update(need=args.target - n_main, n_links=len(links), morelike=similar,
                         candidates=cands, excluded=sorted(t for t in skip if t in similar), kept=[])

            total = n_main
            out_dir = args.raw_dir / lang / "related" / safe(concept)
            for old in out_dir.glob("*.json"):  # this pair is rebuilt from scratch
                old.unlink()
            for rank, t in enumerate(cands):
                if total >= args.target or len(entry["kept"]) >= args.max_related:
                    break
                page = fetch_page(lang, t)
                if page is None:
                    continue
                n = n_sentences(page["text"])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{len(entry['kept']):02d}_{safe(page['title'])}.json").write_text(json.dumps(
                    {"concept": concept, "lang": lang, "related_to": main_title,
                     "morelike_rank": similar.index(t), "fetched_at": now, **page},
                    ensure_ascii=False, indent=1))
                entry["kept"].append({"title": page["title"], "sentences": n,
                                      "morelike_rank": similar.index(t)})
                total += n
            entry["total_sentences"] = total
            print(f"[{lang}] {concept}: main {n_main} + {len(entry['kept'])} related "
                  f"({', '.join(k['title'] for k in entry['kept'])}) -> {total}", flush=True)
            report.setdefault(concept, {})[lang] = entry
            rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
