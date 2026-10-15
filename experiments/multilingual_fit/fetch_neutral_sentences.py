#!/usr/bin/env python3
"""Neutral sentence sets for ru/de/es/fr, sampled the way the English set was.

The English neutral set (data/neutral_sentences.json) is 300 random sentences
from English Wikipedia (paper: "another n sentences from English Wikipedia"),
from 280 different articles. The first foreign sets (fetch_wiki_sentences.py)
took the FIRST sentence of random articles instead, i.e. mostly definitions
("X is a commune in France"), a different style from the concept sentences.

Here: draw random main-namespace articles of at least --min-bytes, split the
full text with sentences.py, and take ONE uniformly random usable sentence per
article, until --n sentences. Fixed seed.

Output: <out-dir>/neutral_sentences_<lang>.json
    [{"sentence", "title", "url", "position", "n_sentences_in_article"}]
and <out-dir>/neutral_report.json.

Usage:
    .venv/bin/python -I experiments/multilingual_fit/fetch_neutral_sentences.py
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_wiki_articles import api, fetch_page  # noqa: E402
from sentences import usable_sentences  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="*", default=["ru", "de", "es", "fr"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min-bytes", type=int, default=3000,
                    help="skip shorter articles (stubs: often only the defining first sentence)")
    ap.add_argument("--min-sentences", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=HERE / "data_v2")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = args.out_dir / "neutral_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    for lang in args.langs:
        if (args.out_dir / f"neutral_sentences_{lang}.json").exists():
            print(f"[{lang}] already done, skipped (delete the file to redo)", flush=True)
            continue
        rng = random.Random(f"{args.seed}-neutral-{lang}")
        got, seen, n_drawn, n_short = [], set(), 0, 0
        while len(got) < args.n:
            d = api(lang, action="query", generator="random", grnnamespace=0, grnlimit=20, prop="info")
            for p in d.get("query", {}).get("pages", []):
                if len(got) >= args.n or p["title"] in seen:
                    continue
                seen.add(p["title"])
                n_drawn += 1
                if p.get("length", 0) < args.min_bytes:
                    n_short += 1
                    continue
                page = fetch_page(lang, p["title"])
                sents = usable_sentences(page["text"], lang) if page else []
                if len(sents) < args.min_sentences:
                    n_short += 1
                    continue
                i = rng.randrange(len(sents))
                got.append({"sentence": sents[i], "title": page["title"], "url": page["url"],
                            "position": i, "n_sentences_in_article": len(sents)})
            print(f"[{lang}] {len(got)}/{args.n}", flush=True)
        (args.out_dir / f"neutral_sentences_{lang}.json").write_text(
            json.dumps(got, ensure_ascii=False, indent=1))
        report[lang] = {"n": len(got), "articles_drawn": n_drawn, "skipped_short": n_short,
                        "min_bytes": args.min_bytes, "seed": args.seed}
        rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
