#!/usr/bin/env python3
"""Check the data_v2 neutral sets against the reference rules of sentences.py, with the
full article of every sentence (the rules need it: back-matter sections, whole lines).

For each neutral sentence: fetch its article (today's revision; saved to
wiki_raw/<lang>/neutral/<title>.json, so a rerun reads the saved copy), split it with
usable_sentences(), and
  - keep the sentence if it is still a usable sentence of the article;
  - replace it if the article has it but the reference rules drop it: another uniformly
    random usable sentence of the same article (fixed seed), or, if the article has fewer
    than --min-sentences usable sentences, one from a new random article (same rules as
    fetch_neutral_sentences.py);
  - if today's article no longer contains it (edited since it was drawn): keep it when the
    sentence passes the rules on its own, and mark it "not_in_current_text".

Output (in --data-dir, default data_v2/):
    neutral_sentences_<lang>.json            updated in place (replaced entries get the
                                             article's "revid")
    neutral_removed_references.json          every replaced sentence: lang, title, the rule,
                                             the full source line, and its replacement
    neutral_report.json                      "recheck" summary per language

Usage:
    .venv/bin/python -I experiments/multilingual_fit/recheck_neutral.py
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_wiki_articles as W  # noqa: E402
from fetch_wiki_articles import api, fetch_page, safe  # noqa: E402
from sentences import usable_sentences  # noqa: E402

HERE = Path(__file__).resolve().parent


def article(lang, title, raw_dir):
    path = raw_dir / lang / "neutral" / f"{safe(title)}.json"
    if path.exists():
        return json.loads(path.read_text())
    page = fetch_page(lang, title)
    if page is not None:
        page["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(page, ensure_ascii=False, indent=1))
    return page


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="*", default=["ru", "de", "es", "fr"])
    ap.add_argument("--data-dir", type=Path, default=HERE / "data_v2")
    ap.add_argument("--raw-dir", type=Path, default=HERE / "wiki_raw")
    ap.add_argument("--min-bytes", type=int, default=3000)
    ap.add_argument("--min-sentences", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds after each API request")
    args = ap.parse_args()
    W.SLEEP = args.sleep

    rep_path = args.data_dir / "neutral_report.json"
    report = json.loads(rep_path.read_text())
    log_path = args.data_dir / "neutral_removed_references.json"
    log = [x for x in (json.loads(log_path.read_text()) if log_path.exists() else [])
           if x["lang"] not in args.langs]
    for lang in args.langs:
        path = args.data_dir / f"neutral_sentences_{lang}.json"
        got = json.loads(path.read_text())
        rng = random.Random(f"{args.seed}-neutral-recheck-{lang}")
        in_set = {e["sentence"] for e in got}
        seen_titles = {e["title"] for e in got}
        counts = {"kept": 0, "replaced": 0, "not_in_current_text": 0, "missing_article": 0}
        for i, e in enumerate(got):
            page = article(lang, e["title"], args.raw_dir)
            removed = []
            sents = usable_sentences(page["text"], lang, removed) if page else []
            if e["sentence"] in sents:
                counts["kept"] += 1
                continue
            hit = next((r for r in removed if r["sentence"] == e["sentence"]), None)
            if hit is None:  # article edited (or gone) since the sentence was drawn
                alone = []
                if usable_sentences(e["sentence"], lang, alone) == [e["sentence"]]:
                    e["status"] = "not_in_current_text" if page else "missing_article"
                    counts[e["status"]] += 1
                    continue
                hit = {"sentence": e["sentence"], "line": e["sentence"], "rule": alone[0]["rule"]
                       if alone else "shape"}
            # replace: same article if it still has enough usable sentences, else a new one
            pool = [s for s in sents if s not in in_set]
            while len(sents) < args.min_sentences or not pool:
                d = api(lang, action="query", generator="random", grnnamespace=0, grnlimit=1, prop="info")
                p = d["query"]["pages"][0]
                if p["title"] in seen_titles or p.get("length", 0) < args.min_bytes:
                    continue
                seen_titles.add(p["title"])
                page = article(lang, p["title"], args.raw_dir)
                sents = usable_sentences(page["text"], lang) if page else []
                pool = [s for s in sents if s not in in_set]
            new = rng.choice(pool)
            in_set.discard(e["sentence"]); in_set.add(new)
            got[i] = {"sentence": new, "title": page["title"], "url": page["url"], "revid": page["revid"],
                      "position": sents.index(new), "n_sentences_in_article": len(sents)}
            log.append({"lang": lang, "title": e["title"], "rule": hit["rule"], "sentence": e["sentence"],
                        "line": hit["line"], "replacement": new, "replacement_title": page["title"]})
            counts["replaced"] += 1
            print(f"[{lang}] {i}: replaced ({hit['rule']}): {e['sentence'][:80]}", flush=True)
        path.write_text(json.dumps(got, ensure_ascii=False, indent=1))
        report[lang]["recheck"] = {"date": datetime.now(timezone.utc).date().isoformat(), **counts}
        rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
        print(f"[{lang}] {counts}", flush=True)


if __name__ == "__main__":
    main()
