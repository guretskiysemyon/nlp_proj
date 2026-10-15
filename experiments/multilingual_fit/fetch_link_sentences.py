#!/usr/bin/env python3
"""Sentences that link to the concept's main article, for the short (concept, language) pairs.

A third source of extra sentences, next to the related articles (fetch_related_articles.py)
and the category / search pool (fetch_extra_articles.py). Wikipedia articles that link to
the main article ("what links here") are listed; in each, the sentence holding the link is
kept. The link fixes the sense: a link to "Héroïne" is the drug, not a heroine; a link to
"Saint-Valentin" is the holiday, not Valentino or a Saint-Valentin commune. The sentence can
still be a passing mention ("on Valentine's Day, X released ..."); the LLM judge decides.

Per pair: backlinks in the main namespace, direct and through redirects to the main
article; a fixed-seed random sample of --per-needed x (sentences missing to --n) articles,
at least --min-articles and at most --max-articles; the rendered page
(action=parse, so date templates come out as text); only body paragraphs (<p>), not lists,
tables, infoboxes or navboxes; split with sentences.py, same 6-60 word and reference-line
rules. Pairs run shortest first. Resumable: articles already in the output are skipped;
a rerun with a larger sample continues along the same shuffled order.

Output (data/, data_v2/, related/ and extra/ are not touched):
    wiki_raw/<lang>/links/<Concept>.json   {"concept", "lang", "main_title", "targets",
                                            "n_linking", "order" (all linking articles,
                                            shuffled), "sample_size", "articles": {title:
                                            {"pageid", "revid", "url", "sentences"}}}
    wiki_raw/links_report.json             counts per (concept, lang)

Usage:
    .venv/bin/python -I experiments/multilingual_fit/fetch_link_sentences.py
"""
import argparse
import json
import random
import re
import sys
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_concept_sets import STEMS  # noqa: E402
import fetch_wiki_articles  # noqa: E402
from fetch_wiki_articles import api, safe  # noqa: E402
from sentences import split_sentences, usable_sentences  # noqa: E402

HERE = Path(__file__).resolve().parent
LANGS = ["ru", "de", "es", "fr"]
MARK = ""  # private-use character put right after the link text
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
SKIP_TAGS = {"table", "ul", "ol", "dl", "figure", "style", "script", "math"}
SKIP_CLASSES = ("reference", "noprint", "mw-editsection", "mwe-math", "navbox", "infobox", "metadata")


def norm_title(t: str) -> str:
    t = t.replace("_", " ").strip()
    return t[:1].upper() + t[1:]


class Paragraphs(HTMLParser):
    """Text of the top-level <p> paragraphs, with MARK after each link to one of `targets`."""

    def __init__(self, targets: set):
        super().__init__(convert_charrefs=True)
        self.targets = targets
        self.stack = []  # (tag, skip, is_target_link)
        self.skip = 0
        self.buf = None  # text of the open paragraph
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            if tag == "br" and self.buf is not None and not self.skip:
                self.buf.append(" ")
            return
        a = dict(attrs)
        skip = tag in SKIP_TAGS or any(c in (a.get("class") or "") for c in SKIP_CLASSES)
        target = False
        if tag == "a" and not self.skip and self.buf is not None:
            # "/wiki/T" (legacy parser), "https://xx.wikipedia.org/wiki/T" or "./T" (Parsoid)
            m = re.match(r"(?:(?:https?:)?//[^/]+)?/wiki/([^?#]+)|\./([^?#]+)", a.get("href") or "")
            if m:
                target = norm_title(urllib.parse.unquote(m.group(1) or m.group(2))) in self.targets
        if tag == "p" and not self.skip and self.buf is None:
            self.buf = []
        self.stack.append((tag, skip, target))
        self.skip += skip

    def handle_endtag(self, tag):
        if not any(t == tag for t, _, _ in self.stack):
            return
        while self.stack:
            t, skip, target = self.stack.pop()
            self.skip -= skip
            if target and self.buf is not None:
                self.buf.append(MARK)
            if t == "p" and self.buf is not None and not self.skip:
                self.paragraphs.append("".join(self.buf))
                self.buf = None
            if t == tag:
                break

    def handle_data(self, data):
        if self.buf is not None and not self.skip:
            self.buf.append(data)


def link_sentences(html: str, targets: set, lang: str) -> list:
    p = Paragraphs(targets)
    p.feed(html)
    out = []
    for para in p.paragraphs:
        if MARK not in para:
            continue
        for piece in split_sentences(para.replace("\n", " "), lang):
            if MARK in piece:
                kept = usable_sentences(piece.replace(MARK, ""), lang)
                if len(kept) == 1 and kept[0] not in out:
                    out.append(kept[0])
    return out


def backlinks(lang: str, title: str):
    """(linking article titles, target titles = the main title + redirects to it)."""
    linking, targets, cont = [], {norm_title(title)}, {}
    while True:
        r = api(lang, action="query", list="backlinks", bltitle=title, blnamespace=0,
                bllimit="max", blredirect=1, **cont)
        for b in r["query"]["backlinks"]:
            if b.get("redirect"):
                targets.add(norm_title(b["title"]))
            else:
                linking.append(b["title"])
            linking += [x["title"] for x in b.get("redirlinks", []) if x.get("ns", 0) == 0]
        if "continue" not in r:
            break
        cont = dict(r["continue"])
    return list(dict.fromkeys(linking)), targets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=HERE / "wiki_raw")
    ap.add_argument("--concepts", nargs="*", default=list(STEMS))
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--all-pairs", action="store_true",
                    help="every (concept, lang), not only the pairs short in --short-from")
    ap.add_argument("--short-from", type=Path, default=HERE / "data_v2" / "concept_report.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--per-needed", type=int, default=4,
                    help="linking articles sampled per missing sentence (most give 0-1, and the judge drops some)")
    ap.add_argument("--min-articles", type=int, default=100)
    ap.add_argument("--max-articles", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between API requests")
    args = ap.parse_args()
    fetch_wiki_articles.SLEEP = args.sleep

    short = json.loads(args.short_from.read_text())
    pairs = [(c, l) for l in args.langs for c in args.concepts]
    if not args.all_pairs:
        pairs = [(c, l) for c, l in pairs if short[c][l]["used"] < args.n]
    pairs.sort(key=lambda p: short[p[0]][p[1]]["used"])  # shortest first
    print(f"{len(pairs)} pairs: " + ", ".join(f"{c}/{l}" for c, l in pairs), flush=True)

    main_report = json.loads((args.raw_dir / "report.json").read_text())
    rep_path = args.raw_dir / "links_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    for concept, lang in pairs:
        main_title = main_report[concept]["langs"][lang]["title"]
        out = args.raw_dir / lang / "links" / f"{safe(concept)}.json"
        if out.exists():
            doc = json.loads(out.read_text())
        else:
            linking, targets = backlinks(lang, main_title)
            order = linking[:]
            random.Random(f"{args.seed}-links-{concept}-{lang}").shuffle(order)
            doc = {"concept": concept, "lang": lang, "main_title": main_title, "targets": sorted(targets),
                   "n_linking": len(linking), "order": order, "sample_size": 0, "articles": {}}
        need = args.n - short[concept][lang]["used"]
        size = min(args.max_articles, max(args.min_articles, args.per_needed * need))
        doc["sample_size"] = max(doc["sample_size"], min(size, len(doc["order"])))
        targets = set(doc["targets"])
        out.parent.mkdir(parents=True, exist_ok=True)
        todo = [t for t in doc["order"][:doc["sample_size"]] if t not in doc["articles"]]
        for i, t in enumerate(todo):
            try:
                r = api(lang, action="parse", page=t, prop="text|revid", redirects=1,
                        disableeditsection=1, disabletoc=1)["parse"]
            except (KeyError, RuntimeError):
                doc["articles"][t] = {"error": True, "sentences": []}
                continue
            doc["articles"][t] = {
                "pageid": r.get("pageid"), "revid": r.get("revid"),
                "url": f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(r["title"].replace(" ", "_")),
                "sentences": link_sentences(r["text"], targets, lang)}
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
                print(f"  [{lang}] {concept}: {len(doc['articles'])}/{doc['sample_size']}", flush=True)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))

        rx = re.compile(STEMS[concept], re.I)
        sents = list(dict.fromkeys(s for a in doc["articles"].values() for s in a["sentences"]))
        report.setdefault(concept, {})[lang] = {
            "main_title": main_title, "have": short[concept][lang]["used"], "n_linking": doc["n_linking"],
            "sampled": doc["sample_size"], "fetched": sum(not a.get("error") for a in doc["articles"].values()),
            "articles_with_sentence": sum(bool(a["sentences"]) for a in doc["articles"].values()),
            "sentences": len(sents), "sentences_naming_stem": sum(bool(rx.search(s)) for s in sents),
            "per_needed": args.per_needed, "min_articles": args.min_articles,
            "max_articles": args.max_articles, "seed": args.seed}
        rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        r = report[concept][lang]
        print(f"[{lang}] {concept}: have {r['have']}, {r['n_linking']} linking, {r['fetched']} fetched, "
              f"{r['articles_with_sentence']} with a sentence, {r['sentences']} sentences", flush=True)


if __name__ == "__main__":
    main()
