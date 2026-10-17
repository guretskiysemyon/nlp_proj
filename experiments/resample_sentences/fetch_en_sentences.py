#!/usr/bin/env python3
"""Re-sample English concept / neutral sentence sets from Wikipedia.

Imitates how data/concept_sentences.json and data/neutral_sentences.json were
built, as far as can be inferred from those files:
  concept: paragraph text of the concept's Wikipedia article, numeric [n]
           citations removed, split into sentences, >= 5 words, sample N.
  neutral: one random sentence (>= 5 words) from each of N random English
           Wikipedia articles, stored with its url/title.

Concept text comes from the live article (one API call). Neutral articles come
from a local random sample of the wikimedia/wikipedia dump, made by
sample_wiki_dump.py, so no API is needed and the page set is reproducible.
Only prose lines (ending in . ! ? or a closing quote) are used, which drops the
dump's section headings and list items.

Two deliberate fixes vs. the original neutral file: numeric [n] citations are
stripped (as on the concept side), and maintenance boilerplate such as
"You can help Wikipedia by expanding it." is dropped.

Output (experiments/resample_sentences/data/):
    concept_sentences_<slug>.json   [{"concept", "source", "revid", "sentences": [...]}]
    neutral_sentences.json          [{"url", "title", "sentence"}]
    report.json                     pool sizes and overlap with the original files

Usage:
    python experiments/resample_sentences/sample_wiki_dump.py    # once: 1k random dump pages
    python experiments/resample_sentences/fetch_en_sentences.py --concept "Harry Potter"
"""
import argparse
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
OUT = HERE / "data"
UA = "EMBER-Embedding-Erasure-course-project/0.1 (academic coursework; low-volume)"
SENT_SPLIT = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[\"“'(\[]?[A-Z0-9])")
CITATION = re.compile(r"\[\d+\]|\[[a-z]\]|\[note \d+\]")
BOILERPLATE = re.compile(r"You can help Wikipedia|This article .* is a stub|is a stub\.", re.I)
PROSE_END = re.compile(r"[.!?][\"'”’)]*$")
MIN_WORDS = 5


def api(**params):
    params.update(format="json", formatversion="2")
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    err = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            time.sleep(0.5)
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


class _ParagraphText(HTMLParser):
    """Collect the text of top-level <p> elements, skipping style/script and reference superscripts."""

    def __init__(self):
        super().__init__()
        self.paragraphs, self._buf, self._in_p = [], [], 0
        self._skip_tag, self._skip = None, 0  # tag being skipped and its nesting depth

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if self._skip:
            self._skip += tag == self._skip_tag
        elif tag in ("style", "script") or (tag == "sup" and "reference" in cls):
            self._skip_tag, self._skip = tag, 1
        elif tag == "p":
            self._in_p += 1

    def handle_endtag(self, tag):
        if self._skip:
            self._skip -= tag == self._skip_tag
        elif tag == "p" and self._in_p:
            self._in_p -= 1
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text:
                self.paragraphs.append(text)
            self._buf = []

    def handle_data(self, data):
        if self._in_p and not self._skip:
            self._buf.append(data)


def page_paragraphs(title: str):
    d = api(action="parse", page=title, prop="text|revid", redirects=1)
    p = _ParagraphText()
    p.feed(d["parse"]["text"])
    return d["parse"]["title"], d["parse"]["revid"], p.paragraphs


def split_sentences(paragraphs):
    out = []
    for para in paragraphs:
        for s in SENT_SPLIT.split(CITATION.sub("", para)):
            s = s.strip()
            if len(s.split()) >= MIN_WORDS and not BOILERPLATE.search(s):
                out.append(s)
    return list(dict.fromkeys(out))  # dedupe, keep order


def neutral_sentences(pages_path: Path, n: int, rng: random.Random, exclude: set):
    """One random sentence from each of n random pages of the local dump sample."""
    pages = [json.loads(line) for line in pages_path.open(encoding="utf-8")]
    got = []
    for page in rng.sample(pages, len(pages)):
        paras = [line.strip() for line in page["text"].splitlines() if PROSE_END.search(line.strip())]
        sents = [s for s in split_sentences(paras) if s not in exclude]
        if sents:
            got.append({"url": page["url"], "title": page["title"], "sentence": rng.choice(sents)})
            if len(got) == n:
                return got
    raise ValueError(f"Only {len(got)} of {len(pages)} pages in {pages_path} had a usable sentence; need {n}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", default="Harry Potter", help="Name as in data/concept_sentences.json.")
    ap.add_argument("--title", default=None, help="Wikipedia title (defaults to --concept).")
    ap.add_argument("--n-concept", type=int, default=300)
    ap.add_argument("--n-neutral", type=int, default=300)
    ap.add_argument("--seed", type=int, default=43, help="Differs from 42 so the sample is new.")
    ap.add_argument("--pages", type=Path, default=OUT / "wiki_dump_pages.jsonl",
                    help="Local dump sample from sample_wiki_dump.py.")
    ap.add_argument("--skip-concept", action="store_true")
    ap.add_argument("--skip-neutral", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = random.Random(args.seed)

    orig_concept = next(e["sentences"] for e in json.loads((ROOT / "data/concept_sentences.json").read_text())
                        if e["concept"] == args.concept)
    orig_neutral = {e["sentence"] for e in json.loads((ROOT / "data/neutral_sentences.json").read_text())}

    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    if not args.skip_concept:
        canon, revid, paras = page_paragraphs(args.title or args.concept)
        pool = split_sentences(paras)
        sents = rng.sample(pool, min(args.n_concept, len(pool)))
        slug = re.sub(r"\W+", "_", args.concept).strip("_")
        (OUT / f"concept_sentences_{slug}.json").write_text(json.dumps([{
            "concept": args.concept,
            "source": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(canon.replace(" ", "_")),
            "revid": revid,
            "sentences": sents,
        }], ensure_ascii=False, indent=1))
        report.update({"concept": args.concept, "revid": revid, "seed": args.seed, "pool_size": len(pool),
                       "n_concept": len(sents),
                       "concept_overlap_with_original": len(set(sents) & set(orig_concept)),
                       "original_concept_found_in_pool": len(set(pool) & set(orig_concept))})
        print(f"[concept] '{canon}' rev {revid}: pool={len(pool)}, using {len(sents)}, "
              f"overlap with original={report['concept_overlap_with_original']}")

    if not args.skip_neutral:
        # Separate RNG so the neutral sample does not depend on whether the concept step ran.
        neutral = neutral_sentences(args.pages, args.n_neutral, random.Random(args.seed), exclude=orig_neutral)
        (OUT / "neutral_sentences.json").write_text(json.dumps(neutral, ensure_ascii=False, indent=1))
        report.update({"neutral_source": str(args.pages.relative_to(ROOT)), "neutral_seed": args.seed,
                       "n_neutral": len(neutral)})
        print(f"[neutral] {len(neutral)} sentences from {args.pages.name}")
    report_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
