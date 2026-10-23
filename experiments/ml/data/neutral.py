"""Native neutral material per language: page partition into neutral / held-out / background pages.

Pages (native Wikipedia, Hugging Face dump wikimedia/wikipedia 20231101.<lang>):
    en      resample_sentences/data/wiki_dump_pages.jsonl + results/ml/data/dump/wiki_dump_pages_en.jsonl
            (pages of EMBER's data/neutral_sentences.json are left out: those 300 sentences are the English neutral set)
    others  multilingual_hp/data/dump/ + dump_ext/ + results/ml/data/dump/  (deduplicated by url)
Pages titled like any concept page (any language) are dropped.

Per page: multilingual_hp.native_neutral.prose_paragraphs -> splitting.split_sentences -> looks_like_prose, minus
sentences whose template signature (first/last 3 words) was already seen in that language (so near-duplicates such
as "X is a village in ..." do not repeat). Pages without such a sentence are dropped. The usable pages are shuffled
(random.Random(SEED)), each page's sentences too, and split in page order into
    neutral     the first N_NEUTRAL_PAGES pages (not for en): concept-specific neutral sets are drawn from them
                (build_sets.py: per concept, one sentence per page that has no concept term, a second pass if short)
    heldout     the next pages: one sentence each with no term of any concept, until N_HELDOUT
    background  all remaining pages, round robin (1st sentence of every page, then 2nd, ...) until N_BACKGROUND,
                minus any sentence that occurs in a concept page pool
Neutral, held-out and background sentences therefore come from disjoint pages.

Output: neutral/partition_<lang>.json  {"lang","neutral_pages":[{"url","title","sentences"}],"heldout":[...],
        "background":[...], "stats"}; neutral_heldout_<lang>.json; background_ml.json {lang: [{"text","url","title"}]}

    python -m experiments.ml.data.neutral [--lang ...]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from experiments.ml.data.common import (BACKGROUND_PATH, DUMP, LANGS, N_BACKGROUND, N_HELDOUT, NEUTRAL, OUT,
                                        PAGES_PATH, POOLS, REPO, SEED, concept_terms, concepts, dump_json, load_json,
                                        slug, term_matcher)
from native_neutral import MIN_PAGE_CHARS, looks_like_prose, prose_paragraphs, signature  # noqa: E402
from splitting import EN_DUMP_PAGES, split_sentences  # noqa: E402

HP_DATA = REPO / "experiments" / "multilingual_hp" / "data"
N_NEUTRAL_PAGES = 450


def page_files(lang: str):
    if lang == "en":
        return [EN_DUMP_PAGES, DUMP / "wiki_dump_pages_en.jsonl"]
    return [HP_DATA / "dump" / f"wiki_dump_pages_{lang}.jsonl", HP_DATA / "dump_ext" / f"wiki_dump_pages_{lang}.jsonl",
            DUMP / f"wiki_dump_pages_{lang}.jsonl"]


def load_pages(lang: str):
    seen, pages = set(), []
    for f in page_files(lang):
        if not f.exists():
            print(f"[{lang}] WARNING missing {f}", flush=True)
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    p = json.loads(line)
                    if p["url"] not in seen:
                        seen.add(p["url"])
                        pages.append(p)
    return pages


def all_titles():
    pages = load_json(PAGES_PATH)
    return {c: {l: v["title"] for l, v in d.items()} for c, d in pages.items()}


def concept_pool_texts(lang: str):
    out = set()
    for c in concepts():
        p = POOLS / slug(c) / f"{lang}.json"
        if p.exists():
            out.update(load_json(p)["sentences"])
    if lang == "en":
        for e in load_json(REPO / "data" / "concept_sentences.json"):
            out.update(e["sentences"])
    return out


def partition(lang: str):
    titles = all_titles()
    concept_page_titles = {t.casefold() for d in titles.values() for t in d.values() if t}
    any_term = term_matcher(sorted({t for c, d in titles.items() for t in concept_terms(c, d)}))
    excluded_urls = set()
    if lang == "en":
        excluded_urls = {x["url"] for x in load_json(REPO / "data" / "neutral_sentences.json")}
    pages = load_pages(lang)
    rng = random.Random(SEED)
    rng.shuffle(pages)
    stats = {"pages": len(pages), "short": 0, "concept_title": 0, "ember_neutral_page": 0, "no_sentence": 0}
    used_sigs, usable = set(), []
    for p in pages:
        if p["url"] in excluded_urls:
            stats["ember_neutral_page"] += 1
            continue
        if p["title"].casefold() in concept_page_titles:
            stats["concept_title"] += 1
            continue
        text = p.get("text") or ""
        if len(text) < MIN_PAGE_CHARS:
            stats["short"] += 1
            continue
        sents = []
        for s in split_sentences(prose_paragraphs(text, lang), lang):
            if looks_like_prose(s, lang) and signature(s) not in used_sigs:
                used_sigs.add(signature(s))
                sents.append(s)
        if not sents:
            stats["no_sentence"] += 1
            continue
        rng.shuffle(sents)
        usable.append({"url": p["url"], "title": p["title"], "sentences": sents})
    stats["usable_pages"] = len(usable)
    stats["usable_sentences"] = sum(len(p["sentences"]) for p in usable)

    n_neu = 0 if lang == "en" else N_NEUTRAL_PAGES
    neutral_pages, rest = usable[:n_neu], usable[n_neu:]
    heldout, i = [], 0
    while len(heldout) < N_HELDOUT and i < len(rest):
        p = rest[i]
        i += 1
        s = next((s for s in p["sentences"] if not any_term(s)), None)
        if s is not None:
            heldout.append({"text": s, "url": p["url"], "title": p["title"]})
    bg_pages = rest[i:]
    concept_texts = concept_pool_texts(lang)
    background, depth = [], 0
    while len(background) < N_BACKGROUND and any(depth < len(p["sentences"]) for p in bg_pages):
        for p in bg_pages:
            if depth < len(p["sentences"]) and p["sentences"][depth] not in concept_texts:
                background.append({"text": p["sentences"][depth], "url": p["url"], "title": p["title"]})
                if len(background) >= N_BACKGROUND:
                    break
        depth += 1
    stats.update(neutral_pages=len(neutral_pages), neutral_page_sentences=sum(len(p["sentences"]) for p in neutral_pages),
                 heldout=len(heldout), background=len(background), background_pages=len(bg_pages),
                 background_max_per_page=depth)
    if len(heldout) < N_HELDOUT or len(background) < N_BACKGROUND:
        print(f"[{lang}] WARNING short: {stats}", flush=True)
    print(f"[{lang}] {stats}", flush=True)
    dump_json(NEUTRAL / f"partition_{lang}.json", {"lang": lang, "seed": SEED, "neutral_pages": neutral_pages,
                                                    "heldout": heldout, "background": background, "stats": stats})
    dump_json(OUT / f"neutral_heldout_{lang}.json", heldout)
    return background


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", nargs="+", default=list(LANGS))
    args = ap.parse_args()
    for lang in args.lang:
        partition(lang)
    bg = {}
    for lang in LANGS:
        p = NEUTRAL / f"partition_{lang}.json"
        if p.exists():
            bg[lang] = load_json(p)["background"]
    dump_json(BACKGROUND_PATH, bg)
    print(f"background: { {l: len(v) for l, v in bg.items()} } -> {BACKGROUND_PATH}")


if __name__ == "__main__":
    main()
