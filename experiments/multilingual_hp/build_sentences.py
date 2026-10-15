"""Build the multilingual Harry Potter concept sentences (N_PER_LANG per language).

Stages (run any subset, in order):
    fetch      one langlinks call on en:"Harry Potter", then one parse call per language -> data/raw/<lang>.json
               (a language whose raw file exists is never re-fetched unless --force)
    split      raw HTML -> <p> paragraphs -> split_sentences -> data/pools/<lang>.json
    concept    seeded sample of N_PER_LANG per language -> data/concept_sentences.json (ConceptDataset format)
    translate  translate_neutral.run(force=...)
    native     native_neutral.run(force=...)

    python experiments/multilingual_hp/build_sentences.py --stage fetch split concept
"""
import argparse
import json
import random
import sys
import urllib.parse
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "multilingual_fit"))
sys.path.insert(0, str(HERE.parent / "resample_sentences"))

import splitting  # noqa: E402
from splitting import CONCEPT, CONCEPT_PATH, CONCEPT_TITLES, LANGS, N_PER_LANG, POOLS, RAW, SEED, split_sentences  # noqa: E402
from fetch_wiki_sentences import api  # noqa: E402
from fetch_en_sentences import _ParagraphText  # noqa: E402

STAGES = ["fetch", "split", "concept", "translate", "native", "extended"]


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def fetch(force: bool):
    todo = [l for l in LANGS if force or not (RAW / f"{l}.json").exists()]
    if not todo:
        print("fetch: all raw files exist, no API calls")
        return
    titles = {}
    for en_title in CONCEPT_TITLES.get(CONCEPT, [CONCEPT]):  # first English title with interlanguage links wins
        titles = {"en": en_title}
        if any(l != "en" for l in todo):
            d = api("en", action="query", titles=en_title, prop="langlinks", lllimit="max", redirects=1)
            for ll in d["query"]["pages"][0].get("langlinks", []):
                titles[ll["lang"]] = ll["title"]
        if all(l in titles for l in todo):
            break
    for lang in todo:
        if lang not in titles:
            print(f"fetch: no {lang} langlink for {CONCEPT!r}, skipping")
            continue
        p = api(lang, action="parse", page=titles[lang], prop="text|revid", redirects=1)["parse"]
        title = p["title"]
        _dump(RAW / f"{lang}.json", {
            "lang": lang, "title": title, "revid": p["revid"],
            "source": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "html": p["text"],
        })
        print(f"fetch: {lang} {title!r} rev {p['revid']} ({len(p['text'])} chars html)")


def split():
    for lang in LANGS:
        if not (RAW / f"{lang}.json").exists():
            print(f"split: no raw page for {lang}, skipping")
            continue
        raw = json.loads((RAW / f"{lang}.json").read_text(encoding="utf-8"))
        parser = _ParagraphText()
        parser.feed(raw["html"])
        sents = split_sentences(parser.paragraphs, lang)
        _dump(POOLS / f"{lang}.json", {"lang": lang, "title": raw["title"], "revid": raw["revid"], "sentences": sents})
        print(f"split: {lang} {len(parser.paragraphs)} paragraphs -> {len(sents)} sentences")


def concept():
    sentences, langs, sources = [], [], {}
    for lang in LANGS:
        if not (POOLS / f"{lang}.json").exists():
            print(f"concept: WARNING no {lang} article, language skipped")
            continue
        raw = json.loads((RAW / f"{lang}.json").read_text(encoding="utf-8"))
        pool = json.loads((POOLS / f"{lang}.json").read_text(encoding="utf-8"))["sentences"]
        n = min(N_PER_LANG, len(pool))
        if n < N_PER_LANG:
            print(f"concept: WARNING {lang} pool has only {len(pool)} sentences (< {N_PER_LANG})")
        picked = random.Random(SEED).sample(pool, n)
        sentences += picked
        langs += [lang] * n
        sources[lang] = {"title": raw["title"], "revid": raw["revid"], "source": raw["source"],
                         "pool_size": len(pool), "n_used": n}
    _dump(CONCEPT_PATH, [{"concept": CONCEPT, "sentences": sentences, "langs": langs, "sources": sources, "seed": SEED}])
    print(f"concept: wrote {len(sentences)} sentences -> {CONCEPT_PATH}")


def _use_concept(concept: str):
    """Point the module-level paths at one concept's data directory."""
    global CONCEPT, RAW, POOLS, CONCEPT_PATH
    paths = splitting.concept_paths(concept)
    CONCEPT, RAW, POOLS, CONCEPT_PATH = concept, paths["raw"], paths["pools"], paths["concept"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", choices=STAGES, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--concept", default=CONCEPT, help="Concept name as in data/concept_sentences.json.")
    args = ap.parse_args()
    _use_concept(args.concept)
    for stage in STAGES:
        if stage not in args.stage:
            continue
        if stage == "fetch":
            fetch(args.force)
        elif stage == "split":
            split()
        elif stage == "concept":
            concept()
        elif stage == "translate":
            from translate_neutral import run as run_translate
            run_translate(force=args.force)
        elif stage == "native":
            from native_neutral import run as run_native
            run_native(force=args.force)
        elif stage == "extended":  # concept_ext + both extended neutral sets (cell 3.2.h)
            from extended_sets import build_concept
            from extended_native import run as run_ext_native
            from extended_translated import run as run_ext_translated
            build_concept(force=args.force)
            run_ext_native(force=args.force)
            run_ext_translated(force=args.force)


if __name__ == "__main__":
    main()
