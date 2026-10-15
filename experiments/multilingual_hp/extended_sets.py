"""Extended sentence sets for the fixed-Z refit (notebook cell 3.2.h).

concept: every sentence of the Harry Potter article in each language (pools/<lang>.json, already fetched and split),
         at most MAX_PER_LANG per language.
neutral: per language, as many sentences as that language's concept sentences, at most one per page, in two
         variants chosen by NEUTRAL_SOURCE in the notebook:
           native      pages of the language's own Wikipedia dump   -> extended_native.run
           translated  distinct English dump pages, Gemini-translated -> extended_translated.run

Output (experiments/multilingual_hp/data/extended/):
    concept_ext.json               [{"sentence", "lang", "title"}]
    neutral_ext_native.json        [{"sentence", "lang", "url", "title", "source": "native"}]
    neutral_ext_translated.json    [{"sentence", "lang", "en_sentence", "url", "title", "source"}]

Usage:
    python experiments/multilingual_hp/extended_sets.py --kind concept native translated
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from splitting import CONCEPT, DATA, LANGS, POOLS, SEED, concept_paths  # noqa: E402

EXT = DATA / "extended"
CONCEPT_EXT_PATH = EXT / "concept_ext.json"
NEUTRAL_EXT_NATIVE_PATH = EXT / "neutral_ext_native.json"
NEUTRAL_EXT_TRANSLATED_PATH = EXT / "neutral_ext_translated.json"
DUMP_EXT = DATA / "dump_ext"
MAX_PER_LANG = 1000


def build_concept(force: bool = False, concept: str = CONCEPT):
    paths = concept_paths(concept)
    out_path = paths["extended"] / "concept_ext.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    out = []
    for lang in LANGS:
        if not (paths["pools"] / f"{lang}.json").exists():
            continue
        pool = json.loads((paths["pools"] / f"{lang}.json").read_text())
        sents = pool["sentences"]
        if len(sents) > MAX_PER_LANG:
            sents = random.Random(SEED).sample(sents, MAX_PER_LANG)
        out += [{"sentence": s, "lang": lang, "title": pool["title"]} for s in sents]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def concept_counts(concept: str = CONCEPT) -> dict:
    """Number of extended concept sentences per language, i.e. how many neutral sentences each language needs."""
    counts = {lang: 0 for lang in LANGS}
    for r in build_concept(concept=concept):
        counts[r["lang"]] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", nargs="+", choices=["concept", "native", "translated"], required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--concept", default=CONCEPT)
    args = ap.parse_args()
    if "concept" in args.kind:
        build_concept(args.force, concept=args.concept)
        print("[concept_ext]", concept_counts(args.concept))
    if "native" in args.kind:
        from extended_native import run as run_native
        run_native(force=args.force)
    if "translated" in args.kind:
        from extended_translated import run as run_translated
        run_translated(force=args.force)


if __name__ == "__main__":
    main()
