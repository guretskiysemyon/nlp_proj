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


# ---------------------------------------------------------------- capped sets: up to per_lang sentences per language
# concept: per language, the 40 V* concept sentences (always kept, so V*_ext contains V*), then a seeded random sample
#          of the rest of the page up to min(per_lang, |page|); with topup="translated", filled up to per_lang with
#          topup_translated.py's Gemini translations of the other languages' pages. Rows carry "source".
# neutral: per language, the V* neutral sentences of replicate `rep` plus the first (n_lang - |base|) sentences of that
#          replicate's shared translated pool (data/neutral_rep{rep}/ext_pool_translated.json), n_lang = concept count.

def _suffix(per_lang: int, topup) -> str:
    return f"_n{per_lang}" + ("_tr" if topup == "translated" else "")


def ext_paths(concept: str = CONCEPT, rep: int = 0, per_lang=None, topup=None) -> dict:
    """Concept / neutral extended-set files. per_lang None: the original whole-page sets (concept_ext.json and
    neutral_ext_translated_rep{rep}.json)."""
    ext = concept_paths(concept)["extended"]
    if per_lang is None:
        from splitting import neutral_paths
        return {"concept": ext / "concept_ext.json", "neutral": neutral_paths(concept, rep)["ext"]}
    sfx = _suffix(per_lang, topup)
    return {"concept": ext / f"concept_ext{sfx}.json", "neutral": ext / f"neutral_ext_translated_rep{rep}{sfx}.json"}


def build_concept_n(concept: str = CONCEPT, per_lang: int = 300, topup=None, force: bool = False):
    paths = concept_paths(concept)
    out_path = ext_paths(concept, per_lang=per_lang, topup=topup)["concept"]
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    entry = json.loads(paths["concept"].read_text())[0]
    vstar = {}
    for s, lang in zip(entry["sentences"], entry["langs"]):
        vstar.setdefault(lang, []).append(s)
    tr = []
    if topup == "translated":
        tr_path = paths["extended"] / "topup_translated.json"
        assert tr_path.exists(), f"{tr_path} missing: run experiments/multilingual_hp/topup_translated.py first"
        tr = json.loads(tr_path.read_text())
    out = []
    for lang in LANGS:
        pool_path = paths["pools"] / f"{lang}.json"
        if not pool_path.exists():
            continue
        pool = json.loads(pool_path.read_text())
        keep = list(dict.fromkeys(vstar.get(lang, [])))
        rest = [s for s in pool["sentences"] if s not in set(keep)]
        n_native = min(per_lang, len(keep) + len(rest))
        keep += random.Random(f"{SEED}-n{per_lang}-{lang}").sample(rest, max(0, n_native - len(keep)))
        out += [{"sentence": s, "lang": lang, "title": pool["title"], "source": "native"} for s in keep]
        if topup == "translated" and len(keep) < per_lang:
            add = [r for r in tr if r["lang"] == lang][:per_lang - len(keep)]
            out += [{**r, "source": "translated"} for r in add]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def build_neutral_n(concept: str = CONCEPT, rep: int = 0, per_lang: int = 300, topup=None, force: bool = False):
    from splitting import neutral_paths
    out_path = ext_paths(concept, rep, per_lang, topup)["neutral"]
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    counts = {lang: 0 for lang in LANGS}
    for r in build_concept_n(concept, per_lang, topup):
        counts[r["lang"]] += 1
    base = json.loads(neutral_paths(concept, rep)["base"].read_text())
    pool = json.loads((DATA / f"neutral_rep{rep}" / "ext_pool_translated.json").read_text())
    out = []
    for lang in LANGS:
        rows = [r for r in base if r["lang"] == lang][:counts[lang]]
        seen = {r["sentence"] for r in rows}
        for r in pool[lang]:          # pool order; skip translations that repeat a sentence already taken
            if len(rows) >= counts[lang]:
                break
            if r["sentence"] not in seen:
                rows.append(r)
                seen.add(r["sentence"])
        if len(rows) < counts[lang]:
            print(f"  warning: neutral pool rep{rep}/{lang} gives {len(rows)} of {counts[lang]} distinct sentences "
                  f"(neutral_replicates.py --min-pool to grow it)")
        out += rows
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", nargs="+", choices=["concept", "native", "translated", "capped"], required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--concept", default=CONCEPT)
    ap.add_argument("--per-lang", type=int, default=300, help="capped: sentences per language")
    ap.add_argument("--topup", choices=["translated", "none"], default="translated", help="capped: fill short pages")
    ap.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2], help="capped: neutral replicates")
    args = ap.parse_args()
    if "capped" in args.kind:
        topup = None if args.topup == "none" else args.topup
        rows = build_concept_n(args.concept, args.per_lang, topup, args.force)
        print("[concept capped]", {l: sum(r["lang"] == l for r in rows) for l in LANGS},
              "translated:", sum(r["source"] == "translated" for r in rows))
        for rep in args.reps:
            n = build_neutral_n(args.concept, rep, args.per_lang, topup, args.force)
            print(f"[neutral capped rep{rep}]", {l: sum(r["lang"] == l for r in n) for l in LANGS})
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
