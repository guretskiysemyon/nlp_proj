"""Full and sample sentence sets per concept (results/ml/data/full/, results/ml/data/sample/).

full/<slug>.json  {lang: {"concept": [{"text","source"[, "src_en"]}], "neutral": [{"text","source","url"}]}}
    concept  en: EMBER's 300 English concept sentences (data/concept_sentences.json), source "native".
             others: random.Random(SEED).sample of min(N_FULL, pool) native page sentences (pools/<slug>/<lang>.json);
             if the pool is short, NLLB translations (translations/nllb_<lang>.jsonl) of the queued English sources
             (translations/tasks.json, in queue order) that pass QA, are not duplicates of a sentence already in the
             set, and are not identical to their source, up to N_FULL; source "translated", src_en = English source.
    neutral  en: EMBER's data/neutral_sentences.json (all 300), source "native".
             others: as many as the concept side, from the neutral pages of neutral/partition_<lang>.json (page order):
             pass 1 takes per page its first sentence without a concept term (common.concept_terms of this
             concept), pass 2 the second, ... ("one or few per page").
sample/<slug>_N<N>.json  same layout, first N items of a Random(SEED) permutation of each list (so N20 ⊂ N30 ⊂ ...).
full/_summary.json  per concept and language: native / translated / neutral counts.

QA for a translation: target-script share of letters >= 0.6 (hi Devanagari, th Thai, others Latin);
character ratio target/source in [0.3, 3.0]; no degenerate repetition (a word making up > 30% of >= 10 words).

    python -m experiments.ml.data.build_sets
"""
from __future__ import annotations

import argparse
import collections
import random
import re

from experiments.ml.data.common import (FULL, LANGS, N_FULL, NEUTRAL, PAGES_PATH, POOLS, REPO, SAMPLE, SAMPLE_NS, SEED,
                                        concept_terms, concepts, dump_json, load_json, slug, term_matcher)
from experiments.ml.data.translate_nllb import TASKS_PATH, load_cache

SCRIPT = {"hi": re.compile(r"[ऀ-ॿ]"), "th": re.compile(r"[฀-๿]")}
LATIN = re.compile(r"[A-Za-zÀ-ɏ]")


def qa_ok(src: str, tgt: str, lang: str) -> bool:
    if not tgt or tgt.strip() == src.strip():
        return False
    letters = [c for c in tgt if c.isalpha()]
    pat = SCRIPT.get(lang, LATIN)
    if not letters or sum(1 for c in letters if pat.match(c)) / len(letters) < 0.6:
        return False
    if not 0.3 <= len(tgt) / max(1, len(src)) <= 3.0:
        return False
    words = tgt.split()
    if len(words) >= 10 and collections.Counter(words).most_common(1)[0][1] / len(words) > 0.3:
        return False
    return True


def concept_side(concept: str, lang: str, ember, tasks, caches, qa_stats):
    if lang == "en":
        return [{"text": s, "source": "native"} for s in ember[concept]]
    p = POOLS / slug(concept) / f"{lang}.json"
    pool = load_json(p)["sentences"] if p.exists() else []
    native = random.Random(SEED).sample(pool, min(N_FULL, len(pool)))
    out = [{"text": s, "source": "native"} for s in native]
    seen = set(native)
    for src in tasks.get(lang, {}).get(slug(concept), []):
        if len(out) >= N_FULL:
            break
        tgt = caches[lang].get(src)
        if tgt is None:
            qa_stats["missing"] += 1
            continue
        if not qa_ok(src, tgt, lang) or tgt in seen:
            qa_stats["failed"] += 1
            continue
        seen.add(tgt)
        out.append({"text": tgt, "source": "translated", "src_en": src})
    return out


def neutral_side(concept: str, lang: str, n: int, titles, partitions, ember_neutral):
    if lang == "en":
        return [{"text": x["sentence"], "source": "native", "url": x["url"]} for x in ember_neutral]
    has_term = term_matcher(concept_terms(concept, titles[concept]))
    pages = partitions[lang]["neutral_pages"]
    eligible = [[s for s in p["sentences"] if not has_term(s)] for p in pages]
    out, depth = [], 0
    while len(out) < n and any(depth < len(e) for e in eligible):
        for p, e in zip(pages, eligible):
            if depth < len(e):
                out.append({"text": e[depth], "source": "native", "url": p["url"]})
                if len(out) >= n:
                    break
        depth += 1
    if len(out) < n:
        print(f"WARNING [{concept}] {lang}: only {len(out)}/{n} neutral sentences", flush=True)
    return out


def subsample(full: dict, n: int):
    out = {}
    for lang, d in full.items():
        out[lang] = {}
        for side in ("concept", "neutral"):
            items = d[side]
            order = list(range(len(items)))
            random.Random(SEED).shuffle(order)
            out[lang][side] = [items[i] for i in order[:n]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=None)
    sel = ap.parse_args().concepts
    ember = {e["concept"]: e["sentences"] for e in load_json(REPO / "data" / "concept_sentences.json")}
    ember_neutral = load_json(REPO / "data" / "neutral_sentences.json")
    tasks = load_json(TASKS_PATH) if TASKS_PATH.exists() else {}
    caches = {l: load_cache(l) for l in LANGS if l != "en"}
    pages = load_json(PAGES_PATH)
    titles = {c: {l: v["title"] for l, v in d.items()} for c, d in pages.items()}
    partitions = {l: load_json(NEUTRAL / f"partition_{l}.json") for l in LANGS if l != "en"}
    summary_path = FULL / "_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    for c in (sel or concepts()):
        full, summary[c] = {}, {}
        for lang in LANGS:
            qa_stats = {"failed": 0, "missing": 0}
            cs = concept_side(c, lang, ember, tasks, caches, qa_stats)
            ns = neutral_side(c, lang, len(cs), titles, partitions, ember_neutral)
            full[lang] = {"concept": cs, "neutral": ns}
            n_tr = sum(1 for x in cs if x["source"] == "translated")
            summary[c][lang] = {"native": len(cs) - n_tr, "translated": n_tr, "neutral": len(ns),
                                "page_exists": pages[c][lang]["exists"], "n_words": pages[c][lang]["n_words"],
                                "qa_failed": qa_stats["failed"], "missing_translation": qa_stats["missing"]}
            if len(cs) < N_FULL:
                print(f"WARNING [{c}] {lang}: concept side {len(cs)}/{N_FULL} ({qa_stats})", flush=True)
        # disjointness of concept and neutral within a language
        for lang in LANGS:
            inter = {x["text"] for x in full[lang]["concept"]} & {x["text"] for x in full[lang]["neutral"]}
            assert not inter, (c, lang, list(inter)[:3])
        dump_json(FULL / f"{slug(c)}.json", full)
        for n in SAMPLE_NS:
            dump_json(SAMPLE / f"{slug(c)}_N{n}.json", subsample(full, n))
        print(f"[{c}] " + " ".join(f"{l}:{s['native']}+{s['translated']}t/{s['neutral']}n"
                                   for l, s in summary[c].items()), flush=True)
    dump_json(summary_path, {c: summary[c] for c in concepts() if c in summary})


if __name__ == "__main__":
    main()
