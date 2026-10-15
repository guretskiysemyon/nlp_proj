#!/usr/bin/env python3
"""Translated extended neutral set (notebook cell 3.2.h, NEUTRAL_SOURCE = "translated").

Per language, n_lang = extended_sets.concept_counts()[lang] English sentences, one per page of the local 1,000-page
English dump sample (distinct pages within a language; languages may share pages), excluding the English sentences
of the base translated set. en stays English; the other languages are translated by Gemini in batches of 40,
run in parallel. Each batch is cached in data/extended/translations/<lang>_<batch>.json and reused while its
English sentences match.

Output: data/extended/neutral_ext_translated.json
    [{"sentence", "lang", "en_sentence", "url", "title", "source": "translated"|"original"}]

Usage:
    python experiments/multilingual_hp/extended_translated.py [--force] [--workers 8]
"""
import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "resample_sentences"))

from extended_sets import EXT, NEUTRAL_EXT_TRANSLATED_PATH, concept_counts  # noqa: E402
from fetch_en_sentences import neutral_sentences  # noqa: E402
from splitting import EN_DUMP_PAGES, LANGS, NEUTRAL_TRANSLATED_PATH, SEED  # noqa: E402
import translate_neutral  # noqa: E402

BATCH = 40
TRANS_DIR = EXT / "translations"


def english_groups():
    base = json.loads(NEUTRAL_TRANSLATED_PATH.read_text())
    exclude = {r["en_sentence"] for r in base} | {r["sentence"] for r in base if r["lang"] == "en"}
    counts = concept_counts()
    return {lang: neutral_sentences(EN_DUMP_PAGES, counts[lang], random.Random(SEED + i), exclude)
            for i, lang in enumerate(LANGS)}


def _batch(lang, b, group, force, trans_dir=TRANS_DIR):
    path = trans_dir / f"{lang}_{b:03d}.json"
    en = [g["sentence"] for g in group]
    if path.exists() and not force:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if [c["en_sentence"] for c in cached] == en:
            return lang, b, cached, True
    for attempt in range(1, 4):  # translate() gives up after MAX_ROUNDS; retry the batch instead of failing the run
        try:
            trans = translate_neutral.translate(en, lang)
            break
        except RuntimeError as e:
            err = str(e)
            print(f"  {lang} batch {b}: attempt {attempt}/3 failed: {err}", flush=True)
    else:
        # Sentences Gemini keeps echoing are usually titles of works, which stay untranslated: keep the English for
        # those ids and translate the rest of the batch.
        keep = {int(i) for i in re.search(r"ids \[([\d, ]+)\]", err).group(1).split(",")}
        rest = [i for i in range(len(en)) if i not in keep]
        part = iter(translate_neutral.translate([en[i] for i in rest], lang))
        trans = [en[i] if i in keep else next(part) for i in range(len(en))]
        print(f"  {lang} batch {b}: kept English for ids {sorted(keep)}", flush=True)
    rows = [{"id": i, "en_sentence": g["sentence"], "translation": t, "url": g["url"], "title": g["title"]}
            for i, (g, t) in enumerate(zip(group, trans))]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return lang, b, rows, False


def run(force: bool = False, workers: int = 8):
    groups = english_groups()
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    translate_neutral._model()  # create the shared client once, before the threads use it
    jobs = [(lang, b, groups[lang][b * BATCH:(b + 1) * BATCH])
            for lang in LANGS if lang != "en" for b in range((len(groups[lang]) + BATCH - 1) // BATCH)]
    done, n_called, failed = {}, 0, []
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(_batch, lang, b, group, force) for lang, b, group in jobs]
        for fut in as_completed(futures):
            lang, b, rows, cached = fut.result()
            if rows is None:
                failed.append((lang, b))
                continue
            done[(lang, b)] = rows
            n_called += not cached
            print(f"[{len(done)}/{len(jobs)}] {lang} batch {b}: {'cached' if cached else 'translated'}", flush=True)

    if failed:
        raise RuntimeError(f"batches still untranslated (re-run to retry only these): {sorted(failed)}")

    out = []
    for lang in LANGS:
        if lang == "en":
            out += [{"sentence": g["sentence"], "lang": lang, "en_sentence": g["sentence"], "url": g["url"],
                     "title": g["title"], "source": "original"} for g in groups[lang]]
            continue
        for b in range((len(groups[lang]) + BATCH - 1) // BATCH):
            out += [{"sentence": r["translation"], "lang": lang, "en_sentence": r["en_sentence"], "url": r["url"],
                     "title": r["title"], "source": "translated"} for r in done[(lang, b)]]
    NEUTRAL_EXT_TRANSLATED_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(out)} sentences ({n_called} Gemini batches translated) -> {NEUTRAL_EXT_TRANSLATED_PATH}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(force=args.force, workers=args.workers)
