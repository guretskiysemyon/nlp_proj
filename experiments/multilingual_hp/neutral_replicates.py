#!/usr/bin/env python3
"""Shared translated neutral sets, 3 replicates, for every concept in splitting.CONCEPT_TITLES.

Replicate r (0, 1, 2) has its own English draw from the local 1,000-page English dump sample (seed SEED for r = 0, which
reproduces the original files, else SEED + 100 * r). Within a replicate each language uses at most one sentence per page;
replicates may share pages.

    base  320 sentences, 40 per language; en stays English, the rest is Gemini-translated.
          r = 0 is the original data/neutral_sentences_translated.json (not rebuilt).
    ext   per language, the largest extended concept count over all concepts (N_max), drawn in a fixed seeded order
          and translated in cached batches of 40. Each concept takes the first n_lang of that order, so one translation
          serves every concept, and r = 0 reuses the batches of the Harry Potter run.

Output:
    data/neutral_rep{r}/neutral_sentences_translated.json      base, r = 1, 2
    data/neutral_rep{r}/ext_pool_translated.json               ext pool (N_max per language)
    <concept extended dir>/neutral_ext_translated_rep{r}.json  per concept, first n_lang per language
    data/neutral_replicates_report.json

Usage:
    python experiments/multilingual_hp/neutral_replicates.py [--reps 0 1 2] [--workers 8]
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "resample_sentences"))

from extended_sets import concept_counts  # noqa: E402
from extended_translated import BATCH, TRANS_DIR, _batch  # noqa: E402
from fetch_en_sentences import neutral_sentences  # noqa: E402
from splitting import CONCEPT_TITLES, DATA, EN_DUMP_PAGES, LANGS, N_PER_LANG, SEED, neutral_paths  # noqa: E402
import translate_neutral  # noqa: E402


def rep_dir(r: int) -> Path:
    return DATA / f"neutral_rep{r}"


def rep_seed(r: int) -> int:
    return SEED if r == 0 else SEED + 100 * r


def _translate_jobs(jobs, workers):
    """jobs: (key, lang, batch, group, trans_dir) -> {key: rows}; batches are cached, so a re-run only redoes failures."""
    translate_neutral._model()  # shared client, created before the threads use it
    out, failed, called = {}, [], 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(_batch, lang, b, group, False, tdir): key for key, lang, b, group, tdir in jobs}
        for fut in as_completed(futs):
            try:
                _, _, rows, cached = fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append((futs[fut], repr(e)[:200]))
                continue
            out[futs[fut]] = rows
            called += not cached
            print(f"  [{len(out)}/{len(jobs)}] {futs[fut]}: {'cached' if cached else 'translated'}", flush=True)
    if failed:
        raise RuntimeError(f"untranslated batches (re-run to retry only these): {failed}")
    return out, called


def build_base(r: int, workers: int):
    if r == 0:
        return json.loads(neutral_paths(rep=0)["base"].read_text()), 0
    path = neutral_paths(rep=r)["base"]
    pool = neutral_sentences(EN_DUMP_PAGES, len(LANGS) * N_PER_LANG, random.Random(rep_seed(r)), set())
    groups = {lang: pool[i * N_PER_LANG:(i + 1) * N_PER_LANG] for i, lang in enumerate(LANGS)}
    tdir = rep_dir(r) / "translations_base"
    tdir.mkdir(parents=True, exist_ok=True)
    done, called = _translate_jobs([(lang, lang, 0, groups[lang], tdir) for lang in LANGS if lang != "en"], workers)
    rows = []
    for lang in LANGS:
        if lang == "en":
            rows += [{"sentence": g["sentence"], "lang": lang, "en_sentence": g["sentence"], "url": g["url"],
                      "title": g["title"], "source": "original"} for g in groups[lang]]
        else:
            rows += [{"sentence": t["translation"], "lang": lang, "en_sentence": t["en_sentence"], "url": t["url"],
                      "title": t["title"], "source": "translated"} for t in done[lang]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows, called


def build_ext(r: int, base_rows, n_max: dict, workers: int):
    exclude = {x["en_sentence"] for x in base_rows} | {x["sentence"] for x in base_rows if x["lang"] == "en"}
    seed = rep_seed(r)
    groups = {lang: neutral_sentences(EN_DUMP_PAGES, n_max[lang], random.Random(seed + i), exclude)
              for i, lang in enumerate(LANGS)}
    tdir = TRANS_DIR if r == 0 else rep_dir(r) / "translations_ext"   # r = 0 reuses the Harry Potter batches
    tdir.mkdir(parents=True, exist_ok=True)
    jobs = [((lang, b), lang, b, groups[lang][b * BATCH:(b + 1) * BATCH], tdir)
            for lang in LANGS if lang != "en" for b in range((n_max[lang] + BATCH - 1) // BATCH)]
    done, called = _translate_jobs(jobs, workers)
    pool = {}
    for lang in LANGS:
        if lang == "en":
            pool[lang] = [{"sentence": g["sentence"], "lang": lang, "en_sentence": g["sentence"], "url": g["url"],
                           "title": g["title"], "source": "original"} for g in groups[lang]]
        else:
            pool[lang] = [{"sentence": t["translation"], "lang": lang, "en_sentence": t["en_sentence"], "url": t["url"],
                           "title": t["title"], "source": "translated"}
                          for b in range((n_max[lang] + BATCH - 1) // BATCH) for t in done[(lang, b)]]
    rep_dir(r).mkdir(parents=True, exist_ok=True)
    (rep_dir(r) / "ext_pool_translated.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    return pool, called


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-pool", type=int, default=0,
                    help="ext pool size per language at least this (the capped sets of extended_sets.build_neutral_n "
                         "need per_lang - 40); the pool order is prefix-stable, so only the tail is translated")
    args = ap.parse_args()

    counts = {c: concept_counts(c) for c in CONCEPT_TITLES}
    n_max = {lang: max(args.min_pool, *(cnt[lang] for cnt in counts.values())) for lang in LANGS}
    print("N_max per language:", n_max)
    report_path = DATA / "neutral_replicates_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for r in args.reps:
        print(f"=== replicate {r} (seed {rep_seed(r)})")
        base_rows, c_base = build_base(r, args.workers)
        pool, c_ext = build_ext(r, base_rows, n_max, args.workers)
        for concept, cnt in counts.items():
            rows = [x for lang in LANGS for x in pool[lang][:cnt[lang]]]
            out = neutral_paths(concept, r)["ext"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        report[str(r)] = {"seed": rep_seed(r), "n_max": n_max, "gemini_batches_base": c_base,
                          "gemini_batches_ext": c_ext, "base_n": len(base_rows)}
        print(f"replicate {r}: base {len(base_rows)} sentences, ext pool {sum(map(len, pool.values()))}, "
              f"Gemini batches translated: base {c_base}, ext {c_ext}")
    report["concept_counts"] = counts
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
