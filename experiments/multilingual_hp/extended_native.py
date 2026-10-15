#!/usr/bin/env python3
"""Extended native neutral set: for each language, exactly concept_counts()[lang] sentences (as many as that
language's extended concept sentences), at most one per Wikipedia page, pages from that language's own Wikipedia.

Pages (Hugging Face dump wikimedia/wikipedia, config 20231101.<lang>, via sample_wiki_dump.sample_dump):
    en      reuses splitting.EN_DUMP_PAGES (the 1k-page English sample, no download)
    others  DUMP_EXT/wiki_dump_pages_<lang>.jsonl (DUMP_N_GROUPS row groups x DUMP_PER_GROUP pages, seed SEED+1)
    top-up  if a language comes up short, DUMP_EXT/wiki_dump_pages_<lang>_topup<k>.jsonl (seed SEED+1+k) is
            sampled and its (shuffled) pages are appended after the existing ones, so earlier picks stay stable.
All samples are cached and reused unless --force (which re-downloads everything but EN_DUMP_PAGES).

Selection mirrors native_neutral.pick_lang (random.Random(SEED) shuffle, skip pages < MIN_PAGE_CHARS, prose
paragraphs, splitting.split_sentences, template-signature dedupe, one random sentence per page), plus:
    - pages titled like the language's Harry Potter article (or containing an HP title) are skipped
    - sentences that are in the extended concept set are dropped

Output: NEUTRAL_EXT_NATIVE_PATH  [{"sentence", "lang", "url", "title", "source": "native"}] in LANGS order
        EXT/neutral_ext_native_report.json  per-language page / filter statistics

Usage:
    python experiments/multilingual_hp/extended_native.py [--force]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "resample_sentences"))

from extended_sets import DUMP_EXT, EXT, NEUTRAL_EXT_NATIVE_PATH, build_concept, concept_counts  # noqa: E402
from native_neutral import MIN_PAGE_CHARS, prose_paragraphs, signature  # noqa: E402
from sample_wiki_dump import sample_dump  # noqa: E402
from splitting import EN_DUMP_PAGES, LANGS, SEED, split_sentences  # noqa: E402

DUMP_N_GROUPS, DUMP_PER_GROUP = 60, 25
MAX_TOPUPS = 3
SAMPLE_RETRIES = 8  # Hugging Face answers bursts of range reads with 429; back off and resample (deterministic)
REPORT_PATH = EXT / "neutral_ext_native_report.json"
HP_MARKERS = ("Harry Potter", "हैरी पॉटर", "แฮร์รี่ พอตเตอร์")


def page_file(lang: str, k: int = 0) -> Path:
    """k = 0: the main sample (EN_DUMP_PAGES for en); k >= 1: the k-th top-up sample."""
    if k == 0:
        return EN_DUMP_PAGES if lang == "en" else DUMP_EXT / f"wiki_dump_pages_{lang}.jsonl"
    return DUMP_EXT / f"wiki_dump_pages_{lang}_topup{k}.jsonl"


def sample_with_retry(config: str, path: Path, seed: int):
    for attempt in range(SAMPLE_RETRIES):
        try:
            return sample_dump(config, path, n_groups=DUMP_N_GROUPS, per_group=DUMP_PER_GROUP, seed=seed)
        except Exception as e:  # noqa: BLE001 - network errors (429, timeouts) from fsspec/aiohttp/urllib
            if attempt == SAMPLE_RETRIES - 1:
                raise
            wait = min(60 * 2 ** attempt, 900)
            print(f"[retry] {config}: {type(e).__name__}: {str(e)[:120]} -> sleeping {wait}s", flush=True)
            time.sleep(wait)


def load_file(lang: str, k: int, force: bool, downloads: list):
    path = page_file(lang, k)
    if not (lang == "en" and k == 0) and (force or not path.exists()):
        sample_with_retry(f"20231101.{lang}", path, seed=SEED + 1 + k)
        downloads.append(str(path))
    with path.open(encoding="utf-8") as f:
        pages = [json.loads(line) for line in f if line.strip()]
    random.Random(SEED + k).shuffle(pages)
    return pages


def is_hp_page(title: str, hp_title: str) -> bool:
    return title == hp_title or any(m in title for m in HP_MARKERS)


def pick(lang: str, n: int, pages, concept_sents: set, hp_title: str, state: dict):
    """Continue picking from `pages` (already ordered) into state until n sentences; returns state."""
    rng, st = state["rng"], state["stats"]
    for page in pages:
        if len(state["out"]) >= n:
            break
        st["pages_available"] += 1
        if page["url"] in state["seen_urls"]:
            st["duplicate_page"] += 1
            continue
        state["seen_urls"].add(page["url"])
        if is_hp_page(page["title"], hp_title):
            st["hp_page"] += 1
            continue
        text = page.get("text") or ""
        if len(text) < MIN_PAGE_CHARS:
            st["short"] += 1
            continue
        sents = split_sentences(prose_paragraphs(text, lang), lang)
        n_split = len(sents)
        sents = [s for s in sents if s not in concept_sents and s not in state["used_sents"]]
        st["concept_or_dup_sentence_drops"] += n_split - len(sents)
        n_before = len(sents)
        sents = [s for s in sents if signature(s) not in state["used_sigs"]]
        st["template_sentence_drops"] += n_before - len(sents)
        if not sents:
            if n_before:
                st["template_page_drops"] += 1
            else:
                st["no_sentence"] += 1
            continue
        st["pages_passing_filter"] += 1
        s = rng.choice(sents)
        state["used_sigs"].add(signature(s))
        state["used_sents"].add(s)
        state["out"].append({"sentence": s, "lang": lang, "url": page["url"], "title": page["title"],
                             "source": "native"})
    return state


def run(force: bool = False):
    DUMP_EXT.mkdir(parents=True, exist_ok=True)
    concept = build_concept()
    counts = concept_counts()
    concept_sents = {r["sentence"] for r in concept}
    hp_titles = {r["lang"]: r["title"] for r in concept}
    rows, report = [], {}
    for lang in LANGS:
        n = counts[lang]
        downloads = []
        state = {"rng": random.Random(SEED), "out": [], "seen_urls": set(), "used_sigs": set(), "used_sents": set(),
                 "stats": dict(pages_available=0, duplicate_page=0, hp_page=0, short=0, no_sentence=0,
                               template_page_drops=0, template_sentence_drops=0, concept_or_dup_sentence_drops=0,
                               pages_passing_filter=0)}
        files = []
        for k in range(MAX_TOPUPS + 1):
            # top-up files that already exist are always loaded so reruns reproduce the same selection
            if k > 0 and len(state["out"]) >= n and not page_file(lang, k).exists():
                break
            pages = load_file(lang, k, force, downloads)
            files.append({"path": str(page_file(lang, k)), "n_pages": len(pages)})
            pick(lang, n, pages, concept_sents, hp_titles.get(lang, ""), state)
        out, st = state["out"], state["stats"]
        if len(out) < n:
            raise RuntimeError(f"[{lang}] only {len(out)}/{n} sentences after {MAX_TOPUPS} top-ups ({st}, {files})")
        report[lang] = {"needed": n, "picked": len(out), "files": files, "downloaded": downloads,
                        "topups_used": len(files) - 1, **st}
        print(f"[{lang}] need {n}: files {[f['n_pages'] for f in files]} pages, scanned {st['pages_available']}, "
              f"hp {st['hp_page']}, short {st['short']}, no prose {st['no_sentence']}, "
              f"template-only pages {st['template_page_drops']} (sentences {st['template_sentence_drops']}), "
              f"concept/dup sentences {st['concept_or_dup_sentence_drops']} -> {len(out)}"
              + (f" | downloaded {downloads}" if downloads else ""), flush=True)
        rows += out
    NEUTRAL_EXT_NATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEUTRAL_EXT_NATIVE_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {len(rows)} sentences -> {NEUTRAL_EXT_NATIVE_PATH}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download the non-English dump samples (and top-ups)")
    run(ap.parse_args().force)
