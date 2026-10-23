"""Extra native Wikipedia pages for the neutral / held-out / background sentences.

multilingual_hp already holds 1,900 dump pages per language (dump/ + dump_ext/) and 1,000 English pages; the
5,000-sentence background needs more for most languages. This samples further pages from the Hugging Face dump
wikimedia/wikipedia (config 20231101.<lang>) with resample_sentences.sample_dump (HTTP range reads of whole row
groups, seed SEED + 100) into results/ml/data/dump/wiki_dump_pages_<lang>.jsonl (kept if it exists).

    python -m experiments.ml.data.dump_extra [--lang hi th ...]
"""
from __future__ import annotations

import argparse
import time

from experiments.ml.data.common import DUMP, LANGS, SEED
from sample_wiki_dump import sample_dump  # noqa: E402 (experiments/resample_sentences)

# (row groups, pages per group) per language, sized from the usable-page yield of the existing samples
PLAN = {"en": (120, 25), "hi": (100, 25), "de": (40, 25)}
DEFAULT = (80, 25)


def path(lang: str):
    return DUMP / f"wiki_dump_pages_{lang}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=list(LANGS))
    args = ap.parse_args()
    for lang in args.lang:
        if path(lang).exists():
            print(f"[{lang}] exists, skipped", flush=True)
            continue
        n_groups, per_group = PLAN.get(lang, DEFAULT)
        for attempt in range(8):
            try:
                sample_dump(f"20231101.{lang}", path(lang), n_groups=n_groups, per_group=per_group, seed=SEED + 100)
                break
            except Exception as e:  # noqa: BLE001  429s / timeouts from HF
                wait = min(60 * 2 ** attempt, 900)
                print(f"[{lang}] {type(e).__name__}: {e}; retry in {wait}s", flush=True)
                time.sleep(wait)
        print(f"[{lang}] done -> {path(lang)}", flush=True)


if __name__ == "__main__":
    main()
