#!/usr/bin/env python
"""Whitening targets + whitened prep for every concept in data/concept_sentences.json that lacks them.

    python -m experiments.varwidth.build_all_targets [--concepts ...]

Page source: the local raw page under experiments/multilingual_hp/data/concepts/<slug>/raw/en.json when present,
otherwise the current English Wikipedia page (fetch_en_sentences.page_paragraphs, as background.py does for
Ancient Rome). Then prep.build_target_page ("page" key) and prep.prepare(concept, "whitened").
"""
import argparse
import json
import sys

from experiments.varwidth.common import ROOT, slug
from experiments.varwidth import prep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", nargs="*", default=None)
    a = p.parse_args()
    concepts = a.concepts or [e["concept"] for e in json.loads((ROOT / "data" / "concept_sentences.json").read_text())]
    sys.path.insert(0, str(ROOT / "experiments" / "resample_sentences"))
    from fetch_en_sentences import page_paragraphs, split_sentences

    pages = ROOT / "experiments" / "whitening" / "data" / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    for c in concepts:
        target = prep.WH_DIR / "targets" / f"{slug(c)}.json"
        local = ROOT / "experiments" / "multilingual_hp" / "data" / "concepts" / slug(c) / "raw" / "en.json"
        try:
            if not target.exists():
                if not local.exists():
                    page_out = pages / f"{slug(c)}.json"
                    if not page_out.exists():
                        title, revid, paras = page_paragraphs(c)
                        sents = split_sentences(paras)
                        page_out.write_text(json.dumps({"concept": c, "title": title, "revid": revid, "sentences": sents},
                                                       ensure_ascii=False, indent=1))
                        print(f"[page] {c}: fetched '{title}' rev {revid}: {len(sents)} sentences", flush=True)
                prep.build_target_page(c, local)
            prep.prepare(c, "whitened")
            print(f"[ok] {c}", flush=True)
        except Exception as e:  # keep going; report at the end
            print(f"[FAIL] {c}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
