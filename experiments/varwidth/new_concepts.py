#!/usr/bin/env python
"""Extra EMBER concepts built from their English Wikipedia page, like the 18 in data/concept_sentences.json.

    python -m experiments.varwidth.new_concepts [--concepts ...] [--fallback ...] [--n 300] [--seed 42]

Per concept (name = Wikipedia title): fetch the page once (fetch_en_sentences.page_paragraphs), split it
(split_sentences), sample n sentences with random.Random(seed), page order kept. A page with < n sentences is
replaced by the next --fallback concept. Writes/merges:
    data/concept_sentences_extra.json            [{"concept", "source": "wikipedia", "title", "revid", "sentences"}]
    experiments/whitening/data/pages/<slug>.json the full split page, as build_all_targets writes it, so the
                                                 whitening target uses the same revision
Then: python -m experiments.varwidth.build_all_targets --concepts <the concepts>
"""
import argparse
import json
import random
import sys

from experiments.varwidth.common import ROOT, slug
from experiments.varwidth.prep import CONCEPTS_EXTRA

DEFAULT = ["French Revolution", "Tuberculosis", "Cricket", "Buddhism", "Thanksgiving"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", nargs="*", default=DEFAULT)
    p.add_argument("--fallback", nargs="*", default=["Plutonium", "Hinduism"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    sys.path.insert(0, str(ROOT / "experiments" / "resample_sentences"))
    from fetch_en_sentences import page_paragraphs, split_sentences

    pages = ROOT / "experiments" / "whitening" / "data" / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    main_names = {e["concept"] for e in json.loads((ROOT / "data" / "concept_sentences.json").read_text())}
    extra = {e["concept"]: e for e in json.loads(CONCEPTS_EXTRA.read_text())} if CONCEPTS_EXTRA.exists() else {}
    queue, fallback, done = list(a.concepts), list(a.fallback), []
    while queue:
        c = queue.pop(0)
        assert c not in main_names, f"{c!r} is already in data/concept_sentences.json"
        title, revid, paras = page_paragraphs(c)
        sents = split_sentences(paras)
        if len(sents) < a.n:
            print(f"[skip] {c}: '{title}' rev {revid} has {len(sents)} < {a.n} sentences", flush=True)
            if not fallback:
                raise SystemExit("no fallback concept left")
            queue.append(fallback.pop(0))
            continue
        keep = sorted(random.Random(a.seed).sample(range(len(sents)), a.n))
        extra[c] = {"concept": c, "source": "wikipedia", "title": title, "revid": revid,
                    "sentences": [sents[i] for i in keep]}
        (pages / f"{slug(c)}.json").write_text(json.dumps({"concept": c, "title": title, "revid": revid,
                                                            "sentences": sents}, ensure_ascii=False, indent=1))
        print(f"[ok] {c}: '{title}' rev {revid}: {len(sents)} sentences, kept {a.n}", flush=True)
        done.append(c)
    CONCEPTS_EXTRA.write_text(json.dumps(list(extra.values()), ensure_ascii=False, indent=1))
    print(f"-> {CONCEPTS_EXTRA} ({len(extra)} concepts); new: {done}")


if __name__ == "__main__":
    main()
