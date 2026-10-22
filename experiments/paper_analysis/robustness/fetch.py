"""Fetch the Wikipedia pages WP4 needs and build the new concept sentence sets and CS targets (no Gemini).

    .venv/bin/python -m experiments.paper_analysis.robustness.fetch

As experiments/varwidth/new_concepts.py: page_paragraphs + split_sentences (experiments/resample_sentences), keep
n = 300 sentences sampled with random.Random(42), page order kept; a page with < 300 sentences is replaced by the next
fallback concept. Adjacent pages (part B) are fetched and split only. Writes, all under results/paper_analysis/robustness/data/:
    pages/<slug>.json              {"concept", "title", "revid", "sentences": [all page sentences]}
    concept_sentences_pa.json      [{"concept", "source": "wikipedia", "title", "revid", "sentences": [300]}]
    targets/<slug>.json            {"page": [token ids of the whole page]}, as prep.build_target_page (minus the
                                   excluded special / unused / byte-fallback ids of experiments/whitening/common.py)
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys

from experiments.paper_analysis.robustness.concepts import (ABSTRACT_NEW, ABSTRACT_SHORT, ADJACENT, CONCEPTS_FILE,
                                                            FALLBACK, N_SENTS, PAGES, S_SHORT, SAMPLE_SEED, TARGETS,
                                                            slug)
from experiments.varwidth.common import ROOT


def _fetch_mod():
    p = str(ROOT / "experiments" / "resample_sentences")
    if p not in sys.path:
        sys.path.insert(0, p)
    import fetch_en_sentences
    return fetch_en_sentences


def _whitening_common():
    spec = importlib.util.spec_from_file_location("whitening_common", ROOT / "experiments" / "whitening" / "common.py")
    wc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wc)
    return wc


def get_page(title: str) -> dict:
    path = PAGES / f"{slug(title)}.json"
    if path.exists():
        return json.loads(path.read_text())
    fe = _fetch_mod()
    t, revid, paras = fe.page_paragraphs(title)
    page = {"concept": title, "title": t, "revid": revid, "sentences": fe.split_sentences(paras)}
    PAGES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(page, ensure_ascii=False, indent=1))
    print(f"[page] {title}: '{t}' rev {revid}: {len(page['sentences'])} sentences", flush=True)
    return page


def page_target_ids(sentences: list[str]) -> list[int]:
    """Token ids of the sentences (no specials), minus the whitening module's excluded ids (build_target_page)."""
    wc = _whitening_common()
    ex = wc.excluded_ids("gemma")
    tok = wc.tokenizer("gemma")
    ids = set()
    for seq in tok(sentences, add_special_tokens=False)["input_ids"]:
        ids.update(seq)
    return sorted(int(i) for i in ids if i not in ex)


def main():
    TARGETS.mkdir(parents=True, exist_ok=True)
    # sanity: our target builder reproduces the project's Harry Potter target from the project's page file
    hp_page = json.loads((ROOT / "experiments" / "whitening" / "data" / "pages" / "Harry_Potter.json").read_text())
    hp_ref = json.loads((ROOT / "experiments" / "whitening" / "data" / "gemma" / "targets" / "Harry_Potter.json")
                        .read_text())["page"]
    assert page_target_ids(hp_page["sentences"]) == hp_ref, "target builder does not reproduce Harry_Potter.json"
    print("[check] target builder reproduces experiments/whitening/data/gemma/targets/Harry_Potter.json")

    entries = json.loads(CONCEPTS_FILE.read_text()) if CONCEPTS_FILE.exists() else []
    have = {e["concept"] for e in entries}
    queue, fallback = list(ABSTRACT_NEW) + list(ABSTRACT_SHORT), list(FALLBACK)
    while queue:
        c = queue.pop(0)
        n_keep = S_SHORT if c in ABSTRACT_SHORT else N_SENTS
        page = get_page(c)
        sents = page["sentences"]
        if len(sents) < n_keep:
            print(f"[skip] {c}: {len(sents)} < {N_SENTS} sentences; using fallback", flush=True)
            queue.append(fallback.pop(0))
            continue
        if c not in have:
            keep = sorted(random.Random(SAMPLE_SEED).sample(range(len(sents)), n_keep))
            entries.append({"concept": c, "source": "wikipedia", "title": page["title"], "revid": page["revid"],
                            "sentences": [sents[i] for i in keep]})
            have.add(c)
        tpath = TARGETS / f"{slug(c)}.json"
        if not tpath.exists():
            tpath.write_text(json.dumps({"page": page_target_ids(sents)}))
        print(f"[ok] {c}: {len(sents)} page sentences, target {len(json.loads(tpath.read_text())['page'])} ids",
              flush=True)
    CONCEPTS_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    print(f"-> {CONCEPTS_FILE}: {[e['concept'] for e in entries]}")

    for concept, titles in ADJACENT.items():
        for t in titles:
            get_page(t)


if __name__ == "__main__":
    main()
