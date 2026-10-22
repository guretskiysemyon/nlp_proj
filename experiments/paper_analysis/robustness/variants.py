"""Build every concept-sentence variant WP4 fits, check the pipeline against the project's prep, list the fit jobs.

    .venv/bin/python -m experiments.paper_analysis.robustness.variants [--check]

Variants (results/paper_analysis/robustness/data/variants/<variant>.json):
    <Slug>          the 300 concept sentences: project data for ENTITY / ABSTRACT_OLD / ADJACENT keys, fetch.py's for
                    ABSTRACT_NEW
    <Slug>__S190    190 sentences: ABSTRACT_SHORT (fetch.py's sample) and ENTITY sub-sampled from the project's 300
                    (random.Random(42).sample, file order kept)
    <Slug>__u<pp>   control: the same positions replaced by sentences of UNRELATED project concepts (no name filter
                    needed), interleaved round-robin after a random.Random(42) shuffle per page; fitted with EMBER, CS
    <Slug>__f<pp>   contamination: positions perm[:n] of the 300 (perm = random.Random(42) permutation) replaced by the
                    first n adjacent sentences, n = round(f * 300). Adjacent sentences: each adjacent page's sentences
                    minus those containing a name form of the concept (case-insensitive), each page shuffled with
                    random.Random(42), interleaved round-robin. Nested in f (f = 0.1 is a subset of f = 0.25 ...).
"""
from __future__ import annotations

import argparse
import json
import random

from experiments.paper_analysis.robustness.concepts import (ABSTRACT_NEW, ABSTRACT_OLD, ABSTRACT_SHORT, ADJACENT,
                                                            CONCEPTS_FILE, ENTITY, FRACTIONS, NAME_FORMS, PAGES,
                                                            S_SHORT, SAMPLE_SEED, SEEDS, UNRELATED, slug, variant_name)
from experiments.paper_analysis.robustness.pipeline import write_variant
from experiments.varwidth.common import ROOT


def project_sentences(concept: str) -> list[str]:
    for f in ("concept_sentences.json", "concept_sentences_extra.json"):
        for e in json.loads((ROOT / "data" / f).read_text()):
            if e["concept"] == concept:
                return list(e["sentences"])
    raise KeyError(concept)


def new_sentences(concept: str) -> list[str]:
    for e in json.loads(CONCEPTS_FILE.read_text()):
        if e["concept"] == concept:
            return list(e["sentences"])
    raise KeyError(concept)


def adjacent_pool(concept: str, kind: str = "f") -> tuple[list[str], list[str]]:
    """Interleaved adjacent (kind "f") or unrelated (kind "u") sentences, and the page each came from."""
    forms = [w.lower() for w in NAME_FORMS[concept]]
    per_page = []
    for t in (ADJACENT[concept] if kind == "f" else UNRELATED):
        s = (json.loads((PAGES / f"{slug(t)}.json").read_text())["sentences"] if kind == "f"
             else project_sentences(t))
        s = [x for x in s if not any(w in x.lower() for w in forms)]
        random.Random(SAMPLE_SEED).shuffle(s)
        per_page.append((t, s))
    out, src = [], []
    for i in range(max(len(s) for _, s in per_page)):
        for t, s in per_page:
            if i < len(s):
                out.append(s[i])
                src.append(t)
    return out, src


def build_all() -> dict:
    """{variant: concept} for every variant; writes the variant files."""
    out = {}
    for c in ENTITY + ABSTRACT_OLD + [k for k in ADJACENT if k not in ENTITY]:
        out[slug(c)] = c
        write_variant(slug(c), c, project_sentences(c))
    for c in ABSTRACT_NEW:
        out[slug(c)] = c
        write_variant(slug(c), c, new_sentences(c))
    for c in ABSTRACT_SHORT:
        v = f"{slug(c)}__S{S_SHORT}"
        out[v] = c
        write_variant(v, c, new_sentences(c))
    for c in ENTITY:
        s = project_sentences(c)
        keep = sorted(random.Random(SAMPLE_SEED).sample(range(len(s)), S_SHORT))
        v = f"{slug(c)}__S{S_SHORT}"
        out[v] = c
        write_variant(v, c, [s[i] for i in keep])
    for c in ADJACENT:
        base = project_sentences(c)
        perm = list(range(len(base)))
        random.Random(SAMPLE_SEED).shuffle(perm)
        for kind in ("f", "u"):
            pool, src = adjacent_pool(c, kind)
            for f in FRACTIONS:
                if f == 0:
                    continue
                n = round(f * len(base))
                assert n <= len(pool), (c, f, n, len(pool))
                s = list(base)
                for k in range(n):
                    s[perm[k]] = pool[k]
                v = variant_name(c, f, kind)
                out[v] = c
                write_variant(v, c, s, {"f": f, "kind": kind, "replaced_positions": sorted(perm[:n]),
                                        "inserted_from": {t: src[:n].count(t) for t in sorted(set(src))}})
    return out


def jobs(variants: dict) -> list[tuple[str, str, str, int]]:
    """(variant, concept, cond, seed). Part A: EMBER, CS, CSGS on the base / S190 variants; part B: + CS_sc, and all
    four conditions on the contaminated variants."""
    J = []
    contam = set(ADJACENT)
    for v, c in variants.items():
        conds = ["EMBER", "CS", "CSGS"]
        if c in contam and "__S" not in v:
            conds.append("CS_sc")
        if "__u" in v:                                   # unrelated-page control: EMBER and CS only
            conds = ["EMBER", "CS"]
        for cond in conds:
            for s in SEEDS:
                J.append((v, c, cond, s))
    return J


def check():
    """The pipeline reproduces the project's prep for Harry Potter (V*, codes, A plain and whitened)."""
    import torch
    from experiments.paper_analysis.robustness.pipeline import build_A, variant_meta
    for c in ("Harry Potter", "Uranium"):
        meta = variant_meta(slug(c), c)
        for matrix, target in (("plain", None), ("whitened", "page")):
            ref = torch.load(ROOT / "results" / "varwidth" / "prep" / f"{slug(c)}_{matrix}_S300.pt", weights_only=False)
            assert ref["vstar_ids"] == meta["vstar_ids"], "V* differs"
            assert (ref["codes"] == meta["codes"]).all(), "codes differ"
            A, _ = build_A(meta, matrix, target)
            err = float((A - ref["A"]).abs().max() / ref["A"].abs().max())
            print(f"[check] {c} {matrix}: V* and codes equal, max|A - A_ref| / max|A_ref| = {err:.2e}")
            assert err < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    v = build_all()
    print(f"{len(v)} variants, {len(jobs(v))} fit jobs")
    (CONCEPTS_FILE.parent / "variants.json").write_text(json.dumps(v, indent=1))
    if a.check:
        check()


if __name__ == "__main__":
    main()
