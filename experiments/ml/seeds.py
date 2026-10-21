#!/usr/bin/env python
"""Coverage across fit seeds: A2, A3 (and A3z, which shares A3's selection) refit and reselected with other ALS seeds.

    ML_CONCEPTS=all python -m experiments.ml.seeds --seed 1 [--device cpu]
    python -m experiments.ml.seeds --report            # results/ml/seeds/COVERAGE.md

Same sample (data seed 0, N = 30), same everything as the main run; only fit.SEED changes. Outputs go to
results/ml/seeds/fs<seed>/<slug>/<approach>/ (fit.pt, selection.json, judge.json); the main runs (fit seed 42) are
read from results/ml/runs/. The A3 lexicon is read from the cache (results/ml/lexicons.json), never regenerated.
No evaluation: this measures coverage (does an approach select any concept feature) and edit-set stability.
"""
from __future__ import annotations

import argparse
import json

from experiments.ml import fit as F
from experiments.ml import select as S
from experiments.ml.common import RESULTS, concepts, run_dir, slug

OUT = RESULTS / "seeds"
APPS = ("A2", "A3")


def sdir(concept, seed, approach):
    return run_dir(concept, approach) if seed == 42 else OUT / f"fs{seed}" / slug(concept) / approach


def edited(concept, seed, approach):
    import torch
    from experiments.ml.evaluate import run_edit
    d = sdir(concept, seed, approach)
    if not (d / "selection.json").exists():
        return None
    return set(run_edit(d / "fit.pt", d / "selection.json"))


def run(seed, device):
    F.SEED = seed
    F.run_dir = lambda c, a, lang=None: sdir(c, seed, a)
    S.run_dir = F.run_dir
    for c in concepts():
        for a in APPS:
            if (sdir(c, seed, a) / "selection.json").exists():
                continue
            F.run(c, a, N=30, device=device)
            S.run(c, a)


def report():
    import os
    os.environ.setdefault("ML_CONCEPTS", "all")
    seeds = [42] + sorted(int(p.name[2:]) for p in OUT.glob("fs*"))
    rows, cov = [], {(a, s): 0 for a in APPS for s in seeds}
    for c in concepts():
        cells = []
        for a in APPS:
            sets = {s: edited(c, s, a) for s in seeds}
            for s in seeds:
                cov[a, s] += bool(sets[s])
            n = [len(json.loads((sdir(c, s, a) / "selection.json").read_text())["F_C"]) if sets[s] is not None else "–"
                 for s in seeds]
            cells.append("/".join(map(str, n)))
        rows.append(f"| {c} | " + " | ".join(cells) + " |")
    out = ["# Coverage across fit seeds (A2, A3; A3z = A3's selection)", "",
           f"Selected concept features per fit seed ({', '.join(map(str, seeds))}); 0 = no edit.", "",
           "| Concept | A2 | A3 / A3z |", "|---|---|---|", *rows, "",
           "| Concepts with an edit | " + " | ".join(f"seed {s}" for s in seeds) + " | any seed |",
           "|---|" + "---|" * (len(seeds) + 1)]
    for a in APPS:
        anys = sum(any(edited(c, s, a) for s in seeds) for c in concepts())
        out.append(f"| {a} | " + " | ".join(str(cov[a, s]) for s in seeds) + f" | {anys} |")
    (OUT / "COVERAGE.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.seed, a.device)


if __name__ == "__main__":
    main()
