#!/usr/bin/env python
"""Pre-registered analysis of the seed-union approaches (PREREG_NEW.md amendment 1): H4, H5 and coverage.

    python -m experiments.ml.analyze_union          # writes results/ml/UNION_RESULTS.md and prints it

Same unit score and tests as analyze_prereg (strict test score, no edit = 0, the 18 original concepts on their v2
scores; paired sign and Wilcoxon tests, bootstrap 95% CI over concepts).
"""
from __future__ import annotations

from statistics import mean

from experiments.ml.analyze_prereg import compare, concept_sets, unit
from experiments.ml.common import RESULTS

APPS = ("A1", "A3z", "A1U", "A3zU")


def main():
    orig, new = concept_sets()
    U = {}
    for c in orig + new:
        for a in APPS:
            U[c, a] = unit(c, a, use_v2=c in orig)
    out = ["# Seed-union analysis (PREREG_NEW.md amendment 1)", "",
           "Strict test score; no edit = 0; original 18 on v2 scores. Paired over concepts; CI = bootstrap 95%; "
           "p = two-sided sign test and Wilcoxon signed-rank. H4 supported if the CI lower bound of A3zU − A1U "
           "is above −0.05.", ""]
    for name, cs in (("All 38 concepts", orig + new), ("New 20 concepts", new), ("Original 18 concepts", orig)):
        out += [f"## {name}", "",
                "| | " + " | ".join(APPS) + " |", "|---|" + "---|" * len(APPS),
                "| Concepts with an edit | " + " | ".join(str(sum(U[c, a]["edit"] for c in cs)) for a in APPS) + " |",
                "| Mean strict score | " + " | ".join(f"{mean(U[c, a]['strict'] for c in cs):.3f}" for a in APPS) + " |",
                "| Mean ungated score | " + " | ".join(f"{mean(U[c, a]['ungated'] for c in cs):.3f}" for a in APPS) + " |", ""]
        for key in ("strict", "ungated"):
            out += [f"**Paired tests, {key}.**", "",
                    "| Comparison | n | means | mean diff [95% CI] | wins–losses | p sign | p Wilcoxon |",
                    "|---|---|---|---|---|---|---|",
                    "| *H4: all concepts* | | | | | | |",
                    compare(U, cs, "A3zU", "A1U", key, lambda c: True),
                    "| *H4: concepts both edit* | | | | | | |",
                    compare(U, cs, "A3zU", "A1U", key, lambda c: U[c, "A1U"]["edit"] and U[c, "A3zU"]["edit"]),
                    "| *H5: all concepts* | | | | | | |",
                    compare(U, cs, "A1U", "A1", key, lambda c: True),
                    compare(U, cs, "A3zU", "A3z", key, lambda c: True),
                    "| *Reference: single seed, all concepts* | | | | | | |",
                    compare(U, cs, "A3z", "A1", key, lambda c: True), ""]
    p = RESULTS / "UNION_RESULTS.md"
    p.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
