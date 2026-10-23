#!/usr/bin/env python
"""Pre-registered analysis (PREREG_NEW.md): H1-H3 and coverage on the 20 new concepts and on all 38.

    python -m experiments.ml.analyze_prereg          # writes results/ml/PREREG_RESULTS.md and prints it

Unit score: strict test score (0 if a gate fails on test; "no edit" = 0). The 18 original concepts enter with their
v2 scores (rescore_v2.json, cleaned it/pt/hi/th questions); the new 20 were built with the v2 pipeline. Ungated
scores are reported alongside. Tests: paired sign test and Wilcoxon signed-rank across concepts (two-sided),
bootstrap 95% CI of the mean difference over concepts (10,000 resamples, seed 0).
"""
from __future__ import annotations

import json
import random
from statistics import mean

from experiments.ml.common import LANGS, RESULTS, run_dir

APPS = ("A1", "A2", "A3", "A3z")


def concept_sets():
    import os
    from experiments.ml.common import concepts
    os.environ["ML_CONCEPTS"] = "orig"
    orig = concepts()
    os.environ["ML_CONCEPTS"] = "new"
    new = concepts()
    return orig, new


def unit(c, a, use_v2):
    e = json.loads((run_dir(c, a) / "eval.json").read_text())
    if not e.get("done"):
        raise RuntimeError(f"not done: {c} {a}")
    if not e.get("n_edited"):
        return {"edit": False, "strict": 0.0, "ungated": 0.0}
    strict = e["test"]["score"]
    per = e["test"]["per_lang"]
    if use_v2 and (run_dir(c, a) / "rescore_v2.json").exists():
        r = json.loads((run_dir(c, a) / "rescore_v2.json").read_text())
        strict, per = r["score"], r.get("per_lang", per)
    elig = [L for L in LANGS if per[L].get("eligible", True)]
    return {"edit": True, "strict": strict, "ungated": mean(per[L]["H"] for L in elig) if elig else 0.0}


def sign_test(d):
    from math import comb
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    return pos, neg, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilcoxon(d):
    try:
        from scipy.stats import wilcoxon as w
        d = [x for x in d if x != 0]
        return float(w(d).pvalue) if len(d) >= 2 else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def boot(d, n=10000, seed=0):
    if not d:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(mean(rng.choice(d) for _ in d) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def compare(U, cs, a, b, key, cond):
    sh = [c for c in cs if cond(c)]
    d = [U[c, a][key] - U[c, b][key] for c in sh]
    if not sh:
        return f"| {a} − {b} | 0 | – | – | – | – | – |"
    pos, neg, ps = sign_test(d)
    lo, hi = boot(d)
    return (f"| {a} − {b} | {len(sh)} | {mean(U[c, a][key] for c in sh):.3f} vs {mean(U[c, b][key] for c in sh):.3f} "
            f"| {mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}] | {pos}–{neg} | {ps:.3f} | {wilcoxon(d):.3f} |")


def main():
    orig, new = concept_sets()
    U = {}
    for c in orig:
        for a in APPS:
            U[c, a] = unit(c, a, use_v2=True)
    for c in new:
        for a in APPS:
            U[c, a] = unit(c, a, use_v2=False)
    out = ["# Pre-registered analysis (PREREG_NEW.md)", "",
           "Strict test score; no edit = 0; original 18 with v2 (cleaned-question) scores. Differences are paired over "
           "concepts; CI = bootstrap 95%; p = two-sided sign test and Wilcoxon signed-rank.", ""]
    for name, cs in (("New 20 concepts (confirmation)", new), ("All 38 concepts", orig + new),
                     ("Original 18 concepts (for reference)", orig)):
        out += [f"## {name}", "",
                "| | " + " | ".join(APPS) + " |", "|---|" + "---|" * len(APPS),
                "| Concepts with an edit | " + " | ".join(str(sum(U[c, a]["edit"] for c in cs)) for a in APPS) + " |",
                "| Mean strict score (all) | " + " | ".join(f"{mean(U[c, a]['strict'] for c in cs):.3f}" for a in APPS) + " |",
                "| Mean ungated score (all) | " + " | ".join(f"{mean(U[c, a]['ungated'] for c in cs):.3f}" for a in APPS) + " |", ""]
        for key in ("strict", "ungated"):
            out += [f"**Paired tests, {key}.**", "",
                    "| Comparison | n | means | mean diff [95% CI] | wins–losses | p sign | p Wilcoxon |",
                    "|---|---|---|---|---|---|---|",
                    "| *H1* (concepts both edit) | | | | | | |",
                    compare(U, cs, "A2", "A1", key, lambda c: U[c, "A1"]["edit"] and U[c, "A2"]["edit"]),
                    "| *H2* (concepts A3 edits) | | | | | | |",
                    compare(U, cs, "A3z", "A3", key, lambda c: U[c, "A3"]["edit"]),
                    "| *H3* (concepts all four edit) | | | | | | |",
                    compare(U, cs, "A3z", "A2", key, lambda c: all(U[c, a]["edit"] for a in APPS)),
                    "| *H3, any edit by either* | | | | | | |",
                    compare(U, cs, "A3z", "A2", key, lambda c: U[c, "A2"]["edit"] or U[c, "A3z"]["edit"]),
                    "| *A3z vs A1 (any edit by either)* | | | | | | |",
                    compare(U, cs, "A3z", "A1", key, lambda c: U[c, "A1"]["edit"] or U[c, "A3z"]["edit"]), ""]
    p = RESULTS / "PREREG_RESULTS.md"
    p.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
