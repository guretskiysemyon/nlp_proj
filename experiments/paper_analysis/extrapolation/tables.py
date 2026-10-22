"""Markdown tables for REPORT.md from summary.json / extra.json / runs_metrics.csv (printed to stdout).

    .venv/bin/python -m experiments.paper_analysis.extrapolation.tables
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from experiments.paper_analysis.extrapolation.common import CONCEPTS, DICTS, OUT
from experiments.paper_analysis.extrapolation.pool import FULL_LABEL
from experiments.paper_analysis.extrapolation.score import RULES, SCORES


def f(e, d=2):
    return "–" if e is None or (isinstance(e, float) and np.isnan(e)) else f"{e:.{d}f}"


def ci(e, d=2):
    return f"{f(e['mean'], d)} [{f(e['ci'][0], d)}, {f(e['ci'][1], d)}]"


def main():
    S = json.loads((OUT / "summary.json").read_text())
    X = json.loads((OUT / "extra.json").read_text())
    print("### Ranking of new tokens (balanced ratio; concept means over 17 concepts, bootstrap 95% CI)\n")
    print("| dictionary | score | P@1 | P@5 | P@10 (C*) | P@20 | P(S+A)@10 | AUC S (C*, n=6) | P@10 all new | N* in top 10 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for D in DICTS:
        for sc in SCORES:
            e = S["ranking"][f"balanced|{D}|{sc}"]
            print(f"| {D} | {sc} | {f(e['cP1']['mean'])} | {f(e['cP5']['mean'])} | {ci(e['cP10'])} | "
                  f"{f(e['cP20']['mean'])} | {f(e['cPSA10']['mean'])} | {f(e['cAUC']['mean'])} | "
                  f"{f(e['P10']['mean'])} | {f(e['Nstar10']['mean'])} |")
    b = S["base_rate"]["balanced"]
    print(f"\nRandom order: P(S) = {b['cS']:.3f} among C*, {b['S']:.3f} among all new tokens (balanced); "
          f"natural: {S['base_rate']['natural']['S']:.3f}.\n")
    print("### Natural ratio (P@10 over all new tokens / N* in top 10)\n")
    print("| dictionary | " + " | ".join(SCORES) + " |")
    print("|---|" + "---|" * len(SCORES))
    for D in DICTS:
        print(f"| {D} | " + " | ".join(f"{f(S['ranking'][f'natural|{D}|{sc}']['P10']['mean'])} / "
                                       f"{f(S['ranking'][f'natural|{D}|{sc}']['Nstar10']['mean'])}" for sc in SCORES) + " |")
    print("\n### Admission rules (balanced; mean per run, concept means)\n")
    print("| dictionary | rule | admitted | S | A | U page | N* | prec S | prec S+A | edited (C*) | support kept |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for D in DICTS:
        for rule in RULES:
            e = S["admission"][f"balanced|{D}|{rule}"]
            print(f"| {D} | {rule} | {f(e['n_admit']['mean'], 1)} | {f(e['nS']['mean'], 1)} | {f(e['nA']['mean'], 1)} | "
                  f"{f(e['nU_page']['mean'], 1)} | {f(e['nNstar']['mean'], 1)} | {f(e['prec']['mean'])} | "
                  f"{f(e['precSA']['mean'])} | {f(e['n_admit_Cstar']['mean'], 1)} | {f(e['retain']['mean'])} |")
    print("\n### Admission, natural ratio (admitted / N* / prec S)\n")
    print("| dictionary | " + " | ".join(RULES) + " |")
    print("|---|" + "---|" * len(RULES))
    for D in DICTS:
        print(f"| {D} | " + " | ".join(
            f"{f(S['admission'][f'natural|{D}|{r}']['n_admit']['mean'], 1)} / "
            f"{f(S['admission'][f'natural|{D}|{r}']['nNstar']['mean'], 1)} / "
            f"{f(S['admission'][f'natural|{D}|{r}']['prec']['mean'])}" for r in RULES) + " |")
    print("\n### Recall of the S tokens among C* (6 fully labelled concepts with >= 1 S token)\n")
    print("| dictionary | frac | width | cutoff | argmax | share50 | coef top 10 | top 20 | top 50 | top 100 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    R = X["recall_full_concepts"]
    for D in DICTS:
        print(f"| {D} | " + " | ".join(f(R[f"{D}|recall_{r}"]["mean"]) for r in RULES) + " | " +
              " | ".join(f(R[f"{D}|recall_coef@{k}"]["mean"]) for k in (10, 20, 50, 100)) + " |")
    print("\n### Per concept (balanced): token counts, base rate, P@10 (C*) of the joint coefficient\n")
    rows = list(csv.DictReader(open(OUT / "runs_metrics.csv")))
    pc = defaultdict(list)
    for r in rows:
        if r["cond"] == "balanced":
            pc[(r["concept"], r["dict"])].append(float(r["coef_cP10"]))
    print("| concept | new page tokens | C* | N* | runs E/CS/Ci | P(S) random | EMBER | CS | CINDER | labels |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in CONCEPTS:
        t = S["tokens"][c]["balanced"]
        runs = "/".join(str(len(pc.get((c, D), []))) for D in DICTS)
        vals = " | ".join(f(np.mean(pc[(c, D)])) if pc.get((c, D)) else "–" for D in DICTS)
        print(f"| {c} | {X['page_count_distribution'][c]['n_new_page']} | {t['n_Cstar']} | "
              f"{t['n_new'] - t['n_page_new']} | {runs} | {f(t['cS'], 3)} | {vals} | "
              f"{'all' if c in FULL_LABEL else 'pooled'} |")


if __name__ == "__main__":
    main()
