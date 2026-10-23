"""Sensitivity of the in-sample cut-off rule: admit a new C* token when its oriented joint coefficient reaches the
q-quantile of the coefficients of the saved support tokens (q = 0 is the cut-off rule up to sign handling).

    .venv/bin/python -m experiments.paper_analysis.extrapolation.cutoff_sweep

Output: results/paper_analysis/extrapolation/cutoff_sweep.json {dict: {q: {n_admit, nS, prec, precSA, n}}}
(per run, then concept means; precision over runs with >= 1 admitted token).
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np

from experiments.paper_analysis.extrapolation.analyze import load_labels
from experiments.paper_analysis.extrapolation.common import CONCEPTS, DICTS, OUT, TOKDIR, write_json
from experiments.varwidth.common import slug

QS = (0.0, 0.25, 0.5, 0.75)


def main():
    labels = load_labels()
    acc = defaultdict(lambda: defaultdict(list))
    for c in CONCEPTS:
        td = json.loads((TOKDIR / f"{slug(c)}.json").read_text())
        L = labels[c]
        new = td["new"]
        g = td["conds"]["balanced"]["groups"]
        cm = np.array([g.get(str(t)) == "C*" for t in new])
        R = pickle.load(open(OUT / "runs" / f"{slug(c)}.pkl", "rb"))
        for (D, seed), r in R.items():
            if not r["F_C"]:
                continue
            sup = r["scores_vstar"]["coef"][r["support_vstar"]]
            for q in QS:
                thr = np.quantile(sup, q)
                sel = np.nonzero(cm & (r["scores"]["coef"] >= thr))[0]
                lab = [L.get(int(new[i]), "?") for i in sel]
                lab = [x for x in lab if x != "?"]
                acc[(D, q)][c].append((len(sel), sum(x == "S" for x in lab), sum(x != "U" for x in lab), len(lab)))
    out = {}
    for D in DICTS:
        out[D] = {}
        for q in QS:
            pcs = acc[(D, q)]
            n_ad = [np.mean([x[0] for x in v]) for v in pcs.values()]
            nS = [np.mean([x[1] for x in v]) for v in pcs.values()]
            prec = [np.mean([x[1] / x[3] for x in v if x[3]]) for v in pcs.values() if any(x[3] for x in v)]
            precSA = [np.mean([x[2] / x[3] for x in v if x[3]]) for v in pcs.values() if any(x[3] for x in v)]
            miss = sum(x[0] - x[3] for v in pcs.values() for x in v)
            out[D][str(q)] = {"n_admit": float(np.mean(n_ad)), "nS": float(np.mean(nS)), "prec": float(np.mean(prec)),
                              "precSA": float(np.mean(precSA)), "n_concepts_prec": len(prec), "unlabelled": int(miss)}
    write_json(OUT / "cutoff_sweep.json", out)
    for D, v in out.items():
        for q, e in v.items():
            print(D, q, {k: round(x, 3) if isinstance(x, float) else x for k, x in e.items()})


if __name__ == "__main__":
    main()
