"""What the TJ judge sees: composition of the top-20 tokens (|Y| order, select.py) of every judged A2 / A3 candidate.

    python -m experiments.paper_analysis.ml_geometry.top20

Per candidate: share of the 20 tokens that are concept-only (code 1), whose language label is English
(p(en|t) >= 0.8), that are Latin script, and the max-language share of the 20 (unweighted mean of p(L|t)).
Writes top20.csv (merged later with features.csv.gz on concept, approach, seed, feature).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.ml_geometry import tokens as TK
from experiments.paper_analysis.ml_geometry.common import OUT, SEEDS, TOP_N, TAU, concepts, fit_dir, script_of


def main():
    torch.set_num_threads(4)
    rows = []
    for c in concepts():
        tk = TK.load(c)
        for s in SEEDS:
            for a in ("A2", "A3"):
                d = fit_dir(c, a, s)
                fit = torch.load(d / "fit.pt", weights_only=False, map_location="cpu")
                sel = json.loads((d / "selection.json").read_text())
                Y = fit["Y"].float()
                vstar = fit["vstar_ids"]
                codes = np.asarray(fit["codes"]).astype(int)
                for cand in sel["candidates"]:
                    f = cand["feature"]
                    y = Y[f]
                    nz = torch.nonzero(y).flatten()
                    o = nz[torch.argsort(y[nz].abs(), descending=True)][:20].numpy()
                    ids = [int(vstar[i]) for i in o]
                    P = TK.lang_probs(tk, ids)
                    rows.append({"concept": c, "approach": a, "seed": s, "feature": f,
                                 "t20_conc_only": float((codes[o] == 1).mean()),
                                 "t20_en": float((P[:, 0] >= 0.8).mean()),
                                 "t20_latin": float(np.mean([script_of(t) == "lat" for t in ids])),
                                 "t20_maxshare": float(P.mean(0).max())})
        print(c, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "top20.csv", index=False)


if __name__ == "__main__":
    main()
