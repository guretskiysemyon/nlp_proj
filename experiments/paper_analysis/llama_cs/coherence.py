#!/usr/bin/env python
"""Judge-free semantic coherence of features, measured in the OTHER model's input-embedding space.

For a feature: its first 20 distinct normalized source words (as for the lexicon match, llama_cs.metrics); each word
is embedded with the other model's tokenizer (" " + word, pieces averaged) and background-whitened input embeddings;
coherence = mean pairwise cosine of these word vectors (features with < 5 words are skipped). Random word sets give
~0; topical sets (religion, sports, months) are clearly positive. Using the other model's space avoids scoring a fit
with the geometry it was fitted on.

Per fit: coh_top20 (mean over the 20 highest-rho features, the judge window), coh_all (all k features),
coh_cf (the lexicon-identified concept feature).

    python -m experiments.paper_analysis.llama_cs.coherence [--configs ...]
Output: results/paper_analysis/llama_cs/coherence.json  {"<model>|<concept>|<config>|<seed>": {...}}
"""
from __future__ import annotations

import argparse
import glob
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from experiments.paper_analysis.llama_cs.common import (FITS, OUT, ROOT, read_json, slug, tokenizer, vstar,
                                                        whitener, embeddings, write_json)
from experiments.paper_analysis.llama_cs.metrics import _word_ctx

OTHER = {"llama": "gemma", "gemma": "llama"}


@lru_cache(maxsize=200_000)
def word_vec(model: str, w: str):
    tok = tokenizer(model)
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    E = embeddings(model)
    bw = whitener(model)
    u = (E.index_select(0, torch.tensor(ids)).double() - bw.mu) @ bw.W
    v = u.mean(0)
    return (v / v.norm()).float()


def coherence(words, model):
    vs = [word_vec(model, w) for w in words]
    vs = [v for v in vs if v is not None]
    if len(vs) < 5:
        return None
    V = torch.stack(vs)
    C = V @ V.T
    n = len(vs)
    return float((C.sum() - C.diagonal().sum()) / (n * (n - 1)))


def fit_words(Y, ids, model, concept):
    from experiments.varwidth import overlap_judge as oj
    counts, cw, tok = _word_ctx(model, concept)
    mags = Y.abs()
    out = []
    for f in range(Y.shape[0]):
        nz = torch.nonzero(mags[f] > 1e-12).flatten()
        order = nz[torch.sort(mags[f][nz], descending=True, stable=True).indices].tolist()
        out.append(oj.feature_words([ids[j] for j in order], counts, n=20, tok_str=lambda t: tok.decode([t])))
    return out


def run_fit(model, concept, Y, ids, met):
    words = fit_words(Y, ids, model, concept)
    other = OTHER[model]
    coh = [coherence(w, other) for w in words]
    r = np.asarray(met["rho_all"])
    top = np.argsort(-r, kind="stable")[:20]
    vals = lambda fs: [coh[f] for f in fs if coh[f] is not None]
    return {"coh_top20": float(np.mean(vals(top))) if vals(top) else None,
            "coh_all": float(np.mean(vals(range(len(coh))))),
            "coh_cf": coh[met["cf"]], "coh_list": coh}


def main():
    from experiments.paper_analysis.llama_cs.fits import load_Y
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--gemma", action="store_true", help="also the existing Gemma raw / CS fits (seeds 42-44)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    path = OUT / "coherence.json"
    res = read_json(path) if path.exists() else {}
    n_new = 0
    for p in sorted(glob.glob(str(FITS / "llama" / "*" / "*" / "seed*" / "metrics.json"))):
        met = json.loads(Path(p).read_text())
        if a.configs and met["config"] not in a.configs:
            continue
        key = f"llama|{met['concept']}|{met['config']}|{met['seed']}"
        if key in res:
            continue
        Y = load_Y(Path(p).with_name("fit.pt"))
        ids = vstar("llama", met["concept"])["vstar_ids"]
        res[key] = {k: v for k, v in run_fit("llama", met["concept"], Y, ids, met).items() if k != "coh_list"}
        n_new += 1
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[key].items()}, flush=True)
        if n_new % 10 == 0:
            write_json(path, res, indent=None)
    if a.gemma:
        gref = read_json(OUT / "gemma_ref.json")
        tags = {"raw": "fixed_refit-False_plain", "cs_e0.05_g1": "fixed_refit-False"}
        for key0, met in gref.items():
            c, cfg, seed = key0.split("|")
            if int(seed) > 44 or (a.configs and cfg not in a.configs):
                continue
            key = f"gemma|{key0}"
            if key in res:
                continue
            prep = torch.load(ROOT / "results" / "varwidth" / "prep" / f"{slug(c)}_plain_S300.pt", map_location="cpu",
                              weights_only=False)
            Y = torch.load(ROOT / "results" / "varwidth" / slug(c) / tags[cfg] / f"seed{seed}" / "fit.pt",
                           map_location="cpu", weights_only=False)["Y"].float()
            res[key] = {k: v for k, v in run_fit("gemma", c, Y, list(prep["vstar_ids"]), met).items() if k != "coh_list"}
            print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[key].items()}, flush=True)
    write_json(path, res, indent=None)


if __name__ == "__main__":
    main()
