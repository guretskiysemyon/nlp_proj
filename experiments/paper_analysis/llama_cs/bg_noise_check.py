#!/usr/bin/env python
"""Speculation test: is Llama's weak page signal (few eigenvalues of M above the noise null) caused by a noisier
background estimate (60,467 types for d = 4096, d/n_B = 0.068, vs Gemma 107,472 for d = 2304, 0.021)?
Re-estimate GEMMA's background from a random subset with Llama's ratio (n_B = 2304 / 0.0677 ~ 34,000 types) and
recompute n_signal (eigenvalues of the page M above a same-size random-type null) for the 4 fit concepts.

    python -m experiments.paper_analysis.llama_cs.bg_noise_check [--device cuda:0]
Output: results/paper_analysis/llama_cs/spectra/bg_noise_check.json
"""
from __future__ import annotations

import argparse
import random

import torch

from experiments.paper_analysis.llama_cs.common import DATA, FIT_CONCEPTS, OUT, embeddings, page_target, read_json, write_json


def stats(E, ids, dev):
    X = E.index_select(0, torch.tensor(ids)).to(dev).double()
    mu = X.mean(0)
    Xc = X - mu
    lam, Q = torch.linalg.eigh(Xc.T @ Xc / len(ids))
    return mu, lam.clamp_min(0), Q


def n_signal(E, mu, W, tgt, pool, dev):
    U = (E.index_select(0, torch.tensor(tgt)).to(dev).double() - mu) @ W
    lam = torch.linalg.eigvalsh(U.T @ U / len(tgt))
    nul = random.Random(1).sample(pool, len(tgt))
    U0 = (E.index_select(0, torch.tensor(nul)).to(dev).double() - mu) @ W
    lam0 = torch.linalg.eigvalsh(U0.T @ U0 / len(nul))
    return int((lam > lam0[-1]).sum()), float(lam[-1]), float(lam0[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    E = embeddings("gemma")
    types = read_json(DATA / "bg_wiki_gemma_types.json")
    d = E.shape[1]
    q_llama = 4096 / 60467
    out = {}
    for name, ids in (("full_107k", types), ("sub_llama_ratio", random.Random(0).sample(types, round(d / q_llama)))):
        mu, lam, Q = stats(E, ids, a.device)
        W = (Q * (lam + 1e-4 * lam.mean()).pow(-0.5)) @ Q.T
        res = {}
        for c in FIT_CONCEPTS:
            tgt = page_target("gemma", c)
            ts = set(tgt)
            pool = [t for t in ids if t not in ts]
            res[c] = dict(zip(("n_signal", "lam1", "null_lam1"), n_signal(E, mu, W, tgt, pool, a.device)))
        out[name] = {"n_B": len(ids), "d_over_nB": d / len(ids), "concepts": res}
        print(name, out[name], flush=True)
    write_json(OUT / "spectra" / "bg_noise_check.json", out)


if __name__ == "__main__":
    main()
