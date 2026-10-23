#!/usr/bin/env python
"""Sparse-MF fits on Llama-3.1-8B-Instruct English V* (300 concept + 300 neutral sentences): EMBER (raw E, RS-1%)
and CS (background-whitened, page-sharpened, RS-1%) over an (eta, gamma) grid, plus controls.

    python -m experiments.paper_analysis.llama_cs.fits --concepts "Harry Potter" ... --seeds 42 43 44 \
        --configs raw cs_e0.05_g1 ... [--device cuda:0]

Config names:
    raw                      A = E_{V*}^T (EMBER, k = 200, fixed width 1%, refit False)
    white                    A = ((E - mu_B) W)^T            (gamma = 0, whitening only)
    cs_e<eta>_g<gamma>       A = ((E - mu_B) W S_eta^{gamma/2})^T, target = English page token types
    cs_e<eta>_g<gamma>_ml8   same, target = Llama token types of the concept's 300 sentences in 8 languages
                             (the multilingual study's pooled target; m >> d)
    cs_e<eta>_g<gamma>_lf    same, leak-free target: page types minus every V* type
    cs_e<eta>_g<gamma>_sub<f> target = a seeded random fraction f of the page types
    cs_..._mp | cs_..._top<r>  spectral truncation: keep g_j = (1-eta) lam_j + eta for the top r directions of M and
                             set g = 1 (background level) elsewhere; mp: r = # eigenvalues above the largest eigenvalue
                             of a same-size random-type null (spectra.py)
    <any>_k<k>               same matrix, rank k instead of the model default (Llama 200, Gemma 100)
Model: --model llama (default) or gemma (for controls; k = 100).
Output: results/paper_analysis/llama_cs/fits/<model>/<slug>/<config>/seed<N>/{fit.pt, metrics.json}
        results/paper_analysis/llama_cs/fits/<model>/<slug>/transform_diag.json (energy shares per config)
"""
from __future__ import annotations

import argparse
import random
import re
import time

import numpy as np
import torch

from experiments.paper_analysis.llama_cs.common import (FITS, MODELS, OUT, background, embeddings, ml_pooled_target,
                                                        page_target, read_json, slug, vstar, whitener, write_json)
from experiments.paper_analysis.llama_cs.metrics import fit_metrics

GRID = [f"cs_e{e:g}_g{g:g}" for e in (0.05, 0.1, 0.2, 0.5) for g in (0.5, 1)]


def parse(cfg: str):
    m = re.fullmatch(r"cs_e([\d.]+)_g([\d.]+)(?:_(ml8|lf|sub[\d.]+))?(?:_(mp|top\d+))?", cfg)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), m.group(3) or "page", m.group(4)


class ConceptData:
    def __init__(self, model: str, concept: str, device: str):
        self.model, self.concept, self.device = model, concept, device
        self.v = vstar(model, concept)
        self.ids = self.v["vstar_ids"]
        E = embeddings(model)
        self.E_v = E.index_select(0, torch.tensor(self.ids)).to(device)          # (n, d) float32
        self._U = None
        self._eig = {}

    @property
    def U(self):
        if self._U is None:
            bw = whitener(self.model)
            self.W = bw.W.to(self.device)
            self.mu = bw.mu.to(self.device)
            self._U = (self.E_v.double() - self.mu) @ self.W
        return self._U

    def target_ids(self, kind: str):
        if kind == "page":
            return page_target(self.model, self.concept)
        if kind == "ml8":
            return ml_pooled_target(self.concept)
        if kind == "lf":
            vs = set(self.ids)
            return [t for t in page_target(self.model, self.concept) if t not in vs]
        if kind.startswith("sub"):
            pg = page_target(self.model, self.concept)
            f = float(kind[3:])
            return sorted(random.Random(0).sample(pg, round(f * len(pg))))
        raise ValueError(kind)

    def eig(self, kind: str):
        if kind not in self._eig:
            ids = self.target_ids(kind)
            E = embeddings(self.model)
            _ = self.U
            Ut = (E.index_select(0, torch.tensor(ids)).to(self.device).double() - self.mu) @ self.W
            M = Ut.T @ Ut / Ut.shape[0]
            lam, P = torch.linalg.eigh(M)
            self._eig[kind] = (lam.flip(0).clamp_min(0), P.flip(1), len(ids), set(ids))
        return self._eig[kind]

    def A(self, cfg: str) -> tuple[torch.Tensor, dict]:
        if cfg == "raw":
            return self.E_v.T.contiguous(), {}
        if cfg == "white":
            return self.U.T.float().contiguous(), {}
        eta, gamma, kind, trunc = parse(cfg)
        lam, P, m, tset = self.eig(kind)
        g = (1 - eta) * lam + eta
        r = None
        if trunc is not None:   # spectral truncation: directions outside the top r get g = 1 (background level)
            if trunc == "mp":   # r = # eigenvalues above the largest eigenvalue of a same-size random-type null
                lam0 = torch.load(OUT / "spectra" / f"spectra_{self.model}_{slug(self.concept)}.pt")["lam_null"]
                r = int((lam.cpu() > lam0[0]).sum())
            else:
                r = int(trunc[3:])
            g = torch.where(torch.arange(len(g), device=g.device) < r, g, torch.ones_like(g))
        Sp = (P * g.pow(gamma / 2)) @ P.T
        X = self.U @ Sp
        return X.T.float().contiguous(), {"eta": eta, "gamma": gamma, "target": kind, "m": m, "trunc_r": r,
                                           "d": int(lam.numel()), "on_target": [t in tset for t in self.ids]}


def energy_diag(A: torch.Tensor, codes: np.ndarray, on_target=None) -> dict:
    e = (A.double() ** 2).sum(0).cpu().numpy()
    out = {"energy_share_C": float(e[codes == 1].sum() / e.sum()), "count_share_C": float((codes == 1).mean()),
           "energy_share_N": float(e[codes == 0].sum() / e.sum()), "count_share_N": float((codes == 0).mean())}
    if on_target is not None:
        ot = np.asarray(on_target)
        out["energy_share_on_target"] = float(e[ot].sum() / e.sum())
        out["count_share_on_target"] = float(ot.mean())
    return out


def save_fit(path, fit, meta):
    Y = fit["Y"].float().cpu()
    nz = torch.nonzero(Y)
    torch.save({"Z": fit["Z"].half().cpu(), "Y_idx": nz.to(torch.int32), "Y_val": Y[nz[:, 0], nz[:, 1]],
                "Y_shape": tuple(Y.shape), "rel_err": float(fit["rel_err"]), "n_iter": int(fit["n_iter"]),
                "elapsed": float(fit["elapsed"]), "stop_reason": fit.get("stop_reason"), **meta}, path)


def load_Y(path) -> torch.Tensor:
    f = torch.load(path, map_location="cpu", weights_only=False)
    Y = torch.zeros(f["Y_shape"])
    Y[f["Y_idx"][:, 0].long(), f["Y_idx"][:, 1].long()] = f["Y_val"]
    return Y


def run(model, concepts, seeds, configs, device):
    from experiments.varwidth.common import VWConfig
    from experiments.varwidth.core import fit_varwidth
    pass
    for concept in concepts:
        cd = ConceptData(model, concept, device)
        cdir = FITS / model / slug(concept)
        dpath = cdir / "transform_diag.json"
        diag = read_json(dpath) if dpath.exists() else {}
        for cfg in configs:
            A = None
            for seed in seeds:
                out = cdir / cfg / f"seed{seed}"
                if (out / "metrics.json").exists():
                    continue
                base, kk = (re.fullmatch(r"(.*)_k(\d+)", cfg).groups() if re.fullmatch(r"(.*)_k(\d+)", cfg)
                            else (cfg, MODELS[model]["rank"]))
                if A is None:
                    A, meta = cd.A(base)
                    if cfg not in diag:
                        diag[cfg] = {**energy_diag(A, cd.v["codes"], meta.get("on_target")),
                                     **{k: v for k, v in meta.items() if k != "on_target"}}
                        write_json(dpath, diag)
                vcfg = VWConfig(method="fixed", rank=int(kk), sparsity=0.01, refit=False)
                t0 = time.time()
                fit = fit_varwidth(A, vcfg, seed=seed, device=device)
                out.mkdir(parents=True, exist_ok=True)
                save_fit(out / "fit.pt", fit, {k: v for k, v in meta.items() if k != "on_target"})
                met = fit_metrics(fit["Y"], cd.ids, cd.v["codes"], model, concept)
                met.update({"concept": concept, "config": cfg, "seed": seed, "model": model,
                            "rel_err": float(fit["rel_err"]), "n_iter": int(fit["n_iter"]),
                            "fit_s": float(fit["elapsed"])})
                write_json(out / "metrics.json", met, indent=None)
                print(f"[{model} {concept} {cfg} s{seed}] iters={fit['n_iter']} err={fit['rel_err']:.4f} "
                      f"n_tau={met['n_tau']} cf: lo={met['cf_lo']} rank={met['cf_rank']} rho={met['cf_rho']:.1f} "
                      f"pur={met['cf_purity']:.2f} auc={met['auc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        del cd
        torch.cuda.empty_cache() if device.startswith("cuda") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    ap.add_argument("--concepts", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--configs", nargs="+", default=["raw"] + GRID)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run(a.model, a.concepts, a.seeds, a.configs, a.device)


if __name__ == "__main__":
    main()
