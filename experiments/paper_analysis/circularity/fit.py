"""WP7 fits: sparse MF of the raw matrix or of the CS matrix built from a chosen target, eta and gamma.

    python -m experiments.paper_analysis.circularity.fit --plan main [--device cuda] [--shard 0/1] [--method fixed]
    python -m experiments.paper_analysis.circularity.fit --plan etagamma [--seeds 42]
    python -m experiments.paper_analysis.circularity.fit --check        # X(page, 0.05, 1) == cached whitened A?

Plans:
    main      18 paper concepts x {disjoint, related, leakfree, wrong, pagesub} x seeds 42-44, at eta = 0.05,
              gamma = 1 (raw and page at the default are the cached L0 / L1 fits: --import-cached)
    etagamma  15 paper concepts outside the tuning trio x eta {0.05, 0.2, 0.5} x gamma {0.5, 1, 2}, page target
Fit: experiments/varwidth/core.fit_varwidth with VWConfig(method="fixed", refit=False) (RS-1%, tag fixed_refit-False,
EMBER's rule) or --method rn (GS, tag rn_refit-False). Output: OUT/fits/<Slug>/<cond>/seed<N>.pt with the sparse Y
(indices + values), rel_err, n_iter, stop_reason, elapsed and target diagnostics. Z is not stored (refits are
deterministic given the seed and device and take seconds).
"""
from __future__ import annotations

import argparse
import itertools
import time

import torch

from experiments.paper_analysis.circularity import metrics
from experiments.paper_analysis.circularity.common import (DEFAULT_ETA, DEFAULT_GAMMA, ETA_GRID, GAMMA_GRID, OUT,
                                                         FITS, PAPER_CONCEPTS, SEEDS, TARGETS, TEXTS, TUNING_TRIO,
                                                         TransformBuilder, cond_name, fit_path, load_prep, read_json,
                                                         slug)


def plan(name: str, seeds, method: str) -> list[tuple]:
    """List of (concept, target, eta, gamma, method, seed)."""
    jobs = []
    if name == "main":
        for c in PAPER_CONCEPTS:
            for t in [x for x in TARGETS if x != "page"]:     # raw and page: imported from the cached L0/L1 fits
                for s in seeds:
                    jobs.append((c, t, DEFAULT_ETA, DEFAULT_GAMMA, method, s))
    elif name == "etagamma":
        for c in [x for x in PAPER_CONCEPTS if x not in TUNING_TRIO]:
            for eta, g in itertools.product(ETA_GRID, GAMMA_GRID):
                if (eta, g) == (DEFAULT_ETA, DEFAULT_GAMMA):
                    continue                                    # = the cached L1 fit (imported)
                for s in seeds:
                    jobs.append((c, "page", eta, g, method, s))
    elif name == "etagamma_related":     # the same grid with the non-circular "related" target (extension)
        for c in [x for x in PAPER_CONCEPTS if x not in TUNING_TRIO]:
            for eta, g in itertools.product(ETA_GRID, GAMMA_GRID):
                for s in seeds:
                    jobs.append((c, "related", eta, g, method, s))
    else:
        raise ValueError(name)
    return jobs


def save_fit(path, fit, extra):
    Y = fit["Y"].detach().float().cpu()
    nz = torch.nonzero(Y, as_tuple=False).to(torch.int32)
    out = {"Y_idx": nz, "Y_val": Y[nz[:, 0].long(), nz[:, 1].long()].clone(), "shape": tuple(Y.shape),
           "rel_err": float(fit["rel_err"]), "n_iter": fit.get("n_iter"), "stop_reason": fit.get("stop_reason"),
           "elapsed": float(fit.get("elapsed", 0.0)), "config": fit.get("config"), "seed": fit.get("seed"), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(out, tmp)
    tmp.replace(path)


def dense_top(fit, A, m, n_top: int = 5) -> dict:
    """float16 dense codes of the n_top features by (lexicon hits, rho), for re-scoring without a refit."""
    Z = fit["Z"].detach().double().cpu()
    G = Z.T @ Z + metrics.RIDGE * torch.eye(Z.shape[1], dtype=torch.float64)
    D = torch.linalg.solve(G, Z.T @ A.detach().double().cpu())
    feats = sorted(range(len(m["rho_all"])), key=lambda i: (m["hits_all"][i], m["rho_all"][i]), reverse=True)[:n_top]
    return {"features": feats, "codes": D[feats].to(torch.float16)}


CACHED_TAG = {("raw", "fixed"): "fixed_refit-False_plain", ("page", "fixed"): "fixed_refit-False",
              ("raw", "rn"): "rn_refit-False_plain", ("page", "rn"): "rn_refit-False"}


def import_cached(concepts, seeds, method: str):
    """raw / page(default eta, gamma) = the existing L0 / L1 (or GS) fits of results/varwidth: same code, same
    config, same seeds (a GPU refit of Ancient Rome seed 42 reproduced the cached L1 fit exactly). Metrics only."""
    from experiments.varwidth.common import run_dir
    for c in concepts:
        lab = torch.load(OUT / "labels" / f"{slug(c)}.pt", weights_only=False)
        for t in ("raw", "page"):
            A = load_prep(c, "plain" if t == "raw" else "whitened")["A"]
            ids = None if t == "raw" else read_json(TEXTS / "targets" / f"{slug(c)}.json")["ids"]["page"]
            for seed in seeds:
                out = fit_path(c, cond_name(t, DEFAULT_ETA, DEFAULT_GAMMA, method), seed)
                if out.exists():
                    continue
                src = run_dir(c, CACHED_TAG[(t, method)], seed) / "fit.pt"
                fit = torch.load(src, weights_only=False, map_location="cpu")
                assert fit["config"]["method"] == method and fit["config"]["refit"] is False and fit["seed"] == seed
                m = metrics.compute(fit["Y"], fit["Z"], A, lab, TARGETS)
                diag = {} if t == "raw" else {"n_target": len(ids)}
                save_fit(out, fit, {"concept": c, "target": t, "eta": DEFAULT_ETA, "gamma": DEFAULT_GAMMA,
                                    "method": method, "tag": CACHED_TAG[(t, method)], "diag": diag,
                                    "device": "cached", "source": str(src), "metrics": m,
                                    "dense_top": dense_top(fit, A, m)})
        print(f"imported {c} ({method})", flush=True)


def load_Y(path) -> tuple[torch.Tensor, dict]:
    d = torch.load(path, weights_only=False, map_location="cpu")
    Y = torch.zeros(d["shape"], dtype=torch.float32)
    idx = d["Y_idx"].long()
    Y[idx[:, 0], idx[:, 1]] = d["Y_val"]
    return Y, d


def claim(c: str, cond: str, seed: int) -> bool:
    """Per-(concept, condition, seed) claim so several workers (SLURM tasks, local processes) can share one plan.
    An old-style group claim (".claim", seeds 42-44) also counts."""
    import os
    if (FITS / slug(c) / cond / ".claim").exists() and seed in (42, 43, 44):
        return False
    lock = FITS / slug(c) / cond / f".claim_s{seed}"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.uname().nodename} {os.getpid()} {time.ctime()}".encode())
    os.close(fd)
    return True


def run(jobs, device: str, builder: TransformBuilder | None, reverse: bool = False):
    from experiments.varwidth.common import VWConfig
    from experiments.varwidth.core import fit_varwidth
    jobs = [j for j in jobs if not fit_path(j[0], cond_name(j[1], j[2], j[3], j[4]), j[5]).exists()]
    print(f"{len(jobs)} jobs to run on {device}", flush=True)
    key = lambda j: (j[0], j[1], j[2], j[3])  # noqa: E731
    jobs.sort(key=key, reverse=reverse)
    t_all = time.time()
    done = 0
    for (c, t, eta, g), grp in itertools.groupby(jobs, key=key):
        grp = [j for j in grp if not fit_path(c, cond_name(t, eta, g, j[4]), j[5]).exists()
               and claim(c, cond_name(t, eta, g, j[4]), j[5])]
        if not grp:
            continue
        t0 = time.time()
        prep = load_prep(c, "plain")
        if t == "raw":
            A, diag = prep["A"].float().contiguous(), {}
        else:
            ids = read_json(TEXTS / "targets" / f"{slug(c)}.json")["ids"][t]
            A, diag = builder.X(prep["A"].T.double(), ids, eta, g)
        t_build = time.time() - t0
        lab = torch.load(OUT / "labels" / f"{slug(c)}.pt", weights_only=False)
        for (_, _, _, _, method, seed) in grp:
            cfg = VWConfig(method=method, refit=False)
            fit = fit_varwidth(A, cfg, seed, device)
            m = metrics.compute(fit["Y"], fit["Z"], A, lab, TARGETS)
            save_fit(fit_path(c, cond_name(t, eta, g, method), seed), fit,
                     {"concept": c, "target": t, "eta": eta, "gamma": g, "method": method, "tag": cfg.tag(),
                      "diag": diag, "device": device, "metrics": m, "dense_top": dense_top(fit, A, m)})
            done += 1
            print(f"[{done}/{len(jobs)}] {c} {cond_name(t, eta, g, method)} s{seed}: rel_err={fit['rel_err']:.4f} "
                  f"iters={fit['n_iter']} ({fit['elapsed']:.1f}s fit, {t_build:.1f}s build) "
                  f"[{time.time() - t_all:.0f}s total]", flush=True)


def check(builder: TransformBuilder):
    """X for the default target must reproduce the cached whitened prep matrix (same code path as prep.py)."""
    for c in ["Harry Potter", "Golf", "Gun"]:
        plain, wh = load_prep(c, "plain"), load_prep(c, "whitened")
        ids = read_json(TEXTS / "targets" / f"{slug(c)}.json")["ids"]["page"]
        A, diag = builder.X(plain["A"].T.double(), ids, wh["eta"], wh["gamma"])
        rel = float(torch.linalg.norm(A - wh["A"]) / torch.linalg.norm(wh["A"]))
        print(f"{c}: |A - A_cached| / |A_cached| = {rel:.2e}; n_target {diag['n_target']} vs cached {wh['n_target']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="main")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--method", default="fixed", choices=["fixed", "rn"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--import-cached", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.import_cached:
        import_cached(a.concepts or PAPER_CONCEPTS, a.seeds, a.method)
        return
    builder = TransformBuilder()
    if a.check:
        check(builder)
        return
    jobs = plan(a.plan, a.seeds, a.method)
    if a.concepts:
        jobs = [j for j in jobs if j[0] in a.concepts]
    if a.targets:
        jobs = [j for j in jobs if j[1] in a.targets]
    i, n = (int(x) for x in a.shard.split("/"))
    groups = sorted({(j[0], j[1], j[2], j[3]) for j in jobs})
    mine = set(groups[i::n])
    run([j for j in jobs if (j[0], j[1], j[2], j[3]) in mine], a.device, builder, a.reverse)


if __name__ == "__main__":
    main()
