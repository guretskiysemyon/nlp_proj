#!/usr/bin/env python3
"""Choose eps_rel per model and background by split-half cross-validated whitening.

Split the background types in halves A/B (seeded). Fit W on A (sharpen.BackgroundWhitener), compute the covariance
of the whitened B (centred on its own mean): W Sigma_B W, error = ||W Sigma_B W - I||_F / sqrt(d). Same with A/B
swapped; average. Chosen eps_rel = the smallest eps whose error is within 10% of the minimum.

Sanity check of "noise": for every eps, the 15 background tokens with the largest whitened norm ||W (e - mu_B)||
(W fit on the whole background). With too small eps these are dominated by tokens that are extreme in the
smallest-variance (noisiest) directions.

Output: data/<model>/eps_select.json and data/<model>/eps_select.md.
Usage: python experiments/whitening/eps_select.py [--models gemma llama]
"""
from __future__ import annotations

import argparse
import math
import random
import time

import torch

from common import Embeddings, SEED, chunked_stats, decode, log, model_dir, read_json, write_json
from sharpen import BackgroundWhitener

EPS_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
BACKGROUNDS = ["wiki", "full"]
TOL = 0.10
TOP_TOK = 15


def cv_error(stats_fit: dict, stats_eval: dict, eps_rel: float) -> float:
    bw = BackgroundWhitener(stats_fit, eps_rel)
    cov_b = (stats_eval["Q"] * stats_eval["lam"]) @ stats_eval["Q"].T
    M = bw.W @ cov_b @ bw.W
    d = M.shape[0]
    return float(torch.linalg.matrix_norm(M - torch.eye(d, dtype=M.dtype)) / math.sqrt(d))


def top_norm_tokens(emb: Embeddings, ids: list[int], stats: dict, eps_rel: float, k: int = TOP_TOK):
    bw = BackgroundWhitener(stats, eps_rel)
    norms = []
    for i in range(0, len(ids), 8192):
        U = bw.whiten(emb.rows(ids[i:i + 8192], torch.float64))
        norms.append(U.norm(dim=1))
    norms = torch.cat(norms)
    top = torch.topk(norms, k).indices.tolist()
    return [{"id": ids[j], "token": decode(emb.model, [ids[j]])[0], "norm": round(float(norms[j]), 2)} for j in top], \
        float(norms.median())


def run(model: str):
    md = model_dir(model)
    out = md / "eps_select.json"
    if out.exists():
        log(f"[{model}] eps_select exists, skipping")
        return
    emb = Embeddings(model)
    res = {"eps_grid": EPS_GRID, "rule": f"smallest eps_rel with mean CV error <= (1+{TOL}) * min", "backgrounds": {}}
    try:
        for bg in BACKGROUNDS:
            t0 = time.time()
            ids = read_json(md / f"bg_{bg}_types.json")
            perm = random.Random(SEED).sample(ids, len(ids))
            h = len(perm) // 2
            sa, sb = chunked_stats(emb, perm[:h]), chunked_stats(emb, perm[h:2 * h])
            full = torch.load(md / f"bg_{bg}_stats.pt")
            rows = []
            for e in EPS_GRID:
                ab, ba = cv_error(sa, sb, e), cv_error(sb, sa, e)
                toks, med = top_norm_tokens(emb, ids, full, e)
                rows.append({"eps_rel": e, "err_AB": ab, "err_BA": ba, "err": (ab + ba) / 2,
                             "median_whitened_norm": med, "top_tokens": toks})
            emin = min(r["err"] for r in rows)
            chosen = min(r["eps_rel"] for r in rows if r["err"] <= (1 + TOL) * emin)
            res["backgrounds"][bg] = {"n_types": len(ids), "rows": rows, "chosen_eps_rel": chosen,
                                      "argmin_eps_rel": min(rows, key=lambda r: r["err"])["eps_rel"]}
            log(f"[{model}/{bg}] errors " + " ".join(f"{r['eps_rel']:g}:{r['err']:.3f}" for r in rows)
                + f" -> eps_rel={chosen:g} ({time.time() - t0:.0f}s)")
    finally:
        emb.free()
    write_json(out, res)
    lines = [f"# eps selection: {model}", "", res["rule"], ""]
    for bg, r in res["backgrounds"].items():
        lines += [f"## background = {bg} (n = {r['n_types']}); chosen eps_rel = {r['chosen_eps_rel']:g}", "",
                  "| eps_rel | err A->B | err B->A | mean | median ||Wx|| | top-15 tokens by whitened norm |",
                  "|---|---|---|---|---|---|"]
        for row in r["rows"]:
            toks = ", ".join(repr(t["token"]).replace("|", "\\|") for t in row["top_tokens"])
            lines.append(f"| {row['eps_rel']:g} | {row['err_AB']:.3f} | {row['err_BA']:.3f} | {row['err']:.3f} | "
                         f"{row['median_whitened_norm']:.1f} | {toks} |")
        lines.append("")
    (md / "eps_select.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma", "llama"])
    args = ap.parse_args()
    for m in args.models:
        t0 = time.time()
        run(m)
        log(f"[{m}] eps_select done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
