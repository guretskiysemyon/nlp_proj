#!/usr/bin/env python
"""Whitened vs non-whitened (plain) side by side: fixed 1%, fixed 2%, m1, m3, rn — no refit, seeds 42-46.

    python -m experiments.varwidth.whitened_vs_plain [--cap 0.05]

Each cell is "whitened / non-whitened". Judged metrics come from summary.json; judge-free metrics track the concept
direction per matrix (reference = mean fixed-1% concept direction of the same concept and matrix), as in
focus_report.py. Writes results/varwidth/whitened_vs_plain.md.
"""
import argparse
import itertools

import numpy as np
import torch

from experiments.varwidth import focus_report as fr
from experiments.varwidth.common import RESULTS, VWConfig, slug

CONFIG_SPECS = [("fixed 1%", "fixed", {}), ("fixed 2%", "fixed", {"sparsity": 0.02}), ("m1", "m1", {}), ("m3", "m3", {}),
                ("rn", "rn", {})]
COLUMNS = [("kept", "Judge kept"), ("nfc", "\\|F_C\\| mean"), ("share", "Concept share"), ("width", "Width (±dev)"),
           ("neutral", "Neutral share"), ("pure", "Pure"), ("dir_worst", "Dir worst"), ("jaccard", "Jaccard")]


def cells(concept, matrix, configs):
    """{config name: {column: formatted cell}} for one concept and matrix."""
    codes = torch.load(RESULTS / "prep" / f"{slug(concept)}_{matrix}_S300.pt", weights_only=False)["codes"]
    runs = {name: [r for r in (fr.load(concept, tag, matrix, s) for s in fr.SEEDS) if r] for name, tag in configs}
    ref = [r["Zn"][:, fr.concept_feature(r)] for r in runs["fixed 1%"] if fr.concept_feature(r) is not None]
    if not ref:  # no fixed-1% run kept the concept: fall back to any judged concept feature of any config
        ref = [r["Zn"][:, fr.concept_feature(r)] for rs in runs.values() for r in rs if fr.concept_feature(r) is not None]
    u = torch.linalg.svd(torch.stack(ref, 1), full_matrices=False)[0][:, 0] if ref else None
    out = {}
    for name, _ in configs:
        rs = runs[name]
        if not rs:
            out[name] = {}
            continue
        share = [r["summary"]["C"] / r["summary"]["n_distinct_tokens"] for r in rs if r["summary"]["n_distinct_tokens"]]
        d = {"kept": f"{sum(bool(r['summary']['F_C']) for r in rs)}/{len(rs)}",
             "nfc": f"{np.mean([len(r['summary']['F_C']) for r in rs]):.1f}",
             "share": f"{np.mean(share):.2f}" if share else "—"}
        if u is not None:
            W, N, Z, S = [], [], [], []
            for r in rs:
                i = int((u @ r["Zn"]).abs().argmax())
                sup = torch.nonzero(r["Y"][i]).flatten().numpy()
                W.append(len(sup)); N.append(int((codes[sup] == 0).sum())); Z.append(r["Zn"][:, i]); S.append(set(sup.tolist()))
            m = float(np.mean(W))
            cos = [float((a @ b).abs()) for a, b in itertools.combinations(Z, 2)]
            d.update(width=f"{m:.0f} (±{np.mean([abs(w - m) for w in W]):.0f})",
                     neutral=f"{100 * sum(N) / max(sum(W), 1):.0f}%", pure=f"{sum(n <= 5 for n in N)}/{len(N)}",
                     dir_worst=f"{min(cos):.2f}" if cos else "—", jaccard=f"{fr.jaccard_mean(S):.2f}")
        out[name] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=0.05, help="row cap of the variable-width configs (default 0.05, the cap of the saved runs)")
    cap = ap.parse_args().cap
    configs = [(name, VWConfig(method=m, refit=False, cap=cap, **kw).tag()) for name, m, kw in CONFIG_SPECS]
    L = [f"# Whitened vs non-whitened: fixed 1%, fixed 2%, m1, m3, rn (no refit, row cap {cap:.0%}, seeds 42–46)", "",
         "Each cell: **whitened / non-whitened**. Judge kept = seeds with non-empty F_C. Judge-free metrics track the "
         "concept direction (reference = mean fixed-1% concept direction of the same concept and matrix). Width = tokens "
         "(± typical deviation in tokens). Pure = seeds with ≤ 5 neutral tokens. Dir worst = lowest pairwise |cos z| "
         "across seeds. Concept share = C / distinct F_C tokens.", ""]
    for concept in fr.CONCEPTS:
        wh, pl = cells(concept, "whitened", configs), cells(concept, "plain", configs)
        L += [f"## {concept}", "", "| Config | " + " | ".join(h for _, h in COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
        for name, _ in configs:
            L.append(f"| {name} | " + " | ".join(f"{wh[name].get(k, '—')} / {pl[name].get(k, '—')}" for k, _ in COLUMNS) + " |")
        L.append("")
    out = RESULTS / "whitened_vs_plain.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
