#!/usr/bin/env python
"""Head-to-head of the focused set: fixed 1%, fixed 2%, m1-m4, all without refit, seeds 42-46.

    python -m experiments.varwidth.focus_report

One table per (concept, matrix). Judged metrics come from summary.json (F_C after tau + judge); judge-free metrics
track the concept direction in every fit: reference = mean direction of the fixed-1% concept features of the same
concept and matrix, tracked feature = argmax |cos| per run. Writes results/varwidth/focus_report.md.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.varwidth.common import RESULTS, VWConfig, slug

CONCEPTS = ["Ancient Rome", "Harry Potter", "Valentine's Day"]
MATRICES = ["whitened", "plain"]
ONLY_WHITENED = {"Valentine's Day"}
SEEDS = range(42, 47)
CONFIG_SPECS = [("fixed 1%", "fixed", {}), ("fixed 2%", "fixed", {"sparsity": 0.02}),
                ("m1 contribution", "m1", {}), ("m2 noise-normalized", "m2", {}), ("m3 entropy", "m3", {}),
                ("m4 L1 ALS", "m4", {}), ("rn row-normalized", "rn", {}), ("rnn row-normalized − noise", "rnn", {})]


def configs(cap: float) -> list:
    """(name, run tag) for the no-refit configs at row cap `cap` (fixed ignores the cap)."""
    return [(name, VWConfig(method=m, refit=False, cap=cap, **kw).tag()) for name, m, kw in CONFIG_SPECS]


CONFIGS = configs(VWConfig().cap)


def load(concept, tag, matrix, seed):
    d = RESULTS / slug(concept) / (tag + ("_plain" if matrix == "plain" else "")) / f"seed{seed}"
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    f = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)
    Z = f["Z"].float()
    return {"summary": s, "Zn": Z / Z.norm(dim=0, keepdim=True).clamp_min(1e-12), "Y": f["Y"].float(),
            "rel_err": float(f["rel_err"]), "elapsed": float(f["elapsed"]), "n_iter": int(f["n_iter"])}


def concept_feature(run):
    s = run["summary"]
    if not s["F_C"]:
        return None
    rho = {int(c["feature"]): c["ratio_abs"] for c in s["candidates"]}
    return max(s["F_C"], key=lambda i: rho.get(int(i), -1))


def jaccard_mean(sets):
    pairs = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if a | b]
    return float(np.mean(pairs)) if pairs else float("nan")


def ms(x, nd=1):
    x = [v for v in x if v == v]
    if not x:
        return "—"
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}" if len(x) > 1 else f"{x[0]:.{nd}f}"


def width_cells(x):
    """(mean, typical deviation, range) in tokens: deviation = mean |w - mean| over seeds."""
    x = [v for v in x if v == v]
    if not x:
        return "—", "—", "—"
    m = float(np.mean(x))
    dev = float(np.mean([abs(v - m) for v in x]))
    return f"{m:.0f}", f"±{dev:.0f}", (f"{min(x):.0f}–{max(x):.0f}" if len(x) > 1 else f"{x[0]:.0f}")


def table(concept, matrix, codes):
    runs = {name: [r for r in (load(concept, tag, matrix, s) for s in SEEDS) if r] for name, tag in CONFIGS}
    ref = [r["Zn"][:, concept_feature(r)] for r in runs["fixed 1%"] if concept_feature(r) is not None]
    u = torch.linalg.svd(torch.stack(ref, 1), full_matrices=False)[0][:, 0] if ref else None
    judged, free = [], []
    for name, _ in CONFIGS:
        rs = runs[name]
        if not rs:
            judged.append([name, 0] + ["—"] * 11)
            free.append([name, 0] + ["—"] * 10)
            continue
        nfc = [len(r["summary"]["F_C"]) for r in rs]
        cw = [r["summary"]["tokens_per_feature"].get(str(concept_feature(r)), np.nan) if concept_feature(r) is not None
              else np.nan for r in rs]
        sh = [r["summary"]["C"] / r["summary"]["n_distinct_tokens"] for r in rs if r["summary"]["n_distinct_tokens"]]
        sets = [set(int(j) for i in r["summary"]["F_C"] for j in torch.nonzero(r["Y"][int(i)]).flatten().tolist()) for r in rs]
        cwv = [w for w in cw if w == w]
        cv = np.std(cwv, ddof=1) / np.mean(cwv) if len(cwv) > 1 else float("nan")
        judged.append([name, len(rs), " ".join(f"{k}:{nfc.count(k)}" for k in sorted(set(nfc))), ms(cw),
                       f"{cv:.3f}" if cv == cv else "—", *width_cells(cw)[1:], ms(sh, 3),
                       ms([r["summary"]["C"] for r in rs]), ms([r["summary"]["N"] for r in rs]),
                       f"{jaccard_mean(sets):.3f}", ms([r["rel_err"] for r in rs], 4), ms([r["elapsed"] for r in rs], 0)])
        if u is None:
            continue
        W, C, N, Z, S = [], [], [], [], []
        for r in rs:
            i = int((u @ r["Zn"]).abs().argmax())
            sup = torch.nonzero(r["Y"][i]).flatten().numpy()
            c = codes[sup]
            W.append(len(sup)); C.append(int((c == 1).sum())); N.append(int((c == 0).sum())); Z.append(r["Zn"][:, i]); S.append(set(sup.tolist()))
        cos = [float((a @ b).abs()) for a, b in itertools.combinations(Z, 2)]
        free.append([name, len(rs), ms(W), f"{np.std(W, ddof=1) / np.mean(W):.3f}" if len(W) > 1 else "—", *width_cells(W)[1:],
                     ms(C), ms(N), f"{100 * sum(N) / max(sum(W), 1):.1f}%", f"{sum(n <= 5 for n in N)}/{len(N)}",
                     f"{np.mean(cos):.3f} / {min(cos):.3f}" if cos else "—", f"{jaccard_mean(S):.3f} ({len(set.intersection(*S))})"])
    return judged, free


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def main():
    import argparse
    global CONFIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=VWConfig().cap, help="row cap of the variable-width configs (default: VWConfig default)")
    cap = ap.parse_args().cap
    CONFIGS = configs(cap)
    L = [f"# Focused comparison: all methods without refit (seeds 42–46, row cap {cap:.0%})", "",
         "Configs: fixed 1% (the paper), fixed 2%, m1 contribution, m2 noise-normalized, m3 entropy, m4 L1 ALS — all "
         "without the refit step, 1% total budget unless noted. Whitened and non-whitened (plain) are separate tables.", "",
         "- **Judged** (after τ = 2 and the Gemini judge): |F_C| per run (count:runs), concept width = tokens of the "
         "highest-ρ F_C feature (mean over seeds; typical deviation = average number of tokens a seed's width is away from that mean; range = smallest–largest), concept share = C / distinct tokens of F_C, C and N tokens, "
         "token-set Jaccard across seeds.",
         "- **Judge-free**: the concept direction tracked in every fit (reference = mean of the fixed-1% concept "
         "directions of the same concept and matrix). Pure = at most 5 neutral tokens. Direction = pairwise |cos z| "
         "(mean / worst pair). Jaccard (n) = mean pairwise Jaccard of the tracked supports (tokens in all seeds).", ""]
    for concept in CONCEPTS:
        L += [f"## {concept}", ""]
        for matrix in MATRICES:
            if matrix != "whitened" and concept in ONLY_WHITENED:
                continue
            prep = RESULTS / "prep" / f"{slug(concept)}_{matrix}_S300.pt"
            codes = torch.load(prep, weights_only=False)["codes"]
            judged, free = table(concept, matrix, codes)
            L += [f"### {concept}, {'whitened' if matrix == 'whitened' else 'non-whitened'}", "", "**Judged**", ""]
            L += md(["Config", "Runs", "\\|F_C\\|", "Concept width", "Width CV", "Typical deviation (tokens)", "Range (tokens)", "Concept share", "C", "N",
                     "Jaccard", "rel_err", "Fit s"], judged) + [""]
            L += ["**Judge-free (tracked concept direction)**", ""]
            L += md(["Config", "Runs", "Width", "Width CV", "Typical deviation (tokens)", "Range (tokens)", "C", "N", "Neutral share", "Pure", "Direction mean / worst",
                     "Jaccard (in all)"], free) + [""]
    out = RESULTS / ("focus_report.md" if cap == VWConfig().cap else f"focus_report_cap{cap:g}.md")
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
