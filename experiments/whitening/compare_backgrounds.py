#!/usr/bin/env python3
"""Quick test from Instructions.txt: does the background (wiki types vs full vocabulary) matter?

Per model:
 1. principal-angle sines between the top-50 eigenvector subspaces of wiki vs full, wiki vs latin, full vs latin.
 2. per concept and background (wiki, full) at that background's chosen eps, the model's chosen eta, gamma = 1,
    target = page: the top-5 cPCA directions (Sharpener.P[:, :5]); the background's own whitened types are
    projected on each and the 15 largest |projection| tokens listed, marked * when the token is on the concept page.
 3. feature overlap: the rho >= 2 features of the wiki and full fits (gamma = 1 and gamma = 0) matched by Jaccard
    of their token supports (best match per wiki feature).
Output: reports/compare_backgrounds_<model>.md (+ data/<model>/compare_backgrounds.json).
Usage: python experiments/whitening/compare_backgrounds.py [--models gemma llama] [--threads 8]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import (CONCEPTS, REPORTS, Embeddings, decode, log, model_dir, principal_angle_sines_top, read_json, slug,
                    write_json)
from eta_sweep import run_fit
from sharpen import BackgroundWhitener, Sharpener

TOP_DIRS, TOP_TOK, TOP_EIG = 5, 15, 50


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def supports(model, concept, cfg_json):
    fit = torch.load(model_dir(model) / "fits" / slug(concept) / f"{cfg_json['config']}.pt")
    Y = fit["G"].T.abs()
    ids = fit["ids"]
    return {f: {ids[j] for j in torch.nonzero(Y[f] > 1e-12).flatten().tolist()} for f in cfg_json["tau_features"]}


def overlap(model, concept, r_wiki, r_full):
    sw, sf = supports(model, concept, r_wiki), supports(model, concept, r_full)
    best = []
    for f, s in sw.items():
        j, g = max(((jaccard(s, t), g) for g, t in sf.items()), default=(0.0, None))
        best.append({"wiki_feature": f, "full_feature": g, "jaccard": round(j, 3)})
    uw = set().union(*sw.values()) if sw else set()
    uf = set().union(*sf.values()) if sf else set()
    return {"n_tau_wiki": len(sw), "n_tau_full": len(sf), "matches": best,
            "mean_best_jaccard": float(np.mean([b["jaccard"] for b in best])) if best else 0.0,
            "n_match_ge_0.5": sum(b["jaccard"] >= 0.5 for b in best), "union_jaccard": round(jaccard(uw, uf), 3)}


def run(model: str):
    md = model_dir(model)
    out_json = md / "compare_backgrounds.json"
    if out_json.exists():
        res = read_json(out_json)
    else:
        res = {}
    d = read_json(md / "defaults.json")
    eps_sel = read_json(md / "eps_select.json")["backgrounds"]
    eps = {bg: eps_sel[bg]["chosen_eps_rel"] for bg in ("wiki", "full")}
    eta = d["eta"]
    stats = {bg: torch.load(md / f"bg_{bg}_stats.pt") for bg in ("wiki", "full", "latin")}

    if "angles" not in res:
        res["angles"] = {}
        for a, b in (("wiki", "full"), ("wiki", "latin"), ("full", "latin")):
            s = principal_angle_sines_top(stats[a], stats[b], TOP_EIG)
            res["angles"][f"{a}_vs_{b}"] = {"mean_sine": float(s.mean()), "max_sine": float(s.max()),
                                            "min_sine": float(s.min()), "n_sine_lt_0.5": int((s < 0.5).sum())}
        write_json(out_json, res)

    if "directions" not in res:
        emb = Embeddings(model)
        res["directions"] = {}
        try:
            for c in CONCEPTS:
                v = torch.load(md / "vstar" / f"{slug(c)}.pt")
                page = set(v["page_ids"])
                for bg in ("wiki", "full"):
                    bw = BackgroundWhitener(stats[bg], eps[bg])
                    sh = Sharpener(bw.whiten(v["E_page"]), eta, 1.0)
                    P = sh.P[:, :TOP_DIRS]
                    ids = read_json(md / f"bg_{bg}_types.json")
                    proj = torch.cat([bw.whiten(emb.rows(ids[i:i + 8192], torch.float64)) @ P
                                      for i in range(0, len(ids), 8192)])  # (N, 5)
                    dirs = []
                    for j in range(TOP_DIRS):
                        top = torch.topk(proj[:, j].abs(), TOP_TOK).indices.tolist()
                        tids = [ids[t] for t in top]
                        toks = decode(model, tids)
                        dirs.append({"gamma": float(sh.g[j]),
                                     "tokens": [{"token": tk, "in_page": tid in page,
                                                 "sign": "+" if proj[t, j] > 0 else "-"}
                                                for tk, tid, t in zip(toks, tids, top)],
                                     "frac_in_page": sum(tid in page for tid in tids) / TOP_TOK})
                    res["directions"][f"{c}|{bg}"] = dirs
                    log(f"[{model}/{c}/{bg}] cPCA in-page fractions: "
                        + " ".join(f"{x['frac_in_page']:.2f}" for x in dirs))
        finally:
            emb.free()
        write_json(out_json, res)

    if "overlap" not in res:
        res["overlap"] = {}
        for c in CONCEPTS:
            for gname, gamma in (("g1", 1.0), ("g0", 0.0)):
                rw = run_fit(model, c, "wiki", eps["wiki"], eta, gamma, "page")
                rf = run_fit(model, c, "full", eps["full"], eta, gamma, "page")
                o = overlap(model, c, rw, rf)
                o["wiki_metrics"] = {k: rw[k] for k in ("rho_max", "n_tau", "purity", "coverage", "auc")}
                o["full_metrics"] = {k: rf[k] for k in ("rho_max", "n_tau", "purity", "coverage", "auc")}
                res["overlap"][f"{c}|{gname}"] = o
                write_json(out_json, res)
    write_report(model, res, eps, eta)


def write_report(model, res, eps, eta):
    L = [f"# Background comparison: {model}", "",
         f"eps_rel: wiki={eps['wiki']:g}, full={eps['full']:g}; eta={eta:g}; gamma=1; target=page tokens", "",
         f"## 1. Top-{TOP_EIG} eigenvector subspaces (principal-angle sines)", "",
         "| pair | mean sine | max | min | #sines < 0.5 |", "|---|---|---|---|---|"]
    for k, a in res["angles"].items():
        L.append(f"| {k} | {a['mean_sine']:.3f} | {a['max_sine']:.3f} | {a['min_sine']:.3f} | {a['n_sine_lt_0.5']} |")
    L += ["", f"## 2. Top-{TOP_DIRS} cPCA directions: top-{TOP_TOK} background tokens by |projection| "
              "(* = token on the concept page)", ""]
    for c in CONCEPTS:
        L.append(f"### {c}")
        for bg in ("wiki", "full"):
            dirs = res["directions"][f"{c}|{bg}"]
            L.append(f"**{bg}** (mean in-page fraction {np.mean([x['frac_in_page'] for x in dirs]):.2f})")
            L.append("")
            for j, x in enumerate(dirs):
                toks = ", ".join(("*" if t["in_page"] else "") + repr(t["token"]) for t in x["tokens"])
                L.append(f"- p{j + 1} (γ={x['gamma']:.1f}, in-page {x['frac_in_page']:.2f}): {toks}")
            L.append("")
    L += ["## 3. Concept-feature overlap (rho >= 2 features, best Jaccard match of token supports wiki -> full)", "",
          "| concept | gamma | n_tau wiki | n_tau full | mean best J | #J>=0.5 | union J | AUC wiki / full |"
          " purity wiki / full |", "|---|---|---|---|---|---|---|---|---|"]
    for k, o in res["overlap"].items():
        c, g = k.split("|")
        L.append(f"| {c} | {g[1:]} | {o['n_tau_wiki']} | {o['n_tau_full']} | {o['mean_best_jaccard']:.2f} | "
                 f"{o['n_match_ge_0.5']} | {o['union_jaccard']:.2f} | {o['wiki_metrics']['auc']:.3f} / "
                 f"{o['full_metrics']['auc']:.3f} | {o['wiki_metrics']['purity']:.2f} / "
                 f"{o['full_metrics']['purity']:.2f} |")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"compare_backgrounds_{model}.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma", "llama"])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for m in args.models:
        t0 = time.time()
        run(m)
        log(f"[{m}] compare_backgrounds done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
