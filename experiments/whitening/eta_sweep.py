#!/usr/bin/env python3
"""Judge-free eta sweep: whiten (wiki background, chosen eps) + sharpen (target = concept page tokens, gamma = 1),
then the paper's sparse MF on the transformed V*, scored with label-based proxies.

V* is reproduced exactly as in the notebooks (cells 3.1.b/3.2.a): ConceptDataset(CONCEPT, neutral_sample_seed=42),
TransformerLens to_tokens = BOS + tokenizer(s, add_special_tokens=False) (left padding), V* = unique ids minus
{pad, bos, eos}; labels = ember.utils.build_token_label_codes (1 concept-only, 0 neutral-only, 2 both).

Configs per (model, concept): raw (E_V as in the paper, no centring), g0 (whitening only), and eta in ETA_GRID with
gamma = 1. Proxies per fit (Y = G^T, k x n):
    rho_max, n_tau (#features with rho >= 2)            from compute_embedding_stats(...).ratio_abs
    purity   = share of concept-only tokens in the support of the best-rho feature
    coverage = share of concept-only tokens in the union support of features with rho >= 2
    auc      = ROC AUC, concept-only (1) vs neutral-only (0), score = max |Y| over rho >= 2 features (0 if none)
    rel_err  = ||X - Z Y||_F / ||X||_F,  fit_s = fit time
ETA choice per model: for each concept rank the 5 eta configs on each of (auc, purity, coverage, rho_max)
(rank 1 = best, ties share the average rank), average over the 4 metrics x 3 concepts; the lowest mean rank wins,
ties -> smaller eta. Written to data/<model>/defaults.json.

Then (--judge) a Gemini confirmation at the chosen eta (and for reference on raw): the top-5 rho >= 2 features per
concept, 20 top |Y| tokens -> get_description -> classify_membership_with_desc; selected = is_member and conf >= 0.85.

Every fit is cached as data/<model>/fits/<slug>/<config>.{json,pt}; re-runs skip finished configs.
Usage: python experiments/whitening/eta_sweep.py --models gemma [--judge] [--threads 8]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import time

import numpy as np
import torch

from common import (CONCEPTS, DATA, MODELS, SEED, Embeddings, decode, excluded_ids, load_env, log, model_dir,
                    read_json, slug, tokenizer, write_json)
from sharpen import BackgroundWhitener, Sharpener, Transform

ETA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5]
TAU = 2.0
SPARSITY, RIDGE, MAX_ITER, PATIENCE = 0.01, 1e-4, 20_000, 200
EXPECTED_VSTAR = {("gemma", "Harry Potter"): 4780, ("gemma", "Ancient Rome"): 4760,
                  ("gemma", "Valentine's Day"): 4852, ("llama", "Harry Potter"): 5025}
METRICS = ["auc", "purity", "coverage", "rho_max"]
JUDGE_MODEL = "models/gemini-3.5-flash-lite"
JUDGE_TOP, JUDGE_TOKENS, JUDGE_CONF = 5, 20, 0.85


# ---------------------------------------------------------------- V* and cached rows
def vstar(model: str, concept: str):
    from ember.local_datasets import ConceptDataset
    from ember.utils import build_token_label_codes
    tok = tokenizer(model)
    ds = ConceptDataset(concept_name=concept, neutral_sample_seed=42)
    # TL: pad_token falls back to eos when the tokenizer has none (Llama); bos/pad/eos removed as in the notebook.
    special = {t for t in (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                           tok.bos_token_id, tok.eos_token_id) if t is not None}
    ids = set()
    for s, _ in ds:
        ids.update(tok(s, add_special_tokens=False)["input_ids"])
    ids.add(tok.bos_token_id)
    vs = sorted(int(t) for t in ids - special)
    codes = build_token_label_codes(concept, ds, tok)
    return vs, codes


def prep(model: str):
    """Cache E_V rows, labels, and page-target rows per concept (needs the embedding once)."""
    md = model_dir(model) / "vstar"
    md.mkdir(parents=True, exist_ok=True)
    todo = [c for c in CONCEPTS if not (md / f"{slug(c)}.pt").exists()]
    if not todo:
        return
    emb = Embeddings(model)
    try:
        ex = excluded_ids(model)
        for c in todo:
            vs, codes = vstar(model, c)
            exp = EXPECTED_VSTAR.get((model, c))
            log(f"[{model}/{c}] |V*| = {len(vs)} (notebook: {exp}) "
                f"{'OK' if exp in (None, len(vs)) else 'MISMATCH'}; special in V*: {len(set(vs) & ex)}")
            tgt = read_json(model_dir(model) / "targets" / f"{slug(c)}.json")
            torch.save({"ids": vs, "codes": [codes.get(t, 0) for t in vs], "E": emb.rows(vs),
                        "page_ids": tgt["page"], "E_page": emb.rows(tgt["page"]),
                        "pmw_ids": tgt["page_minus_wiki"], "E_pmw": emb.rows(tgt["page_minus_wiki"])
                        if tgt["page_minus_wiki"] else None}, md / f"{slug(c)}.pt")
    finally:
        emb.free()


# ---------------------------------------------------------------- one fit
def config_name(bg, eps, eta, gamma, target):
    if bg == "raw":
        return "raw"
    if gamma == 0:
        return f"{bg}_eps{eps:g}_g0"
    return f"{bg}_eps{eps:g}_eta{eta:g}_g{gamma:g}_{target}"


def transform_X(model, concept, bg, eps, eta, gamma, target):
    v = torch.load(model_dir(model) / "vstar" / f"{slug(concept)}.pt")
    if bg == "raw":
        return v, v["E"].double(), None
    stats = torch.load(model_dir(model) / f"bg_{bg}_stats.pt")
    bw = BackgroundWhitener(stats, eps)
    sh = None
    if gamma != 0:
        if target == "page":
            Et = v["E_page"]
        elif target == "page_minus_vstar":  # leak-free control: no V* token (hence no labelled token) in the target
            vs = set(v["ids"])
            Et = v["E_page"][[i not in vs for i in v["page_ids"]]]
        else:
            Et = v["E_pmw"]
        sh = Sharpener(bw.whiten(Et), eta, gamma)
    T = Transform(bw, sh)
    return v, T.forward(v["E"]), T


def proxies(Y: torch.Tensor, ids, codes, concept):
    from ember.utils import compute_embedding_stats
    from sklearn.metrics import roc_auc_score
    codes = np.asarray(codes)
    tok2code = dict(zip(ids, codes.tolist()))
    st = compute_embedding_stats(Y.T.contiguous(), ids, tok2code, concept)
    rho = st["ratio_abs"].to_numpy()
    A = Y.abs().numpy()  # (k, n)
    sel = np.where(rho >= TAU)[0]
    best = int(np.argmax(rho))
    supp = A[best] > 1e-12
    purity = float((codes[supp] == 1).mean()) if supp.any() else 0.0
    is_c = codes == 1
    union = (A[sel] > 1e-12).any(0) if len(sel) else np.zeros(A.shape[1], bool)
    coverage = float(union[is_c].mean()) if is_c.any() else 0.0
    score = A[sel].max(0) if len(sel) else np.zeros(A.shape[1])
    m = (codes == 1) | (codes == 0)
    auc = float(roc_auc_score((codes[m] == 1).astype(int), score[m]))
    order = np.argsort(-rho)
    return {"rho_max": float(rho.max()), "n_tau": int(len(sel)), "best_feature": best, "purity": purity,
            "coverage": coverage, "auc": auc, "rho_sorted_top10": [round(float(r), 3) for r in rho[order[:10]]],
            "tau_features": [int(f) for f in order if rho[f] >= TAU]}


def run_fit(model, concept, bg, eps=None, eta=None, gamma=1.0, target="page", force=False):
    from ember.utils import SparseMatrixFactorization, fit_with_ridge, set_seed
    name = config_name(bg, eps, eta, gamma, target)
    out = model_dir(model) / "fits" / slug(concept)
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{name}.json").exists() and not force:
        return read_json(out / f"{name}.json")
    v, X, T = transform_X(model, concept, bg, eps, eta, gamma, target)
    A = X.T.float().contiguous()
    set_seed(SEED)
    smf = SparseMatrixFactorization(MODELS[model]["rank"], fitting_device="cpu", g_sparsity=SPARSITY)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        ridge = fit_with_ridge(smf, A, MAX_ITER, patience=PATIENCE, base_reg=RIDGE)
    fit_s = time.time() - t0
    stop = [ln for ln in buf.getvalue().splitlines() if "Stopping early" in ln]
    Z, G = smf.F_.detach().float(), smf.G_.detach().float()
    Y = G.T.contiguous()
    rel = float(torch.linalg.matrix_norm(A - Z @ Y) / torch.linalg.matrix_norm(A))
    res = {"model": model, "concept": concept, "config": name, "bg": bg, "eps_rel": eps, "eta": eta,
           "gamma": 0.0 if bg == "raw" else gamma, "target": target if gamma else None, "n": len(v["ids"]),
           "rel_err": rel, "fit_s": round(fit_s, 1), "ridge": ridge, "stop": stop[-1] if stop else "max_iter",
           **proxies(Y, v["ids"], v["codes"], concept)}
    if T is not None and T.sh is not None:
        res["top_enrichment"] = [round(float(g), 3) for g in T.sh.g[:10]]
    torch.save({"Z": Z, "G": G, "ids": v["ids"]}, out / f"{name}.pt")
    write_json(out / f"{name}.json", res)
    log(f"[{model}/{concept}] {name}: rho_max={res['rho_max']:.2f} n_tau={res['n_tau']} purity={res['purity']:.2f} "
        f"cov={res['coverage']:.2f} auc={res['auc']:.3f} rel_err={rel:.3f} ({fit_s:.0f}s, {res['stop']})")
    return res


# ---------------------------------------------------------------- eta choice
def _ranks(vals):
    """Rank 1 = largest; ties get the average rank."""
    vals = np.asarray(vals, float)
    order = np.argsort(-vals, kind="stable")
    ranks = np.empty(len(vals))
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def choose_eta(model, eps, results):
    table, mean_rank = [], {e: [] for e in ETA_GRID}
    for c in CONCEPTS:
        rows = [results[(c, e)] for e in ETA_GRID]
        for m in METRICS:
            for e, r in zip(ETA_GRID, _ranks([row[m] for row in rows])):
                mean_rank[e].append(r)
    mr = {e: float(np.mean(v)) for e, v in mean_rank.items()}
    best = min(ETA_GRID, key=lambda e: (mr[e], e))
    for (c, key), r in results.items():
        table.append({k: r.get(k) for k in ("concept", "config", "eta", "gamma", "rho_max", "n_tau", "purity",
                                            "coverage", "auc", "rel_err", "fit_s")})
    return best, mr, table


def sweep(model):
    prep(model)
    eps = read_json(model_dir(model) / "eps_select.json")["backgrounds"]["wiki"]["chosen_eps_rel"]
    results = {}
    t_all = time.time()
    for c in CONCEPTS:
        results[(c, "raw")] = run_fit(model, c, "raw")
        results[(c, "g0")] = run_fit(model, c, "wiki", eps, None, 0.0)
        for e in ETA_GRID:
            results[(c, e)] = run_fit(model, c, "wiki", eps, e, 1.0, "page")
    best, mr, table = choose_eta(model, eps, results)
    rule = ("per concept, rank the 5 eta configs (gamma=1, target=page, wiki background) on each of "
            "auc, purity, coverage, rho_max (1 = best, ties averaged); mean over 4 metrics x 3 concepts; "
            "lowest mean rank wins, ties -> smaller eta")
    write_json(model_dir(model) / "defaults.json", {"eps_rel": eps, "eta": best, "gamma": 1.0, "background": "wiki",
                                                    "target": "page", "chosen_by": rule, "mean_rank": mr,
                                                    "table": table})
    log(f"[{model}] eta mean ranks {mr} -> eta={best} (sweep {time.time() - t_all:.0f}s)")
    return best


# ---------------------------------------------------------------- judge
def judge(model):
    load_env()
    from ember.interpret_features import SimpleGeminiClient, classify_membership_with_desc, get_description
    d = read_json(model_dir(model) / "defaults.json")
    path = model_dir(model) / "judge.json"
    cache = read_json(path) if path.exists() else {}
    client = None
    tok = tokenizer(model)
    for c in CONCEPTS:
        for name in (config_name("wiki", d["eps_rel"], d["eta"], 1.0, "page"), "raw"):
            res = read_json(model_dir(model) / "fits" / slug(c) / f"{name}.json")
            fit = torch.load(model_dir(model) / "fits" / slug(c) / f"{name}.pt")
            Y = fit["G"].T
            v = torch.load(model_dir(model) / "vstar" / f"{slug(c)}.pt")
            codes = np.asarray(v["codes"])
            for rank_i, f in enumerate(res["tau_features"][:JUDGE_TOP]):
                key = f"{slug(c)}|{name}|{f}"
                if key in cache:
                    continue
                mags = Y[f].abs()
                top = [j for j in torch.argsort(mags, descending=True)[:JUDGE_TOKENS].tolist() if mags[j] > 0]
                toks = [tok.decode([v["ids"][j]]) for j in top]
                if client is None:
                    client = SimpleGeminiClient(JUDGE_MODEL)
                desc = get_description(client, toks)
                is_mem, conf = classify_membership_with_desc(client, c, desc)
                cache[key] = {"concept": c, "config": name, "feature": f, "rho": res["rho_sorted_top10"][rank_i], "tokens": toks,
                              "labels": [int(codes[j]) for j in top], "description": desc, "is_member": is_mem,
                              "confidence": conf, "selected": bool(is_mem and (conf or 0) >= JUDGE_CONF)}
                write_json(path, cache)
                log(f"[{model}/{c}/{name}] f{f}: member={is_mem} conf={conf} :: {desc[:90]}")
    summary = {}
    for k, r in cache.items():
        s = summary.setdefault(f"{r['concept']}|{r['config']}", {"judged": 0, "selected": 0, "features": []})
        s["judged"] += 1
        s["selected"] += int(r["selected"])
        if r["selected"]:
            s["features"].append(r["feature"])
    write_json(model_dir(model) / "judge_summary.json", summary)
    for k, s in summary.items():
        log(f"[{model}] judge {k}: {s['selected']}/{s['judged']} selected {s['features']}")


def control(model):
    """Leak-free check at the chosen eta: the page target contains ~all concept-only V* tokens (labels leak into
    the proxies), so also sharpen with target = page tokens NOT in V*."""
    d = read_json(model_dir(model) / "defaults.json")
    rows = {}
    for c in CONCEPTS:
        r = run_fit(model, c, "wiki", d["eps_rel"], d["eta"], 1.0, "page_minus_vstar")
        v = torch.load(model_dir(model) / "vstar" / f"{slug(c)}.pt")
        r["target_size"] = len(set(v["page_ids"]) - set(v["ids"]))
        rows[c] = {k: r[k] for k in ("config", "target_size", "rho_max", "n_tau", "purity", "coverage", "auc",
                                     "rel_err")}
    d["control_page_minus_vstar"] = rows
    write_json(model_dir(model) / "defaults.json", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma", "llama"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--control", action="store_true", help="only the leak-free target control (after the sweep)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for m in args.models:
        if args.prep_only:
            prep(m)
            continue
        if args.control:
            control(m)
            continue
        sweep(m)
        if args.judge:
            judge(m)


if __name__ == "__main__":
    main()
