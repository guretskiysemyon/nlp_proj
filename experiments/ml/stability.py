#!/usr/bin/env python
"""Sample-sensitivity check for A2/A3 (robustness, not part of the PREREG grid).

    python -m experiments.ml.stability sample  --concept ... --seeds 1 2 3
    python -m experiments.ml.stability fit     --concept ... --seeds 1 2 3 --approach A2 A3 [--device cpu]
    python -m experiments.ml.stability select  --concept ... --seeds 1 2 3 --approach A2 A3
    python -m experiments.ml.stability report  --concept ... --seeds 1 2 3

Redraws the N=30 sample with build_sets.subsample under a different Random seed (the existing
results/ml/data/sample/<slug>_N30.json files use data SEED=0), then runs fit.run / select.run unchanged, with
run_dir / load_sample / lexicon redirected (monkeypatched in this process only). Everything is written under
results/ml/stability/; the seed-0 reference is read from results/ml/runs/. The fit seed stays common.SEED=42.
The A3 lexicon is read from results/ml/lexicons.json (read only, never regenerated).
Judge verdicts are reused across runs (and from the seed-0 runs) when the top-20 token strings are identical.

Extension (fit-seed variance and A1): variants named by directory under results/ml/stability/runs/
    s1 s2 s3   sample seed 1/2/3, fit seed 42          (A2, A3)
    s0cpu      sample seed 0, fit seed 42, CPU          (A2, A3; A1 = full per-language sets)
    fs1 fs2    sample seed 0, fit seed 1/2, CPU         (A2, A3, A1)
    ref        results/ml/runs/ (GPU, seed 0 / fit 42), read only
    python -m experiments.ml.stability vfit    --concept ... --variants fs1 fs2 --approach A1 A2 A3
    python -m experiments.ml.stability vselect --concept ... --variants fs1 fs2 --approach A1 A2 A3 --cap-nis 0.48
    python -m experiments.ml.stability vreport --concept ...
vselect stops before a selection when this process's Gemini spend (from GEMINI_USAGE_LOG) reaches --cap-nis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import experiments.ml.fit as F
import experiments.ml.select as S
from experiments.ml.common import RESULTS, RUNS, TAU, TOP_K_TOKENS, TOP_N, save_json, slug

OUT = RESULTS / "stability"
N = 30
REF_SEED = 0          # seed of the existing sample files (experiments/ml/data/common.py SEED)


CPU_CTRL = -1         # pseudo-seed: the existing seed-0 sample refit on CPU (device-nondeterminism control)


def sample_path(concept, seed):
    if seed == CPU_CTRL:
        return F.DATA / "sample" / f"{slug(concept)}_N{N}.json"      # read only
    return OUT / "sample" / f"{slug(concept)}_N{N}_s{seed}.json"


def sdir(concept, seed, approach):
    return RUNS / slug(concept) / approach if seed == REF_SEED else OUT / "runs" / (f"s{seed}" if seed != CPU_CTRL else "s0cpu") / slug(concept) / approach


def _redirect(seed):
    F.run_dir = lambda c, a, lang=None: sdir(c, seed, a)
    S.run_dir = F.run_dir
    F.load_sample = lambda c, n: json.loads(sample_path(c, seed).read_text())
    _lexicon_ro()


# ---------------------------------------------------------------- variants (extension)
VARIANTS = {"s1": (1, 42), "s2": (2, 42), "s3": (3, 42), "s0cpu": (0, 42), "fs1": (0, 1), "fs2": (0, 2)}


def vdir(concept, variant, approach, lang=None):
    d = RUNS / slug(concept) / approach if variant == "ref" else OUT / "runs" / variant / slug(concept) / approach
    return d / lang if lang else d


def _vredirect(variant):
    sample_seed, fit_seed = VARIANTS[variant]
    F.run_dir = lambda c, a, lang=None: vdir(c, variant, a, lang)
    S.run_dir = F.run_dir
    F.SEED = fit_seed                     # fit._fit reads the module global at call time
    F.load_sample = lambda c, n: json.loads(sample_path(c, CPU_CTRL if sample_seed == 0 else sample_seed).read_text())
    _lexicon_ro()


def _lexicon_ro():
    def lexicon_ro(c):
        return json.loads(S.LEXICON_PATH.read_text())[c]["words"]
    S.lexicon = lexicon_ro


# ---------------------------------------------------------------- sample
def make_sample(concept, seed):
    import experiments.ml.data.build_sets as B
    p = sample_path(concept, seed)
    if seed == CPU_CTRL or p.exists():
        return
    old = B.SEED
    B.SEED = seed
    try:
        save_json(p, B.subsample(F.load_full(concept), N))
    finally:
        B.SEED = old


# ---------------------------------------------------------------- judge-cache seeding
def seed_judge_cache(concept, fit_path):
    """Pre-fill fit_path's judge.json for candidates whose top-20 token strings match an existing verdict for the
    same concept (verdicts are concept-specific). Returns (n_candidates, n_reused)."""
    from ember.utils import compute_embedding_stats
    fit = torch.load(fit_path, weights_only=False)
    Y, vstar = fit["Y"].float(), fit["vstar_ids"]
    t2c = dict(zip(vstar, [int(c) for c in fit["codes"]]))
    stats = compute_embedding_stats(Y.T, vstar, t2c, concept).sort_values("ratio_abs", ascending=False)
    feats = [int(f) for f in stats[stats["ratio_abs"] >= TAU].head(TOP_N)["feature"]]
    # index restricted to this concept: walk the concept's own run dirs only
    idx = {}
    for p in list((RUNS / slug(concept)).glob("A[23]/judge.json")) + \
            list((RUNS / slug(concept)).glob("A1/*/judge.json")) + \
            list((OUT / "runs").glob(f"*/{slug(concept)}/A[23]/judge.json")) + \
            list((OUT / "runs").glob(f"*/{slug(concept)}/A1/*/judge.json")):
        for e in json.loads(p.read_text()).values():
            if "verdict" in e:
                idx.setdefault(tuple(e["tokens"]), e)
    cache_path = fit_path.with_name("judge.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    reused = 0
    for f in feats:
        if str(f) in cache:
            continue
        y = Y[f]
        nz = torch.nonzero(y).flatten()
        order = nz[torch.argsort(y[nz].abs(), descending=True)]
        items = tuple(S.tok_str(vstar[int(i)]) for i in order[:TOP_K_TOKENS])
        if items in idx:
            cache[str(f)] = {**idx[items], "reused": True}
            reused += 1
    save_json(cache_path, cache)
    return len(feats), reused


# ---------------------------------------------------------------- report
def edited_tokens(fit_path, sel_path):
    """Token ids touched by F_C, as evaluate.run_edit: nonzero Y on a selected feature and code == 1."""
    fit = torch.load(fit_path, weights_only=False)
    F_C = json.loads(Path(sel_path).read_text())["F_C"]
    if not F_C:
        return set()
    Y = fit["Y"].float()[F_C]
    codes = torch.as_tensor(fit["codes"])
    cols = torch.nonzero((Y != 0).any(0) & (codes == 1)).flatten()
    return {int(fit["vstar_ids"][int(j)]) for j in cols}


def report(concepts, seeds, approaches):
    seeds = [REF_SEED] + [s for s in seeds if s != REF_SEED]
    rows = {}
    for c in concepts:
        for a in approaches:
            per = {}
            for s in seeds:
                d = sdir(c, s, a)
                if not (d / "selection.json").exists():
                    continue
                sel = json.loads((d / "selection.json").read_text())
                summ = json.loads((d / "fit_summary.json").read_text())
                per[s] = {"F_C": sel["F_C"], "n_tau": sel["n_tau"],
                          "tokens": sorted(edited_tokens(d / "fit.pt", d / "selection.json")),
                          "rel_err_sample": summ["rel_err_sample"], "rel_err_full": summ["rel_err_full"],
                          "n_sample": summ["n_sample"],
                          "selected": [{k: x[k] for k in ("feature", "rho", "description", "TJ", "EJ", "width")}
                                       | ({"ej_matches": x["ej"]["matches"]} if "ej" in x else {})
                                       for x in sel["candidates"] if x["selected"]],
                          "n_TJ": sum(x["TJ"] for x in sel["candidates"]),
                          "n_EJ": sum(x["EJ"] for x in sel["candidates"])}
            jac = {}
            ks = list(per)
            for i, s in enumerate(ks):
                for t in ks[i + 1:]:
                    A, B = set(per[s]["tokens"]), set(per[t]["tokens"])
                    jac[f"{s}-{t}"] = (len(A & B) / len(A | B)) if (A | B) else None
            rows[f"{c}|{a}"] = {"per_seed": per, "jaccard": jac}
    save_json(OUT / "report.json", rows)
    return rows


def spend_nis(log_path, from_line=0):
    """NIS of all log lines after `from_line` (conservative: counts every writer of the shared log)."""
    tot = 0.0
    with open(log_path) as fh:
        for i, line in enumerate(fh):
            if i >= from_line:
                tot += json.loads(line)["nis"]
    return tot


def _jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else None


def vreport(concepts, variants=("ref", "s0cpu", "fs1", "fs2")):
    from experiments.ml.common import LANGS
    from experiments.ml.data.io import page_priority
    ext = {}
    for c in concepts:
        for a in ("A2", "A3"):
            per = {}
            for v in variants:
                d = vdir(c, v, a)
                if not (d / "selection.json").exists():
                    continue
                sel = json.loads((d / "selection.json").read_text())
                summ = json.loads((d / "fit_summary.json").read_text())
                per[v] = {"F_C": sel["F_C"], "tokens": sorted(edited_tokens(d / "fit.pt", d / "selection.json")),
                          "rel_err_sample": summ["rel_err_sample"], "rel_err_full": summ["rel_err_full"],
                          "selected": [{k: x[k] for k in ("feature", "rho", "description", "TJ", "EJ")}
                                       for x in sel["candidates"] if x["selected"]]}
            ks = list(per)
            ext[f"fitseed|{c}|{a}"] = {"per_variant": per, "jaccard": {
                f"{u}-{w}": _jac(per[u]["tokens"], per[w]["tokens"]) for i, u in enumerate(ks) for w in ks[i + 1:]}}
        per = {}
        prio = page_priority(c)
        for v in variants:
            langs, merged, src = {}, set(), {}
            for L in prio:                                   # evaluate.edits order: first language keeps the token
                d = vdir(c, v, "A1", L)
                if not (d / "selection.json").exists():
                    langs = None
                    break
                sel = json.loads((d / "selection.json").read_text())
                summ = json.loads((d / "fit_summary.json").read_text())
                toks = edited_tokens(d / "fit.pt", d / "selection.json")
                for t in toks:
                    src.setdefault(t, L)
                merged |= toks
                langs[L] = {"F_C": sel["F_C"], "n_tok": len(toks), "rel_err": summ["rel_err"],
                            "selected": [x["description"] for x in sel["candidates"] if x["selected"]]}
            if langs is None:
                continue
            per[v] = {"langs": {L: langs[L] for L in LANGS}, "merged_tokens": sorted(merged),
                      "edited_by_lang": {L: sum(x == L for x in src.values()) for L in LANGS}}
        ks = list(per)
        ext[f"A1|{c}"] = {"per_variant": per, "jaccard_merged": {
            f"{u}-{w}": _jac(per[u]["merged_tokens"], per[w]["merged_tokens"]) for i, u in enumerate(ks) for w in ks[i + 1:]},
            "jaccard_per_lang": {f"{u}-{w}": {L: _jac(_lang_tokens(c, u, L), _lang_tokens(c, w, L)) for L in LANGS}
                                 for i, u in enumerate(ks) for w in ks[i + 1:]}}
    path = OUT / "report.json"
    rep = json.loads(path.read_text()) if path.exists() else {}
    rep["extension"] = ext
    save_json(path, rep)
    return ext


def _lang_tokens(c, v, L):
    d = vdir(c, v, "A1", L)
    return edited_tokens(d / "fit.pt", d / "selection.json")


def vmain(a):
    import os
    log = os.environ["GEMINI_USAGE_LOG"]
    for v in a.variants:
        assert v in VARIANTS, v
        _vredirect(v)
        for c in a.concept:
            full = F.load_full(c) if a.stage == "vselect" else None
            for ap_ in a.approach:
                if a.stage == "vfit":
                    F.run(c, ap_, N, a.device)
                    continue
                if ap_ == "A1":
                    from experiments.ml.common import LANGS
                    jobs = [(vdir(c, v, "A1", L) / "fit.pt", *F._flatten(full, [L])[::2], False) for L in LANGS]
                else:
                    s_, _, ls_ = F._flatten(full)
                    jobs = [(vdir(c, v, ap_) / "fit.pt", s_, ls_, ap_ == "A3")]
                for fit_path, sents, ls_, use_ej in jobs:
                    if fit_path.with_name("selection.json").exists() or \
                            not fit_path.with_name("fit_summary.json").exists():   # done, or fit not finished yet
                        continue
                    spent = spend_nis(log, a.log_from)
                    if spent >= a.cap_nis:
                        print(f"BUDGET STOP at {spent:.4f} NIS before {fit_path}", flush=True)
                        return
                    n, r = seed_judge_cache(c, fit_path)
                    sel = S.select_run(c, fit_path, sents, ls_, use_ej=use_ej)
                    print(f"[{v} {ap_} {c} {fit_path.parent.name}] reused {r}/{n} F_C={sel['F_C']} "
                          f"spent={spend_nis(log, a.log_from):.4f}", flush=True)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("vfit", "vselect", "vreport"):
        ap = argparse.ArgumentParser()
        ap.add_argument("stage", choices=["vfit", "vselect", "vreport"])
        ap.add_argument("--concept", nargs="+", required=True)
        ap.add_argument("--variants", nargs="+", default=["fs1", "fs2"])
        ap.add_argument("--approach", nargs="+", default=["A2", "A3"], choices=["A1", "A2", "A3"])
        ap.add_argument("--device", default="cpu")
        ap.add_argument("--cap-nis", type=float, default=0.48)
        ap.add_argument("--log-from", type=int, default=0, help="usage-log line where this budget starts")
        a = ap.parse_args()
        if a.stage == "vreport":
            vreport(a.concept)
        else:
            vmain(a)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "fit", "select", "report"])
    ap.add_argument("--concept", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--approach", nargs="+", default=["A2", "A3"], choices=["A2", "A3"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.stage == "report":
        report(a.concept, a.seeds, a.approach)
        return
    for s in a.seeds:
        assert s != REF_SEED, "seed 0 is the existing reference run"
        _redirect(s)
        for c in a.concept:
            if a.stage == "sample":
                make_sample(c, s)
                continue
            for ap_ in a.approach:
                if a.stage == "fit":
                    F.run(c, ap_, N, a.device)
                else:
                    n, r = seed_judge_cache(c, sdir(c, s, ap_) / "fit.pt")
                    print(f"[s{s}] judge cache: {r}/{n} candidates reused", flush=True)
                    S.run(c, ap_)


if __name__ == "__main__":
    main()
