#!/usr/bin/env python
"""CINDER features for one or more concepts, written in EMBER's file format (CPU + Gemini).

CINDER = Ori's ladder rung L123 (experiments/varwidth/layer_eval.py: RUNGS["L123"] = config L12 + judge HJ):
    L1  contrastive sharpening: whitened matrix (wiki background, page target, eta 0.05, gamma 1, eps_rel 1e-4)
    L2  GS-norm: VWConfig(method="rn", refit=False) -> tag rn_refit-False (floor 0.2%, cap 3%), seed 42
    L3  two-way judge HJ = TJ or LO@5 on the top-20 features by rho (select.JUDGE_TOP_N): TJ = EMBER's two-stage judge
        (gemini-3.5-flash-lite), LO = how many of the 20 concept words (j3_data/concept_words_v1.json, prompt_v1.txt)
        equal one of the feature's first 20 normalized source words (overlap_judge.feature_words, j3_eval.j3_count)

Per concept:
    1. whitening target if missing (experiments/varwidth/build_all_targets.py)
    2. experiments.varwidth.run --method rn --no-refit --seed 42 (skipped when the run exists: Ori's runs for the 6
       concepts were copied from his folder on 2026-10-18 and reproduce bit-for-bit up to 5e-6) -> results/varwidth/
       <concept>/rn_refit-False/seed42/{fit.pt, judge.json, summary.json}
    3. layer_eval.build_cache -> layer_cache.json (TJ, LO, edit ids per candidate); F_C = layer_eval.fc(cache, "HJ")
    4. EMBER format under mf_outputs/cinder/google_gemma-2-2b-it/ (rank100/seed42):
           pickles/.../embedding/embedding.pkl       {"nmf": .F_ (d, k), .G_ (|V*|, k), "vprime_token_ids"}
           csvs/.../embedding/token_features.csv     per feature: all nonzero tokens and their labels
           csvs/.../embedding/stats_embed.csv
           interpretations/.../embedding/from_activation.csv     every candidate with TJ / LO / HJ
           interpretations/.../embedding/potential_features.csv  the HJ-accepted features (what the edit uses)
       F maps the fit's directions back to embedding space: F = (Z^T T_inv)^T, G = Y^T, so
       e_t - mu_B ~ F G_t^T and EMBER's edit e_t <- e_t - delta F G_t^T is CINDER's edit (= erase.py's edit).
    5. judge_rerun/Cinder/features/<Concept>/: candidates.csv, summary.json, tj_judge_cache.json (tracked in git)

    python -m cinder.build_features [--concepts Cannabis Gun ...] [--force-fit]
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from cinder import common as C


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def ensure_target(concept: str) -> None:
    from experiments.varwidth import prep
    from experiments.varwidth.common import slug
    if not (prep.WH_DIR / "targets" / f"{slug(concept)}.json").exists():
        from experiments.varwidth import build_all_targets
        argv, sys.argv = sys.argv, ["build_all_targets", "--concepts", concept]
        try:
            build_all_targets.main()
        finally:
            sys.argv = argv


def ori_run(concept: str, force: bool) -> dict:
    from experiments.varwidth import run
    args = run.parse(["--concept", concept, "--method", C.CINDER_METHOD, "--no-refit", "--seed", str(C.SEED),
                      "--matrix", C.CINDER_MATRIX, "--no-widest"] + (["--force"] if force else []))
    out = run.run_one(args)
    return {"dir": out, "summary": json.loads((out / "summary.json").read_text()),
            "fit": torch.load(out / "fit.pt", weights_only=False)}


def two_way_judge(concept: str, run_dir) -> pd.DataFrame:
    """Ori's L123 judge on the top-20-by-rho candidates: TJ, LO (concept words v1), HJ = TJ or LO@5."""
    from experiments.varwidth.layer_eval import LO_T, build_cache, fc
    from experiments.varwidth.select import JudgeCache
    cache = build_cache(concept, "L12", C.SEED)
    hj = {c["feature"] for c in fc(cache, "HJ")}
    jt = JudgeCache(run_dir / "judge.json")
    rows = []
    for c in cache["candidates"]:
        f = int(c["feature"])
        v = jt.verdicts.get(f)
        rows.append({"feature": f, "rank": c["rank"], "ratio_abs": c["rho"], "width": c["width"],
                     "concept_share": c["share"], "description": jt.descriptions.get(f),
                     "is_member": None if v is None else v[0], "confidence": None if v is None else v[1],
                     "tj": bool(c["tj"]), "lo": int(c["lo"] or 0), "ej": int(c["lo"] or 0) >= LO_T,
                     "twj": f in hj, "feature_words": c["fw20"], "n_edit_ids": len(c["edit_ids"])})
    return pd.DataFrame(rows), cache


def write_ember_format(concept: str, prep: dict, fit: dict, cand: pd.DataFrame) -> dict:
    from ember.utils import collect_feature_rows_for_embeddings, compute_embedding_stats, get_pipeline_path
    from experiments.varwidth.prep import TokenizerOnly, get_tokenizer

    Z, Y = fit["Z"].double(), fit["Y"].float()
    T_inv = prep["T_inv"].double()
    F = (Z.T @ T_inv).T.float().contiguous()             # (d, k): column i = direction i in embedding space
    G = Y.T.contiguous()                                 # (|V*|, k)
    vstar = list(prep["vstar_ids"])
    assert int((Y != 0).sum(1).max()) <= 200, "a feature wider than collect_feature_rows' 200-token list"

    def p(kind, fname):
        return get_pipeline_path(str(C.MF_ROOT), C.MODEL_SAFE, kind, C.RANK, C.safe(concept), "embedding", fname,
                                 seed=C.SEED)

    with open(p("pickles", "embedding.pkl"), "wb") as fh:
        pickle.dump({"nmf": SimpleNamespace(F_=F, G_=G), "vprime_token_ids": vstar, "source": "cinder",
                     "method": C.CINDER_METHOD, "matrix": C.CINDER_MATRIX, "seed": C.SEED,
                     "mu_B": prep["mu_B"].float()}, fh)

    label = {tid: "both" if c == 2 else concept if c == 1 else "Neutral" for tid, c in prep["token_to_code"].items()}
    rows = collect_feature_rows_for_embeddings(G, vstar, label, TokenizerOnly(get_tokenizer()), C.MODEL_SAFE,
                                               concept, C.RANK)
    pd.DataFrame(rows).to_csv(p("csvs", "token_features.csv"), index=False)
    compute_embedding_stats(G, vstar, prep["token_to_code"], concept).to_csv(p("csvs", "stats_embed.csv"), index=False)

    from experiments.varwidth.select import TOP_K_TOKENS
    from experiments.varwidth.judge_compare import row_order
    tok_str = get_tokenizer().convert_ids_to_tokens
    judge_tokens = {int(f): [str(t).replace("▁", " ") for t in tok_str([vstar[j] for j in row_order(Y, int(f))[:TOP_K_TOKENS]])]
                    for f in cand["feature"]}
    interp = pd.DataFrame({
        "model": C.MODEL_NAME, "concept": concept, "rank": C.RANK, "layer": 0,
        "feature": cand["feature"].astype(int), "metric_score": cand["ratio_abs"], "source": "activating_tokens",
        "n_tokens_used": TOP_K_TOKENS, "tokens_provided": cand["feature"].map(lambda f: json.dumps(judge_tokens[int(f)])),
        "description": cand["description"], "is_member": cand["is_member"], "confidence": cand["confidence"],
        "tj": cand["tj"], "lo": cand["lo"], "ej": cand["ej"],
        "feature_words": cand["feature_words"].apply(json.dumps), "twj": cand["twj"]})
    interp.to_csv(p("interpretations", "from_activation.csv"), index=False)
    interp[interp["twj"].astype(bool)].to_csv(p("interpretations", "potential_features.csv"), index=False)

    # Mapping check: the fit's reconstruction mapped back equals F G^T (exact up to float error), and its error
    # in embedding space (centred on mu_B).
    A = prep["A"].double()                                             # (d, n) = X^T
    back = (Z @ Y.double()).T @ T_inv                                  # (n, d)
    FGt = (F.double() @ G.double().T).T
    E_c = A.T @ T_inv                                                  # e - mu_B, from X = (e - mu_B) M
    return {"mapping_max_abs_diff": float((back - FGt).abs().max()),
            "rel_err_X": float(fit["rel_err"]),
            "rel_err_embedding_space": float(torch.linalg.norm(E_c - FGt) / torch.linalg.norm(E_c))}


def build(concept: str, force_fit: bool) -> dict:
    from experiments.varwidth.prep import prepare
    t0 = time.time()
    ensure_target(concept)
    prep = prepare(concept, C.CINDER_MATRIX)
    run = ori_run(concept, force_fit)
    fit, summ = run["fit"], run["summary"]
    cand, cache = two_way_judge(concept, run["dir"])
    low = cand[cand["twj"] & (cand["ratio_abs"] < 2.0)]
    if len(low):   # the edit keeps metric_score >= 2 (EMBER's ratio_thresh): an accepted feature below it would be lost
        raise ValueError(f"{concept}: accepted features with rho < 2: {low['feature'].tolist()}")
    check = write_ember_format(concept, prep, fit, cand)

    C.use_cinder_features()
    from ember.erasure.embed_edit import concept_edit_tokens
    edited = concept_edit_tokens(C.MODEL_NAME, concept, C.RANK, C.SEED, ratio_thresh=2.0) if cand["twj"].any() else []
    ori_edit = sorted({t for c in cache["candidates"] if c["feature"] in set(cand.loc[cand.twj, "feature"])
                       for t in c["edit_ids"]})

    fdir = C.FEATURES_OUT / C.safe(concept)
    fdir.mkdir(parents=True, exist_ok=True)
    cand.to_csv(fdir / "candidates.csv", index=False)
    shutil.copy2(run["dir"] / "judge.json", fdir / "tj_judge_cache.json")
    from experiments.varwidth.j3_words import concept_lists
    out = {
        "concept": concept, "method": "CINDER = Ori's L123 (CS whitened + GS-norm rn refit=False + HJ = TJ or LO@5)",
        "seed": C.SEED, "config": summ["config"], "matrix": C.CINDER_MATRIX, "vstar": summ["vstar"],
        "judge_rule": summ.get("judge_rule"), "n_tau": cache.get("n_tau"),
        "eta": prep.get("eta"), "eps_rel": prep.get("eps_rel"), "gamma": prep.get("gamma"),
        "n_target": prep.get("n_target"), "n_iter": summ.get("n_iter"), "stop_reason": summ.get("stop_reason"),
        "widths_min_med_max": [int(np.min(summ["widths"])), int(np.median(summ["widths"])), int(np.max(summ["widths"]))],
        "concept_words_v1": concept_lists("v1").get(concept),
        "n_candidates": int(len(cand)),
        "F_TJ": sorted(int(f) for f in cand.loc[cand["tj"], "feature"]),
        "F_LO5": sorted(int(f) for f in cand.loc[cand["ej"], "feature"]),
        "F_TWJ": sorted(int(f) for f in cand.loc[cand["twj"], "feature"]),
        "n_tokens_edited": len(edited), "n_edit_ids_layer_cache": len(ori_edit),
        "edited_tokens_top30": [t for t, _ in edited[:30]],
        "check": check, "ori_run_dir": str(run["dir"].relative_to(C.ROOT)), "seconds": round(time.time() - t0, 1),
    }
    C.write_json(fdir / "summary.json", out)
    log(f"{concept}: candidates {out['n_candidates']}, TJ {out['F_TJ']}, LO@5 {out['F_LO5']}, HJ {out['F_TWJ']}, "
        f"{out['n_tokens_edited']} tokens edited (layer_cache: {len(ori_edit)}); mapping diff "
        f"{check['mapping_max_abs_diff']:.2e}, rel err X {check['rel_err_X']:.3f} / embedding "
        f"{check['rel_err_embedding_space']:.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concepts", nargs="+", default=C.CONCEPTS)
    ap.add_argument("--force-fit", action="store_true", help="refit even if Ori's run directory exists")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    torch.set_num_threads(a.threads)
    rows = []
    for c in a.concepts:
        try:
            rows.append(build(c, a.force_fit))
        except Exception as e:  # keep going; report at the end
            log(f"[FAIL] {c}: {type(e).__name__}: {e}")
            rows.append({"concept": c, "error": f"{type(e).__name__}: {e}"})
    keep = ["concept", "n_candidates", "F_TJ", "F_LO5", "F_TWJ", "n_tokens_edited", "widths_min_med_max", "error"]
    table = pd.DataFrame([{k: r.get(k) for k in keep} for r in rows])
    print(table.to_string(index=False))
    if any("error" in r for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
