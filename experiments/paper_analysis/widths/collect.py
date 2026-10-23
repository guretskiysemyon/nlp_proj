#!/usr/bin/env python
"""WP3 data collection: token labels, dense re-solves, concept-feature rankings and directions, one pickle per concept.

    .venv/bin/python -m experiments.paper_analysis.widths.collect [--concepts ...] [--workers 4] [--force]

Per concept (results/paper_analysis/widths/cache/<slug>.pkl):
  labels   per V* token: code (1 C / 0 N / 2 B), c_C / c_N (occurrences in the 300 concept / neutral sentences),
           c_W (occurrences in the Wikipedia sample, wikifreq.py), ex = ln[(c_C + .5)/N_C] - ln[(c_W + .5)/N_W],
           word (normalized most frequent source word, as overlap_judge.feature_words), lex (word in the 20-word
           concept_words_v1 lexicon); and breadth proxies.
  runs     per (config, seed) in TAGS x SEEDS:
             dense re-solve Y' = (Z^T Z + lambda I)^-1 Z^T A with the saved Z (lambda = 1e-4, float64);
             check Y' = Y on the support (relative error per row) and that each row's support is the top-|S| of its
             dense row;
             concept feature = the top-20-by-rho candidate (layer_cache.json) with the most lexicon hits, ties by rho
             (the rule of layer_report's judge-free table); its width, index, LO, TJ, hand label;
             dense ranking of the concept feature by share |y'_ij| / ||y'_i|| (first TOP_KEEP positions + shares);
             widths of all features (serial-number check); GS global threshold (min support share of the free rows);
             the concept feature's z (fit space) and, for CS fits, z mapped back to the raw space (z~ T_inv).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from experiments.paper_analysis.widths.common import CACHE, RIDGE, SEEDS, TAGS, TOP_KEEP, concepts, run_dir, slug


def _source_word(counter: Counter | None, tok_str: str):
    from experiments.varwidth import overlap_judge as oj
    if counter:
        merged = Counter()
        for (_, word), k in counter.items():
            merged[word] += k
        word = oj._piece(merged.most_common(1)[0][0], tok_str)
    else:
        word = tok_str
    return oj.normalize(word)


def token_labels(concept: str, prep: dict, wiki: dict) -> dict:
    from ember.judge_words import token_word_counts
    from experiments.varwidth.j3_words import concept_lists
    from experiments.varwidth.prep import build_dataset, get_tokenizer
    tok = get_tokenizer()
    ds = build_dataset(concept)
    sC = [s for s, l in ds.data if l == concept]
    sN = [s for s, l in ds.data if l != concept]
    V = len(tok)

    def counts(sents):
        c = np.zeros(V, dtype=np.int64)
        for ids in tok(sents, add_special_tokens=False)["input_ids"]:
            c += np.bincount(np.asarray(ids, dtype=np.int64), minlength=V)
        return c

    cC, cN = counts(sC), counts(sN)
    vstar = np.asarray(prep["vstar_ids"])
    wc = token_word_counts(tok, sC + sN)
    wcC = token_word_counts(tok, sC)
    words = [_source_word(wc.get(int(t)), tok.decode([int(t)])) for t in vstar]
    lexicon = concept_lists("v1").get(concept, [])
    lexset = set(lexicon)
    N_C, N_W = int(cC.sum()), int(wiki["n_tokens"])
    c_W = wiki["cf"][vstar]
    ex = np.log((cC[vstar] + 0.5) / N_C) - np.log((c_W + 0.5) / N_W)
    codes = np.asarray(prep["codes"])
    # sanity: code 1 <=> occurs in S_C and not in S_N
    agree = float(np.mean((codes == 1) == ((cC[vstar] > 0) & (cN[vstar] == 0))))
    words_C = {_source_word(c, "") for c in wcC.values()}
    return {"vstar": vstar, "tokens": list(prep["tokens"]), "code": codes, "c_C": cC[vstar], "c_N": cN[vstar],
            "c_W": c_W, "ex": ex, "word": words, "lex": np.array([w in lexset for w in words]),
            "lexicon": lexicon, "lex_in_SC": sorted(lexset & words_C), "N_C": N_C, "N_W": N_W,
            "code_agree": agree, "n_sent_C": len(sC), "n_sent_N": len(sN)}


def page_sentences(concept: str) -> int | None:
    from experiments.varwidth.common import ROOT
    p = ROOT / "experiments" / "whitening" / "data" / "pages" / f"{slug(concept)}.json"
    return len(json.loads(p.read_text())["sentences"]) if p.exists() else None


def concept_feature(cache: dict) -> dict:
    """The rule of layer_report's judge-free table: max lexicon hits among the top-20 rho candidates, ties by rho."""
    from experiments.varwidth.layer_report import labels
    best = max(cache["candidates"], key=lambda x: ((x["lo"] or 0), x["rho"]))
    lab = labels().get(best["key"])
    return {"feature": int(best["feature"]), "rho_rank": int(best["rank"]), "rho": float(best["rho"]),
            "lo": int(best["lo"] or 0), "tj": best["tj"], "share_cache": float(best["share"]),
            "width_cache": int(best["width"]), "hand": lab, "fw20": best["fw20"],
            "lo_all": [int(c["lo"] or 0) for c in cache["candidates"]],
            "feat_all": [int(c["feature"]) for c in cache["candidates"]]}


def one_run(A: torch.Tensor, d: dict, T_inv: torch.Tensor | None) -> dict:
    fit = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)
    Z, Y = fit["Z"].double(), fit["Y"].double()
    k = Z.shape[1]
    G = Z.T @ Z
    Yd = torch.linalg.solve(G + RIDGE * torch.eye(k, dtype=G.dtype), Z.T @ A)        # (k, n)
    S = Y != 0
    w = S.sum(1)
    # Y' = Y on the support
    diff = ((Yd - Y) * S).norm(dim=1) / Y.norm(dim=1).clamp_min(1e-30)
    # support = top-|S| of the dense row by |y'|?
    order = torch.sort(Yd.abs(), dim=1, descending=True, stable=True).indices
    top_ok = []
    for i in range(k):
        m = int(w[i])
        top_ok.append(bool(S[i, order[i, :m]].all()))
    share = Yd.abs() / Yd.norm(dim=1, keepdim=True)
    min_sup = torch.where(S, share, torch.full_like(share, float("inf"))).min(1).values
    max_off = torch.where(~S, share, torch.full_like(share, -1.0)).max(1).values
    cache = json.loads((d / "layer_cache.json").read_text())
    cf = concept_feature(cache)
    priv = {"Z": Z, "S": S, "share": share.float(), "order": order[:, :TOP_KEEP], "T_inv": T_inv}
    rec = {"rel_diff_support": diff.numpy().astype(np.float32), "top_ok": np.array(top_ok),
           "widths": w.numpy(), "min_sup_share": min_sup.numpy(), "max_off_share": max_off.numpy(),
           "m_min": int(fit["m_min"]), "m_max": int(fit["m_max"]), "K": int(fit["K"]), "rel_err": float(fit["rel_err"]),
           "n_iter": int(fit["n_iter"]), "znorm": Z.norm(dim=0).numpy(), **cf,
           **feature_record(priv, cf["feature"]), "_priv": priv}            # _priv is dropped before pickling
    return rec


def feature_record(priv: dict, f: int) -> dict:
    """Dense ranking (first TOP_KEEP positions + shares), support, width and direction of feature f of a run."""
    o = priv["order"][f].numpy()
    z = priv["Z"][:, f]
    return {"feature": int(f), "width": int(priv["S"][f].sum()), "order": o.astype(np.int32),
            "share": priv["share"][f][o].numpy().astype(np.float32),
            "share_sq_total_top": float((priv["share"][f][o].double() ** 2).sum()),
            "support": torch.nonzero(priv["S"][f]).squeeze(1).numpy().astype(np.int32),
            "z": z.float().numpy(), "z_raw": None if priv["T_inv"] is None else (z @ priv["T_inv"]).float().numpy()}


def _unit(M: torch.Tensor) -> torch.Tensor:
    return M / M.norm(dim=0, keepdim=True).clamp_min(1e-30)


def _jac(a, b) -> float:
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(a & b) / max(len(a | b), 1)


def _best(za: torch.Tensor, priv_b: dict, raw_b: bool, T_inv: torch.Tensor) -> tuple[int, torch.Tensor]:
    """|cos| of direction za with every feature of run b; za and b in the same space (raw_b: map b's CS features
    back to the raw space first)."""
    Zb = priv_b["Z"]
    if raw_b:
        Zb = (Zb.T @ T_inv).T
    c = (_unit(Zb).T @ (za / za.norm())).abs()
    return int(c.argmax()), c


def cross_match(runs: dict, T_inv: torch.Tensor) -> dict:
    """Best-match checks. Same seed: the concept feature of config a against every feature of config b (CS space for
    CS pairs; raw space for pairs with a raw fit, CS directions mapped back by T_inv). Cross seed (same config): the
    concept feature of seed s against every feature of seed t (seed-noise baseline). Each entry: best index, its
    |cos|, whether it is b's own concept feature, |cos| with b's concept feature, and the support Jaccard of a's
    concept feature with b's best match and with b's concept feature."""
    out = {}
    pairs = [("L1", "L12"), ("L12", "L1"), ("L12", "RS2"), ("L1", "RS2"), ("L0", "L1"), ("L0", "L12"), ("L12", "L0"),
             ("L0", "rnP"), ("rnP", "L0"), ("L12", "rnP")]
    raw = {"L0", "rnP"}
    jobs = [((a, s), (b, s)) for s in SEEDS for a, b in pairs]
    jobs += [((c, s), (c, t)) for c in ("L0", "L1", "L12", "rnP") for s in SEEDS for t in SEEDS if s != t]
    for (a, s), (b, t) in jobs:
        if (a, s) not in runs or (b, t) not in runs:
            continue
        ra, rb = runs[(a, s)], runs[(b, t)]
        if a in raw or b in raw:     # compare in the raw space
            za = torch.as_tensor(ra["z"] if a in raw else ra["z_raw"]).double()
            j, c = _best(za, rb["_priv"], b not in raw, T_inv)
        else:
            j, c = _best(torch.as_tensor(ra["z"]).double(), rb["_priv"], False, T_inv)
        fb = rb["feature"]
        out[(a, s, b, t)] = {"best": j, "best_cos": float(c[j]), "is_concept": j == fb, "cos_concept": float(c[fb]),
                             "same_index": j == ra["feature"],
                             "jac_best": _jac(ra["support"], torch.nonzero(rb["_priv"]["S"][j]).squeeze(1).numpy()),
                             "jac_concept": _jac(ra["support"], rb["support"]),
                             "width_best": int(rb["_priv"]["S"][j].sum())}
    return out


def anchor_matches(runs: dict, T_inv: torch.Tensor) -> dict:
    """For each seed, the feature of L1 / RS2 / L0 / rnP closest (|cos|) to CINDER's concept feature (L12, the anchor):
    its full feature_record, so natural tails can be compared on the same concept direction across width rules."""
    out = {}
    for s in SEEDS:
        if ("L12", s) not in runs:
            continue
        za = runs[("L12", s)]
        for b in ("L1", "RS2", "L0", "rnP"):
            if (b, s) not in runs:
                continue
            raw_b = b in ("L0", "rnP")
            z = torch.as_tensor(za["z_raw"] if raw_b else za["z"]).double()
            j, c = _best(z, runs[(b, s)]["_priv"], False, T_inv)
            out[(b, s)] = {**feature_record(runs[(b, s)]["_priv"], j), "cos": float(c[j]),
                           "is_concept": j == runs[(b, s)]["feature"]}
    return out


def collect(concept: str, force: bool = False) -> str:
    torch.set_num_threads(int(os.environ.get("WP3_THREADS", "1")))
    from experiments.paper_analysis.widths.wikifreq import load as load_wiki
    from experiments.varwidth.prep import prepare
    path = CACHE / f"{slug(concept)}.pkl"
    if path.exists() and not force:
        return f"{concept}: cached"
    t0 = time.time()
    preps = {m: prepare(concept, m, verbose=False) for m in ("plain", "whitened")}
    assert preps["plain"]["vstar_ids"] == preps["whitened"]["vstar_ids"]
    lab = token_labels(concept, preps["plain"], load_wiki())
    lab["n_target"] = int(preps["whitened"]["n_target"])
    lab["page_sents"] = page_sentences(concept)
    T_inv = preps["whitened"]["T_inv"].double()
    runs = {}
    for cfg, (tag, matrix) in TAGS.items():
        A = preps[matrix]["A"].double()
        for s in SEEDS:
            d = run_dir(concept, tag, s)
            if not ((d / "fit.pt").exists() and (d / "layer_cache.json").exists()):
                continue
            runs[(cfg, s)] = one_run(A, d, T_inv if matrix == "whitened" else None)
        del A
    xm = cross_match(runs, T_inv)
    am = anchor_matches(runs, T_inv)
    for r in runs.values():
        r.pop("_priv")
    out = {"concept": concept, "labels": lab, "runs": runs, "cross": xm, "anchor": am}
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return f"{concept}: {len(runs)} runs ({time.time() - t0:.0f}s)"


def _job(a):
    c, force = a
    try:
        return collect(c, force)
    except Exception as e:  # noqa: BLE001
        import traceback
        return f"{c}: ERR {type(e).__name__}: {e}\n{traceback.format_exc()}"


def load(concept: str) -> dict:
    with open(CACHE / f"{slug(concept)}.pkl", "rb") as fh:
        return pickle.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cs = a.concepts or concepts()
    if a.workers <= 1:
        for c in cs:
            print(_job((c, a.force)), flush=True)
        return
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_job, [(c, a.force) for c in cs]):
            print(r, flush=True)


if __name__ == "__main__":
    main()
