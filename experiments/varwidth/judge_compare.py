#!/usr/bin/env python
"""Three judges on the rho >= 2 candidates of one existing run: J_tok (run.py's judge.json), J_words (same 2 stages,
stage 1 sees the "words" format of ember/judge_words.py) and J3 (overlap_judge.py: concept words vs feature words).

    python -m experiments.varwidth.judge_compare --concept Heroin --seed 42 [--tag rn_refit-False] [--workers 16]
        [--llm-match] [--force]

Reads run_dir/fit.pt + judge.json; caches judge_words.json ({"descriptions", "verdicts"}) there and writes judges.json:
    {"concept", "tag", "seed", "concept_words": {N: [...]}, "llm_match",
     "candidates": [{feature, ratio_abs, width, concept_share, neutral_share, judge_tokens, words_items,
                     feature_words, desc_tok, desc_words, j_tok: [member, conf, accept], j_words: [...],
                     j3: {N: {exact_pairs, n_pairs, accept}}}],     # N in overlap_judge.N_SIZES ("10" = the spec)
     "features": {f: {ratio_abs, width, concept_share, neutral_share, feature_words}}   # every feature, for the report}
J3 is exact (normalized-equality) matching, no Gemini call per candidate. --llm-match also runs the Gemini matcher at
N = 10 (cached in overlap.json) and adds j3["10"]["llm"] = {pairs, n_pairs, accept}. Missing J_tok verdicts are
filled into judge.json with the standard judge. --force redoes J_words (and the LLM matches).
"""
from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np
import torch

from experiments.varwidth import overlap_judge as oj
from experiments.varwidth.common import ROOT, VWConfig, run_dir
from experiments.varwidth.select import (CONFIDENCE, TOP_K_TOKENS, JudgeCache, _run_pool, get_client, judge,
                                         rho_candidates, save_json)

DEFAULT_TAG = VWConfig(method="rn", refit=False).tag()


def accepted(v) -> bool:
    return v is not None and v[0] is True and (v[1] or 0) >= CONFIDENCE


def row_order(Y: torch.Tensor, f: int) -> list:
    """Nonzero columns of row f by descending |y|, ties by column (as collect_feature_rows_for_embeddings)."""
    mags = Y[f].abs()
    nz = torch.nonzero(mags > 1e-12).flatten()
    return nz[torch.sort(mags[nz], descending=True, stable=True).indices].tolist()


def compare(concept: str, seed: int, tag: str = DEFAULT_TAG, workers: int = 16, llm_match: bool = False,
            force: bool = False) -> dict:
    from ember.interpret_features import classify_membership_with_desc, get_description
    from ember.judge_words import desc_sys_prompt, judge_items, token_word_counts
    from experiments.varwidth.prep import build_dataset, get_tokenizer, prepare

    t0 = time.time()
    d = run_dir(concept, tag, seed)
    Y = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)["Y"].float()
    prep = prepare(concept, "plain" if "_plain" in tag else "whitened", verbose=False)
    vstar, codes = prep["vstar_ids"], np.asarray(prep["codes"])
    stats, cand, _ = rho_candidates(Y, prep)
    rho = dict(zip(stats["feature"].astype(int), stats["ratio_abs"].astype(float)))
    feats = cand["feature"].astype(int).tolist()
    tok = get_tokenizer()
    tok_str = lambda t: tok.decode([t])  # noqa: E731
    counts = token_word_counts(tok, [s for s, _ in build_dataset(concept).data])
    client = get_client()

    jt = JudgeCache(d / "judge.json")
    judge(concept, {f: t for f, t in zip(feats, cand["judge_tokens"])}, jt, workers)   # no-op when complete
    jw = JudgeCache(d / "judge_words.json", use_cache=not force)
    ov_path = d / "overlap.json"
    ov = {} if force or not ov_path.exists() else {int(k): v for k, v in json.loads(ov_path.read_text()).items()}
    lock = threading.Lock()
    cws = {n: oj.concept_words(concept, n, client) for n in oj.N_SIZES}
    cw = cws[oj.N_WORDS]

    orders = {f: row_order(Y, f) for f in range(Y.shape[0])}
    fwords = {f: oj.feature_words([vstar[j] for j in o], counts, tok_str=tok_str) for f, o in orders.items()}
    items = {f: judge_items([vstar[j] for j in orders[f][:TOP_K_TOKENS]], tok_str, counts, "words", False)
             for f in feats}
    sys_words = desc_sys_prompt("words", False)

    def llm_j3(f):
        r = ov.get(f)
        if r is not None and (r["concept_words"], r["feature_words"], r.get("v")) == (cw, fwords[f], oj.MATCH_VERSION):
            return r
        r = {**oj.match(cw, fwords[f], client, llm=True), "concept_words": cw, "feature_words": fwords[f]}
        with lock:
            ov[f] = r
            save_json(ov_path, ov)
        return r

    def one(f):
        if f not in jw.descriptions:
            jw.put_desc(f, get_description(client, items[f], sys_prompt=sys_words))
        if f not in jw.verdicts:
            jw.put_verdict(f, classify_membership_with_desc(client, concept, jw.descriptions[f]))
        if llm_match:
            llm_j3(f)

    _run_pool(one, feats, workers)

    def shares(f):
        sup = np.asarray(orders[f], dtype=int)
        n = max(len(sup), 1)
        return len(sup), float((codes[sup] == 1).sum() / n), float((codes[sup] == 0).sum() / n)

    rows = []
    for r in cand.itertuples():
        f = int(r.feature)
        w, cs, ns = shares(f)
        j3 = {}
        for n, words in cws.items():
            ex = oj.exact_pairs(words, fwords[f])
            j3[str(n)] = {"exact_pairs": ex, "n_pairs": len(ex), "accept": len(ex) >= oj.MIN_PAIRS}
        if llm_match:
            j3[str(oj.N_WORDS)]["llm"] = {k: ov[f][k] for k in ("pairs", "n_pairs", "accept")}
        vt, vw = jt.verdicts.get(f), jw.verdicts.get(f)
        rows.append({"feature": f, "ratio_abs": float(r.ratio_abs), "width": w, "concept_share": cs,
                     "neutral_share": ns, "judge_tokens": list(r.judge_tokens), "words_items": items[f],
                     "feature_words": fwords[f], "desc_tok": jt.descriptions.get(f), "desc_words": jw.descriptions[f],
                     "j_tok": [None if vt is None else vt[0], None if vt is None else vt[1], accepted(vt)],
                     "j_words": [vw[0], vw[1], accepted(vw)],
                     "j3": j3})
    feat_info = {}
    for f in range(Y.shape[0]):
        w, cs, ns = shares(f)
        feat_info[f] = {"ratio_abs": rho.get(f), "width": w, "concept_share": cs, "neutral_share": ns,
                        "feature_words": fwords[f]}
    out = {"concept": concept, "tag": tag, "seed": seed, "concept_words": {str(n): w for n, w in cws.items()},
           "llm_match": llm_match,
           "candidates": rows, "features": feat_info, "elapsed": time.time() - t0}
    save_json(d / "judges.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--llm-match", action="store_true", help="also run the Gemini word matcher (N = 10)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    out = compare(a.concept, a.seed, a.tag, a.workers, a.llm_match, a.force)
    print(f"{a.concept} seed {a.seed} [{a.tag}]")
    for n, w in out["concept_words"].items():
        print(f"  concept words @{n}: {', '.join(w)}")
    sizes = list(out["concept_words"])
    print(f"{'feat':>4} {'rho':>9} {'C%':>4}  tok    words  J3@" + "/".join(sizes) + "  feature words | pairs @" + sizes[-1])
    for c in out["candidates"]:
        flag = lambda v: f"{'Y' if v[2] else '-'}{v[1] if v[1] is not None else '?':<5}"  # noqa: E731
        j3 = "/".join(f"{c['j3'][n]['n_pairs']}{'Y' if c['j3'][n]['accept'] else ''}" for n in sizes)
        print(f"{c['feature']:>4} {c['ratio_abs']:>9.3g} {100 * c['concept_share']:>3.0f}%  {flag(c['j_tok'])} "
              f"{flag(c['j_words'])} {j3:<9} {', '.join(c['feature_words'])} | "
              f"{', '.join(a for a, _ in c['j3'][sizes[-1]]['exact_pairs'])}")
    print(f"({out['elapsed']:.0f}s) -> {run_dir(a.concept, a.tag, a.seed) / 'judges.json'}")


if __name__ == "__main__":
    main()
