#!/usr/bin/env python
"""Selection stage: rho filter + judge on the fits of experiments/ml/fit.py (CPU + Gemini).

    python -m experiments.ml.select --concept "Harry Potter" --approach A1 A2 A3

Candidates (all approaches): features with rho >= 2 (EMBER's ratio_abs), the top 10 by rho.
TJ (all approaches): EMBER's 2-stage Gemini judge on the top-20 tokens by |Y| (decoded token strings), accept when
is_member and confidence >= 0.85.
A3 adds EJ@5: >= 5 exact matches between the feature's first 20 normalized source words (over the full multilingual
set) and the concept's lexicon (20 words per language, one Gemini call per concept, union over languages). TWJ = TJ or EJ.
Writes run_dir/(lang/)selection.json; the judge cache is run_dir/(lang/)judge.json.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from concurrent.futures import ThreadPoolExecutor

import torch

from experiments.ml.common import (CONFIDENCE, DATA, EJ_T, EJ_WORDS, LANGS, RESULTS, TAU, TOP_K_TOKENS, TOP_N,
                                   run_dir, save_json, tokenizer)
from experiments.ml.fit import _flatten, load_full

LEXICON_PATH = RESULTS / "lexicons.json"
LEXICON_PROMPT = """Given the concept '{concept}', list the {n} words most strongly associated with it, in each of these \
languages: {langs}.
Rules:
- Start with the concept's own name and its basic word forms when they are single words: the name itself, its \
adjective or demonym, as they are written in that language.
- Then the entities, people, places, objects, actions and terms that an ordinary person with general knowledge would \
immediately connect to this concept. Prefer well-known words over expert or niche terminology.
- Every word must be specific to this concept: avoid words that fit its broader category just as well (e.g. for \
'World War II' not 'war', 'battle' or 'soldier').
- Write each language's words the way they appear in that language's Wikipedia (its own script).
Single words, lowercase where the script has case, no duplicates or mere plural variants.
Return ONLY a JSON object mapping each language code to a list of {n} strings."""


def client():
    from experiments.varwidth.select import get_client
    return get_client()


def tok_str(t: int) -> str:
    tok = tokenizer()
    s = tok.decode([int(t)])
    return s if s.strip() and "�" not in s else tok.convert_ids_to_tokens(int(t))


# ---------------------------------------------------------------- lexicon (A3)
def parse_lexicon(raw: str) -> dict:
    """{lang: [words]} from the reply: strict JSON first, else each '"xx": [ ... ]' block read with a regex
    (tolerates trailing commas, comments and a reply cut off mid-list)."""
    import re
    s = raw.strip().strip("`")
    try:
        obj = json.loads(s[s.find("{"): s.rfind("}") + 1])
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    out = {}
    for m in re.finditer(r'"([a-z]{2})"\s*:\s*\[(.*?)(?:\]|$)', s, flags=re.DOTALL):
        out[m.group(1)] = re.findall(r'"([^"\n]+)"', m.group(2))
    return out


def lexicon(concept: str) -> dict:
    from experiments.varwidth.overlap_judge import _dedupe, normalize
    LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEXICON_PATH.with_suffix(".lock"), "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        cache = json.loads(LEXICON_PATH.read_text()) if LEXICON_PATH.exists() else {}
        if concept in cache:
            return cache[concept]["words"]
        raw = client().generate(LEXICON_PROMPT.format(concept=concept, n=EJ_WORDS, langs=", ".join(LANGS)))
        obj = parse_lexicon(raw)
        words = {L: _dedupe(normalize(w) for w in obj.get(L, []) if isinstance(w, str))[:EJ_WORDS] for L in LANGS}
        cache[concept] = {"words": words, "raw": raw}
        save_json(LEXICON_PATH, cache)
        return words


# ---------------------------------------------------------------- one run
def select_run(concept: str, fit_path, sentences, langs, use_ej: bool, workers: int = 8) -> dict:
    from ember.interpret_features import classify_membership_with_desc, get_description
    from ember.utils import compute_embedding_stats
    tokenizer()                                       # load once here: a first (lazy) import inside the threads races
    client()
    fit = torch.load(fit_path, weights_only=False)
    Y = fit["Y"].float()
    vstar = fit["vstar_ids"]
    t2c = dict(zip(vstar, [int(c) for c in fit["codes"]]))
    stats = compute_embedding_stats(Y.T, vstar, t2c, concept).sort_values("ratio_abs", ascending=False)
    cand = stats[stats["ratio_abs"] >= TAU].head(TOP_N)
    cache_path = fit_path.with_name("judge.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    feats = [int(f) for f in cand["feature"]]
    toks = {}
    for f in feats:
        y = Y[f]
        nz = torch.nonzero(y).flatten()
        order = nz[torch.argsort(y[nz].abs(), descending=True)]
        toks[f] = [vstar[int(i)] for i in order]

    def one(f):
        key = str(f)
        if key in cache and "verdict" in cache[key]:
            return
        items = [tok_str(t) for t in toks[f][:TOP_K_TOKENS]]
        desc = get_description(client(), items)
        cache[key] = {"tokens": items, "description": desc,
                      "verdict": list(classify_membership_with_desc(client(), concept, desc))}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, feats))
    save_json(cache_path, cache)

    ej = {}
    if use_ej:
        from ember.judge_words import token_word_counts
        from experiments.varwidth.overlap_judge import feature_words, normalize
        counts = token_word_counts(tokenizer(), sentences, langs)
        lex = lexicon(concept)
        union = list(dict.fromkeys(w for L in LANGS for w in lex.get(L, [])))
        for f in feats:
            fw = feature_words(toks[f], counts, n=EJ_WORDS, tok_str=tok_str)
            hits = [w for w in union if w in {normalize(x) for x in fw}]
            ej[f] = {"feature_words": fw, "matches": hits, "accept": len(hits) >= EJ_T}
    out = []
    for r in cand.itertuples():
        f = int(r.feature)
        is_m, conf = cache[str(f)]["verdict"]
        tj = bool(is_m) and float(conf or 0) >= CONFIDENCE
        e = ej.get(f, {}).get("accept", False)
        out.append({"feature": f, "rho": float(r.ratio_abs), "description": cache[str(f)]["description"],
                    "is_member": is_m, "confidence": conf, "TJ": tj, "EJ": e, "selected": tj or e,
                    "width": len(toks[f]), **({"ej": ej[f]} if f in ej else {})})
    F_C = sorted(c["feature"] for c in out if c["selected"])
    sel = {"concept": concept, "F_C": F_C, "n_tau": int((stats["ratio_abs"] >= TAU).sum()), "candidates": out,
           "judge": "TWJ" if use_ej else "TJ"}
    save_json(fit_path.with_name("selection.json"), sel)
    return sel


def run(concept: str, approach: str):
    full = load_full(concept)
    if approach == "A1":
        for L in LANGS:
            s, _, ls = _flatten(full, [L])
            sel = select_run(concept, run_dir(concept, "A1", L) / "fit.pt", s, ls, use_ej=False)
            print(f"[A1 {concept} {L}] n_tau={sel['n_tau']} F_C={sel['F_C']}", flush=True)
        return
    s, _, ls = _flatten(full)
    sel = select_run(concept, run_dir(concept, approach) / "fit.pt", s, ls, use_ej=approach == "A3")
    print(f"[{approach} {concept}] n_tau={sel['n_tau']} F_C={sel['F_C']} "
          f"(TJ {sum(c['TJ'] for c in sel['candidates'])}, EJ {sum(c['EJ'] for c in sel['candidates'])})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, nargs="+")
    ap.add_argument("--approach", required=True, nargs="+", choices=["A1", "A2", "A3"])
    a = ap.parse_args()
    for c in a.concept:
        for ap_ in a.approach:
            run(c, ap_)


if __name__ == "__main__":
    main()
