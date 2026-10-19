#!/usr/bin/env python
"""Collateral-aware erasure (PREREG amendment 3): edit sets of every variant, the blind token pool, question checks.

    python -m experiments.varwidth.collateral pool                       # edit sets + blind token pool (no GPU)
    python -m experiments.varwidth.collateral check --version v0 [--base] # programmatic checks of a question draft
    python -m experiments.varwidth.collateral freeze --version <chosen>   # data/collateral_questions.json + hash

Edit sets per (concept, seed) are cached in results/varwidth/layers/collateral/<slug>/edits_s<seed>.pt as
{variant: {"ids": [token ids], "C": (n, d) concept components in the TL-scaled embedding space, "F": features,
"fit": cfg, "dense": bool}} with
  ember        L0 fit, TJ, sparse Y (as published)          cinder   L12 fit, HJ (Ancient Rome: EJ word list v1r)
  ember_judge  L0 fit, TJ v WJ+share, sparse Y              cs_gs    L12 fit, TJ, sparse Y
  ee / 1b      EMBER's tokens x EMBER / CINDER direction (dense Y on that method's F_C)
  1a / cross   CINDER's top-N C tokens by mu_j = |c_j| / |e_j| x CINDER / EMBER direction, N = EMBER's edit count
Dense Y = (Z^T Z + lam I)^-1 Z^T A (core.py step 2); Y_dense * S == Y is asserted (refit = False).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import os
import json
import statistics

from pathlib import Path

import numpy as np
import torch

from experiments.varwidth.common import RESULTS, ROOT, run_dir, slug
from experiments.varwidth.layer_eval import CONFIGS, RUNGS, build_cache, fc

POC = ["Harry Potter", "Ancient Rome", "Valentine's Day"]
SEEDS = (42, 43)
# + one seed per concept where both EMBER and CINDER select features, so each concept has a full 2x2 (amendment 3)
EXTRA_SEED = {"Ancient Rome": 44, "Valentine's Day": 46}


def seeds(concept):
    return SEEDS + ((EXTRA_SEED[concept],) if concept in EXTRA_SEED else ())
OUT = RESULTS / "layers" / "collateral"
V1R = {"Ancient Rome"}          # concepts whose EJ uses the plan-B prompt v1r (prompt example removed)
QFILE = ROOT / "data" / "collateral_questions.json"


def fc_rows(concept, rung, seed):
    """F_C rows of a rung; for V1R concepts the EJ count is recomputed with the v1r concept words."""
    from experiments.varwidth.j3_eval import j3_count
    from experiments.varwidth.j3_words import concept_lists
    cfg, judge = RUNGS[rung]
    cache = build_cache(concept, cfg, seed)
    if concept in V1R:
        cw = concept_lists("v1r")[concept]
        cache = {**cache, "candidates": [{**c, "lo": j3_count(cw, c["fw20"])} for c in cache["candidates"]]}
    return fc(cache, judge)


_FITS = {}


def load_fit(concept, cfg, seed):
    """(prep, Z, Y sparse, Y dense, E rows (n, d) in the TL-scaled space, whitened) of a run."""
    k = (concept, cfg, seed)
    if k not in _FITS:
        from experiments.varwidth.prep import prepare
        tag = CONFIGS[cfg]
        whit = not tag.endswith("_plain")
        prep = prepare(concept, "whitened" if whit else "plain", verbose=False)
        fit = torch.load(run_dir(concept, tag, seed) / "fit.pt", map_location="cpu", weights_only=False)
        Z, Y, A = fit["Z"].double(), fit["Y"].double(), prep["A"].double()
        lam = float(fit["config"].get("ridge", 1e-4))
        Yd = torch.linalg.solve(Z.T @ Z + lam * torch.eye(Z.shape[1], dtype=Z.dtype), Z.T @ A)
        S = Y != 0
        err = ((Yd[S] - Y[S]).norm() / Y[S].norm()).item()
        assert err < 1e-4, f"{concept} {cfg} {seed}: dense Y != sparse Y on the support (rel err {err:.2e})"
        E = A.T @ prep["T_inv"].double() + prep["mu_B"].double() if whit else A.T
        _FITS[k] = (prep, Z, Y, Yd, E, whit, err)
    return _FITS[k]


def components(concept, cfg, seed, F, cols, dense):
    """(n, d) concept components sum_{i in F} Y_it z_i of columns `cols`, in the TL-scaled embedding space."""
    prep, Z, Y, Yd, E, whit, _ = load_fit(concept, cfg, seed)
    M = Yd if dense else Y
    C = (Z[:, F] @ M[F][:, cols]).T
    return C @ prep["T_inv"].double() if whit else C


def sparse_set(concept, rung, seed):
    cfg, _ = RUNGS[rung]
    F = [r["feature"] for r in fc_rows(concept, rung, seed)]
    prep, Z, Y, *_ = load_fit(concept, cfg, seed)
    codes = np.asarray(prep["codes"])
    if not F:
        return {"ids": [], "cols": [], "C": torch.zeros(0, Z.shape[0]), "F": [], "fit": cfg, "dense": False}
    cols = np.where((Y[F].abs() > 0).any(0).numpy() & (codes == 1))[0].tolist()
    return {"ids": [int(prep["vstar_ids"][j]) for j in cols], "cols": cols,
            "C": components(concept, cfg, seed, F, cols, False).float(), "F": F, "fit": cfg, "dense": False}


def ember_n(concept, seed, ember):
    """N for the 2x2: EMBER's edit count, or (EMBER selected nothing) the median over seeds 42-46."""
    if ember["ids"]:
        return len(ember["ids"]), False
    ns = [len(sparse_set(concept, "L0", s)["ids"]) for s in range(42, 47)]
    ns = [n for n in ns if n] or [0]
    return int(statistics.median(ns)), True


def edit_sets(concept, seed, force=False):
    path = OUT / slug(concept) / f"edits_s{seed}.pt"
    if path.exists() and not force:
        sets = torch.load(path, weights_only=False)
        if "cinder_rs" not in sets:                 # amendment 5: CINDER-RS = CS fit with RS-1% + TWJ
            sets["cinder_rs"] = sparse_set(concept, "L13", seed)
            torch.save(sets, path)
        return sets
    sets = {"ember": sparse_set(concept, "L0", seed), "cinder": sparse_set(concept, "L123", seed),
            "ember_judge": sparse_set(concept, "L0W", seed), "cs_gs": sparse_set(concept, "L12", seed),
            "cinder_rs": sparse_set(concept, "L13", seed)}
    em, ci = sets["ember"], sets["cinder"]
    prep = load_fit(concept, "L0", seed)[0]
    codes, vstar = np.asarray(prep["codes"]), prep["vstar_ids"]
    N, fallback = ember_n(concept, seed, em)
    info = {"N": N, "N_fallback": fallback}
    for _, cfg in (("ember", "L0"), ("cinder", "L12")):
        info[f"dense_err_{cfg}"] = load_fit(concept, cfg, seed)[6]
    if em["F"]:
        sets["ee"] = {**em, "C": components(concept, "L0", seed, em["F"], em["cols"], True).float(), "dense": True}
    if em["F"] and ci["F"]:
        sets["1b"] = {**em, "C": components(concept, "L12", seed, ci["F"], em["cols"], True).float(),
                      "F": ci["F"], "fit": "L12", "dense": True}
    if ci["F"] and N:
        allc = np.where(codes == 1)[0].tolist()
        Call = components(concept, "L12", seed, ci["F"], allc, True)
        E = load_fit(concept, "L12", seed)[4][allc]
        mu = (Call.norm(dim=1) / E.norm(dim=1)).numpy()
        top = [allc[i] for i in np.argsort(-mu)[:N]]
        Yraw = load_fit(concept, "L12", seed)[3][ci["F"]][:, allc].abs().sum(0).numpy()
        top_raw = {allc[i] for i in np.argsort(-Yraw)[:N]}
        info["topN_overlap_rawY"] = len(set(top) & top_raw) / N
        ids = [int(vstar[j]) for j in top]
        sets["1a"] = {"ids": ids, "cols": top, "C": components(concept, "L12", seed, ci["F"], top, True).float(),
                      "F": ci["F"], "fit": "L12", "dense": True}
        if em["F"]:
            sets["cross"] = {"ids": ids, "cols": top, "C": components(concept, "L0", seed, em["F"], top, True).float(),
                             "F": em["F"], "fit": "L0", "dense": True}
    sets["_info"] = info
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sets, path)
    return sets


def relative_mu(concept, seed, s):
    """Per-token mu_j = |c_j| / |e_j| of an edit set (e from the plain prep, which equals the TL-scaled rows)."""
    prep = load_fit(concept, "L0", seed)[0]
    E = prep["A"].double().T[s["cols"]]
    return (s["C"].double().norm(dim=1) / E.norm(dim=1)).numpy()


def pool(concepts=POC):
    from experiments.varwidth.prep import get_tokenizer
    tok = get_tokenizer()
    for c in concepts:
        member = {}
        for s in seeds(c):
            sets = edit_sets(c, s)
            for v, st in sets.items():
                if v.startswith("_"):
                    continue
                for t in st["ids"]:
                    member.setdefault(t, set()).add(f"{v}_s{s}")
            print(c, s, {v: len(st["ids"]) for v, st in sets.items() if not v.startswith("_")}, sets["_info"], flush=True)
        toks = sorted(({"id": t, "token": tok.decode([t])} for t in member),
                      key=lambda x: (x["token"].strip().lower(), x["token"]))
        d = OUT / "pool"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{slug(c)}.json").write_text(json.dumps({"concept": c, "tokens": toks}, ensure_ascii=False, indent=1))
        (d / f"{slug(c)}_membership.json").write_text(json.dumps({str(t): sorted(m) for t, m in member.items()}))
        print(f"{c}: pool of {len(toks)} tokens", flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---- question drafts ------------------------------------------------------------------------------------------
# Draft format (writer agents): {"concept": str, "tokens": [{"token": <exact pool string>, "class": "multi" |
# "exclusive", "note": str, "items": [{"q", "correct_answer", "options": [correct, d1, d2, d3], "context"}]}]}

def item_key(it):
    return hashlib.sha1(f"{it['q']}\x00{it['correct_answer']}".encode()).hexdigest()[:12]


def token_hits(tok, tid, it):
    """Where token id `tid` occurs as the MC prompt shows the item: 'q' (question line), 'a' (correct option), ''."""
    if tid in tok(f"Question: {it['q']}\n", add_special_tokens=False)["input_ids"]:
        return "q"
    if tid in tok(f"A. {it['correct_answer']}", add_special_tokens=False)["input_ids"]:
        return "a"
    return ""


def item_problems(it):
    p = []
    opts = it.get("options") or []
    if len(opts) != 4 or len(set(o.strip().lower() for o in opts)) != 4:
        p.append("needs 4 distinct options")
    if opts and opts[0] != it.get("correct_answer"):
        p.append("options[0] must equal correct_answer")
    return p


@functools.lru_cache(maxsize=None)
def exclusivity(concept):
    """{token id: log10 of (frequency in the concept sentences) / (frequency in 3,000 Wikipedia articles)}."""
    import math
    from collections import Counter

    from experiments.varwidth.prep import build_dataset, get_tokenizer
    tok = get_tokenizer()
    sents = [s for s, lbl in build_dataset(concept).data if lbl == concept]
    cc = Counter(t for s in sents for t in tok(s, add_special_tokens=False)["input_ids"])
    nc = sum(cc.values())
    wiki = json.loads((RESULTS / "layers" / "text" / "general_wiki.json").read_text())
    nw, wc = wiki["n_tokens"], wiki["counts"]
    return {t: math.log10((cc[t] / nc) / ((wc.get(str(t), 0) + 1) / nw)) for t in cc}


def tokcheck(draft_path, pool_path=None):
    """Objective checks of a draft (no GPU): token-id presence, option format, coverage of the pool."""
    from experiments.varwidth.prep import get_tokenizer
    tok = get_tokenizer()
    d = json.loads(open(draft_path).read())
    pool_path = pool_path or OUT / "pool" / f"{slug(d['concept'])}.json"
    pool_tok = {t["token"]: t["id"] for t in json.loads(open(pool_path).read())["tokens"]}
    rep = {"concept": d["concept"], "tokens": {}, "unknown_tokens": [], "missing_tokens": []}
    seen = set()
    for e in d["tokens"]:
        if e["token"] not in pool_tok:
            rep["unknown_tokens"].append(e["token"])
            continue
        seen.add(e["token"])
        tid = pool_tok[e["token"]]
        rows = []
        for it in e.get("items", []):
            where = token_hits(tok, tid, it)
            rows.append({"key": item_key(it), "q": it["q"], "token_in": where,
                         "problems": item_problems(it) + ([] if where else ["token id not in question or answer"])})
        rep["tokens"][e["token"]] = {"class": e.get("class"), "n": len(rows),
                                     "n_ok": sum(not r["problems"] for r in rows), "items": rows}
    rep["missing_tokens"] = [t for t in pool_tok if t not in seen]
    multi = [v for v in rep["tokens"].values() if v["class"] == "multi"]
    rep["summary"] = {"tokens_listed": len(seen), "pool": len(pool_tok), "multi": len(multi),
                      "exclusive": sum(v["class"] == "exclusive" for v in rep["tokens"].values()),
                      "items": sum(v["n"] for v in multi), "items_ok": sum(v["n_ok"] for v in multi)}
    rep["summary"]["ok_rate"] = rep["summary"]["items_ok"] / max(1, rep["summary"]["items"])
    return rep


def basecheck(drafts, out):
    """GPU: the unedited model's MC answer on every item of the drafts -> {item key: correct}."""
    from ember.evals.mc import evaluate_mc_generation, prepare_mc_items
    from ember.local_datasets import MCQAItem
    from experiments.varwidth.erase import load_model
    res = json.loads(open(out).read()) if os.path.exists(out) else {}
    _, w = load_model()
    for p in drafts:
        d = json.loads(open(p).read())
        items = {item_key(it): it for e in d["tokens"] for it in e.get("items", [])
                 if not item_problems(it) and item_key(it) not in res}
        if not items:
            continue
        keys = list(items)
        mc = prepare_mc_items([MCQAItem(d["concept"], "Collateral", "val", items[k]["q"], items[k]["options"],
                                        items[k]["correct_answer"]) for k in keys], seed=42)
        _, _, recs = evaluate_mc_generation(w, mc)
        res.update({k: bool(r["is_correct"]) for k, r in zip(keys, recs)})
        open(out, "w").write(json.dumps(res))
        print(f"{p}: {sum(res[k] for k in keys)}/{len(keys)} correct", flush=True)


def stats(draft_path, base_path=None):
    """Objective statistics of a draft for the selector: token-id pass, base accuracy, coverage, exclusivity flags."""
    rep = tokcheck(draft_path)
    base = json.loads(open(base_path).read()) if base_path and os.path.exists(base_path) else {}
    d = json.loads(open(draft_path).read())
    ex = exclusivity(d["concept"])
    pool_tok = {t["token"]: t["id"] for t in json.loads((OUT / "pool" / f"{slug(d['concept'])}.json").read_text())["tokens"]}
    usable, flags = {}, []
    for t, v in rep["tokens"].items():
        e = ex.get(pool_tok[t])
        if v["class"] == "exclusive" and e is not None and e < 1.0:
            flags.append({"token": t, "class": "exclusive", "log10_concept_over_wiki": round(e, 2)})
        if v["class"] == "multi":
            ok = [r for r in v["items"] if not r["problems"] and base.get(r["key"], not base)]
            usable[t] = len(ok)
    s = rep["summary"]
    s["base_acc"] = (sum(base[r["key"]] for v in rep["tokens"].values() for r in v["items"] if r["key"] in base)
                     / max(1, sum(r["key"] in base for v in rep["tokens"].values() for r in v["items"]))) if base else None
    s["tokens_with_2plus_usable"] = sum(n >= 2 for n in usable.values())
    s["tokens_with_4_usable"] = sum(n >= 4 for n in usable.values())
    s["exclusive_but_common_in_wiki"] = flags
    s["missing_tokens"] = rep["missing_tokens"]
    return s


def freeze(chosen: dict, base_path, out_path=None):
    """chosen: {concept: draft path}. Keep usable items (token id present, base model correct), at most 4 per
    multi-meaning token, 2 validation / 2 test (alternating in key order); drop tokens with < 2 usable items."""
    from experiments.varwidth.prep import get_tokenizer
    tok = get_tokenizer()
    base = json.loads(open(base_path).read())
    out = {}
    for c, spec in chosen.items():
        p, drop = (spec, []) if isinstance(spec, str) else (spec["path"], spec.get("drop", []))
        norm = lambda x: " ".join(x.lower().split())  # noqa: E731
        drop_q = [norm(q) for q in drop]
        d = json.loads(open(p).read())
        n_drop = 0
        for e in d["tokens"]:
            keep = [it for it in e.get("items", []) if not any(norm(it["q"]).startswith(q) for q in drop_q)]
            n_drop += len(e.get("items", [])) - len(keep)
            e["items"] = keep
        drop_tok = {q for q in drop if any(q == e["token"] for e in d["tokens"])}
        d["tokens"] = [e for e in d["tokens"] if e["token"] not in drop_tok]
        pool_tok = {t["token"]: t["id"] for t in json.loads((OUT / "pool" / f"{slug(c)}.json").read_text())["tokens"]}
        toks, dropped = {}, []
        for e in d["tokens"]:
            if e.get("class") != "multi" or e["token"] not in pool_tok:
                continue
            tid = pool_tok[e["token"]]
            ok = [it for it in e["items"] if not item_problems(it) and token_hits(tok, tid, it) and base.get(item_key(it))]
            ok = sorted(ok, key=item_key)[:4]
            if len(ok) < 2:
                dropped.append(e["token"])
                continue
            toks[str(tid)] = {"token": e["token"], "val": ok[0::2], "test": ok[1::2]}
        out[c] = {"source": str(p), "tokens": toks, "dropped": dropped, "selector_dropped_items": n_drop,
                  "selector_dropped_tokens": sorted(drop_tok)}
        print(f"{c}: {len(toks)} tokens kept, {sum(len(v['val']) + len(v['test']) for v in toks.values())} items, "
              f"dropped {dropped}", flush=True)
    q = Path(out_path) if out_path else QFILE
    q.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"frozen {q} sha256[:16] = {sha(q)}")


class Editor:
    """Applies e_t <- e_t - delta * c_t (TL-scaled rows, written back / sqrt(d)) or arbitrary new rows; restores."""

    def __init__(self, model):
        import math
        self.W = model.get_input_embeddings().weight
        self.scale = math.sqrt(self.W.shape[1])
        self.snap = {}

    def _orig(self, ids):
        idx = torch.tensor(ids, dtype=torch.long, device=self.W.device)
        for t in ids:
            if t not in self.snap:
                self.snap[t] = self.W[t].detach().float().clone()
        return idx, torch.stack([self.snap[t] for t in ids]) * self.scale

    def set_rows(self, ids, new_scaled):
        """new_scaled: (n, d) rows in the TL-scaled space."""
        self.restore()
        if not ids:
            return
        idx, _ = self._orig(ids)
        with torch.no_grad():
            self.W.index_copy_(0, idx, (new_scaled.to(self.W.device).float() / self.scale).to(self.W.dtype))

    def subtract(self, ids, C, delta):
        if not ids:
            self.restore()
            return
        _, E = self._orig(ids)
        self.set_rows(ids, E - delta * C.to(E.device).float())

    def orig_scaled(self, ids):
        return self._orig(ids)[1]

    def restore(self):
        if self.snap:
            ids = list(self.snap)
            idx = torch.tensor(ids, dtype=torch.long, device=self.W.device)
            with torch.no_grad():
                self.W.index_copy_(0, idx, torch.stack([self.snap[t] for t in ids]).to(self.W.dtype))


def premise(out_path):
    """Step 0: neutral-context prompts on the unedited model and on EMBER / CINDER (sparse, published edits) at
    delta 1 and 200, greedy, 80 new tokens."""
    from experiments.varwidth.erase import load_model
    P = json.loads((OUT / "premise_prompts.json").read_text())
    model, w = load_model()
    ed = Editor(model)
    res = json.loads(open(out_path).read()) if os.path.exists(out_path) else {}
    for c, spec in P.items():
        sets = edit_sets(c, spec["seed"])
        prompts = spec["prompts"]
        for name, (v, delta) in {"unedited": (None, 0), "ember_d1": ("ember", 1.0), "ember_d200": ("ember", 200.0),
                                 "cinder_d1": ("cinder", 1.0), "cinder_d200": ("cinder", 200.0)}.items():
            k = f"{c}|{name}"
            if k in res:
                continue
            if v is None:
                ed.restore()
            else:
                ed.subtract(sets[v]["ids"], sets[v]["C"], delta)
            gens = w.generate_multiple(prompts, max_new_tokens=80, do_sample=False, batch_size=10)
            res[k] = {"n_edit": 0 if v is None else len(sets[v]["ids"]), "gens": gens}
            open(out_path, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
            print(k, flush=True)
    ed.restore()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pool", "tokcheck", "basecheck", "stats", "freeze", "premise"))
    ap.add_argument("--concepts", nargs="*", default=POC)
    ap.add_argument("--draft", nargs="*", default=[])
    ap.add_argument("--base", default=str(OUT / "drafts" / "base_answers.json"))
    ap.add_argument("--chosen", help="json {concept: draft path}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.what == "pool":
        pool(a.concepts)
    elif a.what == "tokcheck":
        for p in a.draft:
            rep = tokcheck(p)
            print(json.dumps(rep["summary"]))
            for t, v in rep["tokens"].items():
                for r in v["items"]:
                    if r["problems"]:
                        print(f"  {t!r}: {r['q'][:90]!r} -> {'; '.join(r['problems'])}")
            if rep["unknown_tokens"]:
                print("  tokens not in the pool (use the exact strings):", rep["unknown_tokens"])
            if rep["missing_tokens"]:
                print("  pool tokens not in the draft:", rep["missing_tokens"])
    elif a.what == "basecheck":
        basecheck(a.draft, a.base)
    elif a.what == "stats":
        for p in a.draft:
            print(p, json.dumps(stats(p, a.base), ensure_ascii=False))
    elif a.what == "premise":
        premise(str(OUT / "premise_generations.json"))
    elif a.what == "freeze":
        freeze(json.loads(open(a.chosen).read()), a.base, a.out)


if __name__ == "__main__":
    main()
