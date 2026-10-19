#!/usr/bin/env python
"""Layer ladder (results/varwidth/layers/PREREG.md): per-run candidate caches, F_C per rung, labeling pools.

    python -m experiments.varwidth.layer_eval cache [--concepts ...] [--configs L0 L1 L12 ...] [--force]
    python -m experiments.varwidth.layer_eval pool  [--out PATH]          # unlabeled groups to label (label-blind)

Configs (CONFIGS): L0 = plain Fixed-1%, L1 = CW Fixed-1%, L12 = CW RNP (also rung L123 with HJ), and the
judge-free comparison configs (fixed2, m1, m3 at cap 3%; plain RNP). Rungs (RUNGS) map to (config, judge).

cache writes <run>/layer_cache.json: the top-20 candidates by rho (the judge set, PREREG) with rho, width, concept
share, TJ / CJ verdicts (judge.json / judge_words.json, when present), the first 20 normalized source words, the LO
count (concept_words_v1) and the group key; the run's n_tau; and the concept-only token ids of every candidate's
support (E for any F_C is the union over its features). The cache is rebuilt when judge files are newer.
Runs without judge files (judge-free configs) still get rho / words / LO / supports.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from experiments.varwidth.common import RESULTS, run_dir

SEEDS = range(42, 47)
CONFIGS = {"L0": "fixed_refit-False_plain", "L1": "fixed_refit-False", "L12": "rn_refit-False",
           "fixed2": "fixed_sparsity-0.02_refit-False", "m1": "m1_refit-False", "m3": "m3_refit-False",
           "rnP": "rn_refit-False_plain"}
JUDGED = ("L0", "L1", "L12")
RUNGS = {"L0": ("L0", "TJ"), "L1": ("L1", "TJ"), "L12": ("L12", "TJ"), "L123": ("L12", "HJ"),
         # exploratory, added after the erasure anchor (PREREG amendment 2): the judges without CS, and TJ v WJ+share
         "L0H": ("L0", "HJ"), "L0W": ("L0", "TW"), "L12W": ("L12", "TW"),
         # PREREG amendment 5: CINDER with EMBER's row sparsity (CS + RS-1% + TWJ), no global sparsity
         "L13": ("L1", "HJ")}
LO_T = 5
OUT = RESULTS / "layers"


def _accepted(v) -> bool:
    from experiments.varwidth.select import CONFIDENCE
    return v is not None and v[0] is True and (v[1] or 0) >= CONFIDENCE


_CTX = {}


def _concept_ctx(concept: str, matrix: str):
    """prep, tokenizer word counts and LO concept words (cached per process)."""
    k = (concept, matrix)
    if k not in _CTX:
        from ember.judge_words import token_word_counts
        from experiments.varwidth.j3_words import concept_lists
        from experiments.varwidth.prep import build_dataset, get_tokenizer, prepare
        tok = get_tokenizer()
        prep = prepare(concept, matrix, verbose=False)
        counts = _CTX.get(("counts", concept))
        if counts is None:
            counts = _CTX[("counts", concept)] = token_word_counts(tok, [s for s, _ in build_dataset(concept).data])
        _CTX[k] = (prep, counts, concept_lists("v1").get(concept), tok)
    return _CTX[k]


def build_cache(concept: str, cfg: str, seed: int, force: bool = False) -> dict | None:
    from experiments.varwidth import overlap_judge as oj
    from experiments.varwidth.j3_eval import group_key, j3_count
    from experiments.varwidth.judge_compare import row_order
    from experiments.varwidth.select import JUDGE_TOP_N, JudgeCache, rho_candidates
    tag = CONFIGS[cfg]
    d = run_dir(concept, tag, seed)
    if not (d / "fit.pt").exists():
        return None
    path = d / "layer_cache.json"
    srcs = [d / n for n in ("fit.pt", "judge.json", "judge_words.json") if (d / n).exists()]
    if path.exists() and not force and path.stat().st_mtime >= max(p.stat().st_mtime for p in srcs):
        return json.loads(path.read_text())
    prep, counts, cw, tok = _concept_ctx(concept, "plain" if tag.endswith("_plain") else "whitened")
    vstar, codes = prep["vstar_ids"], np.asarray(prep["codes"])
    Y = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)["Y"].float()
    stats, cand, _ = rho_candidates(Y, prep, top_n=JUDGE_TOP_N)
    jt = JudgeCache(d / "judge.json") if (d / "judge.json").exists() else None
    jw = JudgeCache(d / "judge_words.json") if (d / "judge_words.json").exists() else None
    rows = []
    for rank, r in enumerate(cand.itertuples()):
        f = int(r.feature)
        order = row_order(Y, f)
        sup = np.asarray(order, dtype=int)
        fw = oj.feature_words([vstar[j] for j in order], counts, n=20, tok_str=lambda t: tok.decode([t]))
        vt = jt.verdicts.get(f) if jt else None
        vw = jw.verdicts.get(f) if jw else None
        rows.append({"feature": f, "rank": rank, "rho": float(r.ratio_abs), "width": len(sup),
                     "share": float((codes[sup] == 1).mean()) if len(sup) else 0.0,
                     "tj": None if vt is None else _accepted(vt), "cj": None if vw is None else _accepted(vw),
                     "fw20": fw, "lo": None if cw is None else j3_count(cw, fw), "key": group_key(concept, fw),
                     "edit_ids": [int(vstar[j]) for j in sup if codes[j] == 1]})
    out = {"concept": concept, "config": cfg, "tag": tag, "seed": seed,
           "n_tau": int((stats["ratio_abs"] >= 2.0).sum()), "n_c_vstar": int((codes == 1).sum()),
           "candidates": rows}
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    return out


def load(concept: str, cfg: str, seed: int) -> dict | None:
    p = run_dir(concept, CONFIGS[cfg], seed) / "layer_cache.json"
    return json.loads(p.read_text()) if p.exists() else None


def fc(cache: dict, judge: str, lo_t: int = LO_T) -> list:
    """F_C (candidate rows) of a run under a judge: TJ, CJ, CJS (CJ and share >= .5), LO, HJ (TJ or LO@t),
    HJR (CJS or LO@4), TW (TJ or CJS)."""
    def ok(c):
        lo = c["lo"] or 0
        cjs = bool(c["cj"]) and c["share"] >= 0.5
        return {"TJ": bool(c["tj"]), "CJ": bool(c["cj"]), "CJS": cjs, "LO": lo >= lo_t,
                "HJ": bool(c["tj"]) or lo >= lo_t, "HJR": cjs or lo >= 4, "TW": bool(c["tj"]) or cjs}[judge]
    return [c for c in cache["candidates"] if ok(c)]


def edit_set(rows: list) -> set:
    return {t for c in rows for t in c["edit_ids"]}


def _cache_job(args):
    c, cfg, s, force = args
    try:
        r = build_cache(c, cfg, s, force)
        return (c, cfg, s, "ok" if r else "no fit")
    except Exception as e:  # noqa: BLE001
        return (c, cfg, s, f"ERR {type(e).__name__}: {e}")


def main():
    from dotenv import load_dotenv

    from experiments.varwidth.common import ROOT
    from experiments.varwidth.judge_report import concepts
    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("cache",))
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cs = a.concepts or concepts()
    # one process per concept keeps the per-concept word counts warm
    jobs = [[(c, cfg, s, a.force) for cfg in a.configs for s in SEEDS] for c in cs]
    with ProcessPoolExecutor(a.workers) as ex:
        for out in ex.map(_concept_jobs, jobs):
            for r in out:
                if r[3] != "ok":
                    print(*r, flush=True)
            print(f"{out[0][0]}: {sum(r[3] == 'ok' for r in out)}/{len(out)} cached", flush=True)


def _concept_jobs(js):
    return [_cache_job(j) for j in js]


if __name__ == "__main__":
    main()
