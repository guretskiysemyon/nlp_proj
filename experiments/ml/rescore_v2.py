#!/usr/bin/env python
"""Re-score finished units on the cleaned (v2) MC question sets for it/pt/hi/th (results/ml/data/qa_v2).

    python -m experiments.ml.rescore_v2 [--approach A1 A2 A3 A3z] [--check]

Queue worker like worker.py (claims in results/ml/claims_rescore_v2, O_EXCL + stale takeover). Takes only units
whose eval.json has done=true and n_edited>0, skips units with rescore_v2.json. Per unit:
  * v2 baselines of the UNEDITED model for it/pt/hi/th (QA_train, QA_test, SimdomQA_test), cached per concept in
    results/ml/baselines_v2/<slug>_{train,test}.json;
  * edit at eval.json's best_delta (NOT re-selected: the delta was chosen on v1 train; only the test split is
    reported), QA_test and SimdomQA_test for it/pt/hi/th with the v2 items;
  * mmlu/ppl/mmlu_frac/fluency of those languages and everything of en/de/fr/es come from eval.json['test'];
  * efficacy/specificity/H/eligible recomputed as evaluate.score does, using its mc_norm/hm/agg/MIN_BASE_QA.
    Eligibility follows evaluate.score exactly: the baseline passed to score() for the test split is the TEST
    baseline, so eligible = v2 QA_test baseline >= MIN_BASE_QA (consistent with the unchanged languages);
    'eligible_train' (v2 QA_train baseline >= MIN_BASE_QA) is stored alongside for reference.
Output: run_dir/rescore_v2.json {per_lang (8 langs merged), score, score_v1, gated, best_delta, ...}.

v2 items: io.load_mc is reused unchanged (same seeded, language-aligned option permutation, same English-position
correct letter); only its per-language file reader io._mc is replaced IN THIS PROCESS so that it/pt/hi/th read
qa_v2 and every other language (en, used for the correct position, included) reads v1. No file is modified.
--check: CPU-only, no model: verifies v1 vs v2 correct letters for every item and that en/de/fr/es are identical.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from functools import lru_cache

from experiments.ml.common import APPROACHES, LANGS, RESULTS, concepts, run_dir, save_json, slug

V2_LANGS = ("it", "pt", "hi", "th")
QA_V1 = RESULTS / "data" / "qa"
QA_V2 = RESULTS / "data" / "qa_v2"
CLAIMS = RESULTS / "claims_rescore_v2"
BASE_DIR = RESULTS / "baselines_v2"
STALE = 90 * 60
OUT = "rescore_v2.json"


# ---------------------------------------------------------------- v2 items (process-local patch)
@lru_cache(maxsize=None)
def _mc_v2(lang: str):
    d = QA_V2 if lang in V2_LANGS else QA_V1
    return json.loads((d / f"mc_questions_{lang}.json").read_text(encoding="utf-8"))


def use_v2():
    import experiments.ml.evaldata.io as io
    io._mc = _mc_v2          # load_mc looks _mc up at call time


def check():
    """Item alignment v1 vs v2 without a model."""
    import experiments.ml.evaldata.io as io
    from experiments.ml.evaldata.common import SPLITS
    v1 = {L: [[it.correct_letter for it in io.load_mc(L, c, s)] for c in concepts() for s in SPLITS] for L in LANGS}
    v1q = {L: [[(it.question, it.options_by_letter) for it in io.load_mc(L, c, s)] for c in concepts() for s in SPLITS]
           for L in LANGS}
    use_v2()
    n = changed = 0
    for L in LANGS:
        k = 0
        for c in concepts():
            for s in SPLITS:
                items = io.load_mc(L, c, s)
                assert [it.correct_letter for it in items] == v1[L][k], (L, c, s)
                q = [(it.question, it.options_by_letter) for it in items]
                if L not in V2_LANGS:
                    assert q == v1q[L][k], (L, c, s)
                changed += sum(a != b for a, b in zip(q, v1q[L][k]))
                n += len(items)
                k += 1
        # every correct letter's text is the file's correct_answer
        for c in concepts():
            for s in SPLITS:
                for it, raw in zip(io.load_mc(L, c, s), _mc_v2(L)[c][s]):
                    assert it.options_by_letter[it.correct_letter] == raw["correct_answer"].strip()
    print(f"OK: {n} items over {len(LANGS)} langs, correct letters v1 == v2 everywhere, "
          f"en/de/fr/es identical, {changed} it/pt/hi/th items with changed text")


# ---------------------------------------------------------------- queue
def eligible_unit(c, a):
    p = run_dir(c, a) / "eval.json"
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    return d.get("done", False) and d.get("n_edited", 0) > 0


def done(c, a):
    return (run_dir(c, a) / OUT).exists()


def claim(c, a) -> bool:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    p = CLAIMS / f"{slug(c)}__{a}"
    me = f"{socket.gethostname()} {os.getpid()} {os.environ.get('SLURM_JOB_ID', 'local')} {time.time():.0f}"
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
        os.write(fd, me.encode())
        os.close(fd)
        return True
    except FileExistsError:
        bt = BASE_DIR / f"{slug(c)}_test.json"     # progress = claim touched or the concept's baselines written
        last = max(p.stat().st_mtime, bt.stat().st_mtime if bt.exists() else 0)
        if time.time() - last > STALE:
            p.write_text(me + " (took over)")
            return True
        return False


# ---------------------------------------------------------------- scoring
def baseline_v2(model, tok, concept, split):
    from experiments.ml.evaluate import mc_acc, mc_items
    path = BASE_DIR / f"{slug(concept)}_{split}.json"
    b = json.loads(path.read_text()) if path.exists() else {}
    sets = {"qa": f"QA_{split}"} | ({"simdom": "SimdomQA_test"} if split == "test" else {})
    for L in V2_LANGS:
        for k, s in sets.items():
            if k not in b.get(L, {}):
                b.setdefault(L, {})[k] = mc_acc(model, tok, mc_items(concept, L, s), L)["acc"]
                save_json(path, b)
    return b


def rescore(model, tok, editor, concept, approach):
    from experiments.ml.evaluate import MIN_BASE_QA, agg, edits, hm, mc_acc, mc_items, mc_norm
    ev = json.loads((run_dir(concept, approach) / "eval.json").read_text())
    editor.restore()                              # unedited model for the baselines
    btr, bte = baseline_v2(model, tok, concept, "train"), baseline_v2(model, tok, concept, "test")
    best = ev["best_delta"]
    per = {L: dict(ev["test"]["per_lang"][L]) for L in LANGS}
    if best:
        edit, _ = edits(concept, approach)
        editor.set_edit(edit)
        editor.apply(best)
    try:
        for L in V2_LANGS:
            qa = mc_acc(model, tok, mc_items(concept, L, "QA_test"), L)
            sd = mc_acc(model, tok, mc_items(concept, L, "SimdomQA_test"), L)
            p = per[L]
            eff = 1 - mc_norm(qa["acc"], bte[L]["qa"])
            sdf = mc_norm(sd["acc"], bte[L]["simdom"])
            p.update({"qa": qa["acc"], "simdom": sd["acc"], "invalid_qa": qa["invalid"], "efficacy": eff,
                      "simdom_frac": sdf, "specificity": hm([p["mmlu_frac"], sdf])})
            p["H"] = hm([eff, p["specificity"], p["fluency"]])
            p["eligible"] = bte[L]["qa"] >= MIN_BASE_QA
            p["eligible_train"] = btr[L]["qa"] >= MIN_BASE_QA
            p["qa_version"] = "v2"
    finally:
        editor.restore()
    gated = ev["test"]["gated"]                   # MMLU/fluency gates are unchanged
    res = {"concept": concept, "approach": approach, "best_delta": best, "delta_selected_on": "v1 train",
           "v2_langs": list(V2_LANGS), "baseline_v2": {"train": btr, "test": bte},
           "per_lang": per, "gated": gated, "score": 0.0 if gated else agg(per), "score_v1": ev["test"]["score"]}
    save_json(run_dir(concept, approach) / OUT, res)
    print(f"[rescore {approach} {concept}] delta={best} v1={res['score_v1']:.3f} v2={res['score']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", nargs="+", default=list(APPROACHES) + ["A3z"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    use_v2()
    from experiments.ml.evaluate import Editor, load_model
    model, tok = load_model()
    editor = Editor(model)
    while True:                                   # re-scan: units finish in the main evaluation meanwhile
        todo = [(c, x) for c in concepts() for x in a.approach if eligible_unit(c, x) and not done(c, x)]
        took = False
        for c, x in todo:
            if done(c, x) or not claim(c, x):
                continue
            t0 = time.time()
            print(f"[rescore {socket.gethostname()}] start {c} {x}", flush=True)
            try:
                rescore(model, tok, editor, c, x)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[rescore] FAILED {c} {x}: {type(e).__name__}: {e}", flush=True)
                import torch
                torch.cuda.empty_cache()
                (CLAIMS / f"{slug(c)}__{x}").unlink(missing_ok=True)
                editor.restore()
                continue
            took = True
            print(f"[rescore] done {c} {x} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if not took:
            break


if __name__ == "__main__":
    main()
