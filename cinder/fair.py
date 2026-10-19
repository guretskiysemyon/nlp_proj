#!/usr/bin/env python
"""Fair comparison of EMBER and CINDER (docs/Part 2/plan.md, "Fair comparison (option B)").

Identical for both methods: Gemma-2-2b-it, the English concept / neutral sentences, rank 100, seed 42, judge model
gemini-3.5-flash-lite, EMBER's edit code, the delta grid 0.5-200 (no extra steps), the pick rule and the tests.
Only the method differs:
    ember    regular EMBER: the paper's factorization (mf_outputs/, seed 42) + EMBER's two-stage judge rerun with gemini-3.5
             (mf_outputs/en_rejudge35) -> mf_outputs/ember35/
    cinder   Ori's L123 (CS + GS-norm + two-way judge, gemini-3.5) -> mf_outputs/cinder/ (cinder.build_features)
    ember_hj EMBER's factorization + Ori's judge only (L3: top-20 by rho, TJ = the gemini-3.5 verdicts above with
             confidence >= 0.85, or LO@5) -> mf_outputs/ember_hj/. Where it keeps the same features as `ember` the
             edit is identical (checked token by token), so `ember`'s results are reused and only the concepts whose
             features differ run on the GPU.

Per (method, concept), `run` does on one GPU:
    grid    every delta: the edit, MMLU (50 train), QA_train, SimdomQA_train with the fixed MC scoring
            (judge_rerun/mc_fixed.py); per-question records saved
    pick    (CPU) Part 1's fast screen without the Alpaca judge: deltas with MMLU below 90% of the pristine model are
            out; best harmonic of efficacy (1 - QA_train fraction of pristine) and specificity (harmonic of the MMLU
            and SimdomQA_train fractions); ties -> smaller delta; nothing above 0 -> no edit
    final   at the pick: QA_test + SimdomQA_test in en/de/es/fr/ru (crosslingual.rerun_fixed.full_cell: the
            baseline's items, shuffle and scoring), MMLU test (1000), and the edit check against the grid row
A concept whose judge keeps no feature is "no edit": no GPU, the pristine model's results count.

Outputs: judge_rerun/Fair/<method>/<Concept>/{grid.csv, records/, pick.json}
         results/crosslingual/google_gemma-2-2b-it/fair_<method>/<Concept>/{<lang>.json, mmlu_test.json, edit.json}
         judge_rerun/Fair/report.md (+ csv) from `report`

    python -m cinder.fair build-ember35                       # CPU, once
    python -m cinder.fair build-ember-hj                      # CPU, once (after build-ember35)
    python -m cinder.fair run --method cinder --concept Golf  # GPU (cinder/run_fair.sbatch)
    python -m cinder.fair run --method ember_hj --concepts Uranium Heroin   # one model load, several concepts
    python -m cinder.fair pick --method ember               # CPU, all concepts with a grid
    python -m cinder.fair report                              # CPU
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinder import common as C  # noqa: E402

ALL_CONCEPTS = ["Culture of Greece", "Golf", "Republic of Ireland", "Ancient Rome", "Baseball", "Uranium",
                "Harry Potter", "Cannabis", "Gambling", "Gun", "Pornography", "Valentine's Day", "World War II",
                "Nazism", "Heroin", "Halloween", "Artificial intelligence", "COVID-19 pandemic"]
# CINDER tuning concepts (eta, gamma: Harry Potter, Ancient Rome, Valentine's Day) and the 4 examples of Ori's
# concept-word prompt (Culture of Greece, Ancient Rome, AI, World War II): left out of the "clean" subset.
NOT_CLEAN = {"Harry Potter", "Ancient Rome", "Valentine's Day", "Culture of Greece", "Artificial intelligence",
             "World War II"}
ROOTS = {"ember": C.ROOT / "mf_outputs" / "ember35", "cinder": C.MF_ROOT,
         "ember_hj": C.ROOT / "mf_outputs" / "ember_hj"}
PAIRS = [("cinder", "ember"), ("ember_hj", "ember"), ("cinder", "ember_hj")]     # (a, b): report a − b
GRID = list(C.EMBER_GRID)
SETS = ("mmlu", "qa", "simdom")
LANGS = ["en", "de", "es", "fr", "ru"]
FAIR = C.ROOT / "judge_rerun" / "Fair"
XL = C.ROOT / "results" / "crosslingual" / C.MODEL_SAFE
SMOKE = int(os.environ.get("FAIR_SMOKE", "0"))     # > 0: only that many items per set (CPU smoke test)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def fair_base() -> Path:
    return Path(os.environ.get("FAIR_OUT", FAIR))       # FAIR_OUT / FAIR_XL: redirect outputs (smoke tests)


def out_dir(method, concept):
    return fair_base() / method / C.safe(concept)


def xl_dir(method, concept):
    return Path(os.environ.get("FAIR_XL", XL)) / f"fair_{method}" / C.safe(concept)


def pot_path(method, concept):
    return (ROOTS[method] / C.MODEL_SAFE / "interpretations" / f"rank{C.RANK}" / f"seed{C.SEED}" / C.safe(concept)
            / "embedding" / "potential_features.csv")


def kept_features(method, concept) -> list[int]:
    import pandas as pd
    p = pot_path(method, concept)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    return sorted(int(f) for f, m in zip(df.get("feature", []), df.get("metric_score", [])) if float(m) >= 2.0)


# ---------------------------------------------------------------- EMBER with the gemini-3.5 judge
def build_ember35():
    """mf_outputs/ember35/: the paper's factorization (pickles, token csvs: symlinks) + the 3.5 judge's kept list."""
    src, rej, dst = C.EMBER_MF_ROOT / C.MODEL_SAFE, C.ROOT / "mf_outputs" / "en_rejudge35" / C.MODEL_SAFE, \
        ROOTS["ember"] / C.MODEL_SAFE
    rel = Path(f"rank{C.RANK}") / f"seed{C.SEED}"
    rows = []
    for c in ALL_CONCEPTS:
        s = C.safe(c)
        for kind in ("pickles", "csvs"):
            link = dst / kind / rel / s / "embedding"
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(src / kind / rel / s / "embedding", target_is_directory=True)
        idir = dst / "interpretations" / rel / s / "embedding"
        idir.mkdir(parents=True, exist_ok=True)
        for f in ("potential_features.csv", "from_activation.csv"):
            shutil.copy2(rej / "interpretations" / rel / s / "embedding" / f, idir / f)
        old = kept_features_at(C.EMBER_MF_ROOT, c)
        new = kept_features("ember", c)
        rows.append({"concept": c, "kept_paper_judge": old, "kept_gemini35": new, "same": old == new})
    C.write_json(FAIR / "ember35_features.json", rows)
    for r in rows:
        log(f"{r['concept']}: paper judge {r['kept_paper_judge']} -> gemini-3.5 {r['kept_gemini35']}"
            f"{'' if r['same'] else '  (changed)'}")


def kept_features_at(root, concept):
    import pandas as pd
    p = (root / C.MODEL_SAFE / "interpretations" / f"rank{C.RANK}" / f"seed{C.SEED}" / C.safe(concept) / "embedding"
         / "potential_features.csv")
    df = pd.read_csv(p)
    return sorted(int(f) for f, m in zip(df["feature"], df["metric_score"]) if float(m) >= 2.0)


# ---------------------------------------------------------------- EMBER with Ori's judge (L3 only)
def edit_tokens(method, concept):
    """(token, magnitude) list of the edit, from the method's feature folder (CPU)."""
    from ember.erasure import embed_edit, features
    old, features.MF_OUTPUTS_ROOT = features.MF_OUTPUTS_ROOT, ROOTS[method]
    try:
        return [(t, round(float(m), 6)) for t, m in
                embed_edit.concept_edit_tokens(C.MODEL_NAME, concept, C.RANK, C.SEED, ratio_thresh=2.0)]
    finally:
        features.MF_OUTPUTS_ROOT = old


def build_ember_hj():
    """mf_outputs/ember_hj/: EMBER's factorization (symlinks) + Ori's HJ on its top-20 features by rho.
    TJ reuses EMBER's gemini-3.5 verdicts (mf_outputs/en_rejudge35: same two-stage judge, 20 tokens, confidence
    0.85); LO = concept words v1 vs the feature's first 20 source words (layer_eval.build_cache's recipe)."""
    import pandas as pd
    from ember.erasure.features import _load_nmf_pickle
    from ember.utils import compute_embedding_stats
    from experiments.varwidth import overlap_judge as oj
    from experiments.varwidth.j3_eval import j3_count
    from experiments.varwidth.judge_compare import row_order
    from experiments.varwidth.layer_eval import LO_T, _concept_ctx
    from experiments.varwidth.select import CONFIDENCE, JUDGE_TOP_N
    src, rej, dst = C.EMBER_MF_ROOT / C.MODEL_SAFE, C.ROOT / "mf_outputs" / "en_rejudge35" / C.MODEL_SAFE, \
        ROOTS["ember_hj"] / C.MODEL_SAFE
    rel = Path(f"rank{C.RANK}") / f"seed{C.SEED}"
    rows = []
    for c in ALL_CONCEPTS:
        s = C.safe(c)
        for kind in ("pickles", "csvs"):
            link = dst / kind / rel / s / "embedding"
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(src / kind / rel / s / "embedding", target_is_directory=True)
        d = _load_nmf_pickle(src / "pickles" / rel / s / "embedding" / "embedding.pkl")
        G, vp = d["nmf"].G_.detach().float().cpu(), list(d["vprime_token_ids"])
        prep, counts, cw, tok = _concept_ctx(c, "whitened")
        assert vp == list(prep["vstar_ids"]), f"{c}: EMBER's V' differs from Ori's V*"
        top = compute_embedding_stats(G, vp, prep["token_to_code"], c) \
            .sort_values("ratio_abs", ascending=False).head(JUDGE_TOP_N)
        judged = pd.read_csv(rej / "interpretations" / rel / s / "embedding" / "from_activation.csv").set_index("feature")
        cand = []
        for _, r in top.iterrows():
            f = int(r["feature"])
            if f not in judged.index:
                raise KeyError(f"{c}: feature {f} (rho {r['ratio_abs']:.2f}) has no gemini-3.5 verdict")
            v = judged.loc[f]
            assert abs(float(v["metric_score"]) - float(r["ratio_abs"])) < 1e-3, (c, f)
            fw = oj.feature_words([vp[j] for j in row_order(G.T, f)], counts, n=20, tok_str=lambda t: tok.decode([t]))
            tj = str(v["is_member"]).strip().lower() == "true" and float(v["confidence"]) >= CONFIDENCE
            lo = j3_count(cw, fw) if cw else 0
            cand.append({**v.to_dict(), "feature": f, "tj": tj, "lo": lo, "ej": lo >= LO_T, "twj": tj or lo >= LO_T,
                         "feature_words": json.dumps(fw)})
        cand = pd.DataFrame(cand)
        idir = dst / "interpretations" / rel / s / "embedding"
        idir.mkdir(parents=True, exist_ok=True)
        cand.to_csv(idir / "from_activation.csv", index=False)
        cand[cand["twj"]].to_csv(idir / "potential_features.csv", index=False)
        new, old = kept_features("ember_hj", c), kept_features("ember", c)
        same_edit = new == old and edit_tokens("ember_hj", c) == edit_tokens("ember", c)
        rows.append({"concept": c, "kept_ember": old, "kept_ember_hj": new,
                     "tj_top20": sorted(int(f) for f in cand.loc[cand.tj, "feature"]),
                     "lo5": sorted(int(f) for f in cand.loc[cand.ej, "feature"]),
                     "lo": {int(f): int(l) for f, l in zip(cand.feature, cand.lo) if l},
                     "kept_paper_judge": kept_features_at(C.EMBER_MF_ROOT, c),
                     "same_features": new == old, "same_edit": same_edit,
                     "n_tokens_edited": len(edit_tokens("ember_hj", c)) if new else 0})
        log(f"{c}: EMBER {old} -> EMBER + Ori's judge {new} (TJ@top20 {rows[-1]['tj_top20']}, LO@5 {rows[-1]['lo5']})"
            f"{'  same edit -> reuse EMBER' if same_edit else '  (changed: GPU run)' if new else '  no feature'}")
    C.write_json(FAIR / "ember_hj_features.json", rows)


def same_as_ember(concept) -> bool:
    """ember_hj's edit is identical to ember's: reuse ember's results (written by build_ember_hj)."""
    p = FAIR / "ember_hj_features.json"
    return p.exists() and any(r["concept"] == concept and r["same_edit"] for r in json.loads(p.read_text()))


def src_method(method, concept) -> str:
    """Whose result files to read: ember_hj's identical-edit concepts point at ember's."""
    return "ember" if method == "ember_hj" and same_as_ember(concept) else method


# ---------------------------------------------------------------- the GPU part
class Model:
    def __init__(self, method):
        from ember.erasure import embed_edit, features
        from ember.erasure.model_loader import load_hf_model
        from ember.evals.mmlu import load_mmlu_items
        from ember.evals.model_wrap import ensure_wrapped_model
        from ember.local_datasets import load_mmlu_indices
        features.MF_OUTPUTS_ROOT = ROOTS[method]
        self.embed_edit = embed_edit
        self.model, tok = load_hf_model(C.MODEL_NAME)
        self.tm = ensure_wrapped_model(self.model, tok)
        self.pristine = embed_edit.snapshot(self.model)
        self.mmlu_train = load_mmlu_items(load_mmlu_indices("train"))
        self.mmlu_test = load_mmlu_items(load_mmlu_indices("test"))
        if SMOKE:
            self.mmlu_train, self.mmlu_test = self.mmlu_train[:SMOKE], self.mmlu_test[:SMOKE]

    def edit(self, concept, delta):
        self.embed_edit.restore(self.model, self.pristine)
        return self.embed_edit.apply_concept_embed_edit_factored(
            hf_model=self.model, model_name=C.MODEL_NAME, concept_name=concept, delta_embed=delta,
            rank=C.RANK, seed=C.SEED, ratio_thresh=2.0)

    def restore(self):
        self.embed_edit.restore(self.model, self.pristine)


def train_items(concept):
    from ember.evals.mc import prepare_mc_items
    from ember.local_datasets import load_mc_qa_items
    qa = prepare_mc_items(load_mc_qa_items("qa_train", concept=concept), seed=C.SEED)
    sim = prepare_mc_items(load_mc_qa_items("simdom_train", concept=concept), seed=C.SEED)
    return (qa[:SMOKE], sim[:SMOKE]) if SMOKE else (qa, sim)


def grid(m: Model, method, concept, deltas):
    from judge_rerun.mc_fixed import evaluate_mc_fixed, evaluate_mmlu_fixed
    import pandas as pd
    od = out_dir(method, concept)
    (od / "records").mkdir(parents=True, exist_ok=True)
    path = od / "grid.csv"
    rows = {float(r["delta_embed"]): r.to_dict() for _, r in pd.read_csv(path).iterrows()} if path.exists() else {}
    qa, sim = train_items(concept)
    for d in deltas:
        if d in rows:
            continue
        t0 = time.time()
        info = m.edit(concept, d)
        res = {"mmlu": evaluate_mmlu_fixed(m.tm, m.mmlu_train), "qa": evaluate_mc_fixed(m.tm, qa),
               "simdom": evaluate_mc_fixed(m.tm, sim)}
        m.restore()
        row = {"concept": concept, "method": method, "delta_embed": d, "k_features_embed": int(info["k_features_embed"]),
               "n_tokens_edited": int(info["n_tokens_edited"])}
        for s in SETS:
            row[f"{s}_acc"], row[f"{s}_invalid_rate"] = res[s]["acc"], res[s]["invalid_rate"]
        row["wall_time_s"] = round(time.time() - t0, 1)
        C.write_json(od / "records" / f"delta_{d}.json", {**row, **{s: res[s] for s in SETS}})
        rows[d] = row
        pd.DataFrame(sorted(rows.values(), key=lambda r: r["delta_embed"])).to_csv(path, index=False)
        log(f"{method} {concept} delta={d}: k={row['k_features_embed']} tokens={row['n_tokens_edited']} "
            f"mmlu {row['mmlu_acc']:.2f} qa {row['qa_acc']:.2f} simdom {row['simdom_acc']:.2f} ({row['wall_time_s']:.0f}s)")


def pick_from_grid(g, concept) -> dict:
    """Part 1's fast screen (pass2_delta_search._metrics_for_delta, alpaca=None) + argmax harmonic, ties smaller."""
    from judge_rerun.pass2_delta_search import _load_baselines, _metrics_for_delta
    bases = _load_baselines(C.BASELINE_DIR, concept, C.JUDGE_ID, "original")
    g = g.sort_values("delta_embed").reset_index(drop=True)
    ms = [_metrics_for_delta(r, bases, None, C.MIN_MMLU, C.MIN_ALPACA) for _, r in g.iterrows()]
    ok = [m for m in ms if m.get("gate") != "min_mmlu"]
    best = None
    for m in ok:                                   # ascending delta, strict > keeps the smaller one on ties
        if best is None or m["harmonic"] > best["harmonic"]:
            best = m
    if best is None or best["harmonic"] <= 0:
        return {"concept": concept, "delta": 0.0, "no_edit": True, "reason": "no delta passes the gates"}
    return {"concept": concept, "delta": float(best["delta_embed"]), "no_edit": False,
            "harmonic": best["harmonic"], "efficacy": best["efficacy"], "specificity": best["specificity"],
            "rule": "fast harmonic (efficacy, specificity), MMLU gate 0.9, no Alpaca", "baselines": bases,
            "all": [{k: m.get(k) for k in ("delta_embed", "qa_acc", "efficacy", "specificity", "harmonic", "gate")}
                    for m in ms]}


def pick(method, concept) -> dict:
    import pandas as pd
    od = out_dir(method, concept)
    if not kept_features(method, concept):
        p = {"concept": concept, "method": method, "delta": 0.0, "no_edit": True, "reason": "no feature kept"}
    else:
        p = {"method": method, **pick_from_grid(pd.read_csv(od / "grid.csv"), concept)}
    C.write_json(od / "pick.json", p)
    log(f"{method} {concept}: pick delta={p['delta']}" + (f" ({p['reason']})" if p["no_edit"] else
                                                         f" harmonic={p['harmonic']:.3f}"))
    return p


def final(m: Model, method, concept, p: dict):
    import torch
    from crosslingual.rerun_fixed import Generator, _write_json, full_cell
    from crosslingual.run_edited import _git_commit
    from judge_rerun.mc_fixed import evaluate_mc_fixed, evaluate_mmlu_fixed
    xd = xl_dir(method, concept)
    if p["no_edit"]:
        _write_json(xd / "edit.json", {"concept": concept, "method": method, "no_edit": True, "reason": p["reason"]})
        return
    import pandas as pd
    g = pd.read_csv(out_dir(method, concept) / "grid.csv")
    row = g[g.delta_embed == p["delta"]].iloc[0]
    if SMOKE:                                       # smoke test: only the first items of each test set
        import crosslingual.rerun_fixed as rf
        if not getattr(rf.load_mc_items, "_smoke", False):
            orig = rf.load_mc_items
            rf.load_mc_items = lambda *a, **k: orig(*a, **k)[:SMOKE]
            rf.load_mc_items._smoke = True
    gen = Generator(m.tm)
    t0 = time.time()
    info = m.edit(concept, p["delta"])
    try:
        with torch.no_grad():
            qa, sim = train_items(concept)
            check = {"mmlu_acc": evaluate_mmlu_fixed(m.tm, m.mmlu_train)["acc"],
                     "qa_acc": evaluate_mc_fixed(m.tm, qa)["acc"], "simdom_acc": evaluate_mc_fixed(m.tm, sim)["acc"]}
            mmlu = evaluate_mmlu_fixed(m.tm, m.mmlu_test)
            for L in LANGS:
                cells = full_cell(gen, concept, L, C.SEED)
                _write_json(xd / f"{L}.json", {"concept": concept, "language": L, "method": method,
                                               "hp": {"delta_embed": p["delta"]}, "results": cells,
                                               "scoring": "mc_fixed (32 tokens, parse_letter_fixed)"})
    finally:
        m.restore()
    diffs = {k: round(check[k] - float(row[k]), 4) for k in check}
    _write_json(xd / "mmlu_test.json", {"concept": concept, "hp": {"delta_embed": p["delta"]}, **mmlu})
    _write_json(xd / "edit.json", {
        "concept": concept, "method": method, "no_edit": False, "hp": {"delta_embed": p["delta"]}, "edit_info": info,
        "check": {"english_train": check, "diff_vs_grid": diffs,
                  "reproduces_grid": all(v == 0 for v in diffs.values())
                  and int(info["n_tokens_edited"]) == int(row["n_tokens_edited"])},
        "mmlu_test": {"acc": mmlu["acc"], "invalid": mmlu["invalid_rate"], "n": mmlu["n"]},
        "seconds": round(time.time() - t0, 1), "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    log(f"{method} {concept} final at delta={p['delta']}: edit check "
        f"{'OK' if all(v == 0 for v in diffs.values()) else 'DIFFERS ' + str(diffs)}, MMLU test {mmlu['acc']:.3f}")


def run(method, concepts, deltas):
    """One model load for all `concepts` of one method."""
    m = None
    for concept in ([concepts] if isinstance(concepts, str) else concepts):
        if src_method(method, concept) != method:
            log(f"{method} {concept}: same edit as {src_method(method, concept)} -> reused, no GPU work")
            continue
        od = out_dir(method, concept)
        od.mkdir(parents=True, exist_ok=True)
        if not kept_features(method, concept):
            final(None, method, concept, pick(method, concept))
            log(f"{method} {concept}: no feature kept -> no edit, no GPU work")
            continue
        m = m or Model(method)
        grid(m, method, concept, deltas)
        p = pick(method, concept)
        final(m, method, concept, p)


# ---------------------------------------------------------------- report (CPU)
def report():
    import numpy as np
    import pandas as pd
    from crosslingual.run_edited import _retention
    base_dir = XL / "baseline_fixed"
    rows = []
    for method in ROOTS:
        for c in ALL_CONCEPTS:
            sm = src_method(method, c)                  # ember_hj with ember's exact edit: ember's files
            ed_path = xl_dir(sm, c) / "edit.json"
            pk_path = out_dir(sm, c) / "pick.json"
            if not ed_path.exists() or not pk_path.exists():
                continue
            ed, pk = json.loads(ed_path.read_text()), json.loads(pk_path.read_text())
            r = {"concept": c, "method": method, "reused_from": "" if sm == method else sm, "clean": c not in NOT_CLEAN,
                 "n_features": len(kept_features(method, c)), "delta": pk["delta"], "no_edit": pk["no_edit"],
                 "train_harmonic": 0.0 if pk["no_edit"] else round(pk["harmonic"], 3)}
            r["n_tokens_edited"] = 0 if ed["no_edit"] else ed["edit_info"]["n_tokens_edited"]
            pristine_mmlu = json.loads((base_dir / "_mmlu.json").read_text())["mmlu_test"]["acc"]
            r["mmlu_test"] = pristine_mmlu if ed["no_edit"] else ed["mmlu_test"]["acc"]
            r["edit_check"] = None if ed["no_edit"] else ed["check"]["reproduces_grid"]
            for L in LANGS:
                base = json.loads((base_dir / C.safe(c) / f"{L}.json").read_text())
                pq, ps = base["results"]["QA_test"]["accuracy"], base["results"]["SimdomQA_test"]["accuracy"]
                if ed["no_edit"]:
                    qa, sd = pq, ps
                else:
                    cell = json.loads((xl_dir(sm, c) / f"{L}.json").read_text())["results"]
                    qa, sd = cell["QA_test"]["accuracy"], cell["SimdomQA_test"]["accuracy"]
                ret = _retention(qa, pq)
                r[f"ret_{L}"] = None if ret is None else round(ret, 3)
                r[f"simdom_{L}"] = round(sd - ps, 3)            # change vs pristine
            other = [r[f"ret_{L}"] for L in LANGS[1:] if r[f"ret_{L}"] is not None]
            r["leak"] = round(float(np.mean(other)) - r["ret_en"], 3) if other and r["ret_en"] is not None else None
            rows.append(r)
    D = pd.DataFrame(rows)
    fb = fair_base()
    fb.mkdir(parents=True, exist_ok=True)
    D.to_csv(fb / "report.csv", index=False)
    L_ = ["# Fair comparison: EMBER vs CINDER (option B)", "",
          "Same model, data, seed 42, judge model (gemini-3.5-flash-lite), grid, pick rule (fast harmonic, MMLU gate, "
          "no Alpaca) and tests; only the method differs. ember = EMBER (paper's factorization and judge), ember_hj = "
          "EMBER's factorization + Ori's judge (top-20 by rho, TJ or LO@5), cinder = Ori's L123. ember_hj rows with "
          "reused_from = ember have exactly ember's edit, so ember's results. ret_L = retention of QA_test in L "
          "(1 = nothing erased); leak = mean(ret de/es/fr/ru) - ret_en; simdom_L = SimdomQA_test accuracy change vs the "
          "pristine model. No feature kept → no edit → the pristine model's numbers.", ""]
    if D.empty:
        (fb / "report.md").write_text("\n".join(L_ + ["No results yet."]) + "\n")
        return
    show = ["concept", "method", "n_features", "n_tokens_edited", "delta", "train_harmonic", "ret_en", "ret_de",
            "ret_es", "ret_fr", "ret_ru", "leak", "simdom_en", "mmlu_test", "edit_check", "reused_from"]
    L_ += ["## Per concept", "", md(D.sort_values(["concept", "method"])[show]), ""]
    for label, sub in (("all 18 concepts", D), ("clean 12 (no CINDER tuning / lexicon-prompt concepts)", D[D.clean])):
        agg = sub.groupby("method")[["ret_en", "ret_de", "ret_es", "ret_fr", "ret_ru", "leak", "simdom_en",
                                     "mmlu_test", "train_harmonic"]].mean().round(3).reset_index()
        agg.insert(1, "concepts", sub.groupby("method").size().values)
        agg.insert(2, "no_edit", sub.groupby("method").no_edit.sum().values)
        L_ += [f"## Means, {label}", "", md(agg), ""]
        piv = sub.pivot(index="concept", columns="method")
        for a_, b_ in PAIRS:
            if not {a_, b_} <= set(sub.method):
                continue
            L_ += [f"Paired over concepts ({label}), {a_} − {b_}:", ""]
            for col in ("ret_en", "leak", "train_harmonic"):
                dlt = (piv[col][a_] - piv[col][b_]).dropna()
                L_.append(f"- {col}: mean {dlt.mean():+.3f}, {a_} lower in {int((dlt < 0).sum())}, higher in "
                          f"{int((dlt > 0).sum())}, equal in {int((dlt == 0).sum())} of {len(dlt)} concepts"
                          f"{wilcoxon(dlt)}")
            L_.append("")
    L_ += ["Lower ret_en = more English erasure; lower leak = less leak; higher train_harmonic = better search score.", ""]
    (fb / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def wilcoxon(d) -> str:
    try:
        from scipy.stats import wilcoxon as w
        d = d[d != 0]
        return f" (Wilcoxon p = {w(d).pvalue:.3f})" if len(d) >= 5 else ""
    except Exception:  # noqa: BLE001
        return ""


def md(df) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and v != v) else str(v) for v in r) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("build-ember35", "build-ember-hj", "run", "pick", "report"))
    ap.add_argument("--method", choices=list(ROOTS))
    ap.add_argument("--concept")
    ap.add_argument("--concepts", nargs="+", help="several concepts in one job (one model load)")
    ap.add_argument("--deltas", type=float, nargs="+", default=GRID)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    if a.stage == "build-ember35":
        build_ember35()
    elif a.stage == "build-ember-hj":
        build_ember_hj()
    elif a.stage == "run":
        run(a.method, a.concepts or [a.concept], a.deltas)
    elif a.stage == "pick":
        for c in ALL_CONCEPTS:
            if (out_dir(a.method, c) / "grid.csv").exists() or not kept_features(a.method, c):
                pick(a.method, c)
    else:
        report()


if __name__ == "__main__":
    main()
