#!/usr/bin/env python
"""EMBER embedding erasure of one ladder rung (PREREG "Erasure anchor") on gemma-2-2b-it, MC protocol.

    python -m experiments.varwidth.erase baselines --concepts ...          # original model, once (cached)
    python -m experiments.varwidth.erase run --concept "Harry Potter" --rung L0 L1 L12 L123 [--seed 42] [--alpaca-n 50]

F_C = the rung's judged features (layer_eval.fc); edited tokens = concept-only tokens of support(F_C) (EMBER
embed_edit.py); c_t = sum_{i in F_C} Y_it z_i, mapped back from the whitened space with T_inv (row form,
x = (e - mu_B) M, so Delta e = Delta x M^-1; the mean cancels); e_t <- e_t - delta c_t on the TL-scaled rows, written
back / sqrt(d) (the notebook's apply_ember_edit). lm_head is untied (model_loader.load_hf_model).

Grid (validation split: qa_train / simdom_train 50 MC each, MMLU train 50) over delta in {0.5, 1, 2, 5, 10, 50, 100,
200}: gate MMLU-frac >= 0.9, H2 = HM(efficacy, specificity), pick max H2 (ties: smaller delta; all 0: no edit).
Unlike the paper, Alpaca is not in the grid (Gemini cost, PREREG); at the chosen delta the test split is scored:
qa_test, simdom_test, MMLU test (1000), and Alpaca (first --alpaca-n test prompts, instruction + fluency judged by
Gemini) -> efficacy = 1 - qa_frac, specificity = HM(mmlu_frac, simdom_frac), coherence = HM(instr_frac, flu_frac),
H = HM(efficacy, specificity, coherence) (+ H2 without coherence). Normalizations as ember/evals/orchestrator.py.

Outputs results/varwidth/layers/erasure/<slug>/<rung>_s<seed>/{grid.json, result.json}; grid rows are saved after
every delta, so a killed job resumes. Baselines: results/varwidth/layers/erasure/baselines.json.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from experiments.varwidth.common import RESULTS, ROOT, run_dir, slug
from experiments.varwidth.layer_eval import CONFIGS, RUNGS, build_cache, fc

MODEL = "google/gemma-2-2b-it"
GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
MIN_MMLU = 0.9
OUT = RESULTS / "layers" / "erasure"
BASE = OUT / "baselines.json"


def hm(vals):
    from ember.evals.harmonic import harmonic_mean
    return harmonic_mean(vals)


def mc_norm(acc, base):
    return min(1.0, max((acc - 0.25) / (base - 0.25), 0.0)) if base != 0.25 else 0.0


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


class Evals:
    """Eval sets and scorers on one wrapped model."""

    def __init__(self, concept, alpaca_n):
        from ember.evals.mc import prepare_mc_items
        from ember.evals.mmlu import load_mmlu_items
        from ember.local_datasets import load_mc_qa_items, load_mmlu_indices
        self.mc = {f"{s}_{sp}": prepare_mc_items(load_mc_qa_items(f"{s}_{sp}", concept), seed=42)
                   for s in ("qa", "simdom") for sp in ("train", "test")} if concept else {}
        self.mmlu = {sp: load_mmlu_items(load_mmlu_indices(sp)) for sp in ("train", "test")}
        self.alpaca_n = alpaca_n

    def mc_acc(self, w, name):
        from ember.evals.mc import evaluate_mc_generation
        c, _, _ = evaluate_mc_generation(w, self.mc[name])
        return c / len(self.mc[name])

    def mmlu_acc(self, w, split):
        from ember.evals.mmlu import evaluate_mmlu_generation
        c, _, _ = evaluate_mmlu_generation(w, self.mmlu[split])
        return c / len(self.mmlu[split])

    def alpaca(self, w, split="test", n=None):
        """Mean instruction / fluency scores (0-2) on the first n (default alpaca_n) prompts of the split."""
        from ember.evals.alpaca import load_alpaca_eval_local
        from ember.evals.gemini import GeminiEvaluator
        from ember.local_datasets import load_alpaca_indices
        items = [ex["instruction"] for ex in load_alpaca_eval_local() if "instruction" in ex]
        inst = [items[i] for i in load_alpaca_indices(split)[:(n or self.alpaca_n)]]
        gens = w.generate_multiple(inst, max_new_tokens=200, do_sample=False, batch_size=8)
        ev = GeminiEvaluator()

        def one(p):
            i, c = p
            try:
                return ev.score_alpaca_instruct(i, c), ev.score_alpaca_fluency(c + ".")
            except RuntimeError as e:
                if type(e).__name__ == "BudgetExceeded":
                    raise
                return 0, 0
        with ThreadPoolExecutor(16) as ex:
            sc = list(ex.map(one, zip(inst, gens)))
        return float(np.mean([s[0] for s in sc])), float(np.mean([s[1] for s in sc]))


def load_model():
    from ember.erasure.model_loader import load_hf_model
    from ember.evals.model_wrap import WrappedHFModel
    model, tok = load_hf_model(MODEL)
    model.eval()
    model.config.use_cache = True
    return model, WrappedHFModel(model, tok)


def _locked_update(path, fn):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(path.read_text()) if path.exists() else {}
        fn(d)
        save(path, d)
        return d


def baselines(concepts, alpaca_n):
    have = json.loads(BASE.read_text()) if BASE.exists() else {}
    todo = [c for c in concepts if c not in have]
    glob_todo = [k for k in ("mmlu_train", "mmlu_test", f"alpaca{alpaca_n}") if k not in have.get("_global", {})]
    if not todo and not glob_todo:
        return have
    _, w = load_model()
    if glob_todo:
        ev = Evals(None, alpaca_n)
        g = {}
        if "mmlu_train" in glob_todo:
            g["mmlu_train"] = ev.mmlu_acc(w, "train")
        if "mmlu_test" in glob_todo:
            g["mmlu_test"] = ev.mmlu_acc(w, "test")
        if f"alpaca{alpaca_n}" in glob_todo:
            g[f"alpaca{alpaca_n}"] = ev.alpaca(w)
        _locked_update(BASE, lambda d: d.setdefault("_global", {}).update(g))
        print("global baselines", g, flush=True)
    for c in todo:
        ev = Evals(c, alpaca_n)
        r = {k: ev.mc_acc(w, k) for k in ev.mc}
        _locked_update(BASE, lambda d: d.__setitem__(c, r))
        print(c, r, flush=True)
    return json.loads(BASE.read_text())


def edit_vectors(concept, rung, seed):
    """(token ids, c_t rows in TL-scaled embedding space, F_C features, info)."""
    cfg, judge = RUNGS[rung]
    cache = build_cache(concept, cfg, seed)
    rows = fc(cache, judge)
    F = [c["feature"] for c in rows]
    from experiments.varwidth.prep import prepare
    tag = CONFIGS[cfg]
    whit = not tag.endswith("_plain")
    prep = prepare(concept, "whitened" if whit else "plain", verbose=False)
    fit = torch.load(run_dir(concept, tag, seed) / "fit.pt", map_location="cpu", weights_only=False)
    Z, Y = fit["Z"].double(), fit["Y"].double()
    codes = np.asarray(prep["codes"])
    if not F:
        return [], torch.zeros(0, Z.shape[0]), F, {"n_edit": 0}
    touched = (Y[F].abs() > 0).any(0).numpy()
    cols = np.where(touched & (codes == 1))[0]
    C = (Z[:, F] @ Y[F][:, cols]).T                     # (n_edit, d) in the factorized space
    if whit:
        C = C @ prep["T_inv"].double()                  # back to the TL-scaled embedding space
    ids = [int(prep["vstar_ids"][j]) for j in cols]
    return ids, C.float(), F, {"n_edit": len(ids), "whitened": whit}


def check_space(concept, rung, seed, W, scale):
    """The fit's matrix columns must equal the transformed HF rows (the edit space matches the model)."""
    from experiments.varwidth.prep import prepare
    cfg, _ = RUNGS[rung]
    whit = not CONFIGS[cfg].endswith("_plain")
    prep = prepare(concept, "whitened" if whit else "plain", verbose=False)
    js = list(range(0, len(prep["vstar_ids"]), max(1, len(prep["vstar_ids"]) // 50)))[:50]
    e = W[torch.tensor([prep["vstar_ids"][j] for j in js], device=W.device)].double().cpu() * scale
    x = (e - prep["mu_B"].double()) @ prep["T_matrix"].double() if whit else e
    a = prep["A"][:, js].T.double()
    cos = torch.nn.functional.cosine_similarity(x, a, dim=1).min().item()
    ratio = (x.norm(dim=1) / a.norm(dim=1))
    assert cos > 0.99 and (ratio - 1).abs().max() < 0.02, (cos, ratio.min().item(), ratio.max().item())
    return cos


def run(concept, rung, seed, alpaca_n, mw=None, paper_grid=False):
    """mw: (model, wrapped) to reuse (its embedding is restored from the snapshot afterwards). paper_grid: the
    paper's own grid (Alpaca on the 150 validation prompts at every delta that passes the MMLU gate, Alpaca gate 0.9,
    pick by H = HM(efficacy, specificity, coherence)); output dir <rung>_s<seed>_papergrid."""
    d = OUT / slug(concept) / (f"{rung}_s{seed}" + ("_papergrid" if paper_grid else ""))
    if (d / "result.json").exists():
        print(f"done: {d / 'result.json'}")
        return
    base = json.loads(BASE.read_text())
    bc, bg = base[concept], base["_global"]
    t0 = time.time()
    ids, C, F, info = edit_vectors(concept, rung, seed)
    for other in ([] if paper_grid else RUNGS):   # same config and same F_C (TWJ adds nothing to TJ): copy the edit
        o = OUT / slug(concept) / f"{other}_s{seed}" / "result.json"
        if other != rung and RUNGS[other][0] == RUNGS[rung][0] and o.exists():
            r = json.loads(o.read_text())
            if sorted(r["F_C"]) == sorted(F) and "copied_from" not in r:
                save(d / "result.json", {**r, "rung": rung, "copied_from": other})
                print(f"{concept} {rung}: F_C = {F} as in {other}, result copied", flush=True)
                return
    model, w = mw or load_model()
    W = model.get_input_embeddings().weight
    scale = math.sqrt(W.shape[1])
    info["space_cos"] = check_space(concept, rung, seed, W, scale)
    idx = torch.tensor(ids, dtype=torch.long, device=W.device)
    orig = W.index_select(0, idx).detach().float().clone() if ids else None

    def set_delta(delta):
        if not ids:
            return
        with torch.no_grad():
            e_new = orig * scale - delta * C.to(W.device)
            W.index_copy_(0, idx, (e_new / scale).to(W.dtype))

    ev = Evals(concept, alpaca_n)
    if paper_grid and "alpaca_train150" not in bg:
        set_delta(0.0)
        bg["alpaca_train150"] = list(ev.alpaca(w, "train", 150))
        _locked_update(BASE, lambda dd: dd["_global"].update(alpaca_train150=bg["alpaca_train150"]))
    gpath = d / "grid.json"
    grid = json.loads(gpath.read_text()) if gpath.exists() else []
    done = {g["delta"] for g in grid}
    for delta in GRID if ids else []:
        if delta in done:
            continue
        set_delta(delta)
        r = {"delta": delta, "mmlu_acc": ev.mmlu_acc(w, "train")}
        r["mmlu_frac"] = mc_norm(r["mmlu_acc"], bg["mmlu_train"])
        if r["mmlu_frac"] < MIN_MMLU:
            r.update(H2=0.0, gate="mmlu")
        else:
            for n in ("qa", "simdom"):
                r[f"{n}_acc"] = ev.mc_acc(w, f"{n}_train")
                r[f"{n}_frac"] = mc_norm(r[f"{n}_acc"], bc[f"{n}_train"])
            r["efficacy"] = 1 - r["qa_frac"]
            r["specificity"] = hm([r["mmlu_frac"], r["simdom_frac"]])
            r.update(H2=hm([r["efficacy"], r["specificity"]]), gate="")
            if paper_grid:
                ai, af = ev.alpaca(w, "train", 150)
                bi, bf = bg["alpaca_train150"]
                r.update(instr_frac=min(1.0, ai / bi), flu_frac=min(1.0, af / bf))
                r["coherence"] = hm([r["instr_frac"], r["flu_frac"]])
                r["H3"] = hm([r["efficacy"], r["specificity"], r["coherence"]])
                if r["instr_frac"] < 0.9 or r["flu_frac"] < 0.9:
                    r.update(H3=0.0, gate="alpaca")
        grid.append(r)
        save(gpath, grid)
        print(f"{concept} {rung} delta={delta}: H2={r['H2']:.3f} {r['gate']}", flush=True)
    hk = "H3" if paper_grid else "H2"
    passed = [g for g in grid if g.get(hk, 0) > 0]
    best = 0.0 if not passed else sorted(passed, key=lambda g: (-g[hk], g["delta"]))[0]["delta"]
    set_delta(best)
    res = {"concept": concept, "rung": rung, "seed": seed, "F_C": F, "edit_ids": ids, **info, "delta": best}
    res["qa_acc"], res["simdom_acc"] = ev.mc_acc(w, "qa_test"), ev.mc_acc(w, "simdom_test")
    res["mmlu_acc"] = ev.mmlu_acc(w, "test")
    res["qa_frac"] = mc_norm(res["qa_acc"], bc["qa_test"])
    res["simdom_frac"] = mc_norm(res["simdom_acc"], bc["simdom_test"])
    res["mmlu_frac"] = mc_norm(res["mmlu_acc"], bg["mmlu_test"])
    res["efficacy"] = 1 - res["qa_frac"]
    res["specificity"] = hm([res["mmlu_frac"], res["simdom_frac"]])
    if alpaca_n:
        bi, bf = bg[f"alpaca{alpaca_n}"]
        if best == 0.0:
            ai, af = bi, bf
        else:
            ai, af = ev.alpaca(w)
        res.update(alpaca_instr=ai, alpaca_flu=af, instr_frac=min(1.0, ai / bi), flu_frac=min(1.0, af / bf))
        res["coherence"] = hm([res["instr_frac"], res["flu_frac"]])
        res["H"] = hm([res["efficacy"], res["specificity"], res["coherence"]])
    res["H2"] = hm([res["efficacy"], res["specificity"]])
    res["seconds"] = time.time() - t0
    save(d / "result.json", res)
    set_delta(0.0)   # restore the original rows for the next rung on the same model
    print(json.dumps({k: v for k, v in res.items() if k != "edit_ids"}), flush=True)


def _copyable(concept, rung, seed) -> bool:
    ids, C, F, info = edit_vectors(concept, rung, seed)
    for other in RUNGS:
        o = OUT / slug(concept) / f"{other}_s{seed}" / "result.json"
        if other != rung and RUNGS[other][0] == RUNGS[rung][0] and o.exists():
            if sorted(json.loads(o.read_text())["F_C"]) == sorted(F):
                return True
    return False


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("baselines", "run"))
    ap.add_argument("--concepts", nargs="*", default=[])
    ap.add_argument("--concept")
    ap.add_argument("--rung", choices=list(RUNGS), nargs="+")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpaca-n", type=int, default=50)
    ap.add_argument("--paper-grid", action="store_true", help="the paper's grid with Alpaca at every delta (fidelity)")
    a = ap.parse_args()
    if a.what == "baselines":
        baselines(a.concepts, a.alpaca_n)
    else:
        sfx = "_papergrid" if a.paper_grid else ""
        todo = [r for r in a.rung if not (OUT / slug(a.concept) / f"{r}_s{a.seed}{sfx}" / "result.json").exists()]
        mw = None
        for r in todo:   # one model load (~7 min from NFS) for all rungs of the concept, only when needed
            if mw is None and (a.paper_grid or not _copyable(a.concept, r, a.seed)):
                mw = load_model()
            run(a.concept, r, a.seed, a.alpaca_n, mw, a.paper_grid)


if __name__ == "__main__":
    main()
