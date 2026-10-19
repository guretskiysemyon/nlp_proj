#!/usr/bin/env python
"""CPU-only diagnosis of why A3 (ML-CINDER) erases less than A2 (ML-EMBER), from saved fits / selections / evals.

    CUDA_VISIBLE_DEVICES= python -m experiments.ml.diagnose_a3

Reads results/ml/runs/<slug>/{A2,A3}; writes results/ml/diagnose_a3/{edits.json,curves.json}. No model, no Gemini.
Per concept: edited tokens, ||c_t||/||e_t|| at delta=1, overlap with A2, cos(c_A3, c_A2) on shared tokens,
cos(c_t, e_t) and the share of c_t along e_t (the direction EMBER removes), the share of c_t's energy in the top
background principal directions (what the model "reads" most), token frequency in the full set, and per-delta
efficacy curves.
"""
from __future__ import annotations

import json
from collections import Counter

import torch

from experiments.ml.common import DATA, LANGS, RESULTS, embeddings, slug, tokenizer
from experiments.ml.evaluate import run_edit
from experiments.ml.fit import _flatten, load_full

OUT = RESULTS / "diagnose_a3"
RUNS = RESULTS / "runs"


def q(x: torch.Tensor):
    x = x.float()
    return {k: round(float(torch.quantile(x, p)), 4) for k, p in (("p10", .1), ("p50", .5), ("p90", .9))} | \
        {"mean": round(float(x.mean()), 4)}


def stack(edit: dict, ids):
    return torch.stack([edit[t].float() for t in ids])


def main():
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    E = embeddings()
    bg = torch.load(DATA / "bg_ml_llama.pt")
    V, lam = bg["Q"], bg["lam"]
    order = torch.argsort(lam, descending=True)
    V, lam = V.double()[:, order].float(), lam.double()[order].float()
    tok = tokenizer()
    rep, curves = {}, {}
    for d in sorted(RUNS.iterdir()):
        a2, a3 = d / "A2", d / "A3"
        if not ((a3 / "selection.json").exists() and json.loads((a3 / "selection.json").read_text())["F_C"]):
            continue
        concept = json.loads((a3 / "selection.json").read_text())["concept"]
        full = load_full(concept)
        s, lab, ls = _flatten(full)
        freq = Counter()
        for x, l_ in zip(s, lab):
            if l_ == "concept":
                freq.update(tok(x, add_special_tokens=False)["input_ids"])
        r = {}
        eds = {}
        for A, p in (("A2", a2), ("A3", a3)):
            if not (p / "selection.json").exists() or not json.loads((p / "selection.json").read_text())["F_C"]:
                continue
            ed = run_edit(p / "fit.pt", p / "selection.json")
            eds[A] = ed
            ids = sorted(ed)
            C, Et = stack(ed, ids), E[ids]
            ratio = C.norm(dim=1) / Et.norm(dim=1)
            cos_e = torch.nn.functional.cosine_similarity(C, Et, dim=1)
            proj = (C @ V)                                   # coordinates on background PCs
            en = proj.pow(2)
            top = {f"top{k}": round(float((en[:, :k].sum(1) / en.sum(1)).mean()), 4) for k in (10, 100, 1000)}
            # variance-weighted norm: how large the edit is in units of background spread (= what whitening removes)
            wnorm = (proj.pow(2) / lam.clamp_min(1e-12)).sum(1).sqrt()
            enorm = ((Et - Et.mean(0)) @ V).pow(2).div(lam.clamp_min(1e-12)).sum(1).sqrt()
            ev = json.loads((p / "eval.json").read_text()) if (p / "eval.json").exists() else None
            bd = ev.get("best_delta") if ev else None
            f = torch.tensor([freq[t] for t in ids], dtype=torch.float)
            r[A] = {"n_edited": len(ids), "F_C": json.loads((p / "selection.json").read_text())["F_C"],
                    "ratio_c_e": q(ratio), "cos_c_e": q(cos_e), "pc_energy": top,
                    "whitened_norm_ratio": q(wnorm / enorm.clamp_min(1e-9)),
                    "concept_freq": q(f), "share_freq1": round(float((f <= 1).float().mean()), 4),
                    "best_delta": bd,
                    "eff_ratio_at_best": q(ratio * bd) if bd else None,
                    "test_score": ev["test"]["score"] if ev and "test" in ev else None,
                    "top_tokens": [tok.decode([t]) for t in sorted(ids, key=lambda t: -float(ed[t].norm()))[:15]]}
            if ev:
                cv = {}
                for dl, g in sorted(ev["grid"].items(), key=lambda kv: float(kv[0])):
                    el = [v for v in g["per_lang"].values() if v.get("eligible", True)]
                    cv[dl] = {"score": round(g["score"], 3), "gated": g["gated"],
                              "eff": {L: round(v.get("efficacy", -1), 2) for L, v in g["per_lang"].items()},
                              "mean_eff": round(sum(v.get("efficacy", 0) for v in el) / max(1, len(el)), 3),
                              "min_mmlu": round(min(v.get("mmlu_frac", 9) for v in g["per_lang"].values()), 2),
                              "min_flu": round(min(v.get("fluency", 9) for v in g["per_lang"].values()), 3)}
                curves.setdefault(concept, {})[A] = cv
                r[A]["test_eff"] = None if "test" not in ev else {L: round(v["efficacy"], 2) for L, v in ev["test"]["per_lang"].items()}
        if "A2" in eds:
            sh = sorted(set(eds["A2"]) & set(eds["A3"]))
            u = set(eds["A2"]) | set(eds["A3"])
            r["overlap"] = {"shared": len(sh), "jaccard": round(len(sh) / len(u), 3),
                            "a3_only": len(set(eds["A3"]) - set(eds["A2"])),
                            "a2_only": len(set(eds["A2"]) - set(eds["A3"]))}
            if sh:
                c2, c3 = stack(eds["A2"], sh), stack(eds["A3"], sh)
                r["overlap"]["cos_c3_c2"] = q(torch.nn.functional.cosine_similarity(c2, c3, dim=1))
                r["overlap"]["norm_c3_over_c2"] = q(c3.norm(dim=1) / c2.norm(dim=1))
                # how much of A2's edit (the direction known to erase) A3 delivers: projection of c3 on c2-hat
                r["overlap"]["proj_c3_on_c2_over_c2"] = q((c3 * c2).sum(1) / c2.pow(2).sum(1))
        rep[concept] = r
        print(concept, json.dumps(r, ensure_ascii=False)[:1500], flush=True)
    (OUT / "edits.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (OUT / "curves.json").write_text(json.dumps(curves, indent=1, ensure_ascii=False))




# ---------------------------------------------------------------- layer-0 read gain (python -m ... read)
def read_gain():
    """How strongly layer 0 reads a unit direction: ||W_in (g * v_hat)|| for W_in = [q;k;v;gate;up] of layer 0 and
    g its input RMSNorm weight, for c_t (A2, A3) and e_t. Ratio c/e > 1 means the edit direction is read more than a
    typical embedding direction."""
    from safetensors import safe_open
    from experiments.ml.common import model_dir
    torch.set_num_threads(8)
    md = model_dir()
    wm = json.loads((md / "model.safetensors.index.json").read_text())["weight_map"]
    names = ["input_layernorm.weight"] + [f"self_attn.{p}_proj.weight" for p in "qkv"] + \
        [f"mlp.{p}_proj.weight" for p in ("gate", "up")]
    W = {}
    for n in names:
        key = f"model.layers.0.{n}"
        with safe_open(str(md / wm[key]), framework="pt") as f:
            W[n] = f.get_tensor(key).float()
    g = W.pop("input_layernorm.weight")
    Win = torch.cat(list(W.values()))
    E = embeddings()

    def gain(X):
        X = X / X.norm(dim=1, keepdim=True)
        return (X * g) @ Win.T

    rnd = torch.randn(2000, E.shape[1])
    base_e = gain(E[torch.randint(0, E.shape[0], (2000,))]).norm(dim=1).median()
    base_r = gain(rnd).norm(dim=1).median()
    out = {"median_gain_random_dir": round(float(base_r), 3), "median_gain_embedding": round(float(base_e), 3)}
    for d in sorted(RUNS.iterdir()):
        r = {}
        for A in ("A2", "A3"):
            p = d / A
            if not (p / "selection.json").exists() or not json.loads((p / "selection.json").read_text())["F_C"]:
                continue
            ed = run_edit(p / "fit.pt", p / "selection.json")
            ids = sorted(ed)
            gc = gain(stack(ed, ids)).norm(dim=1)
            ge = gain(E[ids]).norm(dim=1)
            r[A] = {"gain_c_over_random": round(float(gc.median() / base_r), 3),
                    "gain_e_over_random": round(float(ge.median() / base_r), 3)}
        rz = OUT / f"refitz_{d.name}.pt"
        if rz.exists():
            gz = gain(torch.load(rz)["C"]).norm(dim=1)
            r["A3_refitZ"] = {"gain_c_over_random": round(float(gz.median() / base_r), 3)}
        if "A3" in r:
            out[d.name] = r
            print(d.name, r, flush=True)
    (OUT / "read_gain.json").write_text(json.dumps(out, indent=1))
    print(out)




# ---------------------------------------------------------------- candidate fix, offline only (python -m ... refitz)
def refit_z():
    """Fix candidate: keep A3's support (Y, F_C, edited tokens) but re-solve Z in the ORIGINAL embedding space with Y
    fixed (ridge), c_t = Z_orig[:, F_C] y_t. Reports ||c||/||e||, cos(c, e), cos with A2's c on shared tokens."""
    torch.set_num_threads(8)
    E = embeddings()
    out = {}
    for d in sorted(RUNS.iterdir()):
        p = d / "A3"
        if not (p / "selection.json").exists():
            continue
        F_C = json.loads((p / "selection.json").read_text())["F_C"]
        if not F_C:
            continue
        fit = torch.load(p / "fit.pt", weights_only=False)
        Y = fit["Y"].double()
        A = E[fit["vstar_ids"]].double().T
        Zo = torch.linalg.solve(Y @ Y.T + 1e-4 * torch.eye(Y.shape[0], dtype=torch.float64), Y @ A.T).T.float()
        ed3 = run_edit(p / "fit.pt", p / "selection.json")
        codes = torch.as_tensor(fit["codes"])
        Yf = Y.float()[F_C]
        cols = torch.nonzero((Yf != 0).any(0) & (codes == 1)).flatten()
        C = (Zo[:, F_C] @ Yf[:, cols]).T
        ids = [int(fit["vstar_ids"][int(j)]) for j in cols]
        Et = E[ids]
        r = {"ratio_c_e": q(C.norm(dim=1) / Et.norm(dim=1)),
             "cos_c_e": q(torch.nn.functional.cosine_similarity(C, Et, dim=1)),
             "cos_with_a3_unmapped": q(torch.nn.functional.cosine_similarity(C, stack(ed3, ids), dim=1))}
        a2 = d / "A2"
        if (a2 / "selection.json").exists() and json.loads((a2 / "selection.json").read_text())["F_C"]:
            ed2 = run_edit(a2 / "fit.pt", a2 / "selection.json")
            sh = [i for i, t in enumerate(ids) if t in ed2]
            if sh:
                r["cos_with_a2_shared"] = q(torch.nn.functional.cosine_similarity(
                    C[sh], stack(ed2, [ids[i] for i in sh]), dim=1))
        out[d.name] = r
        print(d.name, r, flush=True)
        torch.save({"ids": ids, "C": C}, OUT / f"refitz_{d.name}.pt")
    (OUT / "refit_z.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    {"read": read_gain, "refitz": refit_z}.get((sys.argv[1:] or [""])[0], main)()
