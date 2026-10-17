"""Synthetic variable-width runs for testing evaluate.py (NOT real results; no SLURM, no Gemini).

Each fake run starts from one of the earlier whitened seed-sweep fits (runs/whiten/gemma/Ancient_Rome/wiki_page*/
seed<N>/smf.pt, Z (d,k) and Y (k,n)) and runs a few iterations of the spec's loop (dense Y-step, Allocate, optional
support refit, Z-update) with a simplified allocation per method, on the real whitened A when it can be rebuilt
(results/varwidth/prep/, or experiments/varwidth/prep.py redirected into <root>/prep), else on A ~ Z@Y + noise.
The "judge" is faked: a feature is a concept feature when |cos z| with the sweep's concept feature is >= 0.6
(method m1 also selects a second feature in odd seeds to exercise the split code; m4 selects nothing in one seed).
Everything is written under --root (default: a scratch dir), never under results/varwidth/<concept>.

    python -m experiments.varwidth.fake_runs --root /tmp/vwfake [--seeds 42 43 44 45 46 47] [--iters 25]
    python -m experiments.varwidth.evaluate --concept "Ancient Rome" --root /tmp/vwfake
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.varwidth.common import (RESULTS, ROOT, VWConfig, budgets, churn, mask_from_row_widths,
                                         pool_floor_cap, rel_err, slug)

CONCEPT = "Ancient Rome"
SWEEP_FIT = ROOT / "runs" / "whiten" / "gemma" / "Ancient_Rome" / "wiki_page_g1_varnnz-per_row-uniform-nz0.5-ca11a0"
SWEEP_JSON = ROOT / "results" / "ancient_rome_seed_sweep"
LAM = 1e-4
MERGE_THR = 0.15  # spec: 0.4; lowered so the fake split pairs exercise the merge path

# (tag, VWConfig kwargs, extra) ; extra: S (|S| concept sentences), merge_from (tag whose fits are merged)
CONFIGS = [
    ("fixed_refit-False", dict(method="fixed", refit=False), {}),
    ("fixed", dict(method="fixed"), {}),
    ("m1", dict(method="m1"), {}),
    ("m2", dict(method="m2"), {}),
    ("m3", dict(method="m3"), {}),
    ("m4", dict(method="m4"), {}),
    ("m2_sparsity-0.005", dict(method="m2", sparsity=0.005), {}),
    ("m2_sparsity-0.02", dict(method="m2", sparsity=0.02), {}),
    ("m2_sparsity-0.04", dict(method="m2", sparsity=0.04), {}),
    ("m2_S200", dict(method="m2"), {"S": 200}),
    ("m1_merge", dict(method="m1"), {"merge_from": "m1"}),
]


# ------------------------------------------------------------------------------------------------ data
def get_prep(root: Path, S: int, Z0: torch.Tensor, Y0: torch.Tensor) -> dict:
    name = f"{slug(CONCEPT)}_whitened_S{S}.pt"
    for p in (RESULTS / "prep" / name, root / "prep" / name):
        if p.exists():
            return torch.load(p, weights_only=False)
    try:  # real prep (tokenizer + embeddings from the HF cache), redirected into <root>/prep
        from experiments.varwidth import prep
        prep.PREP_DIR = root / "prep"
        return prep.prepare(CONCEPT, "whitened", S)
    except Exception as e:  # noqa: BLE001 -- testing fallback only
        print(f"[fake] prep failed ({e!r}); using A = Z@Y + noise with fake tokens/codes")
    g = torch.Generator().manual_seed(S)
    A = Z0 @ Y0
    A = A + 0.5 * A.std() * torch.randn(A.shape, generator=g)
    n = A.shape[1]
    keep = torch.arange(n) if S >= 300 else torch.randperm(n, generator=g)[: int(n * S / 300)].sort().values
    rng = np.random.default_rng(S)
    out = {"A": A[:, keep].contiguous(), "vstar_ids": keep.tolist(), "tokens": [f"tok{int(i)}" for i in keep],
           "codes": rng.choice([0, 1, 2], size=len(keep), p=[0.5, 0.38, 0.12])}
    (root / "prep").mkdir(parents=True, exist_ok=True)
    torch.save(out, root / "prep" / name)
    return out


# ------------------------------------------------------------------------------------------------ allocation
def allocate(cfg: VWConfig, Yd: torch.Tensor, Z: torch.Tensor, K: int, m_min: int, m_max: int, state: dict, it: int):
    k, n = Yd.shape
    a = Yd.abs()
    if cfg.method == "fixed":
        return mask_from_row_widths(a, torch.full((k,), K // k, dtype=torch.long))
    if cfg.method in ("m1", "m4"):
        s = a * Z.norm(dim=0)[:, None]
        if cfg.method == "m4":  # stand-in for the lasso support: shrink small entries first
            s = (s - 0.5 * s.median(dim=1, keepdim=True).values).clamp_min(0) + 1e-12 * s
        return pool_floor_cap(s, K, m_min, m_max)
    if cfg.method == "m2":
        med = Yd.median(dim=1, keepdim=True).values
        sigma = (1.4826 * (Yd - med).abs().median(dim=1, keepdim=True).values).clamp_min(1e-8)
        return pool_floor_cap(a / sigma, K, m_min, m_max)
    if cfg.method == "m3":
        if "m" not in state or it % cfg.m3_every == 0:
            W = max(1, math.ceil(cfg.m3_window * n))
            top = a.topk(W, dim=1).values
            med = Yd.median(dim=1, keepdim=True).values
            sigma = (1.4826 * (Yd - med).abs().median(dim=1, keepdim=True).values).clamp_min(1e-8)
            w = (top - cfg.m3_c * sigma).clamp_min(0) ** 2
            sw = w.sum(1, keepdim=True)
            p = w / sw.clamp_min(1e-30)
            H = -(p * p.clamp_min(1e-30).log()).sum(1)
            r = torch.where(sw[:, 0] > 0, H.exp(), torch.full_like(H, float(m_min)))
            m = (r / r.sum() * K).floor().long().clamp(m_min, m_max)
            for _ in range(50):  # crude exact-K repair within [m_min, m_max]
                diff = K - int(m.sum())
                if diff == 0:
                    break
                room = (m < m_max) if diff > 0 else (m > m_min)
                idx = torch.nonzero(room)[:, 0]
                idx = idx[torch.argsort(r[idx], descending=diff > 0)][: abs(diff)]
                m[idx] += 1 if diff > 0 else -1
            state["m"] = m
        return mask_from_row_widths(a, state["m"])
    raise ValueError(cfg.method)


def refit_support(S: torch.Tensor, G: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """y_{S_j,j} = (G_SS + lam I)^-1 B_{S,j} per token (G = Z^T Z, B = Z^T A)."""
    k, n = S.shape
    Y = torch.zeros(k, n, dtype=B.dtype)
    cnt = S.sum(0)
    one = torch.nonzero(cnt == 1)[:, 0]
    if len(one):
        rows = S[:, one].float().argmax(0)
        Y[rows, one] = B[rows, one] / (G[rows, rows] + LAM)
    for j in torch.nonzero(cnt > 1)[:, 0].tolist():
        idx = torch.nonzero(S[:, j])[:, 0]
        Y[idx, j] = torch.linalg.solve(G[idx][:, idx] + LAM * torch.eye(len(idx), dtype=G.dtype), B[idx, j])
    return Y


def run_loop(cfg, A, Z, Y, iters, K, m_min, m_max, mask=None):
    k = Z.shape[1]
    I = torch.eye(k, dtype=A.dtype)
    ch, errs, state = [], [], {}
    prev = mask
    for it in range(iters):
        G = Z.T @ Z
        B = Z.T @ A
        Yd = torch.linalg.solve(G + LAM * I, B)
        S = allocate(cfg, Yd, Z, K, m_min, m_max, state, it)
        assert int(S.sum()) == K
        Y = refit_support(S, G, B) if cfg.refit else Yd * S
        ch.append(churn(prev, S, K))
        prev = S
        Z = A @ Y.T @ torch.linalg.inv(Y @ Y.T + LAM * I)
        nrm = Y.norm(dim=1).clamp_min(1e-12)
        Y, Z = Y / nrm[:, None], Z * nrm[None, :]
        errs.append(rel_err(A, Z, Y))
    return Z, Y, ch, errs, prev


# ------------------------------------------------------------------------------------------------ fake judge
def ratio_abs(Y: torch.Tensor, codes: np.ndarray) -> np.ndarray:
    a = Y.abs().numpy()
    c = codes
    conc = a[:, c != 0].sum(1) / max(1, (c != 0).sum())
    neu = a[:, c == 0].sum(1) / max(1, (c == 0).sum())
    return conc / (neu + 1e-9)


def fake_select(Z, Y, codes, ref_z, method, seed, descs):
    Zn = Z / Z.norm(dim=0).clamp_min(1e-12)
    cos = (Zn.T @ (ref_z / ref_z.norm())).abs().numpy()
    ratio = ratio_abs(Y, codes)
    order = np.argsort(-cos)
    sel = [int(order[0])] if cos[order[0]] >= 0.6 else []
    if method == "m1" and seed % 2 == 1:
        sel.append(int(order[1]))
    if method == "m4" and seed == 45:
        sel = []
    cand_ids = [i for i in np.argsort(-ratio)[:20] if ratio[i] >= 2.0]
    cand_ids = list(dict.fromkeys(cand_ids + [int(order[0]), int(order[1])]))
    cands = [{"feature": int(i), "ratio_abs": float(ratio[i]) + (10.0 if i == order[0] else 0.0),
              "description": descs.get(str(int(i)), f"feature {int(i)} (fake description)"),
              "is_member": int(i) in sel, "confidence": 0.95, "selected": int(i) in sel} for i in cand_ids]
    return sel, cands


# ------------------------------------------------------------------------------------------------ one run
def summarize(tag, cfg_dict, seed, Z, Y, prep, F_C, cands, descs, info, extra):
    codes = np.asarray(prep["codes"])
    widths = (Y != 0).sum(1).long()
    sup = sorted(set(int(j) for i in F_C for j in torch.nonzero(Y[i])[:, 0].tolist()))
    c = codes[sup] if sup else np.array([], dtype=int)
    widest = [{"feature": int(i), "width": int(widths[i]),
               "description": descs.get(str(int(i)), f"feature {int(i)} (fake description)")}
              for i in torch.argsort(widths, descending=True, stable=True)[:10].tolist()]
    s = {"concept": CONCEPT, "tag": tag, "seed": seed, "config": cfg_dict, "n_concept_features": len(F_C),
         "F_C": F_C, "n_distinct_tokens": len(sup),
         "tokens_per_feature": {str(i): int(widths[i]) for i in F_C},
         "C": int((c == 1).sum()), "N": int((c == 0).sum()), "B": int((c == 2).sum()),
         "rel_err": info["rel_err"], "vstar": int(Y.shape[1]), "fit_elapsed": info["elapsed"],
         "total_elapsed": info["elapsed"] + 30.0, "n_iter": info["n_iter"], "lockin_iter": info["lockin_iter"],
         "widths": widths.tolist(), "candidates": cands, "widest": widest}
    s.update(extra)
    return s


def lockin(ch, tol=0.005):
    last_bad = max([i for i, x in enumerate(ch) if x >= tol], default=-1)
    return last_bad + 1


def save_run(out: Path, fit: dict, summary: dict, prep):
    out.mkdir(parents=True, exist_ok=True)
    torch.save(fit, out / "fit.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    toks = prep["tokens"]
    lines = ["| " + " | ".join(f"f{i}" for i in summary["F_C"]) + " |", "|" + "---|" * max(1, len(summary["F_C"]))]
    Y = fit["Y"]
    cols = []
    for i in summary["F_C"]:
        nz = torch.nonzero(Y[i])[:, 0]
        nz = nz[torch.argsort(Y[i, nz].abs(), descending=True)]
        cols.append([f"{toks[j]} [{'NCB'[int(prep['codes'][j])]}] {float(Y[i, j]):.2f}" for j in nz.tolist()])
    for r in range(max((len(c) for c in cols), default=0)):
        lines.append("| " + " | ".join(c[r] if r < len(c) else "" for c in cols) + " |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="output root (NOT results/varwidth)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--tags", nargs="*", default=None)
    a = ap.parse_args()
    root = a.root.resolve()
    assert root != RESULTS.resolve(), "write fake runs outside results/varwidth"
    torch.manual_seed(0)
    preps = {}
    fits = {}
    for tag, kw, extra in CONFIGS:
        if a.tags and tag not in a.tags:
            continue
        cfg = VWConfig(**kw)
        cfg_dict = cfg.to_dict()
        S = extra.get("S", 300)
        if S != 300:
            cfg_dict["S"] = S
        for seed in a.seeds:
            t0 = time.time()
            sm = torch.load(SWEEP_FIT / f"seed{seed}" / "smf.pt", weights_only=False)
            sw = json.loads((SWEEP_JSON / f"whitened_seed{seed}.json").read_text())
            jpath = SWEEP_FIT / f"seed{seed}" / "judge.json"
            descs = json.loads(jpath.read_text()).get("descriptions", {}) if jpath.exists() else {}
            Z0, Y0 = sm["Z"].double(), sm["Y"].double()
            if S not in preps:
                preps[S] = get_prep(root, S, sm["Z"], sm["Y"])
            prep = preps[S]
            A = prep["A"].double()
            n = A.shape[1]
            K, m_min, m_max = budgets(n, cfg)
            ref_z = Z0[:, sw["F_C"][0]]
            mextra = {}
            if "merge_from" in extra:
                Z, Y, mask, ch, errs = fits[(extra["merge_from"], seed)]
                Z, Y, ch2, errs2, mextra = merge_step(cfg, A, Z.clone(), Y.clone(), mask, K, m_min, m_max, prep,
                                                      ref_z, seed, descs, a.iters)
                ch, errs = ch + ch2, errs + errs2
            else:
                Y_init = Y0 if Y0.shape[1] == n else torch.zeros(Y0.shape[0], n, dtype=A.dtype)
                Z, Y, ch, errs, mask = run_loop(cfg, A, Z0.clone(), Y_init, a.iters, K, m_min, m_max)
                fits[(tag, seed)] = (Z, Y, mask, ch, errs)
            F_C, cands = fake_select(Z, Y, np.asarray(prep["codes"]), ref_z, cfg.method, seed, descs)
            if mextra.get("merge", {}).get("accepted"):
                F_C = sorted(set(F_C) - {p["b"] for p in mextra["merge"]["pairs"] if p.get("accepted")})
            el = time.time() - t0
            info = {"rel_err": errs[-1], "elapsed": el, "n_iter": len(ch), "lockin_iter": lockin(ch)}
            sx = {}
            if tag == "m2":  # fake judge reruns: 3 reps, flip one feature now and then
                rng = np.random.default_rng(seed)
                sx["judge_reps"] = [F_C if rng.random() > 0.3 else sorted(set(F_C) ^ {int(rng.integers(100))})
                                    for _ in range(3)]
            sx.update(mextra)
            summ = summarize(tag, cfg_dict, seed, Z, Y, prep, F_C, cands, descs, info, sx)
            fit = {"Z": Z.float(), "Y": Y.float(), "widths": (Y != 0).sum(1).long(), "churn": ch,
                   "rel_err_hist": errs, "rel_err": errs[-1], "n_iter": len(ch), "lockin_iter": info["lockin_iter"],
                   "elapsed": el, "stop_reason": "max_iter (fake)", "K": K, "m_min": m_min, "m_max": m_max,
                   "config": cfg_dict, "seed": seed, "extra": {"fake": True}}
            save_run(root / slug(CONCEPT) / tag / f"seed{seed}", fit, summ, prep)
            print(f"[fake] {tag} seed{seed}: F_C={F_C} widths[F_C]={[int(summ['widths'][i]) for i in F_C]} "
                  f"rel_err={errs[-1]:.4f} ({el:.1f}s)", flush=True)


def merge_step(cfg, A, Z, Y, mask, K, m_min, m_max, prep, ref_z, seed, descs, iters):
    """Method 5 on a finished fit: merge concept-feature pairs with |cos z| >= 0.4, run more iterations, accept if
    rel_err rises by < 0.002."""
    F_C, _ = fake_select(Z, Y, np.asarray(prep["codes"]), ref_z, cfg.method, seed, descs)
    Zn = Z / Z.norm(dim=0).clamp_min(1e-12)
    pair_cos = [{"a": a_, "b": b_, "cos": float((Zn[:, a_] @ Zn[:, b_]).abs())}
                for i, a_ in enumerate(F_C) for b_ in F_C[i + 1:]]
    e0 = rel_err(A, Z, Y)
    pairs, ch, errs = [], [], []
    for pc in pair_cos:
        if pc["cos"] < MERGE_THR:
            pairs.append({**pc, "accepted": False, "reason": "cos below threshold"})
            continue
        a_, b_ = pc["a"], pc["b"]
        sup = (Y[a_] != 0) | (Y[b_] != 0)
        M = torch.outer(Z[:, a_], Y[a_]) + torch.outer(Z[:, b_], Y[b_])
        U, s, Vh = torch.linalg.svd(M[:, sup], full_matrices=False)
        Z2, Y2 = Z.clone(), Y.clone()
        Z2[:, a_] = U[:, 0] * s[0]
        Y2[a_] = 0
        Y2[a_, sup] = Vh[0]
        Z2[:, b_] = torch.randn(Z.shape[0], dtype=Z.dtype)
        Y2[b_] = 0
        Z2, Y2, ch, errs, _ = run_loop(cfg, A, Z2, Y2, max(5, iters // 3), K, m_min, m_max, mask)
        e1 = errs[-1]
        acc = (e1 - e0) < 0.002
        pairs.append({**pc, "merged_feature": a_, "rel_err_before": e0, "rel_err_after": e1,
                      "delta_rel_err": e1 - e0, "passes_rho": True, "passes_judge": True, "accepted": bool(acc)})
        if acc:
            Z, Y = Z2, Y2
        break
    merge = {"threshold": MERGE_THR, "concept_pair_cosines": pair_cos, "pairs": pairs, "rel_err_before": e0,
             "rel_err_after": rel_err(A, Z, Y), "accepted": any(p.get("accepted") for p in pairs)}
    return Z, Y, ch, errs, {"merge": merge}


if __name__ == "__main__":
    main()
