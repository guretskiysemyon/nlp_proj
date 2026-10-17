"""One (concept, config, seed) run of the variable-width study -> results/varwidth/<concept>/<tag>/seed<N>/.

    python -m experiments.varwidth.run --concept "Ancient Rome" --method m2 --seed 42
        [--matrix whitened|plain] [--sparsity 0.01] [--S 300] [--init randn|svd] [--no-refit] [--no-floorcap]
        [--continuation] [--judge-reps 0] [--merge-from TAG] [--tag-suffix SFX] [--threads 8] [--force]

Steps: prep (cached) -> core.fit_varwidth on prep["A"] -> select_features (rho + 2-stage judge, cached in judge.json)
-> stage-1 descriptions of the 10 widest features -> fit.pt, summary.json, tables.md (schema: common.py docstring).

tag = cfg.tag() [+ "_plain"] [+ "_S<S>"] [+ tag suffix]. With --merge-from TAG (method 5) the output tag is
TAG + "_merge" [+ suffix] and the config is the source run's.

A run whose summary.json exists is skipped unless --force; with --judge-reps N on an existing run only the missing
judge reruns are done (the fit is frozen).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.varwidth.common import VWConfig, rel_err, run_dir

MERGE_COS = 0.4
MERGE_MAX_RISE = 0.002
MERGE_EXTRA_ITERS = 300
N_WIDEST = 10


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


def make_tag(cfg: VWConfig, matrix: str, S: int, suffix: str = "") -> str:
    tag = cfg.tag()
    if matrix == "plain":
        tag += "_plain"
    if S != 300:
        tag += f"_S{S}"
    return tag + suffix


def stub_fit(path: str, A: torch.Tensor, cfg: VWConfig, seed: int) -> dict:
    """Testing only: an old smf.pt (keys Z, Y, rel_err) dressed up as a fit_varwidth result."""
    ck = torch.load(path, weights_only=False)
    Z, Y = ck["Z"].float(), ck["Y"].float()
    hist = [float(x) for x in ck.get("loss_history", [])]
    return {"Z": Z, "Y": Y, "widths": (Y != 0).sum(1).long(), "churn": [], "rel_err_hist": hist,
            "rel_err": rel_err(A, Z, Y), "n_iter": len(hist), "lockin_iter": -1, "elapsed": float(ck.get("elapsed", 0)),
            "stop_reason": "stub", "K": int((Y != 0).sum()), "m_min": int((Y != 0).sum(1).min()),
            "m_max": int((Y != 0).sum(1).max()), "config": cfg.to_dict(), "seed": seed, "extra": {"stub": path}}


def do_fit(A, cfg, seed, stub=None) -> dict:
    if stub:
        return stub_fit(stub, A, cfg, seed)
    from experiments.varwidth import core  # lazy: core.py is optional for prep/select work
    return core.fit_varwidth(A, cfg, seed, "cpu")


def widest_features(widths, n=N_WIDEST):
    w = torch.as_tensor(widths)
    order = torch.sort(w, descending=True, stable=True).indices[:n].tolist()
    return [(int(f), int(w[f])) for f in order]


def selection_summary(fit: dict, prep: dict, sel: dict, widest_desc: dict, widest) -> dict:
    from experiments.varwidth.select import concept_token_counts
    Y = fit["Y"]
    cnt = concept_token_counts(Y, prep, sel["F_C"])
    return {"n_concept_features": len(sel["F_C"]), "F_C": sel["F_C"], "n_distinct_tokens": cnt["n_distinct_tokens"],
            "tokens_per_feature": cnt["tokens_per_feature"], "C": cnt["C"], "N": cnt["N"], "B": cnt["B"],
            "rel_err": float(fit["rel_err"]), "widths": [int(x) for x in (Y != 0).sum(1).tolist()],
            "candidates": sel["candidates"],
            "widest": [{"feature": f, "width": w, "description": widest_desc.get(f)} for f, w in widest]}


def judge_reps(Y, prep, out: Path, n: int, existing: list, workers: int) -> list:
    from experiments.varwidth.select import select_features
    reps = list(existing)
    for r in range(len(reps) + 1, n + 1):
        t0 = time.time()
        s = select_features(Y, prep, out / f"judge_rep{r}.json", use_cache=True, workers=workers)
        reps.append(s["F_C"])
        log(f"judge rerun {r}/{n}: F_C = {s['F_C']} ({time.time() - t0:.1f}s)")
    return reps


def write_tables(out: Path, Y, prep, F_C, title):
    from experiments.varwidth.select import wide_table_md
    (out / "tables.md").write_text(wide_table_md(Y, prep, F_C, title))


def run_one(a) -> Path:
    from experiments.varwidth.prep import prepare
    from experiments.varwidth.select import describe_features, select_features
    t_all = time.time()
    timings = {}

    if a.merge_from:
        return run_merge(a)

    cfg = VWConfig(method=a.method, sparsity=a.sparsity, init=a.init, refit=not a.no_refit,
                   floorcap=not a.no_floorcap, continuation=a.continuation)
    tag = make_tag(cfg, a.matrix, a.S, a.tag_suffix)
    out = run_dir(a.concept, tag, a.seed)
    summ_path = out / "summary.json"

    if summ_path.exists() and not a.force:
        summ = json.loads(summ_path.read_text())
        have = summ.get("judge_reps", [])
        if a.judge_reps > len(have):
            log(f"{out}: exists; adding judge reruns {len(have) + 1}..{a.judge_reps} on the frozen fit")
            prep = prepare(a.concept, a.matrix, a.S, a.sub_seed)
            fit = torch.load(out / "fit.pt", weights_only=False)
            t0 = time.time()
            summ["judge_reps"] = judge_reps(fit["Y"], prep, out, a.judge_reps, have, a.workers)
            summ.setdefault("timings", {})["judge_reps"] = time.time() - t0
            save_json(summ_path, summ)
        else:
            log(f"skip {out}: summary.json exists (use --force)")
        return out
    if a.force and out.exists():
        for p in list(out.glob("judge*.json")):
            p.replace(p.with_name(p.name + ".stale"))
    out.mkdir(parents=True, exist_ok=True)
    log(f"run {a.concept!r} {tag} seed {a.seed} -> {out}")

    t0 = time.time()
    prep = prepare(a.concept, a.matrix, a.S, a.sub_seed)
    timings["prep"] = time.time() - t0
    A = prep["A"]

    t0 = time.time()
    fit = do_fit(A, cfg, a.seed, a.stub_fit)
    timings["fit"] = time.time() - t0
    fit["Z"], fit["Y"] = fit["Z"].detach().float().cpu(), fit["Y"].detach().float().cpu()
    fit.setdefault("config", cfg.to_dict())
    fit.setdefault("seed", a.seed)
    torch.save(fit, out / "fit.pt")
    log(f"fit: rel_err={fit['rel_err']:.4f} iters={fit.get('n_iter')} stop={fit.get('stop_reason')} "
        f"({timings['fit']:.1f}s)")

    t0 = time.time()
    sel = select_features(fit["Y"], prep, out / "judge.json", workers=a.workers)
    timings["select"] = time.time() - t0
    log(f"F_C = {sel['F_C']}  ({len(sel['candidates'])} candidates, {timings['select']:.1f}s)")

    t0 = time.time()
    widest = widest_features(fit["Y"].ne(0).sum(1))
    wdesc = {} if a.no_widest else describe_features(fit["Y"], prep, [f for f, _ in widest], out / "judge.json",
                                                    workers=a.workers, feat_tokens=sel["feat_tokens"])
    timings["describe"] = time.time() - t0

    summ = {"concept": a.concept, "tag": tag, "seed": a.seed, "config": cfg.to_dict(), "matrix": a.matrix,
            "n_concept_sents": a.S, "sub_seed": a.sub_seed, "vstar": len(prep["vstar_ids"]),
            **selection_summary(fit, prep, sel, wdesc, widest),
            "fit_elapsed": float(fit.get("elapsed", timings["fit"])), "n_iter": fit.get("n_iter"),
            "lockin_iter": fit.get("lockin_iter"), "stop_reason": fit.get("stop_reason")}
    write_tables(out, fit["Y"], prep, sel["F_C"], f"{a.concept} / {tag} / seed {a.seed}")

    if a.judge_reps > 0:
        t0 = time.time()
        summ["judge_reps"] = judge_reps(fit["Y"], prep, out, a.judge_reps, [], a.workers)
        timings["judge_reps"] = time.time() - t0
    summ["timings"] = timings
    summ["total_elapsed"] = time.time() - t_all
    save_json(summ_path, summ)
    log(f"done {out}  total {summ['total_elapsed']:.1f}s  (prep {timings['prep']:.1f}, fit {timings['fit']:.1f}, "
        f"select {timings['select']:.1f}, describe {timings['describe']:.1f})")
    return out


# ---------------------------------------------------------------- method 5
def merged_slots(merged: dict, pairs) -> list[int]:
    """Feature slots that hold the merged features: core's report (extra["merged_pairs"]: dicts with "a" = the kept
    slot; skipped pairs are not in it), else the first of each pair."""
    mp = (merged.get("extra") or {}).get("merged_pairs")
    if mp is not None:
        return [int(x["a"]) for x in mp]
    for src in (merged, merged.get("extra", {}) or {}):
        for key in ("merged_slots", "merged_features", "merged"):
            v = src.get(key)
            if v is None:
                continue
            slots = []
            for x in v:
                if isinstance(x, dict):
                    slots.append(int(x.get("slot", x.get("feature", x.get("kept")))))
                elif isinstance(x, (list, tuple)):
                    slots.append(int(x[0]))
                else:
                    slots.append(int(x))
            return slots
    return [int(p[0]) for p in pairs]


def run_merge(a) -> Path:
    from experiments.varwidth.prep import prepare
    from experiments.varwidth.select import describe_features, select_features
    t_all = time.time()
    src = run_dir(a.concept, a.merge_from, a.seed)
    src_summ = json.loads((src / "summary.json").read_text())
    tag = f"{a.merge_from}_merge{a.tag_suffix}"
    out = run_dir(a.concept, tag, a.seed)
    summ_path = out / "summary.json"
    if summ_path.exists() and not a.force:
        log(f"skip {out}: summary.json exists (use --force)")
        return out
    if a.force and out.exists():
        for p in list(out.glob("judge*.json")):
            p.replace(p.with_name(p.name + ".stale"))
    out.mkdir(parents=True, exist_ok=True)
    cfg = VWConfig(**src_summ["config"])
    matrix, S = src_summ.get("matrix", "whitened"), src_summ.get("n_concept_sents", 300)
    prep = prepare(a.concept, matrix, S, src_summ.get("sub_seed", 0))
    A = prep["A"]
    fit = torch.load(src / "fit.pt", weights_only=False)
    Z = fit["Z"].float()
    Zn = torch.nn.functional.normalize(Z, dim=0)

    members = sorted(c["feature"] for c in src_summ["candidates"] if c["is_member"] is True)
    F_C = set(src_summ["F_C"])
    pair_cos = [{"a": i, "b": j, "abs_cos": float((Zn[:, i] @ Zn[:, j]).abs()), "both_in_F_C": i in F_C and j in F_C}
                for i, j in itertools.combinations(members, 2)]
    qualifying = [(p["a"], p["b"]) for p in pair_cos if p["both_in_F_C"] and p["abs_cos"] >= MERGE_COS]
    merge = {"source_tag": a.merge_from, "cos_threshold": MERGE_COS, "max_rel_err_rise": MERGE_MAX_RISE,
             "extra_iters": MERGE_EXTRA_ITERS, "member_pairs": pair_cos, "qualifying_pairs": qualifying,
             "source_F_C": src_summ["F_C"], "source_rel_err": src_summ["rel_err"]}
    keep = {k: src_summ[k] for k in ("n_concept_features", "F_C", "n_distinct_tokens", "tokens_per_feature", "C", "N",
                                     "B", "rel_err", "widths", "candidates", "widest") if k in src_summ}
    summ = {"concept": a.concept, "tag": tag, "seed": a.seed, "config": cfg.to_dict(), "matrix": matrix,
            "n_concept_sents": S, "vstar": len(prep["vstar_ids"]), "n_iter": src_summ.get("n_iter"),
            "lockin_iter": src_summ.get("lockin_iter"), "fit_elapsed": src_summ.get("fit_elapsed")}

    if not qualifying:
        log(f"merge: no pair of F_C features with |cos| >= {MERGE_COS} (member pairs: {len(pair_cos)})")
        merge.update(accepted=False, reason="no qualifying pair")
        shutil.copy2(src / "fit.pt", out / "fit.pt")
        summ.update(keep)
        write_tables(out, fit["Y"], prep, src_summ["F_C"], f"{a.concept} / {tag} / seed {a.seed} (no merge)")
    else:
        from experiments.varwidth import core
        log(f"merge: pairs {qualifying}")
        t0 = time.time()
        merged = core.merge_features(A, fit, qualifying, cfg=cfg, seed=a.seed, extra_iters=MERGE_EXTRA_ITERS)
        merged["Z"], merged["Y"] = merged["Z"].detach().float().cpu(), merged["Y"].detach().float().cpu()
        merged.setdefault("rel_err", rel_err(A, merged["Z"], merged["Y"]))
        merge_elapsed = time.time() - t0
        torch.save(merged, out / "fit_merged.pt")
        sel = select_features(merged["Y"], prep, out / "judge.json", workers=a.workers)
        slots = merged_slots(merged, qualifying)
        rise = float(merged["rel_err"]) - float(src_summ["rel_err"])
        passes = {s: s in sel["F_C"] for s in slots}
        accepted = rise < MERGE_MAX_RISE and all(passes.values())
        merge.update(accepted=accepted, merged_slots=slots, slot_passes=passes, rel_err_merged=float(merged["rel_err"]),
                     rel_err_rise=rise, merged_F_C=sel["F_C"], merged_candidates=sel["candidates"],
                     merged_widths=[int(x) for x in (merged["Y"] != 0).sum(1).tolist()], merge_elapsed=merge_elapsed,
                     reason=("accepted" if accepted else
                             f"rejected: rise {rise:.4f}" + ("" if all(passes.values()) else f", slots pass {passes}")))
        log(f"merge: rel_err {src_summ['rel_err']:.4f} -> {merged['rel_err']:.4f} (rise {rise:+.4f}), "
            f"merged F_C {sel['F_C']}, slots {passes} -> {'ACCEPT' if accepted else 'REJECT'}")
        if accepted:
            torch.save(merged, out / "fit.pt")
            widest = widest_features(merged["Y"].ne(0).sum(1))
            wdesc = describe_features(merged["Y"], prep, [f for f, _ in widest], out / "judge.json",
                                      workers=a.workers, feat_tokens=sel["feat_tokens"])
            summ.update(selection_summary(merged, prep, sel, wdesc, widest))
            write_tables(out, merged["Y"], prep, sel["F_C"], f"{a.concept} / {tag} / seed {a.seed} (merged)")
        else:
            shutil.copy2(src / "fit.pt", out / "fit.pt")
            summ.update(keep)
            write_tables(out, fit["Y"], prep, src_summ["F_C"], f"{a.concept} / {tag} / seed {a.seed} (merge rejected)")
    summ["merge"] = merge
    summ["total_elapsed"] = time.time() - t_all
    save_json(summ_path, summ)
    log(f"done {out}")
    return out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--concept", required=True)
    p.add_argument("--method", default="m2", help="fixed | m1 | m2 | m3 | m4 (ignored with --merge-from)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--matrix", choices=["whitened", "plain"], default="whitened")
    p.add_argument("--sparsity", type=float, default=0.01)
    p.add_argument("--S", type=int, default=300)
    p.add_argument("--sub-seed", type=int, default=0)
    p.add_argument("--init", choices=["randn", "svd"], default="randn")
    p.add_argument("--no-refit", action="store_true")
    p.add_argument("--no-floorcap", action="store_true")
    p.add_argument("--continuation", action="store_true")
    p.add_argument("--judge-reps", type=int, default=0)
    p.add_argument("--merge-from", default=None)
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    p.add_argument("--workers", type=int, default=8, help="concurrent Gemini calls")
    p.add_argument("--no-widest", action="store_true", help="skip the stage-1 descriptions of the 10 widest features")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stub-fit", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    run_one(a)


if __name__ == "__main__":
    main()
