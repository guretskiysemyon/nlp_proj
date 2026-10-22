"""Exact selection cost of A1 / A2 / A3 (/ A3z) from the saved fits, selections, judge caches, Gemini log and run logs.

    python -m experiments.paper_analysis.ml_geometry.cost

Per (concept, fit seed, approach):
  factorizations   A1: one per language fit (8); A2, A3: 1; A3z: 0 (one ridge solve with A3's Y fixed)
  rows, n_iter     fit_summary.json (A1: sum over languages; A2/A3: rows of the sample fit, + one dense solve on n_full)
  fit_secs         fit_summary.json "secs". A1 writes a cumulative clock over its 8 languages (t0 is set before the
                   language loop in experiments/ml/fit.run), so the A1 total is the sum of the last value of every
                   monotone run of the per-language values (a resumed call restarts the clock). Wall time includes
                   tokenizing the sentence sets, the CS transform (A3) and the extrapolation solve (A2/A3).
                   Seed 42: local GPU (c-001, run_local.sh --device cuda); seeds 1, 2: CPU (seeds.py default, SLURM
                   CPU jobs for A1).
  judged           candidates sent to the judge (selection.json; <= 10 per fit), 2 Gemini calls each
  lexicon          A3 only: one Gemini call per concept (results/ml/lexicons.json), shared by all seeds
Gemini tokens and NIS: results/ml/gemini_usage.jsonl has no purpose tag for judge/lexicon calls, so per-call means are
taken from its untagged records (judge-like: output < 150 tokens and prompt < 1000; lexicon-like: output >= 150 and
prompt < 1000), and the predicted call counts are reconciled with the log, per time window.
Evaluation: "[worker] done <concept> <approach> in N min" lines of the worker logs (GPU minutes per unit).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.ml_geometry.common import (LANGS, MLRES, OUT, SEEDS, concepts, embeddings, fit_dir,
                                                           is_new, save_json, slug)


def read_usage():
    raw = (MLRES / "gemini_usage.jsonl").read_bytes()
    n_nul = raw.count(b"\0")
    recs, bad = [], []
    for line in raw.replace(b"\0", b"\n").decode("utf8", "replace").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except ValueError:
            bad.append(line)
    df = pd.DataFrame(recs)
    df["hour"] = df["ts"].str[:13]

    def kind(r):
        if r["tag"].startswith("ml_qa"):
            return "qa_generation"
        if r["p"] >= 1000:
            return "qa_generation_untagged"
        return "lexicon_like" if r["o"] >= 150 else "judge_like"
    df["kind"] = df.apply(kind, axis=1)
    return df, n_nul, bad


def a1_total_secs(vals):
    tot, last = 0.0, None
    for v in vals:
        if last is not None and v < last:
            tot += last
        last = v
    return tot + (last or 0.0)


def selection_rows():
    rows = []
    for c in concepts():
        for s in SEEDS:
            for a in ("A1", "A2", "A3"):
                if a == "A1":
                    fs = [json.loads((fit_dir(c, a, s, L) / "fit_summary.json").read_text()) for L in LANGS]
                    sels = [json.loads((fit_dir(c, a, s, L) / "selection.json").read_text()) for L in LANGS]
                    judged = sum(len(x["candidates"]) for x in sels)
                    jfiles = sum(len(json.loads((fit_dir(c, a, s, L) / "judge.json").read_text()))
                                 if (fit_dir(c, a, s, L) / "judge.json").exists() else 0 for L in LANGS)
                    rows.append({"concept": c, "seed": s, "approach": a, "factorizations": 8,
                                 "rows": sum(f["n"] for f in fs), "n_iter": sum(f["n_iter"] for f in fs),
                                 "row_iters": sum(f["n"] * f["n_iter"] for f in fs),
                                 "fit_secs": a1_total_secs([f["secs"] for f in fs]), "judged": judged,
                                 "judge_cache_entries": jfiles, "lexicon": 0,
                                 "n_FC": sum(len(x["F_C"]) for x in sels),
                                 "langs_with_FC": sum(bool(x["F_C"]) for x in sels)})
                else:
                    f = json.loads((fit_dir(c, a, s) / "fit_summary.json").read_text())
                    sel = json.loads((fit_dir(c, a, s) / "selection.json").read_text())
                    jp = fit_dir(c, a, s) / "judge.json"
                    rows.append({"concept": c, "seed": s, "approach": a, "factorizations": 1, "rows": f["n_sample"],
                                 "rows_extrapolated": f["n_full"], "n_iter": f["n_iter"], "fit_secs": f["secs"],
                                 "row_iters": f["n_sample"] * f["n_iter"],
                                 "judged": len(sel["candidates"]),
                                 "judge_cache_entries": len(json.loads(jp.read_text())) if jp.exists() else 0,
                                 "lexicon": int(a == "A3" and s == 42), "n_FC": len(sel["F_C"]),
                                 "langs_with_FC": None})
    df = pd.DataFrame(rows)
    df["judge_calls"] = 2 * df["judged"]
    df["new"] = df["concept"].map(is_new)
    return df


def eval_times():
    pat = re.compile(r"\[worker\] done (.+) (A1U|A3zU|A1|A2|A3z|A3) in (\d+) min")
    out = []
    for p in sorted((MLRES / "logs").glob("*")):
        if p.suffix not in (".log", ".out"):
            continue
        for line in p.read_text(errors="replace").splitlines():
            m = pat.search(line)
            if m:
                out.append({"concept": m.group(1), "approach": m.group(2), "minutes": int(m.group(3)), "log": p.name})
    return pd.DataFrame(out)


def time_a3z(concepts_=("Baseball", "Harry Potter", "Gun")):
    """Wall time of the A3z ridge re-solve (experiments/ml/a3z.py), re-run here on CPU without writing anything."""
    torch.set_num_threads(4)
    E = embeddings()
    out = []
    for c in concepts_:
        fit = torch.load(fit_dir(c, "A3") / "fit.pt", weights_only=False, map_location="cpu")
        t0 = time.time()
        Y = fit["Y"].double()
        A = E[fit["vstar_ids"]].double().T
        Zo = torch.linalg.solve(Y @ Y.T + 1e-4 * torch.eye(Y.shape[0], dtype=torch.float64), Y @ A.T).T
        out.append({"concept": c, "secs_cpu4": time.time() - t0, "n": A.shape[1]})
        del Zo
    return out


def main():
    use, n_nul, bad = read_usage()
    sel = selection_rows()
    ev = eval_times()
    # ---- Gemini per-call means (untagged selection-type calls)
    jl = use[use.kind == "judge_like"]
    ll = use[use.kind == "lexicon_like"]
    per_call = {"judge": {"n": len(jl), "p": jl.p.mean(), "o": jl.o.mean(), "nis": jl.nis.mean()},
                "lexicon": {"n": len(ll), "p": ll.p.mean(), "o": ll.o.mean(), "nis": ll.nis.mean()}}
    # ---- reconcile predicted selection calls with the log, per window
    windows = {"seed 42, original 18 (Oct 18 20h)": (["2026-10-18T20"], (sel.seed == 42) & ~sel.new),
               "seed 42, new 20 (Oct 20 16h)": (["2026-10-20T16"], (sel.seed == 42) & sel.new),
               "seeds 1 + 2, 38 concepts (Oct 21 10-12h)": (["2026-10-21T10", "2026-10-21T11", "2026-10-21T12"],
                                                            sel.seed != 42)}
    recon = []
    for name, (hours, m) in windows.items():
        u = use[use.hour.isin(hours)]
        pred_j = int(sel[m].judge_calls.sum())
        pred_l = int(sel[m].lexicon.sum())
        recon.append({"window": name, "log_records": len(u), "log_judge_like": int((u.kind == "judge_like").sum()),
                      "log_lexicon_like": int((u.kind == "lexicon_like").sum()), "predicted_judge_calls": pred_j,
                      "predicted_lexicon_calls": pred_l, "log_nis": float(u.nis.sum())})
    other = use[~use.hour.isin(sum((w[0] for w in windows.values()), []))]
    hours_other = other.groupby(["hour", "kind"]).agg(n=("nis", "size"), nis=("nis", "sum")).reset_index()
    # ---- per-approach totals (judge + lexicon), seed 42 and per seed
    sel["gemini_calls"] = sel.judge_calls + sel.lexicon
    sel["nis"] = sel.judge_calls * per_call["judge"]["nis"] + sel.lexicon * per_call["lexicon"]["nis"]
    sel["tok_in"] = sel.judge_calls * per_call["judge"]["p"] + sel.lexicon * per_call["lexicon"]["p"]
    sel["tok_out"] = sel.judge_calls * per_call["judge"]["o"] + sel.lexicon * per_call["lexicon"]["o"]
    sel.to_csv(OUT / "cost_per_unit.csv", index=False)
    agg = sel.groupby(["seed", "approach"]).agg(
        concepts=("concept", "size"), factorizations=("factorizations", "sum"), rows=("rows", "sum"),
        n_iter=("n_iter", "sum"), row_iters=("row_iters", "sum"), fit_secs=("fit_secs", "sum"), judged=("judged", "sum"),
        judge_calls=("judge_calls", "sum"), lexicon=("lexicon", "sum"), gemini_calls=("gemini_calls", "sum"),
        tok_in=("tok_in", "sum"), tok_out=("tok_out", "sum"), nis=("nis", "sum"),
        units_with_edit=("n_FC", lambda x: int((x > 0).sum()))).reset_index()
    agg.to_csv(OUT / "cost_totals.csv", index=False)
    # ---- evaluation GPU minutes
    ev.to_csv(OUT / "eval_minutes.csv", index=False)
    ev_last = ev.groupby(["concept", "approach"]).minutes.last().reset_index()
    ev_sum = ev_last.groupby("approach").minutes.agg(["count", "median", "mean", "sum", "min", "max"]).reset_index()
    out = {"usage_log": {"records": len(use), "unparseable_fragments": len(bad), "nul_bytes": n_nul,
                         "total_nis": float(use.nis.sum()), "by_kind": use.groupby("kind").nis.agg(["size", "sum"])
                         .reset_index().to_dict("records")},
           "per_call": per_call, "reconcile": recon, "other_hours": hours_other.to_dict("records"),
           "totals": agg.to_dict("records"), "eval_minutes": ev_sum.to_dict("records"),
           "eval_duplicates": int(len(ev) - len(ev_last)), "a3z_solve": time_a3z()}
    save_json(OUT / "cost.json", out)
    pd.set_option("display.width", 250)
    print(json.dumps(out["usage_log"], indent=1, default=str))
    print(json.dumps(per_call, indent=1, default=float))
    print(pd.DataFrame(recon).to_string())
    print(hours_other.to_string())
    print(agg.to_string())
    print(ev_sum.to_string())
    print(out["a3z_solve"])


if __name__ == "__main__":
    main()
