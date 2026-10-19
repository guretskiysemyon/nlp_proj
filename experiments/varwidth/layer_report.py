#!/usr/bin/env python
"""Layer-ladder analysis (results/varwidth/layers/PREREG.md) on the layer_eval caches.

    python -m experiments.varwidth.layer_report pool   [--out PATH]   # unlabeled groups to label, label-blind
    python -m experiments.varwidth.layer_report human  [--n 60]       # stratified blind sample for the user
    python -m experiments.varwidth.layer_report noise                 # TJ re-judge of 10% of L12 original candidates
    python -m experiments.varwidth.layer_report report                # -> results/varwidth/layers/*.md

pool: candidate groups of the judged configs (L0, L1, L12) accepted by TJ, CJ or LO >= 2, or in the top 3 by rho,
    with no label in j3_data/labels.json (group key = concept | sorted first 6 feature words) -> CSV (key, concept,
    n, share, words); judge flags are not exported.
noise: a seed-0 random 10% of the L12 top-20 candidates of the 23 original concepts, judged again with the same
    2-stage TJ into <run>/judge_noise.json; flip rate = share whose accept decision changes.
report: ladder.md (primary endpoint and the outcome / proxy table per scope), layers.md (per-layer diagnostics,
    L2 method table, cross-ablations, judge table) and erasure.md (erasure anchor, proxy validity, fidelity).
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.varwidth.common import RESULTS, ROOT, run_dir, slug
from experiments.varwidth.layer_eval import CONFIGS, JUDGED, RUNGS, SEEDS, edit_set, fc, load

OUT = RESULTS / "layers"
LABELS = ROOT / "experiments" / "varwidth" / "j3_data" / "labels.json"
PROMPT = {"Culture of Greece", "Ancient Rome", "Artificial intelligence", "World War II"}
NEW = ["Judaism", "Karate", "Existentialism", "Prime number", "The Matrix", "Louvre", "American Revolutionary War",
       "Robot", "Sigmund Freud", "Giraffe", "Surfing", "Flush toilet", "Cloud", "Doctor of Philosophy", "Tourism"]
WIDE = ["Cloud", "Doctor of Philosophy", "Tourism"]
ERASE = ["Baseball", "COVID-19 pandemic", "Gambling", "Golf", "Gun", "Halloween", "Harry Potter", "Heroin",
         "Pornography", "Uranium"]
FIDELITY = ["Baseball", "Culture of Greece", "Gambling", "Harry Potter", "COVID-19 pandemic"]
PAPER_T6 = {"Con": 52.1, "Sim": 84.6, "MM": 53.6}   # EMBER, Gemma-2-2B-it, Table 6 (5 seeds, 5 concepts)


def concepts():
    from experiments.varwidth.judge_report import concepts as cs
    return cs()


def scopes():
    cs = concepts()
    return {"non-prompt (primary)": [c for c in cs if c not in PROMPT], "original": [c for c in cs if c not in NEW],
            "new": NEW, "wide": WIDE, "all": cs}


def labels() -> dict:
    lab = json.loads(LABELS.read_text())
    for p in sorted(LABELS.parent.glob("labels_batch_*.json")):
        for k, v in json.loads(p.read_text()).items():
            lab.setdefault(k, v)
    return lab


# ---------------------------------------------------------------- pool / human sample
def pool(out: Path):
    lab = labels()
    rows = {}
    for c in concepts():
        for cfg in JUDGED:
            for s in SEEDS:
                ca = load(c, cfg, s)
                if not ca:
                    continue
                for x in ca["candidates"]:
                    if x["key"] in lab:
                        continue
                    if x["tj"] or x["cj"] or (x["lo"] or 0) >= 2 or x["rank"] < 3:
                        r = rows.setdefault(x["key"], {"key": x["key"], "concept": c, "n": 0, "share": [],
                                                       "words": ", ".join(x["fw20"])})
                        r["n"] += 1
                        r["share"].append(x["share"])
    df = pd.DataFrame(rows.values())
    if df.empty:
        print("nothing to label")
        return df
    df["share"] = df["share"].map(lambda v: round(float(np.mean(v)), 2))
    df = df.sort_values(["concept", "key"])
    df.to_csv(out, index=False)
    print(f"{len(df)} unlabeled groups over {df.concept.nunique()} concepts -> {out}")
    return df


def human(n: int):
    """Stratified (config x Claude label x original/new) blind sample of labeled groups that occur in F_C or top 3."""
    lab = labels()
    seen = {}
    for c in concepts():
        for cfg in JUDGED:
            for s in SEEDS:
                ca = load(c, cfg, s)
                for x in (ca or {}).get("candidates", []):
                    if x["key"] in lab and (x["tj"] or (x["lo"] or 0) >= 5 or x["rank"] < 3):
                        seen.setdefault(x["key"], (cfg, lab[x["key"]], "new" if c in NEW else "original",
                                                   ", ".join(x["fw20"])))
    strata = {}
    for k, v in seen.items():
        strata.setdefault(v[:3], []).append(k)
    rng = random.Random(0)
    keys = []
    order = sorted(strata)
    while len(keys) < n and any(strata[s] for s in order):
        for s in order:
            if strata[s] and len(keys) < n:
                keys.append(strata[s].pop(rng.randrange(len(strata[s]))))
    rng.shuffle(keys)
    df = pd.DataFrame([{"id": i, "concept": k.split(" | ")[0], "words": seen[k][3], "your_label (P/B/N)": ""}
                       for i, k in enumerate(keys)])
    df.to_csv(OUT / "human_labels.csv", index=False)
    (OUT / "human_labels_key.json").write_text(json.dumps({i: k for i, k in enumerate(keys)}))
    print(f"{len(df)} groups -> {OUT / 'human_labels.csv'}")


# ---------------------------------------------------------------- judge noise
def noise():
    from experiments.varwidth.select import CONFIDENCE, TOP_K_TOKENS, JudgeCache, feature_tokens, judge
    from experiments.varwidth.prep import prepare
    import torch
    items = [(c, s, x["feature"]) for c in concepts() if c not in NEW for s in SEEDS
             for x in (load(c, "L12", s) or {}).get("candidates", [])]
    pick = random.Random(0).sample(items, len(items) // 10)
    by = {}
    for c, s, f in pick:
        by.setdefault((c, s), []).append(f)
    for (c, s), fs in by.items():
        d = run_dir(c, CONFIGS["L12"], s)
        prep = prepare(c, "whitened", verbose=False)
        Y = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)["Y"].float()
        ft = feature_tokens(Y, prep)
        judge(c, {f: ft.loc[f, "activating_tokens"][:TOP_K_TOKENS] for f in fs}, JudgeCache(d / "judge_noise.json"))
    print(f"re-judged {len(pick)} candidates in {len(by)} runs")


def noise_rate() -> dict | None:
    from experiments.varwidth.layer_eval import _accepted
    from experiments.varwidth.select import JudgeCache
    n = flips = flips_member = 0
    for c in concepts():
        for s in SEEDS:
            d = run_dir(c, CONFIGS["L12"], s)
            if not (d / "judge_noise.json").exists():
                continue
            a, b = JudgeCache(d / "judge.json").verdicts, JudgeCache(d / "judge_noise.json").verdicts
            for f, v in b.items():
                if f in a:
                    n += 1
                    flips += _accepted(a[f]) != _accepted(v)
                    flips_member += bool(a[f][0]) != bool(v[0])
    return {"n": n, "accept_flip_rate": flips / n, "member_flip_rate": flips_member / n} if n else None


# ---------------------------------------------------------------- text proxies
_TXT = {}


def text(name):
    if name not in _TXT:
        p = OUT / "text" / f"{name}.json"
        d = json.loads(p.read_text())
        _TXT[name] = (d["n_tokens"], {int(k): v for k, v in d["counts"].items()}, d["n_units"])
    return _TXT[name]


def proxies(E: set, concept: str) -> dict:
    n, cnt, _ = text(f"heldout_{slug(concept)}")
    out = {"coverage": sum(cnt.get(t, 0) for t in E) / max(n, 1)}
    for g in ("general_eval", "general_wiki"):
        ng, cg, _ = text(g)
        out[f"collateral_{g.split('_')[1]}"] = 1e4 * sum(cg.get(t, 0) for t in E) / ng   # per 10k tokens
    return out


# ---------------------------------------------------------------- per-run rows
def run_rows(lab: dict) -> pd.DataFrame:
    rows = []
    for c in concepts():
        for rung, (cfg, jd) in RUNGS.items():
            for s in SEEDS:
                ca = load(c, cfg, s)
                if not ca or (jd == "TJ" and any(x["tj"] is None for x in ca["candidates"])):
                    continue
                F = fc(ca, jd)
                ls = [lab.get(x["key"]) for x in F]
                E = edit_set(F)
                rows.append({"concept": c, "rung": rung, "seed": s, "n_fc": len(F), "n_edit": len(E),
                             "unlabeled": sum(v is None for v in ls),
                             "clean": bool(F) and all(v == "P" for v in ls),
                             "clean_len": bool(F) and all(v in ("P", "B") for v in ls),
                             "hit": any(v == "P" for v in ls), "hit_len": any(v in ("P", "B") for v in ls),
                             "contam": any(v == "N" for v in ls),
                             "contam_len": any(v == "N" for v in ls),
                             "empty": not F, **proxies(E, c), "E": E})
    return pd.DataFrame(rows)


def cand_rows(lab: dict, cfgs=JUDGED) -> pd.DataFrame:
    rows = []
    for c in concepts():
        for cfg in cfgs:
            for s in SEEDS:
                ca = load(c, cfg, s)
                if not ca:
                    continue
                for x in ca["candidates"]:
                    rows.append({"concept": c, "config": cfg, "seed": s, **{k: x[k] for k in
                                 ("feature", "rank", "rho", "width", "share", "tj", "cj", "lo", "key")},
                                 "lab": lab.get(x["key"]), "E": set(x["edit_ids"]), "n_tau": ca["n_tau"],
                                 "n_c_vstar": ca["n_c_vstar"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics
def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilcoxon(d):
    from scipy.stats import wilcoxon as w
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(w(d, zero_method="zsplit").pvalue)
    except ValueError:
        return float("nan")


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def mdt(df: pd.DataFrame, floatfmt=3) -> list:
    def f(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    cols = list(df.columns)
    return ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] + \
           ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]


def per_concept(R: pd.DataFrame, col: str) -> pd.DataFrame:
    return R.groupby(["concept", "rung"])[col].mean().unstack("rung")


# ---------------------------------------------------------------- report
RUNG_NAMES = {"L0": "EMBER", "L1": "+CS", "L12": "+CS+GS", "L123": "+CS+GS+TWJ (CINDER)"}
PAIRS = [("L0", "L1"), ("L1", "L12"), ("L12", "L123")]


def ladder(R: pd.DataFrame) -> list:
    L = ["# Layer ladder: EMBER → +CS → +CS+GS → +CS+GS+TWJ (CINDER)", "",
         "Pre-registered in PREREG.md. Unit = concept (mean over its seeds). Clean = F_C non-empty and every feature P "
         "(strict; lenient: P or B). Hit = some F_C feature P. Contaminated = some F_C feature N. Unlabeled features "
         "count as not P. Coverage = share of held-out concept-text tokens that are edited (E = concept-only tokens of "
         "support(F_C)). Collateral = occurrences of edited tokens per 10k tokens of general text (eval = MMLU + "
         "Alpaca; wiki = held-out Wikipedia sample).", ""]
    sc = scopes()
    prim = R[R.concept.isin(sc["non-prompt (primary)"])]
    P = per_concept(prim, "clean").dropna()
    L += ["## Primary endpoint: strict clean-hit rate, 34 non-prompt concepts", ""]
    rows, ps = [], []
    for a, b in PAIRS:
        d = (P[b] - P[a]).values
        rows.append({"comparison": f"{RUNG_NAMES[a]} → {RUNG_NAMES[b]}", "concepts": len(d),
                     "before": P[a].mean(), "after": P[b].mean(), "Δ": d.mean(),
                     "95% CI": "[{:.3f}, {:.3f}]".format(*boot_ci(d)), "Wilcoxon p": wilcoxon(d)})
        ps.append(rows[-1]["Wilcoxon p"])
    for r, p in zip(rows, holm(ps)):
        r["Holm p"] = p
    L += mdt(pd.DataFrame(rows)) + [""]
    L += [f"Overall L0 → L1+2+3: Δ = {(P['L123'] - P['L0']).mean():.3f}, 95% CI "
          "[{:.3f}, {:.3f}], Wilcoxon p = {:.2g}.".format(*boot_ci((P['L123'] - P['L0']).values),
                                                        wilcoxon((P['L123'] - P['L0']).values)), ""]
    cols = ["clean", "clean_len", "hit", "contam", "empty", "n_fc", "n_edit", "coverage", "collateral_eval",
            "collateral_wiki", "unlabeled"]
    for name, cs in sc.items():
        S = R[R.concept.isin(cs)]
        if S.empty:
            continue
        L += [f"## Scope: {name} ({S.concept.nunique()} concepts, {len(S) // 4} runs per rung)", ""]
        rows = []
        for rung in RUNGS:
            G = S[S.rung == rung].groupby("concept")[cols].mean()
            row = {"rung": RUNG_NAMES[rung]}
            for c in cols:
                row[c] = G[c].mean()
            lo, hi = boot_ci(G["clean"].values)
            row["clean 95% CI"] = f"[{lo:.2f}, {hi:.2f}]"
            rows.append(row)
        L += mdt(pd.DataFrame(rows)) + [""]
    return L


def auc_cvn(concept, matrix):
    import torch
    from sklearn.metrics import roc_auc_score
    p = RESULTS / "prep" / f"{slug(concept)}_{matrix}_S300.pt"
    if not p.exists():
        return float("nan")
    d = torch.load(p, weights_only=False)
    A, codes = d["A"].double(), np.asarray(d["codes"])
    C, N = codes == 1, codes == 0
    u = A[:, C].mean(1) - A[:, N].mean(1)
    proj = (A.T @ u).numpy()
    m = C | N
    return float(roc_auc_score(C[m], proj[m]))


def layers(R: pd.DataFrame, K: pd.DataFrame) -> list:
    from scipy.stats import spearmanr
    sc = scopes()
    L = ["# Per-layer diagnostics", "", "Secondary / exploratory (PREREG): effect sizes and CIs, no significance claims.", ""]
    # ---- L1
    L += ["## L1 (CS): concept recoverability, independent of the judge", "",
          "Oracle hit@k = a P-labeled feature among the top-k by ρ (share of runs, mean over concepts). "
          "n_tau = features with ρ ≥ 2 (the paper's candidate count; its scale changes under CW). AUC = C-vs-N "
          "separability of the V* columns along the mean-difference direction (prep matrix only).", ""]
    rows = []
    for name, cs in sc.items():
        for cfg in ("L0", "L1"):
            S = K[(K.config == cfg) & K.concept.isin(cs)]
            if S.empty:
                continue
            g = S.groupby(["concept", "seed"])
            row = {"scope": name, "config": {"L0": "EMBER matrix, RS-1%", "L1": "CS, RS-1%"}[cfg]}
            for k in (1, 3, 5, 20):
                row[f"hit@{k}"] = g.apply(lambda x: bool(((x["rank"] < k) & (x["lab"] == "P")).any())) \
                    .groupby("concept").mean().mean()
            row["n_tau"] = g.n_tau.first().groupby("concept").mean().mean()
            top = S[S["rank"] == 0]
            row["top-ρ share"] = top.groupby("concept").share.mean().mean()
            rows.append(row)
    L += mdt(pd.DataFrame(rows)) + [""]
    aucs = [{"concept": c, "AUC plain": auc_cvn(c, "plain"), "AUC CW": auc_cvn(c, "whitened")} for c in concepts()]
    A = pd.DataFrame(aucs)
    L += [f"C-vs-N AUC over {A.dropna().shape[0]} concepts: plain {A['AUC plain'].mean():.3f}, CW "
          f"{A['AUC CW'].mean():.3f}; CW higher for {(A['AUC CW'] > A['AUC plain']).sum()} / {A.dropna().shape[0]}.", ""]
    # ---- L2
    L += ["## L2 (GS): fragmentation, coverage, breadth adaptivity, stability", "",
          "Fragmentation = distinct P groups among the top-20 per run. Best-P = the P candidate with the largest "
          "held-out coverage. Width adaptivity = Spearman over concepts between the best-P feature's width (mean over "
          "seeds) and breadth = number of concept-only tokens in V*. Stability = mean pairwise Jaccard across seeds "
          "of the edited set E of the rung's F_C.", ""]
    rows = []
    for cfg in ("L1", "L12"):
        S = K[(K.config == cfg)].copy()
        S["cov"] = [proxies(e, c)["coverage"] for e, c in zip(S.E, S.concept)]
        P = S[S.lab == "P"]
        frag = S.groupby(["concept", "seed"]).apply(lambda x: x[x.lab == "P"].key.nunique()).groupby("concept").mean()
        best = P.sort_values("cov").groupby(["concept", "seed"]).tail(1)
        bw = best.groupby("concept").agg(width=("width", "mean"), cov=("cov", "mean"), breadth=("n_c_vstar", "first"))
        rho = spearmanr(bw.width, bw.breadth).statistic if len(bw) > 2 and bw.width.nunique() > 1 else float("nan")
        rows.append({"config": {"L1": "CS, RS-1%", "L12": "CS, GS-norm"}[cfg], "fragmentation": frag.mean(),
                     "best-P width": bw.width.mean(), "best-P coverage": bw["cov"].mean(),
                     "Spearman(width, breadth)": rho, "concepts with a P feature": len(bw)})
    L += mdt(pd.DataFrame(rows)) + [""]
    rows = []
    for rung in ("L1", "L12", "L123"):
        jac = []
        for c in concepts():
            Es = [e for e in R[(R.rung == rung) & (R.concept == c)].E if e]
            if len(Es) >= 2:
                jac.append(np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(Es, 2)]))
        rows.append({"rung": RUNG_NAMES[rung], "seed Jaccard of E": float(np.mean(jac)), "concepts": len(jac)})
    L += mdt(pd.DataFrame(rows)) + [""]
    L += ["### Wide topics (Cloud, Doctor of Philosophy, Tourism): best-P width per config", ""]
    for cfg in ("L1", "L12"):
        S = K[(K.config == cfg) & (K.lab == "P") & K.concept.isin(WIDE)]
        L += [f"- {cfg}: " + ", ".join(f"{c} {g.width.mean():.0f} tokens ({g.seed.nunique()} seeds with P)"
                                       for c, g in S.groupby("concept"))]
    L += [""]
    # ---- L2 method table (judge-free)
    L += ["## L2 width-rule table at cap 3% (judge-free; 38 concepts)", "",
          "Concept feature = the top-20 candidate with the most EJ matches (ties: higher ρ), so no Gemini or labels "
          "are needed. EJ ≥ 5 = that feature passes EJ@5. Width and neutral share of that feature; seed Jaccard of its "
          "edited tokens; coverage on held-out concept text; collateral per 10k general tokens.", ""]
    rows = []
    for cfg, name in (("L1", "CS, RS-1%"), ("fixed2", "CS, RS-2%"), ("m1", "CS, GS-gain"), ("m3", "CS, GS-entropy"),
                      ("L12", "CS, GS-norm"), ("L0", "no CS, RS-1% (EMBER)"), ("rnP", "no CS, GS-norm")):
        per = []
        for c in concepts():
            sel, breadth = [], np.nan
            for s in SEEDS:
                ca = load(c, cfg, s)
                if not ca:
                    continue
                breadth = ca["n_c_vstar"]
                sel.append(max(ca["candidates"], key=lambda x: ((x["lo"] or 0), x["rho"])))
            if not sel:
                continue
            Es = [set(x["edit_ids"]) for x in sel]
            jac = np.mean([len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(Es, 2)]) \
                if len(Es) > 1 else np.nan
            pr = [proxies(set(x["edit_ids"]), c) for x in sel]
            per.append({"lo5": np.mean([(x["lo"] or 0) >= 5 for x in sel]), "width": np.mean([x["width"] for x in sel]),
                        "share": np.mean([x["share"] for x in sel]), "jac": jac,
                        "cov": np.mean([p["coverage"] for p in pr]), "col": np.mean([p["collateral_wiki"] for p in pr]),
                        "breadth": breadth})
        if not per:
            continue
        P = pd.DataFrame(per)
        rho = spearmanr(P.width, P.breadth).statistic if P.width.nunique() > 1 else float("nan")
        rows.append({"config": name, "concepts": len(P), "EJ ≥ 5": P.lo5.mean(), "width": P.width.mean(),
                     "concept share": P.share.mean(), "seed Jaccard": P.jac.mean(), "coverage": P["cov"].mean(),
                     "collateral (wiki)": P.col.mean(), "Spearman(width, breadth)": rho})
    L += mdt(pd.DataFrame(rows)) + [""]
    # ---- cross-ablations
    L += ["## Cross-ablations", "", "EJ@5 / TWJ applied to the fixed-width rungs (L3 without L2); columns TJ, LO = EJ@5, HJ = TWJ, strict clean-hit rate "
          "on the non-prompt concepts (mean over concepts):", ""]
    rows = []
    np_cs = sc["non-prompt (primary)"]
    lab = labels()
    for cfg in ("L0", "L1", "L12"):
        row = {"config": {"L0": "no CS, RS-1%", "L1": "CS, RS-1%", "L12": "CS, GS-norm"}[cfg]}
        for jd in ("TJ", "LO", "HJ"):  # TJ, EJ@5, TWJ
            vals = []
            for c in np_cs:
                v = []
                for s in SEEDS:
                    ca = load(c, cfg, s)
                    if not ca or any(x["tj"] is None for x in ca["candidates"]):
                        continue
                    F = fc(ca, jd)
                    v.append(bool(F) and all(lab.get(x["key"]) == "P" for x in F))
                if v:
                    vals.append(np.mean(v))
            row[jd] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)
    L += mdt(pd.DataFrame(rows)) + [""]
    # ---- judges
    L += ["## L3 judges on the CS + GS-norm runs (top-20 candidates)", "",
          "Micro precision / recall over candidates (strict: P positive; lenient: P or B), runs hit = runs with an "
          "accepted P feature, macro recall = mean over concepts. Recall is over the labeled pool (TJ, CJ, LO ≥ 2 or "
          "top 3 by ρ), so it is an upper bound.", ""]
    nr = noise_rate()
    if nr:
        L += [f"TJ noise (re-judged {nr['n']} candidates): accept-decision flip rate {nr['accept_flip_rate']:.3f}, "
              f"is_member flip rate {nr['member_flip_rate']:.3f}.", ""]
    S0 = K[K.config == "L12"]
    for name in ("non-prompt (primary)", "original", "new"):
        S = S0[S0.concept.isin(sc[name]) & ~S0.concept.isin(PROMPT)]
        L += [f"### {name} ({S.concept.nunique()} concepts)", ""]
        rows = []
        lo = S["lo"].fillna(0)
        cj, tj = S.cj.fillna(False).astype(bool), S.tj.fillna(False).astype(bool)
        cjs = cj & (S.share >= 0.5)
        rules = {"TJ (EMBER)": tj, "WJ": cj, "WJ+share": cjs, "EJ@3": lo >= 3, "EJ@4": lo >= 4, "EJ@5": lo >= 5,
                 "TWJ = TJ ∨ EJ@5": tj | (lo >= 5), "TJ ∨ EJ@4": tj | (lo >= 4), "TJ ∨ WJ+share": tj | cjs,
                 "TWJ-R = WJ+share ∨ EJ@4": cjs | (lo >= 4)}
        for rn, acc in rules.items():
            row = {"rule": rn, "accepts": int(acc.sum())}
            for lenient in (False, True):
                y = S.lab.eq("P") | (lenient & S.lab.eq("B"))
                tp, fp, fn = int((acc & y).sum()), int((acc & ~y).sum()), int((~acc & y).sum())
                t = "len" if lenient else "strict"
                row[f"P {t}"] = tp / max(tp + fp, 1)
                row[f"R {t}"] = tp / max(tp + fn, 1)
                if not lenient:
                    row["runs hit"] = int(S.assign(h=acc & y).groupby(["concept", "seed"]).h.any().sum())
                    row["macro R"] = S.assign(h=acc & y, y=y).groupby("concept").apply(
                        lambda g: g.h.sum() / g.y.sum() if g.y.sum() else np.nan).mean()
            rows.append(row)
        L += [f"Runs: {S.groupby(['concept', 'seed']).ngroups}; runs with a P candidate: "
              f"{int(S.assign(y=S.lab.eq('P')).groupby(['concept', 'seed']).y.any().sum())}.", ""]
        L += mdt(pd.DataFrame(rows)) + [""]
    return L


def erasure() -> list:
    from scipy.stats import spearmanr
    L = ["# Erasure anchor (EMBER, gemma-2-2b-it, MC protocol, seed 42)", "",
         "Grid on the validation split gated by MMLU-frac ≥ 0.9, δ = argmax HM(efficacy, specificity); test split at δ: "
         "Con = concept MC accuracy, Sim = similar-domain MC accuracy, MM = MMLU (1000), coherence from 50 AlpacaEval "
         "prompts (Gemini judge). efficacy = 1 − qa_frac, specificity = HM(mmlu_frac, simdom_frac), "
         "H = HM(efficacy, specificity, coherence). Alpaca is not in the grid (Gemini budget): a deviation from the "
         "paper applied identically to every rung.", ""]
    rows = []
    for p in sorted((OUT / "erasure").glob("*/*_s42/result.json")):
        r = json.loads(p.read_text())
        rows.append(r)
    if not rows:
        return L + ["(no erasure results yet)"]
    D = pd.DataFrame(rows)
    D["Con"], D["Sim"], D["MM"] = 100 * D.qa_acc, 100 * D.simdom_acc, 100 * D.mmlu_acc
    base = json.loads((OUT / "erasure" / "baselines.json").read_text())
    S = D[D.concept.isin(ERASE)]
    cols = ["Con", "Sim", "MM", "efficacy", "specificity", "coherence", "H", "H2", "delta", "n_edit"]
    cols = [c for c in cols if c in S]
    T = S.groupby("rung")[cols].mean().reindex([r for r in RUNGS if r in set(S.rung)])
    T.insert(0, "concepts", S.groupby("rung").concept.nunique())
    T.index = [RUNG_NAMES[r] for r in T.index]
    bq = np.mean([base[c]["qa_test"] for c in ERASE if c in base]) * 100
    bs = np.mean([base[c]["simdom_test"] for c in ERASE if c in base]) * 100
    L += [f"Unedited baseline over the same concepts: Con {bq:.1f}, Sim {bs:.1f}, MM "
          f"{100 * base['_global']['mmlu_test']:.1f}.", ""]
    L += mdt(T.reset_index().rename(columns={"index": "rung"}), 2) + [""]
    W = S.pivot_table(index="concept", columns="rung", values="H" if "H" in S else "H2")
    L += ["Paired H differences over concepts:", ""]
    for a, b in PAIRS + [("L0", "L123")]:
        if a in W and b in W:
            d = (W[b] - W[a]).dropna().values
            L += [f"- {RUNG_NAMES[a]} → {RUNG_NAMES[b]}: Δ H = {d.mean():+.3f}, 95% CI [{boot_ci(d)[0]:+.3f}, "
                  f"{boot_ci(d)[1]:+.3f}], {int((d > 0).sum())} up / {int((d < 0).sum())} down of {len(d)}"]
    L += [""]
    # proxy validity
    lab = labels()
    pr = []
    for r in S.itertuples():
        cfg, jd = RUNGS[r.rung]
        ca = load(r.concept, cfg, 42)
        F = fc(ca, jd)
        E = edit_set(F)
        pr.append({"clean": float(bool(F) and all(lab.get(x["key"]) == "P" for x in F)), **proxies(E, r.concept),
                   "efficacy": r.efficacy, "specificity": r.specificity, "H": getattr(r, "H", np.nan)})
    Pdf = pd.DataFrame(pr)
    L += [f"## Proxy validity (Spearman over {len(Pdf)} concept × rung points)", ""]
    rows = []
    for x in ("clean", "coverage", "collateral_eval", "collateral_wiki"):
        row = {"proxy": x}
        for y in ("efficacy", "specificity", "H"):
            ok = Pdf[[x, y]].dropna()
            row[y] = spearmanr(ok[x], ok[y]).statistic if len(ok) > 3 and ok[x].nunique() > 1 else float("nan")
        rows.append(row)
    L += mdt(pd.DataFrame(rows)) + [""]
    F0 = D[(D.rung == "L0") & D.concept.isin(FIDELITY)]
    if len(F0):
        L += ["## Fidelity of L0 (our EMBER reimplementation vs the paper)", "",
              f"Paper Table 6 (EMBER, Gemma, 5 concepts × 5 seeds): Con {PAPER_T6['Con']}, Sim {PAPER_T6['Sim']}, "
              f"MM {PAPER_T6['MM']}. Ours (L0, seed 42, {len(F0)} of the same 5 concepts): Con {F0.Con.mean():.1f}, "
              f"Sim {F0.Sim.mean():.1f}, MM {F0.MM.mean():.1f}.", ""]
        L += mdt(F0[["concept", "Con", "Sim", "MM", "delta", "n_edit"]], 1) + [""]
    return L


def report():
    lab = labels()
    R = run_rows(lab)
    K = cand_rows(lab)
    (OUT / "ladder.md").write_text("\n".join(ladder(R)))
    (OUT / "layers.md").write_text("\n".join(layers(R, K)))
    (OUT / "erasure.md").write_text("\n".join(erasure()))
    R.drop(columns="E").to_csv(OUT / "runs.csv", index=False)
    print("\n".join((OUT / "ladder.md").read_text().splitlines()[:40]))
    print(f"-> {OUT}/ladder.md, layers.md, erasure.md, runs.csv")


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pool", "human", "noise", "report"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    if a.what == "pool":
        pool(Path(a.out) if a.out else OUT / "pool.csv")
    elif a.what == "human":
        human(a.n)
    elif a.what == "noise":
        noise()
    else:
        report()


if __name__ == "__main__":
    main()
