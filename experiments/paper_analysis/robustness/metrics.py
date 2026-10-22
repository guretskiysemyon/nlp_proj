"""WP4 per-run and per-concept metrics from features.parquet (+ blind labels for part A).

    .venv/bin/python -m experiments.paper_analysis.robustness.metrics A    # -> runs_A.csv, concepts_A.csv, stats_A.json
    .venv/bin/python -m experiments.paper_analysis.robustness.metrics B    # -> runs_B.csv, curves_B.csv, stats_B.json

Part A (labels: P specific, B broader / parent topic, N not the concept; top-10 features by rho of each run):
    has_P@k      a P feature among the top-k by rho (k = 1, 3, 10);  P_rank = rho rank of the best P feature (NaN: none)
    nP, nB, nN   counts among the top 10;  P_hits / P_cshare = lexicon hits (of 20 source words) and concept-only
                 share of the 20 strongest tokens of the best P feature
    lex_rank5    judge-free: rho rank of the best-ranked feature with >= 5 lexicon hits among its 20 source words
                 (EJ@5 of the project), over all 100 features; lex_rank3 likewise with >= 3
    stability    per (concept, S, cond): mean pairwise Jaccard of the 20-token sets of the best P features over seeds
Part B: labels as in part A (best-ranked P feature among the top 10 by rho; P@k, P_rank, its intruder share and
    lexicon precision). Judge-free: the concept feature of a run = the feature with the most hits of the SPECIFIC
    lexicon (specific_lexicons; ties: better rho rank), found_spec3 = >= 3 such hits. Reported: its rho rank, rho, hits / 20 (lexicon precision), concept-only share and intruder share
    (top-20 tokens that occur only in the inserted adjacent sentences), the intruder share of the top-rho feature,
    the best rank of an "adjacent" feature (intruder share >= 0.5), and the Jaccard of the concept feature's top-20
    token set with the same seed's f = 0 concept feature.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from experiments.paper_analysis.robustness.analyze import LABELS, PART_A_CONDS, TOP_RHO, load_labels  # noqa: F401
from experiments.paper_analysis.robustness.concepts import ABSTRACT_SHORT, ADJACENT, OUT, slug


def jacc(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else np.nan


def mean_pairwise_jaccard(sets) -> float:
    sets = [s for s in sets if s is not None]
    if len(sets) < 2:
        return np.nan
    return float(np.mean([jacc(a, b) for a, b in itertools.combinations(sets, 2)]))


def S_of(variant: str) -> int:
    return 190 if "__S190" in variant else 300


# ---------------------------------------------------------------- part A
def part_A():
    df = pd.read_parquet(OUT / "features.parquet")
    A = df[df.group.isin(["abstract", "entity"]) & df.cond.isin(PART_A_CONDS)].copy()
    key = json.loads((LABELS / "hidden_key.json").read_text())
    lab = load_labels()
    A["item"] = [key.get(f"{r.variant}|{r.cond}|{r.seed}|{r.feature}") for r in A.itertuples()]
    A["label"] = A["item"].map(lambda i: lab.get(i, (None, ""))[0] if i else None)
    A["note"] = A["item"].map(lambda i: lab.get(i, (None, ""))[1] if i else None)
    top = A[A["rank"] <= TOP_RHO]
    missing = top.label.isna().sum()
    rows = []
    for (v, c, cond, s), g in A.groupby(["variant", "concept", "cond", "seed"]):
        t = g[g["rank"] <= TOP_RHO].sort_values("rank")
        P = t[t.label == "P"]
        best = P.iloc[0] if len(P) else None
        lex5 = g[g.lex_hits >= 5]["rank"].min()
        lex3 = g[g.lex_hits >= 3]["rank"].min()
        rows.append({"variant": v, "concept": c, "S": S_of(v), "group": g.group.iloc[0], "cond": cond, "seed": s,
                     "has_P@1": int((P["rank"] <= 1).any()), "has_P@3": int((P["rank"] <= 3).any()),
                     "has_P@10": int(len(P) > 0), "P_rank": float(best["rank"]) if best is not None else np.nan,
                     "P_rho": float(best["rho"]) if best is not None else np.nan,
                     "nP": int((t.label == "P").sum()), "nB": int((t.label == "B").sum()),
                     "nN": int((t.label == "N").sum()),
                     "P_hits": float(best["lex_hits"]) if best is not None else np.nan,
                     "P_cshare": float(best["c_share"]) if best is not None else np.nan,
                     "P_tids": list(best["top_tids"]) if best is not None else None,
                     "top1_label": t.label.iloc[0], "top1_note": t.note.iloc[0],
                     "lex_rank5": float(lex5) if pd.notna(lex5) else np.nan,
                     "lex_rank3": float(lex3) if pd.notna(lex3) else np.nan,
                     "max_lex_hits": int(g.lex_hits.max()),
                     "B_notes": [n for n in t[t.label == "B"].note], "N_notes": [n for n in t[t.label == "N"].note]})
    R = pd.DataFrame(rows)
    conc = []
    for (c, S, grp, cond), g in R.groupby(["concept", "S", "group", "cond"]):
        conc.append({"concept": c, "S": S, "group": grp, "cond": cond, "n_seeds": len(g),
                     "P@1": g["has_P@1"].mean(), "P@3": g["has_P@3"].mean(), "P@10": g["has_P@10"].mean(),
                     "P_rank_med": g.P_rank.median(), "P_rank_mean_cap11": g.P_rank.fillna(11).mean(),
                     "nP": g.nP.mean(), "nB": g.nB.mean(), "nN": g.nN.mean(), "P_hits": g.P_hits.mean(),
                     "P_cshare": g.P_cshare.mean(), "stability": mean_pairwise_jaccard(g.P_tids.tolist()),
                     "lex5@10": (g.lex_rank5 <= 10).mean(), "lex_rank5_med": g.lex_rank5.median(),
                     "lex3@10": (g.lex_rank3 <= 10).mean()})
    C = pd.DataFrame(conc)
    R.drop(columns=["P_tids"]).to_csv(OUT / "runs_A.csv", index=False)
    C.to_csv(OUT / "concepts_A.csv", index=False)
    st = stats_A(R, C)
    st["unlabelled_top10_slots"] = int(missing)
    (OUT / "stats_A.json").write_text(json.dumps(st, indent=1, default=float))
    print(json.dumps(st, indent=1, default=float))
    return R, C


def boot_ci(x, n=10000, seed=0):
    x = np.asarray([v for v in x if pd.notna(v)], float)
    if len(x) == 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def stats_A(R, C) -> dict:
    from scipy.stats import mannwhitneyu, wilcoxon
    out = {}
    for S in (300, 190):
        cs = C[C.S == S]
        d = {}
        for grp in ("abstract", "entity"):
            for cond in PART_A_CONDS:
                g = cs[(cs.group == grp) & (cs.cond == cond)]
                d[f"{grp}/{cond}"] = {"n_concepts": len(g), "P@1": g["P@1"].mean(), "P@3": g["P@3"].mean(),
                                      "P@10": g["P@10"].mean(), "P@10_ci": boot_ci(g["P@10"]),
                                      "P_rank_mean_cap11": g.P_rank_mean_cap11.mean(),
                                      "lex5@10": g["lex5@10"].mean(), "stability": g.stability.mean(),
                                      "nP": g.nP.mean(), "nB": g.nB.mean(), "nN": g.nN.mean()}
        # abstract vs entity under EMBER (concept = unit), and paired CS - EMBER within abstract concepts
        for metric in ("P@10", "P@3", "P_rank_mean_cap11"):
            a = cs[(cs.group == "abstract") & (cs.cond == "EMBER")][metric]
            e = cs[(cs.group == "entity") & (cs.cond == "EMBER")][metric]
            if len(a) and len(e):
                p = mannwhitneyu(a, e, alternative="two-sided").pvalue
                d[f"MWU EMBER abstract vs entity {metric}"] = {"abstract_mean": a.mean(), "entity_mean": e.mean(),
                                                               "p": float(p)}
            for grp in ("abstract", "entity"):
                for cond in ("CS", "CSGS"):
                    x = cs[(cs.group == grp) & (cs.cond == cond)].set_index("concept")[metric]
                    y = cs[(cs.group == grp) & (cs.cond == "EMBER")].set_index("concept")[metric]
                    diff = (x - y.reindex(x.index)).dropna()
                    try:
                        p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
                    except ValueError:
                        p = np.nan
                    d[f"paired {cond}-EMBER {grp} {metric}"] = {"n": len(diff), "mean_diff": diff.mean(),
                                                                "ci": boot_ci(diff), "wilcoxon_p": p,
                                                                "n_pos": int((diff > 0).sum()),
                                                                "n_neg": int((diff < 0).sum())}
        out[f"S{S}"] = d
    return out


# ---------------------------------------------------------------- part B
def f_of(variant: str) -> float:
    for k in ("__f", "__u"):
        if k in variant:
            return int(variant.split(k)[1]) / 100
    return 0.0


def kind_of(variant: str) -> str:
    return "unrelated" if "__u" in variant else "adjacent"


def specific_lexicons() -> dict:
    """Concept lexicon minus every word (normalized) of the eligible adjacent sentences: the words that set the concept
    apart from its neighbours (HP: names; Baseball: 7 words; Ancient Rome: 14). Used to find the concept feature."""
    from experiments.paper_analysis.robustness.concepts import lexicon
    from experiments.paper_analysis.robustness.variants import adjacent_pool
    from experiments.varwidth import overlap_judge as oj
    out = {}
    for c in ADJACENT:
        pool, _ = adjacent_pool(c, "f")
        words = {oj.normalize(w) for s in pool for w in s.split()}
        out[c] = [w for w in lexicon(c) if w not in words]
    return out


def part_B():
    df = pd.read_parquet(OUT / "features.parquet")
    B = df[df.concept.isin(list(ADJACENT)) & ~df.variant.str.contains("__S")].copy()
    B["f"] = B.variant.map(f_of)
    B["kind"] = B.variant.map(kind_of)
    spec = specific_lexicons()
    B["spec_hits"] = [sum(w in set(fw) for w in spec[c]) for c, fw in zip(B.concept, B.fw20)]
    keyB = json.loads((LABELS / "hidden_key_B.json").read_text()) if (LABELS / "hidden_key_B.json").exists() else {}
    keyA = json.loads((LABELS / "hidden_key.json").read_text())
    lab = load_labels()
    slot = B.variant + "|" + B.cond + "|" + B.seed.astype(str) + "|" + B.feature.astype(str)
    B["label"] = [lab.get(keyB.get(k) or keyA.get(k), (None, ""))[0] for k in slot]
    # f = 0 rows serve both kinds
    B0 = B[B.f == 0].copy()
    B0["kind"] = "unrelated"
    B = pd.concat([B, B0], ignore_index=True)
    rows = []
    for (c, kind, f, cond, s), g in B.groupby(["concept", "kind", "f", "cond", "seed"]):
        t = g[g["rank"] <= TOP_RHO].sort_values("rank")
        P = t[t.label == "P"]
        bp = P.iloc[0] if len(P) else None
        g = g.sort_values(["spec_hits", "rank"], ascending=[False, True])
        cf = g.iloc[0]
        t1 = g[g["rank"] == 1].iloc[0]
        adj = g[g.intr_share >= 0.5]
        rows.append({"concept": c, "kind": kind, "f": f, "cond": cond, "seed": s,
                     "labelled": int(t.label.notna().all()), "has_P@1": int((P["rank"] <= 1).any()),
                     "has_P@3": int((P["rank"] <= 3).any()), "has_P@10": int(len(P) > 0),
                     "P_rank": float(bp["rank"]) if bp is not None else np.nan,
                     "P_rho": float(bp["rho"]) if bp is not None else np.nan,
                     "P_prec": float(bp["lex_hits"]) / 20 if bp is not None else np.nan,
                     "P_intr": float(bp["intr_share"]) if bp is not None else np.nan,
                     "P_tids": list(bp["top_tids"]) if bp is not None else None,
                     "P_disp": list(bp["top_disp"]) if bp is not None else None,
                     "nP": int((t.label == "P").sum()), "nB": int((t.label == "B").sum()),
                     "cf_feature": int(cf.feature),
                     "cf_rank": int(cf["rank"]), "cf_rho": float(cf.rho), "cf_hits": int(cf.lex_hits),
                     "cf_prec": cf.lex_hits / 20, "found5": int(cf.lex_hits >= 5),
                     "cf_spec": int(cf.spec_hits), "found_spec3": int(cf.spec_hits >= 3),
                     "cf_label": cf.label,
                     "cf_cshare": float(cf.c_share), "cf_intr": float(cf.intr_share), "cf_intr_any": int(cf.intr_any),
                     "cf_width": int(cf.width), "cf_tids": list(cf.top_tids), "cf_disp": list(cf.top_disp),
                     "top1_hits": int(t1.lex_hits), "top1_intr": float(t1.intr_share),
                     "top1_is_cf": int(t1.feature == cf.feature),
                     "adj_best_rank": float(adj["rank"].min()) if len(adj) else np.nan,
                     "n_adj_top10": int(((g.intr_share >= 0.5) & (g["rank"] <= 10)).sum()),
                     "lex_rank5": float(g[g.lex_hits >= 5]["rank"].min()) if (g.lex_hits >= 5).any() else np.nan})
    R = pd.DataFrame(rows)
    base = R[R.f == 0].drop_duplicates(["concept", "cond", "seed"]).set_index(["concept", "cond", "seed"])
    R["jacc_f0"] = [jacc(r.cf_tids, base.cf_tids.get((r.concept, r.cond, r.seed), [])) for r in R.itertuples()]
    R["P_jacc_f0"] = [jacc(r.P_tids, base.P_tids.get((r.concept, r.cond, r.seed)) or [])
                      if r.P_tids is not None else np.nan for r in R.itertuples()]
    curves = []
    for (c, kind, cond, f), g in R.groupby(["concept", "kind", "cond", "f"]):
        curves.append({"concept": c, "kind": kind, "cond": cond, "f": f, "n": len(g),
                       "labelled": g.labelled.mean(), "P@1": g["has_P@1"].mean(), "P@3": g["has_P@3"].mean(),
                       "P@10": g["has_P@10"].mean(), "P_rank_med": g.P_rank.median(),
                       "P_rank_mean_cap11": g.P_rank.fillna(11).mean(), "P_prec": g.P_prec.mean(),
                       "P_intr": g.P_intr.mean(), "P_jacc_f0": g.P_jacc_f0.mean(),
                       "P_stability": mean_pairwise_jaccard(g.P_tids.tolist()), "nP": g.nP.mean(),
                       "found5": g.found5.mean(), "found_spec3": g.found_spec3.mean(),
                       "cf_spec": g.cf_spec.mean(),
                       "cf_rank_mean": g.cf_rank.mean(), "cf_rank_med": g.cf_rank.median(),
                       "cf_rank_min": g.cf_rank.min(), "cf_rank_max": g.cf_rank.max(),
                       "cf_rho_med": g.cf_rho.median(), "cf_prec": g.cf_prec.mean(), "cf_prec_sd": g.cf_prec.std(),
                       "cf_cshare": g.cf_cshare.mean(), "cf_intr": g.cf_intr.mean(), "cf_intr_max": g.cf_intr.max(),
                       "top1_is_cf": g.top1_is_cf.mean(), "top1_intr": g.top1_intr.mean(),
                       "adj_best_rank_med": g.adj_best_rank.median(), "n_adj_top10": g.n_adj_top10.mean(),
                       "jacc_f0": g.jacc_f0.mean(), "stability": mean_pairwise_jaccard(g.cf_tids.tolist())})
    K = pd.DataFrame(curves)
    R.drop(columns=["cf_tids", "P_tids"]).to_csv(OUT / "runs_B.csv", index=False)
    K.to_csv(OUT / "curves_B.csv", index=False)
    print(K.round(3).to_string())
    return R, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("A", "B"))
    a = ap.parse_args()
    {"A": part_A, "B": part_B}[a.part]()


if __name__ == "__main__":
    main()
