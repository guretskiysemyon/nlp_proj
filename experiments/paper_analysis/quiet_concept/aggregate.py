"""Tables and paired statistics over concepts from spectra/*.json, features/*.json and frequency/summary.json.

    python -m experiments.paper_analysis.quiet_concept.aggregate

Unit = concept (median over its seeds first). Paired tests: Wilcoxon signed-rank; 95% CI = percentile bootstrap
(10,000 resamples of concepts) of the median paired difference. Scopes: all 38 concepts and the 18 EMBER concepts.
Writes tables/*.csv and stats.json under results/paper_analysis/quiet_concept/.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from experiments.paper_analysis.quiet_concept.common import OUT, concepts, dump
from experiments.varwidth.common import slug

CONDS = ["raw", "gs_raw", "rawp", "ctr", "wo", "cs", "gs"]
SPACES = ["raw", "rawp", "ctr", "wo", "cs"]
NOT_FOUND = 101          # rho rank assigned when no feature has >= 5 lexicon hits (censored)


def boot_ci(d: np.ndarray, B: int = 10_000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    meds = np.median(d[rng.integers(0, len(d), (B, len(d)))], axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def paired(a, b, name_a: str, name_b: str) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a
    w = stats.wilcoxon(a, b) if np.any(d != 0) else None
    return {"n": int(len(a)), f"median_{name_a}": float(np.median(a)), f"median_{name_b}": float(np.median(b)),
            "median_diff": float(np.median(d)), "ci95_median_diff": boot_ci(d),
            f"n_{name_b}_better(lower)": int((d < 0).sum()), f"n_{name_b}_higher": int((d > 0).sum()),
            "n_tie": int((d == 0).sum()), "wilcoxon_p": float(w.pvalue) if w else 1.0}


def load_features() -> pd.DataFrame:
    rows = []
    for c in concepts("all"):
        p = OUT / "features" / f"{slug(c)}.json"
        if p.exists():
            rows += json.loads(p.read_text())
    F = pd.DataFrame(rows)
    F["found"] = F.hits >= 5
    F["rank_c"] = np.where(F.found, F.rho_rank, NOT_FOUND)        # censored rank: not found -> 101
    return F


def decomposition(F: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """Key metrics along raw -> (raw with mu^_B projected out) -> centred -> whiten-only -> CS (-> CS+GS):
    medians over concepts (concept = median over its seeds)."""
    rows = []
    agg = F.groupby(["concept", "cond"]).agg(found=("found", "mean"), top1=("rank_c", lambda x: (x == 1).mean()),
                                             supp_N=("supp_N", "median"), z2=("z2_pct", "median"),
                                             rank=("rho_rank", "median")).reset_index()
    for cond in ("raw", "rawp", "ctr", "wo", "cs", "gs"):
        A = agg[agg.cond == cond]
        sp = {"gs": "cs"}.get(cond, cond)
        r = {"matrix": cond,
             "runs_with_concept_feature": float(F[F.cond == cond].found.mean()) if len(A) else None,
             "runs_rank1": float((F[F.cond == cond].rank_c == 1).mean()) if len(A) else None,
             "neutral_tokens_in_feature": float(A.supp_N.median()) if len(A) else None,
             "z2_percentile": float(A.z2.median()) if len(A) else None,
             "rho_rank_of_max_lexicon_feature": float(A["rank"].median()) if len(A) else None}
        if f"{sp}_spec_pr" in T:
            r.update({"spectrum_participation_ratio": float(T[f"{sp}_spec_pr"].median()),
                      "top1_pc_share": float(T[f"{sp}_spec_top1"].median()),
                      "svd_matched_rank": float(T[f"svdrank_{cond}"].median()) if f"svdrank_{cond}" in T else None,
                      "lex_dir_pca_rank": float(T[f"{sp}_lex_pca_rank"].median()),
                      "lex_dir_x_avg": float(T[f"{sp}_lex_x_avg_dir"].median()),
                      "lex_dir_topm_purity": float(T[f"{sp}_lex_Vm_purity_C"].median()),
                      "lexmd_pca_rank": float(T[f"{sp}_lexmd_pca_rank"].median()),
                      "md_pca_rank": float(T[f"{sp}_md_pca_rank"].median()),
                      "ge_pca_rank": float(T[f"{sp}_ge_pca_rank"].median()),
                      "seeded_purity": float(T[f"{sp}_seeded_purity_C"].median()),
                      "seeded_value_pct": (float(T[f"{sp}_seeded_value_pct_{cond}"].median())
                                           if f"{sp}_seeded_value_pct_{cond}" in T else None),
                      "C_over_N_token_energy": float(T[f"{sp}_tok_C_over_N_mean"].median())})
        rows.append(r)
    return pd.DataFrame(rows)


def load_spectra() -> dict:
    out = {}
    for c in concepts("all"):
        p = OUT / "spectra" / f"{slug(c)}.json"
        if p.exists():
            out[c] = json.loads(p.read_text())
    return out


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    paper = set(concepts("paper"))
    S = {}
    # ------------------------------------------------------------ features: rho rank of the concept feature
    F = load_features()
    F.drop(columns=["words", "rho_top20", "hits_by_rank", "F_C"]).to_csv(OUT / "tables" / "features_all_runs.csv",
                                                                        index=False)
    agg = F.groupby(["concept", "cond"]).agg(
        rank_med=("rank_c", "median"), rank_raw_med=("rho_rank", "median"), hits_med=("hits", "median"),
        found_rate=("found", "mean"), top1_rate=("rank_c", lambda x: float((x == 1).mean())),
        top20_rate=("rank_c", lambda x: float((x <= 20).mean())), rho_med=("rho", "median"),
        z2_pct_med=("z2_pct", "median"), dir_pct_med=("dir_pct", "median"), supp_C=("supp_C", "median"),
        supp_N=("supp_N", "median"), coh_pct=("coh_pct", "median"), feat_coh=("feat_coh_median", "median"),
        best_coh=("best_coh", "median"), tokE=("supp_tok_energy_rel", "median"),
        n_seeds=("seed", "nunique")).reset_index()
    agg.to_csv(OUT / "tables" / "feature_rank_by_concept_cond.csv", index=False)
    wide = agg.pivot(index="concept", columns="cond", values="rank_med")
    wide.to_csv(OUT / "tables" / "rho_rank_wide.csv")
    S["rho_rank"] = {}
    for scope, cs_ in (("all", list(wide.index)), ("paper", [c for c in wide.index if c in paper])):
        W = wide.loc[cs_]
        S["rho_rank"][scope] = {f"{a}_vs_{b}": paired(W[a], W[b], a, b)
                                for a, b in (("raw", "cs"), ("raw", "wo"), ("wo", "cs"), ("raw", "ctr"),
                                             ("ctr", "wo"), ("raw", "gs"), ("gs_raw", "gs"))
                                if a in W and b in W}
    # run-level rates
    S["runs"] = {}
    for cond in CONDS:
        G = F[F.cond == cond]
        if len(G):
            S["runs"][cond] = {"n_runs": int(len(G)), "n_concepts": int(G.concept.nunique()),
                               "found_rate": float(G.found.mean()), "top1_rate": float((G.rank_c == 1).mean()),
                               "top3_rate": float((G.rank_c <= 3).mean()), "top20_rate": float((G.rank_c <= 20).mean()),
                               "median_rank_found": float(G[G.found].rho_rank.median()) if G.found.any() else None,
                               "z2_pct_median": float(G.z2_pct.median()), "dir_pct_median": float(G.dir_pct.median()),
                               "supp_N_median": float(G.supp_N.median()), "supp_C_median": float(G.supp_C.median()),
                               "coh_pct_median": float(G.coh_pct.median()),
                               "tokE_median": float(G.supp_tok_energy_rel.median()),
                               "rho_max_median": float(G.rho_max.median())}
    # per-concept paired comparisons of every feature metric along raw -> ctr -> wo -> cs (and raw -> cs, raw -> gs)
    steps = (("raw", "rawp"), ("rawp", "cs"), ("raw", "ctr"), ("ctr", "wo"), ("wo", "cs"), ("raw", "cs"), ("raw", "wo"),
             ("ctr", "cs"),
             ("raw", "gs"), ("gs_raw", "gs"), ("cs", "gs"))
    S["feature_metrics"] = {}
    for scope, sel in (("all", None), ("paper", paper)):
        d = {}
        for v in ("found_rate", "top1_rate", "top20_rate", "rank_med", "rank_raw_med", "hits_med", "z2_pct_med",
                  "dir_pct_med", "supp_N", "supp_C", "coh_pct", "tokE"):
            W = agg.pivot(index="concept", columns="cond", values=v)
            if sel is not None:
                W = W.loc[[c for c in W.index if c in sel]]
            for a, b in steps:
                if a in W and b in W:
                    d[f"{v}|{a}->{b}"] = paired(W[a], W[b], a, b)
        S["feature_metrics"][scope] = d
    # norm puzzle: z2 percentile of the concept feature, raw vs cs
    zp = agg.pivot(index="concept", columns="cond", values="z2_pct_med")
    S["norm_puzzle_z2_pct"] = paired(zp["raw"], zp["cs"], "raw", "cs")
    S["norm_puzzle_z2_pct_wo"] = paired(zp["raw"], zp["wo"], "raw", "wo") if "wo" in zp else None
    zp.to_csv(OUT / "tables" / "z2_pct_wide.csv")
    # ------------------------------------------------------------ spectra
    SP = load_spectra()
    rows = []
    for c, r in SP.items():
        row = {"concept": c, "paper": c in paper, "n": r["n"], "n_C": r["n_C"], "n_N": r["n_N"],
               "n_lex": r["n_lex_tokens"], "g1": r["cpca"]["g1"], "n_g_gt1": r["cpca"]["n_g_gt1"],
               "p1_cos2_pc1_cs": r["cpca"]["p1_cos2_pc1_cs"], "p1_share_cs": r["cpca"]["p1_share_cs"],
               "neutral_wh_energy_per_dir": r["cpca"]["neutral_whitened_mean_energy_per_dir"],
               "concept_wh_energy_per_dir": r["cpca"]["concept_whitened_mean_energy_per_dir"]}
        for sp in SPACES:
            R = r[sp]
            for k, v in R["spectrum"].items():
                row[f"{sp}_spec_{k}"] = v
            for k in ("C_share", "C_over_N_mean", "top1pct_token_share"):
                row[f"{sp}_tok_{k}"] = R["tokens"][k]
            for dname, D in R["dirs"].items():
                for k in ("share", "pca_rank", "x_avg_dir", "Vm_purity_C", "Vm_purity_N", "Vm_purity_lex",
                          "cos2_top10", "bgvar_rel", "bgvar_rank"):
                    row[f"{sp}_{dname}_{k}"] = D.get(k)
                for k, v in D.items():
                    if k.startswith("Vm_pct_"):
                        row[f"{sp}_{dname}_{k}"] = v
            if "lex" in R:
                for k, v in R["lex"].items():
                    row[f"{sp}_lexset_{k}"] = v
            if "seeded" in R:
                for k, v in R["seeded"].items():
                    row[f"{sp}_seeded_{k}"] = v
            for cond, d in R["svd_rank"].items():
                if d:
                    row[f"svdrank_{cond}"] = float(np.median(list(d.values())))
                    row[f"relerr_{cond}"] = float(np.median(list(R["rel_err"][cond].values())))
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tables" / "spectra_by_concept.csv", index=False)
    Dc = decomposition(F, T)
    Dc.to_csv(OUT / "tables" / "decomposition.csv", index=False)
    S["decomposition"] = Dc.to_dict(orient="records")
    S["spectra"] = {}
    for scope, sel in (("all", T.concept.notna()), ("paper", T.paper)):
        X = T[sel]
        d = {}
        for dname in ("md", "ge", "lex", "lexmd"):
            for a, b in (("raw", "cs"), ("raw", "wo"), ("wo", "cs"), ("raw", "ctr"), ("raw", "rawp"),
                         ("ctr", "wo"), ("rawp", "cs")):
                for k in ("pca_rank", "share", "Vm_purity_C", "x_avg_dir"):
                    ca, cb = f"{a}_{dname}_{k}", f"{b}_{dname}_{k}"
                    if ca in X and cb in X:
                        d[f"{dname}_{k}_{a}_vs_{b}"] = paired(X[ca], X[cb], a, b)
        for k in ("purity_C", "lex_share", "value_pct_raw", "frac_L_kept"):
            pass
        for a, b in (("raw", "cs"), ("raw", "wo"), ("wo", "cs"), ("raw", "ctr"), ("ctr", "wo"), ("raw", "rawp")):
            for k in ("purity_C", "lex_share", "frac_L_kept", "value_share"):
                ca, cb = f"{a}_seeded_{k}", f"{b}_seeded_{k}"
                if ca in X and cb in X:
                    d[f"seeded_{k}_{a}_vs_{b}"] = paired(X[ca], X[cb], a, b)
        vp = {"raw": "value_pct_raw", "ctr": "value_pct_ctr", "wo": "value_pct_wo", "cs": "value_pct_cs",
              "rawp": "value_pct_rawp"}
        for a, b in (("raw", "cs"), ("raw", "wo"), ("wo", "cs"), ("raw", "ctr"), ("ctr", "wo"), ("raw", "rawp"),
                     ("rawp", "cs")):
            ca, cb = f"{a}_seeded_{vp[a]}", f"{b}_seeded_{vp[b]}"
            if ca in X and cb in X:
                d[f"seeded_value_pct_{a}_vs_{b}"] = paired(X[ca], X[cb], a, b)
        for a, b in (("raw", "cs"), ("raw", "wo"), ("wo", "cs"), ("raw", "ctr"), ("ctr", "wo"), ("raw", "gs"),
                     ("raw", "rawp"), ("rawp", "cs")):
            ca, cb = f"svdrank_{a}", f"svdrank_{b}"
            if ca in X and cb in X:
                d[f"svdrank_{a}_vs_{b}"] = paired(X[ca], X[cb], a, b)
        for k in ("top1", "pr", "n50"):
            for a, b in (("raw", "cs"), ("raw", "wo"), ("raw", "ctr"), ("raw", "rawp"), ("ctr", "wo"), ("wo", "cs")):
                d[f"spec_{k}_{a}_vs_{b}"] = paired(X[f"{a}_spec_{k}"], X[f"{b}_spec_{k}"], a, b)
        d["tok_C_share_raw_vs_cs"] = paired(X["raw_tok_C_share"], X["cs_tok_C_share"], "raw", "cs")
        d["tok_C_over_N_raw_vs_cs"] = paired(X["raw_tok_C_over_N_mean"], X["cs_tok_C_over_N_mean"], "raw", "cs")
        d["tok_C_over_N_raw_vs_wo"] = paired(X["raw_tok_C_over_N_mean"], X["wo_tok_C_over_N_mean"], "raw", "wo")
        d["g1_median"] = float(X.g1.median())
        d["g1_range"] = [float(X.g1.min()), float(X.g1.max())]
        d["n_g_gt1_median"] = float(X.n_g_gt1.median())
        d["neutral_wh_energy_per_dir_median"] = float(X.neutral_wh_energy_per_dir.median())
        d["concept_wh_energy_per_dir_median"] = float(X.concept_wh_energy_per_dir.median())
        S["spectra"][scope] = d
    freq = OUT / "frequency" / "summary.json"
    if freq.exists():
        S["frequency"] = json.loads(freq.read_text())
    dump(S, OUT / "stats.json")
    print(json.dumps(S["runs"], indent=1))


if __name__ == "__main__":
    main()
