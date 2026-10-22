#!/usr/bin/env python
"""WP3 analysis: seeds / serial number, dense re-solve and natural tails, breadth, RS vs GS directions, GS needs CS.

    .venv/bin/python -m experiments.paper_analysis.widths.analyze

Reads results/paper_analysis/widths/cache/<slug>.pkl (collect.py); writes CSV / JSON tables and tables.md to
results/paper_analysis/widths/. Figures: figures.py.

Definitions (fixed before looking at any width-vs-tail correlation):
  labels along a dense ranking: C = concept-only (code 1); X = page-enriched: occurs in S_C and
      ln[p_page / p_wiki] >= ln(theta_x), theta_x = 20 (robustness 5, 100); LEX = source word in the 20-word lexicon.
  natural tail T (window rule, primary): T = min{m >= 0 : mean(x_{m+1..m+w}) < theta}, x = X labels of the dense
      ranking, w = 20, theta = 0.5 (grid: theta 0.3-0.7, w 10/20/30, labels C / X5 / X20 / X100).
  natural tail T (change point): the single Bernoulli change point of x_1..x_N maximizing the likelihood with
      p_head > p_tail, N = 300 (grid 200 / 400).
  concept direction: CINDER's concept feature (L12, max lexicon hits among the top-20 rho candidates, ties by rho) =
      the anchor; in the other fits of the same seed its best |cos| match (raw fits: anchor mapped back by T_inv).
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, spearmanr

from experiments.paper_analysis.widths.collect import load
from experiments.paper_analysis.widths.common import (OUT, SEEDS, boot_ci, boot_spearman, concepts, slug,
                                                      wilcoxon)

THX = 20
PRIMARY = {"label": "X20", "w": 20, "theta": 0.5}
GRID_THETA = (0.3, 0.4, 0.5, 0.6, 0.7)
GRID_W = (10, 20, 30)
GRID_LABELS = ("C", "X5", "X20", "X100")
CP_N = (200, 300, 400)
CS_CFGS = ("L1", "L12", "RS2")


# ---------------------------------------------------------------- labels and tails
def label_arrays(L: dict) -> dict:
    occ = L["c_C"] > 0
    out = {"C": L["code"] == 1, "N": L["code"] == 0, "LEX": np.asarray(L["lex"], bool)}
    for t in (5, 20, 100):
        out[f"X{t}"] = occ & (L["ex"] >= math.log(t))
    return out


def tail_window(x: np.ndarray, w: int, theta: float) -> int:
    """min m >= 0 with mean(x[m:m+w]) < theta (0-based x; x[m] is rank m+1); len(x) - w if never (censored)."""
    c = np.concatenate([[0], np.cumsum(x, dtype=float)])
    m = np.arange(len(x) - w + 1)
    win = (c[m + w] - c[m]) / w
    hit = np.nonzero(win < theta)[0]
    return int(hit[0]) if len(hit) else int(len(x) - w)


def tail_changepoint(x: np.ndarray, N: int, lo: int = 5) -> int:
    x = np.asarray(x[:N], float)
    c = np.cumsum(x)
    best, bt = -np.inf, lo
    tot = c[-1]

    def ll(k, n):
        if n == 0:
            return 0.0
        p = k / n
        return (k * math.log(p) if k > 0 else 0.0) + ((n - k) * math.log(1 - p) if n - k > 0 else 0.0)

    for T in range(lo, N - lo):
        k1, k2 = c[T - 1], tot - c[T - 1]
        if k1 / T <= k2 / (N - T):
            continue
        v = ll(k1, T) + ll(k2, N - T)
        if v > best:
            best, bt = v, T
    return bt


def tails(x: np.ndarray) -> dict:
    out = {}
    for w in GRID_W:
        for th in GRID_THETA:
            out[f"win_w{w}_t{th}"] = tail_window(x, w, th)
    for N in CP_N:
        out[f"cp_N{N}"] = tail_changepoint(x, N)
    return out


def feature_rows(concept: str, D: dict) -> list:
    """One row per (config, seed, which) feature ranking: which = own (max-LO concept feature of that fit) or anchor
    (L12's concept feature, or its best match in another fit). Tails for every label and rule."""
    L = D["labels"]
    lab = label_arrays(L)
    rows = []
    items = [(cfg, s, "own", r) for (cfg, s), r in D["runs"].items()]
    items += [(cfg, s, "anchor", r) for (cfg, s), r in D["anchor"].items()]
    items += [("L12", s, "anchor", D["runs"][("L12", s)]) for s in SEEDS if ("L12", s) in D["runs"]]
    n = len(L["vstar"])
    for cfg, s, which, r in items:
        o = r["order"]
        sup = r["support"]
        w = int(r["width"])
        row = {"concept": concept, "config": cfg, "seed": s, "which": which, "feature": int(r["feature"]),
               "width": w, "n": n, "m_rs": math.ceil(0.01 * n)}
        if which == "own":
            row.update({"lo": r["lo"], "rho_rank": r["rho_rank"], "hand": r["hand"], "tj": r["tj"]})
        else:
            row.update({"match_cos": r.get("cos", 1.0), "match_is_concept": r.get("is_concept", True)})
        for k, v in lab.items():
            row[f"sup_{k}"] = float(v[sup].mean()) if len(sup) else np.nan
        # neutral share of the part of a GS support beyond the RS width (by dense rank)
        m = row["m_rs"]
        if w > m:
            row["extra_N"] = float(lab["N"][o[m:w]].mean())
            row["extra_X20"] = float(lab["X20"][o[m:w]].mean())
        row["head_N"] = float(lab["N"][o[:min(w, m)]].mean())
        row["lex_in_support"] = int(lab["LEX"][sup].sum())
        row["lex_total"] = int(lab["LEX"].sum())
        sh = r["share"].astype(float)
        row["share_at_cut"] = float(sh[w - 1]) if w <= len(sh) else np.nan
        row["energy_support"] = float((sh[:w] ** 2).sum())
        row["energy_top600"] = float((sh ** 2).sum())
        for lname in GRID_LABELS:
            x = lab[lname][o].astype(float)
            for k, v in tails(x).items():
                row[f"T_{lname}_{k}"] = v
        Tp = row[PRIMARY_COL]
        row["share_at_T"] = float(sh[Tp - 1]) if 0 < Tp <= len(sh) else np.nan
        rows.append(row)
    return rows


PRIMARY_COL = f"T_{PRIMARY['label']}_win_w{PRIMARY['w']}_t{PRIMARY['theta']}"


# ---------------------------------------------------------------- assemble
def assemble():
    F, R, X, C = [], [], [], []
    curves = {}
    for c in concepts():
        D = load(c)
        L = D["labels"]
        lab = label_arrays(L)
        F += feature_rows(c, D)
        for (cfg, s), r in D["runs"].items():
            free = (r["widths"] > r["m_min"]) & (r["widths"] < r["m_max"])
            R.append({"concept": c, "config": cfg, "seed": s, "max_rel_diff": float(r["rel_diff_support"].max()),
                      "top_ok": float(r["top_ok"].mean()), "feature": int(r["feature"]), "width": int(r["width"]),
                      "widths": r["widths"].tolist(), "m_min": r["m_min"], "m_max": r["m_max"],
                      "tau_min": float(r["min_sup_share"][free].min()) if free.any() else np.nan,
                      "max_off_free": float(r["max_off_share"][free].max()) if free.any() else np.nan,
                      "concept_min_sup": float(r["min_sup_share"][r["feature"]]),
                      "concept_max_off": float(r["max_off_share"][r["feature"]]),
                      "znorm_share": float(r["znorm"][r["feature"]] / r["znorm"].sum()),
                      "lo": r["lo"], "hand": r["hand"], "rel_err": r["rel_err"]})
        for k, v in D["cross"].items():
            a, s, b, t = k
            X.append({"concept": c, "a": a, "s": s, "b": b, "t": t, **v})
        C.append({"concept": c, "n": len(L["vstar"]), "n_C": int(lab["C"].sum()), "n_X20": int(lab["X20"].sum()),
                  "n_X100": int(lab["X100"].sum()), "n_lex_tokens": int(lab["LEX"].sum()),
                  "n_lex_words_SC": len(L["lex_in_SC"]), "n_target": L["n_target"], "page_sents": L["page_sents"],
                  "N_C": L["N_C"], "code_agree": L["code_agree"], "base_C": float(lab["C"].mean()),
                  "base_X20": float(lab["X20"].mean())})
        # mean precision curves (anchor directions) for the figure
        for cfg in ("L1", "L12", "RS2", "L0"):
            for lname in ("C", "X20", "LEX"):
                xs = []
                for s in SEEDS:
                    r = D["runs"].get((cfg, s)) if cfg == "L12" else D["anchor"].get((cfg, s))
                    if r is not None:
                        xs.append(lab[lname][r["order"]].astype(float))
                if xs:
                    curves[(c, cfg, lname)] = np.mean(xs, 0)
    F = pd.DataFrame(F)
    R = pd.DataFrame(R)
    X = pd.DataFrame(X)
    C = pd.DataFrame(C)
    return F, R, X, C, curves


# ---------------------------------------------------------------- analyses
def part1_seeds(F: pd.DataFrame, R: pd.DataFrame, X: pd.DataFrame) -> dict:
    out = {}
    own = F[F.which == "own"]
    for cfg in ("L12", "rnP", "L1", "L0"):
        g = own[own.config == cfg].groupby("concept").width
        per = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "min": g.min(), "max": g.max(), "range": g.max() - g.min()})
        # one-way ICC(1): share of width variance between concepts
        x = own[own.config == cfg]
        k = x.groupby("concept").size().mean()
        msb = x.groupby("concept").width.mean().var(ddof=1) * k
        msw = x.groupby("concept").width.var(ddof=1).mean()
        icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else np.nan
        out[cfg] = {"concept_mean_range": [float(per["mean"].min()), float(per["mean"].max())],
                    "median_sd": float(per["sd"].median()), "mean_sd": float(per["sd"].mean()),
                    "median_range": float(per["range"].median()), "max_range": float(per["range"].max()),
                    "icc1": float(icc), "n_concepts": int(len(per)),
                    "kruskal_p": float(kruskal(*[v.values for _, v in x.groupby("concept").width]).pvalue)
                    if per["mean"].nunique() > 1 else np.nan}
        per.round(2).to_csv(OUT / f"seed_widths_{cfg}.csv")
    # anchor-direction widths (same concept direction across seeds; L12 = own)
    anc = F[(F.which == "anchor") & (F.config == "L12")]
    # identification consistency across seeds: L12 concept feature at seed s is the best match of seed t's
    xs = X[(X.a == "L12") & (X.b == "L12") & (X.s != X.t)]
    out["L12_cross_seed_same_feature"] = float((xs.is_concept & (xs.best_cos > 0.9)).mean())
    # width SD within the largest group of seeds whose concept features match each other (|cos| > 0.9)
    rows = []
    for c, g in anc.groupby("concept"):
        e = xs[(xs.concept == c) & xs.is_concept & (xs.best_cos > 0.9)]
        parent = {sd: sd for sd in g.seed}

        def find(u):
            while parent[u] != u:
                u = parent[u]
            return u
        for a, b in zip(e.s, e.t):
            parent[find(a)] = find(b)
        comp = pd.Series({sd: find(sd) for sd in g.seed})
        big = comp.value_counts().index[0]
        members = comp[comp == big].index
        w = g.set_index("seed").width.loc[members]
        rows.append({"concept": c, "n_consistent": len(members), "sd_all": g.width.std(ddof=1),
                     "sd_consistent": w.std(ddof=1) if len(w) > 1 else np.nan, "range_consistent": w.max() - w.min()})
    G = pd.DataFrame(rows)
    G.round(2).to_csv(OUT / "seed_widths_consistent_L12.csv", index=False)
    out["L12_consistent"] = {"concepts_all5_consistent": int((G.n_consistent == 5).sum()),
                             "median_n_consistent": float(G.n_consistent.median()),
                             "median_sd_consistent": float(G.sd_consistent.median()),
                             "mean_sd_consistent": float(G.sd_consistent.mean()),
                             "median_range_consistent": float(G.range_consistent.median()),
                             "median_sd_all": float(G.sd_all.median())}
    # serial number: width vs feature index, all features of all runs
    ser = {}
    for cfg in ("L12", "rnP"):
        rr = R[R.config == cfg]
        rhos = [spearmanr(np.arange(100), w).statistic for w in rr.widths]
        pooled = spearmanr(np.tile(np.arange(100), len(rr)), np.concatenate(rr.widths.values))
        ow = own[own.config == cfg]
        conc = spearmanr(ow.feature, ow.width)
        # concept feature index: uniform over 0..99? (KS on discrete uniform via chi-square on deciles)
        dec = np.bincount(ow.feature.values // 10, minlength=10)
        from scipy.stats import chisquare
        ser[cfg] = {"per_run_rho_mean": float(np.mean(rhos)), "per_run_rho_absmax": float(np.max(np.abs(rhos))),
                    "per_run_p05": float(np.mean([spearmanr(np.arange(100), w).pvalue < 0.05 for w in rr.widths])),
                    "pooled_rho": float(pooled.statistic), "pooled_p": float(pooled.pvalue),
                    "n_runs": int(len(rr)),
                    "concept_index_vs_width_rho": float(conc.statistic), "concept_index_vs_width_p": float(conc.pvalue),
                    "concept_index_deciles": dec.tolist(), "concept_index_uniform_p": float(chisquare(dec).pvalue)}
        # within concept: does the seed with the larger index get the wider feature? (Spearman within concept, mean)
        wc = [spearmanr(g.feature, g.width).statistic for _, g in ow.groupby("concept")
              if g.feature.nunique() > 1 and g.width.nunique() > 1]
        ser[cfg]["within_concept_rho_mean"] = float(np.nanmean(wc))
        ser[cfg]["within_concept_rho_ci"] = boot_ci(wc)
    out["serial"] = ser
    # the slot is set by the init: same seed, different width rule -> same index
    same = X[(X.a == "L12") & (X.s == X.t) & X.b.isin(["L1", "RS2", "rnP"])]
    out["slot_same_index_same_seed"] = {b: float(g.same_index.mean()) for b, g in same.groupby("b")}
    diff_seed = X[(X.a == "L12") & (X.b == "L12") & (X.s != X.t)]
    out["slot_same_index_other_seed"] = float(diff_seed.same_index.mean())
    return out


def concept_table(A: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """Per-concept means over seeds of the anchor direction: widths per config and tails T (column col) per fit."""
    col = col or PRIMARY_COL
    P = A.groupby(["concept", "config"]).agg(width=("width", "mean"), width_sd=("width", "std"), m_rs=("m_rs", "first"),
                                             T=(col, "mean"), T_sd=(col, "std"), cos=("match_cos", "mean"),
                                             n_runs=("seed", "nunique")).reset_index()
    W = P.pivot(index="concept", columns="config")
    T = pd.DataFrame({"m_rs": W["m_rs"]["L12"], "w_L1": W["width"]["L1"], "w_L12": W["width"]["L12"],
                      "w_L12_sd": W["width_sd"]["L12"], "w_RS2": W["width"]["RS2"], "w_rnP_match": W["width"]["rnP"],
                      "T_L1": W["T"]["L1"], "T_L12": W["T"]["L12"], "T_RS2": W["T"]["RS2"], "T_L0": W["T"]["L0"],
                      "T_L1_sd": W["T_sd"]["L1"], "cos_L1": W["cos"]["L1"], "cos_RS2": W["cos"]["RS2"],
                      "n_runs": W["n_runs"]["L12"]})
    T["T_cons"] = T[["T_L1", "T_L12", "T_RS2"]].mean(1)
    return T


def logratio_disp(w, T) -> np.ndarray:
    return np.log((np.asarray(w, float) + 1) / (np.asarray(T, float) + 1))


def tracking_stats(T: pd.DataFrame, ref: str, boot: bool = True) -> dict:
    if not boot:
        ae_gs, ae_rs = (T.w_L12 - T[ref]).abs(), (T.m_rs - T[ref]).abs()
        sg, sr = spearmanr(T.w_L12, T[ref]), spearmanr(T.m_rs, T[ref])
        return {"spearman_wGS": {"rho": float(sg.statistic), "p": float(sg.pvalue)},
                "spearman_mRS": {"rho": float(sr.statistic), "p": float(sr.pvalue)},
                "MAE_GS": float(ae_gs.mean()), "MAE_RS": float(ae_rs.mean()),
                "n_GS_closer": int((ae_gs < ae_rs).sum()), "n_RS_closer": int((ae_rs < ae_gs).sum()),
                "sd_logratio_GS": float(np.std(logratio_disp(T.w_L12, T[ref]), ddof=1)),
                "sd_logratio_RS": float(np.std(logratio_disp(T.m_rs, T[ref]), ddof=1))}
    ae_gs = (T.w_L12 - T[ref]).abs()
    ae_rs = (T.m_rs - T[ref]).abs()
    lg, lr = logratio_disp(T.w_L12, T[ref]), logratio_disp(T.m_rs, T[ref])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(5000):
        i = rng.integers(0, len(lg), len(lg))
        bs.append(np.std(lr[i], ddof=1) - np.std(lg[i], ddof=1))
    return {"spearman_wGS": boot_spearman(T.w_L12, T[ref]), "spearman_mRS": boot_spearman(T.m_rs, T[ref]),
            "MAE_GS": float(ae_gs.mean()), "MAE_RS": float(ae_rs.mean()),
            "MAE_RS_minus_GS": wilcoxon(ae_rs - ae_gs),
            "n_GS_closer": int((ae_gs < ae_rs).sum()), "n_RS_closer": int((ae_rs < ae_gs).sum()),
            # level-free: spread of ln((w+1)/(T+1)) over concepts (a global rescaling of T does not change it)
            "sd_logratio_GS": float(np.std(lg, ddof=1)), "sd_logratio_RS": float(np.std(lr, ddof=1)),
            "sd_logratio_RS_minus_GS_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": int(len(T))}


def part2_dense(F: pd.DataFrame, R: pd.DataFrame, C: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    out = {"verify": {"runs": int(len(R)), "max_rel_diff": float(R.max_rel_diff.max()),
                      "median_max_rel_diff": float(R.max_rel_diff.median()), "top_ok_all": float(R.top_ok.min())}}
    # GS: within a fit, every free row's support is {share > tau} for one global tau
    gs = R[R.config.isin(["L12", "rnP"])]
    out["gs_threshold"] = {cfg: {"tau_median": float(g.tau_min.median()),
                                 "runs_separable": float((g.max_off_free <= g.tau_min * (1 + 1e-4)).mean()),
                                 "concept_rows_at_floor_or_cap": float(((g.width <= g.m_min) | (g.width >= g.m_max)).mean())}
                           for cfg, g in gs.groupby("config")}
    A = F[F.which == "anchor"].copy()
    T = concept_table(A)
    # co-adaptation: does the tail follow the fit's own width?
    out["coadapt"] = {
        "T_RS2_minus_T_L1": wilcoxon(T.T_RS2 - T.T_L1), "T_L12_minus_T_L1": wilcoxon(T.T_L12 - T.T_L1),
        "w_RS2_minus_m_rs_median": float((T.w_RS2 - T.m_rs).median()),
        "spearman_T_L1_T_RS2": boot_spearman(T.T_L1, T.T_RS2), "spearman_T_L1_T_L12": boot_spearman(T.T_L1, T.T_L12),
        "median_T": {k: float(T[k].median()) for k in ("T_L1", "T_L12", "T_RS2", "T_L0")},
        "median_width": {k: float(T[k].median()) for k in ("m_rs", "w_L12", "w_RS2")}}
    # RS cut short vs padded (RS-1% CS fit, the anchor direction); EMBER's raw fit for reference
    T["rs_class"] = np.where(T.T_L1 > 1.2 * T.m_rs, "cut short", np.where(T.T_L1 < 0.8 * T.m_rs, "padded", "about right"))
    T["rs_class_L0"] = np.where(T.T_L0 > 1.2 * T.m_rs, "cut short", np.where(T.T_L0 < 0.8 * T.m_rs, "padded", "about right"))
    out["rs_class_L1"] = T.rs_class.value_counts().to_dict()
    out["rs_class_L0"] = T.rs_class_L0.value_counts().to_dict()
    out["T_over_mrs_L1"] = {"median": float((T.T_L1 / T.m_rs).median()), "iqr": [float((T.T_L1 / T.m_rs).quantile(q)) for q in (.25, .75)]}
    # does GS move the width in the right direction?
    need, gsd = T.T_L1 > T.m_rs, T.w_L12 > T.m_rs
    tab = [[int((need & gsd).sum()), int((need & ~gsd).sum())], [int((~need & gsd).sum()), int((~need & ~gsd).sum())]]
    out["direction_fisher"] = {"rows: T>m_RS / T<=m_RS; cols: GS wider / GS not wider": tab,
                               "p": float(fisher_exact(tab).pvalue)}
    out["tracking"] = {ref: tracking_stats(T, ref) for ref in ("T_L1", "T_L12", "T_RS2", "T_cons")}
    from scipy.stats import theilslopes
    ts = theilslopes(T.w_L12, T.T_L1)
    out["theil_sen_wGS_on_T_L1"] = {"slope": float(ts.slope), "lo": float(ts.low_slope), "hi": float(ts.high_slope),
                                     "intercept": float(ts.intercept)}
    out["GS_minus_T_by_T_tercile"] = {str(k): float(v) for k, v in
                                       (T.w_L12 - T.T_L1).groupby(pd.qcut(T.T_L1, 3, labels=["low", "mid", "high"])).median().items()}
    # mechanism: the dense share at the natural tail of the GS fit's own concept row, relative to the fit's global tau
    a12 = F[(F.which == "anchor") & (F.config == "L12")].merge(R[R.config == "L12"][["concept", "seed", "tau_min"]],
                                                               on=["concept", "seed"])
    ratio = (a12.share_at_T / a12.tau_min).dropna()
    out["share_at_T_over_tau_L12"] = {"median": float(ratio.median()), "iqr": [float(ratio.quantile(.25)), float(ratio.quantile(.75))],
                                      "runs": int(len(ratio))}
    # sensitivity: anchors that are real concept features (judge-free EJ@5, or hand label P)
    own12 = F[(F.which == "own") & (F.config == "L12")].set_index(["concept", "seed"])
    A = A.join(own12[["lo", "hand"]].rename(columns={"lo": "anchor_lo", "hand": "anchor_hand"}), on=["concept", "seed"])
    sens = {}
    for name, keep in (("anchor EJ@5 (LO>=5)", A.anchor_lo >= 5), ("anchor hand label P", A.anchor_hand == "P")):
        Ts = concept_table(A[keep])
        Ts = Ts[Ts.n_runs >= 2]
        sens[name] = {"n_concepts": int(len(Ts)), "runs": int(keep[A.config == "L12"].sum()),
                      **{ref: tracking_stats(Ts, ref) for ref in ("T_L1", "T_cons")},
                      "rs_class_L1": pd.Series(np.where(Ts.T_L1 > 1.2 * Ts.m_rs, "cut short",
                                                        np.where(Ts.T_L1 < 0.8 * Ts.m_rs, "padded", "about right"))).value_counts().to_dict(),
                      "width_sd_median": float(Ts.w_L12_sd.median())}
    out["sensitivity"] = sens
    # robustness of the tracking result over the tail grid
    rob = []
    for col in [c for c in F.columns if c.startswith("T_")]:
        Q = concept_table(A, col)
        for ref in ("T_L1", "T_L12", "T_RS2"):
            st = tracking_stats(Q, ref, boot=False)
            rob.append({"tail": col[2:], "ref": ref, "median_T": float(Q[ref].median()),
                        "spearman_wGS": st["spearman_wGS"]["rho"], "p": st["spearman_wGS"]["p"],
                        "spearman_mRS": st["spearman_mRS"]["rho"], "MAE_GS": st["MAE_GS"], "MAE_RS": st["MAE_RS"],
                        "GS_closer": st["n_GS_closer"], "RS_closer": st["n_RS_closer"],
                        "sd_logratio_GS": st["sd_logratio_GS"], "sd_logratio_RS": st["sd_logratio_RS"],
                        "cut_short": int((Q[ref] > 1.2 * Q.m_rs).sum()), "padded": int((Q[ref] < 0.8 * Q.m_rs).sum()),
                        "T_RS2_minus_T_L1": float((Q.T_RS2 - Q.T_L1).median())})
    rob = pd.DataFrame(rob)
    rob.round(4).to_csv(OUT / "tail_robustness.csv", index=False)
    out["robustness_summary"] = {
        ref: {"n_defs": int(len(g)), "spearman_wGS_median": float(g.spearman_wGS.median()),
              "spearman_wGS_range": [float(g.spearman_wGS.min()), float(g.spearman_wGS.max())],
              "n_rho_pos": int((g.spearman_wGS > 0).sum()), "n_p05": int((g.p < 0.05).sum()),
              "n_rho_gt_mRS": int((g.spearman_wGS > g.spearman_mRS).sum()),
              "n_MAE_GS_lt_RS": int((g.MAE_GS < g.MAE_RS).sum()),
              "n_sdlog_GS_lt_RS": int((g.sd_logratio_GS < g.sd_logratio_RS).sum()),
              "T_RS2_minus_T_L1_median": float(g.T_RS2_minus_T_L1.median()),
              "T_RS2_minus_T_L1_n_pos": int((g.T_RS2_minus_T_L1 > 0).sum())}
        for ref, g in rob.groupby("ref")}
    # the same, excluding the strictest label (X100, tails collapse to ~0) and censored lenient rules (median T > 200)
    ok = rob[~rob["tail"].str.startswith("X100") & (rob.median_T <= 200)]
    out["robustness_summary_reasonable"] = {
        ref: {"n_defs": int(len(g)), "spearman_wGS_range": [float(g.spearman_wGS.min()), float(g.spearman_wGS.max())],
              "n_p05": int((g.p < 0.05).sum()), "n_MAE_GS_lt_RS": int((g.MAE_GS < g.MAE_RS).sum()),
              "n_sdlog_GS_lt_RS": int((g.sd_logratio_GS < g.sd_logratio_RS).sum())}
        for ref, g in ok.groupby("ref")}
    T = T.join(C.set_index("concept"))
    T.round(3).to_csv(OUT / "natural_tail_per_concept.csv")
    return out, T


def part3_breadth(T: pd.DataFrame, F: pd.DataFrame) -> dict:
    own = F[F.which == "own"]
    wP = own[own.config == "rnP"].groupby("concept").width.mean()
    T = T.assign(w_rnP=wP)
    out = {}
    proxies = {"n_C (concept-only tokens in V*)": "n_C", "n_X20 (page-enriched tokens in V*)": "n_X20",
               "n (|V*|)": "n", "page token types (n_target)": "n_target", "page sentences": "page_sents",
               "lexicon words in S_C": "n_lex_words_SC", "natural tail T (L1 direction)": "T_L1",
               "natural tail T (consensus)": "T_cons"}
    for wcol, name in (("w_L12", "CS GS width"), ("w_rnP", "raw GS width"), ("m_rs", "RS-1% width")):
        out[name] = {p: boot_spearman(T[wcol], T[col]) for p, col in proxies.items()}
    # partial: GS width vs n_X20 controlling for n (rank residuals)
    from scipy.stats import rankdata

    def resid(y, x):
        y, x = rankdata(y), rankdata(x)
        b = np.polyfit(x, y, 1)
        return y - np.polyval(b, x)
    out["partial_wGS_nX20_given_n"] = float(np.corrcoef(resid(T.w_L12, T.n), resid(T.n_X20, T.n))[0, 1])
    out["breadth_proxies_intercorr"] = {f"{a}~{b}": float(spearmanr(T[a], T[b]).statistic)
                                        for a, b in itertools.combinations(["n_C", "n_X20", "n_target", "n_lex_words_SC",
                                                                            "T_L1"], 2)}
    return out


def part4_directions(X: pd.DataFrame, R: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    out = {}
    rows = []
    wa = R.set_index(["concept", "config", "seed"]).width
    X = X.copy()
    X["width_a"] = [wa.loc[(c, a, s)] for c, a, s in zip(X.concept, X.a, X.s)]
    inter = X.jac_best * (X.width_a + X.width_best) / (1 + X.jac_best)
    X["overlap_best"] = inter / np.minimum(X.width_a, X.width_best)
    def summ(df, name):
        per = df.groupby("concept").agg(cos=("best_cos", "mean"), jac=("jac_best", "mean"),
                                        overlap=("overlap_best", "mean"),
                                        is_concept=("is_concept", "mean"), same_index=("same_index", "mean"),
                                        cos_concept=("cos_concept", "mean"), jac_concept=("jac_concept", "mean"))
        rows.append(per.assign(pair=name).reset_index())
        return {"n_runs": int(len(df)), "n_concepts": int(per.shape[0]),
                "cos_median": float(per.cos.median()), "cos_mean": float(per.cos.mean()), "cos_ci": boot_ci(per.cos),
                "cos_min_concept": float(per.cos.min()), "frac_runs_cos_gt_0.9": float((df.best_cos > 0.9).mean()),
                "jac_mean": float(per.jac.mean()), "jac_ci": boot_ci(per.jac),
                "overlap_coef_mean": float(per.overlap.mean()),
                "identification_agrees": float(df.is_concept.mean()), "same_index": float(df.same_index.mean()),
                "cos_independent_ids_mean": float(per.cos_concept.mean()),
                "jac_independent_ids_mean": float(per.jac_concept.mean())}
    same = X[X.s == X.t]
    cross = X[X.s != X.t]
    out["L12->L1 same seed (GS vs RS, CS space)"] = summ(same[(same.a == "L12") & (same.b == "L1")], "L12->L1")
    out["L12->RS2 same seed"] = summ(same[(same.a == "L12") & (same.b == "RS2")], "L12->RS2")
    out["L1->L1 across seeds (RS seed noise)"] = summ(cross[(cross.a == "L1") & (cross.b == "L1")], "L1->L1 x-seed")
    out["L12->L12 across seeds (GS seed noise)"] = summ(cross[(cross.a == "L12") & (cross.b == "L12")], "L12->L12 x-seed")
    out["L0->L1 same seed (raw vs CS mapped back)"] = summ(same[(same.a == "L0") & (same.b == "L1")], "L0->L1")
    out["L0->L12 same seed (raw vs CINDER mapped back)"] = summ(same[(same.a == "L0") & (same.b == "L12")], "L0->L12")
    out["L0->L0 across seeds (EMBER seed noise)"] = summ(cross[(cross.a == "L0") & (cross.b == "L0")], "L0->L0 x-seed")
    out["L0->rnP same seed (raw RS vs raw GS)"] = summ(same[(same.a == "L0") & (same.b == "rnP")], "L0->rnP")
    out["rnP->rnP across seeds"] = summ(cross[(cross.a == "rnP") & (cross.b == "rnP")], "rnP->rnP x-seed")
    # paired: is GS-vs-RS (same seed) closer than RS-vs-RS (other seed)?
    a = same[(same.a == "L12") & (same.b == "L1")].groupby("concept").best_cos.mean()
    b = cross[(cross.a == "L12") & (cross.b == "L12")].groupby("concept").best_cos.mean()
    out["paired_GSvsRS_minus_GS_seed_noise"] = wilcoxon((a - b).dropna())
    D = pd.concat(rows)
    D.round(4).to_csv(OUT / "directions_per_concept.csv", index=False)
    return out, D


def part5_gs_needs_cs(F: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Own concept feature (max-LO) of each fit: width, neutral-only share (N), concept-only (C), page-enriched (X20)."""
    own = F[F.which == "own"]
    per = own.groupby(["concept", "config"]).agg(width=("width", "mean"), N=("sup_N", "mean"), C=("sup_C", "mean"),
                                                 X20=("sup_X20", "mean"), lo=("lo", "mean"),
                                                 P=("hand", lambda h: np.mean([v == "P" for v in h if isinstance(v, str)])
                                                    if any(isinstance(v, str) for v in h) else np.nan),
                                                 labeled=("hand", lambda h: np.mean([isinstance(v, str) for v in h])),
                                                 extra_N=("extra_N", "mean"), head_N=("head_N", "mean"),
                                                 extra_X20=("extra_X20", "mean")).reset_index()
    Wd = per.pivot(index="concept", columns="config")
    out = {"means": {cfg: {k: float(Wd[k][cfg].mean()) for k in ("width", "N", "C", "X20", "lo", "P", "labeled")}
                     for cfg in ("L0", "L1", "L12", "rnP")}}
    out["neutral_share_range"] = {cfg: [float(Wd["N"][cfg].min()), float(Wd["N"][cfg].max())] for cfg in ("L0", "L1", "L12", "rnP")}
    for a, b in (("rnP", "L12"), ("rnP", "L0"), ("L12", "L1"), ("L1", "L0"), ("L12", "L0")):
        out[f"N: {a} - {b}"] = wilcoxon(Wd["N"][a] - Wd["N"][b])
        out[f"X20: {a} - {b}"] = wilcoxon(Wd["X20"][a] - Wd["X20"][b])
    # extra width (beyond the RS width) vs head, within GS supports wider than RS
    for cfg in ("rnP", "L12"):
        g = own[(own.config == cfg) & own.extra_N.notna()]
        pc = g.groupby("concept")[["extra_N", "head_N", "extra_X20"]].mean()
        out[f"extra_vs_head_N_{cfg}"] = {"runs": int(len(g)), "concepts": int(len(pc)),
                                          "extra_N_mean": float(pc.extra_N.mean()), "head_N_mean": float(pc.head_N.mean()),
                                          "extra_X20_mean": float(pc.extra_X20.mean()),
                                          "paired": wilcoxon(pc.extra_N - pc.head_N)}
    # does raw GS width track the natural tail in the raw space? (anchor direction: rnP / L0 best matches)
    A = F[F.which == "anchor"]
    T = concept_table(A)
    Tr = pd.DataFrame({"w_L12": T.w_rnP_match, "m_rs": T.m_rs, "T_L0": T.T_L0, "T_L1": T.T_L1})
    out["raw_tracking_T_L0"] = tracking_stats(Tr, "T_L0")
    out["raw_tracking_T_L1"] = tracking_stats(Tr, "T_L1")
    out["note_raw_tracking"] = "w_L12 here = raw GS (rnP) width of the anchor's best match; m_rs = RS-1% width"
    per.round(4).to_csv(OUT / "gs_needs_cs_per_concept.csv", index=False)
    return out, per


def mdt(df: pd.DataFrame, fmt: int = 3) -> str:
    def f(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.{fmt}f}"
        return str(v)
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main():
    F, R, X, C, curves = assemble()
    F.to_csv(OUT / "features.csv", index=False)
    R.drop(columns=["widths"]).to_csv(OUT / "runs.csv", index=False)
    X.to_csv(OUT / "cross_match.csv", index=False)
    C.to_csv(OUT / "concepts.csv", index=False)
    pd.to_pickle(curves, OUT / "curves.pkl")
    res = {"primary_tail": PRIMARY_COL, "n_concepts": int(C.shape[0])}
    res["part1"] = part1_seeds(F, R, X)
    res["part2"], T = part2_dense(F, R, C)
    res["part3"] = part3_breadth(T, F)
    res["part4"], _ = part4_directions(X, R)
    res["part5"], _ = part5_gs_needs_cs(F)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    cols = ["m_rs", "w_L12", "w_L12_sd", "w_RS2", "T_L1", "T_L12", "T_RS2", "T_L0", "rs_class", "n_C", "n_X20",
            "n_lex_words_SC", "n_target"]
    (OUT / "tables.md").write_text("# WP3 tables (generated by analyze.py)\n\n## Per concept (anchor direction, means over 5 seeds)\n\n"
                                   + mdt(T[cols].reset_index().sort_values("w_L12"), 1) + "\n")
    print(json.dumps(res, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
