"""Aggregates, tests and the figure for WP9 (reads features.csv.gz, fits.csv, top20.csv, energy.json, cost.json).

    python -m experiments.paper_analysis.ml_geometry.report

Writes summary.json, coverage_2x2.csv, per_concept.csv, fig_ml_language.{pdf,png} (+ fig_ml_language_data.json).
Concept is the unit for the tests (values averaged over the 3 fit seeds first); Wilcoxon signed-rank, bootstrap 95% CI.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.paper_analysis.ml_geometry.common import COLORS, DOM, OUT, concepts, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
C_A1, C_A2, C_A3 = "#009E73", "#7F7F7F", "#D55E00"


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(np.mean(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    return {"n": len(d), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": boot(d),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p_wilcoxon": p}


def main():
    df = pd.read_csv(OUT / "features.csv.gz", low_memory=False)
    fits = pd.read_csv(OUT / "fits.csv")
    df["tj"] = df["tj"].astype("boolean").fillna(False).astype(bool)
    df["dom"] = df.maxshare >= DOM
    df["acc_tj"] = df.candidate & df.tj
    df["acc_ej"] = df.candidate & df.ej5
    df["acc_twj"] = df.acc_tj | df.acc_ej
    ml = df[df.approach.isin(["A2", "A3"])]
    S = {}
    # ---- Q1a: language-dominated features
    allf = ml.groupby("approach").agg(n=("feature", "size"), frac_dom=("dom", "mean"), med_maxshare=("maxshare", "median"),
                                      med_H=("H", "median")).round(4)
    cand = ml[ml.candidate].groupby("approach").agg(n=("feature", "size"), frac_dom=("dom", "mean"),
                                                    med_maxshare=("maxshare", "median"), tj=("tj", "mean"),
                                                    ej5=("ej5", "mean")).round(4)
    S["all_features"] = allf.to_dict("index")
    S["candidates"] = cand.to_dict("index")
    S["null"] = fits[fits.approach.isin(["A2", "A3"])].groupby("approach")[["null_frac_dom", "null_med_maxshare"]] \
        .mean().round(4).to_dict("index")
    S["dom_maxlang_A2_candidates"] = ml[ml.candidate & ml.dom & (ml.approach == "A2")].maxlang.value_counts().to_dict()
    S["dom_maxlang_A2_all"] = ml[ml.dom & (ml.approach == "A2")].maxlang.value_counts().to_dict()
    S["dom_maxlang_A3_all"] = ml[ml.dom & (ml.approach == "A3")].maxlang.value_counts().to_dict()
    pc = ml.groupby(["concept", "approach"]).agg(frac_dom=("dom", "mean")).unstack()
    pcc = ml[ml.candidate].groupby(["concept", "approach"]).agg(frac_dom=("dom", "mean")).unstack()
    S["test_frac_dom_all_A2_vs_A3"] = paired(pc[("frac_dom", "A2")], pc[("frac_dom", "A3")])
    S["test_frac_dom_cand_A2_vs_A3"] = paired(pcc[("frac_dom", "A2")], pcc[("frac_dom", "A3")])
    # ---- Q1b: does a concept feature exist, where does it rank
    per_fit = ml.groupby(["concept", "seed", "approach"]).apply(lambda x: pd.Series({
        "ej5_any": bool(x.ej5.any()), "ej5_cand": bool((x.ej5 & x.candidate).any()),
        "best_ej5_rank": x[x.ej5].rho_rank.min() if x.ej5.any() else np.nan,
        "selected": bool(x.selected.any()), "n_tau": int((x.rho >= 2).sum())}), include_groups=False).reset_index()
    S["per_fit_means"] = per_fit.groupby("approach")[["ej5_any", "ej5_cand", "selected", "n_tau"]].mean().round(3) \
        .to_dict("index")
    S["best_ej5_rank_median"] = per_fit.groupby("approach").best_ej5_rank.median().to_dict()
    S["ej5_counts_by_seed"] = per_fit.groupby(["approach", "seed"])[["ej5_any", "ej5_cand", "selected"]].sum() \
        .reset_index().to_dict("records")
    # ---- coverage 2x2 (fit x judge), concepts with >= 1 accepted candidate
    g = ml.groupby(["concept", "seed", "approach"])[["acc_tj", "acc_twj"]].any().unstack("approach")
    cov = pd.DataFrame({"raw+TJ (A2 as run)": g[("acc_tj", "A2")], "raw+TWJ": g[("acc_twj", "A2")],
                        "CS+GS+TJ": g[("acc_tj", "A3")], "CS+GS+TWJ (A3 as run)": g[("acc_twj", "A3")]}).reset_index()
    cov.to_csv(OUT / "coverage_2x2.csv", index=False)
    cols = [c for c in cov.columns if c not in ("concept", "seed")]
    S["coverage_2x2_by_seed"] = cov.groupby("seed")[cols].sum().astype(int).to_dict("index")
    S["coverage_2x2_union"] = cov.groupby("concept")[cols].any().sum().astype(int).to_dict()
    pcov = cov.groupby("concept")[cols].sum()
    S["test_cov_TWJ_cs_vs_raw"] = paired(pcov["CS+GS+TWJ (A3 as run)"], pcov["raw+TWJ"])
    S["test_cov_TJ_cs_vs_raw"] = paired(pcov["CS+GS+TJ"], pcov["raw+TJ (A2 as run)"])
    S["test_cov_raw_TWJ_vs_TJ"] = paired(pcov["raw+TWJ"], pcov["raw+TJ (A2 as run)"])
    S["test_cov_A3_vs_A2_as_run"] = paired(pcov["CS+GS+TWJ (A3 as run)"], pcov["raw+TJ (A2 as run)"])
    # ---- TJ acceptance by lexicon evidence
    c = df[df.candidate].copy()
    c["band"] = pd.cut(c.ej_hits, [-1, 0, 2, 4, 7, 20], labels=["0", "1-2", "3-4", "5-7", "8+"])
    tjb = c.groupby(["approach", "band"], observed=False).agg(n=("tj", "size"), tj=("tj", "mean")).reset_index()
    S["tj_by_hits"] = tjb.round(3).to_dict("records")
    # top-20 composition (what the judge sees)
    try:
        t20 = pd.read_csv(OUT / "top20.csv")
        cc = c[c.approach.isin(["A2", "A3"])].merge(t20, on=["concept", "approach", "seed", "feature"])
        S["top20_by_band"] = cc.groupby(["approach", "band"], observed=False)[
            ["t20_conc_only", "t20_en", "t20_latin", "t20_maxshare"]].mean().round(3).reset_index().to_dict("records")
        S["top20_all_cand"] = cc.groupby("approach")[["t20_conc_only", "t20_en", "t20_latin", "t20_maxshare"]] \
            .mean().round(3).to_dict("index")
        S["top20_ej3plus"] = cc[cc.ej_hits >= 3].groupby("approach")[["t20_conc_only", "t20_en", "t20_maxshare"]] \
            .mean().round(3).to_dict("index")
    except FileNotFoundError:
        pass
    # ---- Q2: language spread of concept features
    df["ll"] = df.lex_langs_list.fillna("")
    df["nonlat_lex"] = df.ll.str.contains("hi") | df.ll.str.contains("th")
    q2 = df[df.selected].groupby("approach").agg(
        n=("feature", "size"), med_maxshare=("maxshare", "median"), med_H=("H", "median"),
        n_lang10=("n_lang10", "mean"), multi_script=("n_script5", lambda s: (s >= 2).mean()),
        three_script=("n_script5", lambda s: (s >= 3).mean()), nonlatin_lexicon_word=("nonlat_lex", "mean"),
        deva_thai_mass=("s_deva", "mean"), thai_mass=("s_thai", "mean"), q_en=("q_en", "mean"),
        width=("width", "median")).round(3)
    q2["deva_thai_mass"] = (df[df.selected].groupby("approach").s_deva.mean()
                            + df[df.selected].groupby("approach").s_thai.mean()).round(3)
    S["q2_selected"] = q2.drop(columns="thai_mass").to_dict("index")
    sel3 = df[df.selected & (df.approach.isin(["A2", "A3"]))].groupby(["concept", "approach"]).maxshare.mean().unstack()
    sel3 = sel3.dropna()
    S["test_selected_maxshare_A3_vs_A2"] = paired(sel3["A3"], sel3["A2"])
    # ---- blind labels of EJ-only accepts
    try:
        b = pd.read_csv(OUT / "blind" / "unblinded.csv")
        S["blind_ej_only"] = pd.crosstab(b.approach, b.label).to_dict("index")
    except FileNotFoundError:
        pass
    # ---- energy
    try:
        en = json.loads((OUT / "energy.json").read_text())
        E = pd.DataFrame([{"concept": r["concept"], **{f"{m}_{k}": r[m][k] for m in ("raw", "white_only", "cs")
                                                         for k in ("lang_between", "conc_between", "lang_proj",
                                                                   "pc_eta_lang", "pc_eta_conc")},
                           "en_raw_conc_between": r["en_raw_conc_between"], "shared_frac": r["shared_frac"],
                           "g_en": r["cpca_en"]["g_top"][0], "g_ml": r["cpca_ml"]["g_top"][0],
                           "pur_en": r["cpca_en"]["purity_mean"], "pur_ml": r["cpca_ml"]["purity_mean"]} for r in en])
        E["raw_ratio"] = E.raw_lang_between / E.raw_conc_between
        E["cs_ratio"] = E.cs_lang_between / E.cs_conc_between
        E.to_csv(OUT / "energy_per_concept.csv", index=False)
        S["energy_n"] = len(E)
        S["energy_median"] = E.drop(columns="concept").median().round(5).to_dict()
        S["test_lang_between_raw_vs_cs"] = paired(E.raw_lang_between, E.cs_lang_between)
        S["test_g_en_vs_ml"] = paired(E.g_en, E.g_ml)
        S["test_conc_between_en_vs_ml_raw"] = paired(E.en_raw_conc_between, E.raw_conc_between)
        hp = [r for r in en if r["concept"] == "Harry Potter"][0]
        S["energy_HP"] = {"cpca_en": hp["cpca_en"], "cpca_ml": hp["cpca_ml"]}
    except FileNotFoundError:
        pass
    S["cost"] = json.loads((OUT / "cost.json").read_text())["totals"]
    save_json(OUT / "summary.json", S)
    # per-concept table
    pcon = pd.concat([pc.droplevel(0, axis=1).add_prefix("frac_dom_all_"),
                      pcc.droplevel(0, axis=1).add_prefix("frac_dom_cand_"), pcov], axis=1).round(3)
    pcon.to_csv(OUT / "per_concept.csv")
    figure(ml, fits, cov, tjb)
    print(json.dumps({k: S[k] for k in ("all_features", "candidates", "coverage_2x2_by_seed", "coverage_2x2_union",
                                        "test_cov_TWJ_cs_vs_raw", "test_cov_TJ_cs_vs_raw")}, indent=1, default=str))


def figure(ml, fits, cov, tjb):
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.15), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    data = {}
    # (a) ECDF of per-feature max-language share
    a = ax[0]
    bins = np.linspace(0.1, 1.0, 46)
    for appr, col, lab in (("A2", C_A2, "ML-EMBER (A2)"), ("A3", C_A3, "CINDER (A3)")):
        x = ml[ml.approach == appr].maxshare.values
        h, _ = np.histogram(x, bins=bins)
        a.step(bins[:-1], h / len(x), where="post", color=col, lw=1.1, label=lab)
        data[f"hist_{appr}"] = (h / len(x)).round(5).tolist()
        sel = ml[(ml.approach == appr) & ml.selected].maxshare.values
        yy = -0.012 if appr == "A2" else -0.024
        a.plot(sel, np.full_like(sel, yy), "|", color=col, ms=4, mew=0.8)
        data[f"selected_{appr}"] = sel.round(4).tolist()
    a.axvline(DOM, color="k", lw=0.6, ls=":")
    a.text(DOM + 0.01, a.get_ylim()[1] * 0.85, "language-\ndominated", fontsize=6.5, va="top")
    a.set_xlabel("max-language share of |Y| mass")
    a.set_ylabel("fraction of features")
    a.set_xlim(0.1, 1.0)
    a.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.72))
    a.text(0.02, 0.98, "(a)", transform=a.transAxes, va="top", fontweight="bold")
    data["bins"] = bins.round(4).tolist()
    # (b) coverage 2x2
    b = ax[1]
    cols = ["raw+TJ (A2 as run)", "raw+TWJ", "CS+GS+TJ", "CS+GS+TWJ (A3 as run)"]
    by_seed = cov.groupby("seed")[cols].sum()
    xs = np.arange(4)
    colors = [C_A2, C_A2, C_A3, C_A3]
    for i, cname in enumerate(cols):
        v = by_seed[cname].values
        b.bar(xs[i], v.mean(), color=colors[i] if "TWJ" in cname else "white", edgecolor=colors[i], lw=1.0, width=0.7)
        b.plot(np.full(len(v), xs[i]) + np.array([-0.15, 0, 0.15])[:len(v)], v, "o", ms=2.2, color="k")
    b.set_xticks(xs, ["raw\nTJ", "raw\nTWJ", "CS\nTJ", "CS\nTWJ"])
    b.set_ylabel("concepts covered (of 38)")
    b.set_ylim(0, 25)
    b.text(0.02, 0.98, "(b)", transform=b.transAxes, va="top", fontweight="bold")
    data["coverage_by_seed"] = by_seed.astype(int).to_dict("index")
    # (c) TJ acceptance by lexicon hits
    c = ax[2]
    bands = ["0", "1-2", "3-4", "5-7", "8+"]
    for appr, col, lab, mk in (("A1", C_A1, "A1", "s"), ("A2", C_A2, "A2", "o"),
                               ("A3", C_A3, "A3", "^")):
        t = tjb[tjb.approach == appr].set_index("band").reindex(bands)
        c.plot(range(5), t.tj.values, marker=mk, ms=3, color=col, lw=1.0, label=lab)
        data[f"tj_{appr}"] = {"rate": t.tj.round(4).tolist(), "n": t.n.astype(int).tolist()}
    c.set_xticks(range(5), bands)
    c.set_xlabel("lexicon words in top 20")
    c.set_ylabel("TJ acceptance rate")
    c.set_ylim(0, 1)
    c.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    c.text(0.02, 0.98, "(c)", transform=c.transAxes, va="top", fontweight="bold")
    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_ml_language.{ext}", dpi=300, bbox_inches="tight")
    save_json(OUT / "fig_ml_language_data.json", data)


if __name__ == "__main__":
    main()
