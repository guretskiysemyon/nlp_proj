"""WP7 analysis: tables, paired tests and figures from the fits' stored metrics.

    python -m experiments.paper_analysis.circularity.analyze

Unit = concept (metrics averaged over seeds first). Each target vs raw: paired Wilcoxon signed-rank (two-sided,
zero differences dropped), bootstrap 95% CI of the mean difference (10,000 resamples of concepts, seed 0);
Holm over the primary family (rho_rank, purity_strict) x 6 targets; retention = mean(T - raw) / mean(page - raw).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.circularity.common import (DEFAULT_ETA, DEFAULT_GAMMA, ETA_GRID, FIGS, FITS, GAMMA_GRID,
                                                         OUT, PAPER_CONCEPTS, TARGETS, TEXTS, TUNING_TRIO, read_json,
                                                         slug, write_json)

PRIMARY = ["rho_rank", "purity_strict"]
SECONDARY = ["log2_rank", "purity_C", "purity_strict_any", "purity_lexonly", "lex_top20", "lex_tok", "auc_C", "auc_S",
             "auc_tau", "log_dom", "found5", "found2", "hits"]
LOWER_BETTER = {"rho_rank", "log2_rank"}
ORDER = ["page", "disjoint", "related", "pagesub", "leakfree", "wrong"]
LABEL = {"raw": "raw (EMBER)", "page": "page", "disjoint": "disjoint sentences", "related": "related pages",
         "pagesub": "page subset (size ctrl)", "leakfree": "leak-free", "wrong": "wrong concept"}
COLOR = {"raw": "#7F7F7F", "page": "#0072B2", "disjoint": "#009E73", "related": "#009E73", "pagesub": "#56B4E9",
         "leakfree": "#CC79A7", "wrong": "#000000"}
MARKER = {"page": "o", "disjoint": "s", "related": "D", "pagesub": "o", "leakfree": "^", "wrong": "v"}
RNG_SEED, N_BOOT = 0, 10_000


# ---------------------------------------------------------------- loading
def load_rows() -> pd.DataFrame:
    rows = []
    for p in sorted(FITS.glob("*/*/seed*.pt")):
        d = torch.load(p, weights_only=False, map_location="cpu")
        m = {k: v for k, v in d["metrics"].items() if not isinstance(v, list)}
        rows.append({"concept": d["concept"], "cond": p.parent.name, "target": d["target"], "eta": d["eta"],
                     "gamma": d["gamma"], "method": d["method"], "seed": d["seed"], "rel_err": d["rel_err"],
                     "n_iter": d["n_iter"], "n_target": d["diag"].get("n_target"), **m})
    df = pd.DataFrame(rows)
    df["log2_rank"] = np.log2(df["rho_rank"])
    for c in ("found5", "found2"):
        df[c] = df[c].astype(float)
    return df


# ---------------------------------------------------------------- statistics
def boot_ci(x: np.ndarray, fn=np.mean, n=N_BOOT, seed=RNG_SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    b = fn(x[idx], axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def wilcoxon_p(d: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.all(d == 0):
        return float("nan")
    return float(wilcoxon(d).pvalue)


def holm(pvals: dict) -> dict:
    items = sorted((p, k) for k, p in pvals.items() if not math.isnan(p))
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def paired(per: pd.DataFrame, base: str, cond: str, metric: str) -> dict:
    """per: concept x cond table of seed-mean values for one metric."""
    a = per[cond] - per[base]
    a = a.dropna()
    lo, hi = boot_ci(a.values)
    return {"n": int(len(a)), "mean_base": float(per.loc[a.index, base].mean()),
            "mean_cond": float(per.loc[a.index, cond].mean()), "median_base": float(per.loc[a.index, base].median()),
            "median_cond": float(per.loc[a.index, cond].median()), "mean_diff": float(a.mean()),
            "median_diff": float(a.median()), "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": wilcoxon_p(a.values),
            "n_better": int(((a < 0) if metric in LOWER_BETTER else (a > 0)).sum()),
            "n_worse": int(((a > 0) if metric in LOWER_BETTER else (a < 0)).sum())}


def retention(per: pd.DataFrame, cond: str) -> dict:
    d_t = (per[cond] - per["raw"]).values
    d_p = (per["page"] - per["raw"]).values
    ok = ~(np.isnan(d_t) | np.isnan(d_p))
    d_t, d_p = d_t[ok], d_p[ok]
    r = d_t.mean() / d_p.mean() if d_p.mean() != 0 else float("nan")
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(d_t), size=(N_BOOT, len(d_t)))
    with np.errstate(divide="ignore", invalid="ignore"):
        b = d_t[idx].mean(1) / d_p[idx].mean(1)
    return {"retention": float(r), "ci_lo": float(np.nanpercentile(b, 2.5)), "ci_hi": float(np.nanpercentile(b, 97.5))}


def seed_mean(df: pd.DataFrame, metric: str, key="target") -> pd.DataFrame:
    return df.groupby(["concept", key])[metric].mean().unstack(key)


# ---------------------------------------------------------------- main study
def main_study(df: pd.DataFrame, method: str = "fixed") -> dict:
    sub = df[(df.method == method) & ((df.target == "raw") | ((df.eta == DEFAULT_ETA) & (df.gamma == DEFAULT_GAMMA)))]
    conds = ["raw"] + [t for t in ORDER if t in set(sub.target)]
    res = {"method": method, "n_runs": {t: int((sub.target == t).sum()) for t in conds},
           "n_concepts": int(sub.concept.nunique()), "tests": {}, "retention": {}, "summary": {}, "heldout": {}}
    for metric in PRIMARY + SECONDARY:
        per = seed_mean(sub, metric)
        res["summary"][metric] = {t: {"mean": float(per[t].mean()), "median": float(per[t].median())}
                                  for t in conds if t in per}
        for t in conds[1:]:
            res["tests"][f"{metric}|{t}"] = paired(per, "raw", t, metric)
        if metric in PRIMARY + ["purity_C", "auc_S", "lex_top20", "log2_rank"]:
            for t in conds[2:]:
                res["retention"][f"{metric}|{t}"] = retention(per, t)
    hp = holm({k: v["p_wilcoxon"] for k, v in res["tests"].items() if k.split("|")[0] in PRIMARY})
    for k, v in hp.items():
        res["tests"][k]["p_holm"] = v
    # held-out tokens: tokens that are NOT types of target T, condition T vs raw on the same tokens
    for t in conds[1:]:
        for mm in ("auc_S", "auc_C", "purity_strict"):
            col = f"out_{t}_{mm}"
            if col not in sub:
                continue
            per = sub[sub.target.isin(["raw", t])].groupby(["concept", "target"])[col].mean().unstack("target")
            if t in per:
                res["heldout"][f"{mm}|{t}"] = paired(per, "raw", t, mm)
    # also: does the page target help the tokens that the related / disjoint targets do not contain?
    for t in ("related", "disjoint"):
        col = f"out_{t}_auc_S"
        per = sub[sub.target.isin(["raw", "page", t])].groupby(["concept", "target"])[col].mean().unstack("target")
        res["heldout"][f"auc_S|page_on_not_{t}"] = paired(per, "raw", "page", "auc_S")
    return res


def per_concept_table(df: pd.DataFrame, method: str = "fixed") -> pd.DataFrame:
    sub = df[(df.method == method) & ((df.target == "raw") | ((df.eta == DEFAULT_ETA) & (df.gamma == DEFAULT_GAMMA)))]
    cols = ["rho_rank", "purity_strict", "purity_C", "auc_S", "lex_top20", "found5"]
    t = sub.groupby(["concept", "target"])[cols].mean().unstack("target")
    return t


def composition() -> dict:
    """How much of each label set each target contains (labels_summary.json from labels.py)."""
    rows = read_json(OUT / "labels_summary.json")
    sizes = {r["concept"]: r["sizes"] for r in read_json(TEXTS / "targets_summary.json")}
    out = {}
    for t in ORDER:
        out[t] = {"n_types_median": float(np.median([sizes[r["concept"]][t] for r in rows])),
                  "n_types_min": int(min(sizes[r["concept"]][t] for r in rows)),
                  "n_types_max": int(max(sizes[r["concept"]][t] for r in rows))}
        for k in ("C", "strict", "N"):
            v = [r[f"{k}_in_{t}"] for r in rows if r[f"{k}_in_{t}"] is not None]
            out[t][f"{k}_in_mean"] = float(np.mean(v))
    return out


def dose_response(df: pd.DataFrame, method: str = "fixed") -> dict:
    """Across (concept, target) pairs: gain over raw vs the share of the strict tokens the target contains."""
    from scipy.stats import spearmanr
    lab = {r["concept"]: r for r in read_json(OUT / "labels_summary.json")}
    sub = df[(df.method == method) & ((df.target == "raw") | ((df.eta == DEFAULT_ETA) & (df.gamma == DEFAULT_GAMMA)))]
    out, rows = {}, []
    for metric in ("purity_strict", "log2_rank", "auc_S"):
        per = seed_mean(sub, metric)
        for t in [x for x in ORDER if x in per]:
            for c in per.index:
                if np.isnan(per.loc[c, t]) or np.isnan(per.loc[c, "raw"]):
                    continue
                rows.append({"metric": metric, "concept": c, "target": t, "diff": float(per.loc[c, t] - per.loc[c, "raw"]),
                             "strict_in": lab[c][f"strict_in_{t}"], "C_in": lab[c][f"C_in_{t}"]})
        r = pd.DataFrame([x for x in rows if x["metric"] == metric])
        rr = spearmanr(r["strict_in"], r["diff"])
        nz = r[r.target != "leakfree"]
        rr2 = spearmanr(nz["strict_in"], nz["diff"])
        on = r[r.target.isin(["page", "disjoint", "related", "pagesub"])]
        rr3 = spearmanr(on["strict_in"], on["diff"])
        # within concept: rank the 4 on-topic targets by strict_in and by diff, average the per-concept Spearman
        wc = [spearmanr(g["strict_in"], g["diff"]).statistic for _, g in on.groupby("concept")
              if g["strict_in"].nunique() > 1 and g["diff"].nunique() > 1]
        out[metric] = {"spearman_all": float(rr.statistic), "p_all": float(rr.pvalue), "n_all": int(len(r)),
                       "spearman_no_leakfree": float(rr2.statistic), "p_no_leakfree": float(rr2.pvalue),
                       "spearman_on_topic": float(rr3.statistic), "p_on_topic": float(rr3.pvalue),
                       "n_on_topic": int(len(on)), "within_concept_mean_spearman_on_topic": float(np.nanmean(wc)),
                       "within_concept_n": int(len(wc))}
    pd.DataFrame(rows).to_csv(OUT / f"dose_response_{method}.csv", index=False)
    return out


# ---------------------------------------------------------------- eta x gamma
EG_METRICS = ["rho_rank", "purity_strict", "purity_C", "auc_S", "lex_top20"]


def _ranks(vals, lower_better):
    from scipy.stats import rankdata
    v = np.asarray(vals, float)
    return rankdata(v if lower_better else -v)


def etagamma(df: pd.DataFrame, target: str = "page") -> dict:
    sub = df[(df.method == "fixed") & (df.target == target) & (~df.concept.isin(TUNING_TRIO))].copy()
    sub["cfg"] = sub.apply(lambda r: f"{r.eta:g}|{r.gamma:g}", axis=1)
    cfgs = [f"{e:g}|{g:g}" for e in ETA_GRID for g in GAMMA_GRID]
    have = sub.groupby("cfg").concept.nunique()
    cfgs = [c for c in cfgs if c in have.index]
    concepts = sorted(set.intersection(*[set(sub[sub.cfg == c].concept) for c in cfgs]))
    sub = sub[sub.concept.isin(concepts)]
    raw = df[(df.method == "fixed") & (df.target == "raw") & df.concept.isin(concepts)]
    out = {"concepts": concepts, "n_concepts": len(concepts), "seeds": sorted(sub.seed.unique().tolist()),
           "grid": {}, "mean_rank": {}, "default_vs_best": {}}
    rank_acc = {c: [] for c in cfgs}
    for metric in EG_METRICS:
        per = sub.groupby(["concept", "cfg"])[metric].mean().unstack("cfg")[cfgs]
        out["grid"][metric] = {c: {"mean": float(per[c].mean()), "median": float(per[c].median())} for c in cfgs}
        out["grid"][metric]["raw"] = {"mean": float(raw.groupby("concept")[metric].mean().mean()),
                                      "median": float(raw.groupby("concept")[metric].mean().median())}
        for concept, row in per.iterrows():
            for c, r in zip(cfgs, _ranks(row.values, metric in LOWER_BETTER)):
                rank_acc[c].append(r)
        agg = per.median() if metric in LOWER_BETTER else per.mean()
        best = (agg.idxmin() if metric in LOWER_BETTER else agg.idxmax())
        dflt = f"{DEFAULT_ETA:g}|{DEFAULT_GAMMA:g}"
        d = per[dflt] - per[best]
        lo, hi = boot_ci(d.values)
        out["default_vs_best"][metric] = {"best": best, "default_value": float(agg[dflt]), "best_value": float(agg[best]),
                                          "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                                          "p_wilcoxon": wilcoxon_p(d.values),
                                          "default_rank_of_9": int(_ranks(agg.values, metric in LOWER_BETTER)[
                                              cfgs.index(dflt)])}
    out["mean_rank"] = {c: float(np.mean(v)) for c, v in rank_acc.items()}
    mr = pd.Series(out["mean_rank"]).sort_values()
    out["mean_rank_order"] = mr.index.tolist()
    return out


# ---------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    return plt


def fig_paired(df: pd.DataFrame, method: str = "fixed", name: str = "fig_targets_paired"):
    plt = setup_mpl()
    sub = df[(df.method == method) & ((df.target == "raw") | ((df.eta == DEFAULT_ETA) & (df.gamma == DEFAULT_GAMMA)))]
    targets = [t for t in ORDER if t in set(sub.target)]
    panels = [("log2_rank", r"$\Delta\log_2$ $\rho$-rank of $f^*$" + "\n(< 0: concept ranks higher)"),
              ("purity_strict", r"$\Delta$ strict purity of $f^*$"),
              ("auc_S", r"$\Delta$ AUC strict vs neutral" + "\n(all tokens)"),
              ("heldout_auc_S", r"$\Delta$ AUC strict vs neutral" + "\n(tokens not in the target)")]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.35), sharey=True)
    data_out = []
    rng = np.random.default_rng(0)
    for ax, (metric, xl) in zip(axes, panels):
        for yi, t in enumerate(targets):
            if metric == "heldout_auc_S":
                col = f"out_{t}_auc_S"
                per = sub[sub.target.isin(["raw", t])].groupby(["concept", "target"])[col].mean().unstack("target")
            else:
                per = seed_mean(sub, metric)
            d = (per[t] - per["raw"]).dropna()
            y = len(targets) - 1 - yi
            ax.scatter(d.values, y + rng.uniform(-0.18, 0.18, len(d)), s=7, color=COLOR[t], alpha=0.35, lw=0,
                       marker=MARKER[t], zorder=2)
            lo, hi = boot_ci(d.values)
            ax.plot([lo, hi], [y, y], color=COLOR[t], lw=2, solid_capstyle="round", zorder=3)
            ax.scatter([d.mean()], [y], s=28, color=COLOR[t], edgecolor="white", lw=0.8, marker=MARKER[t], zorder=4)
            for cpt, v in d.items():
                data_out.append({"panel": metric, "target": t, "concept": cpt, "diff_vs_raw": float(v)})
            data_out.append({"panel": metric, "target": t, "concept": "MEAN", "diff_vs_raw": float(d.mean()),
                             "ci_lo": lo, "ci_hi": hi})
        ax.axvline(0, color="#7F7F7F", lw=0.7, ls="--", zorder=1)
        ax.set_xlabel(xl)
        ax.grid(axis="x", color="#E5E5E5", lw=0.5)
    axes[0].set_yticks(range(len(targets)))
    axes[0].set_yticklabels([LABEL[t] for t in reversed(targets)])
    fig.tight_layout(w_pad=0.6)
    save(fig, name, pd.DataFrame(data_out))


def fig_etagamma(eg: dict, df: pd.DataFrame, name: str = "fig_etagamma_heatmap"):
    plt = setup_mpl()
    from matplotlib.patches import Rectangle
    panels = [("rho_rank", "median", r"median $\rho$-rank of $f^*$", "Blues_r"),
              ("purity_strict", "mean", "mean strict purity", "Blues"),
              ("auc_S", "mean", "mean AUC strict vs neutral", "Blues"),
              ("mean_rank", None, "mean rank over 5 metrics\n(1 = best of 9)", "Blues_r")]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.0))
    rows = []
    for ax, (metric, stat, title, cmap) in zip(axes, panels):
        M = np.full((len(ETA_GRID), len(GAMMA_GRID)), np.nan)
        for i, e in enumerate(ETA_GRID):
            for j, g in enumerate(GAMMA_GRID):
                c = f"{e:g}|{g:g}"
                v = eg["mean_rank"].get(c) if metric == "mean_rank" else eg["grid"][metric].get(c, {}).get(stat)
                M[i, j] = np.nan if v is None else v
                rows.append({"panel": metric, "eta": e, "gamma": g, "value": M[i, j]})
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        vmin, vmax = np.nanmin(M), np.nanmax(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                frac = (v - vmin) / (vmax - vmin + 1e-12)
                dark = frac > 0.55 if not cmap.endswith("_r") else frac < 0.45
                ax.text(j, i, f"{v:.1f}" if metric in ("rho_rank", "mean_rank") else f"{v:.2f}", ha="center",
                        va="center", fontsize=7, color="white" if dark else "#222222")
        di, dj = ETA_GRID.index(DEFAULT_ETA), GAMMA_GRID.index(DEFAULT_GAMMA)
        ax.add_patch(Rectangle((dj - 0.5, di - 0.5), 1, 1, fill=False, ec="#D55E00", lw=1.6))
        ax.set_xticks(range(len(GAMMA_GRID)), [f"{g:g}" for g in GAMMA_GRID])
        ax.set_yticks(range(len(ETA_GRID)), [f"{e:g}" for e in ETA_GRID])
        ax.set_xlabel(r"$\gamma$")
        if ax is axes[0]:
            ax.set_ylabel(r"$\eta$")
        ax.set_title(title, fontsize=7.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        if metric != "mean_rank":
            raw = eg["grid"][metric]["raw"][stat]
            ax.set_title(title + (f"\n(raw: {raw:.1f})" if metric == "rho_rank" else f"\n(raw: {raw:.2f})"),
                         fontsize=7.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, name, pd.DataFrame(rows))


def fig_dose(method: str = "fixed", name: str = "fig_dose_response"):
    """Gain over raw (AUC strict vs neutral, dense f* code) vs the share of strict tokens the target contains."""
    plt = setup_mpl()
    r = pd.read_csv(OUT / f"dose_response_{method}.csv")
    r = r[r.metric == "auc_S"]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for t in ORDER:
        g = r[r.target == t]
        ax.scatter(g["strict_in"], g["diff"], s=14, color=COLOR[t], marker=MARKER[t], lw=0.4,
                   edgecolor="white", label=LABEL[t], alpha=0.85, zorder=3)
    ax.axhline(0, color="#7F7F7F", lw=0.7, ls="--", zorder=1)
    ax.set_xlabel("share of the concept's strict tokens in the target")
    ax.set_ylabel("$\\Delta$ AUC strict vs neutral (target $-$ raw)")
    ax.grid(color="#E5E5E5", lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, name, r)


def save(fig, name, data: pd.DataFrame):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=200, bbox_inches="tight")
    data.to_csv(FIGS / f"{name}.csv", index=False)
    import matplotlib.pyplot as plt
    plt.close(fig)


def examples(concepts=("Harry Potter", "Ancient Rome", "Republic of Ireland", "Golf"), seed=42,
             method="fixed") -> dict:
    """Top-20 source words of f* per condition (seed 42), for the appendix."""
    from experiments.paper_analysis.circularity.common import cond_name, fit_path
    out = {}
    for c in concepts:
        out[c] = {}
        for t in ["raw"] + ORDER:
            p = fit_path(c, cond_name(t, DEFAULT_ETA, DEFAULT_GAMMA, method), seed)
            if not p.exists():
                continue
            m = torch.load(p, weights_only=False, map_location="cpu")["metrics"]
            out[c][t] = {"rho_rank": m["rho_rank"], "hits": m["hits"], "purity_strict": round(m["purity_strict"], 3),
                         "words": m["top20_words"], "tokens": m["top20_tokens"]}
    return out


# ---------------------------------------------------------------- run
def main():
    df = load_rows()
    df.to_csv(OUT / "fits_metrics.csv", index=False)
    print(f"{len(df)} fits loaded", flush=True)
    out = {}
    for method in ("fixed", "rn"):
        if ((df.method == method) & (df.target != "raw")).any() and ((df.method == method) & (df.target == "raw")).any():
            out[f"main_{method}"] = main_study(df, method)
            out[f"dose_{method}"] = dose_response(df, method)
            fig_dose(method, "fig_dose_response" + ("" if method == "fixed" else "_gs"))
            per_concept_table(df, method).to_csv(OUT / f"per_concept_{method}.csv")
            fig_paired(df, method, "fig_targets_paired" + ("" if method == "fixed" else "_gs"))
    if ((df.target == "page") & (df.eta != DEFAULT_ETA)).any():
        out["etagamma"] = etagamma(df)
        fig_etagamma(out["etagamma"], df)
    if ((df.target == "related") & (df.eta != DEFAULT_ETA)).any():
        out["etagamma_related"] = etagamma(df, "related")
        fig_etagamma(out["etagamma_related"], df, "fig_etagamma_heatmap_related")
    tsum = read_json(TEXTS / "targets_summary.json") if (TEXTS / "targets_summary.json").exists() else []
    out["targets"] = tsum
    out["composition"] = composition()
    write_json(OUT / "examples_seed42.json", examples())
    write_json(OUT / "analysis.json", out)
    return out


if __name__ == "__main__":
    main()
