#!/usr/bin/env python
"""Aggregate WP8 results: tables (CSV + markdown snippets) and figures (PDF + PNG, data next to each figure).

    python -m experiments.paper_analysis.llama_cs.report
Inputs: results/paper_analysis/llama_cs/fits/<model>/<slug>/<cfg>/seed<N>/metrics.json, spectra/spectra.json,
        gemma_ref.json.   Outputs: results/paper_analysis/llama_cs/{tables,figures}/
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.llama_cs.common import COLORS, FITS, OUT, read_json, slug, write_json

FIG = OUT / "figures"
MODEL_COL = {"gemma": "#009E73", "llama": "#CC79A7"}   # models (not methods): Okabe-Ito green / reddish purple
TAB = OUT / "tables"
NOT_FOUND_RANK = None   # cf_rank is reported only for found concept features (lo >= 5); see cf_found


def derived(m: dict) -> dict:
    """Extra judge-free rank metrics from the per-feature rho and lexicon-hit lists.
    R3 / R5: rho-rank of the highest-rho feature with >= 3 / >= 5 lexicon hits (NaN if none);
    lex3_top20: # features with >= 3 hits among the 20 highest-rho features (the judge window)."""
    r, lo = np.asarray(m["rho_all"]), np.asarray(m["lo_all"])
    order = np.argsort(-r, kind="stable")
    rank = np.empty(len(r), int)
    rank[order] = np.arange(1, len(r) + 1)
    first = lambda L: float(rank[lo >= L].min()) if (lo >= L).any() else np.nan
    return {"R3": first(3), "R5": first(5), "lex3_top20": int((lo[order[:20]] >= 3).sum()), "max_lo": int(lo.max())}


def _coh():
    p = OUT / "coherence.json"
    return read_json(p) if p.exists() else {}


def load_metrics(model="llama") -> pd.DataFrame:
    rows, coh = [], _coh()
    for p in glob.glob(str(FITS / model / "*" / "*" / "seed*" / "metrics.json")):
        m = json.loads(open(p).read())
        row = {k: v for k, v in m.items() if k not in ("lo_all", "rho_all")}
        row.update(derived(m))
        row.update(coh.get(f"{model}|{m['concept']}|{m['config']}|{m['seed']}", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def load_gemma_ref() -> pd.DataFrame:
    p = OUT / "gemma_ref.json"
    if not p.exists():
        return pd.DataFrame()
    d, coh = read_json(p), _coh()
    rows = []
    for key, m in d.items():
        row = {k: v for k, v in m.items() if k not in ("lo_all", "rho_all")}
        row.update(derived(m))
        row.update(coh.get(f"gemma|{key}", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by=("concept", "config")) -> pd.DataFrame:
    g = df.groupby(list(by))
    out = g.agg(n=("seed", "count"), found=("cf_found", "mean"), cf_rank_med=("cf_rank", "median"),
                cf_rank_mean=("cf_rank", "mean"), top1=("top1_is_cf", "mean"), in_top20=("cf_in_top20", "mean"),
                cf_rho_med=("cf_rho", "median"), cf_purity=("cf_purity", "mean"), cf_width=("cf_width", "mean"),
                cf_lo=("cf_lo", "mean"), n_tau=("n_tau", "mean"), frac_tau=("frac_tau", "mean"),
                auc=("auc", "mean"), rel_err=("rel_err", "mean"), n_ej5=("n_ej5", "mean"),
                R3_med=("R3", "median"), lex3_top20=("lex3_top20", "mean"), max_lo=("max_lo", "mean"),
                **({"coh_top20": ("coh_top20", "mean"), "coh_all": ("coh_all", "mean")} if "coh_top20" in df else {}))
    return out.reset_index()


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    return plt


def save(fig, name, data):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    if isinstance(data, pd.DataFrame):
        data.to_csv(FIG / f"{name}.csv", index=False)
    else:
        write_json(FIG / f"{name}.json", data)


def fig_spectrum(concept="Harry Potter", eta=0.05):
    """Eigenvalues g_j of S_eta (page target, eta = 0.05) for Gemma and Llama, and of a same-size random-type null."""
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    data = {}
    sp = read_json(OUT / "spectra" / "spectra.json")
    for model, col, lab in (("gemma", MODEL_COL["gemma"], "Gemma-2-2B"), ("llama", MODEL_COL["llama"], "Llama-3.1-8B")):
        s = torch.load(OUT / "spectra" / f"spectra_{model}_{slug(concept)}.pt")
        lam, lam0, m, d = s["lam"].numpy(), s["lam_null"].numpy(), s["m"], s["d"]
        g, g0 = (1 - eta) * lam + eta, (1 - eta) * lam0 + eta
        j = np.arange(1, d + 1)
        ns = sp[f"{model}|{concept}"]["n_signal"]
        ax.plot(j, g, color=col, lw=1.2, label=f"{lab}, page (m/d = {m / d:.2f})")
        ax.plot(j, g0, color=col, lw=0.9, ls=(0, (2.5, 1.5)), label=f"{lab}, random types (same m)")
        ax.plot([ns], [g[ns - 1]], marker="o", ms=3.5, color=col, mec="white", mew=0.5, ls="none")
        data[model] = {"m": m, "d": d, "n_signal": ns, "g": g.tolist(), "g_null": g0.tolist()}
    ax.axhline(eta, color="0.35", lw=0.5)
    ax.text(1.2, eta * 0.9, f"floor η = {eta:g}", fontsize=6.5, va="top", color="0.25")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index j")
    ax.set_ylabel(r"eigenvalue $g_j$ of $S_\eta$")
    ax.set_xlim(1, 4096)
    ax.set_ylim(0.03, 20)
    ax.legend(frameon=False, loc="lower left", fontsize=6, handlelength=2.2, bbox_to_anchor=(0, 0.12))
    save(fig, "fig_spectrum_hp", {"concept": concept, "eta": eta, **data})
    plt.close(fig)


def fig_grid(df: pd.DataFrame, gem: pd.DataFrame, concepts):
    """Rows: concept-feature rho-rank (lexicon-identified), share of features with rho >= 2; columns: concepts.
    Blue: CS on the (eta, gamma) grid (median over seeds, band = min-max); gray: EMBER (raw); green: CS truncated at
    the noise edge (mp, eta = 0.05, gamma = 1); black star: Gemma CS default (existing fits, seeds 42-44)."""
    plt = setup_mpl()
    etas = [0.05, 0.1, 0.2, 0.5]
    fig, axes = plt.subplots(2, len(concepts), figsize=(6.8, 3.1), sharex=True)
    rows = []
    refs = (("raw", COLORS["ember"], "-", "EMBER (raw E)"), ("cs_e0.05_g1_mp", COLORS["other"], (0, (3, 1.5)),
            "CS, noise-edge truncated"))
    for j, c in enumerate(concepts):
        d = df[df.concept == c]
        for i, metric in enumerate(("cf_rank", "frac_tau")):
            ax = axes[i, j]
            for gamma, ls, mk in ((1.0, "-", "o"), (0.5, "--", "s")):
                ys, lo_, hi_ = [], [], []
                for e in etas:
                    v = d[d.config == f"cs_e{e:g}_g{gamma:g}"][metric]
                    ys.append(v.median() if len(v) else np.nan)
                    lo_.append(v.min() if len(v) else np.nan)
                    hi_.append(v.max() if len(v) else np.nan)
                    rows.append({"concept": c, "metric": metric, "config": f"cs_e{e:g}_g{gamma:g}", "eta": e,
                                 "gamma": gamma, "median": ys[-1], "min": lo_[-1], "max": hi_[-1], "n": len(v)})
                ax.plot(etas, ys, ls=ls, marker=mk, ms=3, lw=1, color=COLORS["cs"],
                        label=f"CS (Llama), γ = {gamma:g}" if (i == 0 and j == 0) else None)
                ax.fill_between(etas, lo_, hi_, color=COLORS["cs"], alpha=0.12 if gamma == 1 else 0.06, lw=0)
            for cfg, col, ls, lab in refs:
                v = d[d.config == cfg][metric]
                if len(v):
                    ax.axhline(v.median(), color=col, lw=1.1, ls=ls, label=lab if (i == 0 and j == 0) else None)
                    rows.append({"concept": c, "metric": metric, "config": cfg, "median": v.median(),
                                 "min": v.min(), "max": v.max(), "n": len(v)})
            g = gem[(gem.concept == c) & (gem.config == "cs_e0.05_g1") & (gem.seed <= 44)] if len(gem) else gem
            if len(g):
                ax.plot([0.05], [g[metric].median()], marker="*", ms=7, color="k", ls="none",
                        label="Gemma CS default" if (i == 0 and j == 0) else None)
                rows.append({"concept": c, "metric": metric, "config": "gemma_cs_e0.05_g1", "median": g[metric].median(),
                             "min": g[metric].min(), "max": g[metric].max(), "n": len(g)})
            ax.set_xscale("log")
            ax.set_xticks(etas)
            ax.set_xticklabels([f"{e:g}" for e in etas])
            ax.minorticks_off()
            if metric == "cf_rank":
                ax.set_yscale("log")
                ax.set_ylim(0.8, 200)
                ax.set_yticks([1, 3, 10, 30, 100])
                ax.set_yticklabels(["1", "3", "10", "30", "100"])
                ax.axhline(20, color="k", lw=0.4, ls=":", alpha=0.6)
                ax.set_title(c, fontsize=8)
                if j == 0:
                    ax.set_ylabel("ρ-rank of the\nconcept feature")
            else:
                ax.set_ylim(0, 1)
                if j == 0:
                    ax.set_ylabel("share of\nfeatures with ρ ≥ 2")
                ax.set_xlabel("η")
    fig.legend(loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.align_ylabels(axes[:, 0])
    save(fig, "fig_grid_rank", pd.DataFrame(rows))
    plt.close(fig)


def fig_truncation(df: pd.DataFrame, concepts):
    """Dose-response: CS (eta = 0.05, gamma = 1) sharpening only the top r eigen-directions of M (rest at background
    level g = 1): r = noise edge (mp, ~12), 50, 100, all d. Mean over seeds; thin lines = concepts, thick = median."""
    plt = setup_mpl()
    cfgs = [("cs_e0.05_g1_mp", None), ("cs_e0.05_g1_top50", 50), ("cs_e0.05_g1_top100", 100), ("cs_e0.05_g1", 4096)]
    metrics = [("frac_tau", "share of features ρ ≥ 2", False), ("cf_rank", "concept-feature ρ-rank", True),
               ("coh_top20", "coherence (top-20 ρ)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.0))
    rows = []
    for ax, (metric, ylab, logy) in zip(axes, metrics):
        if metric not in df:
            continue
        allx, ally = [], []
        for c in concepts:
            d = df[df.concept == c]
            xs, ys = [], []
            for cfg, r in cfgs:
                v = d[d.config == cfg]
                if not len(v):
                    continue
                rr = r if r is not None else float(v["trunc_r"].iloc[0]) if "trunc_r" in v and v["trunc_r"].notna().any() else 12
                y = v[metric].median() if metric == "cf_rank" else v[metric].mean()
                xs.append(rr)
                ys.append(y)
                rows.append({"concept": c, "metric": metric, "config": cfg, "r": rr, "value": y, "n": len(v)})
            ax.plot(xs, ys, color=COLORS["cs"], lw=0.6, alpha=0.45, marker="o", ms=2)
            allx.append(xs)
            ally.append(ys)
            raw = d[d.config == "raw"][metric]
            if len(raw):
                ax.plot([xs[0], xs[-1]], [raw.median() if metric == "cf_rank" else raw.mean()] * 2,
                        color=COLORS["ember"], lw=0.5, alpha=0.5)
        med = np.nanmedian(np.array(ally, dtype=float), axis=0)
        ax.plot(np.nanmedian(np.array(allx, dtype=float), axis=0), med, color=COLORS["cs"], lw=1.6, marker="o", ms=3.5,
                label="CS (Llama), median of 4 concepts")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
            ax.set_yticks([1, 3, 10, 30, 100])
            ax.set_yticklabels(["1", "3", "10", "30", "100"])
        ax.set_xlabel("# sharpened directions r")
        ax.set_ylabel(ylab)
        ax.minorticks_off()
        ax.set_xticks([12, 50, 100, 4096])
        ax.set_xticklabels(["edge", "50", "100", "all"])
    axes[0].plot([], [], color=COLORS["cs"], lw=0.6, alpha=0.45, marker="o", ms=2, label="CS (Llama), per concept")
    axes[0].plot([], [], color=COLORS["ember"], lw=0.8, label="EMBER (raw), per concept")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_truncation", pd.DataFrame(rows))
    plt.close(fig)


def paired(a: pd.Series, b: pd.Series, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired comparison over concepts (index-aligned): median difference a - b, bootstrap 95% CI of the mean
    difference over concepts, Wilcoxon signed-rank p (two-sided)."""
    from scipy.stats import wilcoxon
    d = (a - b).dropna()
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d.values, len(d)).mean() for _ in range(n_boot)] if len(d) else [np.nan]
    try:
        p = float(wilcoxon(d.values).pvalue) if (d != 0).any() else 1.0
    except ValueError:
        p = float("nan")
    return {"n": int(len(d)), "mean_a": float(a[d.index].mean()), "mean_b": float(b[d.index].mean()),
            "median_diff": float(d.median()), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "wilcoxon_p": p}


def stats18(df: pd.DataFrame, gem: pd.DataFrame) -> dict:
    """18 concepts, seeds 42-44: per-concept seed means (cf_rank: seed median, log-scale summaries), paired tests."""
    from scipy.stats import spearmanr
    sp = read_json(OUT / "spectra" / "spectra.json")
    out = {}
    tabs = []
    for model, d in (("llama", df), ("gemma", gem)):
        d = d[d.seed <= 44].copy()
        d["log2_rank"] = np.log2(d["cf_rank"])
        agg = d.groupby(["concept", "config"]).agg(frac_tau=("frac_tau", "mean"), log2_rank=("log2_rank", "median"),
                                                    cf_rank=("cf_rank", "median"), R3=("R3", "median"),
                                                    purity=("cf_purity", "mean"), auc=("auc", "mean"),
                                                    **({"coh_top20": ("coh_top20", "mean")} if "coh_top20" in d else {}),
                                                    n=("seed", "count")).reset_index()
        agg["model"] = model
        agg["m_over_d"] = agg.concept.map(lambda c: sp[f"{model}|{c}"]["m_over_d"])
        agg["n_signal"] = agg.concept.map(lambda c: sp[f"{model}|{c}"]["n_signal"])
        tabs.append(agg)
        piv = {m: agg.pivot(index="concept", columns="config", values=m) for m in
               ("frac_tau", "log2_rank", "R3", "purity", "auc") + (("coh_top20",) if "coh_top20" in agg else ())}
        res = {}
        for a_cfg, b_cfg in (("cs_e0.05_g1", "raw"), ("cs_e0.05_g1_mp", "raw"), ("cs_e0.05_g1_mp", "cs_e0.05_g1"),
                             ("white", "raw"), ("cs_e0.05_g1_mp", "white")):
            if a_cfg not in piv["frac_tau"] or b_cfg not in piv["frac_tau"]:
                continue
            res[f"{a_cfg} - {b_cfg}"] = {m: paired(piv[m][a_cfg], piv[m][b_cfg]) for m in piv}
        cs = agg[agg.config == "cs_e0.05_g1"]
        if len(cs) > 3:
            res["spearman_vs_m_over_d(cs default)"] = {
                m: float(spearmanr(cs["m_over_d"], cs[m], nan_policy="omit")[0]) for m in ("frac_tau", "log2_rank")}
        out[model] = res
    allt = pd.concat(tabs)
    allt.to_csv(TAB / "per_concept_18.csv", index=False)
    write_json(TAB / "stats18.json", out)
    return out


def spectra_table():
    sp = read_json(OUT / "spectra" / "spectra.json")
    rows = []
    for key, r in sp.items():
        row = {k: r[k] for k in ("model", "concept", "d", "m", "m_over_d", "rank", "trace_over_d", "lam_max",
                                 "null_lam_max", "n_signal", "eta_LW", "frac_C_on_page", "span_share_offpage")}
        for e, v in r["per_eta"].items():
            row[f"floor_{e}"] = v["frac_floor"]
            row[f"cond_{e}"] = v["cond"]
        for e, v in r["vstar_diag"].items():
            row[f"self_share_{e}"] = v["self_share_g1"]
            row[f"boost_g1_{e}"] = v["boost_g1"]
            row[f"boost_g0.5_{e}"] = v["boost_g0.5"]
        row["sine_top10"] = r["split_half"]["sine_top10"]
        row["sine_top10_null"] = r["split_half_null"]["sine_top10"]
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "spectra_18.csv", index=False)
    return t


def main():
    df = load_metrics("llama")
    gem = load_gemma_ref()
    TAB.mkdir(parents=True, exist_ok=True)
    if len(gem):
        gmine = load_metrics("gemma")
        gem_all = pd.concat([gem, gmine[gmine.config.isin(["cs_e0.05_g1_mp", "white"])]])
        st = stats18(df, gem_all)
        print(json.dumps(st, indent=1)[:6000])
    spectra_table()
    df.to_csv(TAB / "llama_fits_all.csv", index=False)
    s = summarize(df)
    s.to_csv(TAB / "llama_summary.csv", index=False)
    if len(gem):
        gem.to_csv(TAB / "gemma_ref_all.csv", index=False)
        summarize(gem).to_csv(TAB / "gemma_summary.csv", index=False)
    print(s.to_string())
    if (OUT / "spectra").exists():
        fig_spectrum()
    concepts = [c for c in ["Harry Potter", "Ancient Rome", "Valentine's Day", "Baseball"] if c in set(df.concept)]
    fig_grid(df, gem, concepts)
    fig_truncation(df, concepts)


if __name__ == "__main__":
    main()
