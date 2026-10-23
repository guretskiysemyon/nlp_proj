#!/usr/bin/env python
"""WP6 edit_direction: statistics and figures from gemma_runs.json, gemma_cloud.json, llama_runs.json.

    .venv/bin/python -m experiments.paper_analysis.edit_direction.summarize    # -> summary.json, fig_*.pdf/png/csv

Unit of analysis: the concept. A paired comparison (a vs b) uses the runs (seeds) where both edits exist, averages
each metric over those seeds per concept, and reports the mean concept-level difference with a bootstrap 95% CI
(10,000 resamples over concepts, seed 0) and a two-sided Wilcoxon signed-rank test.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from experiments.paper_analysis.edit_direction.common import COLORS, LABELS, OUT, save_json

METRICS = ["read_qkv", "read_mlp", "read_all", "read_qkv_perp_mu", "rayleigh", "rayleigh_perp_mu", "mu_share",
           "top1_share", "beta_mu", "cos_ce", "cos_ce_perp_mu", "ratio_ce", "removed", "proj_e", "align_via_mu",
           "dec1", "dec10"]
PAIRS = [("ember", "cinder"), ("cinder_z", "cinder"), ("cinder_z", "ember"), ("ember", "ember_c"),
         ("ember_c", "cinder")]
LOG = {"rayleigh", "rayleigh_perp_mu"}


def frame() -> pd.DataFrame:
    rows = json.loads((OUT / "gemma_runs.json").read_text())
    recs = []
    for r in rows:
        for v, m in r["variants"].items():
            rec = {"concept": r["concept"], "seed": r["seed"], "variant": v}
            rec.update({k: m[k] for k in METRICS if k in m})
            rec["dec1"], rec["dec10"] = m["dec_share"][0], m["dec_share"][-1]
            for i, x in enumerate(m["dec_share"]):
                rec[f"d{i}"] = x
            for i, x in enumerate(m["dec_share_perp_mu"]):
                rec[f"dp{i}"] = x
            recs.append(rec)
    return pd.DataFrame(recs)


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(df: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    w = df.pivot_table(index=["concept", "seed"], columns="variant", values=metric)
    if a not in w or b not in w:
        return {}
    w = w[[a, b]].dropna()
    if metric in LOG:
        w = np.log10(w)
    c = w.groupby(level=0).mean()
    d = (c[a] - c[b]).to_numpy()
    lo, hi = boot_ci(d)
    p = float(stats.wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d != 0) else None
    return {"n_concepts": int(len(c)), "n_runs": int(len(w)), "mean_a": float(c[a].mean()), "mean_b": float(c[b].mean()),
            "diff": float(d.mean()), "ci95": [lo, hi], "wilcoxon_p": p, "a_gt_b": int((d > 0).sum()),
            "log10": metric in LOG}


def describe(df: pd.DataFrame) -> dict:
    out = {}
    for v, g in df.groupby("variant"):
        c = g.groupby("concept")[METRICS].mean(numeric_only=True)
        out[v] = {"n_runs": int(len(g)), "n_concepts": int(g["concept"].nunique()),
                  **{m: {"run_min": float(g[m].min()), "run_p10": float(g[m].quantile(.1)),
                         "run_median": float(g[m].median()), "run_p90": float(g[m].quantile(.9)),
                         "run_max": float(g[m].max()), "concept_mean": float(c[m].mean())} for m in METRICS}}
    return out


def pair_checks() -> dict:
    rows = json.loads((OUT / "gemma_runs.json").read_text())
    keys = {"cos_map_vs_centered_resolve": "p50", "cos_z_vs_map": "p50", "diff_mu_share": "p50",
            "diff_norm_over_c": "p50"}
    out = {k: [] for k in keys}
    out.update({"cos_map_vs_centered_resolve_min": [], "diff_mu_share_min": [], "Z_map_vs_Z_c": [], "e_read_qkv": [],
                "e_beta_mu": [], "e_mu_share": [], "e_rayleigh": []})
    rec = {k: [] for k in ("all", "edited")}
    ember_rel = []
    for r in rows:
        p = r.get("cinder_pair")
        if not p:
            continue
        for k, s in keys.items():
            out[k].append(p[k][s])
        out["cos_map_vs_centered_resolve_min"].append(p["cos_map_vs_centered_resolve"]["p10"])
        out["diff_mu_share_min"].append(p["diff_mu_share"]["p10"])
        for k in ("e_read_qkv", "e_beta_mu", "e_mu_share", "e_rayleigh"):
            out[k].append(p[k])
        out["Z_map_vs_Z_c"].append(r["recon"]["Z_map_vs_Z_c"])
        for s in rec:
            rec[s].append(r["recon"][s])
        if "ember_Z_vs_resolve" in r["recon"]:
            ember_rel.append(r["recon"]["ember_Z_vs_resolve"])
    res = {k: {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)}
           for k, v in out.items()}
    res["ember_Z_vs_resolve"] = {"median": float(np.median(ember_rel)), "max": float(np.max(ember_rel))}
    rr = {}
    for s, L in rec.items():
        d = pd.DataFrame(L)
        rr[s] = {c: {"median": float(d[c].median()), "min": float(d[c].min()), "max": float(d[c].max())}
                 for c in d.columns}
        rr[s]["affine_map_better_than_o"] = int((d["eucl_raw_map_with_mu"] < d["eucl_raw_o"]).sum())
        rr[s]["n"] = len(d)
    res["recon"] = rr
    return res


def reproduce_d16(df: pd.DataFrame) -> dict:
    """Max |difference| to the read strengths of collateral_diag.a3z_offline (D16, R13) on its 8 runs."""
    from experiments.varwidth.collateral import OUT as COL
    ref = json.loads((COL / "diag" / "a3z_offline.json").read_text())
    diffs, n = [], 0
    for key, row in ref.items():
        concept, seed = key.rsplit(" ", 1)
        for v in ("ember", "cinder", "cinder_z"):
            if v not in row:
                continue
            m = df[(df.concept == concept) & (df.seed == int(seed)) & (df.variant == v)]
            if len(m):
                diffs.append(abs(float(m.read_qkv.iloc[0]) - row[v][2]))
                n += 1
    return {"n_compared": n, "max_abs_diff_read": float(max(diffs)) if diffs else None}


def correlations(df: pd.DataFrame) -> dict:
    out = {}
    for x in ("mu_share", "rayleigh", "top1_share", "beta_mu"):
        r = stats.spearmanr(df[x], df["read_qkv"])
        out[f"read_qkv~{x}"] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(df))}
    r = stats.spearmanr(df["rayleigh_perp_mu"], df["read_qkv_perp_mu"])
    out["read_perp~rayleigh_perp"] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(df))}
    return out


def cloud_summary() -> dict:
    c = json.loads((OUT / "gemma_cloud.json").read_text())
    lam = np.array(c["lam_over_mean"])
    eig = np.array(c["eig"]["qkv"])
    dec = np.minimum(np.arange(len(lam)) * 10 // len(lam), 9)
    out = {"eig_read_qkv_by_decile": [float(eig[dec == i].mean()) for i in range(10)],
           "eig_read_mlp_by_decile": [float(np.array(c["eig"]["mlp"])[dec == i].mean()) for i in range(10)],
           "eig_read_qkv_top5": eig[:5].tolist(), "lam1_over_mean": float(lam[0]),
           "eig_cos_mu_top3": c["eig_cos_mu"][:3],
           "spearman_eig_read_vs_lam": float(stats.spearmanr(eig, lam).statistic),
           "mu": {k: c["mu"][k][0] for k in ("read_qkv", "read_mlp", "read_all", "rayleigh")}}
    for k in ("iso", "tilt", "mu_mix", "vocab"):
        out[k] = {"read_qkv_median": float(np.median(c[k]["read_qkv"])),
                  "read_qkv_p5_p95": [float(np.percentile(c[k]["read_qkv"], 5)),
                                      float(np.percentile(c[k]["read_qkv"], 95))],
                  "rayleigh_median": float(np.median(c[k]["rayleigh"])),
                  "spearman_read_vs_rayleigh": float(stats.spearmanr(c[k]["read_qkv"], c[k]["rayleigh"]).statistic),
                  "spearman_read_vs_mu_share": float(stats.spearmanr(c[k]["read_qkv"], c[k]["mu_share"]).statistic)}
    t = c["tilt"]
    a, rq = np.array(t["alpha"]), np.array(t["read_qkv"])
    out["tilt"]["read_by_alpha"] = {f"{lo:+.0f}..{lo + 1:+.0f}": float(np.median(rq[(a >= lo) & (a < lo + 1)]))
                                    for lo in range(-3, 3)}
    return out


def llama_summary() -> dict:
    L = json.loads((OUT / "llama_runs.json").read_text())
    recs = []
    for r in L["runs"]:
        for v, m in r["variants"].items():
            recs.append({"concept": r["concept"], "variant": v, **{k: m[k] for k in m if k != "dec_share"}})
    d = pd.DataFrame(recs)
    out = {k: L[k] for k in ("base_random_median", "mu_read", "mu_rayleigh", "mu_top1_energy", "lam1_over_mean",
                             "mu_norm_over_median_e", "eig_read_by_decile", "eig_read_qkv_by_decile",
                             "eig_read_top5")}
    out["variants"] = {}
    for v, g in d.groupby("variant"):
        out["variants"][v] = {"n": int(len(g)), **{m: {"min": float(g[m].min()), "median": float(g[m].median()),
                                                       "max": float(g[m].max())}
                                                   for m in ("read_all", "read_qkv", "rayleigh", "mu_share",
                                                             "cos_ce", "removed", "beta_mu")}}
    pr = {}
    for a, b in (("a2", "a3"), ("a3z", "a3"), ("a3z", "a3_c"), ("a3_c", "a3"), ("a2", "a2_c"), ("a2_c", "a3"),
                 ("a3z", "a2")):
        w = d.pivot_table(index="concept", columns="variant", values="read_all")
        if a in w and b in w:
            w = w[[a, b]].dropna()
            x = (w[a] - w[b]).to_numpy()
            pr[f"{a}-{b}"] = {"n": int(len(x)), "mean_a": float(w[a].mean()), "mean_b": float(w[b].mean()),
                              "diff": float(x.mean()), "ci95": boot_ci(x), "a_gt_b": int((x > 0).sum()),
                              "wilcoxon_p": float(stats.wilcoxon(x).pvalue) if len(x) >= 5 else None}
    out["paired_read_all"] = pr
    pc = {}
    for k in ("cos_a3z_vs_a3", "cos_a3c_vs_a3", "cos_a3z_vs_a3c", "diff_zc_mu_share"):
        v = [r["a3_pair"][k]["p50"] for r in L["runs"] if "a3_pair" in r]
        pc[k] = {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)}
    for k in ("rel_err_full_map_with_mu", "rel_err_full_c_with_mu", "rel_err_full_o", "e_read_all", "e_beta_mu"):
        v = [r["a3_pair"][k] for r in L["runs"] if "a3_pair" in r]
        pc[k] = {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)}
    out["a3_pair"] = pc
    return out


# ----------------------------------------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


MARK = {"ember": "o", "ember_c": "D", "cinder": "s", "cinder_z": "^"}
LS = {"ember": "-", "ember_c": "--", "cinder": "-", "cinder_z": "-."}
ORDER = ("ember", "ember_c", "cinder", "cinder_z")


def fig_deciles(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    data = []
    x = np.arange(1, 11)
    for ax, pre, title in ((axes[0], "d", "edit $c_t$"), (axes[1], "dp", "edit with $\\hat\\mu_B$ projected out")):
        ax.axhline(0.1, color="#BBBBBB", lw=0.8, ls=":", zorder=0)
        for v in ORDER:
            if pre == "dp" and v == "cinder_z":
                continue                                  # identical to CINDER once mu_B is projected out (Prop. 1)
            g = df[df.variant == v].groupby("concept")[[f"{pre}{i}" for i in range(10)]].mean()
            m = g.mean().to_numpy()
            ci = np.array([boot_ci(g[f"{pre}{i}"].to_numpy(), n=2000) for i in range(10)])
            lab = LABELS[v] + (" (= CINDER-z)" if pre == "dp" and v == "cinder" else "")
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=COLORS[v], alpha=0.18, lw=0)
            ax.plot(x, m, LS[v], marker=MARK[v], ms=3.5, lw=1.3, color=COLORS[v], label=lab,
                    mfc="white" if v == "ember_c" else COLORS[v])
            for i in range(10):
                data.append({"panel": title, "variant": v, "decile": i + 1, "mean_share": m[i], "ci_lo": ci[i, 0],
                             "ci_hi": ci[i, 1], "n_concepts": len(g)})
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xlabel("$\\Sigma_B$ eigenvalue decile (1 = highest variance)")
        ax.text(0.98, 0.97, title, transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
        ax.text(10.4, 0.1, "isotropic", fontsize=6, color="#888888", va="bottom", ha="right")
    axes[0].set_ylabel("share of edit energy")
    axes[0].legend(frameon=False, loc="lower left", handlelength=2.2)
    axes[1].legend(frameon=False, loc="lower left", handlelength=2.2)
    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_spectral_deciles.{ext}", bbox_inches="tight")
    pd.DataFrame(data).to_csv(OUT / "fig_spectral_deciles.csv", index=False)
    plt.close(fig)


def fig_read(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    _style()
    c = json.loads((OUT / "gemma_cloud.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    ax = axes[0]
    ax.scatter(c["tilt"]["rayleigh"], c["tilt"]["read_qkv"], s=2, color="#CFCFCF", lw=0, rasterized=True,
               label="random, spectrally tilted")
    ax.scatter(np.array(c["lam_over_mean"]), c["eig"]["qkv"], s=2, color="#8FB9D6", lw=0, rasterized=True,
               label="eigenvectors of $\\Sigma_B$")
    ax.scatter(c["mu_mix"]["rayleigh"], c["mu_mix"]["read_qkv"], s=2, color="#A8DCC9", lw=0, rasterized=True,
               label="random + $\\hat\\mu_B$ mixtures")
    pts = df.groupby(["variant", "concept"])[["read_qkv", "rayleigh", "mu_share"]].mean().reset_index()
    for v in ORDER:
        p = pts[pts.variant == v]
        for a, xcol in ((axes[0], "rayleigh"), (axes[1], "mu_share")):
            a.scatter(p[xcol], p["read_qkv"], s=14, marker=MARK[v], color=COLORS[v] if v != "ember_c" else "white",
                      edgecolor=COLORS[v] if v == "ember_c" else "white", lw=0.8 if v == "ember_c" else 0.4,
                      label=LABELS[v], zorder=3)
    for a, xcol in ((axes[0], "rayleigh"), (axes[1], "mu_share")):
        a.scatter([c["mu"][xcol if xcol != "mu_share" else "mu_share"][0]], [c["mu"]["read_qkv"][0]], s=40,
                  marker="*", color="black", zorder=4, label="$\\hat\\mu_B$ (background mean)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.8, 1, 1.5, 2, 3, 5, 8])
    ax.set_yticklabels(["0.8", "1", "1.5", "2", "3", "5", "8"])
    ax.minorticks_off()
    ax.set_xlabel("Rayleigh ratio $v^\\top\\Sigma_B v\\,/\\,(\\|v\\|^2\\bar\\lambda)$")
    ax.set_ylabel("layer-0 read gain (× random)")
    ax.axhline(1, color="#BBBBBB", lw=0.6, ls=":", zorder=0)
    ax2 = axes[1]
    ax2.scatter(c["mu_mix"]["mu_share"], c["mu_mix"]["read_qkv"], s=2, color="#A8DCC9", lw=0, rasterized=True)
    ax2.set_xlabel("energy share along $\\hat\\mu_B$")
    ax2.axhline(1, color="#BBBBBB", lw=0.6, ls=":", zorder=0)
    h, lab = axes[0].get_legend_handles_labels()
    keep = {}
    for hh, ll in zip(h, lab):
        keep.setdefault(ll, hh)
    leg = fig.legend(list(keep.values()), list(keep.keys()), loc="upper center", ncol=4, frameon=False,
                     bbox_to_anchor=(0.5, 1.13), markerscale=1.0, handletextpad=0.3, columnspacing=1.0)
    for hnd in leg.legend_handles:
        if hasattr(hnd, "set_sizes") and hnd.get_sizes() is not None and len(hnd.get_sizes()) and hnd.get_sizes()[0] < 5:
            hnd.set_sizes([12])
    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_read_vs_rayleigh.{ext}", bbox_inches="tight")
    pts.to_csv(OUT / "fig_read_vs_rayleigh_methods.csv", index=False)
    cl = []
    for k in ("tilt", "mu_mix"):
        for r, s, m in zip(c[k]["read_qkv"], c[k]["rayleigh"], c[k]["mu_share"]):
            cl.append({"cloud": k, "read_qkv": r, "rayleigh": s, "mu_share": m})
    for r, s in zip(c["eig"]["qkv"], c["lam_over_mean"]):
        cl.append({"cloud": "eigvec", "read_qkv": r, "rayleigh": s, "mu_share": None})
    pd.DataFrame(cl).to_csv(OUT / "fig_read_vs_rayleigh_cloud.csv", index=False)
    plt.close(fig)


def fig_decomposition(df: pd.DataFrame):
    """Per-concept read gain of each edit variant, Gemma (qkv read, 31 concepts) and Llama (qkv + gate + up read,
    the multilingual study's definition)."""
    import matplotlib.pyplot as plt
    _style()
    L = json.loads((OUT / "llama_runs.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), gridspec_kw={"width_ratios": [4, 5]})
    data = []
    g = df.groupby(["variant", "concept"])["read_qkv"].mean().reset_index()
    cats = [("ember", "EMBER", "ember"), ("ember_c", "EMBER\ncentered Z", "ember_c"),
            ("cinder", "CINDER\n(= centered Z)", "cinder"), ("cinder_z", "CINDER-z", "cinder_z")]
    lcats = [("a2", "ML-EMBER", "ember"), ("a2_c", "ML-EMBER\ncentered Z", "ember_c"),
             ("a3", "ML-CINDER", "cinder"), ("a3_c", "ML-CINDER\ncentered Z", "cinder"), ("a3z", "A3z", "cinder_z")]
    rng = np.random.default_rng(0)
    for ax, model, cs in ((axes[0], "Gemma-2-2B", cats), (axes[1], "Llama-3.1-8B", lcats)):
        for i, (v, lab, col) in enumerate(cs):
            if model.startswith("Gemma"):
                y = g[g.variant == v]["read_qkv"].to_numpy()
                names = g[g.variant == v]["concept"].tolist()
            else:
                rows = [(r["concept"], r["variants"][v]["read_all"]) for r in L["runs"] if v in r["variants"]]
                names, y = [a for a, _ in rows], np.array([b for _, b in rows])
            x = i + rng.uniform(-0.18, 0.18, len(y))
            hollow = v in ("ember_c", "a2_c", "a3_c")
            ax.scatter(x, y, s=9, color="white" if hollow else COLORS[col], edgecolor=COLORS[col],
                       lw=0.7 if hollow else 0.2, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [np.median(y)] * 2, color="black", lw=1.2, zorder=4)
            for nm, yy in zip(names, y):
                data.append({"model": model, "variant": v, "concept": nm, "read_gain": float(yy)})
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([c[1] for c in cs], fontsize=6.5)
        ax.set_yscale("log")
        ticks = [1, 1.5, 2, 3] if model.startswith("Gemma") else [1, 2, 4, 8]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.axhline(1, color="#BBBBBB", lw=0.6, ls=":", zorder=0)
        ax.text(0.02, 0.97, model, transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    axes[0].set_ylabel("layer-0 read gain (× random)")
    fig.tight_layout(w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_read_decomposition.{ext}", bbox_inches="tight")
    pd.DataFrame(data).to_csv(OUT / "fig_read_decomposition.csv", index=False)
    plt.close(fig)


def main():
    df = frame()
    res = {"describe": describe(df), "paired": {}, "checks": pair_checks(), "correlations": correlations(df),
           "reproduce_d16": reproduce_d16(df),
           "cloud": cloud_summary()}
    for a, b in PAIRS:
        res["paired"][f"{a}-{b}"] = {m: paired(df, a, b, m) for m in METRICS}
    res["n_runs_with"] = {v: int((df.variant == v).sum()) for v in ORDER}
    res["n_concepts_with"] = {v: int(df[df.variant == v].concept.nunique()) for v in ORDER}
    if (OUT / "llama_runs.json").exists():
        res["llama"] = llama_summary()
    save_json(OUT / "summary.json", res)
    fig_deciles(df)
    fig_read(df)
    if (OUT / "llama_runs.json").exists():
        fig_decomposition(df)
    print(json.dumps({k: res[k] for k in ("n_runs_with", "n_concepts_with", "checks", "correlations", "reproduce_d16")}, indent=1))
    for k, v in res["paired"].items():
        print(k, {m: (round(x["mean_a"], 3), round(x["mean_b"], 3), x["a_gt_b"], x["n_concepts"],
                      None if x["wilcoxon_p"] is None else f"{x['wilcoxon_p']:.1e}") for m, x in v.items() if x})


if __name__ == "__main__":
    main()
