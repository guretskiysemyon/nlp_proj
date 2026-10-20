#!/usr/bin/env python3
"""Figures for Part1_report.md (Part 1: cross-lingual leakage of English-tuned erasure).

Inputs (all tracked in git, CPU only, a few seconds):
  - docs/Part 1/part1_results.md: per-concept cross-lingual tables (§5.1, §5.3) and the
    selection-score means (§4.1). The raw cross-lingual summary.csv files are git-ignored,
    so the per-concept values are read from these tables (rounded to 2 decimals).
  - judge_rerun/Ember/rescore_mc/.../<Concept>/grid_rescored.csv: every EMBER delta,
    scored with the original 4-token scorer and the fixed 32-token scorer.
  - judge_rerun/Ember/pass2_rescored/.../<Concept>/delta_metrics__qwen2.5-32b-instruct.csv
  - judge_rerun/Ember/pass1_outputs/.../_baseline/baseline_raw.csv: unedited model.
  - report/analysis/paired_fixed.csv: per-concept-convention means, bootstrap intervals.

Writes PNGs to Part1_report/figures/ and prints the numbers the report quotes.

Usage (from the repo root; needs matplotlib, numpy, pandas):
    python Part1_report/make_figures.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
RESULTS_MD = ROOT / "docs" / "Part 1" / "part1_results.md"
GRID = ROOT / "judge_rerun" / "Ember" / "rescore_mc" / "google_gemma-2-2b-it" / "rank100" / "seed42"
PASS2 = ROOT / "judge_rerun" / "Ember" / "pass2_rescored" / "google_gemma-2-2b-it" / "rank100" / "seed42"
BASELINE = ROOT / "judge_rerun" / "Ember" / "pass1_outputs" / "google_gemma-2-2b-it" / "rank100" / "seed42" / "_baseline" / "baseline_raw.csv"
PAIRED = ROOT / "report" / "analysis" / "paired_fixed.csv"

LANGS = ["en", "de", "es", "fr", "ru"]
OTHER = ["de", "es", "fr", "ru"]
METHODS = ["ember", "snmf", "snmf_ember"]
LABEL = {"ember": "EMBER", "snmf": "SNMF", "snmf_ember": "SNMF+EMBER"}
# Categorical slots 1-3 of the reference palette (validated all-pairs, light surface).
COLOR = {"ember": "#2a78d6", "snmf": "#eb6834", "snmf_ember": "#1baf7a"}
MARKER = {"ember": "o", "snmf": "s", "snmf_ember": "^"}
# Metric lines (efficacy / specificity / score): violet, yellow, red (validated all-pairs).
METRIC_COLOR = {"efficacy": "#6250d6", "specificity": "#eda100", "score": "#e34948"}

SURFACE, INK, INK2, MUTED, GRIDC, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OLD = "#b4b2a9"  # neutral gray for "original scorer" series

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})


# ---------------------------------------------------------------- data loading
def _num(s: str) -> float:
    s = s.strip().replace("−", "-").replace("*", "").replace("¹", "").strip()
    s = re.sub(r"\s*\(.*\)$", "", s)
    return float("nan") if s in {"—", "-", ""} else float(s)


def _table_after(lines: list[str], marker: str) -> list[list[str]]:
    """Rows (cells) of the first markdown table after the line that starts with `marker`."""
    i = next(k for k, l in enumerate(lines) if l.startswith(marker))
    while not lines[i].startswith("|"):
        i += 1
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not set("".join(cells)) <= set("-: "):
            rows.append(cells)
        i += 1
    return rows


def load_md():
    lines = RESULTS_MD.read_text().splitlines()
    rows = _table_after(lines, "### 5.1")
    base = pd.DataFrame([[r[0].replace("*", "")] + [_num(c) for c in r[1:]] for r in rows[1:]],
                        columns=["concept"] + LANGS).set_index("concept")
    per = {}
    for m in METHODS:
        rows = _table_after(lines, f"**`{m}`**")
        recs = []
        for r in rows[1:]:
            sim_en, sim_other = (_num(x) for x in r[8].split("/"))
            recs.append({"concept": r[0], "en_acc": _num(r[1]), "r_en": _num(r[2]),
                         **{f"d_{l}": _num(r[3 + k]) for k, l in enumerate(OTHER)},
                         "sim_en": sim_en, "sim_other": sim_other, "mmlu": _num(r[9])})
        df = pd.DataFrame(recs).set_index("concept")
        for l in OTHER:
            df[f"r_{l}"] = df.r_en + df[f"d_{l}"]
        df["r_ru"] = df["r_ru"].where(base.loc[df.index, "ru"] >= 0.40)  # screened cells
        df["leak_mean"] = df[[f"d_{l}" for l in OTHER]].mean(axis=1, skipna=True)
        per[m] = df
    rows = _table_after(lines, "Mean over the 18 concepts")
    sel = {}
    for r in rows[1:]:
        key = {"EMBER": "ember", "SNMF": "snmf", "SNMF+EMBER": "snmf_ember"}[r[0]]
        sel[key] = [_num(c) for c in r[1:5]]
    sel = pd.DataFrame(sel, index=["efficacy", "specificity", "coherence", "score"])
    return base, per, sel


def boot(v, n=10_000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    idx = np.random.default_rng(0).integers(0, len(v), (n, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return v.mean(), lo, hi


def load_grid():
    base = pd.read_csv(BASELINE).set_index("concept")
    recs = []
    for d in sorted(GRID.iterdir()):
        if not (d / "grid_rescored.csv").exists():
            continue
        g = pd.read_csv(d / "grid_rescored.csv")
        m = pd.read_csv(PASS2 / d.name / "delta_metrics__qwen2.5-32b-instruct.csv")
        g = g.merge(m[["delta_embed", "efficacy", "specificity", "harmonic"]], on="delta_embed")
        qb = base.loc[g.concept.iloc[0], "qa_acc"]
        g["efficacy_old"] = 1 - ((g.grid_qa_acc - 0.25) / (qb - 0.25)).clip(0, 1)
        g["qa_base"] = qb
        recs.append(g)
    return pd.concat(recs, ignore_index=True), base


# ---------------------------------------------------------------- helpers
def style(ax, ygrid=True):
    if ygrid:
        ax.yaxis.grid(True, color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}")


def method_legend(ax, **kw):
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="none", ms=7, mec=SURFACE, mew=1.2,
                          label=LABEL[m]) for m in METHODS]
    ax.legend(handles=handles, **kw)


# ---------------------------------------------------------------- figures
def fig_baseline(base):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    rng = np.random.default_rng(1)
    for k, l in enumerate(LANGS):
        v = base[l].values
        jit = rng.uniform(-0.17, 0.17, len(v))
        ok = v >= 0.40
        ax.scatter(k + jit[ok], v[ok], s=22, color="#86b6ef", edgecolor=SURFACE, lw=0.8, zorder=3)
        ax.scatter(k + jit[~ok], v[~ok], s=26, facecolor=SURFACE, edgecolor=INK2, lw=1.0, zorder=3)
        ax.plot([k - 0.28, k + 0.28], [v.mean()] * 2, color="#184f95", lw=2.2, solid_capstyle="round", zorder=4)
        ax.text(k + 0.31, v.mean(), f"{v.mean():.2f}", va="center", fontsize=8.5, color=INK)
    ax.axhline(0.25, color=MUTED, lw=1)
    ax.text(-0.45, 0.255, "chance (0.25)", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0.40, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(-0.45, 0.405, "screen: cells below 0.40 dropped", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(5), ["English", "German", "Spanish", "French", "Russian"])
    ax.set_ylim(0.2, 1.0)
    ax.set_xlim(-0.5, 4.7)
    ax.set_ylabel("Unedited QA_test accuracy")
    ax.set_title("The unedited model knows the concepts less well outside English")
    style(ax)
    ax.scatter([], [], s=22, color="#86b6ef", label="one concept")
    ax.scatter([], [], s=26, facecolor=SURFACE, edgecolor=INK2, label="dropped (< 0.40)")
    ax.plot([], [], color="#184f95", lw=2.2, label="mean over 18 concepts")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig1_pristine_baseline.png")


def fig_selection(sel):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    metrics = ["efficacy", "specificity", "coherence", "score"]
    names = ["Efficacy\n(erases concept)", "Specificity\n(keeps other knowledge)", "Coherence\n(still writes well)",
             "Selection score\n(harmonic mean)"]
    w = 0.24
    for k, m in enumerate(METHODS):
        xs = np.arange(4) + (k - 1) * (w + 0.02)
        ax.bar(xs, sel.loc[metrics, m], width=w, color=COLOR[m], label=LABEL[m], zorder=3)
        for x, v in zip(xs, sel.loc[metrics, m]):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(4), names)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Mean over 18 concepts (QA_train)")
    ax.set_title("English selection scores of the chosen edits")
    style(ax)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig2_selection_scores.png")


def fig_delta_sweep(grid, base):
    deltas = sorted(grid.delta_embed.unique())
    mean = grid.groupby("delta_embed").mean(numeric_only=True)
    sem = grid.groupby("delta_embed").sem(numeric_only=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"wspace": 0.32})
    # (a) concept-QA accuracy, old vs fixed scorer
    for col, inv, color, lab in [("grid_qa_acc", "grid_qa_invalid_rate", OLD, "original scorer (4 tokens)"),
                                 ("qa_acc", "qa_invalid_rate", COLOR["ember"], "fixed scorer (32 tokens)")]:
        a.fill_between(deltas, mean[col] - sem[col], mean[col] + sem[col], color=color, alpha=0.12, lw=0)
        a.plot(deltas, mean[col], color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1, label=lab)
    pb = base.qa_acc.mean()
    a.axhline(pb, color=MUTED, lw=1, ls=(0, (3, 2)))
    a.text(0.5, pb + 0.008, f"unedited {pb:.2f}", fontsize=8, color=INK2, va="bottom")
    a.axhline(0.25, color=MUTED, lw=1)
    a.text(0.5, 0.257, "chance", fontsize=8, color=INK2, va="bottom")
    a.set_xscale("log")
    a.set_xticks(deltas, [f"{d:g}" for d in deltas])
    a.set_ylim(0.2, 0.8)
    a.set_xlabel("EMBER strength δ")
    a.set_ylabel("Concept accuracy, QA_train (mean ± SEM)")
    a.set_title("(a) The 4-token scorer faked erasure at large δ")
    a.legend(loc="center right", bbox_to_anchor=(1.0, 0.38))
    style(a)
    # (b) efficacy / specificity / judge-free score, fixed scorer
    for col, key, lab in [("efficacy", "efficacy", "efficacy"), ("specificity", "specificity", "specificity"),
                          ("harmonic", "score", "HM(eff., spec.)")]:
        b.fill_between(deltas, mean[col] - sem[col], mean[col] + sem[col], color=METRIC_COLOR[key], alpha=0.12, lw=0)
        b.plot(deltas, mean[col], color=METRIC_COLOR[key], lw=2, marker="o", ms=5, mec=SURFACE, mew=1)
        b.text(deltas[-1] * 1.25, mean[col].iloc[-1], lab, va="center", fontsize=8.5, color=INK)
    b.set_xscale("log")
    b.set_xticks(deltas, [f"{d:g}" for d in deltas])
    b.set_xlim(0.4, 900)
    b.minorticks_off()
    a.minorticks_off()
    b.set_ylim(0, 1.05)
    b.set_xlabel("EMBER strength δ")
    b.set_ylabel("Mean over 18 concepts (fixed scorer)")
    b.set_title("(b) Trade-off along δ")
    style(b)
    save(fig, "fig3_ember_delta_sweep.png")
    out = pd.DataFrame({"qa_old": mean.grid_qa_acc, "qa_fixed": mean.qa_acc, "inv_old": mean.grid_qa_invalid_rate,
                        "inv_fixed": mean.qa_invalid_rate, "eff_old": mean.efficacy_old, "eff_fixed": mean.efficacy,
                        "spec": mean.specificity, "H": mean.harmonic})
    print(out.round(3).to_string())


def fig_retention(per):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(5)
    stats = {}
    for k, m in enumerate(METHODS):
        s = np.array([boot(per[m][f"r_{l}"]) for l in LANGS])
        stats[m] = s
        xs = x + (k - 1) * 0.22
        ax.vlines(xs, s[:, 1], s[:, 2], color=COLOR[m], lw=2, zorder=3)
        ax.plot(xs, s[:, 0], ls="none", marker=MARKER[m], ms=8, color=COLOR[m], mec=SURFACE, mew=1.5, zorder=4)
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(4.45, 1.01, "unedited model = 1", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0.0, color=AXIS, lw=0.8)
    ax.text(4.45, 0.01, "chance = 0", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, ["English\n(tuned here)", "German", "Spanish", "French", "Russian"])
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylabel("Retention r_L (lower = more erased)")
    ax.set_title("How much of the concept survives, by question language")
    style(ax)
    method_legend(ax, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    save(fig, "fig4_retention_by_language.png")
    for m in METHODS:
        print(f"  {LABEL[m]:11s}", " ".join(f"{l} {v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]" for l, v in zip(LANGS, stats[m])))


def fig_leak_forest(per):
    rows = [("German", "d_de"), ("Spanish", "d_es"), ("French", "d_fr"), ("Russian", "d_ru"),
            ("Mean of the 4", "leak_mean")]
    # Russian and the mean come from the unrounded per-concept results (paired_fixed.csv);
    # de/es/fr are bootstrapped here from the 2-decimal tables (may differ by 0.01).
    p = pd.read_csv(PAIRED)
    p = p[p.run == "fixed"].set_index(["comparison", "stat"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (_, col) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for k, m in enumerate(METHODS):
            key = {"d_ru": "leak_ru", "leak_mean": "leak_mean"}.get(col)
            v, lo, hi = p.loc[(m, key), ["mean", "lo", "hi"]] if key else boot(per[m][col])
            y = y0 + (1 - k) * 0.24
            ax.hlines(y, lo, hi, color=COLOR[m], lw=2, zorder=3)
            ax.plot(v, y, marker=MARKER[m], ms=8, color=COLOR[m], mec=SURFACE, mew=1.5, zorder=4)
            ax.text(0.66, y, f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.axhspan(-0.45, 0.45, color="#f0efec", zorder=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(-0.25, 0.98)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    ax.set_xlabel("Leak Δ_L = r_L − r_en   (> 0: concept survives better than in English)")
    ax.set_title("Leak per language, mean with 95% bootstrap interval over concepts")
    ax.xaxis.grid(True, color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.55, len(rows) - 0.25)
    ax.text(0.66, len(rows) - 0.48, "mean [95% CI]", fontsize=7.5, color=MUTED)
    method_legend(ax, loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.17))
    save(fig, "fig5_leak_by_language.png")


def fig_collateral(per):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.3})
    # (a) similar-domain retention per language (documented means, part1_results §5.2)
    sim = {"ember": [0.95, 0.98, 0.98, 0.98, 1.00], "snmf": [0.73, 0.63, 0.65, 0.68, 0.59],
           "snmf_ember": [0.76, 0.67, 0.68, 0.73, 0.67]}
    x = np.arange(5)
    for m in METHODS:
        a.plot(x, sim[m], color=COLOR[m], lw=2, marker=MARKER[m], ms=7, mec=SURFACE, mew=1.2, label=LABEL[m])
    a.axhline(1.0, color=MUTED, lw=1, ls=(0, (3, 2)))
    a.set_xticks(x, ["en", "de", "es", "fr", "ru"])
    a.set_ylim(0.4, 1.06)
    a.set_ylabel("SimDom_test retention (higher = less damage)")
    a.set_title("(a) Similar-domain knowledge kept")
    style(a)
    a.legend(loc="lower left")
    # (b) English MMLU, 1000-q test split, one dot per concept
    rng = np.random.default_rng(2)
    for k, m in enumerate(METHODS):
        v = per[m].mmlu.values
        b.scatter(k + rng.uniform(-0.16, 0.16, len(v)), v, s=18, color=COLOR[m], edgecolor=SURFACE, lw=0.8,
                  alpha=0.85, zorder=3, marker=MARKER[m])
        b.plot([k - 0.26, k + 0.26], [v.mean()] * 2, color=INK, lw=2, solid_capstyle="round", zorder=4)
        b.text(k, 0.566, f"{v.mean():.3f}", ha="center", fontsize=8, color=INK)
    b.axhline(0.553, color=MUTED, lw=1, ls=(0, (3, 2)))
    b.text(2.45, 0.5535, "unedited 0.553", ha="right", va="bottom", fontsize=8, color=INK2)
    b.set_xticks(range(3), [LABEL[m] for m in METHODS])
    b.set_ylim(0.42, 0.575)
    b.set_ylabel("English MMLU accuracy (1000 q)")
    b.set_title("(b) General knowledge kept")
    style(b)
    save(fig, "fig6_collateral_damage.png")


def fig_heatmap(per):
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#b8322f"])
    concepts = per["ember"].index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 5.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, m in zip(axes, METHODS):
        mat = per[m].loc[concepts, [f"d_{l}" for l in OTHER]].values
        im = ax.imshow(mat, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
        for (i, j), v in np.ndenumerate(mat):
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=SURFACE, edgecolor=AXIS, hatch="///",
                                           lw=0))
                continue
            ax.text(j, i, f"{v:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00"), ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 0.38 else INK)
        ax.set_xticks(range(4), OTHER)
        ax.xaxis.tick_top()
        ax.set_title(LABEL[m], pad=18, loc="center")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(concepts), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=1.5)
        ax.tick_params(which="minor", length=0)
    axes[0].set_yticks(range(len(concepts)), concepts)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Leak Δ_L (red: survives better than in English;\nblue: erased more than in English)", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Leak per concept and language (hatched: dropped, unedited accuracy < 0.40)", x=0.12, ha="left",
                 y=0.995, fontsize=10, fontweight="semibold")
    save(fig, "fig8_leak_heatmap.png")


def fig_repair():
    p = pd.read_csv(PAIRED)
    p = p[p.comparison.isin(METHODS)]
    stats = [("r_en", "English retention r_en\n(lower = stronger erasure)"),
             ("leak_mean", "Mean leak over de/es/fr/ru"), ("leak_ru", "Leak into Russian")]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), gridspec_kw={"wspace": 0.35})
    for ax, (st, title) in zip(axes, stats):
        for k, m in enumerate(METHODS):
            for j, run in enumerate(["original", "fixed"]):
                r = p[(p.comparison == m) & (p.stat == st) & (p.run == run)].iloc[0]
                xx = k + (j - 0.5) * 0.36
                color = OLD if run == "original" else COLOR[m]
                ax.bar(xx, r["mean"], width=0.32, color=color, zorder=3)
                ax.vlines(xx, r.lo, r.hi, color=INK2, lw=1, zorder=4)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks(range(3), ["EMBER", "SNMF", "SNMF\n+EMBER"])
        ax.set_title(title, fontsize=9, loc="left")
        style(ax)
    axes[0].set_ylim(0, 0.75)
    for ax in axes[1:]:
        ax.set_ylim(-0.1, 0.6)
    h = [plt.Rectangle((0, 0), 1, 1, color=OLD, label="original 4-token scorer"),
         plt.Rectangle((0, 0), 1, 1, color=COLOR["ember"], label="fixed scorer (bar in the method's color)")]
    axes[0].legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.16), ncol=2)
    save(fig, "fig9_scoring_repair.png")


def fig_tradeoff(per):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    p = pd.read_csv(PAIRED)
    p = p[(p.run == "fixed") & p.comparison.isin(METHODS)]
    sim_drop = {"ember": 1 - np.mean([0.95, 0.98, 0.98, 0.98, 1.00]), "snmf": 1 - np.mean([0.73, 0.63, 0.65, 0.68, 0.59]),
                "snmf_ember": 1 - np.mean([0.76, 0.67, 0.68, 0.73, 0.67])}
    for m in METHODS:
        leak = p[(p.comparison == m) & (p.stat == "leak_mean")].iloc[0]
        ax.hlines(sim_drop[m], leak.lo, leak.hi, color=COLOR[m], lw=2, zorder=3)
        ax.plot(leak["mean"], sim_drop[m], marker=MARKER[m], ms=11, color=COLOR[m], mec=SURFACE, mew=1.5, zorder=4)
        ax.text(leak.hi + 0.008, sim_drop[m], LABEL[m], ha="left", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-0.12, 0.42)
    ax.set_ylim(-0.02, 0.42)
    ax.set_xlabel("Mean leak over de/es/fr/ru, 95% CI (right = leaks more)")
    ax.set_ylabel("Similar-domain knowledge lost\n(1 − mean SimDom retention; up = more damage)")
    ax.set_title("No method is good on both axes")
    style(ax)
    ax.xaxis.grid(True, color=GRIDC, lw=0.8)
    ax.plot(0, 0, marker="*", ms=11, color=MUTED, zorder=2)
    ax.text(-0.01, 0.012, "ideal: no leak,\nno damage", ha="right", va="bottom", fontsize=7.5, color=INK2)
    save(fig, "fig7_tradeoff.png")


def fig_protocol():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 54)
    ax.axis("off")
    BLUE_F, BLUE_E = "#e4effc", "#2a78d6"
    GRAY_F, GRAY_E = "#efeeea", "#898781"

    def box(x, y, w, h, fill, edge, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fill, ec=edge,
                                    lw=1.4, ls=ls))

    def arrow(x0, y0, x1, y1, label=None, dy=1.6):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.4, color=INK2))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=8, color=INK2)

    def chips(x, y):
        for k, l in enumerate(LANGS):
            ax.add_patch(FancyBboxPatch((x + k * 4.6, y), 4.0, 3.0, boxstyle="round,pad=0,rounding_size=0.8",
                                        fc=SURFACE, ec=BLUE_E if l == "en" else MUTED, lw=1.1))
            ax.text(x + k * 4.6 + 2.0, y + 1.5, l, ha="center", va="center", fontsize=8.5, color=INK)

    # row 1: the edit path (English only -> one edit -> 5 languages)
    y1, h = 28, 22
    box(1, y1, 28, h, BLUE_F, BLUE_E)
    ax.text(15, y1 + h - 2.5, "English data only", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(15, y1 + h - 7.5, "QA_train\nSimdomQA_train\nMMLU (50 q)\nAlpaca answers (judge)", ha="center",
            va="top", fontsize=8.8, color=INK, linespacing=1.5)
    box(40, y1, 24, h, BLUE_F, BLUE_E)
    ax.text(52, y1 + h - 2.5, "Edited model", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(52, y1 + h - 7.5, "one edit, applied once,\nnever re-tuned\nper language", ha="center", va="top",
            fontsize=8.8, color=INK, linespacing=1.5)
    box(75, y1, 28, h, BLUE_F, BLUE_E)
    ax.text(89, y1 + h - 2.5, "Held-out questions", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(89, y1 + h - 7.5, "QA_test + SimdomQA_test\nprompt fully in the language", ha="center", va="top",
            fontsize=8.8, color=INK, linespacing=1.5)
    chips(89 - 11.5, y1 + 2.8)
    arrow(29, y1 + h / 2, 40, y1 + h / 2, "select")
    arrow(64, y1 + h / 2, 75, y1 + h / 2, "ask")

    # row 2: the unedited model gives the denominators
    y2 = 3
    box(40, y2, 24, h, GRAY_F, GRAY_E)
    ax.text(52, y2 + h - 2.5, "Unedited model", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(52, y2 + h - 7.5, "no edit:\ngives the denominators", ha="center", va="top", fontsize=8.8, color=INK,
            linespacing=1.5)
    box(75, y2, 28, h, GRAY_F, GRAY_E)
    ax.text(89, y2 + h - 2.5, "Same questions", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(89, y2 + h - 7.5, "same items,\nsame option order", ha="center", va="top", fontsize=8.8, color=INK)
    chips(89 - 11.5, y2 + 2.8)
    arrow(64, y2 + h / 2, 75, y2 + h / 2, "ask")
    ax.text(15, y2 + h / 2, "QA_test is never used\nfor selecting the edit", ha="center", va="center", fontsize=8.8,
            color=INK2, style="italic", linespacing=1.5)

    # result
    box(108, 11, 21, 32, SURFACE, INK)
    ax.text(118.5, 40.5, "Result", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(118.5, 35, "retention r_L", ha="center", va="top", fontsize=9.2, fontweight="bold")
    ax.text(118.5, 31.2, "edited vs unedited\naccuracy, in the\nsame language", ha="center", va="top", fontsize=8.5,
            color=INK, linespacing=1.4)
    ax.text(118.5, 20.6, "leak Δ_L", ha="center", va="top", fontsize=9.2, fontweight="bold")
    ax.text(118.5, 16.8, "= r_L − r_en", ha="center", va="top", fontsize=8.5, color=INK)
    arrow(103, y1 + h / 2, 108, 34)
    arrow(103, y2 + h / 2, 108, 20)
    save(fig, "fig0_protocol.png")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    base, per, sel = load_md()
    grid, gbase = load_grid()
    print("Unedited QA_test mean per language:", base.mean().round(3).to_dict())
    print("Selection means:\n", sel)
    fig_protocol()
    fig_baseline(base)
    fig_selection(sel)
    print("EMBER delta sweep (mean over 18 concepts):")
    fig_delta_sweep(grid, gbase)
    print("Retention per language (mean [95% CI]):")
    fig_retention(per)
    fig_leak_forest(per)
    fig_collateral(per)
    fig_heatmap(per)
    fig_repair()
    fig_tradeoff(per)
    print("Per-concept mean leak (concept convention):")
    print(pd.DataFrame({m: per[m].leak_mean for m in METHODS}).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
