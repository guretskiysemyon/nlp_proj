#!/usr/bin/env python3
"""Figures and tables for Part2_report.md (Part 2: CINDER and extrapolation vs EMBER, English-fitted edits).

Inputs (all tracked in git, CPU only, seconds):
  - judge_rerun/Fair/report.csv                       fair run, 18 concepts x arms (final test, 5 languages)
  - Part 2 Experiments/x13_fair_analysis/{stats,coverage,mechanism}.csv
  - Part 2 Experiments/x8_test_mechanism/split.csv    touched vs untouched questions (6 concepts)
  - Part 2 Experiments/x12_token_control/results.csv  token-choice control (6 concepts)

Writes PNGs to Part2_report/figures/ and prints the numbers and markdown tables the report quotes.

Usage (from the repo root; needs matplotlib, numpy, pandas):
    python report/Part2_report/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
P2X = ROOT / "Part 2 Experiments"
FAIR = ROOT / "judge_rerun" / "Fair" / "report.csv"

LANGS = ["en", "de", "es", "fr", "ru"]
FOR = ["de", "es", "fr", "ru"]
UNRELIABLE = {("Harry Potter", "ru"), ("Valentine's Day", "ru")}  # unedited QA_test accuracy < 0.40
ARMS = ["ember", "cinder", "ember_hj", "ember_x", "cinder_x", "ml_sep", "ml_joint"]
LABEL = {"ember": "EMBER", "cinder": "CINDER", "ember_hj": "EMBER+TWJ", "ember_x": "EMBER+X",
         "cinder_x": "CINDER+X", "ml_sep": "Native per-language", "ml_joint": "Native joint"}

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
BLUE_L, ORANGE_L = "#86b6ef", "#f4a98a"
SURFACE, INK, INK2, MUTED, GRIDC, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GRAY = "#b4b2a9"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}")


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)


def boot(v, n=10_000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    idx = np.random.default_rng(0).integers(0, len(v), (n, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return v.mean(), lo, hi


# ---------------------------------------------------------------- data
def load_fair():
    df = pd.read_csv(FAIR)
    df = df[df.method.isin(ARMS)].copy()
    for l in FOR:
        bad = [(c, l) in UNRELIABLE for c in df.concept]
        df.loc[bad, f"ret_{l}"] = np.nan
    df["ret_for"] = df[[f"ret_{l}" for l in FOR]].mean(axis=1, skipna=True)
    df["leak_rel"] = df.ret_for - df.ret_en  # mean over reliable languages, minus English
    return df


# ---------------------------------------------------------------- figure 1: methods diagram
def fig_methods():
    fig, ax = plt.subplots(figsize=(11.4, 4.7))
    ax.set_xlim(0, 142)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def box(x, y, w, h, fill, edge, text, bold=False, fs=8.6, color=INK, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fill, ec=edge, lw=1.3,
                                    ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK2))

    heads = ["1  Matrix", "2  Factorization", "3  Select features", "4  Extrapolate", "5  Edit"]
    xs = [22, 47, 72, 97, 122]
    w = 21
    wl = 19
    for x, h in zip(xs, heads):
        ax.text(x + (w if x != 122 else wl) / 2, 55, h, ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=INK2)
    GF, GE = "#efeeea", GRAY
    rows = [
        ("EMBER\n(Part 1)", 38, [
            ("embeddings of\nconcept + neutral\ntokens", GF, GE),
            ("sparse MF,\ntop 1% of tokens\nper feature", GF, GE),
            ("LLM judge on a\none-sentence\nsummary", GF, GE),
            None,
            ("subtract concept\nfeatures from\nconcept tokens", GF, GE)]),
        ("CINDER", 21, [
            ("Contrastive\nSharpening (CS)\nwhiten, stretch\nconcept directions", "#fde6dc", ORANGE),
            ("Global Sparsity (GS)\none token budget\nfor the whole\nmatrix", "#fde6dc", ORANGE),
            ("Two-Way Judge\n(TWJ): summary\nOR concept\nlexicon match", "#fde6dc", ORANGE),
            None,
            ("same edit;\ndirections\nmapped back", GF, GE)]),
        ("+X\n(either method)", 4, [
            ("as the parent\n(English only)", GF, GE),
            ("as the parent\n(English only)", GF, GE),
            ("as the parent\n(English only)", GF, GE),
            ("features fixed;\nsolve coefficients\nfor native de/es/\nfr/ru tokens;\nadd concept-only", "#d8f1e8", AQUA),
            ("English edit\nkept + added\nforeign tokens", "#d8f1e8", AQUA)]),
    ]
    for label, y, cells in rows:
        h = 15
        ax.text(1, y + h / 2, label, ha="left", va="center", fontsize=9.5, fontweight="bold")
        prev = None
        for i, cell in enumerate(cells):
            if cell is None:
                prev = None
                continue
            text, fill, edge = cell
            x = xs[i]
            box(x, y, w if i < 4 else wl, h, fill, edge, text)
            if prev is not None:
                arrow(prev, y + h / 2, x, y + h / 2)
            prev = x + (w if i < 4 else wl)
        if cells[3] is None:  # skip the extrapolation column: draw a long arrow from step 3 to step 5
            arrow(xs[2] + w, y + h / 2, xs[4], y + h / 2)
            ax.text((xs[2] + w + xs[4]) / 2, y + h / 2 + 1.6, "no extrapolation", ha="center", va="bottom",
                    fontsize=8, color=MUTED)
    ax.text(71, -2.6, "Orange: what CINDER changes.  Green: what extrapolation adds.  All features are fitted on English text only.",
            ha="center", va="center", fontsize=8.6, color=INK2)
    save(fig, "fig1_methods.png")


# ---------------------------------------------------------------- figure 2: English retention, CINDER vs EMBER
def fig_english(df):
    p = df.pivot(index="concept", columns="method", values="ret_en")
    noedit = df.pivot(index="concept", columns="method", values="no_edit")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0.3, 1.04], [0.3, 1.04], color=AXIS, lw=1, zorder=1)
    for c in p.index:
        x, y = p.loc[c, "ember"], p.loc[c, "cinder"]
        ne = bool(noedit.loc[c, "ember"]) or bool(noedit.loc[c, "cinder"])
        ax.scatter(x, y, s=46, facecolor=SURFACE if ne else BLUE, edgecolor=BLUE, lw=1.4, zorder=3)
    lab = {"Culture of Greece": (6, -3, "left"), "Cannabis": (0, 8, "center"), "Gambling": (6, 3, "left"),
           "Gun": (6, -9, "left"), "Nazism": (-3, 8, "right"), "Republic of Ireland": (6, -14, "left"),
           "World War II": (0, 8, "center"), "Halloween": (-8, -3, "right"), "Heroin": (-8, 4, "right"),
           "Uranium": (-8, -9, "right"), "Valentine's Day": (-8, 0, "right"), "Golf": (-6, 6, "right"),
           "Ancient Rome": (6, 2, "left")}
    for c, (dx, dy, ha) in lab.items():
        ax.annotate(c, (p.loc[c, "ember"], p.loc[c, "cinder"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.2, color=INK2, ha=ha)
    ax.set_xlim(0.3, 1.04)
    ax.set_ylim(0.3, 1.08)
    ax.set_xlabel("EMBER: English retention")
    ax.set_ylabel("CINDER: English retention")
    ax.set_title("English erasure, concept by concept")
    ax.text(0.31, 1.06, "CINDER erases less\n(above the line)", fontsize=8, color=MUTED, va="top")
    ax.text(1.02, 0.33, "CINDER erases more\n(below the line)", fontsize=8, color=MUTED, ha="right")
    grid(ax, "both")
    ax.scatter([], [], s=46, facecolor=BLUE, edgecolor=BLUE, label="both methods edit")
    ax.scatter([], [], s=46, facecolor=SURFACE, edgecolor=BLUE, lw=1.4, label="one method makes no edit (retention 1)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    save(fig, "fig4_english_retention.png")


# ---------------------------------------------------------------- figure 3: retention by language, arms
def fig_retention(df):
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    x = np.arange(5)
    style = {"ember": (BLUE, "o", True), "ember_x": (BLUE, "o", False), "cinder": (ORANGE, "s", True),
             "cinder_x": (ORANGE, "s", False)}
    for k, a in enumerate(["ember", "ember_x", "cinder", "cinder_x"]):
        d = df[df.method == a]
        st = np.array([boot(d[f"ret_{l}"]) for l in LANGS])
        color, mk, filled = style[a]
        xs = x + (k - 1.5) * 0.17
        ax.vlines(xs, st[:, 1], st[:, 2], color=color, lw=1.6, zorder=3)
        ax.plot(xs, st[:, 0], ls="none", marker=mk, ms=7.5, color=color, mfc=color if filled else SURFACE, mew=1.6,
                zorder=4, label=LABEL[a])
        print(f"  {LABEL[a]:9s}", " ".join(f"{l} {v[0]:.2f}" for l, v in zip(LANGS, st)))
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.text(4.45, 1.012, "unedited model = 1", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, ["English\n(fitted here)", "German", "Spanish", "French", "Russian"])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0.5, 1.08)
    ax.set_ylabel("Retention (lower = more erased)")
    ax.set_title("Concept retention by question language, 18 concepts")
    grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    save(fig, "fig2_retention_by_language.png")


# ---------------------------------------------------------------- figure 9: paired differences (X14)
def fig_paired():
    s = pd.read_csv(P2X / "x14_followup" / "stats.csv")
    order = [("ember_x − ember", "EMBER+X − EMBER"), ("cinder_x − cinder", "CINDER+X − CINDER"),
             ("cinder − ember", "CINDER − EMBER"), ("cinder_x − ember_x", "CINDER+X − EMBER+X"),
             ("ember_hj − ember", "EMBER+TWJ − EMBER"), ("cinder − ember_hj", "CINDER − EMBER+TWJ"),
             ("ml_sep − ember", "Native per-lang. − EMBER"), ("ml_sep − ember_x", "Native per-lang. − EMBER+X")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (block, title) in zip(axes, [("leak, all 18", "(a) Leak"), ("foreign_ret, all 18", "(b) Foreign retention")]):
        for i, (key, lab) in enumerate(order):
            r = s[(s.analysis == block) & (s.pair == key)].iloc[0]
            y = len(order) - 1 - i
            col = BLUE if r.holm_p < 0.05 else GRAY
            ax.hlines(y, r.ci_low, r.ci_high, color=col, lw=2.2, zorder=3)
            ax.plot(r["mean"], y, "o", ms=8, color=col, mec=SURFACE, mew=1.5, zorder=4)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title)
        ax.set_xlabel("Difference (negative: first arm erases more / leaks less)", fontsize=8)
        ax.xaxis.grid(True, color=GRIDC, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)), [o[1] for o in order][::-1])
    axes[0].set_xlim(-0.14, 0.16)
    axes[1].set_xlim(-0.30, 0.17)
    axes[1].scatter([], [], color=BLUE, s=40, label="survives Holm correction (p < 0.05)")
    axes[1].scatter([], [], color=GRAY, s=40, label="does not")
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.05, -0.2), ncol=2)
    save(fig, "fig3_paired_differences.png")


# ---------------------------------------------------------------- figure 10: all arms, English vs foreign
def fig_all_arms(df):
    style = {"ember": (BLUE, "o", True), "ember_x": (BLUE, "o", False), "ember_hj": (BLUE, "D", True),
             "cinder": (ORANGE, "s", True), "cinder_x": (ORANGE, "s", False),
             "ml_sep": (AQUA, "^", True), "ml_joint": (AQUA, "^", False)}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.25})
    ax = axes[0]
    ax.plot([0.5, 1.0], [0.5, 1.0], color=AXIS, lw=1, zorder=1)
    off = {"ember": (6, 5, "left"), "ember_x": (6, -11, "left"), "ember_hj": (-6, -12, "right"),
           "cinder": (6, 5, "left"), "cinder_x": (6, -11, "left"), "ml_sep": (8, -3, "left"), "ml_joint": (-8, 6, "right")}
    rows = []
    for a, (color, mk, filled) in style.items():
        d = df[df.method == a]
        x = boot(d.ret_en)
        y = boot(d.ret_for)
        ax.plot(x[0], y[0], ls="none", marker=mk, ms=9, color=color, mfc=color if filled else SURFACE, mew=1.7, zorder=4)
        dx, dy, ha = off[a]
        ax.annotate(LABEL[a], (x[0], y[0]), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK, ha=ha)
        rows.append((a, x[0], y[0], int(d.no_edit.sum())))
    ax.set_xlim(0.54, 0.96)
    ax.set_ylim(0.64, 0.95)
    ax.set_xlabel("English retention")
    ax.set_ylabel("Foreign retention (mean of de/es/fr/ru)")
    ax.set_title("(a) All arms, 18 concepts")
    ax.text(0.955, 0.655, "above the line: the concept survives better\nin foreign languages than in English (leak)", fontsize=7.6, color=MUTED, va="bottom", ha="right")
    grid(ax, "both")
    # (b) retention by language
    ax = axes[1]
    x = np.arange(5)
    arms = ["ember", "ember_x", "ml_sep", "ml_joint"]
    for k, a in enumerate(arms):
        d = df[df.method == a]
        st = np.array([boot(d[f"ret_{l}"]) for l in LANGS])
        color, mk, filled = style[a]
        xs = x + (k - 1.5) * 0.17
        ax.vlines(xs, st[:, 1], st[:, 2], color=color, lw=1.6, zorder=3)
        ax.plot(xs, st[:, 0], ls="none", marker=mk, ms=7.5, color=color, mfc=color if filled else SURFACE, mew=1.6,
                zorder=4, label=LABEL[a])
        print(f"  {LABEL[a]:20s}", " ".join(f"{l} {v[0]:.2f}" for l, v in zip(LANGS, st)))
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.set_xticks(x, ["en", "de", "es", "fr", "ru"])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0.4, 1.06)
    ax.set_ylabel("Retention (lower = more erased)")
    ax.set_title("(b) Native-text arms by language")
    grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "fig10_all_arms.png")
    for r in rows:
        print(f"  {LABEL[r[0]]:20s} ret_en {r[1]:.3f} foreign {r[2]:.3f} leak {r[2]-r[1]:+.3f} no_edit {r[3]}")


# ---------------------------------------------------------------- touched vs untouched (all edited concepts)
def fig_touched():
    """Parent edits (EMBER, CINDER) on the held-out QA_test of every concept they edit, from the question-level records
    of the extrapolated arms: a question is touched when the parent's edit changes a token of its text or correct answer."""
    d = pd.read_csv(P2X / "x13_fair_analysis" / "mechanism.csv")
    d["touched"] = d.group.eq("old")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True, gridspec_kw={"wspace": 0.08})
    langs = ["en", "de", "es", "fr", "ru"]
    summary = []
    for ax, (arm, name) in zip(axes, [("ember_x", "EMBER"), ("cinder_x", "CINDER")]):
        a = d[d.arm == arm]
        for j, (flag, col, lab) in enumerate([(True, BLUE, "question touched"), (False, GRAY, "question untouched")]):
            vals, ns = [], []
            for l in langs:
                x = a[(a.lang == l) & (a.touched == flag)]
                vals.append(100 * x.lost_parent.mean() if len(x) else np.nan)
                ns.append(len(x))
            xs = np.arange(5) + (j - 0.5) * 0.36
            ok = np.array([n >= 10 for n in ns])
            ax.bar(xs[ok], np.array(vals)[ok], width=0.32, color=col, zorder=3, label=lab)
            for xx, v, n in zip(xs, vals, ns):
                if n >= 10:
                    ax.text(xx, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
            summary += [(name, l, "touched" if flag else "untouched", n, v) for l, n, v in zip(langs, ns, vals)]
        ax.set_xticks(range(5), langs)
        ax.set_title(f"{name} ({a.concept.nunique()} concepts)")
        grid(ax)
    axes[0].set_ylabel("Correct answers turned wrong (%)")
    axes[1].legend(loc="upper right")
    fig.suptitle("The edit only changes questions that contain an edited token", x=0.125, ha="left", y=1.02,
                 fontsize=10, fontweight="bold")
    save(fig, "fig5_touched_untouched.png")
    print(pd.DataFrame(summary, columns=["parent", "lang", "group", "n", "lost%"]).round(1).to_string())


# ---------------------------------------------------------------- figure 6: coverage vs leak
def fig_coverage():
    c = pd.read_csv(P2X / "x13_fair_analysis" / "coverage.csv")
    rho = c.coverage.rank().corr(c.leak.rank())
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.scatter(c.coverage, c.leak, s=16, color=BLUE, alpha=0.32, edgecolor="none", zorder=2)
    bins = [0, 0.05, 0.2, 0.4, 0.6, 1.01]
    c["bin"] = pd.cut(c.coverage, bins, right=False)
    b = c.groupby("bin", observed=True).agg(cov=("coverage", "mean"), leak=("leak", "mean"), n=("leak", "size"))
    ax.plot(b["cov"], b["leak"], "-o", color=INK, lw=2, ms=7, mec=SURFACE, mew=1.5, zorder=4)
    for _, r in b.iterrows():
        ax.text(r["cov"] + 0.03, r["leak"] + 0.05, f"{r['leak']:+.2f}  (n={int(r['n'])})", ha="left", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Question coverage in the language (share of questions with an edited token)")
    ax.set_ylabel("Leak in that language")
    ax.set_title(f"Less coverage, more leak (Spearman {rho:+.2f})")
    ax.set_xlim(-0.03, 1.03)
    grid(ax, "both")
    save(fig, "fig6_coverage_leak.png")
    print(f"  coverage vs leak: n={len(c)}, rank correlation {rho:+.3f}")
    print(b.round(3).to_string())


# ---------------------------------------------------------------- figure 7: coverage gain by extrapolation
def fig_extrap_coverage():
    c = pd.read_csv(P2X / "x13_fair_analysis" / "coverage.csv")
    m = c[c.arm.isin(["ember", "ember_x", "cinder", "cinder_x"])].groupby(["arm", "lang"]).coverage.mean().unstack()
    print(m.round(2).to_string())
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(4)
    spec = [("ember", BLUE_L, "EMBER"), ("ember_x", BLUE, "EMBER+X"), ("cinder", ORANGE_L, "CINDER"),
            ("cinder_x", ORANGE, "CINDER+X")]
    for k, (a, col, lab) in enumerate(spec):
        vals = [m.loc[a, l] for l in FOR]
        xs = x + (k - 1.5) * 0.2
        ax.bar(xs, vals, width=0.18, color=col, zorder=3, label=lab)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.8, color=INK, rotation=90, va="bottom")
    ax.set_xticks(x, ["German", "Spanish", "French", "Russian"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Question coverage")
    ax.set_title("Extrapolation lets the edit reach foreign questions")
    grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    save(fig, "fig7_extrapolation_coverage.png")


# ---------------------------------------------------------------- figure 8: mechanism of +X
def fig_extrap_mechanism():
    d = pd.read_csv(P2X / "x13_fair_analysis" / "mechanism.csv")
    d = d[d.lang_kind == "foreign"] if "lang_kind" in d.columns else d[d.lang != "en"]
    g = d.groupby(["arm", "group"])[["lost_parent", "lost_x"]].mean() * 100
    n = d.groupby(["arm", "group"]).size()
    print(g.round(1).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True, gridspec_kw={"wspace": 0.08})
    groups = [("new", "only an added\ntoken"), ("old", "an original\nEMBER/CINDER token"), ("none", "no edited\ntoken")]
    for ax, arm, col in zip(axes, ["ember_x", "cinder_x"], [BLUE, ORANGE]):
        for j, (key, colr, lab) in enumerate([("lost_parent", GRAY, "parent edit alone"), ("lost_x", col, "with extrapolation")]):
            vals = [g.loc[(arm, gr), key] for gr, _ in groups]
            xs = np.arange(3) + (j - 0.5) * 0.36
            ax.bar(xs, vals, width=0.32, color=colr, zorder=3, label=lab)
            for xx, v in zip(xs, vals):
                ax.text(xx, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.8, color=INK)
        ax.set_xticks(range(3), [lab for _, lab in groups], fontsize=8)
        ax.set_title(LABEL[arm])
        grid(ax)
    axes[0].set_ylabel("Foreign questions answered wrong (%)")
    axes[0].set_ylim(0, 56)
    axes[1].legend(loc="upper right", ncol=1)
    fig.suptitle("The extra erasure sits in the questions that extrapolation newly reaches", x=0.125, ha="left",
                 y=1.02, fontsize=10, fontweight="bold")
    save(fig, "fig8_extrapolation_mechanism.png")


# ---------------------------------------------------------------- figure 9: token control
def fig_control():
    d = pd.read_csv(P2X / "x12_token_control" / "results.csv")
    m = d.groupby("variant")[["ret_en", "ret_foreign", "leak", "sim_en", "sim_foreign"]].mean()
    print(m.round(3).to_string())
    order = ["EMBER", "X", "F_own", "R_own"]
    names = {"EMBER": "EMBER", "X": "X", "F_own": "F_own", "R_own": "R_own"}
    cols = {"EMBER": GRAY, "X": BLUE, "F_own": BLUE_L, "R_own": "#d9d8d1"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"wspace": 0.3})
    for ax, key, title, lab in [(axes[0], "ret_foreign", "(a) Foreign retention (lower = more erased)", ""),
                                (axes[1], "sim_foreign", "(b) Similar-domain accuracy change, foreign", "")]:
        for i, v in enumerate(order):
            ax.bar(i, m.loc[v, key], width=0.62, color=cols[v], zorder=3, hatch="///" if v == "R_own" else None,
                   edgecolor=SURFACE if v != "R_own" else GRAY, lw=0)
            pts = d[d.variant == v][key].values
            ax.scatter(np.full(len(pts), i) + np.linspace(-0.15, 0.15, len(pts)), pts, s=9, color=INK2, zorder=4)
            ax.text(i, m.loc[v, key] + (0.015 if key == "ret_foreign" else -0.004), f"{m.loc[v, key]:.2f}" if key == "ret_foreign" else f"{m.loc[v, key]:+.3f}",
                    ha="center", va="bottom" if key == "ret_foreign" else "top", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
        ax.set_xticks(range(4), [names[v] for v in order], fontsize=8.5)
        ax.set_title(title, fontsize=9)
        grid(ax)
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylim(-0.06, 0.005)
    save(fig, "fig9_token_control.png")


# ---------------------------------------------------------------- tables for the text
def tables(df):
    print("\n### arm means, all 18 concepts (no edit = unchanged model), reliable cells only")
    rows = []
    for a in ARMS:
        d = df[df.method == a]
        r = {"arm": LABEL[a], "no_edit": int(d.no_edit.sum())}
        for l in LANGS:
            r[l] = d[f"ret_{l}"].mean()
        lk = boot(d.leak_rel)
        r["leak"] = f"{lk[0]:+.2f} [{lk[1]:+.2f}, {lk[2]:+.2f}]"
        r["simdom_en"] = d.simdom_en.mean()
        r["mmlu"] = d.mmlu_test.mean()
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_markdown(index=False) if hasattr(pd.DataFrame, "to_markdown") else pd.DataFrame(rows).round(3).to_string())
    common = set(df.concept.unique())
    for a in ["ember", "cinder", "ember_x", "cinder_x"]:
        common -= set(df[(df.method == a) & df.no_edit.astype(bool)].concept)
    print(f"\n### the {len(common)} concepts that all four arms edit:", sorted(common))
    rows = []
    for a in ARMS:
        d = df[(df.method == a) & df.concept.isin(common)]
        r = {"arm": LABEL[a]}
        for l in LANGS:
            r[l] = d[f"ret_{l}"].mean()
        lk = boot(d.leak_rel)
        r["leak"] = f"{lk[0]:+.2f} [{lk[1]:+.2f}, {lk[2]:+.2f}]"
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string())
    print("\n### per concept: n_features / tokens / delta / ret_en / leak (reliable)")
    for c in sorted(df.concept.unique()):
        line = []
        for a in ARMS:
            r = df[(df.concept == c) & (df.method == a)].iloc[0]
            line.append(f"{a}: {'none' if r.no_edit else f'{int(r.n_features)}f/{int(r.n_tokens_edited)}t d={r.delta:g}'} ret_en={r.ret_en:.2f} leak={r.leak_rel:+.2f}")
        print(c, "|", " | ".join(line))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    df = load_fair()
    fig_methods()
    fig_english(df)
    print("Retention by language (mean over concepts):")
    fig_retention(df)
    fig_paired()
    fig_all_arms(df)
    print("Touched vs untouched:")
    fig_touched()
    print("Coverage vs leak:")
    fig_coverage()
    print("Coverage by arm:")
    fig_extrap_coverage()
    print("Mechanism of +X:")
    fig_extrap_mechanism()
    print("Token control:")
    fig_control()
    tables(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
