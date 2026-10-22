"""Figures of the extrapolation benchmark (PDF + PNG, plotted data saved next to each figure).

    .venv/bin/python -m experiments.paper_analysis.extrapolation.figures

fig_precision_at_k        P@k (share of concept-specific tokens, blind labels) among the edit candidates C* for the
                          four scores, one panel per score, one line per dictionary; dotted: random-order base rate
fig_precision_at_k_all    same over all new tokens (C*, B*, N*), balanced (top row) and natural (bottom row) ratio
fig_precision_vs_pagefreq (a) page-count distribution of new page tokens vs concept-only V* tokens;
                          (b) S share by page count: in-sample support tokens vs new tokens admitted by RS/GS re-solve
fig_admission             mean admitted new tokens per run by label, per admission rule and dictionary (balanced)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.paper_analysis.extrapolation.analyze import FBINS, KS, boot_ci, fbin  # noqa: E402
from experiments.paper_analysis.extrapolation.common import COLORS, CONCEPTS, DICTS, OUT, TOKDIR  # noqa: E402
from experiments.paper_analysis.extrapolation.score import RULES, SCORES  # noqa: E402
from experiments.varwidth.common import slug  # noqa: E402

FIG = OUT / "figures"
STYLE = {"EMBER": dict(ls="-", marker="o"), "CS": dict(ls="--", marker="s"), "CINDER": dict(ls="-", marker="^")}
LABEL = {"EMBER": "EMBER (raw, RS-1%)", "CS": "CS + RS-1%", "CINDER": "CINDER (CS + GS)"}
SCORE_NAME = {"proj": "projection", "cos": "cosine", "coef": "joint LS coefficient", "share": "code share"}
RULE_NAME = {"frac": "fraction\n(RS/GS 1%)", "width": "width\nheld", "cutoff": "in-sample\ncut-off",
             "argmax": "dominant\nfeature", "share50": "share\n>= 0.5"}
STACK = {"S": "#003D2C", "A": "#136F53", "U page": "#2FA27D", "N*": "#6DBF9F"}   # ordinal ramp (validated)


def setup():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "grid.linewidth": 0.5,
                         "grid.color": "#e5e5e5", "pdf.fonttype": 42, "ps.fonttype": 42})
    FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name, data):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (FIG / f"{name}.json").write_text(json.dumps(data, indent=1))


def concept_means(rows, met, D, cond):
    acc = defaultdict(list)
    for r in rows:
        if r["dict"] == D and r["cond"] == cond and r.get(met, "") not in ("", "nan"):
            acc[r["concept"]].append(float(r[met]))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def p_at_k(rows, pref, conds, name, base_key):
    fig, axes = plt.subplots(len(conds), 4, figsize=(6.8, 1.85 * len(conds)), sharey=True, squeeze=False)
    data = {}
    for i, cond in enumerate(conds):
        base = np.mean(list({r["concept"]: float(r[base_key]) for r in rows if r["cond"] == cond}.values()))
        data[f"{cond}|base"] = base
        for j, sc in enumerate(SCORES):
            ax = axes[i, j]
            for D in DICTS:
                m, lo, hi = [], [], []
                for k in KS:
                    v = list(concept_means(rows, f"{sc}_{pref}P{k}", D, cond).values())
                    m.append(np.mean(v))
                    a, b = boot_ci(v, n=2000)
                    lo.append(a)
                    hi.append(b)
                data[f"{cond}|{sc}|{D}"] = {"k": KS, "mean": m, "ci_lo": lo, "ci_hi": hi}
                ax.fill_between(KS, lo, hi, color=COLORS[D], alpha=0.10, lw=0)
                ax.plot(KS, m, color=COLORS[D], lw=1.6, markevery=[0, 4, 9, 14, 19], ms=4,
                        markeredgecolor="white", markeredgewidth=0.6, label=LABEL[D], **STYLE[D])
            ax.axhline(base, color="black", ls=":", lw=1, label="random order")
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(axis="y")
            ax.set_ylim(0, 0.6)
            if i == 0:
                ax.set_title(SCORE_NAME[sc])
            if i == len(conds) - 1:
                ax.set_xlabel("k (top-k new tokens)")
            if j == 0:
                lab = "precision@k"
                ax.set_ylabel(lab if len(conds) == 1 else f"{cond}\n{lab}")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06 if len(conds) == 1
                                                                                  else -0.02))
    fig.tight_layout(rect=(0, 0.06 if len(conds) == 1 else 0.03, 1, 1))
    save(fig, name, data)


def freq_figure():
    fr = list(csv.DictReader(open(OUT / "freq.csv")))
    names = [b for _, _, b in FBINS]
    # (a) page-count distribution, concept means
    dist = {"new page tokens": [], "concept-only V* tokens": []}
    for c in CONCEPTS:
        td = json.loads((TOKDIR / f"{slug(c)}.json").read_text())
        toks = td["tokens"]
        newp = [toks[str(t)]["c_page"] for t in td["new"] if toks[str(t)]["c_page"] > 0]
        vC = [toks[str(t)]["c_page"] for t in td["vstar"] if toks[str(t)]["vcode"] == 1]
        for key, arr in (("new page tokens", newp), ("concept-only V* tokens", vC)):
            b = [fbin(x) for x in arr]
            dist[key].append([b.count(n) / len(b) for n in names])
    distm = {k: np.mean(v, 0).tolist() for k, v in dist.items()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.2), gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.arange(len(names))
    w = 0.38
    for i, (key, col) in enumerate((("concept-only V* tokens", STACK["A"]), ("new page tokens", STACK["N*"]))):
        a1.bar(x + (i - 0.5) * (w + 0.02), distm[key], w, color=col, label=key)
    a1.set_xticks(x, names)
    a1.set_xlabel("occurrences on the page")
    a1.set_ylabel("share of tokens")
    a1.set_title("(a) page frequency", loc="left")
    a1.grid(axis="y")
    a1.legend(frameon=False, loc="upper right")
    # (b) S share per bin: in-sample support vs new admitted (frac), per dictionary; base = all new page tokens
    data = {"distribution": {"bins": names, **distm}}

    def curve(kind, D):
        out_m, out_lo, out_hi, ok = [], [], [], []
        for b in names:
            pc = defaultdict(list)
            for r in fr:
                if r["kind"] == kind and r["dict"] == D and fbin(int(r["c_page"])) == b:
                    pc[r["concept"]].append(r["label"] == "S")
            v = [np.mean(z) for z in pc.values()]
            ok.append(len(v) >= 5)
            out_m.append(np.mean(v) if v else np.nan)
            lo, hi = boot_ci(v, n=2000) if v else (np.nan, np.nan)
            out_lo.append(lo)
            out_hi.append(hi)
        return np.array(out_m), np.array(out_lo), np.array(out_hi), np.array(ok)
    for D in ("EMBER", "CINDER"):
        for kind, ls, mfc, lab in (("in-sample support", "-", None, "in-sample support"),
                                   ("new admitted (frac)", "--", "white", "new, admitted")):
            m, lo, hi, ok = curve(kind, D)
            xx = x[ok]
            a2.plot(xx, m[ok], color=COLORS[D], ls=ls, lw=1.6, marker=STYLE[D]["marker"], ms=4.5,
                    markerfacecolor=mfc or COLORS[D], markeredgecolor=COLORS[D] if mfc else "white",
                    label=f"{D}: {lab}")
            data[f"{D}|{kind}"] = {"bins": names, "mean": m.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
                                   "plotted (>=5 concepts)": ok.tolist()}
    m, lo, hi, ok = curve("new page (all)", "-")
    a2.plot(x[ok], m[ok], color="black", ls=":", lw=1.2, marker="x", ms=4, label="all new page tokens")
    data["new page (all)"] = {"bins": names, "mean": m.tolist(), "plotted (>=5 concepts)": ok.tolist()}
    a2.set_xticks(x, names)
    a2.set_xlabel("occurrences on the page")
    a2.set_ylabel("concept-specific share")
    a2.set_ylim(0, 0.85)
    a2.grid(axis="y")
    a2.set_title("(b) precision by page frequency", loc="left")
    a2.legend(frameon=False, loc="upper left", ncol=1, fontsize=6.5, handlelength=3.2)
    fig.tight_layout()
    save(fig, "fig_precision_vs_pagefreq", data)


def admission_figure():
    ad = list(csv.DictReader(open(OUT / "admit.csv")))
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    data = {}
    xpos, xt, xl, bar_x, bar_l = 0, [], [], [], []
    for rule in RULES:
        start = xpos
        for D in DICTS:
            vals = {}
            for lab, col in (("S", "nS"), ("A", "nA"), ("U page", "nU_page"), ("N*", "nNstar")):
                pc = defaultdict(list)
                for r in ad:
                    if r["dict"] == D and r["rule"] == rule and r["cond"] == "balanced":
                        pc[r["concept"]].append(float(r[col]))
                vals[lab] = float(np.mean([np.mean(v) for v in pc.values()]))
            data[f"{rule}|{D}"] = vals
            bottom = 0
            for lab, col in STACK.items():
                ax.bar(xpos, vals[lab], 0.8, bottom=bottom, color=col, edgecolor="white", linewidth=1,
                       label=lab if (rule == RULES[0] and D == "EMBER") else None)
                bottom += vals[lab]
            bar_x.append(xpos)
            bar_l.append({"EMBER": "EMBER", "CS": "CS", "CINDER": "CINDER"}[D])
            xpos += 1
        xt.append((start + xpos - 1) / 2)
        xl.append(RULE_NAME[rule].replace("\n", " "))
        xpos += 0.8
    ax.set_xticks(bar_x, bar_l, rotation=90, fontsize=6)
    for x0, lab in zip(xt, xl):
        ax.text(x0, -0.30, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.set_ylabel("admitted new tokens per run")
    ax.grid(axis="y")
    h, _ = ax.get_legend_handles_labels()
    ax.legend(h[::-1], ["new neutral token (N*)", "unrelated page token (U)", "adjacent (A)",
                        "concept-specific (S)"], frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_admission", data)


def main():
    setup()
    rows = list(csv.DictReader(open(OUT / "runs_metrics.csv")))
    p_at_k(rows, "c", ["balanced"], "fig_precision_at_k", "base_cS")
    p_at_k(rows, "", ["balanced", "natural"], "fig_precision_at_k_all", "base_S")
    freq_figure()
    admission_figure()
    print(f"-> {FIG}")


if __name__ == "__main__":
    main()
