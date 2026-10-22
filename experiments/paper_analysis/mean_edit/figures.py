#!/usr/bin/env python
"""WP6b mean_edit figures (PDF + PNG, plotted data as CSV next to each figure).

    python -m experiments.paper_analysis.mean_edit.figures

fig_mean_curves      validation efficacy and collateral vs r, CINDER family (left) and EMBER family (right)
fig_eff_vs_damage    efficacy vs copy (grid points r <= 2) and efficacy vs context-MC (sweep r in {0.25, 1, 10, 100})
fig_s2_s4            S2 read gain under modified norm weights; S4 layer-0 write gain vs proxy; S4 output KL
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.paper_analysis.mean_edit.run import OUT  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "pdf.fonttype": 42})
STY = {"CINDER": ("#D55E00", "-", "o"), "CINDER-z": ("#E69F00", "-", "s"), "MEAN-C": ("#009E73", "-", "^"),
       "Noise-C": ("#000000", ":", "x"), "EMBER": ("#7F7F7F", "-", "s"), "EMBER-c": ("#0072B2", "-", "o"),
       "MEAN-E": ("#009E73", "--", "^"), "Noise-E": ("#000000", ":", "x")}
FAM = {"C": ["CINDER", "CINDER-z", "MEAN-C", "Noise-C"], "E": ["EMBER-c", "EMBER", "MEAN-E", "Noise-E"]}
R = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]


def save(fig, name, rows, header):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    plt.close(fig)


def mean_curve(curves, v, key):
    runs = curves.get(v, {})
    xs, ys, ns = [], [], []
    for p in R:
        vals = [d[str(p)][key] for d in runs.values() if str(p) in d and d[str(p)].get(key) is not None]
        if vals:
            xs.append(p), ys.append(float(np.mean(vals))), ns.append(len(vals))
    return xs, ys, ns


def fig_curves(A):
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.0), sharex=True)
    rows = []
    for j, fam in enumerate(("C", "E")):
        cur = A[fam]["curves"]
        fl = [x["qa_train"]["del"] for x in A[fam]["floors"].values() if x and "qa_train" in x]
        for i, key in enumerate(("efficacy", "collateral")):
            a = ax[i, j]
            for v in FAM[fam]:
                xs, ys, ns = mean_curve(cur, v, key)
                if not xs:
                    continue
                c, ls, mk = STY[v]
                a.plot(xs, ys, color=c, ls=ls, marker=mk, ms=3.5, label=f"{v} (n={max(ns)})")
                rows += [[fam, v, key, x, y, n] for x, y, n in zip(xs, ys, ns)]
            if key == "efficacy" and fl:
                a.axhline(np.mean(fl), color="#7F7F7F", lw=0.8, ls="--")
                a.text(0.05, np.mean(fl) + 0.015, "deletion floor", fontsize=6.5, color="#555555")
                rows.append([fam, "deletion floor (qa_train)", key, "", float(np.mean(fl)), len(fl)])
            if key == "collateral":
                a.axhline(0.8, color="#BBBBBB", lw=0.8, ls="-")
            a.set_xscale("log")
            a.set_ylim(-0.02, 1.02)
            a.grid(axis="y", color="#E5E5E5", lw=0.5)
            if j == 0:
                a.set_ylabel({"efficacy": "Efficacy (validation)", "collateral": "Collateral (validation)"}[key])
            if i == 1:
                a.set_xlabel("Edit size r (median |Δe| / |e|)")
            if i == 0:
                a.legend(frameon=False, loc="upper left")
    save(fig, "fig_mean_curves", rows, ["family", "variant", "metric", "r", "mean_over_runs", "n_runs"])


def fig_damage(A):
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    rows = []
    a = ax[0]
    for fam in ("C", "E"):
        cc = A[fam]["copy"]["curves"]
        for v in FAM[fam]:
            if v not in cc:
                continue
            by = defaultdict(lambda: ([], []))
            for d in cc[v].values():
                for p, cp, ef, r in d["points"]:
                    by[p][0].append(cp)
                    by[p][1].append(ef)
            ps = sorted(by)
            x = [np.mean(by[p][0]) for p in ps]
            y = [np.mean(by[p][1]) for p in ps]
            c, ls, mk = STY[v]
            a.plot(x, y, color=c, ls=ls, marker=mk, ms=3.5, label=v)
            rows += [["copy", v, p, xx, yy, len(by[p][0])] for p, xx, yy in zip(ps, x, y)]
    a.set_xlabel("Copy rate of the edited words")
    a.set_ylabel("Efficacy (validation)")
    a.invert_xaxis()
    a.grid(color="#E5E5E5", lw=0.5)
    a.legend(frameon=False, fontsize=6.5, ncol=2)
    a = ax[1]
    for fam in ("C", "E"):
        ig = A[fam]["integrity"]["summary"]
        cur = A[fam]["curves"]
        for v in FAM[fam]:
            if v not in ig:
                continue
            xs, ys = [], []
            for nm in ("0.25", "1", "10", "100"):
                if nm not in ig[v]:
                    continue
                p = str(float(nm))
                effs = [d[p]["efficacy"] for d in cur[v].values() if p in d]
                xs.append(ig[v][nm]["ctxmc"])
                ys.append(float(np.mean(effs)))
                rows.append(["ctxmc", v, float(nm), xs[-1], ys[-1], ig[v][nm]["n"]])
            c, ls, mk = STY[v]
            a.plot(xs, ys, color=c, ls=ls, marker=mk, ms=3.5, label=v)
            for nm, x, y in zip(("0.25", "1", "10", "100"), xs, ys):
                if v == "MEAN-C":
                    a.annotate(f"r={nm}", (x, y), fontsize=6, color="#444444", xytext=(3, -8),
                               textcoords="offset points")
    ref = np.mean([A["C"]["integrity"]["summary"]["CINDER"][nm]["ctxmc_orig"] for nm in ("0.25", "1", "10", "100")])
    a.axvline(ref, color="#BBBBBB", lw=0.8)
    a.axvline(0.25, color="#BBBBBB", lw=0.8, ls=":")
    a.text(ref, 0.02, " real word", fontsize=6.5, color="#555555")
    a.text(0.25, 0.02, " chance", fontsize=6.5, color="#555555")
    a.set_xlabel("Context-MC (unrelated MMLU after the word)")
    a.set_ylabel("Efficacy (validation)")
    a.invert_xaxis()
    a.grid(color="#E5E5E5", lw=0.5)
    save(fig, "fig_eff_vs_damage", rows, ["panel", "variant", "r", "x", "efficacy", "n_runs"])


def fig_s2s4():
    S2 = json.loads((OUT / "s2_read_gain.json").read_text())
    S4 = json.loads((OUT / "s4_forward.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.3))
    rows = []
    a = ax[0]
    items = [("μ_B-hat", S2["mu_hat"]), ("EMBER", {k: x["mean"] for k, x in S2["summary"]["ember"].items()}),
             ("CINDER-z", {k: x["mean"] for k, x in S2["summary"]["cinder_z"].items()}),
             ("CINDER", {k: x["mean"] for k, x in S2["summary"]["cinder"].items()}),
             ("EMBER-c", {k: x["mean"] for k, x in S2["summary"]["ember_c"].items()})]
    cols = {"μ_B-hat": "#009E73", "EMBER": "#7F7F7F", "CINDER-z": "#E69F00", "CINDER": "#D55E00", "EMBER-c": "#0072B2"}
    gk = [("actual", "actual g"), ("g1", "g = 1"), ("top8_zero", "top-8 g = 0")]
    w = 0.16
    for i, (nm, d) in enumerate(items):
        xs = np.arange(3) + (i - 2) * w
        a.bar(xs, [d[k] for k, _ in gk], width=w * 0.9, color=cols[nm], label=nm)
        rows += [["s2", nm, k, d[k]] for k, _ in gk]
    a.axhline(1, color="#888888", lw=0.6)
    a.set_xticks(range(3), [lbl for _, lbl in gk])
    a.set_ylabel("Layer-0 read gain (× random)")
    a.legend(frameon=False, fontsize=6, loc="upper right")
    # S4
    by = defaultdict(dict)
    for r in S4["rows"]:
        by[(r["concept"], r["seed"], r["family"], r["token"], r["scale"])][r["dir"]] = r
    names = {"cinder": "CINDER", "cinder_z": "CINDER-z", "mean_c": "MEAN-C", "ember": "EMBER", "ember_c": "EMBER-c",
             "mean_e": "MEAN-E"}
    a = ax[1]
    for k, d in by.items():
        if k[4] != 0.01:
            continue
        base = np.mean([d[f"rand{i}"]["l0_write"] for i in range(4)])
        for n, x in d.items():
            if n.startswith("rand"):
                continue
            c, _, mk = STY[names[n]]
            a.scatter(x["proxy_qkv"], x["l0_write"] / base, s=9, color=c, marker=mk, lw=0.4,
                      edgecolor="white" if mk != "x" else None)
            rows.append(["s4_l0", names[n], k[3], x["proxy_qkv"], x["l0_write"] / base])
    lim = [0.6, 3.6]
    a.plot(lim, lim, color="#BBBBBB", lw=0.6)
    a.set_xlabel("Linear proxy (× random)")
    a.set_ylabel("Layer-0 write gain (× random)")
    a = ax[2]
    order = ["CINDER", "CINDER-z", "MEAN-C", "EMBER-c", "EMBER", "MEAN-E"]
    inv = {v: k for k, v in names.items()}
    for s_i, (sc, off) in enumerate(((0.01, -0.18), (1.0, 0.18))):
        meds = []
        for j, nm in enumerate(order):
            vals = []
            for k, d in by.items():
                if k[4] != sc or inv[nm] not in d:
                    continue
                base = np.mean([d[f"rand{i}"]["kl_tok"] for i in range(4)])
                vals.append(d[inv[nm]]["kl_tok"] / base)
            meds.append(np.median(vals))
            rows.append(["s4_kl", nm, sc, float(np.median(vals)), len(vals)])
        a.bar(np.arange(6) + off, meds, width=0.34, color=[STY[n][0] for n in order],
              alpha=1.0 if sc == 0.01 else 0.45, edgecolor="#333333", lw=0.4,
              label=f"s = {sc:g}")
    a.axhline(1, color="#888888", lw=0.6)
    a.set_xticks(range(6), order, rotation=45, ha="right")
    a.set_ylabel("Token next-token KL (× random)")
    a.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_s2_s4", rows, ["panel", "variant", "key", "x", "y"])


def main():
    A = json.loads((OUT / "analysis.json").read_text())
    fig_curves(A)
    fig_damage(A)
    fig_s2s4()
    print("figures written")


if __name__ == "__main__":
    main()
