"""Section 7 figure: efficacy against word integrity (copy) along the edit-size grid, with the unreadability floors.

For each method, one point per grid value r in {0.05, 0.1, 0.25, 0.5, 1, 2} (the grid points where the copy test was
run, amendment 11): mean over the held-out runs where the method edits (12 concepts x 5 seeds) of
    x = copy: share of edited words the model repeats correctly in a carrier sentence (copygrid.json),
    y = efficacy on the validation questions (grid.json).
Bands: the nonce floor and the deletion floor (efficacy of the UNEDITED model on the test questions with each edit
set's tokens replaced by their nonces or deleted; last_mile.json "floor"), as ranges over the five methods' sets.
Star: the free nonce-matching edit on CINDER's tokens (seed 42, test split; nonce_edit.json, post hoc).

Data (read only): results/varwidth/layers/collateral/<Concept>/s<seed>/sub_<variant>/{grid,copygrid}.json,
                  results/varwidth/layers/collateral/{last_mile,nonce_edit}.json

    .venv/bin/python paper/figures/src/fig_integrity.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COL = ROOT / "results/varwidth/layers/collateral"
CONCEPTS = ["Baseball", "Cannabis", "COVID_19_pandemic", "Gambling", "Golf", "Gun", "Halloween", "Heroin", "Nazism",
            "Pornography", "Republic_of_Ireland", "Uranium"]
SEEDS = range(42, 47)
GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
GREY, ORANGE, AMBER, GREEN, INK, MUTED, HAIR = "#7F7F7F", "#D55E00", "#E69F00", "#009E73", "#222222", "#666666", "#BBBBBB"
# (key, label, colour, line style, filled marker): colour = edit, dashed/open = EMBER's row sparsity instead of GS
METHODS = [("ember", "EMBER", GREY, "-", True), ("cinder", "CINDER", ORANGE, "-", True),
           ("cinder_rs", "CINDER-RS", ORANGE, "--", False), ("cinder_z", "CINDER-z", AMBER, "-", True),
           ("cinder_rs_z", "CINDER-RS-z", AMBER, "--", False)]


def curve(key):
    eff, cp, n = {r: [] for r in GRID}, {r: [] for r in GRID}, 0
    for c in CONCEPTS:
        for s in SEEDS:
            d = COL / c / f"s{s}" / f"sub_{key}"
            if not (d / "grid.json").exists() or not (d / "copygrid.json").exists():
                continue
            n += 1
            for x in json.loads((d / "grid.json").read_text()):
                if x["param"] in eff:
                    eff[x["param"]].append(x["efficacy"])
            for k, x in json.loads((d / "copygrid.json").read_text()).items():
                cp[float(k)].append(x["copy"])
    return np.array([np.mean(cp[r]) for r in GRID]), np.array([np.mean(eff[r]) for r in GRID]), n


def main():
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    floor = json.loads((COL / "last_mile.json").read_text())["floor"]
    nonce = [floor[f"sub:{k}"]["nonce"] for k, *_ in METHODS]
    dele = [floor[f"sub:{k}"]["del"] for k, *_ in METHODS]
    nm = json.loads((COL / "nonce_edit.json").read_text())["variants"]["nm:free"]

    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    for lo, hi, name in ((min(nonce), max(nonce), "nonce floor"), (min(dele), max(dele), "deletion floor")):
        ax.axhspan(lo, hi, color="#E9E9E9", lw=0, zorder=0)
        ax.text(1.015, (lo + hi) / 2, name, fontsize=6.5, color=MUTED, va="center", ha="right", zorder=1)
    ax.axvline(0.9, color=HAIR, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(0.893, 0.445, "copy $\\geq$ 0.9", fontsize=6.5, color=MUTED, ha="right", va="top")
    data = {}
    for key, label, color, ls, filled in METHODS:
        x, y, n = curve(key)
        data[key] = {"r": GRID, "copy": x.tolist(), "efficacy_val": y.tolist(), "runs": n}
        ax.plot(x, y, color=color, ls=ls, lw=1.2, zorder=3, label=label, marker="o", ms=3.6,
                mfc=color if filled else "white", mec=color, mew=0.9)
        if key == "ember":
            for r, xx, yy in zip(GRID, x, y):
                if r >= 0.25:
                    off, ha = {0.25: ((-4, -1.5), "right"), 0.5: ((-4, -6), "right")}.get(r, ((0, -9), "center"))
                    ax.annotate(f"$r$ = {r:g}", (xx, yy), textcoords="offset points", xytext=off, ha=ha,
                                fontsize=6, color=MUTED)
    ax.plot(nm["copy"], nm["eff"], marker="*", ms=8, color=GREEN, mec="white", mew=0.4, ls="none", zorder=5,
            label="nonce-matching edit")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("copy: share of edited words still copied", fontsize=7.5, color=INK)
    ax.set_ylabel("efficacy", fontsize=7.5, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(HAIR)
        ax.spines[s].set_linewidth(0.6)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.01), handlelength=2.2,
              borderaxespad=0.1, labelspacing=0.25)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.16)
    out = ROOT / "paper/figures/integrity.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    data["floors"] = {"nonce": dict(zip([m[0] for m in METHODS], nonce)),
                      "deletion": dict(zip([m[0] for m in METHODS], dele))}
    data["nonce_matching_free"] = {"copy": nm["copy"], "efficacy_test": nm["eff"], "runs": nm["n_edit"]}
    (ROOT / "paper/figures/src/integrity_data.json").write_text(json.dumps(data, indent=1))
    print("->", out)
    for k, v in data.items():
        print(k, v)


if __name__ == "__main__":
    main()
