"""Figure 3 (Section 5.3): paired H3 differences under four selection rules (results.pdf, panel a); the 2x2 at
fixed edit size goes to the appendix (twobytwo.pdf, panel b).

(a) Per-concept paired test H3 difference, each CINDER variant minus EMBER, on the runs where both edit
    (11 held-out concepts, mean over seeds), at the point chosen by each selection rule:
    collateral rule (primary), paper rule (no collateral term), fixed r = 0.5, integrity-gated (copy >= 0.9).
    Small dots: concepts; large marker and bar: mean and 95% cluster-bootstrap CI; * = Holm p < 0.05.
(b) Test efficacy of the four direction x token-set cells at fixed r = 0.5 (primary) and r = 1 (secondary).

Data (read only):
    results/varwidth/layers/collateral/confirm.json     rules collateral|H3v_rn, paper|H3v_rn, fixed05|H3v_rn
    results/varwidth/layers/collateral/last_mile.json   e3 (integrity-gated rule)
    results/varwidth/layers/collateral/twobytwo.json    result_fixed05.json, result_fixed1.json

    .venv/bin/python paper/figures/src/fig_results.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COL = ROOT / "results/varwidth/layers/collateral"
GREY, ORANGE, AMBER, INK, MUTED, HAIR = "#7F7F7F", "#D55E00", "#E69F00", "#222222", "#666666", "#BBBBBB"
# colour = edit (mapped back: orange, original space: amber); fill = widths (GS: filled, EMBER's RS: open)
VARIANTS = [("sub:cinder", "CINDER", ORANGE, True), ("sub:cinder_rs", "CINDER-RS", ORANGE, False),
            ("sub:cinder_z", "CINDER-z", AMBER, True), ("sub:cinder_rs_z", "CINDER-RS-z", AMBER, False)]
RULES = [("collateral", "Collateral\n(primary)"), ("paper", "EMBER's $H$\n(no collateral)"),
         ("fixed05", "Fixed\n$r = 0.5$"), ("e3", "Integrity\n(copy $\\geq$ 0.9)")]


def contrasts():
    c = json.loads((COL / "confirm.json").read_text())["rules"]
    e3 = json.loads((COL / "last_mile.json").read_text())["e3"]["contrasts"]
    return {"collateral": c["collateral|H3v_rn"]["contrasts"], "paper": c["paper|H3v_rn"]["contrasts"],
            "fixed05": c["fixed05|H3v_rn"]["contrasts"], "e3": e3}


def holm(ps):
    order = np.argsort(ps)
    out, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        out[j] = run
    return out


def panel_a(ax):
    C = contrasts()
    rng = np.random.default_rng(0)
    offs = np.linspace(-0.3, 0.3, len(VARIANTS))
    for x, (rule, _) in enumerate(RULES):
        ps = [C[rule][v]["p"] for v, *_ in VARIANTS]
        ph = [C[rule][v].get("p_holm") for v, *_ in VARIANTS]
        ph = holm(np.array(ps)) if any(p is None for p in ph) else np.array(ph)
        for k, (v, _, color, filled) in enumerate(VARIANTS):
            d = C[rule][v]
            ys = np.array([a[0] if isinstance(a, list) else a for a in d["per_concept"].values()])
            xs = x + offs[k] + rng.uniform(-0.045, 0.045, len(ys))
            ax.scatter(xs, ys, s=3.5, color=color, alpha=0.45, lw=0, zorder=2)
            lo, hi = d["ci"]
            ax.plot([x + offs[k]] * 2, [lo, hi], color=color, lw=1.0, zorder=3, solid_capstyle="butt")
            ax.plot(x + offs[k], d["mean"], marker="o", ms=4.2, mec=color, mew=1.0,
                    mfc=color if filled else "white", ls="none", zorder=4)
            if ph[k] < 0.05:
                ax.text(x + offs[k], hi + 0.02, "*", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(0, color=HAIR, lw=0.6, zorder=1)
    for x in range(1, len(RULES)):
        ax.axvline(x - 0.5, color="#E6E6E6", lw=0.5, zorder=0)
    ax.set_xlim(-0.5, len(RULES) - 0.5)
    ax.set_ylim(-0.6, 0.62)
    ax.set_xticks(range(len(RULES)))
    ax.set_xticklabels([r[1] for r in RULES], fontsize=7, color=INK)
    ax.set_ylabel(r"$\Delta H_3$ vs. EMBER", fontsize=7.5, color=INK)
    ax.tick_params(axis="y", labelsize=7, colors=MUTED, length=2)
    ax.tick_params(axis="x", length=0, pad=2)
    handles = [plt.Line2D([], [], marker="o", ms=4.2, mec=c, mew=1.0, mfc=c if f else "white", ls="none", label=n)
               for _, n, c, f in VARIANTS]
    ax.legend(handles=handles, ncol=4, fontsize=6.5, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 0.995), handletextpad=0.2, columnspacing=0.9, borderaxespad=0.0)
    ax.text(-0.135, 1.0, "(a)", transform=ax.transAxes, fontsize=8, color=INK, fontweight="bold", va="bottom")


def panel_b(axes):
    T = json.loads((COL / "twobytwo.json").read_text())
    for ax, (key, title, eff) in zip(axes, (("result_fixed05.json", "$r = 0.5$", "result_fixed05.json"),
                                            ("result_fixed1.json", "$r = 1$", "result_fixed1.json"))):
        m = T[key]["means"]
        e = T[key]["effects"]["Δ_dir|efficacy"]
        rows = [("EMBER's tokens", GREY, [m["sub:ee"]["efficacy"], m["sub:1b"]["efficacy"]]),
                ("CINDER's tokens", ORANGE, [m["sub:cross"]["efficacy"], m["sub:1a"]["efficacy"]])]
        for name, color, ys in rows:
            ax.plot([0, 1], ys, color=color, lw=1.3, marker="o", ms=4, mec="white", mew=0.6, label=name, zorder=3)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0.15, 0.33)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["EMBER\ndirection", "CINDER\ndirection"], fontsize=7, color=INK)
        ax.set_yticks([0.15, 0.2, 0.25, 0.3])
        ax.tick_params(axis="y", labelsize=7, colors=MUTED, length=2)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=2)
        ax.text(0.5, 0.04, f"direction {e['mean']:+.2f} ({e['pos']}/{e['neg']} concepts)".replace("-", "\u2212"),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=MUTED)
    axes[0].set_ylabel("efficacy (test)", fontsize=7.5, color=INK)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.1)
    axes[0].text(-0.33, 1.0, "(b)", transform=axes[0].transAxes, fontsize=8, color=INK, fontweight="bold",
                 va="bottom")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(HAIR)
        ax.spines[s].set_linewidth(0.6)


def main():
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    # Main text (Figure 3): panel (a) only, to fit the page budget; appendix: panel (b).
    fig = plt.figure(figsize=(3.3, 1.95))
    a = fig.add_axes([0.16, 0.2, 0.825, 0.68])
    panel_a(a)
    a.texts[-1].set_visible(False)  # no "(a)" tag on a single-panel figure
    style(a)
    out = ROOT / "paper/figures/results.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print("->", out)

    fig = plt.figure(figsize=(3.3, 1.45))
    gs = fig.add_gridspec(1, 2, wspace=0.12, left=0.16, right=0.985, top=0.86, bottom=0.24)
    b = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    panel_b(b)
    b[0].texts[-1].set_visible(False)
    for ax in b:
        style(ax)
    out = ROOT / "paper/figures/twobytwo.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print("->", out)


if __name__ == "__main__":
    main()
