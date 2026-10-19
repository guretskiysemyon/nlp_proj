#!/usr/bin/env python
"""X6: erasure and side effects against EDIT SIZE, not delta (CPU; reruns whenever new results exist).

At equal delta CINDER's edit is about half of EMBER's (cinder.edit_size). Here every grid point is placed at its
relative edit size x = delta * median_t ||c_t|| / ||e_t|| (per concept and method), so the two methods are compared
at the same amount of change to the edited tokens.
  1. English grid (train split, fixed MC scoring): QA_train, SimdomQA_train, MMLU against x; QA_train interpolated
     (linear in log x) at x = 0.5, 1, 2, 5, 10, 20, 50.
  2. If g1_leak_curves exist: retention per language (QA_test) against x, both methods and their extrapolated
     variants (EMBER+X, CINDER+X: same English edit + native foreign tokens, x5b_build_extrapolated.py).
  3. The picked delta of each method (EMBER: Part 1 pass2_rescored; CINDER: judge_rerun/Cinder/pick, when done).

Output: Part 2 Experiments/x6_strength_curves/{points.csv, matched.csv, english_<C>.png, leak_<C>.png, report.md}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

XS = [0.5, 1, 2, 5, 10, 20, 50]
C = p2x.C


def grid(concept, method):
    p = (C.EMBER_DIR / "rescore_mc" / C.RUN_REL / C.safe(concept) / "grid_rescored.csv" if method == "EMBER"
         else C.GRID_OUT / C.safe(concept) / "grid_rescored.csv")
    return pd.read_csv(p).sort_values("delta_embed") if p.exists() else None


def picks():
    out = {}
    e = pd.read_csv(C.EMBER_DIR / "pass2_rescored" / C.RUN_REL / "best_deltas_summary.csv")
    out.update({("EMBER", r.concept): float(r.best_delta_embed) for r in e.itertuples()})
    for p in C.PICK_OUT.glob(f"*/best_delta__{C.JUDGE_ID}.json"):
        b = json.loads(p.read_text())
        out[("CINDER", b["concept"])] = float(b["best_delta_embed"])
    return out


def style(ax, title, ylabel):
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("relative edit size  δ·‖c‖/‖e‖ (log)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    out = p2x.outdir("x6_strength_curves")
    es = pd.read_csv(C.OUT / "edit_size.csv").dropna(subset=["token_id"])
    smed = es.groupby(["concept", "method"]).s.median().to_dict()
    pk = picks()
    pts, matched = [], []
    for c in p2x.CONCEPTS:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        any_line = False
        for m in p2x.METHODS:
            g = grid(c, m)
            if g is None or (c, m) not in smed:
                continue
            x = g.delta_embed.values * smed[(c, m)]
            for r, xx in zip(g.itertuples(), x):
                pts.append({"concept": c, "method": m, "delta": r.delta_embed, "rel_edit": xx, "qa_acc": r.qa_acc,
                            "simdom_acc": r.simdom_acc, "mmlu_acc": r.mmlu_acc,
                            "picked": pk.get((m, c)) == r.delta_embed})
            row = {"concept": c, "method": m, "picked_delta": pk.get((m, c))}
            for xq in XS:
                row[f"qa@{xq}"] = (round(float(np.interp(np.log(xq), np.log(x), g.qa_acc.values)), 3)
                                   if x.min() <= xq <= x.max() else None)
            matched.append(row)
            for ax, col, lab in zip(axes, ("qa_acc", "simdom_acc", "mmlu_acc"),
                                    ("QA_train (concept)", "SimdomQA_train", "MMLU (50)")):
                ax.plot(x, g[col], marker="o", ms=4, lw=2, color=p2x.COLORS[m], label=m)
                if (m, c) in pk and pk[(m, c)] in set(g.delta_embed):
                    i = list(g.delta_embed).index(pk[(m, c)])
                    ax.plot(x[i], g[col].iloc[i], marker="*", ms=13, color=p2x.COLORS[m], mec="white", mew=1)
            any_line = True
        if any_line:
            for ax, lab in zip(axes, ("QA_train accuracy (lower = more erased)", "SimdomQA_train accuracy",
                                      "MMLU accuracy")):
                style(ax, f"{c}", lab)
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"{c}: English, train split (★ = picked δ)", fontsize=11)
            fig.tight_layout()
            fig.savefig(out / f"english_{C.safe(c)}.png", dpi=130)
        plt.close(fig)

    # leak curves (g1)
    leak_rows = []
    for c in p2x.CONCEPTS:
        curves = {m: p2x.OUT / "g1_leak_curves" / m / C.safe(c) / "curve.csv" for m in p2x.ALL_METHODS}
        have = {m: pd.read_csv(p) for m, p in curves.items() if p.exists()}
        if not have:
            continue
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
        for m, d in have.items():
            base = m.split("+")[0]                    # +X keeps the original English edit: same edit size
            if (c, base) not in smed:
                continue
            d["rel_edit"] = d.delta * smed[(c, base)]
            leak_rows.append(d.assign(method=m))
            for ax, L in zip(axes, p2x.LANGS):
                dl = d[d.language == L].sort_values("rel_edit")
                ax.plot(dl.rel_edit, dl.retention, marker="o", ms=4, lw=2, color=p2x.COLORS[m], label=m)
        for ax, L in zip(axes, p2x.LANGS):
            style(ax, L, "retention (QA_test)" if L == "en" else "")
            ax.set_ylim(-0.05, 1.1)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{c}: retention per language vs edit size (test split, descriptive)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"leak_{C.safe(c)}.png", dpi=130)
        plt.close(fig)

    P, M = pd.DataFrame(pts), pd.DataFrame(matched)
    P.to_csv(out / "points.csv", index=False)
    M.to_csv(out / "matched.csv", index=False)
    if leak_rows:
        pd.concat(leak_rows).to_csv(out / "leak_points.csv", index=False)
    L_ = ["# X6: erasure vs edit size", "",
          "Relative edit size x = δ × median‖c_t‖/‖e_t‖ (per concept and method; `judge_rerun/Cinder/edit_size.csv`). "
          "QA_train accuracy interpolated at equal x (lower = more erased; pristine ≈ the δ→0 end). Figures: "
          "`english_<concept>.png`, `leak_<concept>.png` (when the G1 jobs are done).", "", p2x.md_table(M), ""]
    (out / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
