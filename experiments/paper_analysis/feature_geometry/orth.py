#!/usr/bin/env python
"""WP2 step 1: how orthogonal are the k feature directions, raw (EMBER) vs CS fits?

    .venv/bin/python -m experiments.paper_analysis.feature_geometry.orth

Per run and space (fit = where the fit ran; raw = mapped back to the embedding space, z = z~ T^-1; cs = the CS metric;
w = background whitening W only; rawc = raw with the background-mean direction projected out):
mean / median / p95 of |cos| over the k(k-1)/2 pairs, mutual coherence (max |cos|), log10 condition number of the
Gram matrix of the unit directions, share of pairs with angle < 60 deg (|cos| > 0.5).
Null references: (a) random Gaussian directions (E|cos| = sqrt(2 / (pi d))); (b) "random-support features": 100 random
token subsets of width 48 of the same matrix, z = mean of their token vectors (raw and CS space), which isolates the
anisotropy of the space from anything the fit learns.
Concept feature (judge-free, common.concept_feature): its nearest-neighbour angle (arccos max |cos|) and the
neighbour's words; robustness with the TJ (F_C) and hand-label (P) identifications.
Outputs: orth_runs.csv, orth_concepts.csv, orth_stats.json, concept_nn.csv, fig_cos_ecdf.{pdf,png} (+ _data.csv).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.feature_geometry.common import (COLORS, CONFIGS, NAMES, OUT, SEEDS, boot_ci,
                                                                concept_feature, full, load_features, load_pairs,
                                                                paired, savefig, setup_mpl)

SPACES = ("fit", "raw", "cs", "w", "rawc")


def run_metrics(P: dict) -> dict:
    out = {}
    for sp in SPACES:
        c = np.abs(P[f"cos_{sp}"].astype(np.float64))
        G = full(P[f"cos_{sp}"])
        ev = np.linalg.eigvalsh(G)
        out |= {f"{sp}_mean": c.mean(), f"{sp}_median": np.median(c), f"{sp}_p95": np.percentile(c, 95),
                f"{sp}_coh": c.max(), f"{sp}_logcond": np.log10(ev.max() / max(ev.min(), 1e-12)),
                f"{sp}_lt60": (c > 0.5).mean()}
    out["jac_mean"] = P["jac"].mean()
    out["jac_pos"] = (P["jac"] > 0).mean()
    return out


def random_support_null(concept: str, width: int = 48, k: int = 100, seed: int = 0) -> dict:
    from experiments.varwidth.prep import prepare
    rng = np.random.default_rng(seed)
    out = {}
    for name, matrix in (("raw", "plain"), ("cs", "whitened")):
        A = prepare(concept, matrix, verbose=False)["A"].double()
        n = A.shape[1]
        idx = np.stack([rng.choice(n, width, replace=False) for _ in range(k)])
        Zr = A[:, torch.as_tensor(idx)].mean(-1)          # (d, k)
        Zr = Zr / Zr.norm(dim=0, keepdim=True)
        C = (Zr.T @ Zr).numpy()
        c = np.abs(C[np.triu_indices(k, 1)])
        out[f"null_{name}_mean"] = c.mean()
        out[f"null_{name}_p95"] = np.percentile(c, 95)
        # single token pairs (anisotropy of the token cloud itself)
        t = A[:, torch.as_tensor(rng.choice(n, 400, replace=False))]
        t = t / t.norm(dim=0, keepdim=True)
        Ct = (t.T @ t).numpy()
        out[f"tok_{name}_mean"] = np.abs(Ct[np.triu_indices(400, 1)]).mean()
        out[f"tok_{name}_signed"] = Ct[np.triu_indices(400, 1)].mean()
    return out


def main():
    torch.set_num_threads(4)
    F = load_features()
    words = json.loads((OUT / "words.json").read_text())
    rows, nn_rows, ecdf = [], [], {}
    for (c, cfg, s), g in F.groupby(["concept", "cfg", "seed"], sort=False):
        P = load_pairs(c, cfg, s)
        rows.append({"concept": c, "cfg": cfg, "seed": s} | run_metrics(P))
        if s == SEEDS[0]:
            for sp in ("fit", "raw", "cs"):
                ecdf.setdefault((cfg, sp), []).append(np.abs(P[f"cos_{sp}"]))
        g = g.set_index("feature")
        for how in ("lex", "tj", "p"):
            f = concept_feature(g.reset_index(), how)
            if f is None:
                continue
            for sp in ("fit", "raw"):
                C = np.abs(full(P[f"cos_{sp}"], diag=0.0))
                order = np.argsort(-C[f])
                j = int(order[0])
                nn_rows.append({"concept": c, "cfg": cfg, "seed": s, "how": how, "space": sp, "feature": f,
                                "lo": int(g.loc[f, "lo"]), "nn": j, "nn_angle": float(np.degrees(np.arccos(C[f, j]))),
                                "nn2_angle": float(np.degrees(np.arccos(C[f, order[1]]))),
                                "nn_lo": int(g.loc[j, "lo"]), "nn_jac": float(full(P["jac"], diag=1.0)[f, j]),
                                "mean_abs_cos": float(C[f].sum() / (len(C) - 1)),
                                "words": ", ".join(words[f"{c}|{cfg}|{s}|{f}"][:8]),
                                "nn_words": ", ".join(words[f"{c}|{cfg}|{s}|{j}"][:8])})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "orth_runs.csv", index=False)
    NN = pd.DataFrame(nn_rows)
    NN.to_csv(OUT / "concept_nn.csv", index=False)

    nulls = pd.DataFrame([{"concept": c} | random_support_null(c) for c in R.concept.unique()])
    nulls.to_csv(OUT / "orth_nulls.csv", index=False)
    Cc = R.groupby(["concept", "cfg"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    Cc.to_csv(OUT / "orth_concepts.csv", index=False)

    st = {"n_runs": {cfg: int((R.cfg == cfg).sum()) for cfg in CONFIGS}, "n_concepts": int(R.concept.nunique()),
          "gaussian_null_mean_abs_cos": float(np.sqrt(2 / (np.pi * 2304))),
          "nulls_mean_over_concepts": nulls.mean(numeric_only=True).to_dict(), "by_cfg": {}, "paired": {}}
    for cfg in CONFIGS:
        sub = Cc[Cc.cfg == cfg]
        st["by_cfg"][cfg] = {m: {"mean": float(sub[m].mean()), "ci95": boot_ci(sub[m])}
                             for m in sub.columns if m not in ("concept", "cfg")}
    piv = lambda m: Cc.pivot(index="concept", columns="cfg", values=m)  # noqa: E731
    for m in ("fit_mean", "raw_mean", "cs_mean", "w_mean", "rawc_mean", "fit_coh", "raw_coh", "fit_logcond",
              "raw_logcond", "fit_p95", "raw_p95", "fit_lt60", "raw_lt60"):
        p = piv(m)
        st["paired"][m] = {f"L0->{b}": paired(p["L0"], p[b]) for b in ("L1", "L12")} | \
                          {"L1->L12": paired(p["L1"], p["L12"])}
    # concept-feature nearest neighbour angle (lex identification), per concept mean over seeds
    for how in ("lex", "tj", "p"):
        n = NN[NN.how == how]
        for sp in ("fit", "raw"):
            q = n[n.space == sp].groupby(["concept", "cfg"]).nn_angle.mean().unstack()
            st.setdefault("nn_angle", {})[f"{how}_{sp}"] = {
                "mean_by_cfg": {cfg: float(q[cfg].mean()) for cfg in q.columns},
                "n_concepts_by_cfg": {cfg: int(q[cfg].notna().sum()) for cfg in q.columns},
                **{f"L0->{b}": paired(q["L0"], q[b]) for b in ("L1", "L12") if b in q}}
            q2 = n[n.space == sp].groupby(["concept", "cfg"]).mean_abs_cos.mean().unstack()
            st.setdefault("concept_mean_abs_cos", {})[f"{how}_{sp}"] = {
                "mean_by_cfg": {cfg: float(q2[cfg].mean()) for cfg in q2.columns}}
    hp = NN[(NN.concept == "Harry Potter") & (NN.how == "lex")]
    st["harry_potter_nn"] = hp[["cfg", "seed", "space", "nn_angle", "nn2_angle", "words", "nn_words"]].to_dict("records")
    (OUT / "orth_stats.json").write_text(json.dumps(st, indent=1, default=float))

    # ---------------- figure: ECDF of pairwise |cos| (seed 42, all concepts pooled)
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), sharey=True)
    data = []
    styles = {"L0": "-", "L1": "--", "L12": "-"}
    for ax, sp, lab in ((axes[0], "fit", "|cos|, space the fit ran in"), (axes[1], "raw", "|cos|, raw embedding metric"),
                        (axes[2], "cs", "|cos|, CS metric")):
        for cfg in CONFIGS:
            x = np.sort(np.concatenate(ecdf[(cfg, sp)]))
            y = np.arange(1, len(x) + 1) / len(x)
            sel = np.unique(np.linspace(0, len(x) - 1, 400).astype(int))
            ax.plot(x[sel], y[sel], color=COLORS[cfg], ls=styles[cfg], lw=1.5, label=NAMES[cfg])
            data += [{"space": sp, "cfg": cfg, "abs_cos": float(x[i]), "ecdf": float(y[i])} for i in sel]
        g = np.sqrt(2 / (np.pi * 2304))
        ax.axvline(g, color="#009E73", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_xlabel(lab)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("ECDF over feature pairs")
    axes[0].text(np.sqrt(2 / (np.pi * 2304)) * 1.1, 0.04, "random\ndirections", fontsize=6, color="#009E73")
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    savefig(fig, "fig_cos_ecdf")
    pd.DataFrame(data).to_csv(OUT / "fig_cos_ecdf_data.csv", index=False)
    print(json.dumps({k: st[k] for k in ("by_cfg",)}, indent=1, default=float)[:3000])
    print(json.dumps(st["paired"], indent=1, default=float)[:4000])
    print(json.dumps(st["nn_angle"], indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
