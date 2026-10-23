"""Norm puzzle: is the raw concept feature's large ||z|| the mu_B component?

    python -m experiments.paper_analysis.quiet_concept.norm_puzzle

For every raw (EMBER L0) fit and its judge-free concept feature b (features/<Slug>.json): percentile of ||z_b||^2 among
the 100 features, the same after projecting mu^_B = mu_B/||mu_B|| out of every z_i, the share of ||z_b||^2 along mu^_B vs
the median feature's, and the sign coherence |sum_t y_bt| / sum_t |y_bt| vs the median feature's.
Writes tables/norm_puzzle_runs.csv and norm_puzzle.json.
"""
import json

import numpy as np
import pandas as pd
import torch
from scipy import stats

from experiments.paper_analysis.quiet_concept.common import OUT, SEEDS, concepts, dump, load_fit, matrices, whitener
from experiments.varwidth.common import slug


def main():
    wh = whitener()
    mu = (wh["mu"] / wh["mu"].norm()).float()
    z = np.load(OUT / "frequency" / "wiki_counts.npz")
    counts, total = z["counts"], int(z["total"])
    rows = []
    for c in concepts("all"):
        M = matrices(c)
        ids = np.asarray(M["vstar_ids"])
        lf = np.log10((counts[ids] + 0.5) / total * 1e6)
        tok_mu = (mu.double() @ M["raw"]).numpy()                   # e_t . mu^ per token
        F = {(r["cond"], r["seed"]): r for r in json.loads((OUT / "features" / f"{slug(c)}.json").read_text())}
        for cond in ("raw", "cs", "ctr"):
            for s in SEEDS:
                f = load_fit(c, cond, s)
                if f is None or (cond, s) not in F:
                    continue
                b = F[(cond, s)]["best"]
                Z, Y = f["Z"].float(), f["Y"].float()
                z2 = Z.pow(2).sum(0).numpy()
                mproj = (mu @ Z).numpy()
                z2p = z2 - mproj ** 2
                share = mproj ** 2 / z2
                coh = (Y.sum(1).abs() / Y.abs().sum(1)).numpy()
                o = np.arange(len(z2)) != b
                Ym = (Y.abs() > 1e-12).numpy()
                lf_f = np.array([np.median(lf[Ym[i]]) for i in range(len(z2))])
                mu_f = np.array([np.median(tok_mu[Ym[i]]) for i in range(len(z2))])
                rows.append({"concept": c, "cond": cond, "seed": s, "found": F[(cond, s)]["hits"] >= 5,
                             "z2_pct": float((z2[o] < z2[b]).mean() * 100),
                             "z2_perp_pct": float((z2p[o] < z2p[b]).mean() * 100),
                             "mu_share_b": float(share[b]), "mu_share_med": float(np.median(share[o])),
                             "signcoh_b": float(coh[b]), "signcoh_med": float(np.median(coh[o])),
                             "lf_b": float(lf_f[b]), "lf_med": float(np.median(lf_f[o])),
                             "lf_pct": float((lf_f[o] < lf_f[b]).mean() * 100),
                             "tokmu_b": float(mu_f[b]), "tokmu_med": float(np.median(mu_f[o])),
                             "tokmu_pct": float((mu_f[o] < mu_f[b]).mean() * 100)})
    D = pd.DataFrame(rows)
    (OUT / "tables").mkdir(exist_ok=True)
    D.to_csv(OUT / "tables" / "norm_puzzle_runs.csv", index=False)
    out = {}
    for cond in ("raw", "ctr", "cs"):
        G = D[D.cond == cond].groupby("concept")[["z2_pct", "z2_perp_pct", "mu_share_b", "mu_share_med",
                                                  "signcoh_b", "signcoh_med", "lf_b", "lf_med", "lf_pct",
                                                  "tokmu_b", "tokmu_med", "tokmu_pct"]].median()
        out[cond] = {"n_concepts": int(len(G)), **{k: float(v) for k, v in G.median().items()},
                     "wilcoxon_z2_vs_z2perp_p": float(stats.wilcoxon(G.z2_pct, G.z2_perp_pct).pvalue),
                     "n_z2perp_lower": int((G.z2_perp_pct < G.z2_pct).sum()),
                     "wilcoxon_mushare_b_vs_med_p": float(stats.wilcoxon(G.mu_share_b, G.mu_share_med).pvalue),
                     "n_mushare_b_gt_med": int((G.mu_share_b > G.mu_share_med).sum()),
                     "wilcoxon_lf_b_vs_med_p": float(stats.wilcoxon(G.lf_b, G.lf_med).pvalue),
                     "n_lf_b_lt_med": int((G.lf_b < G.lf_med).sum()),
                     "wilcoxon_tokmu_b_vs_med_p": float(stats.wilcoxon(G.tokmu_b, G.tokmu_med).pvalue),
                     "n_tokmu_b_gt_med": int((G.tokmu_b > G.tokmu_med).sum())}
    dump(out, OUT / "norm_puzzle.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(3)
    main()
