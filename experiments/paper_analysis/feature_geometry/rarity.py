#!/usr/bin/env python
"""WP2 mechanism check: is the raw concept feature's close neighbour a shared "rare token" direction?

    .venv/bin/python -m experiments.paper_analysis.feature_geometry.rarity

Raw space only. Background: top principal direction pc1 of the Wikipedia type cloud (bg_wiki_stats), its variance
share and its cosine with the background mean mu_B. Per concept: rarity direction r = normalized sum_t (logf_mean - logf_t)(e_t - e_mean) over the V*
tokens (the least-squares direction along which raw embeddings vary with -log frequency; logf from the Wikipedia
sample). Per run and the judge-free concept feature c (common.concept_feature):
    cos_r       cos(z_c^raw, r) and the run median over all features
    nn angles   nearest-neighbour angle of c in raw space as is, with r, mu_B, both, or pc1 projected out
    sp_pc1proj_rarity  Spearman(projection on pc1 oriented toward rare tokens, -log frequency) over V*
    nn_logf_z   within-run z-score of the neighbour's |y|-weighted log frequency (is the neighbour a rare-token feature?)
    nn_rel_pct  percentile of MiniLM relatedness(c, nn) among relatedness(c, all others) (50 = chance), for the
                neighbour in the fit space and in the raw space
Token level: beta_t = e_t . mu_B / ||mu_B||^2 (the token's coordinate along the background mean) by log-frequency
decile over the union of V* tokens.
Outputs: rarity_runs.csv, rarity_stats.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.feature_geometry.common import (CONFIGS, OUT, SEEDS, boot_ci, concept_feature,
                                                                load_features, paired, text_emb_path)
from experiments.varwidth.common import run_dir


def unit(M):
    return M / M.norm(dim=0, keepdim=True).clamp_min(1e-30)


def nn_angle(Zu: torch.Tensor, c: int) -> tuple[float, int]:
    C = (Zu.T @ Zu[:, c]).abs()
    C[c] = -1
    j = int(C.argmax())
    return float(np.degrees(np.arccos(min(float(C[j]), 1.0)))), j


def proj_out(Z: torch.Tensor, vs: list) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.stack(vs, 1))
    return Z - Q @ (Q.T @ Z)


def main():
    torch.set_num_threads(4)
    from experiments.varwidth.prep import prepare
    F = load_features()
    E = np.load(text_emb_path())
    kw = {k: i for i, k in enumerate(E["keys_w"])}
    wc = np.load(OUT / "wiki_token_counts.npz")
    from scipy.stats import spearmanr

    from experiments.varwidth.prep import WH_DIR
    bg = torch.load(WH_DIR / "bg_wiki_stats.pt", map_location="cpu")
    lam, Qb = bg["lam"].double(), bg["Q"].double()
    order = lam.argsort(descending=True)
    pc1 = Qb[:, order[0]]
    st_bg = {"lam_top5": lam[order[:5]].tolist(), "lam_median": float(lam.median()),
             "pc1_var_share": float(lam[order[0]] / lam.sum()),
             "cos_mu_pc1": float(abs(pc1 @ bg["mu"].double()) / bg["mu"].double().norm())}
    rows, crow, tok_beta = [], [], {}
    mu_full = bg["mu"].double()
    for concept in F.concept.unique():
        pp = prepare(concept, "plain", verbose=False)
        pw = prepare(concept, "whitened", verbose=False)
        A = pp["A"].double()
        lf = torch.as_tensor(np.log10((wc["counts"][np.asarray(pp["vstar_ids"])] + 1) / float(wc["n_tokens"]) * 1e6))
        Ac = A - A.mean(1, keepdim=True)
        r = Ac @ (lf.mean() - lf)
        r = r / r.norm()
        mu = pw["mu_B"].double()
        mu = mu / mu.norm()
        Tinv = pw["T_inv"].double()
        beta = (A.T @ mu_full / mu_full.norm() ** 2).numpy()          # token's mean coordinate, units of mu_B
        for t, b, f in zip(pp["vstar_ids"], beta, lf.numpy()):
            tok_beta[int(t)] = (float(b), float(f))
        crow.append({"concept": concept, "cos_r_mu": float(r @ mu), "cos_r_pc1": float(abs(r @ pc1)),
                     "sp_pc1proj_rarity": float(spearmanr(((A - pw["mu_B"].double()[:, None]).T @ pc1).numpy(),
                                                        lf.numpy()).statistic * np.sign(float(r @ pc1)) * -1)})
        for cfg, tag in CONFIGS.items():
            for s in SEEDS:
                g = F[(F.concept == concept) & (F.cfg == cfg) & (F.seed == s)].reset_index(drop=True)
                c = concept_feature(g, "lex")
                if c is None:
                    continue
                fit = torch.load(run_dir(concept, tag, s) / "fit.pt", map_location="cpu", weights_only=False)
                Z = fit["Z"].double() * torch.as_tensor(g.sign.values, dtype=torch.float64)
                Zr = Z if cfg == "L0" else (Z.T @ Tinv).T
                Zu = unit(Zr)
                cr = (r @ Zu).numpy()
                out = {"concept": concept, "cfg": cfg, "seed": s, "feature": c, "cos_r": float(cr[c]),
                       "cos_r_median": float(np.median(cr)), "cos_r_pct": float(100 * (cr < cr[c]).mean()),
                       "cos_mu": float((mu @ Zu)[c])}
                for name, vs in (("raw", []), ("raw_minus_r", [r]), ("raw_minus_mu", [mu]), ("raw_minus_both", [r, mu]),
                                 ("raw_minus_pc1", [pc1])):
                    Zp = unit(proj_out(Zr, vs)) if vs else Zu
                    a, j = nn_angle(Zp, c)
                    out[f"nn_{name}"] = a
                    if name == "raw":
                        out["nn_raw_id"] = j
                # fit-space neighbour (for CS this differs from the raw one)
                Zf = unit(Z)
                af, jf = nn_angle(Zf, c)
                out["nn_fit"], out["nn_fit_id"] = af, jf
                lz = (g.logf_w - g.logf_w.mean()) / g.logf_w.std()
                out["nn_raw_logf_z"] = float(lz[out["nn_raw_id"]])
                out["nn_fit_logf_z"] = float(lz[jf])
                V = E["w20"][[kw[f"{concept}|{cfg}|{s}|{f}"] for f in range(len(g))]]
                rel = V @ V[c]
                others = np.delete(np.arange(len(g)), c)
                for nm, j in (("fit", jf), ("raw", out["nn_raw_id"])):
                    out[f"nn_{nm}_rel_pct"] = float(100 * (rel[others] < rel[j]).mean())
                rows.append(out)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rarity_runs.csv", index=False)
    C = R.groupby(["concept", "cfg"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    CR = pd.DataFrame(crow)
    TB = np.array(list(tok_beta.values()))
    dec = pd.qcut(TB[:, 1], 10, labels=False, duplicates="drop")
    st_tok = {"n_tokens": len(TB), "spearman_beta_logf": float(spearmanr(TB[:, 0], TB[:, 1]).statistic),
              "beta_by_logf_decile_rare_to_frequent": pd.Series(TB[:, 0]).groupby(dec).mean().round(3).tolist(),
              "mu_B_norm": float(mu_full.norm())}
    st = {"n_concepts": {cfg: int((C.cfg == cfg).sum()) for cfg in CONFIGS}, "by_cfg": {}, "paired": {},
          "background": st_bg, "token_mu_coordinate": st_tok, "rarity_direction": {m: {"mean": float(CR[m].mean()), "min": float(CR[m].min()),
                                                        "max": float(CR[m].max())} for m in CR.columns if m != "concept"}}
    cols = ["cos_r", "cos_r_median", "cos_r_pct", "cos_mu", "nn_raw", "nn_raw_minus_r", "nn_raw_minus_mu",
            "nn_raw_minus_both", "nn_raw_minus_pc1", "nn_fit", "nn_raw_logf_z", "nn_fit_logf_z", "nn_fit_rel_pct", "nn_raw_rel_pct"]
    for cfg in CONFIGS:
        sub = C[C.cfg == cfg]
        st["by_cfg"][cfg] = {m: {"mean": float(sub[m].mean()), "ci95": boot_ci(sub[m])} for m in cols}
    for m in cols:
        p = C.pivot(index="concept", columns="cfg", values=m)
        st["paired"][m] = {f"L0->{b}": paired(p["L0"], p[b]) for b in ("L1", "L12")}
    L0 = C[C.cfg == "L0"]
    st["L0_within"] = {"nn_raw_minus_r_vs_raw": paired(L0.nn_raw, L0.nn_raw_minus_r),
                       "nn_raw_minus_mu_vs_raw": paired(L0.nn_raw, L0.nn_raw_minus_mu),
                       "nn_raw_minus_both_vs_raw": paired(L0.nn_raw, L0.nn_raw_minus_both),
                       "nn_raw_minus_pc1_vs_raw": paired(L0.nn_raw, L0.nn_raw_minus_pc1),
                       "cos_r_vs_median": paired(L0.cos_r_median, L0.cos_r)}
    (OUT / "rarity_stats.json").write_text(json.dumps(st, indent=1, default=float))
    for cfg, d in st["by_cfg"].items():
        print(cfg, {k: round(v["mean"], 3) for k, v in d.items()})
    print(json.dumps(st["L0_within"], indent=1, default=float))
    print(json.dumps({k: st[k] for k in ("background", "rarity_direction", "token_mu_coordinate")}, indent=1,
                     default=float))
    print(json.dumps({m: st["paired"][m] for m in ("nn_fit_rel_pct", "nn_raw_rel_pct", "nn_raw_logf_z")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
