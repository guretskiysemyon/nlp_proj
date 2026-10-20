#!/usr/bin/env python
"""X5: extrapolating the English features to native foreign tokens (Ori's "ML-EMBER with extrapolation" step;
experiments/ml/fit.py extrapolate() in his ml branch), for both methods. CPU, no model.

Per (concept, method, language L in de/es/fr/ru): token set = all tokens of the English sentences (data/) and of
L's native concept + neutral sentences (data_v2). The fit's feature directions Z are kept FIXED (English-only fit);
coefficients for every token are a dense ridge solve, then the method's own sparsity rule with the new set's budget
(EMBER: top ceil(1% n) per feature; CINDER: GS-norm pool with floor 0.2% / cap 3%, in its whitened space). Edited
tokens = concept-only tokens (English or native L concept sentences, never in either neutral set) with a nonzero
coefficient on a selected feature, as embed_edit's rule. Reported: q_cov (share of the 50 QA_test questions in en and
L that contain an edited token) before vs after, and the added tokens (largest |y| first) to judge their quality.
No erasure is run here.

Output: Part 2 Experiments/x5_extrapolation/{coverage.csv, added_tokens.csv, report.md}
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

C = p2x.C
FOREIGN = ["de", "es", "fr", "ru"]


def fit_pieces(concept, method):
    """(Z (d,k) in the fit's space, transform(E_rows) -> fit-space columns, selected feature ids) or None."""
    import pandas as pd
    from ember.erasure import features
    features.MF_OUTPUTS_ROOT = p2x.METHODS[method]
    pkl, pot = features._embedding_paths(C.MODEL_NAME, concept, C.RANK, C.SEED)
    df = pd.read_csv(pot)
    if df.empty:
        return None
    sel = features.select_embed_feature_ids(df, ratio_thresh=2.0)
    if method == "EMBER":
        F, _, _ = features.load_embedding_payload(pkl)
        return torch.as_tensor(F).double(), (lambda X: X.T), sel
    from experiments.varwidth.prep import prepare
    prep = prepare(concept, "whitened", verbose=False)
    fit = torch.load(C.ROOT / "results" / "varwidth" / C.safe(concept) / "rn_refit-False" / "seed42" / "fit.pt",
                     weights_only=False)
    M, mu = prep["T_matrix"].double(), prep["mu_B"].double()
    return fit["Z"].double(), (lambda X: ((X - mu) @ M).T), sel


def extrapolate(Z, A, method):
    from experiments.varwidth.common import VWConfig, mask_from_row_widths, pool_floor_cap
    cfg = VWConfig(method="rn", refit=False)
    G = Z.T @ Z + cfg.ridge * torch.eye(Z.shape[1], dtype=torch.float64)
    Y = torch.linalg.solve(G, Z.T @ A).float()
    k, n = Y.shape
    if method == "EMBER":
        mask = mask_from_row_widths(Y.abs(), torch.full((k,), math.ceil(cfg.sparsity * n), dtype=torch.long))
    else:
        K = math.ceil(cfg.sparsity * n) * k
        mask = pool_floor_cap(Y.abs() / Y.norm(dim=1, keepdim=True).clamp_min(1e-12), K,
                              math.ceil(cfg.floor * n), math.ceil(cfg.cap * n))
    return Y * mask


def main():
    import edit_coverage as ec
    out = p2x.outdir("x5_extrapolation")
    ec.TOK = p2x.tokenizer()
    E = p2x.embeddings()
    special = set(p2x.tokenizer().all_special_ids)
    cov, added = [], []
    for c in p2x.CONCEPTS:
        en_codes, _ = ec.labels(c, ("en",), str(p2x.DATA_V2))
        for m in p2x.METHODS:
            pieces = fit_pieces(c, m)
            if pieces is None:
                continue
            Z, transform, sel = pieces
            E0 = set(p2x.edit(c, m)[0])
            for L in FOREIGN:
                codes, _ = ec.labels(c, ("en", L), str(p2x.DATA_V2))
                V = sorted(t for t in codes if t not in special)
                Y = extrapolate(Z, transform(E[torch.tensor(V)]), m)
                ys = Y[sel].abs().max(0).values.numpy()
                Ex = {t for t, y in zip(V, ys) if y > 0 and codes.get(t) == 1}
                q0 = {f"q_cov_{x}": float(np.mean([bool((q | a) & E0) for q, a in ec.questions(c, x)])) for x in ("en", L)}
                q1 = {f"q_cov_{x}": float(np.mean([bool((q | a) & Ex) for q, a in ec.questions(c, x)])) for x in ("en", L)}
                new = [(t, y) for t, y in zip(V, ys) if t in Ex and t not in E0]
                new.sort(key=lambda z: -z[1])
                cov.append({"concept": c, "method": m, "language": L, "n_edit_before": len(E0), "n_edit_after": len(Ex),
                            "n_added": len(new), "n_added_not_in_english": sum(t not in en_codes for t, _ in new),
                            "kept_original_%": round(100 * len(E0 & Ex) / max(len(E0), 1)),
                            "q_cov_en_before": round(q0["q_cov_en"], 2), "q_cov_en_after": round(q1["q_cov_en"], 2),
                            "q_cov_L_before": round(q0[f"q_cov_{L}"], 2), "q_cov_L_after": round(q1[f"q_cov_{L}"], 2)})
                for t, y in new[:40]:
                    added.append({"concept": c, "method": m, "language": L, "token": p2x.tok_str([t])[0],
                                  "abs_y": round(float(y), 4), "foreign_only": t not in en_codes})
                print(c, m, L, cov[-1]["n_added"], cov[-1]["q_cov_L_before"], "->", cov[-1]["q_cov_L_after"], flush=True)
    Cv, Ad = pd.DataFrame(cov), pd.DataFrame(added)
    Cv.to_csv(out / "coverage.csv", index=False)
    Ad.to_csv(out / "added_tokens.csv", index=False)
    L_ = ["# X5: extrapolating the English features to native foreign tokens (no erasure run)", "",
          "Z fixed, coefficients solved for English + native L tokens, each method's sparsity rule. `q_cov_L`: share "
          "of L's 50 QA_test questions containing an edited token, before → after. `kept_original_%`: share of the "
          "original English edit set still edited.", "", p2x.md_table(Cv), "",
          "## Added tokens (largest coefficient first, foreign-only marked *)", ""]
    for (c, m, L), g in Ad.groupby(["concept", "method", "language"], sort=False):
        L_.append(f"- {c} / {m} / {L}: " + " ".join(f"{r.token}{'*' if r.foreign_only else ''}" for r in g.head(25).itertuples()))
    (out / "report.md").write_text("\n".join(L_) + "\n")
    print(p2x.md_table(Cv))


if __name__ == "__main__":
    main()
