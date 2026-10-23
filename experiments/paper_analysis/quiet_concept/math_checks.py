"""Numerical checks of the appendix statements (Harry Potter, plus two more concepts).

    python -m experiments.paper_analysis.quiet_concept.math_checks

(1) Gram lemma: the ALS of fit_varwidth depends on A only through K = A^T A. Run the fixed RS-1% fit for 60
    iterations on A and on Q A (Q a random orthogonal d x d matrix); the Y iterates must coincide.
(2) Page kernel: X^T X = ((1 - eta)/m) K_VP K_PV + eta K_VV with K_ab = (e_a - mu_B)^T (Sigma_B + eps I)^{-1} (e_b - mu_B),
    P = page token types (gamma = 1).
(3) Gains: the eigenvalues g_j of S_eta equal (1 - eta) lambda_j + eta, lambda_j the generalized eigenvalues of
    (M_page, Sigma_B + eps I), M_page = (1/m) sum_j (e_j - mu_B)(e_j - mu_B)^T.
(4) E_X(p_j) = g_j E_U(p_j) for every j (checked in spectra.py as check_EX_eq_gEU_maxrel).
Output: results/paper_analysis/quiet_concept/math_checks.json
"""
from __future__ import annotations

import json

import torch

from experiments.paper_analysis.quiet_concept.common import OUT, WH_DIR, dump, matrices, whitener
from experiments.varwidth.common import VWConfig, slug
from experiments.varwidth.core import fit_varwidth


def page_rows(concept: str) -> torch.Tensor:
    from experiments.varwidth.prep import get_tokenizer, load_input_embeddings
    from ember.utils import get_special_token_ids
    from experiments.varwidth.prep import TokenizerOnly
    tok = get_tokenizer()
    special = get_special_token_ids(TokenizerOnly(tok))
    E = load_input_embeddings()
    tgt = json.loads((WH_DIR / "targets" / f"{slug(concept)}.json").read_text())["page"]
    ids = sorted({int(t) for t in tgt if 0 <= int(t) < E.shape[0]} - set(special))
    rows = E[torch.tensor(ids)].double()
    del E
    return rows


def main():
    torch.set_num_threads(4)
    out = {}
    # (1) Gram lemma
    M = matrices("Harry Potter")
    A = M["raw"].float()
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(A.shape[0], A.shape[0], generator=g, dtype=torch.float64))
    cfg = VWConfig(method="fixed", refit=False, max_iter=60, patience=10**6)
    f1 = fit_varwidth(A, cfg, 42)
    f2 = fit_varwidth((Q @ A.double()).float(), cfg, 42)
    sup_same = bool(((f1["Y"].abs() > 0) == (f2["Y"].abs() > 0)).all())
    out["gram_lemma"] = {"iters": 60, "support_identical": sup_same,
                         "Y_max_abs_diff": float((f1["Y"] - f2["Y"]).abs().max()),
                         "rel_err": [f1["rel_err"], f2["rel_err"]],
                         "Z_rotated_max_diff": float((Q @ f1["Z"].double() - f2["Z"].double()).abs().max()
                                                     / f2["Z"].abs().max())}
    print(out["gram_lemma"], flush=True)
    # (2), (3)
    wh = whitener()
    for c in ("Harry Potter", "Ancient Rome", "Valentine's Day"):
        M = matrices(c)
        Ep = page_rows(c)
        Up = (Ep - wh["mu"]) @ wh["W"]
        m = Up.shape[0]
        eta = M["eta"]
        Uv = M["wo"].T
        K_VP = Uv @ Up.T
        Kx = (1 - eta) / m * K_VP @ K_VP.T + eta * Uv @ Uv.T
        X = M["cs"].T
        G = X @ X.T
        rel2 = float((G - Kx).norm() / G.norm())
        Sig = (wh["Q"] * (wh["lam"] + wh["eps"])) @ wh["Q"].T
        Mp = (Ep - wh["mu"]).T @ (Ep - wh["mu"]) / m
        L = torch.linalg.cholesky(Sig)
        Li = torch.linalg.inv(L)
        lam_gen = torch.linalg.eigvalsh(Li @ Mp @ Li.T).flip(0)
        Sp = wh["W_inv"] @ M["T"]
        gS = torch.linalg.eigvalsh((Sp + Sp.T) / 2).flip(0) ** 2
        rel3 = float(((1 - eta) * lam_gen + eta - gS).abs().max() / gS.max())
        out[c] = {"m_page": m, "page_kernel_rel_err": rel2, "gain_identity_max_rel_err": rel3,
                  "lambda1": float(lam_gen[0]), "g1": float(gS[0]), "n_lambda_gt1": int((lam_gen > 1).sum()),
                  "n_lambda_lt_0.1": int((lam_gen < 0.1).sum())}
        print(c, out[c], flush=True)
    dump(out, OUT / "math_checks.json")


if __name__ == "__main__":
    main()
