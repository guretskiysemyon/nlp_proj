"""Background whitening + target sharpening (see Instructions.txt).

Row convention, as in the notebooks: embeddings are rows, E_rows is (n, d).

    stats = fit_background(E_B)                       # mu_B, Q, lam (once per model and background)
    bw = BackgroundWhitener(stats, eps_rel)           # W = Q (lam + eps)^-1/2 Q^T, W_inv
    U = bw.whiten(E_V)                                # (E_V - mu_B) W          (W is symmetric)
    sh = Sharpener(U[target_rows], eta, gamma)        # S_eta = (1-eta) U_C^T U_C / m + eta I = P diag(g) P^T
    X = sh.apply(U)                                   # U S_eta^{gamma/2}
    T = Transform(bw, sh)                             # X = T.forward(E_V);  directions back: T.unmap(z_tilde)

Mapping back is exact for directions (the mean cancels in a difference):
    x_t = T(e_t - mu_B) = sum_i Y_it z~_i   =>   e_t - mu_B = sum_i Y_it T^-1 z~_i,   T^-1 = S^{-gamma/2} W^-1.
In row form a direction z~ (d,) maps back as z = z~ S^{-gamma/2} W_inv.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def fit_background(E_B: torch.Tensor) -> dict:
    """Mean and eigendecomposition of the background covariance (float64). E_B: (N_B, d) rows."""
    E = E_B.double()
    mu = E.mean(0)
    Xc = E - mu
    cov = Xc.T @ Xc / E.shape[0]
    lam, Q = torch.linalg.eigh(cov)
    return {"mu": mu, "lam": lam.clamp_min(0), "Q": Q, "n": E.shape[0]}


def _sym_pow(Q: torch.Tensor, vals: torch.Tensor, p: float) -> torch.Tensor:
    return (Q * vals.pow(p)) @ Q.T


class BackgroundWhitener:
    def __init__(self, stats: dict, eps_rel: float):
        lam, Q = stats["lam"], stats["Q"]
        self.mu = stats["mu"]
        self.eps = eps_rel * lam.mean()
        self.lam_eps = lam + self.eps
        self.W = _sym_pow(Q, self.lam_eps, -0.5)
        self.W_inv = _sym_pow(Q, self.lam_eps, 0.5)

    def whiten(self, E_rows: torch.Tensor) -> torch.Tensor:
        return (E_rows.double() - self.mu) @ self.W


class Sharpener:
    """S_eta = (1 - eta) * second moment of the whitened target (around 0) + eta * I."""

    def __init__(self, U_target: torch.Tensor, eta: float, gamma: float):
        U_c = U_target.double()
        d = U_c.shape[1]
        S = (1 - eta) * (U_c.T @ U_c / U_c.shape[0]) + eta * torch.eye(d, dtype=torch.float64)
        g, P = torch.linalg.eigh(S)
        self.g, self.P = g.flip(0), P.flip(1)  # enrichment ratios, largest first; P[:, j] = cPCA direction j
        self.eta, self.gamma = eta, gamma
        self.Sp = _sym_pow(self.P, self.g, gamma / 2)
        self.Sm = _sym_pow(self.P, self.g, -gamma / 2)

    def apply(self, U: torch.Tensor) -> torch.Tensor:
        return U.double() @ self.Sp


@dataclass
class Transform:
    bw: BackgroundWhitener
    sh: Sharpener | None = None  # None = whitening only (gamma = 0)

    def forward(self, E_rows: torch.Tensor) -> torch.Tensor:
        U = self.bw.whiten(E_rows)
        return U if self.sh is None else self.sh.apply(U)

    def unmap(self, z_tilde: torch.Tensor) -> torch.Tensor:
        """Directions from X-space back to the original embedding space. z_tilde: (..., d) rows or (d, k) columns
        given as rows of z_tilde.T -- pass rows."""
        z = z_tilde.double()
        if self.sh is not None:
            z = z @ self.sh.Sm
        return z @ self.bw.W_inv

    @property
    def matrix(self) -> torch.Tensor:
        """M with X = (E - mu_B) M."""
        return self.bw.W if self.sh is None else self.bw.W @ self.sh.Sp


def principal_angle_sines(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Sines of the principal angles between the column spaces of A and B (both d x r, any scale)."""
    Qa, _ = torch.linalg.qr(A.double())
    Qb, _ = torch.linalg.qr(B.double())
    cos = torch.linalg.svdvals(Qa.T @ Qb).clamp(0, 1)
    return (1 - cos ** 2).clamp_min(0).sqrt()
