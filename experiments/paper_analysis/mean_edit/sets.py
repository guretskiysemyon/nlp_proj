"""WP6b mean_edit: the new edit sets (MEAN-C, EMBER-c, MEAN-E) next to the cached collateral sets.

Every set is a dict like the cached collateral sets (`experiments.varwidth.collateral.edit_sets`): {"ids", "cols",
"C" (n, d) rows to subtract at delta = 1 in the TL-scaled embedding space, "F", "fit", "dense"}; the pipeline's
`Edits.rows("sub:<name>", r)` then applies e' = e - (r / median_t |c_t|/|e_t|) c, i.e. r is computed from each
variant's own edit lengths.

    cinder    cached (CS + GS fit L12, HJ judge, Z mapped back with T^-1)            centred
    cinder_z  cached (CINDER's Y and F_C, Z re-solved on the raw A)                   uncentred
    mean_c    beta_t^z mu_B on CINDER's tokens, beta^z from CINDER-z (WP6 Corollary 2): b = (Y Y^T + lam I)^-1 Y 1_n,
              beta_t = sum_{i in F_C} b_i Y_it. cinder_z = (centred re-solve) + mean_c exactly.
    ember     cached (plain fit L0, TJ judge, the fit's own Z)                        uncentred
    ember_c   EMBER's Y and F_C, Z re-solved on the centred A - mu_B 1^T
    mean_e    beta_t^E mu_B on EMBER's tokens (b from EMBER's Y)                      ember ~= ember_c + mean_e
    mean_cd / mean_ed  the delta-matched mean components: the same rows as mean_c / mean_e, but the grid parameter
              is converted with the PARENT's median length (cinder_z / ember), so mean_cd at grid point r is exactly
              the mean part of CINDER-z at its grid point r (secondary, see PREREG).
"""
from __future__ import annotations

import functools

import numpy as np
import torch

LAM = 1e-4
CINDER_RUNS = [("Harry Potter", 42), ("Harry Potter", 43), ("Ancient Rome", 42), ("Ancient Rome", 44),
               ("Valentine's Day", 42), ("Valentine's Day", 43), ("Valentine's Day", 46)]
EMBER_RUNS = [("Harry Potter", 42), ("Harry Potter", 43), ("Ancient Rome", 43), ("Ancient Rome", 44),
              ("Valentine's Day", 42), ("Valentine's Day", 46)]
DEV_RUNS = [("Harry Potter", 42), ("Harry Potter", 43), ("Ancient Rome", 42), ("Ancient Rome", 43),
            ("Ancient Rome", 44), ("Valentine's Day", 42), ("Valentine's Day", 43), ("Valentine's Day", 46)]
NEW = ("mean_c", "ember_c", "mean_e", "mean_cd", "mean_ed")
PARENT = {"mean_c": "cinder", "mean_cd": "cinder", "ember_c": "ember", "mean_e": "ember", "mean_ed": "ember"}
DELTA_MATCH = {"mean_cd": "cinder_z", "mean_ed": "ember"}


def _b(Y: torch.Tensor) -> torch.Tensor:
    k = Y.shape[0]
    return torch.linalg.solve(Y @ Y.T + LAM * torch.eye(k, dtype=Y.dtype), Y @ torch.ones(Y.shape[1], dtype=Y.dtype))


def _resolve(B: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    k = Y.shape[0]
    return torch.linalg.solve(Y @ Y.T + LAM * torch.eye(k, dtype=Y.dtype), Y @ B.T).T


@functools.lru_cache(maxsize=16)
def new_sets(concept: str, seed: int) -> dict:
    """{name: set} for the new variants of one run (empty set where the parent has no edit), plus '_check'."""
    from experiments.varwidth.collateral import edit_sets, load_fit
    cached = edit_sets(concept, seed)
    prep_w = load_fit(concept, "L12", seed)[0]
    mu = prep_w["mu_B"].double()
    A = load_fit(concept, "L0", seed)[0]["A"].double()             # raw TL-scaled E_V*^T (d, n)
    E_all = A.T
    out, chk = {}, {}
    for parent, cfg, mean_name, cen_name in (("cinder", "L12", "mean_c", None), ("ember", "L0", "mean_e", "ember_c")):
        st = cached[parent]
        names = [mean_name, mean_name[:-1] + mean_name[-1] + "d"] + ([cen_name] if cen_name else [])
        if not st["ids"]:
            for n in names:
                out[n] = {**st, "C": torch.zeros(0, A.shape[0]), "fit": st["fit"] + "-" + n}
            continue
        _, Z, Y, *_ = load_fit(concept, cfg, seed)
        F, cols = st["F"], st["cols"]
        b = _b(Y)
        beta = (b[F][:, None] * Y[F][:, cols]).sum(0)                  # (n_edit,)
        Cm = beta[:, None] * mu[None, :]
        out[mean_name] = {**st, "C": Cm.float(), "fit": st["fit"] + "-mean", "beta": beta.float()}
        out[mean_name[:-1] + mean_name[-1] + "d"] = {**out[mean_name], "fit": st["fit"] + "-mean-dmatch"}
        Cpar = st["C"].double()
        E = E_all[cols]
        c = {"n": len(cols), "beta_median": float(beta.median()), "beta_min": float(beta.min()),
             "mu_norm": float(mu.norm()), "e_norm_median": float(E.norm(dim=1).median()),
             "e_mu_coord_median": float((E @ mu / mu.norm() ** 2).median())}
        if parent == "cinder":
            Cz = cached["cinder_z"]["C"].double()
            diff = Cz - Cpar
            c["z_minus_c_vs_mean_relerr"] = float((diff - Cm).norm() / diff.norm())
            c["z_minus_c_mu_energy_share"] = float(((diff @ mu / mu.norm()) ** 2).sum() / (diff ** 2).sum())
            Zc = _resolve(A - mu[:, None], Y)
            Cc = (Zc[:, F] @ Y[F][:, cols]).T
            c["cinder_vs_centred_resolve_relerr"] = float((Cpar - Cc).norm() / Cc.norm())
            c["cinder_z_vs_centred_plus_mean_relerr"] = float((Cz - Cc - Cm).norm() / Cz.norm())
            par_z = Cz
        else:
            Zc = _resolve(A - mu[:, None], Y)
            Cc = (Zc[:, F] @ Y[F][:, cols]).T
            out[cen_name] = {**st, "C": Cc.float(), "fit": st["fit"] + "-c"}
            c["ember_vs_centred_plus_mean_relerr"] = float((Cpar - Cc - Cm).norm() / Cpar.norm())
            Zo = _resolve(A, Y)
            Co = (Zo[:, F] @ Y[F][:, cols]).T
            c["ember_fitZ_vs_raw_resolve_relerr"] = float((Cpar - Co).norm() / Co.norm())
            c["ember_c_median_mu"] = float(np.median((Cc.norm(dim=1) / E.norm(dim=1)).numpy()))
            c["cos_ember_c_to_ember_median"] = float(torch.nn.functional.cosine_similarity(Cc, Cpar, dim=1).median())
            par_z = Cpar
        for nm, C in (("parent", Cpar), ("mean", Cm), ("uncentred", par_z)):
            mu_t = (C.norm(dim=1) / E.norm(dim=1)).numpy()
            c[f"median_mu_{nm}"] = float(np.median(mu_t))
        c["mean_share_of_uncentred_r"] = c["median_mu_mean"] / c["median_mu_uncentred"]
        chk[parent] = c
    out["_check"] = chk
    return out


def all_sets(concept: str, seed: int) -> dict:
    """The cached collateral sets plus the new ones (a fresh dict; the cache file is not modified)."""
    from experiments.varwidth.collateral import edit_sets
    s = dict(edit_sets(concept, seed))
    s.update({k: v for k, v in new_sets(concept, seed).items() if k != "_check"})
    return s


def install():
    """Monkey-patch the erasure pipeline (in this process only) so that `setup` returns the extended sets and the
    delta-matched variants convert r with the parent's median length. No file outside this WP is modified."""
    import experiments.varwidth.erase_v2 as ev2
    if getattr(ev2, "_mean_edit_installed", False):
        return
    ev2.edit_sets = all_sets
    orig_comp = ev2.Edits.comp

    def comp(self, variant):
        kind, setname = variant.split(":")
        if setname in DELTA_MATCH and kind == "sub":
            if variant in self._cache:
                return self._cache[variant]
            ids, E, C, _ = orig_comp(self, variant)
            par_med = orig_comp(self, f"sub:{DELTA_MATCH[setname]}")[3]
            self._cache[variant] = (ids, E, C, par_med)      # delta = r / median mu of the parent
            return self._cache[variant]
        return orig_comp(self, variant)

    ev2.Edits.comp = comp
    ev2._mean_edit_installed = True
