#!/usr/bin/env python
"""A3z: A3's selection, edit built in the original embedding space (results/ml/diagnose_a3/DIAGNOSIS.md).

    python -m experiments.ml.a3z

For every concept with an A3 fit: keep Y, vstar_ids, codes and F_C from A3; re-solve Z with Y fixed against the
original embeddings (ridge 1e-4), Z_o = argmin ||E_V*^T - Z Y||; write run_dir(concept, "A3z")/fit.pt with Z_o and
no unmap, and copy selection.json. evaluate.run_edit then gives c_t = Z_o[:, F_C] y_t. Then run the queue workers with
--approach A3z.
"""
from __future__ import annotations

import shutil

import torch

from experiments.ml.common import RIDGE, concepts, embeddings, run_dir


def main():
    torch.set_num_threads(8)
    E = embeddings()
    for c in concepts():
        src = run_dir(c, "A3")
        if not (src / "fit.pt").exists():
            continue
        dst = run_dir(c, "A3z")
        dst.mkdir(parents=True, exist_ok=True)
        fit = torch.load(src / "fit.pt", weights_only=False)
        Y = fit["Y"].double()
        A = E[fit["vstar_ids"]].double().T
        Zo = torch.linalg.solve(Y @ Y.T + RIDGE * torch.eye(Y.shape[0], dtype=torch.float64), Y @ A.T).T
        err = float(torch.linalg.norm(A - Zo @ Y) / torch.linalg.norm(A))
        fit.update({"Z": Zo.float(), "unmap": None, "approach": "A3z", "rel_err_orig": err})
        torch.save(fit, dst / "fit.pt")
        shutil.copy(src / "selection.json", dst / "selection.json")
        print(f"[A3z {c}] rel_err in the original space {err:.4f}", flush=True)


if __name__ == "__main__":
    main()
