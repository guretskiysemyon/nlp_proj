"""Copy the project's existing fits of the S = 300 base variants into WP4's compact format (no refit).

    .venv/bin/python -m experiments.paper_analysis.robustness.import_project

Valid because WP4's pipeline reproduces the project's prep exactly (variants.py --check) and a GPU refit of
Harry Potter seed 42 (EMBER and CS) gave the project's CPU fit (identical support, max |dY| < 2e-6).
"""
from __future__ import annotations

import json

import torch

from experiments.paper_analysis.robustness.concepts import DATA, PROJECT_TAG, SEEDS
from experiments.paper_analysis.robustness.pipeline import fit_path, save_fit
from experiments.varwidth.common import run_dir, slug


def main():
    variants = json.loads((DATA / "variants.json").read_text())
    n = 0
    for v, c in variants.items():
        if v != slug(c):
            continue
        for cond, tag in PROJECT_TAG.items():
            for s in SEEDS:
                src = run_dir(c, tag, s) / "fit.pt"
                dst = fit_path(v, cond, s)
                if dst.exists() or not src.exists():
                    continue
                fit = torch.load(src, weights_only=False, map_location="cpu")
                matrix = "plain" if tag.endswith("_plain") else "whitened"
                save_fit(dst, fit, {"variant": v, "concept": c, "cond": cond, "matrix": matrix,
                                    "target": None if matrix == "plain" else "page", "source": str(src)})
                n += 1
    print(f"imported {n} project fits")


if __name__ == "__main__":
    main()
