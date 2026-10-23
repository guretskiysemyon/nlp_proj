#!/usr/bin/env python
"""Seed-union approaches A1U and A3zU (PREREG_NEW.md, amendment 1).

    ML_CONCEPTS=all python -m experiments.ml.union --prep      # A3z fits for seeds 1-2, union.json, reused units
    python -m experiments.ml.worker --approach A1U A3zU         # evaluate the rest

Per fit seed s in SEEDS the approach's own edit is built (A1: merged by page priority; A3z: A3's selection with Z
re-solved on the original embeddings). Pooled edit: every token edited by any seed, c_t = mean of the seeds' c_t.
Seed 42 is the main run (results/ml/runs/); seeds 1 and 2 are results/ml/seeds/fs<s>/.
--prep writes run_dir(concept, U)/union.json ({"seeds_with_edit", "n_edited"}) and, when no seed or only seed 42
edits, the finished eval.json directly (no edit -> score 0; seed 42 only -> the base approach's eval.json, since
evaluation is deterministic).
"""
from __future__ import annotations

import argparse
import json

import torch

from experiments.ml.common import LANGS, RIDGE, concepts, embeddings, run_dir, save_json
from experiments.ml.seeds import sdir

SEEDS = (42, 1, 2)
BASE = {"A1U": "A1", "A3zU": "A3z"}


def seed_edit(concept, seed, base):
    from experiments.ml.data.io import page_priority
    from experiments.ml.evaluate import run_edit
    if base == "A3z":
        d = sdir(concept, seed, "A3z")
        return run_edit(d / "fit.pt", d / "selection.json")
    merged = {}
    for L in page_priority(concept):           # highest priority first: a token keeps the first language's edit
        d = sdir(concept, seed, "A1", L)
        for t, c in run_edit(d / "fit.pt", d / "selection.json").items():
            merged.setdefault(t, c)
    return merged


def pooled(concept, approach):
    """(pooled edit {token: c_t}, seeds whose own edit is non-empty)."""
    per = {s: seed_edit(concept, s, BASE[approach]) for s in SEEDS}
    used = [s for s in SEEDS if per[s]]
    toks = sorted(set().union(*(per[s] for s in used))) if used else []
    return {t: torch.stack([per[s][t].float() for s in used if t in per[s]]).mean(0) for t in toks}, used


def build_a3z_seeds(E):
    """seeds/fs<s>/<slug>/A3z/{fit.pt, selection.json} from the seed's A3 fit (same construction as a3z.py)."""
    import shutil
    for s in SEEDS[1:]:
        for c in concepts():
            src, dst = sdir(c, s, "A3"), sdir(c, s, "A3z")
            if (dst / "fit.pt").exists():
                continue
            fit = torch.load(src / "fit.pt", weights_only=False)
            Y = fit["Y"].double()
            A = E[fit["vstar_ids"]].double().T
            Zo = torch.linalg.solve(Y @ Y.T + RIDGE * torch.eye(Y.shape[0], dtype=torch.float64), Y @ A.T).T
            dst.mkdir(parents=True, exist_ok=True)
            fit.update({"Z": Zo.float(), "unmap": None, "approach": "A3z"})
            torch.save(fit, dst / "fit.pt")
            shutil.copy(src / "selection.json", dst / "selection.json")


def prep():
    torch.set_num_threads(8)
    build_a3z_seeds(embeddings())
    for c in concepts():
        for u, base in BASE.items():
            edit, used = pooled(c, u)
            d = run_dir(c, u)
            save_json(d / "union.json", {"seeds_with_edit": used, "n_edited": len(edit)})
            ev = d / "eval.json"
            if ev.exists() and json.loads(ev.read_text()).get("done"):
                continue
            if not used:
                save_json(ev, {"concept": c, "approach": u, "n_edited": 0, "done": True, "seeds_with_edit": []})
            elif used == [42]:
                r = json.loads((run_dir(c, base) / "eval.json").read_text())
                r.update({"approach": u, "seeds_with_edit": used, "reused_from": base})
                save_json(ev, r)
                for f in ("rescore_v2.json",):  # the 18 original concepts are scored on the v2 questions
                    if (run_dir(c, base) / f).exists():
                        save_json(d / f, json.loads((run_dir(c, base) / f).read_text()))
            print(f"[{u} {c}] seeds {used} n_edited {len(edit)}"
                  f"{' (reused)' if used == [42] else ' (no edit)' if not used else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    a = ap.parse_args()
    if a.prep:
        prep()


if __name__ == "__main__":
    main()
