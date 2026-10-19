#!/usr/bin/env python
"""G1: retention and leak at EVERY delta of the grid, both methods, 5 languages (GPU, Gemma only, no judge).

Part 1 and the main CINDER run measure the held-out questions only at the picked delta. Here the same cells
(QA_test + SimdomQA_test in en/de/es/fr/ru, crosslingual.rerun_fixed.full_cell: the baseline's items and shuffle,
mc_fixed scoring) are scored at each delta of the grid, so retention and leak can be read as curves against edit
strength. Descriptive only: the test split is never used to choose anything.

Output: Part 2 Experiments/g1_leak_curves/<method>/<Concept>/delta_<d>/<lang>.json (per-question records)
        and <method>/<Concept>/curve.csv (retention per language and delta, pristine from baseline_fixed).

    python "Part 2 Experiments/scripts/g1_leak_curves.py" --method CINDER --concept Golf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402

GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
BASE = p2x.ROOT / "results" / "crosslingual" / "google_gemma-2-2b-it" / "baseline_fixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(p2x.ALL_METHODS), required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--deltas", type=float, nargs="+", default=GRID)
    ap.add_argument("--langs", nargs="+", default=p2x.LANGS)
    ap.add_argument("--out", type=Path, default=None, help="default: Part 2 Experiments/g1_leak_curves/<method>/<C>")
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(p2x.ROOT / ".env")
    import pandas as pd
    import torch
    from crosslingual.data import SETS
    from crosslingual.rerun_fixed import Generator, _write_json, full_cell
    from crosslingual.run_edited import _retention as retention   # Part 1's definition
    from ember.erasure import embed_edit, features, model_loader
    from ember.evals.model_wrap import ensure_wrapped_model

    features.MF_OUTPUTS_ROOT = p2x.ALL_METHODS[a.method]
    out = a.out or p2x.outdir(f"g1_leak_curves/{a.method}/{p2x.C.safe(a.concept)}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model, tok = model_loader.load_hf_model(p2x.C.MODEL_NAME, "", dtype=model_loader.pick_dtype("bf16"))
    model.eval()
    tm = ensure_wrapped_model(model, tok)
    gen = Generator(tm)
    snap = embed_edit.snapshot(model)
    rows = []
    for d in a.deltas:
        ddir = out / f"delta_{d}"
        t0 = time.time()
        embed_edit.restore(model, snap)
        info = embed_edit.apply_concept_embed_edit_factored(
            hf_model=model, model_name=p2x.C.MODEL_NAME, concept_name=a.concept, delta_embed=d,
            rank=p2x.C.RANK, seed=p2x.C.SEED, ratio_thresh=2.0)
        if info["n_tokens_edited"] == 0:
            print(f"{a.method} {a.concept}: no edit, stopping", flush=True)
            break
        for L in a.langs:
            path = ddir / f"{L}.json"
            if path.exists():
                cell = json.loads(path.read_text())
            else:
                with torch.no_grad():
                    cell = {"concept": a.concept, "language": L, "method": a.method, "delta": d, "edit_info": info,
                            "results": full_cell(gen, a.concept, L, 42)}
                _write_json(path, cell)
            base = json.loads((BASE / p2x.C.safe(a.concept) / f"{L}.json").read_text())
            r = {"concept": a.concept, "method": a.method, "delta": d, "language": L,
                 "n_tokens_edited": info["n_tokens_edited"], "reliable": base.get("reliable")}
            for s in SETS:
                r[f"{s}_acc"] = cell["results"][s]["accuracy"]
                r[f"{s}_pristine"] = base["results"][s]["accuracy"]
            r["retention"] = retention(r["QA_test_acc"], r["QA_test_pristine"])
            rows.append(r)
        print(f"{a.method} {a.concept} delta={d}: " + " ".join(
            f"{x['language']}={x['QA_test_acc']:.2f}" for x in rows if x["delta"] == d) + f" ({time.time() - t0:.0f}s)",
            flush=True)
        pd.DataFrame(rows).to_csv(out / "curve.csv", index=False)
    embed_edit.restore(model, snap)


if __name__ == "__main__":
    main()
