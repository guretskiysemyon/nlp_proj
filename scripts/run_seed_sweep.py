#!/usr/bin/env python
"""Run steps 3.1-3.2 of a low_ram_demands notebook for one seed, headless.

    python scripts/run_seed_sweep.py {plain|whitened} SEED [--out-dir DIR]

Builds a copy of the notebook that
  - sets SEED (the SMF seed; neutral_sample_seed stays 42) via an injected first cell,
  - sends RUN_DIR (and ORIG_DIR) caches to a per-seed subfolder, so other seeds' caches are untouched,
  - stops right after the "3.2.f" wide-table cell and appends a cell that writes a JSON summary.
The source notebooks are not modified. Outputs in DIR: <name>_seed<N>.ipynb / .html / .json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = {
    "plain": ROOT / "low_ram_demands" / "ancient_rome_varnnz.ipynb",
    "whitened": ROOT / "low_ram_demands" / "ancient_rome_whitened_varnnz.ipynb",
}
NAMES = {"plain": "not_whitened", "whitened": "whitened"}
TARGET = "# One column per selected feature: every token with nonzero Y_{i,t}"

SUMMARY_CELL = '''
_per_feat = {int(f): int((Y[f] != 0).sum()) for f in F_C}
_summary = {"seed": SEED, "n_concept_features": len(F_C), "F_C": [int(f) for f in F_C],
            "n_distinct_tokens": len(all_cols), "tokens_per_feature": _per_feat,
            "n_tokens_summed": sum(_per_feat.values()),
            "C": int(sum(codes[j] == 1 for j in all_cols)), "N": int(sum(codes[j] == 0 for j in all_cols)),
            "B": int(sum(codes[j] == 2 for j in all_cols)), "rel_err": float(fit_out["rel_err"]),
            "vstar": len(vstar_ids), "run_dir": str(RUN_DIR)}
Path(__SUMMARY_PATH__).write_text(json.dumps(_summary, indent=1))
print(_summary)
'''


def build(kind: str, seed: int, summary_path: Path):
    nb = nbformat.read(NOTEBOOKS[kind], as_version=4)
    cut = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.startswith(TARGET))
    nb.cells = nb.cells[: cut + 1]
    patched = 0
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        lines = c.source.split("\n")
        for i, ln in enumerate(lines):
            if ln.startswith("RUN_DIR = Path(") or ln.startswith("ORIG_DIR = Path("):
                code, sep, comment = ln.partition("   #")
                lines[i] = f'{code.rstrip()} / f"seed{{SEED}}"' + (f"   #{comment}" if sep else "")
                patched += 1
        c.source = "\n".join(lines)
    assert patched == (2 if kind == "whitened" else 1), patched
    inject = nbformat.v4.new_code_cell(f"_OVERRIDES = {{'SEED': {seed}}}\nSEED = {seed}")
    inject.metadata["tags"] = ["injected-parameters"]
    nb.cells.insert(1, inject)  # after the cwd cell, before the config cell
    nb.cells.append(nbformat.v4.new_code_cell(SUMMARY_CELL.replace("__SUMMARY_PATH__", repr(str(summary_path)))))
    return nb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=list(NOTEBOOKS))
    p.add_argument("seed", type=int)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "ancient_rome_seed_sweep")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = a.out_dir / f"{NAMES[a.kind]}_seed{a.seed}"
    nb = build(a.kind, a.seed, stem.with_suffix(".json"))
    t0 = time.time()
    client = NotebookClient(nb, kernel_name="python3", timeout=None, resources={"metadata": {"path": str(ROOT / "low_ram_demands")}})
    err = None
    try:
        client.execute()
    except Exception as e:  # keep the partial notebook for debugging
        err = e
    nbformat.write(nb, stem.with_suffix(".ipynb"))
    html, _ = HTMLExporter().from_notebook_node(nb)
    stem.with_suffix(".html").write_text(html)
    print(f"{stem.name}: {'FAILED' if err else 'done'} in {time.time() - t0:.0f}s", flush=True)
    if err:
        print(err, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
