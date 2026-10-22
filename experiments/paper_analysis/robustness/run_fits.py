"""Run WP4's fit jobs (variants.jobs) on one device; existing fits are skipped. Several workers can share the list.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m experiments.paper_analysis.robustness.run_fits \
        --device cuda [--worker 0 --n-workers 1] [--only-variants V ...] [--order part]

Jobs are grouped by (variant, matrix, target) so A is built once per group. --worker i / --n-workers m takes every
m-th group (by a stable order), so a GPU process and CPU SLURM jobs can split the list.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.paper_analysis.robustness.concepts import CONDITIONS, DATA
from experiments.paper_analysis.robustness.pipeline import fit_path, run_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--only-variants", nargs="*", default=None)
    ap.add_argument("--reverse", action="store_true", help="walk the groups from the end (second worker)")
    ap.add_argument("--clear-stale", type=float, default=None, help="first delete .lock files older than this many minutes")
    a = ap.parse_args()
    if a.clear_stale is not None:
        from experiments.paper_analysis.robustness.concepts import FITS
        for lk in FITS.glob("*/*/*.lock"):
            if time.time() - lk.stat().st_mtime > 60 * a.clear_stale:
                print(f"stale lock removed: {lk}", flush=True)
                lk.unlink(missing_ok=True)
    torch.set_num_threads(a.threads)
    from experiments.paper_analysis.robustness.variants import jobs
    variants = json.loads((DATA / "variants.json").read_text())
    J = jobs(variants)
    if a.only_variants:
        J = [j for j in J if j[0] in a.only_variants]
    groups: dict = {}
    for v, c, cond, s in J:
        m, _, t = CONDITIONS[cond]
        groups.setdefault((v, m, t), []).append((v, c, cond, s))
    keys = sorted(groups)
    keys = [k for i, k in enumerate(keys) if i % a.n_workers == a.worker]
    if a.reverse:
        keys = keys[::-1]
    todo = sum(not fit_path(v, cond, s).exists() for k in keys for v, c, cond, s in groups[k])
    print(f"worker {a.worker}/{a.n_workers} on {a.device}: {len(keys)} groups, {todo} fits to do", flush=True)
    t0, cache = time.time(), {}
    for k in keys:
        for v, c, cond, s in groups[k]:
            if fit_path(v, cond, s).exists():
                continue
            lock = fit_path(v, cond, s).with_suffix(".lock")
            try:
                lock.parent.mkdir(parents=True, exist_ok=True)
                fd = open(lock, "x")
                fd.close()
            except FileExistsError:
                continue
            try:
                run_fit(v, c, cond, s, a.device, cache)
            finally:
                lock.unlink(missing_ok=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
