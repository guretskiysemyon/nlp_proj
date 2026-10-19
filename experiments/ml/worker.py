#!/usr/bin/env python
"""Evaluation worker: loads Llama once, then claims (concept, approach) units from a shared queue until none is left.

    python -m experiments.ml.worker [--approach A1 A2 A3]

Local (c-001) and SLURM workers share the queue through claim files results/ml/claims/<slug>__<approach>
(created with O_EXCL, so two workers never take the same unit). A unit is done when its eval.json has done=true.
A claim whose unit has shown no progress (claim and eval.json both untouched) for STALE seconds is taken over:
that covers a preempted SLURM job or a killed local process. Units with edits go first.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time

from experiments.ml.common import APPROACHES, RESULTS, concepts, run_dir, slug

CLAIMS = RESULTS / "claims"
STALE = 90 * 60


def has_edit(c, a):
    from experiments.ml.common import LANGS
    sel = [run_dir(c, a, L) / "selection.json" for L in LANGS] if a == "A1" else [run_dir(c, a) / "selection.json"]
    return any(json.loads(p.read_text())["F_C"] for p in sel if p.exists())


def done(c, a):
    p = run_dir(c, a) / "eval.json"
    return p.exists() and json.loads(p.read_text()).get("done", False)


def claim(c, a) -> bool:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    p = CLAIMS / f"{slug(c)}__{a}"
    me = f"{socket.gethostname()} {os.getpid()} {os.environ.get('SLURM_JOB_ID', 'local')} {time.time():.0f}"
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
        os.write(fd, me.encode())
        os.close(fd)
        return True
    except FileExistsError:
        ev = run_dir(c, a) / "eval.json"
        last = max(p.stat().st_mtime, ev.stat().st_mtime if ev.exists() else 0)
        if time.time() - last > STALE:
            p.write_text(me + " (took over)")
            return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", nargs="+", default=list(APPROACHES))
    a = ap.parse_args()
    units = [(c, x) for c in concepts() for x in a.approach]
    units.sort(key=lambda u: not has_edit(*u))
    from experiments.ml.evaluate import Editor, evaluate, load_model
    model, tok = load_model()
    editor = Editor(model)
    for c, x in units:
        if done(c, x) or not claim(c, x):
            continue
        t0 = time.time()
        print(f"[worker {socket.gethostname()}] start {c} {x}", flush=True)
        try:
            evaluate(model, tok, editor, c, x)
        except Exception as e:  # noqa: BLE001 — log, release the claim, go on
            print(f"[worker] FAILED {c} {x}: {type(e).__name__}: {e}", flush=True)
            (CLAIMS / f"{slug(c)}__{x}").unlink(missing_ok=True)
            editor.restore()
            continue
        print(f"[worker] done {c} {x} in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
