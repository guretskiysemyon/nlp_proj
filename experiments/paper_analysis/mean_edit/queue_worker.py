#!/usr/bin/env python
"""WP6b mean_edit: a claim-based task queue so that a few GPU workers (SLURM jobs and the local GPU) each load the
model once and work through the tasks in priority order.

    python -m experiments.paper_analysis.mean_edit.queue_worker init      # write queue/tasks.json (once)
    python -m experiments.paper_analysis.mean_edit.queue_worker work      # claim and run tasks until none is left
    python -m experiments.paper_analysis.mean_edit.queue_worker status

A task is claimed by an atomic mkdir of queue/claims/<id>; done tasks get queue/done/<id>. A task runs only when the
tasks it requires are done (the worker waits otherwise). Stages skip outputs that exist, so a task whose worker died
can be re-run after removing its claim directory.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback

from experiments.paper_analysis.mean_edit.run import OUT
from experiments.paper_analysis.mean_edit.sets import CINDER_RUNS, EMBER_RUNS

Q = OUT / "queue"


def tid(fam, c, s, phase):
    return f"{phase}_{fam}_{c.replace(' ', '_').replace(chr(39), '')}_{s}"


def init():
    tasks = []
    for fam, runs in (("C", CINDER_RUNS), ("E", EMBER_RUNS)):          # phase 1: own-r grid, copy, integrity
        for c, s in runs:
            tasks.append({"id": tid(fam, c, s, "p1"), "fam": fam, "concept": c, "seed": s,
                          "stages": ["grid", "copy", "integ"], "requires": []})
    for c in ("Harry Potter", "Ancient Rome", "Valentine's Day"):
        tasks.append({"id": f"floor_{c.replace(' ', '_').replace(chr(39), '')}", "fam": None, "concept": c,
                      "seed": None, "stages": ["floor"], "requires": []})
    for phase, stage in (("p2", "final"), ("p3", "dgrid")):           # phase 2: test passes; 3: delta-matched grids
        for fam, runs in (("C", CINDER_RUNS), ("E", EMBER_RUNS)):
            for c, s in runs:
                tasks.append({"id": tid(fam, c, s, phase), "fam": fam, "concept": c, "seed": s, "stages": [stage],
                              "requires": [tid(fam, c, s, "p1")]})
    Q.mkdir(parents=True, exist_ok=True)
    for d in ("claims", "done", "failed"):
        (Q / d).mkdir(exist_ok=True)
    (Q / "tasks.json").write_text(json.dumps(tasks, indent=1))
    print(f"{len(tasks)} tasks")


def status():
    tasks = json.loads((Q / "tasks.json").read_text())
    for t in tasks:
        st = "done" if (Q / "done" / t["id"]).exists() else "FAILED" if (Q / "failed" / t["id"]).exists() else \
            ("claimed " + (Q / "claims" / t["id"] / "owner").read_text().strip()
             if (Q / "claims" / t["id"] / "owner").exists() else "pending")
        print(f"{t['id']:40s} {st}")


def run_task(t):
    from experiments.paper_analysis.mean_edit import run as R
    R.ONLY["family"] = t["fam"]
    for st in t["stages"]:
        t0 = time.time()
        if st == "floor":
            R.st_floor(t["concept"])
        elif st == "grid":
            R.st_grid(t["concept"], t["seed"])
        elif st == "dgrid":
            R.st_grid(t["concept"], t["seed"], delta=True)
        elif st == "copy":
            R.st_copy(t["concept"], t["seed"])
        elif st == "integ":
            R.st_integ(t["concept"], t["seed"])
        elif st == "final":
            R.st_final(t["concept"], t["seed"])
        print(f"=== {t['id']} stage {st}: {time.time() - t0:.0f}s", flush=True)


def work():
    import torch
    from dotenv import load_dotenv

    from experiments.varwidth.common import ROOT
    load_dotenv(ROOT / ".env", override=False)
    torch.set_num_threads(4)
    owner = f"{socket.gethostname()} pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', 'local')}"
    while True:
        tasks = json.loads((Q / "tasks.json").read_text())
        done = {t["id"] for t in tasks if (Q / "done" / t["id"]).exists()}
        left = [t for t in tasks if t["id"] not in done and not (Q / "failed" / t["id"]).exists()]
        if not left:
            print("queue empty", flush=True)
            return
        claimed = None
        for t in left:
            if not all(r in done for r in t["requires"]):
                continue
            try:
                os.mkdir(Q / "claims" / t["id"])
            except FileExistsError:
                continue
            (Q / "claims" / t["id"] / "owner").write_text(owner + f" t={time.strftime('%H:%M:%S')}\n")
            claimed = t
            break
        if claimed is None:
            if all((Q / "claims" / t["id"]).exists() for t in left):
                print("all remaining tasks are claimed by others", flush=True)
                return
            time.sleep(60)                       # waiting for a prerequisite
            continue
        print(f">>> {claimed['id']} ({owner})", flush=True)
        try:
            run_task(claimed)
            (Q / "done" / claimed["id"]).write_text(owner + f" t={time.strftime('%H:%M:%S')}\n")
        except Exception:                        # noqa: BLE001
            (Q / "failed" / claimed["id"]).write_text(traceback.format_exc())
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    {"init": init, "work": work, "status": status}[sys.argv[1]]()
