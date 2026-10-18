#!/usr/bin/env python
"""Run notebooks headless, one after the other, cell by cell (nbclient), saving after every cell.

    python scripts/run_overnight.py NB1.ipynb NB2.ipynb ...            # run in order
    python scripts/run_overnight.py --resume NB1.ipynb NB2.ipynb ...   # skip notebooks already "done"
    python scripts/run_overnight.py --param SMOKE=True NB.ipynb        # quick smoke pass
    python scripts/run_overnight.py --run-name gemma/hp/wiki_page_g1 --param GAMMA=1.0 \
        --stop-before STEP-3.3-START NB.ipynb                          # one config, executed copy elsewhere
    python scripts/run_overnight.py --queue runs/whiten/queue.json     # a list of runs, resumable

- Cells tagged `skip-overnight` are not executed (their old outputs are cleared).
- Each notebook is saved after every cell, so its outputs can be watched while it runs: in place by default, or to
  `--out PATH`. With `--run-name NAME` (and no --out) the executed copy goes to runs/whiten/executed/<NAME>.ipynb, so
  several configurations of one notebook never overwrite each other or the source.
- `--run-name NAME` is the key in status.json (default: the notebook path), so one notebook can be run with different
  `--param` sets as separate entries; `--resume` skips keys whose state is done.
- `--stop-before TEXT` stops at the first cell (code or markdown) whose source contains TEXT; the run is then marked
  done with `partial: true`.
- `--queue FILE.json`: a JSON list of {"notebook", "run_name", "params", "stop_before"} entries ("params" and
  "stop_before" optional), run in order with resume semantics (entries already done are skipped).
- `--param NAME=VALUE` overrides are injected as a first code cell (tag `injected-parameters`) that sets
  `_OVERRIDES = {...}` and the globals themselves; notebooks re-apply `_OVERRIDES` after their config literals.
- Progress goes to runs/status.json ({notebook: {state, cell, started, updated, error}}) and runs/overnight.log.
- On a cell error the notebook is marked failed, its kernel shut down, and the next notebook starts.
  `--resume` skips notebooks whose state is done; failed / running ones restart from the top (their RUN_DIR
  caches make the repeat cheap).
"""
import argparse
import ast
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBER_PY = Path("/homes/orip/.conda/envs/Ember/bin/python")
KERNEL = "ember"
SKIP_TAG = "skip-overnight"
INJECT_TAG = "injected-parameters"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebooks", nargs="*", type=Path, help="notebooks to run, in order, one at a time")
    p.add_argument("--queue", type=Path, default=None, metavar="FILE.json",
                   help='JSON list of {"notebook", "run_name", "params", "stop_before"}; run in order, done entries skipped')
    p.add_argument("--run-name", default=None, metavar="NAME", help="status key for a single notebook (default: its path)")
    p.add_argument("--out", type=Path, default=None, metavar="PATH",
                   help="save the executed copy here (default: runs/whiten/executed/<run-name>.ipynb with --run-name, else in place)")
    p.add_argument("--stop-before", default=None, metavar="TEXT",
                   help="stop before the first cell whose source contains TEXT (run marked done, partial)")
    p.add_argument("--resume", action="store_true", help="skip notebooks whose status is done")
    p.add_argument("--status", type=Path, default=ROOT / "runs" / "status.json", help="status file (default: runs/status.json)")
    p.add_argument("--log", type=Path, default=ROOT / "runs" / "overnight.log", help="log file (default: runs/overnight.log)")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="override injected before the notebook runs (VALUE is a Python literal, else a string); repeatable")
    p.add_argument("--timeout", type=float, default=None, help="per-cell timeout in seconds (default: none)")
    p.add_argument("--kernel", default=KERNEL, help=f"kernel name (default: {KERNEL})")
    args = p.parse_args()
    if bool(args.queue) == bool(args.notebooks):
        p.error("give notebooks or --queue, not both / neither")
    if args.queue and (args.run_name or args.out or args.param or args.stop_before):
        p.error("--queue entries carry their own run_name / params / stop_before; --run-name/--out/--param/--stop-before not allowed")
    if (args.run_name or args.out) and len(args.notebooks) != 1:
        p.error("--run-name / --out apply to exactly one notebook")
    return args


def ensure_packages() -> None:
    """Re-exec under the Ember env if needed, then pip-install nbclient / nbformat / ipykernel when missing."""
    if EMBER_PY.exists() and Path(sys.executable).resolve() != EMBER_PY.resolve() and not os.environ.get("RUN_OVERNIGHT_NO_REEXEC"):
        os.environ["RUN_OVERNIGHT_NO_REEXEC"] = "1"
        os.execv(str(EMBER_PY), [str(EMBER_PY), *sys.argv])
    missing = [m for m in ("nbclient", "nbformat", "ipykernel") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"installing {missing} into {sys.executable}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def parse_params(items) -> dict:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name.isidentifier():
            sys.exit(f"--param expects NAME=VALUE, got {item!r}")
        try:
            out[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[name] = value
    return out


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class Status:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def update(self, nb: str, **fields) -> None:
        self.data.setdefault(nb, {}).update(fields, updated=now())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.path)


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def default_out(path: Path, run_name) -> Path:
    if run_name:
        return ROOT / "runs" / "whiten" / "executed" / f"{run_name}.ipynb"
    return path


def run_notebook(path: Path, key: str, params: dict, args, status: Status, log: Log,
                 out: Path = None, stop_before: str = None) -> bool:
    import nbformat
    from nbclient import NotebookClient

    out = out or path
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if INJECT_TAG not in c.get("metadata", {}).get("tags", [])]
    if params:
        src = (f"# [injected by scripts/run_overnight.py --param] overrides re-applied by the config cell\n"
               f"_OVERRIDES = {params!r}\n"
               f"globals().update(_OVERRIDES)")
        cell = nbformat.v4.new_code_cell(src, metadata={"tags": [INJECT_TAG]})
        nb.cells.insert(0, cell)
    for c in nb.cells:  # clear stale outputs up front, so the saved file only shows this run
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None

    client = NotebookClient(nb, kernel_name=args.kernel, timeout=args.timeout,
                            allow_errors=False, record_timing=True,
                            resources={"metadata": {"path": str(ROOT)}})
    stop_idx = None
    if stop_before:
        hits = [i for i, c in enumerate(nb.cells) if stop_before in c.source]
        if not hits:
            raise ValueError(f"--stop-before {stop_before!r}: no cell of {path} contains it")
        stop_idx = hits[0]
    code_idx = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and (stop_idx is None or i < stop_idx)]
    status.update(key, state="running", cell=None, started=now(), error=None, notebook=str(path), out=str(out),
                  params=params, stop_before=stop_before, partial=None)
    log(f"START {key}  ({len(code_idx)} code cells{f' before cell {stop_idx}' if stop_idx is not None else ''}, "
        f"params={params or {}}, out={out})")
    nbformat.write(nb, out)
    try:
        with client.setup_kernel():
            for n, i in enumerate(code_idx, 1):
                cell = nb.cells[i]
                if SKIP_TAG in cell.get("metadata", {}).get("tags", []):
                    log(f"  {key} cell {i} ({n}/{len(code_idx)}): skipped ({SKIP_TAG})")
                    continue
                first = (cell.source.strip().splitlines() or [""])[0][:80]
                status.update(key, cell=i)
                log(f"  {key} cell {i} ({n}/{len(code_idx)}): {first}")
                t0 = dt.datetime.now()
                try:
                    client.execute_cell(cell, i)
                finally:
                    nbformat.write(nb, out)
                log(f"  {key} cell {i} done in {(dt.datetime.now() - t0).total_seconds():.1f}s")
    except Exception as e:  # CellExecutionError, CellTimeoutError, DeadKernelError, ...
        err = ANSI.sub("", "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        status.update(key, state="failed", error=err[-3000:])
        last = (ANSI.sub("", str(e)).strip().splitlines() or [""])[-1][:200]
        log(f"FAILED {key} at cell {status.data[key].get('cell')}: {type(e).__name__}: {last}")
        try:
            nbformat.write(nb, out)
        except Exception:
            pass
        return False
    status.update(key, state="done", error=None, partial=stop_idx is not None)
    log(f"DONE {key}" + (f" (partial: stopped before cell {stop_idx}, {stop_before!r})" if stop_idx is not None else ""))
    return True


def load_queue(path: Path) -> list:
    entries = json.loads(path.read_text())
    assert isinstance(entries, list), f"{path}: expected a JSON list"
    names = set()
    for n, e in enumerate(entries):
        missing = {"notebook", "run_name"} - set(e)
        unknown = set(e) - {"notebook", "run_name", "params", "stop_before", "out"}
        if missing or unknown:
            sys.exit(f"{path} entry {n}: missing {sorted(missing)} / unknown {sorted(unknown)}")
        if e["run_name"] in names:
            sys.exit(f"{path}: duplicate run_name {e['run_name']!r}")
        names.add(e["run_name"])
    return entries


def abs_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def run_queue(args, status: Status, log: Log) -> None:
    entries = load_queue(abs_path(args.queue))
    for e in entries:
        if not abs_path(e["notebook"]).exists():
            sys.exit(f"queue: no such notebook: {e['notebook']}")
    log(f"=== run_overnight: queue {args.queue} with {len(entries)} entries, kernel={args.kernel}")
    results = {}
    for n, e in enumerate(entries, 1):
        key = e["run_name"]
        if status.data.get(key, {}).get("state") == "done":
            log(f"SKIP [{n}/{len(entries)}] {key} (done)")
            results[key] = "done (skipped)"
            continue
        log(f"QUEUE [{n}/{len(entries)}] {key}")
        path = abs_path(e["notebook"])
        out = abs_path(e["out"]) if e.get("out") else default_out(path, key)
        try:
            ok = run_notebook(path, key, e.get("params") or {}, args, status, log, out=out, stop_before=e.get("stop_before"))
        except Exception as exc:  # e.g. bad stop_before: record and continue with the next entry
            status.update(key, state="failed", error=f"{type(exc).__name__}: {exc}")
            log(f"FAILED {key}: {type(exc).__name__}: {exc}")
            ok = False
        results[key] = "done" if ok else "failed"
    n_done = sum(v.startswith("done") for v in results.values())
    log(f"=== queue summary: {n_done}/{len(results)} done; failed: {[k for k, v in results.items() if v == 'failed']}")
    sys.exit(0 if n_done == len(results) else 1)


def main():
    args = parse_args()
    ensure_packages()
    params = parse_params(args.param)
    os.chdir(ROOT)
    status, log = Status(args.status), Log(args.log)
    if args.queue:
        run_queue(args, status, log)
        return
    paths = [p if p.is_absolute() else (Path.cwd() / p) for p in args.notebooks]
    for p in paths:
        if not p.exists():
            sys.exit(f"no such notebook: {p}")
    log(f"=== run_overnight: {len(paths)} notebook(s), resume={args.resume}, params={params}, kernel={args.kernel}")
    results = {}
    for p in paths:
        key = args.run_name or (str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p.resolve()))
        if args.resume and status.data.get(key, {}).get("state") == "done":
            log(f"SKIP {key} (done)")
            results[key] = "done (skipped)"
            continue
        out = (args.out if args.out.is_absolute() else Path.cwd() / args.out) if args.out else default_out(p, args.run_name)
        try:
            ok = run_notebook(p, key, params, args, status, log, out=out, stop_before=args.stop_before)
        except ValueError as exc:  # bad --stop-before
            sys.exit(str(exc))
        results[key] = "done" if ok else "failed"
    log("=== summary: " + ", ".join(f"{k}: {v}" for k, v in results.items()))
    sys.exit(0 if all(v.startswith("done") for v in results.values()) else 1)


if __name__ == "__main__":
    main()
