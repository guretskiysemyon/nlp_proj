"""Queue of run.py invocations: one line of run.py arguments per run (blank lines and # comments ignored).

    python -m experiments.varwidth.queue LINES.txt --status            # done / failed / running / pending per line
    python -m experiments.varwidth.queue LINES.txt --local 2           # 2 at a time here, OMP threads = 16 // 2
    python -m experiments.varwidth.queue LINES.txt --slurm [--max-parallel 20] [--all] [--dry-run]

--slurm submits one sbatch array (studentkillable / gpu-students, 8 CPUs, 24G, 40 min, no GPU, --exclude=s-005) over
the lines that are not done (all lines with --all); logs in results/varwidth/logs/. done = the run's summary.json
exists; failed = the last attempt exited non-zero (status markers in results/varwidth/logs/status/).
Example line:  --concept "Ancient Rome" --method m2 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.varwidth.common import RESULTS, ROOT, VWConfig, run_dir

LOGS = RESULTS / "logs"
STATUS = LOGS / "status"
HF_HOME = "/home/morg/NLP_2526b/oripetel/.cache/huggingface"
MAX_CPUS = 16


def read_lines(path: Path) -> list[str]:
    out = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def key(line: str) -> str:
    return hashlib.sha1(line.encode()).hexdigest()[:12]


def summary_path(line: str) -> Path:
    from experiments.varwidth.run import make_tag, parse
    a = parse(shlex.split(line))
    if a.merge_from:
        tag = f"{a.merge_from}_merge{a.tag_suffix}"
    else:
        cfg = VWConfig(method=a.method, sparsity=a.sparsity, init=a.init, refit=not a.no_refit,
                       floorcap=not a.no_floorcap, continuation=a.continuation, cap=a.cap)
        tag = make_tag(cfg, a.matrix, a.S, a.tag_suffix)
    return run_dir(a.concept, tag, a.seed) / "summary.json"


def is_done(line: str) -> bool:
    """summary.json exists (and, with --judge-reps N, holds N reruns)."""
    p = summary_path(line)
    if not p.exists():
        return False
    from experiments.varwidth.run import parse
    n = parse(shlex.split(line)).judge_reps
    if n:
        import json
        return len(json.loads(p.read_text()).get("judge_reps", [])) >= n
    return True


def status(line: str) -> str:
    if is_done(line):
        return "done"
    k = key(line)
    rc, start = STATUS / f"{k}.rc", STATUS / f"{k}.start"
    if rc.exists() and (not start.exists() or rc.stat().st_mtime >= start.stat().st_mtime):
        return f"failed(rc={rc.read_text().strip()})"
    if start.exists():
        return "running"
    return "pending"


def exec_line(line: str, threads: int, log_path: Path | None = None) -> int:
    """Run one line (python -m experiments.varwidth.run ...), with start / exit-code markers."""
    STATUS.mkdir(parents=True, exist_ok=True)
    k = key(line)
    (STATUS / f"{k}.start").write_text(f"{time.strftime('%F %T')} {os.uname().nodename}\n{line}\n")
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    cmd = [sys.executable, "-m", "experiments.varwidth.run", *shlex.split(line), "--threads", str(threads)]
    if log_path is None:
        rc = subprocess.call(cmd, cwd=ROOT, env=env)
    else:
        with open(log_path, "a") as f:
            f.write(f"### {time.strftime('%F %T')} {' '.join(shlex.quote(c) for c in cmd)}\n")
            f.flush()
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    (STATUS / f"{k}.rc").write_text(f"{rc}\n")
    return rc


def env_setup() -> None:
    """The HF / Gemini environment, for --local (sbatch scripts do the same in bash)."""
    from dotenv import dotenv_values
    for k, v in dotenv_values(ROOT / ".env").items():
        if v is not None:
            os.environ.setdefault(k, v)
    os.environ.update(HF_HOME=HF_HOME, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false")


def run_local(lines: list[str], n: int, threads: int | None):
    env_setup()
    threads = threads or max(1, MAX_CPUS // n)
    assert n * threads <= MAX_CPUS, (n, threads)
    LOGS.mkdir(parents=True, exist_ok=True)
    todo = [ln for ln in lines if not is_done(ln)]
    print(f"local: {len(todo)} of {len(lines)} lines to run, {n} at a time x {threads} threads", flush=True)

    def one(ln):
        log = LOGS / f"local_{key(ln)}.log"
        t0 = time.time()
        rc = exec_line(ln, threads, log)
        print(f"[{'ok' if rc == 0 else f'FAIL rc={rc}'}] {time.time() - t0:6.0f}s  {ln}   (log {log})", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=n) as ex:
        rcs = list(ex.map(one, todo))
    print(f"local: {sum(r == 0 for r in rcs)} ok, {sum(r != 0 for r in rcs)} failed", flush=True)


SBATCH = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --exclude=s-005
#SBATCH --array=0-{last}{throttle}
#SBATCH --output={logs}/%x_%A_%a.out

umask 002
cd {root}
set -a; source .env; set +a
export HF_HOME={hf_home} HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} PATH=$PWD/.venv/bin:$PATH
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID"
python -m experiments.varwidth.queue {listfile} --exec-index $SLURM_ARRAY_TASK_ID --threads {cpus}
"""


def submit_slurm(lines: list[str], src: Path, a) -> None:
    todo = lines if a.all else [ln for ln in lines if not is_done(ln)]
    if not todo:
        print("slurm: nothing to submit (all done)")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"vw_{src.stem}"
    listfile = LOGS / f"{name}_{stamp}.lines"
    listfile.write_text("\n".join(todo) + "\n")
    script = LOGS / f"{name}_{stamp}.sbatch"
    script.write_text(SBATCH.format(name=name, cpus=a.cpus, mem=a.mem, time=a.time, last=len(todo) - 1,
                                    throttle=f"%{a.max_parallel}" if a.max_parallel else "", logs=LOGS, root=ROOT,
                                    hf_home=HF_HOME, listfile=listfile))
    print(f"slurm: {len(todo)} of {len(lines)} lines -> {script}")
    if a.dry_run:
        print(script.read_text())
        return
    out = subprocess.run(["sbatch", str(script)], capture_output=True, text=True, cwd=ROOT)
    print(out.stdout.strip() or out.stderr.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("lines", type=Path, help="text file: one line of run.py arguments per run")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true")
    g.add_argument("--local", type=int, metavar="N")
    g.add_argument("--slurm", action="store_true")
    g.add_argument("--exec-index", type=int, help=argparse.SUPPRESS)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--all", action="store_true", help="--slurm: submit done lines too (run.py still skips them)")
    p.add_argument("--max-parallel", type=int, default=0, help="--slurm: array throttle (%%N)")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="24G")
    p.add_argument("--time", default="00:40:00")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    lines = read_lines(a.lines)

    if a.exec_index is not None:
        sys.exit(exec_line(lines[a.exec_index], a.threads or 8))
    if a.status:
        counts = {}
        for i, ln in enumerate(lines):
            s = status(ln)
            counts[s.split("(")[0]] = counts.get(s.split("(")[0], 0) + 1
            print(f"{i:4d}  {s:14s}  {ln}")
        print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f"  (total {len(lines)})")
    elif a.local:
        run_local(lines, a.local, a.threads)
    elif a.slurm:
        submit_slurm(lines, a.lines, a)


if __name__ == "__main__":
    main()
