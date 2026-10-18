"""Shared Gemini spend log and hard budget stop (judges, word lists and erasure evals alike).

Every call's usage_metadata (prompt, output and thought tokens) is appended to GEMINI_USAGE_LOG
(default results/gemini_usage.jsonl). Before a call, check() raises BudgetExceeded once the logged total reaches
GEMINI_BUDGET_NIS (default 25). Prices are deliberately conservative (env GEMINI_PRICE_IN_USD / GEMINI_PRICE_OUT_USD
per 1M tokens, default 0.30 / 2.50; thought tokens billed as output; GEMINI_USD_NIS default 3.7).
The total is re-read from the log every REREAD calls per process, so concurrent processes overshoot by at most a few
calls each.

    python -m ember.gemini_budget            # spend so far
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = Path(os.environ.get("GEMINI_USAGE_LOG", ROOT / "results" / "gemini_usage.jsonl"))
REREAD = 25


class BudgetExceeded(RuntimeError):
    pass


def _f(name, default):
    return float(os.environ.get(name, default))


def budget_nis() -> float:
    return _f("GEMINI_BUDGET_NIS", 25.0)


def cost_nis(prompt: int, output: int, thoughts: int) -> float:
    usd = prompt / 1e6 * _f("GEMINI_PRICE_IN_USD", 0.30) + (output + thoughts) / 1e6 * _f("GEMINI_PRICE_OUT_USD", 2.50)
    return usd * _f("GEMINI_USD_NIS", 3.7)


_lock = threading.Lock()
_state = {"total": None, "since": 0}


def logged_total() -> float:
    if not LOG.exists():
        return 0.0
    tot = 0.0
    with LOG.open() as fh:
        for line in fh:
            try:
                tot += json.loads(line)["nis"]
            except Exception:  # noqa: BLE001
                pass
    return tot


def check() -> None:
    with _lock:
        if _state["total"] is None or _state["since"] >= REREAD:
            _state["total"], _state["since"] = logged_total(), 0
        if _state["total"] >= budget_nis():
            raise BudgetExceeded(f"Gemini budget reached: {_state['total']:.2f} >= {budget_nis():.2f} NIS")


def record(resp, tag: str = "") -> None:
    um = getattr(resp, "usage_metadata", None)
    p = int(getattr(um, "prompt_token_count", 0) or 0)
    o = int(getattr(um, "candidates_token_count", 0) or 0)
    t = int(getattr(um, "thoughts_token_count", 0) or 0)
    nis = cost_nis(p, o, t)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "job": os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}"),
           "tag": tag or os.environ.get("GEMINI_TAG", ""), "p": p, "o": o, "t": t, "nis": round(nis, 7)}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOG), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
    try:
        os.write(fd, (json.dumps(rec) + "\n").encode())
    finally:
        os.close(fd)
    with _lock:
        _state["total"] = (_state["total"] or 0.0) + nis
        _state["since"] += 1


def summary() -> dict:
    rows = [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()] if LOG.exists() else []
    by = {}
    for r in rows:
        d = by.setdefault(r.get("tag", ""), {"calls": 0, "p": 0, "o": 0, "t": 0, "nis": 0.0})
        d["calls"] += 1
        for k in ("p", "o", "t", "nis"):
            d[k] += r[k]
    return {"total_nis": sum(d["nis"] for d in by.values()), "calls": len(rows), "budget_nis": budget_nis(), "by_tag": by}


if __name__ == "__main__":
    print(json.dumps(summary(), indent=1))
