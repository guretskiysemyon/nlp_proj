"""Output paths, result schema, and resumability.

Layout::

    results/cross_lang_test/
      <language>/
        <concept_safe>/
          <config>__<wrapper>.json     # wrapper is "native" for language == en
      summary.csv                      # one row per (concept, language, config,
                                       # wrapper, qa_kind); append-only, resumable

Each per-cell JSON holds both ``qa_kind`` blocks, written incrementally: a
crash partway through a cell must not lose the qa_kind that already finished,
and a re-run must skip it. That is why "is this cell done?" is asked per
(cell file, qa_kind) rather than per file -- the file name has no qa_kind axis,
but the work does.

``summary.csv`` is the flat rollup for quick pandas loading; the per-question
records stay in the JSON so a teammate can spot-check specific translated
failures later.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ember.erasure import io as ember_io
from ember.utils import _safe_concept

DEFAULT_OUT_DIR = ember_io.ROOT_DIR / "results" / "cross_lang_test"

SUMMARY_COLUMNS: List[str] = [
    "concept", "language", "config", "wrapper", "wrapper_lang", "qa_kind",
    "model", "rank", "seed", "train_eval", "reliable",
    "qa_acc", "qa_base", "qa_frac",
    "simdom_acc", "simdom_base", "simdom_frac",
    "qa_invalid", "simdom_invalid",
    "mmlu_acc_en", "mmlu_frac_en",
    "efficacy", "specificity", "harmonic",
    "n_qa", "n_simdom", "timestamp",
]

#: Identifies one summary row; used to drop stale rows when ``--force`` re-runs.
SUMMARY_KEY: List[str] = ["concept", "language", "config", "wrapper", "qa_kind"]


# ========================================================================== #
# Paths                                                                       #
# ========================================================================== #

def cell_json_path(out_dir: Path, language: str, concept: str, config: str,
                   wrapper: str) -> Path:
    """``<out_dir>/<language>/<concept_safe>/<config>__<wrapper>.json``."""
    return (Path(out_dir) / language / _safe_concept(concept)
            / f"{config}__{wrapper}.json")


def summary_csv_path(out_dir: Path) -> Path:
    return Path(out_dir) / "summary.csv"


# ========================================================================== #
# Per-cell JSON                                                               #
# ========================================================================== #

def read_cell(path: Path) -> Optional[Dict[str, Any]]:
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[cl-io] WARN: unreadable result {path}: {type(e).__name__}: {e}")
        return None


def cell_has_qa_kind(path: Path, qa_kind: str) -> bool:
    """True when this cell's JSON already holds a completed ``qa_kind`` block."""
    cell = read_cell(path)
    if not cell:
        return False
    block = cell.get(qa_kind)
    return isinstance(block, dict) and "metrics" in block


def write_cell_qa_kind(path: Path, *, identity: Dict[str, Any], qa_kind: str,
                       block: Dict[str, Any]) -> Dict[str, Any]:
    """Merge one qa_kind block into the cell JSON and write it atomically."""
    cell = read_cell(path) or {}
    cell.update(identity)
    cell[qa_kind] = block
    cell["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ember_io.save_json_atomic(Path(path), cell)
    return cell


def build_subset_block(*, accuracy: float, base: float, frac: float, n: int,
                       records: List[Dict[str, Any]],
                       invalid_rate: Optional[float] = None) -> Dict[str, Any]:
    """One ``qa``/``simdom`` entry: accuracy, its denominator, frac, and records."""
    block: Dict[str, Any] = {
        "accuracy": accuracy,
        "base_accuracy": base,
        "frac": frac,
        "n": n,
        "records": records,
    }
    if invalid_rate is not None:
        block["invalid_rate"] = invalid_rate
    return block


# ========================================================================== #
# summary.csv                                                                 #
# ========================================================================== #

def drop_summary_rows(out_dir: Path, key: Dict[str, Any]) -> None:
    """Remove rows matching ``key`` (used by ``--force`` so re-runs don't duplicate)."""
    path = summary_csv_path(out_dir)
    if not path.exists():
        return
    df = ember_io.read_csv_safe(path)
    if df.empty or any(c not in df.columns for c in SUMMARY_KEY):
        return
    mask = pd.Series(True, index=df.index)
    for col in SUMMARY_KEY:
        mask &= df[col].astype(str) == str(key[col])
    if not mask.any():
        return
    df[~mask].to_csv(path, index=False)


def append_summary_row(out_dir: Path, row: Dict[str, Any]) -> None:
    """Append one completed cell to ``summary.csv`` (created with a header on first write)."""
    row = {**row}
    row.setdefault("timestamp",
                   datetime.now(timezone.utc).isoformat(timespec="seconds"))
    ember_io.append_csv_row(summary_csv_path(out_dir), row, columns=SUMMARY_COLUMNS)


def load_summary(out_dir: Path) -> pd.DataFrame:
    return ember_io.read_csv_safe(summary_csv_path(out_dir))


# ========================================================================== #
# Headline metric                                                             #
# ========================================================================== #

def compute_delta_l(summary: pd.DataFrame) -> pd.DataFrame:
    """Add ``delta_l = qa_frac(language) - qa_frac(en)`` per (config, wrapper, qa_kind).

    Δ_L compares a config against **its own** English retention, not against
    pristine -- ``ember``'s French qa_frac versus ``ember``'s English qa_frac.
    English rows are produced by the same uniform code path as every other
    language, so ``r_en`` is just another row here.

    English rows are matched on (concept, config, qa_kind) only: ``en`` has a
    single wrapper condition, which is the comparison point for both the
    English-wrapper and native-wrapper arms of a non-English language.
    """
    if summary.empty or "qa_frac" not in summary.columns:
        return summary
    en = (summary[summary["language"].astype(str) == "en"]
          [["concept", "config", "qa_kind", "qa_frac"]]
          .rename(columns={"qa_frac": "qa_frac_en"})
          .drop_duplicates(subset=["concept", "config", "qa_kind"], keep="last"))
    out = summary.merge(en, on=["concept", "config", "qa_kind"], how="left")
    out["delta_l"] = out["qa_frac"] - out["qa_frac_en"]
    return out


__all__ = [
    "DEFAULT_OUT_DIR", "SUMMARY_COLUMNS", "SUMMARY_KEY",
    "cell_json_path", "summary_csv_path",
    "read_cell", "cell_has_qa_kind", "write_cell_qa_kind", "build_subset_block",
    "drop_summary_rows", "append_summary_row", "load_summary", "compute_delta_l",
]
