#!/usr/bin/env python3
"""Stage D of the cross-lingual repair: summaries against the fixed pristine baseline.

Writes ``<method>_fixed/summary.csv`` in the same format as run_edited.write_summary
(retention = clip((acc - 0.25) / (pristine - 0.25), 0, 1) on QA_test, leak =
retention_L - retention_en, ``reliable`` from the pristine cell), but with the
pristine values from ``baseline_fixed/``, and prints the headline numbers next to the
original run's (``<method>/summary.csv``): English accuracy and retention, leak per
language with a 95% bootstrap over concepts (10,000 resamples, default_rng(0)),
SimDom, English MMLU, invalid rates.

Usage:
    python -m crosslingual.summary_fixed --method ember
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crosslingual.data import SETS
from crosslingual.run_edited import _retention
from crosslingual.reparse_saved import ROOT

OTHER = ("de", "es", "fr", "ru")


def write_summary(method_dir: Path, base_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for edit_path in sorted(method_dir.glob("*/edit.json")):
        edit = json.loads(edit_path.read_text(encoding="utf-8"))
        per_lang = {}
        for lang_path in sorted(edit_path.parent.glob("*.json")):
            if lang_path.name in ("edit.json", "mmlu_test.json"):
                continue
            cell = json.loads(lang_path.read_text(encoding="utf-8"))
            base = json.loads((base_dir / edit_path.parent.name / lang_path.name).read_text(encoding="utf-8"))
            row = {"concept": edit["concept"], "language": cell["language"], "reliable": base["reliable"]}
            for s in SETS:
                row[f"{s}_acc"] = cell["results"][s]["accuracy"]
                row[f"{s}_invalid"] = cell["results"][s]["invalid_rate"]
                row[f"{s}_pristine"] = base["results"][s]["accuracy"]
            row["retention"] = _retention(row["QA_test_acc"], row["QA_test_pristine"])
            row["mmlu_test_acc_en"] = edit["mmlu_test"]["acc"]
            row["mmlu_test_invalid_en"] = edit["mmlu_test"].get("invalid")
            row["hp"] = json.dumps(edit["hp"], sort_keys=True)
            row["mode"] = edit.get("mode", "reparse_only")
            per_lang[cell["language"]] = row
        r_en = per_lang.get("en", {}).get("retention")
        for row in per_lang.values():
            row["leak"] = None if r_en is None or row["retention"] is None else row["retention"] - r_en
            rows.append(row)
    cols = ["concept", "language", "reliable", "QA_test_acc", "QA_test_pristine", "retention", "leak",
            "QA_test_invalid", "SimdomQA_test_acc", "SimdomQA_test_pristine", "SimdomQA_test_invalid",
            "mmlu_test_acc_en", "mmlu_test_invalid_en", "hp", "mode"]
    df = pd.DataFrame(rows)[cols].sort_values(["concept", "language"])
    df.to_csv(method_dir / "summary.csv", index=False)
    return df


def _boot(piv: pd.DataFrame, cols, rng) -> tuple:
    v = piv[list(cols)].values
    idx = rng.integers(0, len(v), (10000, len(v)))
    bs = np.array([np.nanmean(np.nanmean(v[i], axis=0)) for i in idx])
    return float(np.nanmean(np.nanmean(v, axis=0))), *np.percentile(bs, [2.5, 97.5])


def headline(df: pd.DataFrame) -> Dict[str, str]:
    r = df[df.reliable]
    en = df[df.language == "en"]
    piv = r.pivot(index="concept", columns="language", values="leak")
    out = {"English QA_test acc": f"{en.QA_test_acc.mean():.2f}",
           "retention r_en": f"{en.retention.mean():.2f}",
           "retention de/es/fr/ru": " / ".join(f"{r[r.language == l].retention.mean():.2f}" for l in OTHER)}
    for l in OTHER:
        m, lo, hi = _boot(piv, [l], np.random.default_rng(0))
        out[f"leak {l}"] = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    m, lo, hi = _boot(piv, OTHER, np.random.default_rng(0))
    out["mean leak de/es/fr/ru"] = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    pc = r[r.language != "en"].groupby("concept").leak.mean()
    out["concepts with mean leak > 0"] = f"{int((pc > 0).sum())} / {len(pc)}"
    out["SimdomQA_test acc en / mean other"] = f"{en.SimdomQA_test_acc.mean():.2f} / {df[df.language != 'en'].SimdomQA_test_acc.mean():.2f}"
    out["English MMLU test"] = f"{en.mmlu_test_acc_en.mean():.3f}"
    out["QA_test invalid en / de / es / fr / ru"] = " / ".join(
        f"{df[df.language == l].QA_test_invalid.mean():.3f}" for l in ("en",) + OTHER)
    out["SimdomQA_test invalid (all)"] = f"{df.SimdomQA_test_invalid.mean():.3f}"
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("ember", "snmf", "snmf_ember"))
    args = p.parse_args()
    new = write_summary(ROOT / f"{args.method}_fixed", ROOT / "baseline_fixed")
    old = pd.read_csv(ROOT / args.method / "summary.csv")
    h_old, h_new = headline(old), headline(new)
    print(f"| {args.method} | original run | fixed |\n|---|---|---|")
    for k in h_new:
        print(f"| {k} | {h_old.get(k, '')} | {h_new[k]} |")
    print(f"\nmodes: {new.groupby('mode').concept.nunique().to_dict()}; "
          f"summary -> {ROOT / f'{args.method}_fixed' / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
