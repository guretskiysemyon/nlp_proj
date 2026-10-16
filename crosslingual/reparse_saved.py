#!/usr/bin/env python3
"""Re-score saved cross-lingual cells with the fixed letter parser (CPU, no model).

The cross-lingual runs (run_baseline.py / run_edited.py) scored each answer with
ember.evals.mc.parse_letter on a 4-token generation, and read the answer through
WrappedHFModel._extract_response, which cuts at the first "model\\n" in the decoded
text. Three problems, see docs/Part 1/ember_search.md "MC scoring repair":

  - lowercase letters ("c") counted as invalid;
  - an option ending in "model" made the "answer" the rest of the prompt
    ('B. Bellman equation ... Answer only with one letter ...\\nmodel\\nD' read as B;
    the model's real answer is the text after the last "model\\n", here D);
  - "The answer is **C**" cut off after 4 tokens (needs regeneration, not done here).

This script fixes the first two from the saved records, without touching the
originals: every record of every cell is re-parsed with
judge_rerun.mc_fixed.parse_letter_fixed (applied to the text after the last
"model\\n" when the saved answer contains the prompt), accuracy / invalid_rate are
recomputed, and the cells go to <method>_fixed/ in the same format. Each changed
record keeps its old values (old_model_letter, old_is_correct). A record that is
still invalid afterwards is listed in the cell's ``still_invalid`` field: if the
model ran out of tokens it has to be regenerated (an edited model), otherwise it
is a real non-answer.

Usage:
    python -m crosslingual.reparse_saved --method baseline
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosslingual.data import REPO_ROOT, SETS
from crosslingual.run_baseline import MIN_RELIABLE_QA_ACC
from judge_rerun.mc_fixed import parse_letter_fixed

ROOT = REPO_ROOT / "results" / "crosslingual" / "google_gemma-2-2b-it"


def answer_text(raw: str) -> str:
    """The model's own answer: the text after the last 'model\\n' if the prompt leaked in."""
    if "Answer only with one letter" in raw and "model\n" in raw:
        return raw.rsplit("model\n", 1)[1].strip()
    return raw.strip()


def reparse_cell(cell: Dict[str, Any]) -> Dict[str, int]:
    stats = {"changed": 0, "now_valid": 0, "prompt_echo": 0}
    still = []
    for s in SETS:
        res = cell["results"][s]
        for r in res["records"]:
            text = answer_text(r["model_raw"])
            if text != r["model_raw"].strip():
                stats["prompt_echo"] += 1
            letter = parse_letter_fixed(text)
            valid = letter in ("A", "B", "C", "D")
            correct = valid and letter == r["correct_letter"]
            if letter != r["model_letter"]:
                stats["changed"] += 1
                stats["now_valid"] += int(valid and not r["is_valid"])
                r["old_model_letter"], r["old_is_correct"] = r["model_letter"], r["is_correct"]
            r["model_answer_text"] = text
            r["model_letter"], r["is_valid"], r["is_correct"] = letter, valid, correct
            if not valid:
                still.append({"set": s, "index": r["index"], "model_raw": r["model_raw"]})
        n = len(res["records"])
        res["accuracy"] = sum(r["is_correct"] for r in res["records"]) / n
        res["invalid_rate"] = sum(not r["is_valid"] for r in res["records"]) / n
    cell["scoring"] = "reparse_saved (parse_letter_fixed, prompt-echo fix); generation unchanged"
    cell["still_invalid"] = still
    return stats


def write_summary(root: Path, is_baseline: bool) -> Path:
    import pandas as pd
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.glob("*/*.json")):
        if p.name in ("edit.json",):
            continue
        cell = json.loads(p.read_text(encoding="utf-8"))
        row = {"concept": cell["concept"], "language": cell["language"]}
        if is_baseline:
            row.update({"reliable": cell["reliable"], "limit": cell.get("limit")})
        for s in SETS:
            r = cell["results"][s]
            row.update({f"{s}_acc": r["accuracy"], f"{s}_invalid": r["invalid_rate"], f"{s}_n": r["n"]})
        row["still_invalid"] = len(cell["still_invalid"])
        rows.append(row)
    path = root / "summary.csv"
    pd.DataFrame(rows).sort_values(["concept", "language"]).to_csv(path, index=False)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("baseline", "ember", "snmf", "snmf_ember"))
    args = p.parse_args()
    src, dst = ROOT / args.method, ROOT / f"{args.method}_fixed"
    is_baseline = args.method == "baseline"
    total = {"changed": 0, "now_valid": 0, "prompt_echo": 0, "still_invalid": 0, "reliable_flips": 0}
    for path in sorted(src.glob("*/*.json")):
        out = dst / path.parent.name / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "edit.json":
            shutil.copy2(path, out)
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        st = reparse_cell(cell)
        for k in ("changed", "now_valid", "prompt_echo"):
            total[k] += st[k]
        total["still_invalid"] += len(cell["still_invalid"])
        if is_baseline:
            reliable = cell["results"]["QA_test"]["accuracy"] >= MIN_RELIABLE_QA_ACC
            total["reliable_flips"] += int(reliable != cell["reliable"])
            cell["reliable"] = reliable
        out.write_text(json.dumps(cell, ensure_ascii=False, indent=1), encoding="utf-8")
        if st["changed"]:
            print(f"  {cell['concept']}/{cell['language']}: {st['changed']} answers re-read "
                  f"({st['now_valid']} were invalid, {st['prompt_echo']} prompt echo)")
    for extra in src.glob("_*.json"):          # baseline/_mmlu.json
        shutil.copy2(extra, dst / extra.name)
    print(f"[reparse] {args.method}: {total} -> {write_summary(dst, is_baseline)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
