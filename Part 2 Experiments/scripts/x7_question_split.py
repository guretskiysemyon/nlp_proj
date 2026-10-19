#!/usr/bin/env python
"""X7: is erasure limited to questions that contain an edited token? English train split, every delta (CPU).

Each QA_train question is "touched" if its question or correct answer contains a token the method edits, else
"untouched". Accuracy per group and delta, from the grids' per-question records (EMBER: judge_rerun/Ember/
rescore_mc; CINDER: judge_rerun/Cinder/grid). If accuracy falls only on touched questions, the edit removes words,
not the knowledge behind them.

Output: Part 2 Experiments/x7_question_split/{split.csv, report.md}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import pandas as pd  # noqa: E402

C = p2x.C


def rec_dir(concept, method):
    return (C.EMBER_DIR / "rescore_mc" / C.RUN_REL / C.safe(concept) / "records" if method == "EMBER"
            else C.GRID_OUT / C.safe(concept) / "records")


def main():
    out = p2x.outdir("x7_question_split")
    rows = []
    for c in p2x.CONCEPTS:
        for m in p2x.METHODS:
            tids, _ = p2x.edit(c, m)
            E = set(tids)
            if not E:
                continue
            for p in sorted(rec_dir(c, m).glob("delta_*.json"), key=lambda p: float(p.stem[6:])):
                r = json.loads(p.read_text())
                recs = r["qa"]["records"]
                touched = [bool((p2x.text_ids(x["question"]) | p2x.text_ids(x["correct_text"])) & E) for x in recs]
                for grp, flag in (("touched", True), ("untouched", False)):
                    g = [x for x, t in zip(recs, touched) if t == flag]
                    rows.append({"concept": c, "method": m, "delta": float(r["delta_embed"]), "group": grp,
                                 "n": len(g), "acc": round(sum(x["is_correct"] for x in g) / len(g), 3) if g else None})
    D = pd.DataFrame(rows)
    D.to_csv(out / "split.csv", index=False)
    piv = D.pivot_table(index=["concept", "method", "group", "n"], columns="delta", values="acc").reset_index()
    piv.columns = [str(x) for x in piv.columns]
    L = ["# X7: QA_train accuracy on questions that contain an edited token (touched) vs not (untouched)", "",
         "Columns: δ. Lower = more erased. A drop only in `touched` means the edit removes the words, not the "
         "knowledge.", "", p2x.md_table(piv), ""]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
