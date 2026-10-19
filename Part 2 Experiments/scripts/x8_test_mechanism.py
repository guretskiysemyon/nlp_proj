#!/usr/bin/env python
"""X8: held-out questions in 5 languages at the picked delta: touched vs untouched questions, both methods (CPU).

Uses the final-test records (EMBER: results/crosslingual/.../ember_fixed; CINDER: .../cinder, written by
cinder.final_test) and the pristine model's records on the same questions (baseline_fixed). A question is "touched"
if its question or correct answer (in that language) contains a token the method edits. Per method, language and
group: pristine accuracy, edited accuracy, and the share of pristine-correct questions the edit turns wrong.
Explains the leak per question: does a foreign question keep its answer exactly when the edit does not reach it?

Output: Part 2 Experiments/x8_test_mechanism/{split.csv, report.md}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import pandas as pd  # noqa: E402

XL = p2x.ROOT / "results" / "crosslingual" / "google_gemma-2-2b-it"
DIRS = {"EMBER": XL / "ember_fixed", "CINDER": XL / "cinder"}


def main():
    out = p2x.outdir("x8_test_mechanism")
    rows = []
    for c in p2x.CONCEPTS:
        for m, base_dir in DIRS.items():
            tids, _ = p2x.edit(c, m)
            E = set(tids)
            if not E:
                continue
            for L in p2x.LANGS:
                p = base_dir / p2x.C.safe(c) / f"{L}.json"
                b = XL / "baseline_fixed" / p2x.C.safe(c) / f"{L}.json"
                if not p.exists() or not b.exists():
                    continue
                ed = {r["index"]: r for r in json.loads(p.read_text())["results"]["QA_test"]["records"]}
                pr = {r["index"]: r for r in json.loads(b.read_text())["results"]["QA_test"]["records"]}
                for grp in ("touched", "untouched"):
                    sel = [i for i, r in ed.items()
                           if bool((p2x.text_ids(r["question"]) | p2x.text_ids(r["correct_text"])) & E)
                           == (grp == "touched")]
                    if not sel:
                        rows.append({"concept": c, "method": m, "language": L, "group": grp, "n": 0})
                        continue
                    pc = [i for i in sel if pr.get(i, {}).get("is_correct")]
                    rows.append({"concept": c, "method": m, "language": L, "group": grp, "n": len(sel),
                                 "pristine_acc": round(sum(bool(pr.get(i, {}).get("is_correct")) for i in sel) / len(sel), 2),
                                 "edited_acc": round(sum(bool(ed[i]["is_correct"]) for i in sel) / len(sel), 2),
                                 "flipped_to_wrong_%": round(100 * sum(not ed[i]["is_correct"] for i in pc) / len(pc), 1)
                                 if pc else None})
    D = pd.DataFrame(rows)
    D.to_csv(out / "split.csv", index=False)
    L_ = ["# X8: held-out questions at the picked δ, touched vs untouched by the edit", "",
          "`flipped_to_wrong_%`: share of the questions the pristine model answered correctly that the edited model "
          "gets wrong (the erased knowledge).", ""]
    if len(D):
        agg = (D.dropna(subset=["flipped_to_wrong_%"]).groupby(["method", "language", "group"])
               .apply(lambda g: pd.Series({"questions": int(g.n.sum()),
                                           "flipped_%": round(float((g["flipped_to_wrong_%"] * g.n).sum() / g.n.sum()), 1)}))
               .reset_index())
        L_ += ["## All concepts pooled (weighted by questions)", "", p2x.md_table(agg), "", "## Per concept", "",
               p2x.md_table(D), ""]
    else:
        L_.append("No final-test records yet.")
    (out / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_[:30]))


if __name__ == "__main__":
    main()
