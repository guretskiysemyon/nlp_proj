#!/usr/bin/env python
"""X9: does extrapolation reduce the leak? Retention per language at the method's picked delta (CPU).

From the G1 leak curves (test split, descriptive): for each concept, retention of QA_test in en/de/es/fr/ru at the
picked delta (EMBER: Part 1's pick; CINDER: judge_rerun/Cinder/pick), for the plain edit and its extrapolated
variant (+X: same English edit plus native foreign tokens). Also SimdomQA_test accuracy (side effect) per language.
Leak = retention_L - retention_en (lower retention = more erased).

Output: Part 2 Experiments/x9_extrapolation_effect/{table.csv, report.md}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import pandas as pd  # noqa: E402
from x6_strength_curves import picks  # noqa: E402

C = p2x.C


def main():
    out = p2x.outdir("x9_extrapolation_effect")
    pk = picks()
    rows = []
    for c in p2x.CONCEPTS:
        for m in p2x.ALL_METHODS:
            base = m.split("+")[0]
            f = p2x.OUT / "g1_leak_curves" / m / C.safe(c) / "curve.csv"
            if (base, c) not in pk or not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[d.delta == pk[(base, c)]]
            if d.empty:
                continue
            r = {"concept": c, "method": m, "delta": pk[(base, c)]}
            ret = dict(zip(d.language, d.retention))
            for L in p2x.LANGS:
                r[f"ret_{L}"] = None if pd.isna(ret.get(L)) else round(float(ret[L]), 2)
            others = [r[f"ret_{L}"] for L in p2x.LANGS[1:] if r[f"ret_{L}"] is not None]
            r["mean_leak"] = round(sum(others) / len(others) - r["ret_en"], 2) if others and r["ret_en"] is not None else None
            sim = dict(zip(d.language, d.SimdomQA_test_acc))
            r["simdom_mean"] = round(float(pd.Series(sim).mean()), 2)
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "table.csv", index=False)
    L_ = ["# X9: retention per language at the picked δ, with and without extrapolation (test split, descriptive)", "",
          "ret_L = retention of QA_test in language L (1 = nothing erased, 0 = fully erased). mean_leak = mean over "
          "de/es/fr/ru minus en. simdom_mean = SimdomQA_test accuracy averaged over the 5 languages (side effect).", ""]
    L_ += [p2x.md_table(T) if len(T) else "No leak curves at the picked δ yet.", ""]
    (out / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
