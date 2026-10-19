#!/usr/bin/env python
"""CINDER vs EMBER on the 6 concepts (CPU): features, chosen delta, English erasure, leak per language.

Reads EMBER from Part 1 (picks: judge_rerun/Ember/pass2_rescored, final test:
results/crosslingual/google_gemma-2-2b-it/ember_fixed/summary.csv) and CINDER from judge_rerun/Cinder/ and
results/crosslingual/google_gemma-2-2b-it/cinder/summary.csv (written by cinder.final_test). A concept where the
two-way judge accepted no feature has no CINDER edit: the model is the pristine one, so its retention is 1 in
every language and its leak 0; it is listed as "no edit" and counted that way only in the "all 6" means.

Output: judge_rerun/Cinder/comparison.md (and printed).

    python -m cinder.compare
"""
from __future__ import annotations

import json

import pandas as pd

from cinder import common as C
from cinder.final_test import OUT_DIR, XL_ROOT

LANGS = ("en", "de", "es", "fr", "ru")


def ember_rows() -> dict:
    picks = pd.read_csv(C.EMBER_DIR / "pass2_rescored" / C.RUN_REL / "best_deltas_summary.csv").set_index("concept")
    grid = C.EMBER_DIR / "rescore_mc" / C.RUN_REL
    out = {}
    for c in C.CONCEPTS:
        g = pd.read_csv(grid / C.safe(c) / "grid_rescored.csv").iloc[0]
        out[c] = {"k": int(g.k_features_embed), "tokens": int(g.n_tokens_edited),
                  "delta": float(picks.loc[c, "best_delta_embed"]), "h_alpaca": float(picks.loc[c, "best_harmonic_alpaca"])}
    return out


def cinder_rows() -> dict:
    out = {}
    for c in C.CONCEPTS:
        s = json.loads((C.FEATURES_OUT / C.safe(c) / "summary.json").read_text())
        r = {"k": len(s["F_TWJ"]), "tokens": s["n_tokens_edited"], "F_TJ": s["F_TJ"], "F_LO5": s["F_LO5"]}
        best = C.PICK_OUT / C.safe(c) / f"best_delta__{C.JUDGE_ID}.json"
        if best.exists():
            b = json.loads(best.read_text())
            r.update(delta=b["best_delta_embed"], h_alpaca=b["best_harmonic_alpaca"], at_edge=b["at_grid_edge"])
        out[c] = r
    return out


def retention_table(path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df.concept.isin(C.CONCEPTS)]
    return df.pivot(index="concept", columns="language", values="retention")


def main():
    E, Ci = ember_rows(), cinder_rows()
    lines = ["# CINDER vs EMBER (English fit, 6 concepts)", "",
             "## Features and chosen strength", "",
             "| concept | EMBER features / tokens | EMBER δ | EMBER h_alpaca | CINDER features (TJ, LO@5) / tokens | CINDER δ | CINDER h_alpaca |",
             "|---|---|---|---|---|---|---|"]
    for c in C.CONCEPTS:
        e, k = E[c], Ci[c]
        cf = f"{k['k']} (TJ {k['F_TJ'] or '-'}, LO@5 {k["F_LO5"] or "-"}) / {k['tokens']}" if k["k"] else "0: no edit"
        cd = f"{k['delta']:g}" + (" (edge)" if k.get("at_edge") else "") if "delta" in k else ("-" if not k["k"] else "pending")
        ch = f"{k['h_alpaca']:.3f}" if "h_alpaca" in k else "-"
        lines.append(f"| {c} | {e['k']} / {e['tokens']} | {e['delta']:g} | {e['h_alpaca']:.3f} | {cf} | {cd} | {ch} |")

    re_ = retention_table(XL_ROOT / "ember_fixed" / "summary.csv")
    if any(OUT_DIR.glob("*/edit.json")):
        from crosslingual.summary_fixed import write_summary
        write_summary(OUT_DIR, XL_ROOT / "baseline_fixed")
    rc = retention_table(OUT_DIR / "summary.csv")
    lines += ["", "## Retention of concept knowledge on QA_test (lower = more erased; leak = r_L - r_en)", ""]
    if rc is None:
        lines.append("CINDER final test not run yet (python -m cinder.final_test).")
    else:
        for c in C.CONCEPTS:
            if Ci[c]["k"] == 0 and c not in rc.index:
                rc.loc[c] = 1.0                       # no edit: the pristine model
        langs = [l for l in LANGS if l in rc.columns]
        other = [l for l in langs if l != "en"]
        lines += ["| concept | method | " + " | ".join(langs) + " | mean leak " + ("/".join(other) or "-") + " |",
                  "|---|---|" + "---|" * (len(langs) + 1)]
        for c in C.CONCEPTS:
            for name, t in (("EMBER", re_), ("CINDER", rc)):
                if t is None or c not in t.index:
                    lines.append(f"| {c} | {name} | " + " | ".join("-" for _ in langs) + " | - |")
                    continue
                r = t.loc[c]
                leak = f"{sum(r[l] - r['en'] for l in other) / len(other):+.2f}" if other else "-"
                note = " (no edit)" if name == "CINDER" and Ci[c]["k"] == 0 else ""
                lines.append(f"| {c} | {name}{note} | " + " | ".join(f"{r[l]:.2f}" for l in langs) + f" | {leak} |")
        for label, cs in (("concepts CINDER edits", [c for c in C.CONCEPTS if Ci[c]["k"]]), ("all 6", C.CONCEPTS)):
            for name, t in (("EMBER", re_), ("CINDER", rc)):
                sub = t.loc[[c for c in cs if c in t.index]]
                leak = f"{sum((sub[l] - sub['en']).mean() for l in other) / len(other):+.2f}" if other else "-"
                lines.append(f"| **mean, {label}** | {name} | " + " | ".join(f"{sub[l].mean():.2f}" for l in langs)
                             + f" | {leak} |")
    text = "\n".join(lines) + "\n"
    (C.OUT / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
