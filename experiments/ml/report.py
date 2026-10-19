#!/usr/bin/env python
"""Result tables for the multilingual comparison, from results/ml/runs/*/*/{eval,rescore_v2,fit_summary}.json.

    python -m experiments.ml.report            # writes results/ml/REPORT_tables.md and prints it

Test scores follow the protocol (PLAN.md): a test row that fails a gate scores 0; the ungated mean H is listed too.
v2 = it/pt/hi/th re-scored on the cleaned question sets (results/ml/data/qa_v2), δ kept from the v1 selection.
"""
from __future__ import annotations

import json
from statistics import mean

from experiments.ml.common import LANGS, RESULTS, concepts, run_dir

APPROACHES = ("A1", "A2", "A3", "A3z")
NAMES = {"A1": "A1 per-language EMBER", "A2": "A2 multilingual EMBER", "A3": "A3 multilingual CINDER",
         "A3z": "A3z CINDER, original-space edit"}


def load(c, a):
    p = run_dir(c, a) / "eval.json"
    if not p.exists():
        return None
    e = json.loads(p.read_text())
    if not e.get("done"):
        return {"running": True}
    out = {"edit": bool(e.get("n_edited")), "n": e.get("n_edited", 0), "delta": e.get("best_delta", 0.0)}
    t = e.get("test", {})
    out["score"] = t.get("score", 0.0)
    if out["edit"]:
        per = t["per_lang"]
        el = [L for L in LANGS if per[L].get("eligible", True)]
        out["ungated"] = mean(per[L]["H"] for L in el) if el else 0.0
        out["gated"] = t.get("gated", False)
        out["eff"] = {L: per[L].get("efficacy") for L in LANGS}
        out["mmlu"] = min(per[L]["mmlu_frac"] for L in LANGS)
        out["sim"] = mean(per[L].get("simdom_frac", 1.0) for L in LANGS)
    r = run_dir(c, a) / "rescore_v2.json"
    if r.exists() and out["edit"]:
        out["v2"] = json.loads(r.read_text())["score"]
    return out


def cell(r):
    if r is None:
        return "·"
    if r.get("running"):
        return "running"
    if not r["edit"]:
        return "—"
    s = f"{r['score']:.2f}"
    if r.get("gated"):
        s += f"* ({r['ungated']:.2f})"
    if "v2" in r:
        s += f" [{r['v2']:.2f}]"
    return s + f" δ{r['delta']:g}"


def main():
    R = {c: {a: load(c, a) for a in APPROACHES} for c in concepts()}
    L = ["# Multilingual erasure: A1 / A2 / A3 / A3z (Llama-3.1-8B-Instruct, NF4, 8 languages)", "",
         "Test score = mean over eligible languages of H_L = HM(efficacy, specificity, fluency); 0 when a gate fails",
         "(* = gated on test, ungated value in parentheses). [v2] = it/pt/hi/th on the cleaned question sets.",
         "— = no selected feature, no edit (score 0).", "",
         "| Concept | " + " | ".join(NAMES[a] for a in APPROACHES) + " |", "|---" * (len(APPROACHES) + 1) + "|"]
    for c in concepts():
        L.append(f"| {c} | " + " | ".join(cell(R[c][a]) for a in APPROACHES) + " |")
    done = lambda a: [c for c in concepts() if R[c][a] and not R[c][a].get("running")]
    L += ["", "## Summary", "", "| | " + " | ".join(APPROACHES) + " |", "|---" * (len(APPROACHES) + 1) + "|"]
    allfin = [a for a in APPROACHES if len(done(a)) == len(concepts())]
    rows = {
        "concepts finished": [str(len(done(a))) for a in APPROACHES],
        "concepts with an edit": [str(sum(R[c][a]["edit"] for c in done(a))) for a in APPROACHES],
        "mean test score, all 18 (no edit = 0)": [f"{mean(R[c][a]['score'] for c in done(a)):.3f}"
                                                  if a in allfin else "partial" for a in APPROACHES],
        "mean v2 score, all 18": [f"{mean(R[c][a].get('v2', 0.0) for c in done(a)):.3f}"
                                  if a in allfin and all('v2' in R[c][a] for c in done(a) if R[c][a]['edit'])
                                  else "partial" for a in APPROACHES],
    }
    for k, v in rows.items():
        L.append(f"| {k} | " + " | ".join(v) + " |")
    # head-to-head on concepts where every compared approach edits and is finished
    for group in (("A1", "A2", "A3"), ("A1", "A2", "A3", "A3z"), ("A2", "A3z"), ("A3", "A3z")):
        cs = [c for c in concepts() if all(R[c][a] and not R[c][a].get("running") and R[c][a]["edit"] for a in group)]
        if cs:
            L.append("")
            L.append(f"Shared concepts for {'/'.join(group)} (n={len(cs)}: {', '.join(cs)}): " + ", ".join(
                f"{a} {mean(R[c][a]['score'] for c in cs):.3f}" for a in group) +
                "; wins " + ", ".join(f"{a} {sum(max(group, key=lambda x: R[c][x]['score']) == a for c in cs)}"
                                       for a in group))
    # leak profile
    L += ["", "## Efficacy per language at the chosen δ (mean over concepts with an edit, test)", "",
          "| | " + " | ".join(LANGS) + " |", "|---" * (len(LANGS) + 1) + "|"]
    for a in APPROACHES:
        rs = [R[c][a] for c in done(a) if R[c][a]["edit"]]
        if rs:
            L.append(f"| {a} | " + " | ".join(f"{mean(r['eff'][x] or 0 for r in rs):.2f}" for x in LANGS) + " |")
    # cost
    L += ["", "## Selection cost per concept (mean)", "", "| | factorizations | rows factorized | ALS iterations | "
          "judged candidates |", "|---|---|---|---|---|"]
    for a in ("A1", "A2", "A3"):
        fs, rows_, its, js = [], [], [], []
        for c in concepts():
            if a == "A1":
                ps = [run_dir(c, "A1", x) / "fit_summary.json" for x in LANGS]
                if not all(p.exists() for p in ps):
                    continue
                d = [json.loads(p.read_text()) for p in ps]
                fs.append(8); rows_.append(sum(x["n"] for x in d)); its.append(sum(x["n_iter"] for x in d))
                js.append(sum(len(json.loads((p.with_name("judge.json")).read_text()))
                              for p in ps if p.with_name("judge.json").exists()))
            else:
                p = run_dir(c, a) / "fit_summary.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                fs.append(1); rows_.append(d["n_sample"]); its.append(d["n_iter"])
                j = p.with_name("judge.json")
                js.append(len(json.loads(j.read_text())) if j.exists() else 0)
        if fs:
            L.append(f"| {a} | {mean(fs):.0f} | {mean(rows_):,.0f} | {mean(its):,.0f} | {mean(js):.0f} |")
    L.append("| A3z | A3's fit + one k×k ridge solve | | | as A3 |")
    text = "\n".join(L) + "\n"
    (RESULTS / "REPORT_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
