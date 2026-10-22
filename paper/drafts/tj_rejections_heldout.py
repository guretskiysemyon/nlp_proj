"""Read-only recount behind App. D's held-out judge table and its per-concept list of TJ's wrong rejections.

Reuses the layer-ladder caches and labels exactly as results/varwidth/layers/layers.md ("L3 judges ... new (15
concepts)") does: CS + GS-norm runs (config L12), top-20 candidates, labels from experiments/varwidth/j3_data/.
Also prints each P-labelled feature that TJ rejected, with its stage-1 description, verdict and source words.
No model, no Gemini.

    .venv/bin/python paper/drafts/tj_rejections_heldout.py
"""
import pandas as pd
from experiments.varwidth.layer_report import cand_rows, labels, NEW, PROMPT
from experiments.varwidth.layer_eval import load

lab = labels()
K = cand_rows(lab, cfgs=["L12"])
S = K[K.concept.isin(NEW)].copy()
S["lo"] = S["lo"].fillna(0)
tj = S.tj.fillna(False).astype(bool)
cjs = S.cj.fillna(False).astype(bool) & (S.share >= 0.5)
S["TJ"], S["EJ5"], S["TWJ"], S["TWs"] = tj, S.lo >= 5, tj | (S.lo >= 5), tj | cjs
y = S.lab.eq("P"); yl = S.lab.isin(["P", "B"])
print("runs", S.groupby(["concept", "seed"]).ngroups, "runs with P", int(S.assign(y=y).groupby(["concept","seed"]).y.any().sum()))
for r in ["TJ", "EJ5", "TWJ", "TWs"]:
    a = S[r]
    tp, fp, fn = int((a & y).sum()), int((a & ~y).sum()), int((~a & y).sum())
    tpl, fpl, fnl = int((a & yl).sum()), int((a & ~yl).sum()), int((~a & yl).sum())
    hit = int(S.assign(h=a & y).groupby(["concept", "seed"]).h.any().sum())
    print(f"{r}: acc {int(a.sum())} P {tp/max(tp+fp,1):.3f} R {tp/max(tp+fn,1):.3f} hit {hit} | len P {tpl/max(tpl+fpl,1):.3f} R {tpl/max(tpl+fnl,1):.3f}")
rows = []
for c, g in S.groupby("concept"):
    yy = g.lab.eq("P")
    runs_p = int(g.assign(y=yy).groupby("seed").y.any().sum())
    rej = g[yy & ~g.TJ]
    rows.append({"concept": c, "P cands": int(yy.sum()), "TJ rejects P": len(rej),
                 "of which TWs accepts": int(rej.TWs.sum()),
                 "runs w/ P": runs_p,
                 "TJ hit": int(g.assign(h=g.TJ & yy).groupby("seed").h.any().sum()),
                 "TWs hit": int(g.assign(h=g.TWs & yy).groupby("seed").h.any().sum()),
                 "TJ FP": int((g.TJ & ~yy).sum()), "TJ FP (N)": int((g.TJ & g.lab.eq("N")).sum())})
D = pd.DataFrame(rows)
print(D.to_string())
print(D.sum(numeric_only=True))

import json
from experiments.varwidth.common import run_dir
from experiments.varwidth.layer_eval import CONFIGS
print("\n--- TJ-rejected P candidates (held-out) ---")
for _, x in S[y & ~S.TJ].sort_values(["concept", "seed"]).iterrows():
    d = run_dir(x.concept, CONFIGS["L12"], int(x.seed))
    J = json.loads((d / "judge.json").read_text())
    f = str(int(x.feature))
    desc = J["descriptions"].get(f, "?"); v = J["verdicts"].get(f)
    ca = load(x.concept, "L12", int(x.seed))
    fw = next(c["fw20"] for c in ca["candidates"] if c["feature"] == x.feature)
    print(f"{x.concept} s{int(x.seed)} f{f} rank{int(x['rank'])} lo={int(x.lo)} cj={x.cj} share={x.share:.2f} TWs={x.TWs} verdict={v}\n   {desc}\n   words: {', '.join(fw[:12])}")

print("\n--- lenient runs hit ---")
S["WJs"] = cjs
print("runs with P or B:", int(S.assign(y=yl).groupby(["concept","seed"]).y.any().sum()), "P+B cands", int(yl.sum()), "P cands", int(y.sum()))
for r in ["TJ", "WJs", "EJ5", "TWJ", "TWs"]:
    a = S[r]
    print(r, "accepts", int(a.sum()), "strict hit", int(S.assign(h=a & y).groupby(["concept","seed"]).h.any().sum()),
          "lenient hit", int(S.assign(h=a & yl).groupby(["concept","seed"]).h.any().sum()),
          "P %.2f R %.2f | len P %.2f R %.2f" % ((a&y).sum()/max(a.sum(),1), (a&y).sum()/y.sum(), (a&yl).sum()/max(a.sum(),1), (a&yl).sum()/yl.sum()))
