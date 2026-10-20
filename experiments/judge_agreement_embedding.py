#!/usr/bin/env python3
"""Do Gemini-2.5-Flash-Lite (the paper's judge) and Gemini-3.5-Flash-Lite keep the same EMBER
embedding features? (no GPU)

Compares, per concept, the paper's released English embedding interpretations
(mf_outputs/google_gemma-2-2b-it/interpretations/rank100/seed42/<C>/embedding/) with the
re-judge of the same candidates by 3.5 (mf_outputs/en_rejudge35/..., written by
`python -m ember.interpret_features --tracks embedding --outdir mf_outputs/en_rejudge35`).
Same candidates and token lists (ratio >= 2); kept = is_member and confidence >= 0.85.
If a concept's kept list is identical, the EMBER edit (and every Part 1 EMBER result) is
the same under either judge.

Output: <new>/embedding_judge_agreement.csv and .md

Usage: .venv/bin/python -I experiments/judge_agreement_embedding.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "mf_outputs/google_gemma-2-2b-it/interpretations/rank100/seed42"
NEW = ROOT / "mf_outputs/en_rejudge35/google_gemma-2-2b-it/interpretations/rank100/seed42"


def load(base, c):
    p = base / c / "embedding" / "from_activation.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["member"] = d["is_member"].astype(str).str.lower().eq("true")
    d["conf"] = pd.to_numeric(d["confidence"], errors="coerce").fillna(0)
    d["kept"] = d["member"] & (d["conf"] >= 0.85)
    return d.set_index("feature")


rows = []
for cdir in sorted(OLD.iterdir()):
    c = cdir.name
    a, b = load(OLD, c), load(NEW, c)
    if a is None or b is None:
        rows.append({"concept": c, "status": "missing " + ("paper" if a is None else "re-judge")})
        continue
    j = a.join(b, lsuffix="_25", rsuffix="_35", how="inner")
    k25 = sorted(j.index[j.kept_25].tolist())
    k35 = sorted(j.index[j.kept_35].tolist())
    rows.append({
        "concept": c, "status": "ok", "candidates": len(j),
        "same_tokens": bool((j.tokens_provided_25 == j.tokens_provided_35).all()),
        "member_agree": int((j.member_25 == j.member_35).sum()),
        "kept_2.5": " ".join(map(str, k25)), "kept_3.5": " ".join(map(str, k35)),
        "same_kept_list": k25 == k35,
        "flips": "; ".join(f"f{f}: 2.5 {'keep' if j.kept_25[f] else 'drop'} ({j.conf_25[f]:.2f}) / "
                           f"3.5 {'keep' if j.kept_35[f] else 'drop'} ({j.conf_35[f]:.2f}): "
                           f"{str(j.description_35[f])[:80]}"
                           for f in j.index if j.kept_25[f] != j.kept_35[f]),
    })
t = pd.DataFrame(rows)
NEW.parent.parent.parent.mkdir(parents=True, exist_ok=True)
out = ROOT / "mf_outputs/en_rejudge35"
t.to_csv(out / "embedding_judge_agreement.csv", index=False)
ok = t[t.status == "ok"]
lines = ["# EMBER embedding features: paper judge (2.5) vs gemini-3.5-flash-lite", "",
         f"Concepts compared: {len(ok)} of {len(t)}. Same kept list: **{int(ok.same_kept_list.sum())} of {len(ok)}**. "
         f"Decisions agreeing (is_member): {int(ok.member_agree.sum())} of {int(ok.candidates.sum())}.", "",
         "| concept | candidates | decisions agree | kept 2.5 | kept 3.5 | same list |", "|---|---|---|---|---|---|"]
for _, r in t.iterrows():
    if r.status != "ok":
        lines.append(f"| {r.concept} | {r.status} | | | | |")
        continue
    lines.append(f"| {r.concept} | {r.candidates} | {r.member_agree} | {r['kept_2.5'] or '-'} | {r['kept_3.5'] or '-'} | "
                 f"{'yes' if r.same_kept_list else '**no**'} |")
flips = ok[ok.flips.astype(bool)]
if len(flips):
    lines += ["", "## Changed decisions", ""] + [f"- **{r.concept}**: {r.flips}" for _, r in flips.iterrows()]
(out / "embedding_judge_agreement.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0)
