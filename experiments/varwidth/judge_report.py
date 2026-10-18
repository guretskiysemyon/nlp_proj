#!/usr/bin/env python
"""J_tok vs J_words vs J3@N over all judges.json (judge_compare.py) of one run tag, all concepts, seeds 42-46.

    python -m experiments.varwidth.judge_report [--tag rn_refit-False] [--out PATH]

J3@N = exact overlap of N concept words with the feature's 10 words, accept at >= 4 pairs; N = 10 is the spec, 20 / 30
are sensitivity. Tracked concept feature per run = argmax |u @ Zn|, u = top singular vector of the runs' own
J_tok-accepted top-rho feature directions over seeds (fallback: top-rho J_words, then J3@10, J3@30 accepts).
Writes results/varwidth/judge_compare.md (judge_compare_<tag>.md for a non-default tag).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from experiments.varwidth.common import RESULTS, ROOT, run_dir
from experiments.varwidth.judge_compare import DEFAULT_TAG
from experiments.varwidth.overall_report import md
from experiments.varwidth.overlap_judge import MIN_PAIRS, N_SIZES, N_WORDS

SEEDS = range(42, 47)
THRESHOLDS = (3, 4, 5)
N_DISAGREE = 30
CELLS = ("both", "only stage 2", "only J3", "neither")


def concepts() -> list:
    out = []
    for name in ("concept_sentences.json", "concept_sentences_extra.json"):
        p = ROOT / "data" / name
        if p.exists():
            out += [e["concept"] for e in json.loads(p.read_text()) if e["concept"] not in out]
    return out


def cand_row(concept, seed, J, c) -> dict:
    j3 = c["j3"]
    llm = j3[str(N_WORDS)].get("llm")
    r = {"concept": concept, "seed": seed, "feature": c["feature"], "rho": c["ratio_abs"], "share": c["concept_share"],
         "tok": c["j_tok"][2], "words": c["j_words"][2], "fwords": c["feature_words"],
         "desc_words": c["desc_words"], "desc_tok": c["desc_tok"], "cwords": J["concept_words"][str(N_WORDS)],
         "llm_n": None if llm is None else llm["n_pairs"], "llm_pairs": None if llm is None else llm["pairs"]}
    for n in N_SIZES:
        r[f"np{n}"] = j3[str(n)]["n_pairs"]
        r[f"pairs{n}"] = [a for a, _ in j3[str(n)]["exact_pairs"]]
        r[f"j3_{n}"] = r[f"np{n}"] >= MIN_PAIRS
    return r


def judge_cols() -> list:
    """(column, label) of every judge / pipeline flag."""
    return ([("tok", "J_tok"), ("words", "J_words")] + [(f"j3_{n}", f"J3@{n}") for n in N_SIZES]
            + [(f"w_or_{n}", f"J_words∨J3@{n}") for n in N_SIZES] + [(f"t_or_{n}", f"J_tok∨J3@{n}") for n in N_SIZES])


def load_concept(concept: str, tag: str):
    """(candidate rows, run rows) for one concept; `tracked` marks each run's tracked concept feature."""
    runs = {}
    for s in SEEDS:
        d = run_dir(concept, tag, s)
        if (d / "judges.json").exists() and (d / "fit.pt").exists():
            Z = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)["Z"].float()
            runs[s] = (json.loads((d / "judges.json").read_text()), Z / Z.norm(dim=0, keepdim=True).clamp_min(1e-12))
    if not runs:
        return [], []
    rows = {s: pd.DataFrame([cand_row(concept, s, J, c) for c in J["candidates"]]) for s, (J, _) in runs.items()}
    for df in rows.values():
        if len(df):
            for n in N_SIZES:
                df[f"w_or_{n}"] = df["words"] | df[f"j3_{n}"]
                df[f"t_or_{n}"] = df["tok"] | df[f"j3_{n}"]

    def top(s, key):
        df = rows[s]
        acc = df[df[key]] if len(df) else df
        return int(acc.loc[acc["rho"].idxmax(), "feature"]) if len(acc) else None

    ref = []
    for key in ("tok", "words", f"j3_{N_WORDS}", f"j3_{max(N_SIZES)}"):
        ref = [Zn[:, top(s, key)] for s, (_, Zn) in runs.items() if top(s, key) is not None]
        if ref:
            break
    u = torch.linalg.svd(torch.stack(ref, 1), full_matrices=False)[0][:, 0] if ref else None
    cands, rr = [], []
    for s, (J, Zn) in runs.items():
        df = rows[s]
        tracked = int((u @ Zn).abs().argmax()) if u is not None else None
        df["tracked"] = (df["feature"] == tracked) if len(df) else []
        cands += df.to_dict("records")
        tr = df[df["tracked"]].iloc[0] if len(df) and df["tracked"].any() else None
        info = J["features"].get(str(tracked), {}) if tracked is not None else {}
        run = {"concept": concept, "seed": s, "tracked": tracked, "tracked_cand": tr is not None,
               "tracked_rho": info.get("ratio_abs"), "tracked_share": info.get("concept_share"),
               "tracked_words": info.get("feature_words", []), "cwords": J["concept_words"][str(N_WORDS)],
               "tracked_np": {n: (int(tr[f"np{n}"]) if tr is not None else None) for n in N_SIZES}}
        for col, _ in judge_cols():
            run[f"found_{col}"] = bool(len(df) and df[col].any())
            run[f"tr_{col}"] = bool(tr is not None and tr[col])
        for n in N_SIZES:
            for t in THRESHOLDS:
                run[f"found_{n}_{t}"] = bool(len(df) and (df[f"np{n}"] >= t).any())
                run[f"tr_{n}_{t}"] = bool(tr is not None and tr[f"np{n}"] >= t)
        rr.append(run)
    return cands, rr


def cell(a, b) -> str:
    return CELLS[0] if a and b else CELLS[1] if a else CELLS[2] if b else CELLS[3]


def esc(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def mdt(df: pd.DataFrame) -> list:
    return md(df.map(esc)) if len(df) else ["(none)"]


def words(ws) -> str:
    return ", ".join(ws) or "—"


def pairs(ps) -> str:
    return "; ".join(f"{a}={b}" if a != b else a for a, b in ps) or "—"


def crosstab(C: pd.DataFrame, stage2: str, j3: str, title: str, per_concept: bool = True) -> list:
    C = C.assign(cell=[cell(a, b) for a, b in zip(C[stage2], C[j3])])
    g = C.groupby("cell")
    tot = pd.DataFrame({"cell": CELLS,
                        "candidates": [len(g.get_group(k)) if k in g.groups else 0 for k in CELLS],
                        "mean concept share": [f"{g.get_group(k)['share'].mean():.2f}" if k in g.groups else "—"
                                               for k in CELLS]})
    L = [f"### {title}", ""] + mdt(tot) + [""]
    if per_concept:
        per = []
        for concept, gc in C.groupby("concept", sort=False):
            row = {"concept": concept}
            for k in CELLS:
                x = gc[gc["cell"] == k]
                row[k] = f"{len(x)} ({x['share'].mean():.2f})" if len(x) else "0"
            per.append(row)
        L += ["Per concept: candidates (mean concept share)", ""] + mdt(pd.DataFrame(per)) + [""]
    return L


def review(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"concept": df["concept"], "seed": df["seed"], "feature": df["feature"],
                         "rho": df["rho"].map(lambda v: f"{v:.3g}"), "C share": df["share"].map(lambda v: f"{v:.2f}"),
                         "tracked": df["tracked"].map(lambda v: "✓" if v else ""),
                         "J_tok": df["tok"].map(lambda v: "✓" if v else ""),
                         "J_words": df["words"].map(lambda v: "✓" if v else ""),
                         "pairs @" + "/".join(map(str, N_SIZES)): ["/".join(str(r[f"np{n}"]) for n in N_SIZES)
                                                                  for _, r in df.iterrows()],
                         "feature words": df["fwords"].map(words),
                         f"matched @{max(N_SIZES)}": df[f"pairs{max(N_SIZES)}"].map(words),
                         "description (words)": df["desc_words"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else RESULTS / ("judge_compare.md" if a.tag == DEFAULT_TAG else f"judge_compare_{a.tag}.md")
    cs, rs = [], []
    for c in concepts():
        x, y = load_concept(c, a.tag)
        cs += x
        rs += y
    if not cs:
        raise SystemExit(f"no judges.json under {RESULTS}/*/{a.tag}/seed*")
    C, R = pd.DataFrame(cs), pd.DataFrame(rs)
    hi = max(N_SIZES)
    L = [f"# Judges compared: J_tok vs J_words vs J3 (tag {a.tag})", "",
         f"{R['concept'].nunique()} concepts, {len(R)} runs, {len(C)} rho ≥ 2 candidates.", "",
         "- **J_tok**: the paper's 2-stage judge (stage 1 sees the top-20 tokens). **J_words**: same, stage 1 sees "
         "'token → source words'. Both accept if is_member and confidence ≥ 0.85.",
         f"- **J3@N**: N concept words (Gemini, once per concept and N) vs the feature's first 10 distinct normalized "
         f"source words, exact match after normalization; accept at ≥ {MIN_PAIRS} matches. N = {N_WORDS} is the spec, "
         "the other sizes are sensitivity.",
         "- **Concept share**: C tokens / support of the feature. **Tracked**: the feature closest to the concept "
         "direction (see module docstring); seeds where it is not a candidate count as not accepted.", ""]

    # (1) candidate cross tables
    L += ["## 1. Candidate-level agreement (stage 2 = the 2-stage judge's accept)", ""]
    for n in (N_WORDS, hi):
        L += crosstab(C, "words", f"j3_{n}", f"J_words × J3@{n}") + crosstab(C, "tok", f"j3_{n}", f"J_tok × J3@{n}")
    for n in N_SIZES:
        if n not in (N_WORDS, hi):
            L += crosstab(C, "words", f"j3_{n}", f"J_words × J3@{n} (overall)", False)
            L += crosstab(C, "tok", f"j3_{n}", f"J_tok × J3@{n} (overall)", False)

    # (2) run level
    L += ["## 2. Run level", ""]
    for pre, title in (("found", "Seeds where the concept is found (any candidate accepted)"),
                       ("tr", "Seeds where the tracked concept feature is accepted")):
        rows = []
        for concept, g in list(R.groupby("concept", sort=False)) + [("**all**", R)]:
            row = {"concept": concept, "runs": len(g)}
            if pre == "tr":
                row["tracked is cand."] = int(g["tracked_cand"].sum())
            row.update({lab: int(g[f"{pre}_{col}"].sum()) for col, lab in judge_cols()})
            rows.append(row)
        L += [f"### {title}", ""] + mdt(pd.DataFrame(rows)) + [""]

    # (3) N x threshold grid
    L += ["## 3. J3 sensitivity: concept-word list size N × threshold t (matches ≥ t)", ""]
    rows = []
    for n in N_SIZES:
        for t in THRESHOLDS:
            acc = C[C[f"np{n}"] >= t]
            rows.append({"N": n, "t": t, "accepted candidates": len(acc),
                         "mean concept share": f"{acc['share'].mean():.2f}" if len(acc) else "—",
                         "share < 0.5": int((acc["share"] < 0.5).sum()),
                         "also J_words": int(acc["words"].sum()), "also J_tok": int(acc["tok"].sum()),
                         "runs found": f"{int(R[f'found_{n}_{t}'].sum())}/{len(R)}",
                         "tracked accepted": f"{int(R[f'tr_{n}_{t}'].sum())}/{len(R)}"})
    L += mdt(pd.DataFrame(rows)) + ["", "Matches per candidate (count of candidates):", ""]
    L += mdt(pd.DataFrame({f"N={n}": C[f"np{n}"].value_counts() for n in N_SIZES}).fillna(0).astype(int)
             .rename_axis("matches").reset_index()) + [""]

    # (4) LLM matcher vs exact
    L += [f"## 4. LLM matcher vs deterministic exact match (N = {N_WORDS})", ""]
    M = C[C["llm_n"].notna()]
    if len(M):
        ex = M[f"np{N_WORDS}"]
        diff = M[M["llm_n"] != ex]
        L += [f"Candidates with the LLM match: {len(M)} ({M['concept'].nunique()} concepts, "
              f"{M.groupby(['concept', 'seed']).ngroups} runs). LLM n_pairs ≠ exact: {len(diff)} ({len(diff) / len(M):.1%}).", ""]
        L += mdt(pd.DataFrame([{"t": t, "accept differs": int(((M["llm_n"] >= t) != (ex >= t)).sum()),
                                "LLM accepts, exact not": int(((M["llm_n"] >= t) & (ex < t)).sum())}
                               for t in THRESHOLDS])) + [""]
        D = diff.assign(extra=diff["llm_n"] - diff[f"np{N_WORDS}"]).sort_values("extra", ascending=False).head(N_DISAGREE)
        L += [f"Up to {N_DISAGREE} disagreeing cases:", ""]
        L += mdt(pd.DataFrame({"concept": D["concept"], "seed": D["seed"], "feature": D["feature"],
                               "concept words": D["cwords"].map(words), "feature words": D["fwords"].map(words),
                               "LLM-only pairs": [pairs([p for p in ps if p[0] != p[1]]) for ps in D["llm_pairs"]],
                               "exact": D[f"pairs{N_WORDS}"].map(words)})) + [""]
    else:
        L += ["LLM matcher not run in this sweep (judge_compare --llm-match).", ""]

    # (5) eye review
    L += ["## 5. Eye review", ""]
    for n in (N_WORDS, hi):
        L += [f"### J3@{n} accepts that J_words rejects", ""] + mdt(review(C[C[f"j3_{n}"] & ~C["words"]])) + [""]
    L += [f"### J_words accepts that J3@{N_WORDS} rejects", ""]
    L += mdt(review(C[C["words"] & ~C[f"j3_{N_WORDS}"]])) + [""]
    L += [f"### Runs with nothing accepted under J_words∨J3@{N_WORDS}", ""]
    N = R[~R[f"found_w_or_{N_WORDS}"]]
    fmt = lambda v, f: "—" if v is None or pd.isna(v) else format(v, f)  # noqa: E731
    L += mdt(pd.DataFrame({"concept": N["concept"], "seed": N["seed"],
                           "tracked": N["tracked"].map(lambda v: fmt(v, "d") if v is None or pd.isna(v) else int(v)),
                           "rho": N["tracked_rho"].map(lambda v: fmt(v, ".3g")),
                           "C share": N["tracked_share"].map(lambda v: fmt(v, ".2f")),
                           "matches @" + "/".join(map(str, N_SIZES)):
                               N["tracked_np"].map(lambda d: "/".join("—" if d[n] is None else str(d[n]) for n in N_SIZES)),
                           "tracked feature words": N["tracked_words"].map(words),
                           f"concept words @{N_WORDS}": N["cwords"].map(words)})) + [""]
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
