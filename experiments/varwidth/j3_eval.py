#!/usr/bin/env python
"""J3 (20 x 20) vs J_tok / J_words on hand-labeled candidates: pool export, confusion tables, threshold sweep.

    python -m experiments.varwidth.j3_eval pool   [--scope all|new|wide|noprompt] [--min-j3 2]   # features to label
    python -m experiments.varwidth.j3_eval report [--scope ...] [--version v1] [--out PATH]

Candidates: every rho candidate of every run with a judges.json (judge_compare.py) and fw20 words (j3_words features).
J3@t: >= t concept words of j3_data/concept_words_<version>.json equal (normalized) to one of the feature's 20 words;
each concept word counts once. W = J_words and concept share >= 0.5.
Labels (j3_data/labels.json, {group key: P|B|N}): a group key is "concept | sorted first 6 feature words", so a feature
recurring over seeds is labeled once. P = specific to the concept; B = borderline (broader category or adjacent,
e.g. generic nuclear for Uranium); N = not the concept. Strict counts B as negative, lenient as positive.
Unlabeled candidates count as negatives, so FN only counts misses some rule caught and TN is uninformative.
pool lists the unlabeled groups that any rule accepts (J_tok, W or J3 >= --min-j3), without judge flags (label blind),
to a CSV under results/varwidth/j3/; add their labels to labels.json.
Scopes: all; new = the concepts added after the prompt was chosen (NEW_CONCEPTS, the held-out set); wide = its broad
topics (WIDE_CONCEPTS);
noprompt = all minus the concepts used as examples in prompt_v1 (PROMPT_EXAMPLES).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.varwidth.common import RESULTS
from experiments.varwidth.j3_words import DATA, FW20, all_concepts, concept_lists

PROMPT_EXAMPLES = {"Culture of Greece", "Ancient Rome", "Artificial intelligence", "World War II"}
NEW_CONCEPTS = ["Judaism", "Karate", "Existentialism", "Prime number", "The Matrix", "Louvre",
                "American Revolutionary War", "Robot", "Sigmund Freud", "Giraffe", "Surfing", "Flush toilet",
                "Cloud", "Doctor of Philosophy", "Tourism"]
WIDE_CONCEPTS = ["Cloud", "Doctor of Philosophy", "Tourism"]   # broad topics among NEW_CONCEPTS
OUT = RESULTS / "j3"
KEY = ["concept", "seed"]


def group_key(concept: str, fw: list) -> str:
    return f"{concept} | {', '.join(sorted(fw[:6]))}"


def j3_count(cw: list, fw: list) -> int:
    fs = set(fw)
    return sum(w in fs for w in dict.fromkeys(cw))


def candidates(version: str = "v1") -> pd.DataFrame:
    from experiments.varwidth.judge_compare import DEFAULT_TAG
    from experiments.varwidth.judge_report import load_concept
    fw20 = json.loads(FW20.read_text())
    cw = concept_lists(version)
    rows = []
    for c in all_concepts():
        if c not in cw:
            continue
        for x in load_concept(c, DEFAULT_TAG)[0]:
            fw = fw20.get(f"{c}|{x['seed']}|{x['feature']}")
            if fw is None:
                continue
            rows.append({k: x[k] for k in ("concept", "seed", "feature", "share", "tok", "words", "tracked")}
                        | {"fw20": fw, "j3": j3_count(cw[c], fw), "key": group_key(c, fw)})
    C = pd.DataFrame(rows)
    C["W"] = C.words & (C.share >= 0.5)
    labels = json.loads((DATA / "labels.json").read_text())
    C["lab"] = C.key.map(labels)
    return C


def in_scope(C: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "new":
        return C[C.concept.isin(NEW_CONCEPTS)]
    if scope == "wide":
        return C[C.concept.isin(WIDE_CONCEPTS)]
    if scope == "noprompt":
        return C[~C.concept.isin(PROMPT_EXAMPLES)]
    return C


def rules(C: pd.DataFrame) -> dict:
    j = C.j3
    return {"J_tok": C.tok, "J_words": C.words, "W (J_words ∧ share≥.5)": C.W,
            "J3@3": j >= 3, "J3@4": j >= 4, "J3@5": j >= 5,
            "J_tok ∨ W (combined)": C.tok | C.W, "J_tok ∨ J3@3": C.tok | (j >= 3),
            "J_tok ∨ J3@4": C.tok | (j >= 4), "J_tok ∨ J3@5": C.tok | (j >= 5),
            "W ∨ J3@4": C.W | (j >= 4), "W ∨ J3@5": C.W | (j >= 5)}


def confusion(C: pd.DataFrame, accept: pd.Series, lenient: bool) -> dict:
    y = C.lab.eq("P") | (lenient & C.lab.eq("B"))
    tp, fp, fn = int((accept & y).sum()), int((accept & ~y).sum()), int((~accept & y).sum())
    hit = int(C.assign(h=accept & y).groupby(KEY).h.any().sum())
    return {"TP": tp, "FP": fp, "FN": fn, "precision": round(tp / max(tp + fp, 1), 2),
            "recall": round(tp / max(tp + fn, 1), 2), "runs hit": hit}


def mdt(df: pd.DataFrame) -> list:
    cols = list(df.columns)
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
           ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]


def report(C: pd.DataFrame, scope: str) -> list:
    S = in_scope(C, scope)
    unl = int((S.lab.isna() & (S.tok | S.W | (S.j3 >= 3))).sum())
    L = [f"## Scope: {scope} — {S.concept.nunique()} concepts, {S.groupby(KEY).ngroups} runs, {len(S)} candidates", "",
         f"Labels in scope: {S.drop_duplicates('key').lab.value_counts().to_dict()}; unlabeled accepted candidates "
         f"(J_tok, W or J3 ≥ 3): {unl}.", ""]
    for lenient in (False, True):
        y = S.lab.eq("P") | (lenient & S.lab.eq("B"))
        L += [f"### {'Lenient (B = positive)' if lenient else 'Strict (B = negative)'}: {int(y.sum())} positives, "
              f"{int(S.assign(y=y).groupby(KEY).y.any().sum())} runs with a positive", ""]
        L += mdt(pd.DataFrame([{"rule": k} | confusion(S, a[S.index], lenient) for k, a in rules(C).items()])) + [""]
    L += ["### J3 threshold sweep (strict P/R | lenient P/R | strict runs hit)", ""]
    rows = []
    for name, base in (("J3", None), ("J_tok ∨ J3", S.tok), ("W ∨ J3", S.W)):
        for t in range(2, 9):
            a = (S.j3 >= t) if base is None else (base | (S.j3 >= t))
            s, l = confusion(S, a, False), confusion(S, a, True)
            rows.append({"rule": name, "t": t, "accepts": int(a.sum()), "strict P": s["precision"], "strict R": s["recall"],
                         "lenient P": l["precision"], "lenient R": l["recall"], "runs hit (strict)": s["runs hit"]})
    return L + mdt(pd.DataFrame(rows)) + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pool", "report"))
    ap.add_argument("--scope", default="all", choices=("all", "new", "wide", "noprompt"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--min-j3", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    C = candidates(a.version)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "pool":
        S = in_scope(C, a.scope)
        P = S[S.lab.isna() & (S.tok | S.W | (S.j3 >= a.min_j3))]
        G = P.groupby("key").agg(concept=("concept", "first"), n=("seed", "size"), share=("share", "mean"),
                                 words=("fw20", lambda s: ", ".join(s.iloc[0][:14]))).reset_index()
        path = Path(a.out) if a.out else OUT / f"pool_{a.scope}.csv"
        G.sort_values(["concept", "share"]).to_csv(path, index=False)
        print(f"{len(P)} unlabeled candidates in {len(G)} groups -> {path}")
        return
    scopes = ("new", "wide", "noprompt", "all") if a.scope == "all" else (a.scope,)
    L = [f"# J3 ({a.version}, 20 x 20, exact) vs J_tok / J_words on labeled candidates", ""]
    for sc in scopes:
        L += report(C, sc)
    path = Path(a.out) if a.out else OUT / "j3_eval.md"
    path.write_text("\n".join(L))
    print("\n".join(L))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
