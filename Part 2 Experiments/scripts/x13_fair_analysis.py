#!/usr/bin/env python
"""X13: analyses of the fair run's held-out results (CPU, no model; judge_rerun/Fair + results/crosslingual/fair_*).

All on QA_test / SimdomQA_test in en/de/es/fr/ru, 18 concepts, arms ember, cinder, ember_hj, ember_x, cinder_x.
A cell (concept, language) is reliable when the pristine QA_test accuracy is >= 0.40 (Part 1's rule); unreliable
cells (Harry Potter ru, Valentine's Day ru) are left out of every test here.

    1  stats      paired arm differences in leak (mean of de/es/fr/ru retention - en retention), PRIMARY = all 4
                  languages, 18 concepts: mean, 95% CI from a two-level bootstrap (concepts, then the 50 questions of
                  each cell, paired across arms), Wilcoxon over concepts, Holm over the 6 comparisons; sensitivity:
                  unclipped retention, de/es/fr only
    2  mechanism  extrapolated arm vs its parent, per question: questions the pristine model answers correctly are
                  split into "new" (an added token in the question or correct answer, no parent token), "old" (a
                  parent token) and "none"; share answered wrong after the edit, per group
    3  coverage   coverage (share of QA_test questions with an edited token) vs leak_L = ret_L - ret_en over all
                  edited reliable (concept, foreign language) cells of all arms: Spearman, pooled and per arm
    4  failures   examples: foreign questions still answered correctly by EMBER+X, with and without an edited token,
                  and how the correct answer is tokenized
    5  tokens     the tokens extrapolation adds for the 12 concepts outside Part 2's first 6 (judge_rerun/Fair/x_build.json)

    python "Part 2 Experiments/scripts/x13_fair_analysis.py"

Output: Part 2 Experiments/x13_fair_analysis/{stats.csv, mechanism.csv, coverage.csv, failures.md, report.md}
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = p2x.C
LANGS = ["en", "de", "es", "fr", "ru"]
FOR = LANGS[1:]
ARMS = ["ember", "cinder", "ember_hj", "ember_x", "cinder_x"]
PARENT = {"ember_x": "ember", "cinder_x": "cinder"}
PAIRS = [("ember_x", "ember"), ("cinder_x", "cinder"), ("cinder", "ember"), ("cinder_x", "ember_x"),
         ("ember_hj", "ember"), ("cinder", "ember_hj")]
XL = C.ROOT / "results" / "crosslingual" / C.MODEL_SAFE
OUT = p2x.outdir("x13_fair_analysis")
REL = 0.40
B = 4000
RNG = np.random.default_rng(42)


def fair():
    sys.path.insert(0, str(C.ROOT))
    from cinder import fair as F
    return F


@lru_cache(maxsize=None)
def cell(src, concept, lang, s="QA_test"):
    """Per-question correctness (0/1, length 50) and the records, or None if the file does not exist."""
    p = XL / src / C.safe(concept) / f"{lang}.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())["results"][s]["records"]
    return np.array([bool(r["is_correct"]) for r in recs], dtype=float), recs


def arm_cell(arm, concept, lang, s="QA_test"):
    """The arm's per-question correctness; no edit -> pristine; ember_hj with ember's edit -> ember's files."""
    F = fair()
    src = F.src_method(arm, concept)
    ed = json.loads((F.xl_dir(src, concept) / "edit.json").read_text())
    if ed["no_edit"]:
        return cell("baseline_fixed", concept, lang, s)
    return cell(f"fair_{src}", concept, lang, s)


def no_edit(arm, concept):
    F = fair()
    return json.loads((F.xl_dir(F.src_method(arm, concept), concept) / "edit.json").read_text())["no_edit"]


def retention(acc, base, clip=True):
    r = (acc - 0.25) / (base - 0.25)
    return np.clip(r, 0, 1) if clip else r


@lru_cache(maxsize=None)
def edit_tokens(arm, concept):
    from cinder.edit_size import edit_vectors
    F = fair()
    if no_edit(arm, concept):
        return frozenset()
    return frozenset(edit_vectors(concept, F.ROOTS[F.src_method(arm, concept)])[0])


@lru_cache(maxsize=None)
def q_tokens(concept, lang):
    """Token ids of question + correct answer of each QA_test record (edit_coverage's tokenization)."""
    import edit_coverage as ec
    ec.TOK = p2x.tokenizer()
    _, recs = cell("baseline_fixed", concept, lang)
    return [set(ec.tok_ids(r["question"])) | set(ec.tok_ids(r["correct_text"])) for r in recs]


def concepts():
    return list(fair().ALL_CONCEPTS)


def reliable(concept, lang):
    return cell("baseline_fixed", concept, lang)[0].mean() >= REL


# ---------------------------------------------------------------- 1. statistics
def leak_from(correct, base, concept, langs, clip=True):
    """correct / base: {lang: 0/1 vector}; leak over the reliable languages in `langs`."""
    ret = {L: retention(correct[L].mean(), base[L].mean(), clip) for L in ["en"] + langs if reliable(concept, L)}
    fr = [ret[L] for L in langs if L in ret]
    return float(np.mean(fr) - ret["en"])


def stats():
    from scipy.stats import wilcoxon
    cs = concepts()
    data = {c: {"base": {L: cell("baseline_fixed", c, L)[0] for L in LANGS},
                **{a: {L: arm_cell(a, c, L)[0] for L in LANGS} for a in ARMS}} for c in cs}
    rows = []
    for label, langs, clip in (("primary: de/es/fr/ru, clipped", FOR, True), ("unclipped retention", FOR, False),
                               ("de/es/fr only", ["de", "es", "fr"], True)):
        for a, b in PAIRS:
            d = np.array([leak_from(data[c][a], data[c]["base"], c, langs, clip)
                          - leak_from(data[c][b], data[c]["base"], c, langs, clip) for c in cs])
            boot = np.empty(B)
            for i in range(B):                      # resample concepts, then questions within each cell (paired)
                cc = RNG.integers(0, len(cs), len(cs))
                vals = []
                for j in cc:
                    c = cs[j]
                    idx = {L: RNG.integers(0, 50, 50) for L in LANGS}
                    rs = {k: {L: v[L][idx[L]] for L in LANGS} for k, v in data[c].items()}
                    vals.append(leak_from(rs[a], rs["base"], c, langs, clip) - leak_from(rs[b], rs["base"], c, langs, clip))
                boot[i] = np.mean(vals)
            nz = d[np.abs(d) > 1e-12]
            p = float(wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
            rows.append({"analysis": label, "pair": f"{a} − {b}", "mean": round(float(d.mean()), 3),
                         "ci_low": round(float(np.quantile(boot, 0.025)), 3),
                         "ci_high": round(float(np.quantile(boot, 0.975)), 3),
                         "lower": int((d < -1e-12).sum()), "higher": int((d > 1e-12).sum()),
                         "equal": int((np.abs(d) <= 1e-12).sum()), "wilcoxon_p": round(p, 4)})
        sub = [r for r in rows if r["analysis"] == label]
        order = sorted(range(len(sub)), key=lambda i: sub[i]["wilcoxon_p"])
        running = 0.0
        for k, i in enumerate(order):               # Holm step-down
            running = max(running, min(1.0, (len(sub) - k) * sub[i]["wilcoxon_p"]))
            sub[i]["holm_p"] = round(running, 4)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "stats.csv", index=False)
    return S


# ---------------------------------------------------------------- 2. mechanism
def mechanism():
    rows = []
    for arm, par in PARENT.items():
        for c in concepts():
            if no_edit(arm, c):
                continue
            Ex, Ep = edit_tokens(arm, c), edit_tokens(par, c)
            added = Ex - Ep
            for L in LANGS:
                if not reliable(c, L):
                    continue
                base = cell("baseline_fixed", c, L)[0]
                after_x, after_p = arm_cell(arm, c, L)[0], arm_cell(par, c, L)[0]
                for i, toks in enumerate(q_tokens(c, L)):
                    if not base[i]:
                        continue
                    grp = "old" if toks & Ep else "new" if toks & added else "none"
                    rows.append({"arm": arm, "concept": c, "lang": L, "group": grp,
                                 "lost_parent": 1 - after_p[i], "lost_x": 1 - after_x[i]})
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "mechanism.csv", index=False)
    M["lang_kind"] = np.where(M.lang == "en", "en", "foreign")
    agg = M.groupby(["arm", "lang_kind", "group"]).agg(questions=("lost_x", "size"), lost_parent=("lost_parent", "mean"),
                                                        lost_x=("lost_x", "mean")).round(3).reset_index()
    agg["gain"] = (agg.lost_x - agg.lost_parent).round(3)
    return agg


# ---------------------------------------------------------------- 3. coverage vs leak
def coverage():
    from scipy.stats import spearmanr
    rows = []
    for a in ARMS:
        for c in concepts():
            if no_edit(a, c):
                continue
            E = edit_tokens(a, c)
            base = {L: cell("baseline_fixed", c, L)[0].mean() for L in LANGS}
            ret = {L: float(retention(arm_cell(a, c, L)[0].mean(), base[L])) for L in LANGS}
            cov = {L: float(np.mean([bool(t & E) for t in q_tokens(c, L)])) for L in LANGS}
            for L in FOR:
                if reliable(c, L):
                    rows.append({"arm": a, "concept": c, "lang": L, "coverage": round(cov[L], 3),
                                 "coverage_gap": round(cov["en"] - cov[L], 3), "leak": round(ret[L] - ret["en"], 3)})
    V = pd.DataFrame(rows)
    V.to_csv(OUT / "coverage.csv", index=False)
    out = []
    for name, sub in [("all arms", V)] + [(a, V[V.arm == a]) for a in ARMS]:
        r1, p1 = spearmanr(sub.coverage, sub.leak)
        r2, p2 = spearmanr(sub.coverage_gap, sub.leak)
        out.append({"arms": name, "cells": len(sub), "rho_coverage": round(float(r1), 2), "p": f"{p1:.1e}",
                    "rho_gap": round(float(r2), 2), "p_gap": f"{p2:.1e}"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 4. failure cases
def failures(n=3):
    tok = p2x.tokenizer()
    L_ = ["# X13: where EMBER+X still fails (held-out QA_test, questions the pristine model answers correctly)", ""]
    for L in FOR:
        kept_cov, kept_unc, total = [], [], 0
        for c in concepts():
            if no_edit("ember_x", c) or not reliable(c, L):
                continue
            E = edit_tokens("ember_x", c)
            base = cell("baseline_fixed", c, L)[0]
            after, recs = arm_cell("ember_x", c, L)
            for i, toks in enumerate(q_tokens(c, L)):
                if base[i]:
                    total += 1
                    if after[i]:
                        (kept_cov if toks & E else kept_unc).append((c, recs[i]))
        L_ += [f"## {L}: {len(kept_cov) + len(kept_unc)} of {total} still correct "
               f"({len(kept_unc)} with no edited token, {len(kept_cov)} with one)", ""]
        for kind, lst in (("no edited token", kept_unc), ("edited token present", kept_cov)):
            pick = [lst[i] for i in RNG.choice(len(lst), size=min(n, len(lst)), replace=False)] if lst else []
            for c, r in pick:
                pieces = tok.convert_ids_to_tokens(tok(" " + r["correct_text"], add_special_tokens=False)["input_ids"])
                L_.append(f"- *{kind}* — {c}: {r['question']} → **{r['correct_text']}** (tokens: {' '.join(pieces)})")
        L_.append("")
    (OUT / "failures.md").write_text("\n".join(L_) + "\n")
    return L_


# ---------------------------------------------------------------- 5. added tokens, 12 new concepts
def tokens():
    rows = json.loads((C.ROOT / "judge_rerun" / "Fair" / "x_build.json").read_text())
    import x10_extrapolation_ablation as X10
    english = X10.english_tokens()
    tok = p2x.tokenizer()
    out = []
    for r in rows:
        if r.get("no_feature") or r["concept"] in p2x.CONCEPTS:
            continue
        ids = tok.convert_tokens_to_ids(r["added"])
        out.append({"concept": r["concept"], "arm": r["method"], "n_added": r["n_added"],
                    "whole_word": round(float(np.mean([p2x.is_word_token(t) for t in r["added"]])), 2),
                    "foreign_only": round(float(np.mean([i not in english for i in ids])), 2),
                    "first_25": " ".join(r["added"][:25])})
    return pd.DataFrame(out)


def main():
    S = stats()
    M = mechanism()
    V = coverage()
    failures()
    T = tokens()
    L_ = ["# X13: analyses of the fair run (held-out QA_test, 18 concepts)", "",
          f"Reliable cells only (pristine QA_test ≥ {REL}); leak = mean(ret de/es/fr/ru) − ret_en. See the script "
          "docstring for the definitions.", "",
          "## 1. Paired leak differences (two-level bootstrap 95% CI, Wilcoxon, Holm over 6 comparisons)", "",
          p2x.md_table(S), "",
          "## 2. Mechanism: extrapolated arm vs parent, share of pristine-correct questions lost", "",
          "group: old = has a parent edit token; new = only an added token; none = neither.", "",
          p2x.md_table(M), "",
          "## 3. Coverage vs leak over edited reliable (concept, foreign language) cells", "",
          p2x.md_table(V), "",
          "## 4. Failure cases", "", "See `failures.md`.", "",
          "## 5. Tokens extrapolation adds, 12 concepts outside the first 6", "", p2x.md_table(T), ""]
    (OUT / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
