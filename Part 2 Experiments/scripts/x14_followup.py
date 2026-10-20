#!/usr/bin/env python
"""X14: follow-up analyses of the fair run (CPU, no model), the checks X13 left open.

Same data, reliable-cell rule (pristine QA_test >= 0.40) and bootstrap as X13 (its loaders are reused). Arms: X13's
five plus the native-text arms ml_sep / ml_joint (cinder/ml_arms.py) when all 18 of their results exist.

    1  foreign    paired differences in FOREIGN RETENTION (mean of the reliable de/es/fr/ru retentions; lower = more
                  erased), next to leak. Leak = foreign - English retention, so an arm that erases English more gets a
                  higher leak at the same foreign erasure; foreign retention does not have that problem.
    2  joint      the same paired leak / foreign retention differences on the concepts BOTH arms edit (a no-edit
                  concept has leak 0 and retention 1 by construction, which favours the arm that edits less).
    3  collateral SimdomQA_test (similar domain, 50 questions per language): questions the pristine model answers
                  correctly, split by whether question or correct answer contains an edited token; share lost.
    4  exposure   share of ordinary sentences that contain an edited token: the other 17 concepts' sentences, per
                  language (English data/, foreign data_v2). Where collateral damage can happen; X10 did this for
                  EMBER / X on 6 concepts. No measured damage on general text (no foreign MMLU): a limitation.

    python "Part 2 Experiments/scripts/x14_followup.py"

Output: Part 2 Experiments/x14_followup/{stats.csv, collateral.csv, exposure.csv, report.md}
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import x13_fair_analysis as X13  # noqa: E402

C = p2x.C
LANGS, FOR = X13.LANGS, X13.FOR
OUT = p2x.outdir("x14_followup")
B = 4000
RNG = np.random.default_rng(42)
ML = ["ml_sep", "ml_joint"]


def arms():
    F = X13.fair()
    have = [a for a in ML if all((F.xl_dir(a, c) / "edit.json").exists() for c in F.ALL_CONCEPTS)]
    return list(X13.ARMS) + have


def pairs(A):
    P = list(X13.PAIRS)
    if "ml_sep" in A:
        P += [("ml_sep", "ember"), ("ml_sep", "ember_x")]
    if "ml_joint" in A:
        P += [("ml_joint", "ember"), ("ml_joint", "ember_x")]
    if {"ml_sep", "ml_joint"} <= set(A):
        P += [("ml_joint", "ml_sep")]
    return P


def foreign_ret(correct, base, concept, langs=FOR):
    r = [X13.retention(correct[L].mean(), base[L].mean()) for L in langs if X13.reliable(concept, L)]
    return float(np.mean(r))


def measures(correct, base, concept):
    return {"leak": X13.leak_from(correct, base, concept, FOR), "foreign_ret": foreign_ret(correct, base, concept)}


def holm(sub):
    order = sorted(range(len(sub)), key=lambda i: (np.nan_to_num(sub[i]["wilcoxon_p"], nan=1.0)))
    running = 0.0
    for k, i in enumerate(order):
        p = sub[i]["wilcoxon_p"]
        running = max(running, min(1.0, (len(sub) - k) * (1.0 if p != p else p)))
        sub[i]["holm_p"] = round(running, 4)


# ---------------------------------------------------------------- 1 + 2. paired statistics
def stats(A):
    from scipy.stats import wilcoxon
    cs = X13.concepts()
    data = {c: {"base": {L: X13.cell("baseline_fixed", c, L)[0] for L in LANGS},
                **{a: {L: X13.arm_cell(a, c, L)[0] for L in LANGS} for a in A}} for c in cs}
    rows = []
    for subset in ("all 18", "both edit"):
        for measure in ("leak", "foreign_ret"):
            label = f"{measure}, {subset}"
            for a, b in pairs(A):
                cc_ = [c for c in cs if subset == "all 18" or not (X13.no_edit(a, c) or X13.no_edit(b, c))]
                if len(cc_) < 3:
                    continue
                d = np.array([measures(data[c][a], data[c]["base"], c)[measure]
                              - measures(data[c][b], data[c]["base"], c)[measure] for c in cc_])
                boot = np.empty(B)
                for i in range(B):                   # concepts, then the 50 questions of each cell, paired
                    vals = []
                    for j in RNG.integers(0, len(cc_), len(cc_)):
                        c = cc_[j]
                        idx = {L: RNG.integers(0, 50, 50) for L in LANGS}
                        rs = {k: {L: data[c][k][L][idx[L]] for L in LANGS} for k in ("base", a, b)}
                        vals.append(measures(rs[a], rs["base"], c)[measure] - measures(rs[b], rs["base"], c)[measure])
                    boot[i] = np.mean(vals)
                nz = d[np.abs(d) > 1e-12]
                p = float(wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
                rows.append({"analysis": label, "pair": f"{a} − {b}", "concepts": len(cc_),
                             "mean": round(float(d.mean()), 3), "ci_low": round(float(np.quantile(boot, .025)), 3),
                             "ci_high": round(float(np.quantile(boot, .975)), 3), "lower": int((d < -1e-12).sum()),
                             "higher": int((d > 1e-12).sum()), "equal": int((np.abs(d) <= 1e-12).sum()),
                             "wilcoxon_p": round(p, 4)})
            holm([r for r in rows if r["analysis"] == label])
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "stats.csv", index=False)
    return S


def means(A):
    out = []
    for a in A:
        r = {"arm": a, "no_edit": 0}
        vals = {k: [] for k in ("ret_en", "foreign_ret", "leak")}
        for c in X13.concepts():
            base = {L: X13.cell("baseline_fixed", c, L)[0] for L in LANGS}
            cor = {L: X13.arm_cell(a, c, L)[0] for L in LANGS}
            r["no_edit"] += X13.no_edit(a, c)
            vals["ret_en"].append(float(X13.retention(cor["en"].mean(), base["en"].mean())))
            for k, v in measures(cor, base, c).items():
                vals[k].append(v)
        out.append({**r, **{k: round(float(np.mean(v)), 3) for k, v in vals.items()}})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 3. collateral on SimdomQA_test
@lru_cache(maxsize=None)
def q_tokens_sim(concept, lang):
    import edit_coverage as ec
    ec.TOK = p2x.tokenizer()
    _, recs = X13.cell("baseline_fixed", concept, lang, "SimdomQA_test")
    return [set(ec.tok_ids(r["question"])) | set(ec.tok_ids(r["correct_text"])) for r in recs]


def collateral(A):
    rows = []
    for a in A:
        for c in X13.concepts():
            if X13.no_edit(a, c):
                continue
            E = X13.edit_tokens(a, c)
            for L in LANGS:
                base = X13.cell("baseline_fixed", c, L, "SimdomQA_test")[0]
                after = X13.arm_cell(a, c, L, "SimdomQA_test")[0]
                for i, toks in enumerate(q_tokens_sim(c, L)):
                    if base[i]:
                        rows.append({"arm": a, "concept": c, "lang": L, "touched": bool(toks & E), "lost": 1 - after[i]})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "collateral.csv", index=False)
    D["lang_kind"] = np.where(D.lang == "en", "en", "foreign")
    return (D.groupby(["arm", "lang_kind", "touched"]).agg(questions=("lost", "size"), lost=("lost", "mean"))
            .round(3).reset_index())


# ---------------------------------------------------------------- 4. exposure on ordinary sentences
@lru_cache(maxsize=None)
def sentences(lang):
    """{concept: [token-id sets]} of every concept's sentences in `lang`."""
    import edit_coverage as ec
    ec.TOK = p2x.tokenizer()
    path = C.ROOT / "data" / "concept_sentences.json" if lang == "en" else p2x.DATA_V2 / f"concept_sentences_{lang}.json"
    out = {}
    for e in json.loads(Path(path).read_text()):
        out[e["concept"]] = [set(ec.tok_ids(s if isinstance(s, str) else s["sentence"])) for s in e["sentences"]]
    return out


def exposure(A):
    rows = []
    for a in A:
        for c in X13.concepts():
            if X13.no_edit(a, c):
                continue
            E = X13.edit_tokens(a, c)
            for L in LANGS:
                other = [t for k, v in sentences(L).items() if k != c for t in v]
                rows.append({"arm": a, "concept": c, "lang": L, "sentences": len(other),
                             "exposure": round(float(np.mean([bool(t & E) for t in other])), 4)})
    X = pd.DataFrame(rows)
    X.to_csv(OUT / "exposure.csv", index=False)
    piv = X.pivot_table(index="arm", columns="lang", values="exposure", aggfunc="mean").round(4)
    piv["foreign_mean"] = piv[FOR].mean(axis=1).round(4)
    piv["concepts"] = X.groupby("arm").concept.nunique()
    return piv.reset_index()


def main():
    A = arms()
    S, M = stats(A), means(A)
    Col, Exp = collateral(A), exposure(A)
    L_ = ["# X14: follow-up analyses of the fair run (held-out tests, 18 concepts)", "",
          f"Arms: {', '.join(A)}. Reliable cells only (pristine QA_test ≥ {X13.REL}). foreign_ret = mean retention "
          "over the reliable de/es/fr/ru cells (lower = more erased); leak = foreign_ret − ret_en. Definitions in the "
          "script docstring.", "",
          "## Means over the 18 concepts (no edit counts as retention 1)", "", p2x.md_table(M), "",
          "## 1–2. Paired differences: leak and foreign retention, all 18 concepts and only where both arms edit",
          "", "Two-level bootstrap 95% CI, Wilcoxon over concepts, Holm within each block of rows "
          "(same analysis label).", "", p2x.md_table(S), "",
          "## 3. Collateral on SimdomQA_test: share of pristine-correct questions lost, by whether the question "
          "contains an edited token", "", p2x.md_table(Col), "",
          "## 4. Exposure: share of other concepts' sentences that contain an edited token", "", p2x.md_table(Exp), ""]
    (OUT / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
