"""Metrics, paired tests and tables of the extrapolation benchmark (after tokens -> embrows -> run -> pool -> labels).

    .venv/bin/python -m experiments.paper_analysis.extrapolation.analyze

Ground truth per token (labels/key.json + labels_XX.tsv, blind, rubric.txt): S concept-specific, A adjacent,
U unrelated; new neutral tokens (N*) are U by construction. "prec" = share of S (strict); "precSA" = share of S or A.

Per run (concept x dictionary x seed with a non-empty F_C) and neutral-ratio condition:
    P@k      for each score (proj, cos, coef, share), k = 1..POOL_K, over the condition's new tokens (C*, B*, N*);
             exact because every new page token in a score's top-POOL_K is labelled
    AUC      S vs rest and S+A vs U over all new tokens (FULL_LABEL concepts only: every new page token labelled)
    admit    for each admission rule: #new admitted, #C* admitted (= EMBER's edit-set growth), precision on the
             admitted tokens, N* share, U share (U page tokens + N*), retention of the saved support
Per concept: mean over its runs; paired comparisons use concepts as units (Wilcoxon signed-rank, bootstrap 95% CI).
Outputs: results/paper_analysis/extrapolation/{runs_metrics.csv, admit.csv, summary.json, freq.csv}
"""
from __future__ import annotations

import csv
import glob
import json
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from experiments.paper_analysis.extrapolation.common import CONCEPTS, DICTS, OUT, TOKDIR, write_json
from experiments.paper_analysis.extrapolation.pool import FULL_LABEL, POOL_K
from experiments.paper_analysis.extrapolation.score import RULES, SCORES
from experiments.varwidth.common import slug

LAB = OUT / "labels"
KS = list(range(1, POOL_K + 1))
CONDS = ("balanced", "natural")
FBINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 10 ** 9, "10+")]


def load_labels() -> dict:
    """{concept: {tid: 'S'|'A'|'U'}} for every labelled token."""
    key = json.loads((LAB / "key.json").read_text())
    lab = {}
    for p in sorted(glob.glob(str(LAB / "labels_[0-9]*.tsv"))):
        for line in open(p):
            if line.strip():
                i, l = line.strip().split("\t")
                lab[i] = l
    out = defaultdict(dict)
    for i, k in key.items():
        out[k["concept"]][int(k["tid"])] = lab[i]
    return out


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def paired(a: dict, b: dict) -> dict:
    """a, b: {concept: value}; Wilcoxon on concepts present (non-NaN) in both."""
    cs = [c for c in a if c in b and not np.isnan(a[c]) and not np.isnan(b[c])]
    x, y = np.array([a[c] for c in cs]), np.array([b[c] for c in cs])
    d = x - y
    r = {"n": len(cs), "mean_a": float(x.mean()) if len(cs) else np.nan,
         "mean_b": float(y.mean()) if len(cs) else np.nan, "mean_diff": float(d.mean()) if len(cs) else np.nan,
         "ci_diff": boot_ci(d), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    try:
        r["p_wilcoxon"] = float(wilcoxon(x, y).pvalue) if len(cs) >= 5 and np.any(d != 0) else np.nan
    except ValueError:
        r["p_wilcoxon"] = np.nan
    return r


def auc(score, pos, neg) -> float:
    s_p, s_n = score[pos], score[neg]
    if len(s_p) == 0 or len(s_n) == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([s_p, s_n]))
    return float((r[:len(s_p)].sum() - len(s_p) * (len(s_p) + 1) / 2) / (len(s_p) * len(s_n)))


def fbin(c: int) -> str:
    for lo, hi, name in FBINS:
        if lo <= c <= hi:
            return name
    return "0"


def main():
    labels = load_labels()
    key = json.loads((LAB / "key.json").read_text())
    rows, adm_rows, freq_rows = [], [], []
    base = {}
    for c in CONCEPTS:
        td = json.loads((TOKDIR / f"{slug(c)}.json").read_text())
        toks, new, vstar = td["tokens"], td["new"], td["vstar"]
        L = labels[c]
        R = pickle.load(open(OUT / "runs" / f"{slug(c)}.pkl", "rb"))
        pos_of = {t: i for i, t in enumerate(new)}
        # base rates: S / SA among each condition's new tokens (FULL: exact; else page-new random sample x page share)
        for cond in CONDS:
            g = td["conds"][cond]["groups"]
            n_new = len(g)
            n_page = sum(1 for v in g.values() if v != "N*")
            if c in FULL_LABEL:
                labs = [L[int(t)] for t, v in g.items() if v != "N*"]
            else:   # the random sample only (uniform over new page tokens)
                rs = {int(k["tid"]) for k in key.values() if k["concept"] == c and "random" in k["sets"]}
                labs = [L[t] for t in rs]
            pS = np.mean([x == "S" for x in labs]) * n_page / n_new
            pSA = np.mean([x in "SA" for x in labs]) * n_page / n_new
            cst = {int(t) for t, v in g.items() if v == "C*"}
            if c in FULL_LABEL:
                labc = [L[t] for t in cst]
            else:
                labc = [L[t] for t in rs if t in cst]
            base[(c, cond)] = {"S": float(pS), "SA": float(pSA), "n_new": n_new, "n_page_new": n_page,
                               "cS": float(np.mean([x == "S" for x in labc])),
                               "cSA": float(np.mean([x != "U" for x in labc])), "n_Cstar": len(cst)}
        for (D, seed), r in R.items():
            if not r["F_C"]:
                continue
            for cond in CONDS:
                g = td["conds"][cond]["groups"]
                ids = np.array(sorted(int(t) for t in g))
                idx = np.array([pos_of[t] for t in ids])
                grp = np.array([g[str(t)] for t in ids])
                lab = np.array([("U" if grp[j] == "N*" else L.get(int(t), "?")) for j, t in enumerate(ids)])
                row = {"concept": c, "dict": D, "seed": seed, "cond": cond, "n_F": len(r["F_C"]),
                       "n_new": len(ids), "n_Cstar": int((grp == "C*").sum()), "n_Nstar": int((grp == "N*").sum()),
                       "base_S": base[(c, cond)]["S"], "base_SA": base[(c, cond)]["SA"],
                       "base_cS": base[(c, cond)]["cS"], "base_cSA": base[(c, cond)]["cSA"]}
                for sc in SCORES:
                    v = r["scores"][sc][idx]
                    order = np.argsort(-v, kind="stable")[:POOL_K]
                    top = lab[order]
                    assert "?" not in top, (c, D, seed, cond, sc)
                    for k in KS:
                        row[f"{sc}_P{k}"] = float(np.mean(top[:k] == "S"))
                        row[f"{sc}_PSA{k}"] = float(np.mean(np.isin(top[:k], ["S", "A"])))
                    row[f"{sc}_Nstar10"] = float(np.mean(grp[order[:10]] == "N*"))
                    if c in FULL_LABEL:
                        assert "?" not in lab
                        row[f"{sc}_AUC"] = auc(v, lab == "S", lab != "S")
                        row[f"{sc}_AUCSA"] = auc(v, lab != "U", lab == "U")
                    # edit candidates only: the condition's C* tokens (concept text only; EMBER edits only these)
                    cm = grp == "C*"
                    vc, lc = v[cm], lab[cm]
                    oc = np.argsort(-vc, kind="stable")[:POOL_K]
                    tc = lc[oc]
                    for k in KS:
                        tk = tc[:k][tc[:k] != "?"]                      # pooled concepts: unlabelled tail is rare
                        row[f"{sc}_cP{k}"] = float(np.mean(tk == "S")) if len(tk) else np.nan
                        row[f"{sc}_cPSA{k}"] = float(np.mean(tk != "U")) if len(tk) else np.nan
                    row[f"{sc}_cmiss20"] = int((tc == "?").sum())
                    if c in FULL_LABEL:
                        row[f"{sc}_cAUC"] = auc(vc, lc == "S", lc != "S")
                        row[f"{sc}_cAUCSA"] = auc(vc, lc != "U", lc == "U")
                rows.append(row)
                cd = r["cond"][cond]
                nid = np.array(cd["new_ids"])
                g_n = np.array([g[str(t)] for t in nid])
                for rule in RULES:
                    m = cd["admit_new"][rule]
                    a_ids, a_g = nid[m], g_n[m]
                    a_lab = np.array([("U" if gg == "N*" else L.get(int(t), "?")) for t, gg in zip(a_ids, a_g)])
                    assert "?" not in a_lab, (c, D, seed, cond, rule)
                    n_a = int(m.sum())
                    adm_rows.append({
                        "concept": c, "dict": D, "seed": seed, "cond": cond, "rule": rule, "n_admit": n_a,
                        "n_admit_Cstar": int((a_g == "C*").sum()),
                        "prec": float(np.mean(a_lab == "S")) if n_a else np.nan,
                        "precSA": float(np.mean(a_lab != "U")) if n_a else np.nan,
                        "prec_Cstar": float(np.mean(a_lab[a_g == "C*"] == "S")) if (a_g == "C*").any() else np.nan,
                        "nS": int((a_lab == "S").sum()), "nA": int((a_lab == "A").sum()),
                        "nU_page": int(((a_lab == "U") & (a_g != "N*")).sum()), "nNstar": int((a_g == "N*").sum()),
                        "Nstar_share": float(np.mean(a_g == "N*")) if n_a else np.nan,
                        "U_share": float(np.mean(a_lab == "U")) if n_a else np.nan,
                        "retain": cd["retain"][rule], "n_ext": cd["n_ext"], "n_vstar": r["n"],
                        "width_orig": int(np.sum(r["widths"])), "width_frac": int(np.sum(cd["width_frac"]))})
            # frequency: in-sample support tokens (labelled sample) of this run, by page count
            sup = [vstar[i] for i in np.nonzero(r["support_vstar"])[0] if toks[str(vstar[i])]["vcode"] == 1]
            for t in sup:
                if t in L:
                    freq_rows.append({"concept": c, "dict": D, "seed": seed, "kind": "in-sample support",
                                      "tid": t, "c_page": toks[str(t)]["c_page"], "label": L[t]})
            cd = r["cond"]["balanced"]
            for rule in ("frac", "cutoff"):
                for t in np.array(cd["new_ids"])[cd["admit_new"][rule]]:
                    t = int(t)
                    if toks[str(t)]["c_page"] > 0:
                        freq_rows.append({"concept": c, "dict": D, "seed": seed, "kind": f"new admitted ({rule})",
                                          "tid": t, "c_page": toks[str(t)]["c_page"], "label": L[t]})
        # all new page tokens of FULL concepts, and the random sample elsewhere: base rate by page count
        sample = {int(k["tid"]) for k in key.values() if k["concept"] == c and
                  ("full" in k["sets"] if c in FULL_LABEL else "random" in k["sets"])}
        for t in sample:
            freq_rows.append({"concept": c, "dict": "-", "seed": -1, "kind": "new page (all)", "tid": t,
                              "c_page": toks[str(t)]["c_page"], "label": L[t]})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rr in (("runs_metrics.csv", rows), ("admit.csv", adm_rows), ("freq.csv", freq_rows)):
        with open(OUT / name, "w", newline="") as f:
            fields = list(dict.fromkeys(k for r in rr for k in r))
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rr)
    summarize(rows, adm_rows, freq_rows, base)


def per_concept(rows, keyf, val, filt=lambda r: True) -> dict:
    """{group: {concept: mean over runs}}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if filt(r) and val in r and not (isinstance(r[val], float) and np.isnan(r[val])):
            acc[keyf(r)][r["concept"]].append(r[val])
    return {g: {c: float(np.mean(v)) for c, v in d.items()} for g, d in acc.items()}


def summarize(rows, adm_rows, freq_rows, base):
    S = {"n_concepts": len(CONCEPTS), "concepts": CONCEPTS, "full_label": FULL_LABEL}
    # run counts
    S["runs"] = {D: sum(1 for r in rows if r["dict"] == D and r["cond"] == "balanced") for D in DICTS}
    # token counts
    S["tokens"] = {c: {cond: {k: base[(c, cond)][k] for k in ("n_new", "n_page_new", "n_Cstar", "S", "SA", "cS", "cSA")} for cond in CONDS}
                   for c in CONCEPTS}
    # P@k and AUC per (dict, score, cond): concept means with CI
    tab = {}
    for cond in CONDS:
        for D in DICTS:
            for sc in SCORES:
                f = lambda r, D=D, cond=cond: r["dict"] == D and r["cond"] == cond  # noqa: E731
                ent = {}
                for met in [f"{sc}_{p}P{k}" for p in ("", "c") for k in (1, 5, 10, 20)] + \
                        [f"{sc}_{p}PSA{k}" for p in ("", "c") for k in (5, 10, 20)] + \
                        [f"{sc}_{p}{a}" for p in ("", "c") for a in ("AUC", "AUCSA")] + [f"{sc}_Nstar10"]:
                    pc = per_concept(rows, lambda r: 0, met, f).get(0, {})
                    v = list(pc.values())
                    ent[met.split("_", 1)[1]] = {"mean": float(np.mean(v)) if v else np.nan, "ci": boot_ci(v),
                                                 "n": len(v)}
                tab[f"{cond}|{D}|{sc}"] = ent
    S["ranking"] = tab
    S["base_rate"] = {cond: {m: float(np.mean([base[(c, cond)][m] for c in CONCEPTS])) for m in ("S", "SA", "cS", "cSA")}
                      for cond in CONDS}
    # paired tests: scores within dictionary, dictionaries within score (P@10 strict, AUC)
    tests = {}
    for cond in CONDS:
        for met in ("P10", "P5", "PSA10", "AUC", "cP10", "cP5", "cPSA10", "cAUC"):
            for D in DICTS:
                pcs = {sc: per_concept(rows, lambda r: 0, f"{sc}_{met}",
                                       lambda r, D=D, cond=cond: r["dict"] == D and r["cond"] == cond).get(0, {})
                       for sc in SCORES}
                for a, b in (("coef", "proj"), ("share", "proj"), ("share", "coef"), ("cos", "proj"),
                             ("coef", "cos")):
                    tests[f"{cond}|{met}|{D}|{a}-{b}"] = paired(pcs[a], pcs[b])
            for sc in SCORES:
                pcs = {D: per_concept(rows, lambda r: 0, f"{sc}_{met}",
                                      lambda r, D=D, cond=cond: r["dict"] == D and r["cond"] == cond).get(0, {})
                       for D in DICTS}
                for a, b in (("CINDER", "EMBER"), ("CS", "EMBER"), ("CINDER", "CS")):
                    tests[f"{cond}|{met}|{sc}|{a}-{b}"] = paired(pcs[a], pcs[b])
        # balanced vs natural (same dict/score)
    for met in ("P10", "AUC", "cP10"):
        for D in DICTS:
            for sc in SCORES:
                a = per_concept(rows, lambda r: 0, f"{sc}_{met}",
                                lambda r, D=D: r["dict"] == D and r["cond"] == "balanced").get(0, {})
                b = per_concept(rows, lambda r: 0, f"{sc}_{met}",
                                lambda r, D=D: r["dict"] == D and r["cond"] == "natural").get(0, {})
                tests[f"bal-nat|{met}|{D}|{sc}"] = paired(a, b)
    S["tests"] = tests
    # admission rules: concept means
    adm = {}
    for cond in CONDS:
        for D in DICTS:
            for rule in RULES:
                f = lambda r, D=D, rule=rule, cond=cond: r["dict"] == D and r["rule"] == rule and r["cond"] == cond  # noqa
                ent = {}
                for met in ("n_admit", "n_admit_Cstar", "prec", "precSA", "prec_Cstar", "Nstar_share", "U_share",
                            "retain", "nS", "nA", "nU_page", "nNstar", "width_frac", "width_orig", "n_ext"):
                    pc = per_concept(adm_rows, lambda r: 0, met, f).get(0, {})
                    v = list(pc.values())
                    ent[met] = {"mean": float(np.mean(v)) if v else np.nan, "ci": boot_ci(v), "n": len(v)}
                # pooled precision (sum S / sum admitted over all runs of the concept, then concept mean)
                pooled = defaultdict(lambda: [0, 0, 0])
                for r in adm_rows:
                    if f(r):
                        pooled[r["concept"]][0] += r["nS"]
                        pooled[r["concept"]][1] += r["nS"] + r["nA"]
                        pooled[r["concept"]][2] += r["n_admit"]
                pp = [s / n for s, _, n in pooled.values() if n]
                ent["prec_pooled"] = {"mean": float(np.mean(pp)) if pp else np.nan, "ci": boot_ci(pp), "n": len(pp)}
                ent["S_per_run"] = {"mean": float(np.mean([r["nS"] for r in adm_rows if f(r)]))}
                adm[f"{cond}|{D}|{rule}"] = ent
    S["admission"] = adm
    atests = {}
    for cond in CONDS:
        for D in DICTS:
            for met in ("prec", "U_share", "nS", "n_admit"):
                pcs = {rule: per_concept(adm_rows, lambda r: 0, met,
                                         lambda r, D=D, rule=rule, cond=cond: r["dict"] == D and r["rule"] == rule
                                         and r["cond"] == cond).get(0, {}) for rule in RULES}
                for a, b in (("width", "frac"), ("cutoff", "frac"), ("argmax", "frac"), ("share50", "frac"),
                             ("argmax", "cutoff")):
                    atests[f"{cond}|{D}|{met}|{a}-{b}"] = paired(pcs[a], pcs[b])
        for rule in RULES:
            for met in ("prec", "nS", "U_share"):
                pcs = {D: per_concept(adm_rows, lambda r: 0, met,
                                      lambda r, D=D, rule=rule, cond=cond: r["dict"] == D and r["rule"] == rule
                                      and r["cond"] == cond).get(0, {}) for D in DICTS}
                atests[f"{cond}|{rule}|{met}|CINDER-EMBER"] = paired(pcs["CINDER"], pcs["EMBER"])
    S["admission_tests"] = atests
    # frequency
    fr = {}
    kinds = sorted({r["kind"] for r in freq_rows})
    for kind in kinds:
        for D in ["-"] + list(DICTS):
            sub = [r for r in freq_rows if r["kind"] == kind and r["dict"] == D]
            if not sub:
                continue
            for _, _, b in FBINS:
                s2 = [r for r in sub if fbin(r["c_page"]) == b]
                # per-concept mean of the S share, over concepts with >= 1 token in the bin
                pc = defaultdict(list)
                for r in s2:
                    pc[r["concept"]].append(r["label"] == "S")
                v = [float(np.mean(x)) for x in pc.values()]
                fr[f"{kind}|{D}|{b}"] = {"n_tokens": len(s2), "n_concepts": len(v),
                                         "S_token": float(np.mean([r["label"] == "S" for r in s2])) if s2 else np.nan,
                                         "SA_token": float(np.mean([r["label"] != "U" for r in s2])) if s2 else np.nan,
                                         "S_concept_mean": float(np.mean(v)) if v else np.nan, "ci": boot_ci(v)}
    S["frequency"] = fr
    write_json(OUT / "summary.json", S)
    print(f"-> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
