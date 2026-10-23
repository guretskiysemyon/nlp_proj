"""Supporting analyses of the extrapolation benchmark (run after analyze.py).

    .venv/bin/python -m experiments.paper_analysis.extrapolation.extra

1. Judge-free labels on the C* tokens of every concept (full coverage, no hand labels):
     exclusivity  excl(t) = log10[(c_page/N_page) / ((c_wiki + 0.5)/N_wiki)], positive if excl >= 2 (100x)
     lexicon      a source word in concept_words_v1 (20 words per concept)
   validity: AUC of excl for the blind S label (FULL concepts); P@10 and AUC of each score for the excl label.
2. Recall (FULL concepts): share of all S tokens among C* that each admission rule / top-k recovers.
3. Mechanism of proj ~ coef on CS: per run, Spearman(proj, coef) over new tokens and the concept feature's largest
   |cos| with another dictionary column (coherence), raw vs CS.
4. In-sample vs extrapolated precision, per concept (paired), overall and at page count 1, and the precision the
   in-sample frequency profile predicts for the new tokens (in-sample S share per page-count bin, weighted by the
   new admitted tokens' page-count distribution).
Output: results/paper_analysis/extrapolation/extra.json
"""
from __future__ import annotations

import csv
import json
import math
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from experiments.paper_analysis.extrapolation.analyze import CONDS, auc, boot_ci, fbin, load_labels, paired
from experiments.paper_analysis.extrapolation.common import CONCEPTS, DICTS, OUT, TOKDIR, load_fit, write_json
from experiments.paper_analysis.extrapolation.pool import FULL_LABEL
from experiments.paper_analysis.extrapolation.score import RULES, SCORES
from experiments.varwidth.common import slug

EXCL_T = 2.0


def excl(r: dict, n_page: int, n_wiki: int) -> float:
    return math.log10((r["c_page"] / n_page) / ((r["c_wiki"] + 0.5) / n_wiki)) if r["c_page"] else -np.inf


def main():
    labels = load_labels()
    X = {"excl_threshold": EXCL_T}
    ex_rows, rec_rows, mech_rows = [], [], []
    excl_valid = {}
    lex_counts = {}
    for c in CONCEPTS:
        td = json.loads((TOKDIR / f"{slug(c)}.json").read_text())
        toks, new = td["tokens"], td["new"]
        L = labels[c]
        R = pickle.load(open(OUT / "runs" / f"{slug(c)}.pkl", "rb"))
        pos_of = {t: i for i, t in enumerate(new)}
        g = td["conds"]["balanced"]["groups"]
        cst = np.array(sorted(int(t) for t, v in g.items() if v == "C*"))
        cidx = np.array([pos_of[t] for t in cst])
        ex = np.array([excl(toks[str(t)], td["n_page_tokens"], td["n_wiki_tokens"]) for t in cst])
        lex = np.array([toks[str(t)]["lex"] for t in cst])
        wcnt = np.array([toks[str(t)]["c_wiki"] for t in cst], dtype=float)
        lex_counts[c] = {"C*": len(cst), "lex": int(lex.sum()), "excl_pos": int((ex >= EXCL_T).sum()),
                         "lex_vstar_C": int(sum(toks[str(t)]["lex"] for t in td["vstar"] if toks[str(t)]["vcode"] == 1))}
        if c in FULL_LABEL:
            lab = np.array([L[int(t)] for t in cst])
            excl_valid[c] = {"auc_excl_for_S": auc(ex, lab == "S", lab != "S"),
                             "auc_excl_for_SA": auc(ex, lab != "U", lab == "U"),
                             "S_rate_excl_pos": float(np.mean(lab[ex >= EXCL_T] == "S")) if (ex >= EXCL_T).any() else np.nan,
                             "S_rate_excl_neg": float(np.mean(lab[ex < EXCL_T] == "S")),
                             "excl_pos_share": float(np.mean(ex >= EXCL_T)), "n_S": int((lab == "S").sum()),
                             "n_A": int((lab == "A").sum()), "n_C*": len(cst)}
        for (D, seed), r in R.items():
            if not r["F_C"]:
                continue
            row = {"concept": c, "dict": D, "seed": seed}
            for sc in SCORES:
                v = r["scores"][sc][cidx]
                o = np.argsort(-v, kind="stable")
                row[f"{sc}_exclP10"] = float(np.mean(ex[o[:10]] >= EXCL_T))
                row[f"{sc}_exclAUC"] = auc(v, ex >= EXCL_T, ex < EXCL_T)
                row[f"{sc}_lexAUC"] = auc(v, lex, ~lex) if lex.any() else np.nan
                row[f"{sc}_lex_in_top20"] = int(lex[o[:20]].sum())
                # rarity: does the score track how rare the token is in Wikipedia?
                row[f"{sc}_rho_rarity"] = float(spearmanr(v, -np.log10(wcnt + 0.5))[0])
            row["base_excl"] = float(np.mean(ex >= EXCL_T))
            ex_rows.append(row)
            # recall (FULL concepts): S tokens among C* recovered by each rule / top-k
            if c in FULL_LABEL:
                lab = np.array([L[int(t)] for t in cst])
                nS = int((lab == "S").sum())
                if nS:
                    rr = {"concept": c, "dict": D, "seed": seed, "n_S_Cstar": nS, "n_Cstar": len(cst)}
                    for sc in SCORES:    # channelling test: does the score separate S from A (adjacent) tokens?
                        rr[f"aucSvsA_{sc}"] = auc(r["scores"][sc][cidx], lab == "S", lab == "A")
                    cd = r["cond"]["balanced"]
                    nid = np.array(cd["new_ids"])
                    for rule in RULES:
                        adm = set(int(t) for t in nid[cd["admit_new"][rule]])
                        rr[f"recall_{rule}"] = sum(1 for t, l in zip(cst, lab) if l == "S" and int(t) in adm) / nS
                    for sc in SCORES:
                        o = np.argsort(-r["scores"][sc][cidx], kind="stable")
                        for k in (10, 20, 50, 100):
                            rr[f"recall_{sc}@{k}"] = float((lab[o[:k]] == "S").sum() / nS)
                    rec_rows.append(rr)
            # mechanism: proj vs coef agreement and the concept feature's coherence with the dictionary
            fit = load_fit(c, DICTS[D][0], seed)
            Z = fit["Z"].double()
            Zn = Z / Z.norm(dim=0, keepdim=True)
            C = (Zn.T @ Zn).abs()
            C.fill_diagonal_(0)
            mech_rows.append({"concept": c, "dict": D, "seed": seed,
                              "spearman_proj_coef": float(spearmanr(r["scores"]["proj"], r["scores"]["coef"])[0]),
                              "spearman_proj_coef_Cstar": float(spearmanr(r["scores"]["proj"][cidx],
                                                                          r["scores"]["coef"][cidx])[0]),
                              "maxcos_F": float(np.mean([C[f].max().item() for f in r["F_C"]])),
                              "mean_top5cos_F": float(np.mean([C[f].topk(5).values.mean().item() for f in r["F_C"]])),
                              "mean_abs_cos_all": float(C.sum() / (C.shape[0] * (C.shape[0] - 1)))})
    X["lexicon_excl_counts"] = lex_counts
    X["excl_validity"] = excl_valid
    X["excl_validity_mean"] = {k: float(np.nanmean([v[k] for v in excl_valid.values()]))
                               for k in ("auc_excl_for_S", "auc_excl_for_SA", "S_rate_excl_pos", "S_rate_excl_neg",
                                         "excl_pos_share")}
    # judge-free ranking table (concept means) and tests
    def pc(rows, met, D):
        acc = defaultdict(list)
        for r in rows:
            if r["dict"] == D and not (isinstance(r[met], float) and np.isnan(r[met])):
                acc[r["concept"]].append(r[met])
        return {k: float(np.mean(v)) for k, v in acc.items()}
    jf = {}
    for D in DICTS:
        for sc in SCORES:
            for met in ("exclP10", "exclAUC", "lexAUC", "lex_in_top20", "rho_rarity"):
                v = list(pc(ex_rows, f"{sc}_{met}", D).values())
                jf[f"{D}|{sc}|{met}"] = {"mean": float(np.mean(v)) if v else np.nan, "ci": boot_ci(v), "n": len(v)}
        jf[f"{D}|base_excl"] = float(np.mean(list(pc(ex_rows, "base_excl", D).values())))
    X["judge_free"] = jf
    jt = {}
    for D in DICTS:
        for a, b in (("coef", "proj"), ("share", "coef"), ("share", "proj")):
            for met in ("exclP10", "exclAUC"):
                jt[f"{D}|{met}|{a}-{b}"] = paired(pc(ex_rows, f"{a}_{met}", D), pc(ex_rows, f"{b}_{met}", D))
    for sc in SCORES:
        for met in ("exclP10", "exclAUC", "lexAUC", "rho_rarity"):
            jt[f"{sc}|{met}|CINDER-EMBER"] = paired(pc(ex_rows, f"{sc}_{met}", "CINDER"),
                                                   pc(ex_rows, f"{sc}_{met}", "EMBER"))
    X["judge_free_tests"] = jt
    # recall
    rc = {}
    for D in DICTS:
        for k in [k for k in rec_rows[0] if k.startswith(("recall_", "aucSvsA_"))]:
            v = list(pc(rec_rows, k, D).values())
            rc[f"{D}|{k}"] = {"mean": float(np.mean(v)) if v else np.nan, "n": len(v)}
    X["recall_full_concepts"] = rc
    X["channelling_SvsA_tests"] = {f"{D}|{a}-{b}": paired(pc(rec_rows, f"aucSvsA_{a}", D), pc(rec_rows, f"aucSvsA_{b}", D))
                                   for D in DICTS for a, b in (("coef", "proj"), ("share", "proj"), ("share", "coef"))}
    X["n_S_Cstar_full"] = {c: v["n_S"] for c, v in excl_valid.items()}
    # mechanism
    mech = {}
    for D in DICTS:
        for met in ("spearman_proj_coef", "spearman_proj_coef_Cstar", "maxcos_F", "mean_top5cos_F",
                    "mean_abs_cos_all"):
            v = list(pc(mech_rows, met, D).values())
            mech[f"{D}|{met}"] = {"mean": float(np.mean(v)), "ci": boot_ci(v), "n": len(v)}
    for met in ("spearman_proj_coef_Cstar", "maxcos_F"):
        mech[f"test|{met}|CINDER-EMBER"] = paired(pc(mech_rows, met, "CINDER"), pc(mech_rows, met, "EMBER"))
        mech[f"test|{met}|CS-EMBER"] = paired(pc(mech_rows, met, "CS"), pc(mech_rows, met, "EMBER"))
    X["mechanism"] = mech
    # in-sample vs extrapolated precision (freq.csv from analyze.py)
    fr = list(csv.DictReader(open(OUT / "freq.csv")))
    for r in fr:
        r["c_page"] = int(r["c_page"])
        r["S"] = r["label"] == "S"
    ins = {}
    for D in DICTS:
        for kind in ("in-sample support", "new admitted (frac)", "new admitted (cutoff)"):
            for sel, name in ((lambda r: True, "all"), (lambda r: r["c_page"] == 1, "c1")):
                acc = defaultdict(list)
                for r in fr:
                    if r["dict"] == D and r["kind"] == kind and sel(r):
                        acc[r["concept"]].append(r["S"])
                ins[f"{D}|{kind}|{name}"] = {k: float(np.mean(v)) for k, v in acc.items()}
        # predicted precision of new admitted tokens from the in-sample frequency profile (per concept)
        pred = {}
        for c in CONCEPTS:
            prof = defaultdict(list)
            for r in fr:
                if r["dict"] == D and r["kind"] == "in-sample support" and r["concept"] == c:
                    prof[fbin(r["c_page"])].append(r["S"])
            new_bins = [fbin(r["c_page"]) for r in fr if r["dict"] == D and r["kind"] == "new admitted (frac)"
                        and r["concept"] == c]
            vals = [np.mean(prof[b]) for b in new_bins if prof.get(b)]
            if vals:
                pred[c] = float(np.mean(vals))
        ins[f"{D}|predicted from in-sample profile|all"] = pred
    tests = {}
    for D in DICTS:
        tests[f"{D}|in-sample - new(frac)|all"] = paired(ins[f"{D}|in-sample support|all"],
                                                         ins[f"{D}|new admitted (frac)|all"])
        tests[f"{D}|in-sample - new(frac)|c1"] = paired(ins[f"{D}|in-sample support|c1"],
                                                        ins[f"{D}|new admitted (frac)|c1"])
        tests[f"{D}|predicted - new(frac)|all"] = paired(ins[f"{D}|predicted from in-sample profile|all"],
                                                         ins[f"{D}|new admitted (frac)|all"])
    X["insample_vs_new"] = {"per_concept": ins, "tests": tests}
    # page-count distribution: new page tokens vs in-sample support vs all concept-only V* tokens
    dist = {}
    for c in CONCEPTS:
        td = json.loads((TOKDIR / f"{slug(c)}.json").read_text())
        toks = td["tokens"]
        newp = [toks[str(t)]["c_page"] for t in td["new"] if toks[str(t)]["c_page"] > 0]
        vC = [toks[str(t)]["c_page"] for t in td["vstar"] if toks[str(t)]["vcode"] == 1]
        dist[c] = {"new_page_c1_share": float(np.mean(np.array(newp) == 1)),
                   "vstarC_c1_share": float(np.mean(np.array(vC) == 1)),
                   "new_page_median": float(np.median(newp)), "vstarC_median": float(np.median(vC)),
                   "n_new_page": len(newp), "n_vstarC": len(vC)}
    X["page_count_distribution"] = dist
    X["page_count_distribution_mean"] = {k: float(np.mean([d[k] for d in dist.values()]))
                                         for k in ("new_page_c1_share", "vstarC_c1_share", "new_page_median",
                                                   "vstarC_median")}
    for name, rr in (("judge_free_runs.csv", ex_rows), ("recall_runs.csv", rec_rows), ("mechanism_runs.csv", mech_rows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rr for k in r)), restval="")
            w.writeheader()
            w.writerows(rr)
    write_json(OUT / "extra.json", X)
    print(f"-> {OUT / 'extra.json'}")


if __name__ == "__main__":
    main()
