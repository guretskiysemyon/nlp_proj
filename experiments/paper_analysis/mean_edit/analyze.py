#!/usr/bin/env python
"""WP6b mean_edit: analysis of the grids, copy curves, selected points, integrity tests and floors (CPU).

    python -m experiments.paper_analysis.mean_edit.analyze      # -> results/paper_analysis/mean_edit/analysis.json

Reads the existing collateral results (CINDER, CINDER-z, EMBER, Noise) and this WP's outputs. Unit = run
(concept-seed). Statistics: paired mean differences over runs, bootstrap 95% CI (10,000 resamples of runs, seed 0),
Wilcoxon signed-rank (two-sided, zero differences dropped), wins / losses.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

from experiments.paper_analysis.mean_edit.run import (COLL, INTEG, IOUT_OLD, OUT, SWEEP, grid_rows, points, save,
                                                      vdir)
from experiments.paper_analysis.mean_edit.sets import CINDER_RUNS, EMBER_RUNS, new_sets
from experiments.varwidth.common import slug

R = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
R1 = [1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
DR = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
FAMS = {"C": {"runs": CINDER_RUNS, "cen": "sub:cinder", "unc": "sub:cinder_z", "mean": "sub:mean_c",
              "dmean": "sub:mean_cd", "noise": "noise:cinder", "parent_set": "cinder"},
        "E": {"runs": EMBER_RUNS, "cen": "sub:ember_c", "unc": "sub:ember", "mean": "sub:mean_e",
              "dmean": "sub:mean_ed", "noise": "noise:ember", "parent_set": "ember"}}
NAMES = {"sub:cinder": "CINDER", "sub:cinder_z": "CINDER-z", "sub:mean_c": "MEAN-C", "sub:mean_cd": "MEAN-Cd",
         "noise:cinder": "Noise-C", "sub:ember": "EMBER", "sub:ember_c": "EMBER-c", "sub:mean_e": "MEAN-E",
         "sub:mean_ed": "MEAN-Ed", "noise:ember": "Noise-E"}


def rk(c, s):
    return f"{c}|{s}"


# ------------------------------------------------------------------------------------------------ statistics
def paired(a, b, n_boot=10000):
    """Paired a - b over runs (lists aligned; NaN pairs dropped)."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    if not len(d):
        return {"n": 0}
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    nz = d[np.abs(d) > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 1 else float("nan")
    return {"n": int(len(d)), "mean_a": float(a[ok].mean()), "mean_b": float(b[ok].mean()), "diff": float(d.mean()),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "wins": int((d > 1e-12).sum()),
            "losses": int((d < -1e-12).sum()), "p": p}


# ------------------------------------------------------------------------------------------------ loaders
def val(c, s, v):
    """{param: row} of the validation grid."""
    rows = grid_rows(c, s, v)
    return {} if not rows else {float(x["param"]): x for x in rows}


def copygrid(c, s, v):
    f = vdir(c, s, v) / "copygrid.json"
    return {} if not f.exists() else {float(x["param"]): x for x in json.loads(f.read_text()).values()}


def unedited_test():
    from experiments.varwidth.collateral_report import unedited_test_acc
    return unedited_test_acc()


def renorm(res, ids, base):
    from experiments.varwidth.erase import hm, mc_norm
    ids = {str(t) for t in ids} if res.get("n_edit") else set()
    vals = [mc_norm(a, base[t]) for t, a in res["coll_tok"].items() if t in ids and base.get(t, 0) > 0.25]
    res["collateral_rn"] = float(np.mean(vals)) if vals else 1.0
    res["H3v_rn"] = hm([res["efficacy"], res["specificity"], res["collateral_rn"]])
    return res


def test_result(c, s, v, rule):
    """Test pass at the rule's point ('sel' or 'e3'); collateral renormalised (amendment 6)."""
    from experiments.paper_analysis.mean_edit.run import is_new
    if rule == "sel" and not is_new(v):
        f = vdir(c, s, v, mine=False) / "result.json"
    else:
        f = vdir(c, s, v) / f"result_{rule}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    from experiments.varwidth.collateral import edit_sets
    ids = edit_sets(c, s)[v.split(":")[1] if v.split(":")[1] in ("cinder", "cinder_z", "ember") else
                          {"mean_c": "cinder", "ember_c": "ember", "mean_e": "ember"}[v.split(":")[1]]]["ids"]
    return renorm(r, ids, unedited_test().get(c, {}))


# ------------------------------------------------------------------------------------------------ integrity
_REFS = {}


def refs(c, s):
    k = (c, s)
    if k not in _REFS:
        d = IOUT_OLD / slug(c) / f"s{s}"
        _REFS[k] = {n: json.loads((d / f"ref_{n}.json").read_text()) for n in ("orig", "nonce", "del")}
    return _REFS[k]


def integ_file(c, s, v, par, rule=None):
    """The E2 file of variant v at param par (new folder first, then the existing E2 folder)."""
    cands = []
    nm = v.replace(":", "_")
    for d in (INTEG / slug(c) / f"s{s}", IOUT_OLD / slug(c) / f"s{s}"):
        if not d.exists():
            continue
        for f in d.glob(f"{nm}@*.json"):
            cands.append(f)
    for f in cands:
        j = json.loads(f.read_text())
        if j.get("param") is not None and par is not None and abs(float(j["param"]) - float(par)) < 1e-12:
            return j
    return None


def integ_summary(c, s, j, words=None):
    """Judge-free E2 summary of one config on `words` (default: its own words); refs on the same words."""
    from experiments.varwidth.integrity import COPY, CTX
    R_ = refs(c, s)
    ws = [w for w in (words or j["words"]) if w in j["recs"]]
    if not ws:
        return None
    recs = j["recs"]

    def m(key, src):
        v = [x for w in ws for x in (src[w][key] if isinstance(src[w][key], list) else [src[w][key]])]
        return float(np.mean(v))
    exc = [recs[w]["nll"][k] - R_["nonce"][w]["nll"][k] for w in ws for k in range(len(CTX))]
    norm = lambda x: re.sub(r"\s+", " ", x).strip().strip('"').strip()  # noqa: E731
    dl = [norm(recs[w]["copy_out"][i]) == norm(R_["del"][w]["copy_out"][i]) for w in ws for i in range(len(COPY))]
    return {"n_words": len(ws), "copy": m("copy", recs), "ctx_pass": m("recall", recs), "ctxmc": m("ctxmc", recs),
            "ctxmc_orig": m("ctxmc", R_["orig"]), "ctxmc_nonce": m("ctxmc", R_["nonce"]),
            "ctx_excess": float(np.mean(exc)), "echo": m("echo", recs), "del_likeness": float(np.mean(dl)),
            "z_gt3": j["static"]["z_gt3"], "nn_cos": j["static"]["nn_cos_median"], "r": j["r"]}


# ------------------------------------------------------------------------------------------------ floors
def floors():
    """{run key: {set: {split: {nonce, del, q_touched}}}} as efficacy (1 - mc_norm(acc, unedited acc))."""
    from experiments.varwidth.erase import mc_norm
    out = {}
    for c in ("Harry Potter", "Ancient Rome", "Valentine's Day"):
        f = OUT / "floors" / f"{slug(c)}.json"
        if not f.exists():
            continue
        fl = json.loads(f.read_text())
        for k, x in fl.items():
            if k.startswith("unedited"):
                continue
            s, st, split = k.split("|")
            base = fl[f"unedited|{split}"]
            out.setdefault(rk(c, int(s)), {}).setdefault(st, {})[split] = {
                "nonce": 1 - mc_norm(x["nonce_acc"], base), "del": 1 - mc_norm(x["del_acc"], base),
                "q_touched": x["q_touched"]}
    return out


# ------------------------------------------------------------------------------------------------ analyses
def curves(fam):
    """Validation efficacy / collateral / specificity / H3v per variant, run and r."""
    F = FAMS[fam]
    out = {}
    for v in (F["cen"], F["unc"], F["mean"], F["noise"]):
        for c, s in F["runs"]:
            g = val(c, s, v)
            for p, x in g.items():
                out.setdefault(v, {}).setdefault(rk(c, s), {})[p] = {
                    k: x.get(k) for k in ("r", "efficacy", "collateral", "specificity", "mmlu_frac")}
                from experiments.varwidth.erase import hm
                out[v][rk(c, s)][p]["H3v"] = hm([x["efficacy"], x["specificity"], x.get("collateral", 0.0)])
    return out


def mean_at(cur, v, runs, p, key="efficacy"):
    vals = [cur.get(v, {}).get(rk(c, s), {}).get(p, {}).get(key) for c, s in runs]
    vals = [x for x in vals if x is not None]
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def c1(fam, cur):
    """Fixed r: gain of the uncentred edit over the centred one vs the mean edit alone (validation)."""
    F = FAMS[fam]
    runs = F["runs"]
    per_r = {}
    for p in R:
        row = {}
        for v in (F["cen"], F["unc"], F["mean"], F["noise"]):
            for key in ("efficacy", "collateral", "H3v"):
                row[f"{NAMES[v]}|{key}"] = mean_at(cur, v, runs, p, key)[0]
        a = [cur[F["unc"]][rk(c, s)][p]["efficacy"] for c, s in runs if p in cur.get(F["unc"], {}).get(rk(c, s), {})]
        row["gain"] = row[f"{NAMES[F['unc']]}|efficacy"] - row[f"{NAMES[F['cen']]}|efficacy"]
        row["n"] = len(a)
        per_r[p] = row
    # pooled over R1, per run
    G, M, Nz = [], [], []
    for c, s in runs:
        k = rk(c, s)
        try:
            g_ = np.mean([cur[F["unc"]][k][p]["efficacy"] - cur[F["cen"]][k][p]["efficacy"] for p in R1])
            m_ = np.mean([cur[F["mean"]][k][p]["efficacy"] for p in R1])
            n_ = np.mean([cur[F["noise"]][k][p]["efficacy"] for p in R1])
        except KeyError:
            g_ = m_ = n_ = np.nan
        G.append(g_), M.append(m_), Nz.append(n_)
    G, M, Nz = np.array(G), np.array(M), np.array(Nz)
    ok = ~np.isnan(G) & ~np.isnan(M)
    hi = [10.0, 30.0, 100.0]
    mn = [[cur[F["mean"]][rk(c, s)][p]["efficacy"] - cur[F["noise"]][rk(c, s)][p]["efficacy"] for p in hi]
          for c, s in runs if all(p in cur.get(F["mean"], {}).get(rk(c, s), {}) and
                                  p in cur.get(F["noise"], {}).get(rk(c, s), {}) for p in hi)]
    return {"per_r": per_r, "runs": [rk(c, s) for c, s in runs], "gain_R1_per_run": G.tolist(),
            "mean_R1_per_run": M.tolist(), "noise_R1_per_run": Nz.tolist(),
            "Gz": float(np.nanmean(G)) if ok.any() else float("nan"),
            "Em": float(np.nanmean(M)) if ok.any() else float("nan"),
            "ratio": float(np.nanmean(M) / np.nanmean(G)) if ok.any() else float("nan"),
            "runs_mean_ge_half_gain": int((M[ok] >= 0.5 * G[ok]).sum()), "n": int(ok.sum()),
            "gain_vs0": paired(G, np.zeros_like(G)), "mean_minus_gain": paired(M, G),
            "mean_minus_noise_r10_100": paired([np.mean(x) for x in mn], [0.0] * len(mn))}


def c2(fam, cur):
    """Approximate decomposition at the uncentred edit's grid point r (amendment 1): unc(delta) = cen(delta) +
    mean(delta) exactly; cen(delta) and mean(delta) sit at r_c = r * mu_cen / mu_unc and r_m = r * mu_mean / mu_unc
    (median lengths) on their own grids and are interpolated log-linearly in the achieved r."""
    F = FAMS[fam]
    out = {"per_run": {}, "per_r": {}}

    def interp(g, r):
        rs = sorted(g)
        ach = np.array([g[p]["r"] for p in rs])
        eff = np.array([g[p]["efficacy"] for p in rs])
        return float(np.interp(np.log(r), np.log(ach), eff))
    for c, s in F["runs"]:
        k = rk(c, s)
        if k not in cur.get(F["mean"], {}) or k not in cur.get(F["cen"], {}):
            continue
        chk = new_sets(c, s)["_check"][F["parent_set"]]
        rc_ratio = (chk["median_mu_parent"] if fam == "C" else chk["ember_c_median_mu"]) / chk["median_mu_uncentred"]
        rm_ratio = chk["mean_share_of_uncentred_r"]
        rec = {}
        for p in R:
            if p not in cur[F["unc"]][k]:
                continue
            rec[p] = {"unc": cur[F["unc"]][k][p]["efficacy"], "cen_at_delta": interp(cur[F["cen"]][k], p * rc_ratio),
                      "mean_at_delta": interp(cur[F["mean"]][k], p * rm_ratio), "r_cen": p * rc_ratio,
                      "r_mean": p * rm_ratio}
        out["per_run"][k] = {"ratio_cen": rc_ratio, "ratio_mean": rm_ratio, "points": rec}
    for p in R:
        xs = [x["points"][p] for x in out["per_run"].values() if p in x["points"]]
        if xs:
            out["per_r"][p] = {k2: float(np.mean([x[k2] for x in xs])) for k2 in ("unc", "cen_at_delta",
                                                                                    "mean_at_delta")} | {"n": len(xs)}
    gaps, dms = [], []
    for k, x in out["per_run"].items():
        pts = [x["points"][p] for p in R1 if p in x["points"]]
        if pts:
            gaps.append(np.mean([q["unc"] - q["cen_at_delta"] for q in pts]))
            dms.append(np.mean([q["mean_at_delta"] for q in pts]))
    out["gap_R1"] = float(np.mean(gaps)) if gaps else float("nan")
    out["mean_R1"] = float(np.mean(dms)) if dms else float("nan")
    out["share"] = out["mean_R1"] / out["gap_R1"] if gaps else float("nan")
    out["mean_minus_gap"] = paired(dms, gaps)
    return out


def copy_curves(fam, cur):
    """Efficacy vs copy along the grid (r <= 2.5); efficacy interpolated at copy = 0.9 and 0.5."""
    F = FAMS[fam]
    out = {}
    for v in (F["cen"], F["unc"], F["mean"], F["noise"]):
        for c, s in F["runs"]:
            cg = copygrid(c, s, v)
            g = cur.get(v, {}).get(rk(c, s), {})
            pts = sorted((p, cg[p]["copy"], g[p]["efficacy"], g[p]["r"]) for p in cg if p in g)
            if not pts:
                continue
            res = {"points": pts}
            for target in (0.9, 0.7, 0.5):
                val_ = float("nan")
                for (p1, c1_, e1, _), (p2, c2_, e2, _) in zip(pts, pts[1:]):
                    if c1_ >= target >= c2_ and c1_ != c2_:
                        val_ = e1 + (e2 - e1) * (c1_ - target) / (c1_ - c2_)
                        break
                if np.isnan(val_) and pts[0][1] < target:
                    val_ = float("nan")
                res[f"eff_at_copy{target}"] = val_
            out.setdefault(v, {})[rk(c, s)] = res
    summ = {}
    for target in (0.9, 0.7, 0.5):
        for v in out:
            xs = [x[f"eff_at_copy{target}"] for x in out[v].values()]
            summ.setdefault(str(target), {})[NAMES[v]] = {"mean": float(np.nanmean(xs)) if not all(np.isnan(xs))
                                                          else float("nan"), "n": int(np.sum(~np.isnan(xs)))}
        a = [out.get(F["cen"], {}).get(rk(c, s), {}).get(f"eff_at_copy{target}", np.nan) for c, s in F["runs"]]
        b = [out.get(F["unc"], {}).get(rk(c, s), {}).get(f"eff_at_copy{target}", np.nan) for c, s in F["runs"]]
        m = [out.get(F["mean"], {}).get(rk(c, s), {}).get(f"eff_at_copy{target}", np.nan) for c, s in F["runs"]]
        summ[str(target)]["cen-unc"] = paired(a, b)
        summ[str(target)]["unc-mean"] = paired(b, m)
    return {"curves": {NAMES[v]: x for v, x in out.items()}, "summary": summ}


def integrity(fam):
    """Judge-free E2 summaries: sweep r in {0.25, 1, 10, 100} and the two selected points, per variant."""
    F = FAMS[fam]
    out = defaultdict(dict)
    for v in (F["cen"], F["unc"], F["mean"]):
        for c, s in F["runs"]:
            pts = points(c, s, v)
            for nm, par in [(f"{p:g}", p) for p in SWEEP] + [("sel", pts.get("sel")), ("e3", pts.get("e3"))]:
                if par is None:
                    continue
                j = integ_file(c, s, v, par)
                if j is None:
                    continue
                out[NAMES[v]].setdefault(nm, {})[rk(c, s)] = {"param": par, **integ_summary(c, s, j), "_j": j}
    # paired contrasts on the pairwise common words
    con = {}
    pairs = [(F["mean"], F["unc"]), (F["mean"], F["cen"]), (F["unc"], F["cen"])]
    for a, b in pairs:
        for nm in [f"{p:g}" for p in SWEEP]:
            A_, B_ = out[NAMES[a]].get(nm, {}), out[NAMES[b]].get(nm, {})
            rows = []
            for k in A_:
                if k not in B_:
                    continue
                c, s = k.split("|")
                s = int(s)
                ws = sorted(set(A_[k]["_j"]["words"]) & set(B_[k]["_j"]["words"]))
                sa, sb = integ_summary(c, s, A_[k]["_j"], ws), integ_summary(c, s, B_[k]["_j"], ws)
                rows.append((sa, sb))
            if rows:
                con[f"{NAMES[a]}-{NAMES[b]}@{nm}"] = {key: paired([x[key] for x, _ in rows], [y[key] for _, y in rows])
                                                      for key in ("copy", "ctxmc", "ctx_excess", "echo")}
    summ = {}
    for v, d in out.items():
        for nm, runs in d.items():
            summ.setdefault(v, {})[nm] = {key: float(np.mean([x[key] for x in runs.values()]))
                                          for key in ("copy", "ctx_pass", "ctxmc", "ctxmc_orig", "ctxmc_nonce",
                                                      "ctx_excess", "echo", "del_likeness", "z_gt3", "nn_cos", "r")}
            summ[v][nm]["n"] = len(runs)
            summ[v][nm]["ctxmc_minus_orig"] = float(np.mean([x["ctxmc"] - x["ctxmc_orig"] for x in runs.values()]))
    per_run = {v: {nm: {k: {kk: vv for kk, vv in x.items() if kk != "_j"} for k, x in runs.items()}
                   for nm, runs in d.items()} for v, d in out.items()}
    return {"summary": summ, "contrasts": con, "per_run": per_run}


def selected(fam):
    F = FAMS[fam]
    out = {}
    for v in (F["cen"], F["unc"], F["mean"]):
        for rule in ("sel", "e3"):
            for c, s in F["runs"]:
                r = test_result(c, s, v, rule)
                if r is None:
                    continue
                out.setdefault(rule, {}).setdefault(NAMES[v], {})[rk(c, s)] = {
                    k: r.get(k) for k in ("param", "r", "n_edit", "efficacy", "specificity", "collateral",
                                          "collateral_rn", "H3v_rn")}
    con = {}
    for rule, d in out.items():
        for a, b in ((F["unc"], F["cen"]), (F["mean"], F["cen"]), (F["mean"], F["unc"])):
            A_, B_ = d.get(NAMES[a], {}), d.get(NAMES[b], {})
            ks = [k for k in A_ if k in B_]
            con[f"{rule}:{NAMES[a]}-{NAMES[b]}"] = {key: paired([A_[k][key] for k in ks], [B_[k][key] for k in ks])
                                                    for key in ("efficacy", "collateral_rn", "H3v_rn")}
    summ = {rule: {v: {key: float(np.mean([x[key] for x in runs.values()])) for key in
                       ("efficacy", "specificity", "collateral_rn", "H3v_rn")} |
                   {"n": len(runs), "r_median": float(np.median([x["r"] for x in runs.values()])),
                    "params": [x["param"] for x in runs.values()]}
                   for v, runs in d.items()} for rule, d in out.items()}
    return {"per_run": out, "summary": summ, "contrasts": con}


def verdict(res):
    a = res["C"]["c1"]
    ig = res["C"]["integrity"]["summary"]
    v = {"Gz": a["Gz"], "Em": a["Em"], "efficacy_part": a["Em"] >= 0.5 * a["Gz"]}
    try:
        cm1, cz1, cc1 = (ig[n]["1"]["copy"] for n in ("MEAN-C", "CINDER-z", "CINDER"))
        ctx_m = np.mean([ig["MEAN-C"][nm]["ctxmc"] for nm in ("10", "100")])
        ctx_z = np.mean([ig["CINDER-z"][nm]["ctxmc"] for nm in ("10", "100")])
        v.update(copy1={"MEAN-C": cm1, "CINDER-z": cz1, "CINDER": cc1}, ctxmc_10_100={"MEAN-C": ctx_m,
                                                                                      "CINDER-z": ctx_z})
        dmg = cm1 <= cz1 + 0.10 and ctx_m <= ctx_z + 0.05
        v["damage_part"] = bool(dmg)
        if v["efficacy_part"] and dmg:
            v["S1"] = "supported"
        elif not v["efficacy_part"] and cm1 <= cc1:
            v["S1"] = "refuted"
        else:
            v["S1"] = "mixed"
    except KeyError as e:
        v["S1"] = f"incomplete ({e})"
    return v


def ember_small_r(curE, curC):
    """Coordinator's question (r = 1 2x2): does EMBER gain from its mean component as r grows? EMBER vs EMBER-c vs
    MEAN-E at r = 0.5 and 1 (validation), with the 1b cell (EMBER's tokens, CINDER's direction, dense Y) and the
    CINDER-family analogue."""
    out = {}
    runs = FAMS["E"]["runs"]
    g1b = {rk(c, s): val(c, s, "sub:1b") for c, s in runs}
    for p in (0.25, 0.5, 1.0, 2.0):
        e = [curE.get("sub:ember", {}).get(rk(c, s), {}).get(p, {}).get("efficacy", np.nan) for c, s in runs]
        ec = [curE.get("sub:ember_c", {}).get(rk(c, s), {}).get(p, {}).get("efficacy", np.nan) for c, s in runs]
        me = [curE.get("sub:mean_e", {}).get(rk(c, s), {}).get(p, {}).get("efficacy", np.nan) for c, s in runs]
        b1 = [g1b[rk(c, s)].get(p, {}).get("efficacy", np.nan) for c, s in runs]
        co = {k: [curE.get(v, {}).get(rk(c, s), {}).get(p, {}).get("collateral", np.nan) for c, s in runs]
              for k, v in (("EMBER", "sub:ember"), ("EMBER-c", "sub:ember_c"), ("MEAN-E", "sub:mean_e"))}
        cr = FAMS["C"]["runs"]
        ci = [curC.get("sub:cinder", {}).get(rk(c, s), {}).get(p, {}).get("efficacy", np.nan) for c, s in cr]
        cz = [curC.get("sub:cinder_z", {}).get(rk(c, s), {}).get(p, {}).get("efficacy", np.nan) for c, s in cr]
        out[p] = {"EMBER": float(np.nanmean(e)), "EMBER-c": float(np.nanmean(ec)), "MEAN-E": float(np.nanmean(me)),
                  "1b": float(np.nanmean(b1)), "n_1b": int(np.sum(~np.isnan(b1))),
                  "collateral": {k: float(np.nanmean(v)) for k, v in co.items()},
                  "EMBER-c_minus_EMBER": paired(ec, e), "EMBER_minus_MEAN-E": paired(e, me),
                  "EMBER-c_minus_1b": paired(ec, b1), "1b_minus_EMBER": paired(b1, e),
                  "per_run": {rk(c, s): {"EMBER": a, "EMBER-c": b, "MEAN-E": m, "1b": x}
                              for (c, s), a, b, m, x in zip(runs, e, ec, me, b1)},
                  "CINDER_minus_CINDER-z": paired(ci, cz)}
    return out


def floor_excess(fam, cur, fl):
    """Validation efficacy minus the deletion / nonce floor of the same token set (qa_train)."""
    F = FAMS[fam]
    out = {}
    for v in (F["cen"], F["unc"], F["mean"], F["noise"]):
        for p in (1.0, 10.0, 100.0):
            d, n = [], []
            for c, s in F["runs"]:
                x = fl.get(rk(c, s), {}).get(F["parent_set"], {}).get("qa_train")
                e = cur.get(v, {}).get(rk(c, s), {}).get(p, {}).get("efficacy")
                if x is None or e is None:
                    continue
                d.append(e - x["del"])
                n.append(e - x["nonce"])
            if d:
                out.setdefault(NAMES[v], {})[p] = {"minus_del": float(np.mean(d)), "minus_nonce": float(np.mean(n)),
                                                   "n": len(d), "runs_above_del": int(np.sum(np.array(d) > 0))}
    fls = [x[F["parent_set"]]["qa_train"] for x in fl.values() if F["parent_set"] in x and "qa_train" in x[F["parent_set"]]]
    flt = [x[F["parent_set"]]["qa_test"] for x in fl.values() if F["parent_set"] in x and "qa_test" in x[F["parent_set"]]]
    out["floor_mean"] = {"qa_train_del": float(np.mean([x["del"] for x in fls])) if fls else None,
                         "qa_train_nonce": float(np.mean([x["nonce"] for x in fls])) if fls else None,
                         "qa_test_del": float(np.mean([x["del"] for x in flt])) if flt else None,
                         "qa_test_nonce": float(np.mean([x["nonce"] for x in flt])) if flt else None,
                         "n": len(fls)}
    return out


def s2s4_summary():
    from scipy.stats import spearmanr
    out = {}
    f2 = OUT / "s2_read_gain.json"
    if f2.exists():
        S2 = json.loads(f2.read_text())
        out["s2"] = {"mu_hat": S2["mu_hat"], "summary": S2["summary"], "gaps": S2["gaps"],
                     "top8_g": S2["top8_g"], "mu_hat_energy_top8": S2["mu_hat_energy_top8"]}
    f4 = OUT / "s4_forward.json"
    if f4.exists():
        S4 = json.loads(f4.read_text())
        by = defaultdict(dict)
        for r in S4["rows"]:
            by[(r["concept"], r["seed"], r["family"], r["token"], r["scale"])][r["dir"]] = r
        res = {}
        for sc in S4["scales"]:
            rel = defaultdict(lambda: defaultdict(list))
            px, py = [], []
            wins = defaultdict(list)
            for k, d in by.items():
                if k[4] != sc:
                    continue
                base = {m: np.mean([d[f"rand{i}"][m] for i in range(4)])
                        for m in ("l0_write", "l1_after", "kl_tok", "kl_after")}
                for n, x in d.items():
                    px.append(x["proxy_qkv"])
                    py.append(x["l0_write"])
                    if not n.startswith("rand"):
                        for m in base:
                            rel[n][m].append(x[m] / base[m])
                fam = k[2]
                cen, unc, mn = ("cinder", "cinder_z", "mean_c") if fam == "C" else ("ember_c", "ember", "mean_e")
                for m in ("l0_write", "l1_after", "kl_tok", "kl_after"):
                    wins[f"unc>cen|{m}"].append(d[unc][m] > d[cen][m])
                    wins[f"mean>cen|{m}"].append(d[mn][m] > d[cen][m])
            res[str(sc)] = {"spearman_proxy_l0": float(spearmanr(px, py)[0]), "n_points": len(px),
                            "median_rel": {n: {m: float(np.median(v)) for m, v in d.items()} for n, d in rel.items()},
                            "share": {k: float(np.mean(v)) for k, v in wins.items()}}
        out["s4"] = res
    return out


def excess_vs_damage(fam, cur, fl, ig):
    """Per (run, variant, r in {1, 10, 100}): efficacy above the deletion floor vs context damage (context-MC of the
    real word minus that of the edited word; answer-NLL excess). Spearman over the pooled points."""
    from scipy.stats import spearmanr
    F = FAMS[fam]
    pts = []
    for v in (F["cen"], F["unc"], F["mean"]):
        for c, s in F["runs"]:
            k = rk(c, s)
            x = fl.get(k, {}).get(F["parent_set"], {}).get("qa_train")
            for nm, p in (("1", 1.0), ("10", 10.0), ("100", 100.0)):
                e = cur.get(v, {}).get(k, {}).get(p, {}).get("efficacy")
                q = ig["per_run"].get(NAMES[v], {}).get(nm, {}).get(k)
                if x is None or e is None or q is None:
                    continue
                pts.append({"variant": NAMES[v], "run": k, "r": p, "excess": e - x["del"],
                            "ctx_drop": q["ctxmc_orig"] - q["ctxmc"], "nll_excess": q["ctx_excess"]})
    if len(pts) < 5:
        return {"n": len(pts)}
    ex = [q["excess"] for q in pts]
    return {"n": len(pts), "spearman_ctx_drop": float(spearmanr(ex, [q["ctx_drop"] for q in pts])[0]),
            "spearman_nll": float(spearmanr(ex, [q["nll_excess"] for q in pts])[0]), "points": pts}


def main():
    res = {}
    fl = floors()
    for fam in ("C", "E"):
        cur = curves(fam)
        res[fam] = {"c1": c1(fam, cur), "c2": c2(fam, cur), "copy": copy_curves(fam, cur), "integrity": integrity(fam),
                    "selected": selected(fam),
                    "curves": {NAMES[v]: x for v, x in cur.items()}}
        res[fam]["floors"] = {k: x.get(FAMS[fam]["parent_set"]) for k, x in fl.items()}
        res[fam]["floor_excess"] = floor_excess(fam, cur, fl)
        res[fam]["excess_vs_damage"] = excess_vs_damage(fam, cur, fl, res[fam]["integrity"])
        res[fam]["_cur"] = cur
    res["ember_small_r"] = ember_small_r(res["E"].pop("_cur"), res["C"].pop("_cur"))
    res["s2s4"] = s2s4_summary()
    res["verdict"] = verdict(res)
    res["checks"] = {rk(c, s): new_sets(c, s)["_check"] for c, s in sorted(set(CINDER_RUNS) | set(EMBER_RUNS))}
    save(OUT / "analysis.json", res)
    print(json.dumps(res["verdict"], indent=1))
    for fam in ("C", "E"):
        x = res[fam]["c1"]
        print(fam, "Gz", round(x["Gz"], 3), "Em", round(x["Em"], 3), "ratio", round(x["ratio"], 2),
              "c2 share", round(res[fam]["c2"].get("share", float("nan")), 2))


if __name__ == "__main__":
    main()
