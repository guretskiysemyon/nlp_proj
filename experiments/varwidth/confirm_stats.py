#!/usr/bin/env python
"""Pre-registered analysis of the confirmatory expansion (PREREG amendment 8). CPU only, from saved files.

    python -m experiments.varwidth.confirm_stats   # -> results/varwidth/layers/collateral/confirm.md + confirm.json

Primary: paired H3v (collateral rule, test split, collateral renormalized by the unedited model's per-token test
accuracy) for each CINDER variant minus EMBER, on runs where both edit. Unit = concept (mean over its seeds), two-sided
Wilcoxon signed-rank across concepts, Holm over the 4 contrasts, 95% cluster-bootstrap CI (concepts resampled).
Secondary: the same at fixed r = 0.5 and under the paper rule (H2); failure rate (H3v < 0.25); edit rate; judge-free
E2 integrity (copy, echo) at the selected point.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

from experiments.varwidth.collateral import EXP_VARIANTS, EXPANSION, OUT, seeds
from experiments.varwidth.collateral_report import NAMES, renorm
from experiments.varwidth.common import slug
from experiments.varwidth.erase import hm

RULES = {"collateral": "result.json", "fixed05": "result_fixed05.json", "paper": "result_paper.json"}
CONTRASTS = [v for v in EXP_VARIANTS if v != "sub:ember"]
IOUT = OUT / "integrity"
FAIL = 0.25
B = 10000


def load(rule):
    """{(concept, seed, variant): result} for the expansion runs under one rule (variants with an empty edit set
    have no result)."""
    R = {}
    for c in EXPANSION:
        for s in seeds(c):
            for v in EXP_VARIANTS:
                f = OUT / slug(c) / f"s{s}" / v.replace(":", "_") / RULES[rule]
                if f.exists():
                    R[(c, s, v)] = json.loads(f.read_text())
    renorm(R)
    for r in R.values():
        r["H2"] = hm([r["efficacy"], r["specificity"]])
    return R


def edits(r):
    return r is not None and r.get("n_edit", 0) > 0


def holm(p):
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(order):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def contrast(R, v, metric):
    """Per-concept mean paired difference (v - EMBER) over seeds where both edit."""
    per = {}
    for c in EXPANSION:
        d = [R[(c, s, v)][metric] - R[(c, s, "sub:ember")][metric] for s in seeds(c)
             if edits(R.get((c, s, v))) and edits(R.get((c, s, "sub:ember")))]
        if d:
            per[c] = (float(np.mean(d)), len(d))
    x = np.array([m for m, _ in per.values()])
    out = {"n_concepts": len(x), "n_pairs": sum(n for _, n in per.values()), "per_concept": per}
    if len(x) == 0:
        return out
    rng = np.random.default_rng(0)
    boot = rng.choice(x, (B, len(x)), replace=True).mean(1)
    out.update(mean=float(x.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               p=float(wilcoxon(x).pvalue) if len(x) >= 2 and np.any(x != 0) else 1.0,
               wins=int((x > 0).sum()), losses=int((x < 0).sum()))
    return out


def by_variant(R, metric):
    """Mean of a metric per variant over the runs where it edits, and over concept means."""
    out = {}
    for v in EXP_VARIANTS:
        xs = [r[metric] for (c, s, vv), r in R.items() if vv == v and edits(r)]
        out[v] = float(np.mean(xs)) if xs else float("nan")
    return out


def rates(R):
    """Edit rate (runs with a non-empty set whose rule picked a point) and failure rate (H3v < 0.25, over all 60 runs;
    a run without an edit counts as a failure of the method)."""
    n = sum(len(seeds(c)) for c in EXPANSION)
    out = {}
    for v in EXP_VARIANTS:
        rs = [R.get((c, s, v)) for c in EXPANSION for s in seeds(c)]
        out[v] = {"has_set": sum(r is not None for r in rs) / n, "edit_rate": sum(edits(r) for r in rs) / n,
                  "fail_rate": sum((not edits(r)) or r["H3v_rn"] < FAIL for r in rs) / n,
                  "fail_rate_when_edits": float(np.mean([r["H3v_rn"] < FAIL for r in rs if edits(r)]))
                  if any(edits(r) for r in rs) else float("nan")}
    return out


def integrity():
    """Judge-free copy / echo at the selected point: {variant: (copy, echo, n runs)} and the paired copy contrast."""
    from experiments.varwidth.integrity import summarize
    S = {}
    for c in EXPANSION:
        for s in seeds(c):
            d = IOUT / slug(c) / f"s{s}"
            for v in EXP_VARIANTS:
                if (d / f"{v.replace(':', '_')}@sel.json").exists():
                    S[(c, s, v)] = summarize(d, f"{v}@sel")["summary"]
    out = {}
    for v in EXP_VARIANTS:
        xs = [x for (c, s, vv), x in S.items() if vv == v]
        out[v] = {"copy": float(np.mean([x["copy"] for x in xs])) if xs else float("nan"),
                  "echo": float(np.mean([x["echo"] for x in xs])) if xs else float("nan"), "n": len(xs)}
    R = {k: {"copy": x["copy"], "n_edit": 1} for k, x in S.items()}
    return out, {v: contrast(R, v, "copy") for v in CONTRASTS}


AGENTS_X = {b: a for a, bs in enumerate([(41, 42, 43), (44, 45, 46), (47, 48, 49), (50, 51, 52), (53, 54, 55),
                                         (56, 57)]) for b in bs}     # amendment 8 labellers: batches per agent


def t3():
    """Expansion T3 (seed 42, selected point): per variant graceful / leak / confab / destruction, destruction of
    exclusive vs multi-meaning words, the paired destruction contrast (unit = concept), and cross-agent agreement
    on the expansion duplicates."""
    from collections import Counter

    from experiments.varwidth.integrity import DESTR, GRACEFUL, JDIR, LEAK, labels, token_class
    key, lab = json.loads((JDIR / "key.json").read_text()), labels()
    rows = {}
    for i, k in key.items():
        if k["concept"] in EXPANSION and i in lab and k["config"].endswith("@sel"):
            rows.setdefault(k["concept"], []).append((k["config"][:-4], k["key"], lab[i]))
    out, per = {}, {}
    for c, rs in rows.items():
        cls_tok = token_class(c)
        meta = json.loads((IOUT / slug(c) / "s42" / "meta.json").read_text())["words"]
        wcls = {w: "multi" if any(cls_tok.get(t) == "multi" for t in v["tokens"]) else
                "exclusive" if any(cls_tok.get(t) == "exclusive" for t in v["tokens"]) else "unclassified"
                for w, v in meta.items()}
        for v, w, lb in rs:
            out.setdefault(v, {"all": Counter(), "exclusive": [], "multi": []})["all"][lb] += 1
            if wcls.get(w) in ("exclusive", "multi"):
                out[v][wcls[w]].append(lb in DESTR)
            per.setdefault((c, v), []).append(lb in DESTR)
    summ = {}
    for v, g in out.items():
        n = sum(g["all"].values())
        summ[v] = {"n": n, "graceful": sum(g["all"][x] for x in GRACEFUL) / n,
                   "leak": sum(g["all"][x] for x in LEAK) / n, "confab": g["all"]["CONFAB"] / n,
                   "destruction": sum(g["all"][x] for x in DESTR) / n,
                   "destr_exclusive": float(np.mean(g["exclusive"])) if g["exclusive"] else float("nan"),
                   "n_exclusive": len(g["exclusive"]),
                   "destr_multi": float(np.mean(g["multi"])) if g["multi"] else float("nan"), "n_multi": len(g["multi"])}
    R = {(c, 42, v): {"destr": float(np.mean(x)), "n_edit": 1} for (c, v), x in per.items()}
    con = {v: contrast(R, v, "destr") for v in CONTRASTS}
    batch_of = {it["id"]: int(f.stem.split("_")[1]) for f in JDIR.glob("batch_*.json")
                for it in json.loads(f.read_text())}
    dups = [(i, i + "d") for i in key if key[i]["concept"] in EXPANSION and i in lab and i + "d" in lab
            and AGENTS_X.get(batch_of.get(i)) != AGENTS_X.get(batch_of.get(i + "d"))]
    agree = {"n": len(dups), "exact": float(np.mean([lab[a] == lab[b] for a, b in dups])) if dups else float("nan"),
             "destruction": float(np.mean([(lab[a] in DESTR) == (lab[b] in DESTR) for a, b in dups]))
             if dups else float("nan"),
             "leak": float(np.mean([(lab[a] in LEAK) == (lab[b] in LEAK) for a, b in dups])) if dups else float("nan")}
    return summ, con, agree


def fmt(x, n=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def main():
    res = {"rules": {}}
    lines = ["# Confirmatory expansion (PREREG amendment 8)", "",
             "12 concepts × seeds 42–46. Test split. Collateral renormalized by the unedited model's per-token test "
             "accuracy. Paired contrasts on runs where both methods edit; unit = concept (mean over seeds); "
             "two-sided Wilcoxon across concepts; Holm over the 4 primary contrasts; 95% cluster-bootstrap CI.", ""]
    for rule, metric in (("collateral", "H3v_rn"), ("fixed05", "H3v_rn"), ("paper", "H2"), ("paper", "H3v_rn")):
        R = load(rule)
        C = {v: contrast(R, v, metric) for v in CONTRASTS}
        adj = holm(np.array([C[v].get("p", 1.0) for v in CONTRASTS]))
        for v, a in zip(CONTRASTS, adj):
            C[v]["p_holm"] = float(a)
        res["rules"][f"{rule}|{metric}"] = {"contrasts": C, "means": by_variant(R, metric), "rates": rates(R)}
        tag = {"collateral": "Primary: collateral rule", "fixed05": "Secondary: fixed r = 0.5",
               "paper": "Secondary: paper rule"}[rule]
        lines += [f"## {tag}, {metric}", "", "| Variant − EMBER | concepts (pairs) | mean Δ | 95% CI | wins/losses | "
                  "Wilcoxon p | Holm p |", "|---|---|---|---|---|---|---|"]
        for v in CONTRASTS:
            x = C[v]
            if "mean" not in x:
                lines.append(f"| {NAMES[v]} | 0 | – | – | – | – | – |")
                continue
            lines.append(f"| {NAMES[v]} | {x['n_concepts']} ({x['n_pairs']}) | {x['mean']:+.3f} | "
                         f"[{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] | {x['wins']}/{x['losses']} | {x['p']:.3f} | "
                         f"{x['p_holm']:.3f} |")
        rt = res["rules"][f"{rule}|{metric}"]["rates"]
        mn = res["rules"][f"{rule}|{metric}"]["means"]
        lines += ["", f"| Variant | mean {metric} (runs where it edits) | edit set exists | edit rate | "
                  f"failure rate (H3v < {FAIL}, no edit = fail) | failure rate when it edits |", "|---|---|---|---|---|---|"]
        for v in EXP_VARIANTS:
            lines.append(f"| {NAMES[v]} | {fmt(mn[v])} | {fmt(rt[v]['has_set'])} | {fmt(rt[v]['edit_rate'])} | "
                         f"{fmt(rt[v]['fail_rate'])} | {fmt(rt[v]['fail_rate_when_edits'])} |")
        lines.append("")
    I, IC = integrity()
    res["integrity"], res["integrity_contrast"] = I, IC
    lines += ["## Secondary: E2 integrity at the selected point (judge-free)", "",
              "| Variant | copy | echo | runs |", "|---|---|---|---|"]
    lines += [f"| {NAMES[v]} | {fmt(I[v]['copy'])} | {fmt(I[v]['echo'])} | {I[v]['n']} |" for v in EXP_VARIANTS]
    lines += ["", "| Variant − EMBER, copy | concepts (pairs) | mean Δ | 95% CI | Wilcoxon p |", "|---|---|---|---|---|"]
    for v in CONTRASTS:
        x = IC[v]
        if "mean" in x:
            lines.append(f"| {NAMES[v]} | {x['n_concepts']} ({x['n_pairs']}) | {x['mean']:+.3f} | "
                         f"[{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] | {x['p']:.3f} |")
    T, TC, AG = t3()
    res["t3"], res["t3_contrast"], res["t3_agreement"] = T, TC, AG
    lines += ["", "## Secondary: T3 labels at the selected point (seed 42, blind agents)", "",
              f"Cross-agent agreement on {AG['n']} expansion duplicates: exact label {fmt(AG['exact'])}, destruction "
              f"yes/no {fmt(AG['destruction'])}, leak yes/no {fmt(AG['leak'])}.", "",
              "| Variant | n | graceful | leak | confab | destruction | destruction, exclusive (n) | "
              "destruction, multi (n) |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {NAMES[v]} | {T[v]['n']} | {fmt(T[v]['graceful'])} | {fmt(T[v]['leak'])} | {fmt(T[v]['confab'])} | "
              f"{fmt(T[v]['destruction'])} | {fmt(T[v]['destr_exclusive'])} ({T[v]['n_exclusive']}) | "
              f"{fmt(T[v]['destr_multi'])} ({T[v]['n_multi']}) |" for v in EXP_VARIANTS if v in T]
    lines += ["", "| Variant − EMBER, destruction | concepts | mean Δ | 95% CI | Wilcoxon p |", "|---|---|---|---|---|"]
    for v in CONTRASTS:
        x = TC[v]
        if "mean" in x:
            lines.append(f"| {NAMES[v]} | {x['n_concepts']} | {x['mean']:+.3f} | [{x['ci'][0]:+.3f}, "
                         f"{x['ci'][1]:+.3f}] | {x['p']:.3f} |")
    (OUT / "confirm.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "confirm.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
