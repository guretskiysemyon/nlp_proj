"""Aggregate metrics + report for the variable-width sparse MF study (instructions_new.txt).

    python -m experiments.varwidth.evaluate --concept "Ancient Rome" [--tags m1 m2 ...] [--root DIR] [--no-figs]

Reads every run <root>/<concept_slug>/<tag>/seed<N>/{summary.json, fit.pt} (layout in common.py; root defaults to
results/varwidth) plus the prep data <root>/prep/<slug>_<matrix>_S<n>.pt (falls back to results/varwidth/prep/) for
token strings, C/N/B codes and A. Writes to <root>/<slug>/:
    report.md     all sections below, each saying "not run yet" when its configs/runs are missing
    metrics.json  per-config metrics + cross-config sections + decision
    runs.csv      one row per run
    figs/*.png    concept width per seed, inclusion curves, width histograms, churn, resolution sweep

A config is a tag directory; its identity is parsed from summary["config"] (VWConfig.to_dict()) as the method plus the
switches that differ from VWConfig(method=...) defaults, plus three switches that are not VWConfig fields: matrix
("plain" from config["matrix"] / summary["matrix"] / a "_plain" tag part), |S| (config["S"] / config["n_concept_sents"]
/ summary["n_concept_sents"] / a "_S<n>" tag part; default 300) and merge (summary["merge"] present, or "merge" in the
tag). The fixed + refit baseline is the config with method "fixed" and no switches.

The concept feature of a run is its F_C feature with the highest ratio_abs in summary["candidates"] (widest F_C
feature if no candidate info). summary["merge"] (method 5) is read as {"threshold", "concept_pair_cosines":
[{"a","b","cos"}], "pairs": [{"a","b","cos","accepted","rel_err_before","rel_err_after", ...}], "rel_err_before",
"rel_err_after", "accepted"}; unknown keys are dumped as JSON.

pick_winner(concept) applies the spec's decision rules and returns the tag to carry forward (see decide()).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from experiments.varwidth.common import METHODS, RESULTS, VWConfig, budgets, slug

LAM = 1e-4
DEFAULT_S = 300
EXPECTED = {  # tail tokens expected to enter the concept support (whole-token, case-insensitive, "▁" stripped)
    "Ancient_Rome": ["Cleopatra", "gladiator", "Constantine", "Augustus", "Colosseum", "Senate", "legion", "Julius",
                     "Pompeii", "Carthage", "Hannibal", "Latin", "emperor", "Nero", "Caesar", "Trajan", "Hadrian"],
}
# spec thresholds (starting points; adjust after stage 1)
CV_RANDOM_BUDGET = 0.29
CV_CARRY = 0.15
ABSORB_SHARE = 0.25
SPEARMAN_REAL = 0.6
M4_WIDTH_TOL = 0.25          # "m4 close to the winner": matched-width Spearman > 0.6 and concept width within 25%
CHURN_TOL = 0.005
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#8a8985"


# ================================================================================================= loading
def describe(summary: dict, tag: str) -> tuple[str, dict]:
    """(method, switches that differ from the defaults) for one run."""
    cfg = dict(summary.get("config") or {})
    method = cfg.get("method") or tag.split("_")[0]
    base = VWConfig(method=method).to_dict() if method in METHODS else {}
    diffs = {}
    for k, v in cfg.items():
        if k in base and k != "method" and not _same(base[k], v):
            diffs[k] = v
    matrix = cfg.get("matrix") or summary.get("matrix") or ("plain" if re.search(r"(^|_)plain(_|$)", tag) else "whitened")
    m = re.search(r"(^|_)S(\d+)(_|$)", tag)
    S = cfg.get("S") or cfg.get("n_concept_sents") or summary.get("n_concept_sents") or (int(m.group(2)) if m else DEFAULT_S)
    merge = bool(summary.get("merge")) or "merge" in tag
    if matrix != "whitened":
        diffs["matrix"] = matrix
    if int(S) != DEFAULT_S:
        diffs["S"] = int(S)
    if merge:
        diffs["merge"] = True
    return method, diffs


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return a == b


class Run:
    def __init__(self, tag: str, seed: int, path: Path):
        self.tag, self.seed, self.path = tag, seed, path
        self.summary = json.loads((path / "summary.json").read_text())
        self.method, self.diffs = describe(self.summary, tag)
        self.F_C = [int(f) for f in self.summary.get("F_C", [])]

    @cached_property
    def fit(self) -> dict | None:
        p = self.path / "fit.pt"
        if not p.exists():
            return None
        d = torch.load(p, map_location="cpu", weights_only=False)
        for key in ("Z", "Y"):
            if key in d and torch.is_tensor(d[key]):
                d[key] = d[key].float().numpy()
        return d

    @property
    def Z(self):
        return None if self.fit is None else self.fit.get("Z")

    @property
    def Y(self):
        return None if self.fit is None else self.fit.get("Y")

    @cached_property
    def Zn(self):
        Z = self.Z
        return None if Z is None else Z / np.maximum(np.linalg.norm(Z, axis=0, keepdims=True), 1e-12)

    @cached_property
    def widths(self) -> np.ndarray:
        if self.Y is not None:
            return (self.Y != 0).sum(1)
        if self.fit is not None and "widths" in self.fit:
            return np.asarray(self.fit["widths"])
        return np.asarray(self.summary.get("widths", []))

    @property
    def n(self) -> int:
        return int(self.Y.shape[1]) if self.Y is not None else int(self.summary.get("vstar", 0))

    @cached_property
    def budget(self) -> tuple[int, int, int]:
        f = self.fit or {}
        if all(k in f for k in ("K", "m_min", "m_max")):
            return int(f["K"]), int(f["m_min"]), int(f["m_max"])
        cfg = {k: v for k, v in (self.summary.get("config") or {}).items() if k in VWConfig().to_dict()}
        try:
            return budgets(self.n, VWConfig(**cfg))
        except Exception:  # noqa: BLE001
            return int(self.widths.sum()), 0, self.n

    @cached_property
    def concept(self) -> int | None:
        if not self.F_C:
            return None
        ratio = {int(c["feature"]): c.get("ratio_abs", float("nan")) for c in self.summary.get("candidates", [])}
        if any(f in ratio and np.isfinite(ratio[f]) for f in self.F_C):
            return max(self.F_C, key=lambda f: ratio.get(f, -np.inf))
        w = self.widths
        return max(self.F_C, key=lambda f: w[f] if len(w) > f else 0)

    def support(self, i: int) -> np.ndarray:
        if self.Y is not None:
            return np.nonzero(self.Y[i])[0]
        return np.array([], dtype=int)

    @cached_property
    def selected(self) -> set:
        """Token indices in the union of the F_C supports."""
        return set(int(j) for i in self.F_C for j in self.support(i))

    @property
    def concept_width(self) -> float:
        if self.concept is None:
            return float("nan")
        if len(self.widths) > self.concept:
            return float(self.widths[self.concept])
        return float(self.summary.get("tokens_per_feature", {}).get(str(self.concept), float("nan")))

    @property
    def share(self) -> float:
        n = self.summary.get("n_distinct_tokens", 0)
        return self.summary.get("C", 0) / n if n else float("nan")


@dataclass
class Group:
    tag: str
    runs: list
    method: str
    diffs: dict
    prep: dict | None = None
    m: dict = field(default_factory=dict)   # metrics

    @property
    def seeds(self):
        return [r.seed for r in self.runs]

    def run(self, seed):
        return next((r for r in self.runs if r.seed == seed), None)

    def tok(self, j: int) -> str:
        if self.prep and "tokens" in self.prep:
            return str(self.prep["tokens"][j])
        return f"id{j}"

    def code(self, j: int) -> int | None:
        if self.prep is not None and "codes" in self.prep:
            return int(self.prep["codes"][j])
        return None

    def label(self, j: int) -> str:
        c = self.code(j)
        return f"{self.tok(j)} [{'NCB'[c]}]" if c is not None else self.tok(j)

    @property
    def is_base(self) -> bool:
        return not self.diffs


def _load_prep(roots: list, concept: str, matrix: str, S: int, n: int, cache: dict) -> dict | None:
    key = (matrix, S)
    if key not in cache:
        cache[key] = None
        for root in roots:
            for p in sorted((root / "prep").glob(f"{slug(concept)}_{matrix}_S{S}*.pt")) if (root / "prep").exists() else []:
                if "_sub" in p.stem:
                    continue
                d = torch.load(p, map_location="cpu", weights_only=False)
                A = d.get("A")
                d["A"] = A.float().numpy() if torch.is_tensor(A) else A
                d["codes"] = np.asarray(d["codes"]) if "codes" in d else None
                cache[key] = d
                break
            if cache[key] is not None:
                break
    d = cache[key]
    if d is not None and len(d.get("vstar_ids", d.get("tokens", []))) != n:
        return None
    return d


def load_groups(concept: str, root: Path | None = None, tags=None) -> dict:
    root = Path(root) if root else RESULTS
    cdir = root / slug(concept)
    groups, cache = {}, {}
    if not cdir.exists():
        return groups
    for tdir in sorted(p for p in cdir.iterdir() if p.is_dir() and p.name != "figs"):
        if tags and tdir.name not in tags:
            continue
        runs = []
        for sdir in sorted(tdir.glob("seed*")):
            if (sdir / "summary.json").exists():
                try:
                    runs.append(Run(tdir.name, int(sdir.name[4:]), sdir))
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"[evaluate] skip {sdir}: {e}")
        if not runs:
            continue
        runs.sort(key=lambda r: r.seed)
        g = Group(tdir.name, runs, runs[0].method, runs[0].diffs)
        matrix = g.diffs.get("matrix", "whitened")
        S = g.diffs.get("S", DEFAULT_S)
        g.prep = _load_prep([root, RESULTS], concept, matrix, S, runs[0].n, cache)
        groups[g.tag] = g
    order = {m: i for i, m in enumerate(METHODS)}
    return dict(sorted(groups.items(), key=lambda kv: (order.get(kv[1].method, 99), len(kv[1].diffs), kv[0])))


def find(groups: dict, method: str, **diffs) -> Group | None:
    hits = [g for g in groups.values() if g.method == method and _diffs_eq(g.diffs, diffs)]
    return max(hits, key=lambda g: len(g.runs)) if hits else None


def _diffs_eq(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)


# ================================================================================================= helpers
def mstd(x, nd=3) -> str:
    x = [v for v in x if v is not None and np.isfinite(v)]
    if not x:
        return "—"
    if len(x) == 1:
        return f"{x[0]:.{nd}f}"
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"


def f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def cv(x) -> float:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 2 or x.mean() == 0:
        return float("nan")
    return float(x.std(ddof=1) / x.mean())


def gini(w) -> float:
    w = np.sort(np.asarray(w, float))
    if w.sum() == 0:
        return float("nan")
    n = len(w)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(w) / (n * w.sum()))


def hungarian(Zn_a, Zn_b):
    """perm[i] = feature of b matched to feature i of a (max total |cos z|), and the matched |cos|."""
    C = np.abs(Zn_a.T @ Zn_b)
    r, c = linear_sum_assignment(-C)
    perm = np.empty(C.shape[0], int)
    perm[r] = c
    return perm, C[np.arange(C.shape[0]), perm]


def stats_row(x) -> dict:
    x = [v for v in x if v is not None and np.isfinite(v)]
    if not x:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, "min": float(np.min(x)),
            "median": float(np.median(x)), "max": float(np.max(x))}


def norm_tok(t: str) -> str:
    return t.replace("▁", "").strip().lower()


def md_table(header, rows) -> list:
    def cell(c):
        return str(c).replace("|", "\\|")
    out = ["| " + " | ".join(map(cell, header)) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return out


NOT_RUN = "_Not run yet._"


# ================================================================================================= per-config metrics
def y_dense_row(run: Run, i: int, A) -> np.ndarray | None:
    if A is None or run.Z is None:
        return None
    Z = run.Z.astype(np.float64)
    G = Z.T @ Z + LAM * np.eye(Z.shape[1])
    e = np.linalg.solve(G, np.eye(Z.shape[1])[i])          # row i of G^-1 (G symmetric)
    return (e @ Z.T) @ A


def _lockin(x) -> float:
    """lockin_iter; -1 / None (churn never settled) -> nan."""
    return float("nan") if x is None or x < 0 else float(x)


def config_metrics(g: Group) -> dict:
    m = {}
    runs = g.runs
    # 1. seed-sweep metrics
    cols = {"n_features": [r.summary.get("n_concept_features", len(r.F_C)) for r in runs],
            "n_tokens": [r.summary.get("n_distinct_tokens", 0) for r in runs],
            "C": [r.summary.get("C", 0) for r in runs], "N": [r.summary.get("N", 0) for r in runs],
            "B": [r.summary.get("B", 0) for r in runs],
            "concept_share": [r.share for r in runs],
            "rel_err": [r.summary.get("rel_err", float("nan")) for r in runs],
            "concept_width": [r.concept_width for r in runs],
            "fit_elapsed": [r.summary.get("fit_elapsed", float("nan")) for r in runs],
            "total_elapsed": [r.summary.get("total_elapsed", float("nan")) for r in runs],
            "n_iter": [r.summary.get("n_iter", float("nan")) for r in runs],
            "lockin_iter": [_lockin(r.summary.get("lockin_iter")) for r in runs]}
    m["stats"] = {k: stats_row([float(x) for x in v]) for k, v in cols.items()}
    m["n_seeds"] = len(runs)
    m["n_features_counts"] = dict(sorted(Counter(cols["n_features"]).items()))
    sets = [{g.label(j) for j in r.selected} for r in runs]
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if a | b]
    freq = pd.Series([t for s in sets for t in s], dtype=object).value_counts()
    m["jaccard"] = float(np.mean(jac)) if jac else float("nan")
    m["union_tokens"] = int(len(freq))
    m["core50"] = int((freq >= len(runs) / 2).sum())
    m["core_all"] = int((freq == len(runs)).sum())
    m["most_frequent"] = [(t, int(c)) for t, c in freq.head(20).items()]
    m["most_frequent_neutral"] = [(t, int(c)) for t, c in freq.items() if t.endswith("[N]")][:10]

    # 2-3. width + direction stability (needs Z)
    have = [r for r in runs if r.concept is not None and r.Zn is not None]
    m["concept_width_cv"] = cv(cols["concept_width"])
    pair_cos, maps, matched_cos = [], [], []
    for a, b in itertools.combinations(have, 2):
        pair_cos.append(float(abs(a.Zn[:, a.concept] @ b.Zn[:, b.concept])))
        perm, mc = hungarian(a.Zn, b.Zn)
        maps.append(bool(perm[a.concept] == b.concept))
        matched_cos.append(float(mc[a.concept]))
    m["direction_stability"] = float(np.mean(pair_cos)) if pair_cos else float("nan")
    m["concept_maps_to_concept"] = float(np.mean(maps)) if maps else float("nan")
    m["concept_matched_cos"] = float(np.mean(matched_cos)) if matched_cos else float("nan")
    tracked = {}
    if have:  # reference = seed whose concept z is most central; track it into every seed (also seeds w/o concept)
        ref = max(have, key=lambda r: np.mean([abs(r.Zn[:, r.concept] @ o.Zn[:, o.concept]) for o in have]))
        for r in runs:
            if r.Zn is None:
                continue
            if r is ref:
                tracked[r.seed] = (r.concept, 1.0)
            else:
                perm, mc = hungarian(ref.Zn, r.Zn)
                tracked[r.seed] = (int(perm[ref.concept]), float(mc[ref.concept]))
        m["tracked_ref_seed"] = ref.seed
    m["tracked"] = {s: {"feature": t[0], "cos": t[1], "width": float(g.run(s).widths[t[0]]),
                        "is_concept": t[0] in g.run(s).F_C} for s, t in tracked.items()}
    m["tracked_width_cv"] = cv([v["width"] for v in m["tracked"].values()])

    # 4. inclusion curve
    m["inclusion"] = inclusion(g, tracked)

    # 5. width distribution
    ginis, floor, cap, dead, pooled = [], [], [], [], []
    for r in runs:
        w = r.widths
        if not len(w):
            continue
        K, mmin, mmax = r.budget
        pooled += list(map(int, w))
        ginis.append(gini(w))
        floor.append(int((w == mmin).sum()) if mmin > 0 else float("nan"))
        cap.append(int((w >= mmax).sum()) if mmax < r.n else float("nan"))
        dead.append(int((w == 0).sum()))
    m["widths_pooled"] = pooled
    m["gini"] = float(np.mean(ginis)) if ginis else float("nan")
    m["at_floor"] = float(np.nanmean(floor)) if floor and np.isfinite(floor).any() else float("nan")
    m["at_cap"] = float(np.nanmean(cap)) if cap and np.isfinite(cap).any() else float("nan")
    m["dead_rows"] = float(np.mean(dead)) if dead else float("nan")
    m["m_min_max"] = runs[0].budget[1:] if runs else None

    # 6. split
    split = []
    for r in runs:
        cs = []
        if r.Zn is not None:
            cs = [float(abs(r.Zn[:, a] @ r.Zn[:, b])) for a, b in itertools.combinations(r.F_C, 2)]
        split.append({"seed": r.seed, "n": len(r.F_C), "union_width": len(r.selected) if r.Y is not None
                      else r.summary.get("n_distinct_tokens"), "widths": [int(r.widths[i]) for i in r.F_C]
                      if len(r.widths) else [], "cos": cs})
    m["split"] = split

    # 7. noise pull on z
    A = g.prep.get("A") if g.prep else None
    pulls = {}
    if A is not None:
        for r in runs:
            if r.concept is None or r.Y is None:
                continue
            i = r.concept
            S = r.support(i)
            if len(S) < 2:
                continue
            y = r.Y[i, S].astype(np.float64)
            R = A[:, S].astype(np.float64) - r.Z.astype(np.float64) @ r.Y[:, S].astype(np.float64) \
                + np.outer(r.Z[:, i], y)
            o = np.argsort(-np.abs(y), kind="stable")
            h = o[: math.ceil(len(S) / 2)]
            zf = R @ y / (y @ y)
            zh = R[:, h] @ y[h] / (y[h] @ y[h])
            pulls[r.seed] = float(zf @ zh / (np.linalg.norm(zf) * np.linalg.norm(zh) + 1e-30))
    m["noise_pull"] = pulls

    # 8. generic absorption
    absorb = []
    for r in runs:
        w = r.widths
        if not len(w) or w.sum() == 0:
            continue
        top = np.argsort(-w, kind="stable")[:10]
        nonc = [int(i) for i in top if int(i) not in r.F_C]
        absorb.append({"seed": r.seed, "share": float(w[top].sum() / w.sum()),
                       "nonconcept_share": float(w[nonc].sum() / w.sum()),
                       "concept_in_top10": [int(i) for i in top if int(i) in r.F_C],
                       "widest": r.summary.get("widest") or [{"feature": int(i), "width": int(w[i])} for i in top]})
    m["absorption"] = absorb
    m["top10_share"] = float(np.mean([a["share"] for a in absorb])) if absorb else float("nan")
    m["top10_nonconcept_share"] = float(np.mean([a["nonconcept_share"] for a in absorb])) if absorb else float("nan")
    m["top10_has_concept"] = float(np.mean([bool(a["concept_in_top10"]) for a in absorb])) if absorb else float("nan")

    # 9. lock-in
    m["churn"] = [list(map(float, r.fit["churn"])) for r in runs if r.fit is not None and r.fit.get("churn")]

    # 11. judge noise
    jr = []
    for r in runs:
        reps = r.summary.get("judge_reps")
        if reps:
            base = set(r.F_C)
            jr.append({"seed": r.seed, "n_reps": len(reps), "changed": sum(set(map(int, x)) != base for x in reps),
                       "reps": reps, "F_C": r.F_C})
    m["judge_reps"] = jr

    # 6b. tail / neutral tokens
    m["tail"] = None
    return m


def inclusion(g: Group, tracked: dict) -> dict:
    """Per token: fraction of seeds in which it is selected (union of F_C supports) vs its mean rank by |y| in the
    concept feature (dense Y-solve row when A is available, else the sparse row; unselected tokens then rank at the
    support edge). Seeds without a concept feature use the tracked (Hungarian-matched) feature for the rank and count
    as 'not selected'."""
    n = g.runs[0].n
    if not n or g.runs[0].Y is None:
        return {}
    A = g.prep.get("A") if g.prep else None
    ranks, sel = [], np.zeros(n)
    dense = A is not None
    for r in g.runs:
        sel[list(r.selected)] += 1
        i = r.concept if r.concept is not None else tracked.get(r.seed, (None,))[0]
        if i is None or r.Y is None:
            continue
        y = y_dense_row(r, i, A) if dense else r.Y[i]
        rank = np.empty(n)
        o = np.argsort(-np.abs(y), kind="stable")
        rank[o] = np.arange(1, n + 1)
        if not dense:
            w = int((r.Y[i] != 0).sum())
            rank[r.Y[i] == 0] = w + 1
        ranks.append(rank)
    if not ranks:
        return {}
    mean_rank = np.mean(ranks, 0)
    frac = sel / len(g.runs)
    mw = np.nanmean([r.concept_width for r in g.runs]) if any(np.isfinite(r.concept_width) for r in g.runs) else 50
    limit = int(max(3 * mw, 30))
    o = np.argsort(mean_rank, kind="stable")
    keep = o[mean_rank[o] <= mean_rank[o][min(limit, n) - 1]]
    fr = frac[keep]
    ever = frac > 0
    win = max(3, int(round(0.1 * mw)))
    sm = np.convolve(fr, np.ones(3) / 3, mode="same") if len(fr) >= 3 else fr
    drop = max((sm[i] - sm[i + win] for i in range(len(sm) - win)), default=float("nan"))
    trans = int(((frac > 0.1) & (frac < 0.9)).sum())
    return {"mean_rank": mean_rank[keep].tolist(), "frac": fr.tolist(), "tokens": [g.label(int(j)) for j in keep],
            "dense": dense, "n_always": int((frac == 1).sum()), "n_ambiguous": int(((frac > 0) & (frac < 1)).sum()),
            "n_ever": int(ever.sum()), "transition_tokens": trans,
            "transition_ratio": float(trans / mw) if mw else float("nan"), "max_window_drop": float(drop),
            "window": win}


def tail_metrics(g: Group, expected: list) -> dict | None:
    if not expected or not g.prep or "tokens" not in g.prep:
        return None
    toks = [norm_tok(str(t)) for t in g.prep["tokens"]]
    out = {}
    for w in expected:
        idx = [j for j, t in enumerate(toks) if t == w.lower()]
        if not idx:
            out[w] = {"in_vstar": False, "vstar_tokens": [], "frac": float("nan")}
            continue
        hits = [any(j in r.selected for j in idx) for r in g.runs]
        out[w] = {"in_vstar": True, "vstar_tokens": [str(g.prep["tokens"][j]) for j in idx],
                  "frac": float(np.mean(hits))}
    fr = [v["frac"] for v in out.values() if v["in_vstar"]]
    return {"tokens": out, "score": float(np.mean(fr)) if fr else float("nan")}


# ================================================================================================= cross-config
def cross_match(gx: Group, gy: Group) -> dict | None:
    rows = []
    for rx in gx.runs:
        ry = gy.run(rx.seed)
        if ry is None or rx.Zn is None or ry.Zn is None or rx.Zn.shape != ry.Zn.shape:
            continue
        perm, mc = hungarian(rx.Zn, ry.Zn)
        wx, wy = rx.widths.astype(float), ry.widths[perm].astype(float)
        rho = spearmanr(wx, wy).statistic if wx.std() > 0 and wy.std() > 0 else float("nan")
        conf = mc > 0.5
        rho_c = (spearmanr(wx[conf], wy[conf]).statistic
                 if conf.sum() > 5 and wx[conf].std() > 0 and wy[conf].std() > 0 else float("nan"))
        cw = (float(rx.widths[rx.concept]), float(ry.widths[perm[rx.concept]])) if rx.concept is not None else None
        rows.append({"seed": rx.seed, "spearman": float(rho), "spearman_conf": float(rho_c),
                     "mean_cos": float(mc.mean()), "n_conf": int(conf.sum()), "concept_widths": cw,
                     "concept_maps": bool(rx.concept is not None and perm[rx.concept] in ry.F_C)})
    if not rows:
        return None
    sp = [r["spearman"] for r in rows if np.isfinite(r["spearman"])]
    cws = [r["concept_widths"] for r in rows if r["concept_widths"]]
    return {"x": gx.tag, "y": gy.tag, "rows": rows, "median_spearman": float(np.median(sp)) if sp else float("nan"),
            "mean_cos": float(np.mean([r["mean_cos"] for r in rows])),
            "concept_width_ratio": float(np.median([b / a for a, b in cws if a])) if cws else float("nan")}


def resolution(groups: dict, method: str) -> dict | None:
    levels = {}
    for g in groups.values():
        if g.method == method and set(g.diffs) <= {"sparsity"}:
            levels[float(g.diffs.get("sparsity", VWConfig().sparsity))] = g
    base = levels.get(VWConfig().sparsity)
    if base is None or len(levels) < 2:
        return None
    out = {}
    for lv, g in sorted(levels.items()):
        rows = []
        for rb in base.runs:
            r = g.run(rb.seed)
            if r is None or rb.concept is None or r.Zn is None or rb.Zn is None or r.Zn.shape[0] != rb.Zn.shape[0]:
                continue
            c = np.abs(r.Zn.T @ rb.Zn[:, rb.concept])
            i = int(np.argmax(c))
            sup = r.support(i)
            codes = np.array([g.code(int(j)) for j in sup]) if g.prep is not None and len(sup) else np.array([])
            rows.append({"seed": r.seed, "feature": i, "cos": float(c[i]), "width": int(len(sup)),
                         "C": int((codes == 1).sum()), "N": int((codes == 0).sum()), "B": int((codes == 2).sum()),
                         "is_concept": i in r.F_C})
        if rows:
            out[lv] = {"tag": g.tag, "rows": rows, **{k: float(np.mean([r[k] for r in rows]))
                                                       for k in ("width", "C", "N", "B", "cos", "is_concept")},
                       "width_std": float(np.std([r["width"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                       "n": base.runs[0].n}
    return out or None


# ================================================================================================= analysis
class Analysis:
    def __init__(self, concept: str, root: Path | None = None, tags=None, expected=None):
        self.concept, self.root = concept, Path(root) if root else RESULTS
        self.groups = load_groups(concept, self.root, tags)
        self.expected = expected if expected is not None else EXPECTED.get(slug(concept), [])
        for g in self.groups.values():
            g.m = config_metrics(g)
            g.m["tail"] = tail_metrics(g, self.expected)
        self.fixed = find(self.groups, "fixed")
        self.fixed_norefit = find(self.groups, "fixed", refit=False)
        self.base = {mth: find(self.groups, mth) for mth in METHODS}
        self.decision = decide(self)
        w = self.decision.get("winner_method")
        pairs = [("m2", "m3"), ("m1", "m2")]
        if w and w != "m4":
            pairs.append((w, "m4"))
        if w != "m2":
            pairs.append(("m2", "m4"))  # stage-3 reading even before a winner exists
        self.cross = {}
        for a, b in pairs:
            ga, gb = self.base.get(a), self.base.get(b)
            self.cross[f"{a} vs {b}"] = cross_match(ga, gb) if ga and gb else None
        self.decision["widths_real"] = widths_real_rule(self)
        self.resolution = {mth: resolution(self.groups, mth) for mth in METHODS}
        self.resolution = {k: v for k, v in self.resolution.items() if v}

    @property
    def outdir(self) -> Path:
        return self.root / slug(self.concept)


def carry_test(an: Analysis, g: Group) -> dict:
    fx = an.fixed
    res = {"tag": g.tag, "cv": g.m["concept_width_cv"]}
    res["cv_ok"] = bool(np.isfinite(res["cv"]) and res["cv"] < CV_CARRY)
    share = g.m["stats"]["concept_share"]["mean"]
    res["share"] = share
    if fx is None:
        res.update(share_ok=None, direction_ok=None, passes=None, note="fixed + refit baseline not run yet")
        return res
    fshare = fx.m["stats"]["concept_share"]["mean"]
    res["share_fixed"] = fshare
    res["share_ok"] = bool(np.isfinite(share) and np.isfinite(fshare) and share >= fshare - 1e-12)
    t, tf = (g.m["tail"] or {}).get("score", float("nan")), (fx.m["tail"] or {}).get("score", float("nan"))
    nN, nNf = g.m["stats"]["N"]["mean"], fx.m["stats"]["N"]["mean"]
    res.update(tail=t, tail_fixed=tf, N=nN, N_fixed=nNf)
    res["tail_up"] = bool(np.isfinite(t) and np.isfinite(tf) and t > tf)
    res["noise_down"] = bool(np.isfinite(nN) and np.isfinite(nNf) and nN < nNf)
    res["direction_ok"] = res["tail_up"] or res["noise_down"]
    res["passes"] = bool(res["cv_ok"] and res["share_ok"] and res["direction_ok"])
    return res


def absorbs(g: Group) -> bool:
    """10 widest features hold > 25% of K and are not concept features: the non-concept ones among them alone hold
    > 25% of K (mean over seeds)."""
    return bool(g.m["top10_nonconcept_share"] > ABSORB_SHARE)


def decide(an: Analysis) -> dict:
    d = {"fixed_tag": an.fixed.tag if an.fixed else None, "tests": {}, "notes": []}
    for mth in ("m1", "m2", "m3", "m4"):
        g = an.base.get(mth)
        if g is not None:
            d["tests"][mth] = carry_test(an, g)
    for g in an.groups.values():  # ablations of m1/m2 (switches other than matrix/S/sparsity/merge): info only
        if g.method in ("m1", "m2") and g.diffs and not set(g.diffs) & {"matrix", "S", "sparsity", "merge"}:
            d["tests"][g.tag] = carry_test(an, g)
    t1, t2 = d["tests"].get("m1"), d["tests"].get("m2")
    p1, p2 = bool(t1 and t1.get("passes")), bool(t2 and t2.get("passes"))
    d["m1_absorbs"] = absorbs(an.base["m1"]) if an.base.get("m1") else None
    winner = None
    if an.fixed is None or (t1 is None and t2 is None):
        d["notes"].append("stage 1 incomplete (needs fixed + refit and methods 1/2): no decision")
    elif p1 and p2:
        if d["m1_absorbs"]:
            winner = "m2"
            d["notes"].append("both 1 and 2 pass; method 1 shows generic absorption -> prefer method 2")
        else:
            winner = min(("m1", "m2"), key=lambda k: d["tests"][k]["cv"])
            d["notes"].append(f"both 1 and 2 pass, no generic absorption in method 1 -> lower concept-width CV: {winner}")
    elif p1 or p2:
        winner = "m1" if p1 else "m2"
        d["notes"].append(f"only method {winner[1]} passes the carry-forward test")
    else:
        d["notes"].append("no method beats fixed + refit on stage 1 -> stop and reframe (width is a tuned quantity; "
                          "the resolution sweep becomes the contribution)")
    d["stop_and_reframe"] = bool(an.fixed is not None and (t1 or t2) and winner is None)
    d["winner_method"] = winner
    d["winner_tag"] = an.base[winner].tag if winner else (an.fixed.tag if d["stop_and_reframe"] else None)
    return d


def widths_real_rule(an: Analysis) -> dict:
    c23 = an.cross.get("m2 vs m3")
    w = an.decision.get("winner_method")
    c4 = an.cross.get(f"{w} vs m4") if w else None
    r = {"m2_m3_spearman": c23["median_spearman"] if c23 else None,
         "m4_spearman": c4["median_spearman"] if c4 else None,
         "m4_concept_width_ratio": c4["concept_width_ratio"] if c4 else None}
    r["m2_m3_ok"] = None if c23 is None else bool(r["m2_m3_spearman"] > SPEARMAN_REAL)
    r["m4_ok"] = None if c4 is None else bool(r["m4_spearman"] > SPEARMAN_REAL
                                              and abs(r["m4_concept_width_ratio"] - 1) <= M4_WIDTH_TOL)
    r["widths_real"] = None if (r["m2_m3_ok"] is None or r["m4_ok"] is None) else bool(r["m2_m3_ok"] and r["m4_ok"])
    return r


def pick_winner(concept: str, root: Path | None = None) -> str | None:
    """Tag to carry forward for `concept` under the spec's decision rules: the winning method's base config (m1 or m2);
    the fixed + refit tag on "stop and reframe"; None while stage 1 is incomplete."""
    return Analysis(concept, root).decision["winner_tag"]


# ================================================================================================= figures
def color_map(groups) -> dict:
    """Fixed hue per method base config (in METHODS order); everything else muted."""
    cm = {}
    for i, mth in enumerate(METHODS):
        cm[mth] = PALETTE[i]
    return cm


def make_figs(an: Analysis, figdir: Path) -> dict:
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 9})
    figdir.mkdir(parents=True, exist_ok=True)
    cm = color_map(an.groups)
    gs = list(an.groups.values())
    out = {}

    def col(g):
        if g.method == "fixed" and g.diffs == {"refit": False}:
            return PALETTE[6]
        return cm.get(g.method, MUTED) if g.is_base else MUTED

    def grid(n):
        c = min(4, n)
        return math.ceil(n / c), c

    # concept width per seed
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(gs) + 2), 3.2))
    for x, g in enumerate(gs):
        w = [r.concept_width for r in g.runs if np.isfinite(r.concept_width)]
        ax.scatter(np.full(len(w), x) + np.linspace(-0.15, 0.15, len(w)) if w else [], w, s=18, color=col(g), zorder=3)
        if w:
            ax.hlines(np.mean(w), x - 0.3, x + 0.3, color="#3d3d3a", lw=1.5, zorder=4)
    ax.set_xticks(range(len(gs)), [g.tag for g in gs], rotation=45, ha="right")
    ax.set_ylabel("concept feature width (tokens)")
    ax.set_title("Concept-feature width per seed (bar = mean)")
    fig.tight_layout()
    fig.savefig(figdir / "concept_width.png", dpi=150)
    plt.close(fig)
    out["concept_width"] = "figs/concept_width.png"

    # inclusion curves: small multiples
    inc = [g for g in gs if g.m["inclusion"]]
    if inc:
        r_, c_ = grid(len(inc))
        fig, axes = plt.subplots(r_, c_, figsize=(3.2 * c_, 2.5 * r_), squeeze=False, sharey=True)
        for ax, g in zip(axes.flat, inc):
            I = g.m["inclusion"]
            ax.plot(I["mean_rank"], I["frac"], ".", ms=4, color=col(g) if col(g) != MUTED else PALETTE[0])
            mw = g.m["stats"]["concept_width"]["mean"]
            if np.isfinite(mw):
                ax.axvline(mw, color=MUTED, lw=1, ls="--")
            ax.set_title(f"{g.tag} (n={len(g.runs)})")
            ax.set_ylim(-0.05, 1.05)
        for ax in axes.flat[len(inc):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("mean rank by |y| in concept feature")
        for ax in axes[:, 0]:
            ax.set_ylabel("fraction of seeds selected")
        fig.suptitle("Inclusion curves (dashed: mean concept width)", fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / "inclusion.png", dpi=150)
        plt.close(fig)
        out["inclusion"] = "figs/inclusion.png"

    # width histograms
    wh = [g for g in gs if g.m["widths_pooled"]]
    if wh:
        r_, c_ = grid(len(wh))
        fig, axes = plt.subplots(r_, c_, figsize=(3.2 * c_, 2.4 * r_), squeeze=False)
        for ax, g in zip(axes.flat, wh):
            w = np.asarray(g.m["widths_pooled"])
            bw = max(1, math.ceil(w.max() / 40))
            ax.hist(w, bins=np.arange(0, w.max() + 2 * bw, bw), color=col(g) if col(g) != MUTED else PALETTE[0],
                    edgecolor="white", linewidth=0.5)
            mm = g.m["m_min_max"]
            if mm and mm[0] > 0:
                ax.axvline(mm[0], color="#3d3d3a", lw=1, ls=":")
            if mm and mm[1] < g.runs[0].n:
                ax.axvline(mm[1], color="#3d3d3a", lw=1, ls=":")
            cw = [r.concept_width for r in g.runs if np.isfinite(r.concept_width)]
            for x in cw:
                ax.axvline(x, color=PALETTE[7], lw=0.6, alpha=0.6)
            ax.set_title(f"{g.tag}: Gini {g.m['gini']:.2f}")
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.7)
        for ax in axes.flat[len(wh):]:
            ax.set_visible(False)
        fig.supxlabel("feature width (tokens), all k features pooled over seeds", fontsize=9)
        fig.supylabel("features (log)", fontsize=9)
        fig.suptitle("Width distribution (dotted: floor/cap; red: concept features)", fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / "width_hist.png", dpi=150)
        plt.close(fig)
        out["width_hist"] = "figs/width_hist.png"

    # churn: median per config, base configs coloured
    ch = [g for g in gs if g.m["churn"]]
    if ch:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for g in ch:
            L = max(map(len, g.m["churn"]))
            M = np.full((len(g.m["churn"]), L), np.nan)
            for i, c in enumerate(g.m["churn"]):
                M[i, : len(c)] = c
            med = np.nanmedian(M, 0)
            base = col(g) != MUTED
            ax.plot(np.arange(1, L + 1), np.where(med > 0, med, np.nan), lw=2 if base else 1, color=col(g),
                    alpha=1 if base else 0.5, label=g.tag if base else None)
        ax.axhline(CHURN_TOL, color="#3d3d3a", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("support churn (median over seeds)")
        ax.set_title("Lock-in: churn by iteration (dashed: 0.5%; grey: ablations; gaps: zero churn)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / "churn.png", dpi=150)
        plt.close(fig)
        out["churn"] = "figs/churn.png"

    # resolution
    if an.resolution:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), squeeze=False)
        for mth, res in an.resolution.items():
            lv = sorted(res)
            for ax, key in zip(axes[0], ("width", "C", "N")):
                ax.plot(lv, [res[l][key] for l in lv], "o-", color=cm.get(mth, MUTED), lw=2, ms=6, label=mth)
                ax.set_xscale("log")
                ax.set_xticks(lv, [f"{l:.3g}" for l in lv])
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.set_xlabel("total sparsity")
                ax.set_title({"width": "tracked concept width", "C": "C tokens", "N": "N tokens"}[key])
        axes[0][0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / "resolution.png", dpi=150)
        plt.close(fig)
        out["resolution"] = "figs/resolution.png"
    return out


# ================================================================================================= report
def tracked_concept(an: Analysis) -> dict:
    """Follow the concept direction in every run, whether or not the judge kept it.

    Reference per (matrix, S): the top singular vector of the fixed+refit concept features (sign-free mean direction)
    of the base fixed config with the same matrix (whitened references serve S200 too: same transform, other V*).
    In each run the tracked feature is argmax_i |cos(z_ref, z_i)|. Separates "the fit lost the concept" from "the
    judge rejected it".
    """
    refs = {}
    for g in an.groups.values():
        if g.method != "fixed" or set(g.diffs) - {"matrix"}:
            continue
        zs = [r.Zn[:, r.concept] for r in g.runs if r.concept is not None and r.Zn is not None]
        if zs:
            u = np.linalg.svd(np.stack(zs, 1), full_matrices=False)[0][:, 0]
            refs[g.diffs.get("matrix", "whitened")] = u
    out = {}
    for g in an.groups.values():
        u = refs.get(g.diffs.get("matrix", "whitened"))
        if u is None:
            continue
        rows = []
        for r in g.runs:
            if r.Zn is None or r.Y is None:
                continue
            cos = np.abs(u @ r.Zn)
            i = int(cos.argmax())
            sup = r.support(i)
            c = np.array([g.code(int(j)) for j in sup]) if g.prep is not None else np.array([])
            cand = {int(x["feature"]): x for x in r.summary.get("candidates", [])}
            rho = cand[i]["ratio_abs"] if i in cand else float("nan")
            rows.append({"seed": r.seed, "feature": i, "cos": float(cos[i]), "width": int(len(sup)),
                         "C": int((c == 1).sum()), "N": int((c == 0).sum()), "B": int((c == 2).sum()),
                         "rho": float(rho), "candidate": i in cand, "selected": i in r.F_C})
        if rows:
            out[g.tag] = rows
    return out


def tracked_section(an: Analysis) -> list:
    tr = tracked_concept(an)
    L = ["### Concept direction tracked in every run (independent of the judge)", ""]
    if not tr:
        return L + [NOT_RUN, ""]
    L += ["Reference = mean direction of the fixed+refit concept features (same matrix). Per run, the feature with the "
          "highest |cos| to it. *Pure* = at most 5 neutral tokens. Shows whether a method loses the concept in the fit "
          "or whether the judge rejects it.", ""]
    rows = []
    for tag, rs in tr.items():
        df = pd.DataFrame(rs)
        pure = df["N"] <= 5
        rows.append([f"`{tag}`", len(df), f(df["cos"].mean(), 2), mstd(df["width"], 1), mstd(df["C"], 1),
                     mstd(df["N"], 1), f"{int(pure.sum())}/{len(df)}", f"{int(df['candidate'].sum())}/{len(df)}",
                     f"{int(df['selected'].sum())}/{len(df)}",
                     f(df.loc[pure, "width"].mean(), 1) if pure.any() else "—",
                     f(df.loc[pure, "C"].mean(), 1) if pure.any() else "—",
                     f(np.nanmedian(df["rho"]), 2) if df["rho"].notna().any() else "—"])
    L += md_table(["Config", "Runs", "|cos| to ref", "Width", "C", "N", "Pure (N≤5)", "ρ ≥ τ", "Judge kept",
                   "Width when pure", "C when pure", "Median ρ"], rows) + [""]
    return L


def report(an: Analysis, figs: dict) -> str:
    gs = an.groups
    L = [f"# Variable-width sparse MF: {an.concept}", "",
         f"Runs read from `{an.outdir}`. Configs: {len(gs)}; runs: {sum(len(g.runs) for g in gs.values())}. "
         f"Spec: `instructions_new.txt`. Erasure (the 2×2) is not run.", ""]
    if not gs:
        return "\n".join(L + [NOT_RUN, ""])
    fx = an.fixed
    L += ["**Columns.** |F_C|: features kept after τ + judge. Concept feature: the F_C feature with the highest "
          "ratio_abs. Distinct tokens: tokens in the union of the F_C supports. Concept share = C / distinct tokens "
          "(runs without a concept feature excluded). Width CV = std/mean (ddof 1) of the concept feature's width over "
          "seeds. Direction stability: mean pairwise |cos z| of the concept features. Jaccard / core tokens: over the "
          "token sets of all seeds (empty runs included), as in the seed sweep.", ""]

    # --- decision
    d = an.decision
    L += ["## Decision", ""]
    L += [f"- Baseline (fixed 1% + refit): `{d['fixed_tag']}`" if d["fixed_tag"] else "- Baseline fixed + refit: not run yet"]
    L += [f"- {n}" for n in d["notes"]]
    L += [f"- **Recommended config to carry forward: `{d['winner_tag']}`**" if d["winner_tag"]
          else "- **No recommendation yet.**"]
    wr = d["widths_real"]
    L += [f"- Widths real? m2–m3 matched-width Spearman {f(wr['m2_m3_spearman'])} (need > {SPEARMAN_REAL}): "
          f"{_yn(wr['m2_m3_ok'])}; m4 vs winner Spearman {f(wr['m4_spearman'])}, concept-width ratio "
          f"{f(wr['m4_concept_width_ratio'], 2)} (need > {SPEARMAN_REAL} and within ±{M4_WIDTH_TOL:.0%}): "
          f"{_yn(wr['m4_ok'])} → {_yn(wr['widths_real'])}", ""]
    if d["tests"]:
        rows = []
        for k, t in d["tests"].items():
            rows.append([f"`{t['tag']}`", f"{f(t['cv'])} {_yn(t['cv_ok'])}",
                         f"{f(t['share'])} vs {f(t.get('share_fixed'))} {_yn(t.get('share_ok'))}",
                         f"{f(t.get('tail'), 2)} vs {f(t.get('tail_fixed'), 2)} {_yn(t.get('tail_up'))}",
                         f"{f(t.get('N'), 2)} vs {f(t.get('N_fixed'), 2)} {_yn(t.get('noise_down'))}",
                         _yn(t.get("passes"))])
        L += ["Carry-forward test against fixed + refit (CV < 0.15, concept share ≥ fixed, and tail tokens enter "
              "more often or neutral tokens drop). Ablation rows are informational; only the base configs of "
              "methods 1 and 2 can win.", ""]
        L += md_table(["Config", "Width CV", "Concept share", "Tail score", "N tokens", "Passes"], rows) + [""]
        m1 = an.base.get("m1")
        if m1:
            L += [f"Method 1 generic absorption: 10 widest features hold {f(m1.m['top10_share'])} of K, the "
                  f"non-concept ones among them {f(m1.m['top10_nonconcept_share'])} (> {ABSORB_SHARE} → absorbs): "
                  f"{_yn(d['m1_absorbs'])}.", ""]

    # --- summary table
    L += ["## Summary (one row per config)", ""]
    rows = []
    for g in gs.values():
        m, s = g.m, g.m["stats"]
        dist = " ".join(f"{k}:{v}" for k, v in m["n_features_counts"].items())
        rows.append([f"`{g.tag}`", m["n_seeds"], dist,
                     f"{f(s['concept_width']['mean'], 1)} ± {f(s['concept_width']['std'], 1)}",
                     f(m["concept_width_cv"]), f(s["concept_share"]["mean"]), f(s["N"]["mean"], 2),
                     f(m["jaccard"]), f"{m['core50']} / {m['core_all']}", f(m["direction_stability"]),
                     f"{f(s['rel_err']['mean'], 4)} ± {f(s['rel_err']['std'], 4)}", f(s["fit_elapsed"]["mean"], 0)])
    L += md_table(["Config", "Seeds", "|F_C| (count:runs)", "Concept width", "Width CV", "Concept share", "N",
                   "Jaccard", "Core ≥50% / all", "Dir. stab.", "rel_err", "Fit s"], rows) + [""]
    if "concept_width" in figs:
        L += [f"![concept width]({figs['concept_width']})", ""]

    # --- 1 seed-sweep metrics
    L += ["## 1. Seed-sweep metrics", ""]
    keys = ["n_features", "n_tokens", "C", "N", "B", "concept_share", "rel_err", "concept_width", "fit_elapsed",
            "total_elapsed", "n_iter", "lockin_iter"]
    rows = []
    for g in gs.values():
        s = g.m["stats"]
        rows.append([f"`{g.tag}`"] + [f"{f(s[k]['mean'], 4 if k == 'rel_err' else 2)} ± "
                                       f"{f(s[k]['std'], 4 if k == 'rel_err' else 2)}" for k in keys])
    L += md_table(["Config"] + keys, rows) + [""]
    rows = [[f"`{g.tag}`", g.m["union_tokens"], f(g.m["jaccard"]), g.m["core50"], g.m["core_all"]] for g in gs.values()]
    L += ["Token stability across seeds:", ""]
    L += md_table(["Config", "Tokens in ≥1 seed", "Mean pairwise Jaccard", "In ≥50% of seeds", "In all seeds"], rows)
    L += [""]
    for g in gs.values():
        n = len(g.runs)
        L += [f"**`{g.tag}`: most frequent tokens** (seeds, of {n}): "
              + (", ".join(f"`{t}` ({c})" for t, c in g.m["most_frequent"]) or "—"),
              "", "Most frequent neutral tokens: " + (", ".join(f"`{t}` ({c})" for t, c in g.m["most_frequent_neutral"])
                                                     or "none"), ""]

    # --- 2/3 width & direction stability
    L += ["## 2–3. Width and direction stability", "",
          f"Pass bars: width CV well below {CV_RANDOM_BUDGET} (the random budget's CV); direction stability at least "
          f"the fixed baseline's ({f(fx.m['direction_stability']) if fx else 'not run yet'}). 'Concept→concept': "
          "fraction of seed pairs where Hungarian matching over all k×k |cos z| sends one seed's concept feature to "
          "the other's. 'Tracked': the most central seed's concept feature matched into every seed (also seeds "
          "without a concept feature).", ""]
    rows = []
    for g in gs.values():
        m = g.m
        tr = m["tracked"]
        rows.append([f"`{g.tag}`", " ".join(f(w, 0) for w in [r.concept_width for r in g.runs]),
                     f(m["concept_width_cv"]), f(m["tracked_width_cv"]),
                     f(m["direction_stability"]), f(m["concept_maps_to_concept"], 2), f(m["concept_matched_cos"]),
                     f"{sum(v['is_concept'] for v in tr.values())}/{len(tr)}" if tr else "—"])
    L += md_table(["Config", "Concept width per seed", "CV", "Tracked CV", "Dir. stab.", "Concept→concept",
                   "Matched |cos|", "Tracked is F_C"], rows) + [""]

    # --- 4 inclusion
    L += ["## 4. Inclusion curve", "",
          "Per token: fraction of seeds where it is in the union of the F_C supports, against its mean rank by |y| in "
          "the concept feature (dense Y-solve row (ZᵀZ+λI)⁻¹ZᵀA when the prep A is available). A real edge shows as "
          "a drop: few tokens with inclusion strictly between 0.1 and 0.9 relative to the width (transition ratio, "
          "lower = sharper) and a large drop over a window of ~10% of the width (max window drop, higher = sharper).", ""]
    rows = []
    for g in gs.values():
        I = g.m["inclusion"]
        if not I:
            rows.append([f"`{g.tag}`", "—", "—", "—", "—", "—", "—"])
            continue
        rows.append([f"`{g.tag}`", "dense" if I["dense"] else "sparse", I["n_ever"], I["n_always"],
                     I["transition_tokens"], f(I["transition_ratio"], 2), f"{f(I['max_window_drop'], 2)} (w={I['window']})"])
    L += md_table(["Config", "Rank from", "Ever selected", "Always", "0.1<frac<0.9", "Transition ratio",
                   "Max window drop"], rows) + [""]
    if "inclusion" in figs:
        L += [f"![inclusion]({figs['inclusion']})", ""]

    # --- 5 width distribution
    L += ["## 5. Width distribution (all k features, pooled over seeds)", "",
          "Pass bar: few rows pinned at the floor (m_min) or cap (m_max).", ""]
    rows = []
    for g in gs.values():
        w = np.asarray(g.m["widths_pooled"])
        mm = g.m["m_min_max"]
        rows.append([f"`{g.tag}`", f"{mm[0]} / {mm[1]}" if mm else "—", f(g.m["gini"]),
                     f(g.m["at_floor"], 1), f(g.m["at_cap"], 1), f(g.m["dead_rows"], 1),
                     f"{int(w.min())}–{int(np.median(w))}–{int(w.max())}" if len(w) else "—"])
    L += md_table(["Config", "m_min / m_max", "Gini", "Rows at floor", "Rows at cap", "Empty rows",
                   "min–median–max width"], rows) + [""]
    if "width_hist" in figs:
        L += [f"![width histogram]({figs['width_hist']})", ""]

    # --- 6 split, tail, noise
    L += ["## 6. Concept split, tail tokens, neutral tokens", "", "### Split", ""]
    rows = []
    for g in gs.values():
        sp = g.m["split"]
        two = [s for s in sp if s["n"] >= 2]
        cosv = [c for s in two for c in s["cos"]]
        rows.append([f"`{g.tag}`", " ".join(str(s["n"]) for s in sp), f(np.mean([s["union_width"] for s in sp
                                                                                 if s["union_width"] is not None]), 1),
                     len(two), " ".join(f"{c:.2f}" for c in cosv) or "—"])
    L += md_table(["Config", "|F_C| per seed", "Mean union width", "Seeds with ≥2", "|cos z| between them"], rows) + [""]
    L += ["### Tail tokens", ""]
    tails = {k: g.m["tail"] for k, g in gs.items() if g.m["tail"]}
    if not an.expected:
        L += ["No expected tail list for this concept (pass `--expected`).", ""]
    elif not tails:
        L += ["Needs the prep tokens; not available.", ""]
    else:
        any_t = next(iter(tails.values()))["tokens"]
        absent = [w for w, v in any_t.items() if not v["in_vstar"]]
        L += ["Fraction of seeds in which any V* token equal to the word (case-insensitive, `▁` stripped, whole token) "
              "is in the F_C support.", ""]
        hdr = ["Token", "V* tokens"] + [f"`{k}`" for k in tails]
        rows = []
        for w, v in any_t.items():
            if not v["in_vstar"]:
                continue
            rows.append([w, " ".join(f"`{t}`" for t in v["vstar_tokens"])]
                        + [f(tails[k]["tokens"][w]["frac"], 2) for k in tails])
        rows.append(["**mean**", ""] + [f(tails[k]["score"], 2) for k in tails])
        L += md_table(hdr, rows) + [""]
        L += [f"Not in V* (cannot enter): {', '.join(absent) if absent else 'none'}. Note: with |S| ≠ 300 the V* "
              "differs, so the in-V* set can differ per config.", ""]
    L += ["### Neutral tokens and concept width vs fixed", ""]
    rows = []
    fw = fx.m["stats"]["concept_width"]["mean"] if fx else float("nan")
    for g in gs.values():
        s = g.m["stats"]
        rows.append([f"`{g.tag}`", f(s["N"]["mean"], 2), f(s["B"]["mean"], 2), f(s["concept_width"]["mean"], 1),
                     f(s["concept_width"]["mean"] / fw, 2) if np.isfinite(fw) else "—",
                     ", ".join(f"`{t}` ({c})" for t, c in g.m["most_frequent_neutral"][:5]) or "—"])
    L += md_table(["Config", "N", "B", "Concept width", "Width / fixed", "Top neutral tokens"], rows) + [""]

    L += tracked_section(an)

    # --- 7 noise pull
    L += ["## 7. Noise pull on z", "",
          "z refit on the residual R = A − Σ_{l≠i} z_l y_l of the concept feature i: z = R_S y_S / ‖y_S‖² on the full "
          "support and on its top half by |y|. Lower cos(z_full, z_half) means the tail moved z.", ""]
    rows = [[f"`{g.tag}`", len(g.m["noise_pull"]), mstd(list(g.m["noise_pull"].values())),
             f(min(g.m["noise_pull"].values()) if g.m["noise_pull"] else float("nan"))]
            for g in gs.values()]
    if all(r[1] == 0 for r in rows):
        L += ["Needs the prep A; not available.", ""]
    else:
        L += md_table(["Config", "Seeds", "cos(z_full, z_half)", "min"], rows) + [""]

    # --- 8 generic absorption
    L += ["## 8. Generic absorption", "",
          f"Share of K held by the 10 widest features (rule: > {ABSORB_SHARE} and not concept features, read as: the "
          "non-concept features among the 10 widest hold > 25% of K). "
          "Descriptions: stage-1 judge descriptions of the median-share seed.", ""]
    rows = [[f"`{g.tag}`", mstd([a["share"] for a in g.m["absorption"]]),
             mstd([a["nonconcept_share"] for a in g.m["absorption"]]),
             f"{sum(a['share'] > ABSORB_SHARE for a in g.m['absorption'])}/{len(g.m['absorption'])}",
             f(g.m["top10_has_concept"], 2)] for g in gs.values()]
    L += md_table(["Config", "Top-10 share of K", "Non-concept top-10 share", "Seeds > 25%", "Seeds with a concept feature in top 10"], rows) + [""]
    for g in gs.values():
        ab = g.m["absorption"]
        if not ab or g.method == "fixed":
            continue
        med = sorted(ab, key=lambda a: a["share"])[len(ab) // 2]
        L += [f"<details><summary><code>{g.tag}</code> seed {med['seed']}: 10 widest (share {med['share']:.3f})</summary>",
              ""]
        L += md_table(["Feature", "Width", "Description"],
                      [[f"f{w['feature']}" + (" **(F_C)**" if w["feature"] in g.run(med["seed"]).F_C else ""),
                        w.get("width", "—"), str(w.get("description", "—")).replace("|", "/")] for w in med["widest"]])
        L += ["", "</details>", ""]

    # --- 9 lock-in
    L += ["## 9. Lock-in", "", f"Lock-in iteration: the iteration after which churn stays under {CHURN_TOL:.1%}.", ""]
    rows = []
    for g in gs.values():
        s = g.m["stats"]
        stops = Counter((r.fit or {}).get("stop_reason", "?") for r in g.runs)
        rows.append([f"`{g.tag}`", f"{f(s['lockin_iter']['median'], 0)} [{f(s['lockin_iter']['min'], 0)}–"
                                  f"{f(s['lockin_iter']['max'], 0)}]",
                     f"{f(s['n_iter']['median'], 0)} [{f(s['n_iter']['min'], 0)}–{f(s['n_iter']['max'], 0)}]",
                     ", ".join(f"{k} ×{v}" for k, v in stops.items())])
    L += md_table(["Config", "Lock-in iter median [min–max]", "Iterations median [min–max]", "Stop reasons"], rows)
    L += [""] + ([f"![churn]({figs['churn']})", ""] if "churn" in figs else [])

    # --- 10 cross-method
    L += ["## 10. Cross-method agreement", "",
          "Per seed: Hungarian matching of the two configs' features by |cos z|, Spearman of the matched widths over "
          f"all k features (and over confident matches, |cos| > 0.5). Rule: m2 vs m3 > {SPEARMAN_REAL}; m4 close to "
          "the winner.", ""]
    rows = []
    for k, c in an.cross.items():
        if c is None:
            rows.append([k, NOT_RUN, "", "", "", ""])
            continue
        rows.append([k, len(c["rows"]), f(c["median_spearman"]),
                     f(np.nanmedian([r["spearman_conf"] for r in c["rows"]]) if any(np.isfinite(r["spearman_conf"])
                                                                                   for r in c["rows"]) else float("nan")),
                     f(c["mean_cos"]), f(c["concept_width_ratio"], 2)])
    L += md_table(["Pair", "Seeds", "Median Spearman (all k)", "Median Spearman (confident)", "Mean matched |cos|",
                   "Concept width ratio y/x"], rows) + [""]

    # --- 11 judge noise
    L += ["## 11. Judge noise", ""]
    jr = {k: g.m["judge_reps"] for k, g in gs.items() if g.m["judge_reps"]}
    if not jr:
        L += [NOT_RUN + " (needs runs with `--judge-reps`)", ""]
    else:
        rows = []
        for k, lst in jr.items():
            tot = sum(x["n_reps"] for x in lst)
            ch = sum(x["changed"] for x in lst)
            rows.append([f"`{k}`", len(lst), tot, f"{ch}/{tot} ({ch / tot:.0%})" if tot else "—",
                         "; ".join(f"s{x['seed']}: {x['F_C']} → {x['reps']}" for x in lst if x["changed"]) or "—"])
        L += md_table(["Config", "Frozen fits", "Reruns", "F_C changed", "Changes"], rows) + [""]

    # --- 12 resolution
    L += ["## 12. Resolution sweep", ""]
    if not an.resolution:
        L += [NOT_RUN + " (needs configs differing only in sparsity)", ""]
    for mth, res in an.resolution.items():
        L += [f"Method `{mth}`: the 1% concept feature tracked to each level by max |cos z| (same seed).", ""]
        rows = []
        lv = sorted(res)
        for i, l in enumerate(lv):
            r = res[l]
            gain = ""
            if i:
                dC, dN = r["C"] - res[lv[i - 1]]["C"], r["N"] - res[lv[i - 1]]["N"]
                gain = f"ΔC {dC:+.1f}, ΔN {dN:+.1f}"
            rows.append([f"{l:.3g}", f"`{r['tag']}`", len(r["rows"]), f"{r['width']:.1f} ± {r['width_std']:.1f}",
                         f"{100 * r['width'] / r['n']:.2f}%", f(r["C"], 1), f(r["N"], 1), f(r["B"], 1), f(r["cos"]),
                         f(r["is_concept"], 2), gain])
        L += md_table(["Sparsity", "Config", "Seeds", "Width", "Width / |V*|", "C", "N", "B", "|cos| to 1%",
                       "Is F_C", "Gain vs previous"], rows) + [""]
        L += ["Reading: a broad concept keeps gaining C tokens at each level; a narrow one starts gaining N tokens "
              "past its edge.", ""]
    if "resolution" in figs:
        L += [f"![resolution]({figs['resolution']})", ""]

    # --- 13 V* dependence
    L += ["## 13. V* dependence (|S|)", ""]
    vrows = []
    for mth in METHODS:
        fam = [g for g in gs.values() if g.method == mth and set(g.diffs) <= {"S"}]
        if len(fam) < 2:
            continue
        for g in sorted(fam, key=lambda g: g.diffs.get("S", DEFAULT_S)):
            s = g.m["stats"]
            n = g.runs[0].n
            vrows.append([f"`{g.tag}`", g.diffs.get("S", DEFAULT_S), n, mstd([r.concept_width for r in g.runs], 1),
                          f"{100 * s['concept_width']['mean'] / n:.2f}%" if n else "—", f(s["C"]["mean"], 1),
                          f(s["N"]["mean"], 1), f(s["n_tokens"]["mean"], 1), f(s["concept_share"]["mean"])])
    if vrows:
        L += ["Different V*, so widths and counts are compared, not tokens by index.", ""]
        L += md_table(["Config", "|S|", "|V*|", "Concept width", "Width / |V*|", "C", "N", "Distinct", "Share"], vrows)
        L += [""]
    else:
        L += [NOT_RUN + " (needs a config differing only in |S|)", ""]

    # --- 14 merge
    L += ["## 14. Merge step (method 5)", ""]
    mg = [g for g in gs.values() if g.diffs.get("merge")]
    if not mg:
        L += [NOT_RUN, ""]
    for g in mg:
        rows, allc, nacc = [], [], 0
        for r in g.runs:
            mi = merge_info(r.summary.get("merge") or {})
            allc += mi["cos"]
            nacc += bool(mi["accepted"])
            rows.append([r.seed, " ".join(f"f{a}–f{b}" for a, b in mi["pairs"]) or "none",
                         " ".join(f"{c:.2f}" for c in mi["pair_cos"]) or "—", _yn(mi["accepted"]),
                         f(mi["before"], 4), f(mi["after"], 4), f(mi["rise"], 4), mi["reason"]])
        L += [f"`{g.tag}`: {nacc}/{len(g.runs)} seeds accepted a merge (threshold "
              f"{merge_info(g.runs[0].summary.get('merge') or {})['threshold']}). All concept-pair |cos z| "
              f"(log for setting the threshold): {' '.join(f'{c:.2f}' for c in sorted(allc)) or 'none'}.", ""]
        L += md_table(["Seed", "Qualifying pairs", "Their |cos z|", "Accepted", "rel_err before", "after", "Rise",
                       "Note"], rows) + [""]

    # --- per-run tables
    L += ["## Per-run results", ""]
    for g in gs.values():
        L += [f"### `{g.tag}`", "", "Switches: " + (", ".join(f"{k}={v}" for k, v in g.diffs.items()) or "none")
              + f" (method {g.method})", ""]
        rows = []
        for r in g.runs:
            s = r.summary
            tpf = ", ".join(f"f{i}: {int(r.widths[i]) if len(r.widths) > i else '?'}" + ("*" if i == r.concept else "")
                            for i in r.F_C) or "—"
            rows.append([r.seed, len(r.F_C), s.get("n_distinct_tokens", "—"), tpf,
                         f"{s.get('C', '—')} / {s.get('N', '—')} / {s.get('B', '—')}", f(s.get("rel_err"), 4),
                         s.get("n_iter", "—"), s.get("lockin_iter", "—"), f(s.get("fit_elapsed"), 0)])
        L += md_table(["Seed", "|F_C|", "Distinct tokens", "Tokens per feature (* concept)", "C / N / B", "rel_err",
                       "Iters", "Lock-in", "Fit s"], rows) + [""]
    L += ["## Files", "", "- `report.md` (this file), `metrics.json`, `runs.csv`, `figs/`.",
          f"- Regenerate: `python -m experiments.varwidth.evaluate --concept \"{an.concept}\"`.", ""]
    return "\n".join(L)


def merge_info(mr: dict) -> dict:
    """Normalise summary["merge"] (run.py schema: member_pairs/qualifying_pairs/source_rel_err/rel_err_merged/
    rel_err_rise; or the fake_runs schema: concept_pair_cosines/pairs/rel_err_before/rel_err_after)."""
    if "member_pairs" in mr:
        cos_of = {(p["a"], p["b"]): p.get("abs_cos") for p in mr["member_pairs"]}
        allc = [p.get("abs_cos") for p in mr["member_pairs"] if p.get("both_in_F_C", True)]
        pairs = [tuple(x) for x in mr.get("qualifying_pairs", [])]
        before, after = mr.get("source_rel_err"), mr.get("rel_err_merged")
        rise = mr.get("rel_err_rise")
        thr = mr.get("cos_threshold")
    else:
        cos_of = {(p["a"], p["b"]): p.get("cos") for p in mr.get("concept_pair_cosines", [])}
        allc = list(cos_of.values())
        qp = [p for p in mr.get("pairs", []) if "rel_err_after" in p]
        pairs = [(p["a"], p["b"]) for p in qp]
        before = qp[0].get("rel_err_before") if qp else mr.get("rel_err_before")
        after = qp[0].get("rel_err_after") if qp else None
        rise = (after - before) if (after is not None and before is not None) else None
        thr = mr.get("threshold")
    return {"cos": [c for c in allc if c is not None], "pairs": pairs,
            "pair_cos": [cos_of.get(p, cos_of.get(p[::-1])) or float("nan") for p in pairs],
            "accepted": mr.get("accepted"), "before": before, "after": after, "rise": rise, "threshold": thr,
            "reason": str(mr.get("reason", ""))}


def _yn(x) -> str:
    return "—" if x is None else ("✓" if x else "✗")


def runs_frame(an: Analysis) -> pd.DataFrame:
    rows = []
    for g in an.groups.values():
        for r in g.runs:
            s = r.summary
            ab = next((a for a in g.m["absorption"] if a["seed"] == r.seed), None)
            rows.append({"tag": g.tag, "method": g.method, "switches": json.dumps(g.diffs), "seed": r.seed,
                         "n_features": len(r.F_C), "F_C": " ".join(map(str, r.F_C)), "concept_feature": r.concept,
                         "concept_width": r.concept_width, "n_tokens": s.get("n_distinct_tokens"), "C": s.get("C"),
                         "N": s.get("N"), "B": s.get("B"), "concept_share": r.share, "rel_err": s.get("rel_err"),
                         "n_iter": s.get("n_iter"), "lockin_iter": s.get("lockin_iter"),
                         "fit_elapsed": s.get("fit_elapsed"), "total_elapsed": s.get("total_elapsed"),
                         "top10_share": ab["share"] if ab else None,
                         "noise_pull_cos": g.m["noise_pull"].get(r.seed),
                         "gini": gini(r.widths) if len(r.widths) else None})
    return pd.DataFrame(rows)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def metrics_json(an: Analysis) -> dict:
    per = {}
    for k, g in an.groups.items():
        m = {kk: v for kk, v in g.m.items() if kk not in ("widths_pooled", "churn")}
        if m.get("inclusion"):
            m["inclusion"] = {kk: v for kk, v in m["inclusion"].items() if kk not in ("mean_rank", "frac", "tokens")}
        m["method"], m["switches"], m["seeds"] = g.method, g.diffs, g.seeds
        per[k] = m
    return _jsonable({"concept": an.concept, "configs": per, "cross_method": an.cross, "resolution": an.resolution,
                      "decision": an.decision})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--tags", nargs="*", default=None, help="only these tag directories")
    ap.add_argument("--root", type=Path, default=None, help="results root (default results/varwidth)")
    ap.add_argument("--expected", nargs="*", default=None, help="tail tokens expected in the concept support")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    an = Analysis(a.concept, a.root, a.tags, a.expected)
    out = an.outdir
    out.mkdir(parents=True, exist_ok=True)
    figs = {} if (a.no_figs or not an.groups) else make_figs(an, out / "figs")
    (out / "report.md").write_text(report(an, figs))
    (out / "metrics.json").write_text(json.dumps(metrics_json(an), indent=1))
    if an.groups:
        runs_frame(an).to_csv(out / "runs.csv", index=False)
    d = an.decision
    print(f"[evaluate] {a.concept}: {len(an.groups)} configs, {sum(len(g.runs) for g in an.groups.values())} runs "
          f"-> {out / 'report.md'}")
    for n in d["notes"]:
        print("  -", n)
    print(f"  winner: {d['winner_tag']}")


if __name__ == "__main__":
    main()
