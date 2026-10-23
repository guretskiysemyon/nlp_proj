#!/usr/bin/env python
"""WP2 step 4: concept splits. How often is a concept spread over >= 2 features, under L0 / L1 / L12?

    .venv/bin/python -m experiments.paper_analysis.feature_geometry.splits items     # blind items to label
    .venv/bin/python -m experiments.paper_analysis.feature_geometry.splits report    # after labeling

Per run: lexicon-matching features (lo >= 5 = EJ@5, any rho rank; also lo >= 3), the EMBER-judge set (TJ, F_C of
the top-20 by rho) and CINDER's judge set (HJ = TJ or EJ@5 within the top-20). A run "splits" under a definition
when >= 2 features qualify.
Split pairs (both features lo >= 5, or both HJ-accepted): sign-aligned cos (fit and raw space), Jaccard of the
supports, widths. Judge-free "cut by width" test: the share of the pair's union support that the single widest
lexicon-matching L12 feature of the same seed covers (V* and token order are shared across configs).
Blind labels: items (pairs of word lists, configs hidden, shuffled) in split_items_blind.csv; labels in
split_labels.json {item_id: SUB | CUT | OFF}: SUB = two distinct sub-topics of the concept, CUT = same kind of
tokens divided between two features, OFF = at least one of the two is not about the concept.
Outputs: splits_runs.csv, split_pairs.csv, splits_stats.json, ancient_rome.md.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.feature_geometry.common import (CONFIGS, OUT, full, load_features, load_pairs,
                                                                paired)
from experiments.varwidth.common import run_dir

ITEMS = OUT / "split_items_blind.csv"
KEY = OUT / "split_items_key.csv"       # unblinding key (not to be read while labeling)
LABELS = OUT / "split_labels.json"


def supports(concept: str, cfg: str, seed: int) -> list:
    Y = torch.load(run_dir(concept, CONFIGS[cfg], seed) / "fit.pt", map_location="cpu", weights_only=False)["Y"]
    S = (Y.abs() > 1e-12).numpy()
    return [set(np.flatnonzero(r)) for r in S]


def build(F: pd.DataFrame, words: dict):
    runs, pairs = [], []
    sup_cache = {}

    def sup(c, cfg, s):
        if (c, cfg, s) not in sup_cache:
            sup_cache[(c, cfg, s)] = supports(c, cfg, s)
        return sup_cache[(c, cfg, s)]

    for (c, s), gs in F.groupby(["concept", "seed"], sort=False):
        g12 = gs[(gs.cfg == "L12") & (gs.lo >= 5)]
        big = None
        if len(g12):
            f12 = int(g12.sort_values(["lo", "width"], ascending=False).iloc[0].feature)
            big = sup(c, "L12", s)[f12]
        for cfg, g in gs.groupby("cfg", sort=False):
            sets = {"lex5": g[g.lo >= 5], "lex3": g[g.lo >= 3], "tj": g[g.tj], "hj": g[g.hj]}
            runs.append({"concept": c, "cfg": cfg, "seed": s, **{f"n_{k}": len(v) for k, v in sets.items()},
                         "conc_width": int(sets["lex5"].width.sum()) if len(sets["lex5"]) else 0})
            P = load_pairs(c, cfg, s)
            Cf, Cr, J = full(P["cos_fit"]), full(P["cos_raw"]), full(P["jac"])
            S = sup(c, cfg, s)
            for kind in ("lex5", "hj"):
                ids = sorted(sets[kind].feature.astype(int))
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        i, j = ids[a], ids[b]
                        u = S[i] | S[j]
                        pairs.append({"concept": c, "cfg": cfg, "seed": s, "kind": kind, "f1": i, "f2": j,
                                      "cos_fit": Cf[i, j], "cos_raw": Cr[i, j], "jac": J[i, j],
                                      "w1": len(S[i]), "w2": len(S[j]),
                                      "lo1": int(g.set_index("feature").lo[i]), "lo2": int(g.set_index("feature").lo[j]),
                                      "l12_cover": (len(u & big) / len(u)) if big is not None else np.nan,
                                      "l12_cover_each": (min(len(S[i] & big) / len(S[i]), len(S[j] & big) / len(S[j]))
                                                         if big is not None else np.nan),
                                      "words1": ", ".join(words[f"{c}|{cfg}|{s}|{i}"][:12]),
                                      "words2": ", ".join(words[f"{c}|{cfg}|{s}|{j}"][:12])})
    return pd.DataFrame(runs), pd.DataFrame(pairs)


def items(Pp: pd.DataFrame):
    """Unique (concept, words1, words2) items of lex5 split pairs, configs hidden, shuffled."""
    D = Pp[Pp.kind == "lex5"].copy()
    D["item"] = D.apply(lambda r: " || ".join([r.concept] + sorted([r.words1, r.words2])), axis=1)
    U = D.drop_duplicates("item")[["item", "concept", "words1", "words2"]].reset_index(drop=True)
    rng = random.Random(7)
    order = list(range(len(U)))
    rng.shuffle(order)
    U = U.iloc[order].reset_index(drop=True)
    U["item_id"] = [f"i{n:03d}" for n in range(len(U))]
    # randomize which list is shown first
    flip = [rng.random() < 0.5 for _ in range(len(U))]
    U["A"] = [b if f else a for a, b, f in zip(U.words1, U.words2, flip)]
    U["B"] = [a if f else b for a, b, f in zip(U.words1, U.words2, flip)]
    U[["item_id", "concept", "A", "B"]].to_csv(ITEMS, index=False)
    D.merge(U[["item", "item_id"]], on="item")[["item_id", "concept", "cfg", "seed", "f1", "f2"]].to_csv(KEY, index=False)
    print(f"{len(U)} blind items -> {ITEMS}")


def report(Rn: pd.DataFrame, Pp: pd.DataFrame, F: pd.DataFrame, words: dict):
    st = {"runs": {}, "paired": {}, "pairs": {}}
    for k in ("lex5", "lex3", "tj", "hj"):
        Rn[f"split_{k}"] = Rn[f"n_{k}"] >= 2
        Rn[f"has_{k}"] = Rn[f"n_{k}"] >= 1
    for cfg in CONFIGS:
        r = Rn[Rn.cfg == cfg]
        st["runs"][cfg] = {"n_runs": len(r), **{f"frac_split_{k}": float(r[f"split_{k}"].mean()) for k in
                                                ("lex5", "lex3", "tj", "hj")},
                           **{f"frac_split_given_has_{k}": float(r[f"split_{k}"].sum() / max(r[f"has_{k}"].sum(), 1))
                              for k in ("lex5", "tj", "hj")},
                           **{f"mean_n_{k}": float(r[f"n_{k}"].mean()) for k in ("lex5", "lex3", "tj", "hj")},
                           "concepts_with_any_split_lex5": int(r.groupby("concept").split_lex5.any().sum())}
    Cc = Rn.groupby(["concept", "cfg"]).mean(numeric_only=True).reset_index()
    for k in ("lex5", "hj", "tj"):
        p = Cc.pivot(index="concept", columns="cfg", values=f"split_{k}")
        st["paired"][f"split_{k}"] = {f"L0->{b}": paired(p["L0"], p[b]) for b in ("L1", "L12")} | \
                                     {"L1->L12": paired(p["L1"], p["L12"])}
    for (kind, cfg), d in Pp.groupby(["kind", "cfg"]):
        st["pairs"][f"{kind}_{cfg}"] = {"n_pairs": len(d), "n_concepts": int(d.concept.nunique()),
                                        "median_cos_fit": float(d.cos_fit.median()),
                                        "median_cos_raw": float(d.cos_raw.median()),
                                        "median_angle_fit": float(np.degrees(np.arccos(d.cos_fit.clip(-1, 1))).median()),
                                        "median_jac": float(d.jac.median()), "frac_jac0": float((d.jac == 0).mean()),
                                        "median_l12_cover": float(d.l12_cover.median()),
                                        "median_l12_cover_each": float(d.l12_cover_each.median())}
    # background: all feature pairs of the same runs (for the cos / jac reference)
    if LABELS.exists():
        lab = json.loads(LABELS.read_text())
        key = pd.read_csv(KEY)
        key["label"] = key.item_id.map(lab)
        st["labels"] = {"n_items": len(lab), "counts_items": pd.Series(lab).value_counts().to_dict(),
                        "by_cfg_pairs": key.groupby("cfg").label.value_counts().unstack(fill_value=0).to_dict("index")}
        Pl = Pp[Pp.kind == "lex5"].merge(key[["concept", "cfg", "seed", "f1", "f2", "label"]],
                                         on=["concept", "cfg", "seed", "f1", "f2"], how="left")
        st["labels"]["by_label"] = {l: {"n": len(d), "median_cos_fit": float(d.cos_fit.median()),
                                        "median_jac": float(d.jac.median()),
                                        "median_l12_cover": float(d.l12_cover.median())}
                                    for l, d in Pl.groupby("label")}
        Pl.to_csv(OUT / "split_pairs_labeled.csv", index=False)
    (OUT / "splits_stats.json").write_text(json.dumps(st, indent=1, default=float))
    Rn.to_csv(OUT / "splits_runs.csv", index=False)
    Pp.to_csv(OUT / "split_pairs.csv", index=False)

    # ---------------- Ancient Rome dossier
    L = ["# Ancient Rome: lexicon-matching features (lo >= 3) per config and seed", "",
         "lo = lexicon hits in the first 20 source words; width = support size; first 14 words shown.", ""]
    ar = F[(F.concept == "Ancient Rome") & (F.lo >= 3)].sort_values(["cfg", "seed", "lo"], ascending=[True, True, False])
    for (cfg, s), g in ar.groupby(["cfg", "seed"]):
        L.append(f"## {cfg} seed {s}")
        for r in g.itertuples():
            L.append(f"- f{r.feature} lo={r.lo} width={r.width} rho={r.rho:.1f} hj={r.hj} tj={r.tj}: "
                     f"{', '.join(words[f'Ancient Rome|{cfg}|{s}|{r.feature}'][:14])}")
        pp = Pp[(Pp.concept == "Ancient Rome") & (Pp.cfg == cfg) & (Pp.seed == s) & (Pp.kind == "lex5")]
        for r in pp.itertuples():
            L.append(f"  - pair f{r.f1}/f{r.f2}: cos_fit {r.cos_fit:+.2f} (angle {np.degrees(np.arccos(r.cos_fit)):.0f}°), "
                     f"cos_raw {r.cos_raw:+.2f}, Jaccard {r.jac:.2f}, L12 covers {r.l12_cover:.0%} of the union")
        L.append("")
    (OUT / "ancient_rome.md").write_text("\n".join(L))
    print(json.dumps(st, indent=1, default=float)[:8000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("items", "report"))
    a = ap.parse_args()
    F = load_features()
    words = json.loads((OUT / "words.json").read_text())
    Rn, Pp = build(F, words)
    if a.what == "items":
        items(Pp)
    else:
        report(Rn, Pp, F, words)


if __name__ == "__main__":
    main()
