"""WP4 analysis: per-feature judge-free stats for every fit, blind-label export / import, per-run metrics.

    .venv/bin/python -m experiments.paper_analysis.robustness.analyze features     # -> features.parquet
    .venv/bin/python -m experiments.paper_analysis.robustness.analyze export       # -> labels/batch_*.txt + key
    .venv/bin/python -m experiments.paper_analysis.robustness.analyze runs         # -> runs_A.csv, runs_B.csv

Per feature (all 100 of every fit): rho (= ember.utils.compute_embedding_stats ratio_abs: mean |Y| over concept-only
tokens / (mean |Y| over neutral-only + 1e-8)), its rank by rho (1 = highest), width, the first 20 distinct source
words (overlap_judge.feature_words over the variant's sentences, as j3_words.gen_features), lexicon hits among them
(j3_eval.j3_count; lexicon = concepts.lexicon), the concept-only share of the 20 strongest tokens, and (part B) the
intruder share: tokens of the 20 strongest that occur in the inserted adjacent sentences but not in the clean S_C.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.robustness.concepts import (ABSTRACT_NEW, ABSTRACT_OLD, ABSTRACT_SHORT, ADJACENT, DATA,
                                                            ENTITY, FITS, OUT, PARENTS, lexicon, slug)
from experiments.paper_analysis.robustness.pipeline import META, VARIANTS, load_Y

TOPN = 20          # tokens / source words per feature
TOP_RHO = 10       # features per run shown for blind labelling
LABELS = OUT / "labels"


def group_of(variant: str, concept: str) -> str:
    if ("__f" in variant or "__u" in variant) and not variant.endswith("__f00"):
        return "B"
    if concept in ABSTRACT_NEW or concept in ABSTRACT_OLD or concept in ABSTRACT_SHORT:
        return "abstract"
    if concept in ENTITY:
        return "entity"
    return "other"


def rho_stats(Y: torch.Tensor, codes: np.ndarray) -> np.ndarray:
    G = Y.abs().numpy()
    c, n = codes == 1, codes == 0
    mc = G[:, c].mean(1) if c.any() else np.zeros(G.shape[0])
    mn = G[:, n].mean(1) if n.any() else np.zeros(G.shape[0])
    return mc / (mn + 1e-8)


def row_order(Y: torch.Tensor, f: int) -> list[int]:
    mags = Y[f].abs()
    nz = torch.nonzero(mags > 1e-12).flatten()
    return nz[torch.sort(mags[nz], descending=True, stable=True).indices].tolist()


_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from experiments.varwidth.prep import get_tokenizer
        _TOK = get_tokenizer()
    return _TOK


def variant_counts(variant: str, meta: dict):
    """token -> Counter{(lang, word)} over the variant's concept + neutral sentences (as j3_words)."""
    from ember.judge_words import token_word_counts
    from experiments.varwidth.prep import NEUTRAL_SAMPLE_SEED
    from ember.local_datasets import ConceptDataset
    ds = ConceptDataset(concept_name=meta["concept"], concept_path=VARIANTS / f"{variant}.json",
                        neutral_sample_seed=NEUTRAL_SAMPLE_SEED)
    return token_word_counts(tok(), [s for s, _ in ds.data])


def intruder_ids(variant: str, concept: str) -> set:
    """Token ids in the inserted adjacent sentences that never occur in the clean S_C (part B only)."""
    if "__f" not in variant and "__u" not in variant:
        return set()
    v = json.loads((VARIANTS / f"{variant}.json").read_text())[0]
    clean = json.loads((VARIANTS / f"{slug(concept)}.json").read_text())[0]["sentences"]
    pos = set(v["replaced_positions"])
    inserted = [s for i, s in enumerate(v["sentences"]) if i in pos]
    ids = lambda ss: {t for seq in tok()(ss, add_special_tokens=False)["input_ids"] for t in seq}  # noqa: E731
    return ids(inserted) - ids(clean)


def word_of(counts: dict, t: int) -> str | None:
    c = counts.get(int(t))
    if not c:
        return None
    merged = Counter()
    for (_, w), k in c.items():
        merged[w] += k
    return merged.most_common(1)[0][0]


def features(only: list | None = None, out_path=None):
    from experiments.varwidth import overlap_judge as oj
    from experiments.varwidth.j3_eval import j3_count
    variants = json.loads((DATA / "variants.json").read_text())
    if only:
        variants = {v: c for v, c in variants.items() if v in only}
    rows = []
    for v, c in sorted(variants.items()):
        meta = torch.load(META / f"{v}.pt", weights_only=False) if (META / f"{v}.pt").exists() else None
        if meta is None:
            from experiments.paper_analysis.robustness.pipeline import variant_meta
            meta = variant_meta(v, c)
        codes, vstar = np.asarray(meta["codes"]), meta["vstar_ids"]
        counts = variant_counts(v, meta)
        intr = intruder_ids(v, c)
        lex = lexicon(c)
        tok_str = lambda t: tok().decode([t])  # noqa: E731
        for fp in sorted((FITS / v).glob("*/seed*.pt")):
            fit = torch.load(fp, weights_only=False)
            Y = load_Y(fit)
            rho = rho_stats(Y, codes)
            rank = np.empty(len(rho), int)
            rank[np.argsort(-rho, kind="stable")] = np.arange(1, len(rho) + 1)
            znorm = fit["Z"].norm(dim=0).numpy()
            for f in range(Y.shape[0]):
                order = row_order(Y, f)
                top = order[:TOPN]
                tids = [vstar[j] for j in top]
                fw = oj.feature_words([vstar[j] for j in order], counts, n=TOPN, tok_str=tok_str)
                rows.append({"variant": v, "concept": c, "cond": fit["cond"], "seed": int(fit["seed"]),
                             "group": group_of(v, c), "feature": f, "rho": float(rho[f]), "rank": int(rank[f]),
                             "width": len(order), "znorm": float(znorm[f]),
                             "lex_hits": j3_count(lex, fw), "fw20": fw,
                             "c_share": float(np.mean([codes[j] == 1 for j in top])) if top else 0.0,
                             "intr_share": float(np.mean([t in intr for t in tids])) if (top and intr) else 0.0,
                             "intr_any": int(sum(t in intr for t in tids)),
                             "top_tids": tids,
                             "top_disp": [disp(tok().convert_ids_to_tokens(t), word_of(counts, t)) for t in tids]})
        print(f"[features] {v}: {len(list((FITS / v).glob('*/seed*.pt')))} fits", flush=True)
    df = pd.DataFrame(rows)
    if only and out_path is None and (OUT / "features.parquet").exists():   # merge: replace these variants' rows
        old = pd.read_parquet(OUT / "features.parquet")
        df = pd.concat([old[~old.variant.isin(list(variants))], df], ignore_index=True)
    out_path = out_path or OUT / "features.parquet"
    df.to_parquet(out_path)
    print(f"-> {out_path} ({len(df)} rows)")
    return df


def disp(piece: str, word: str | None) -> str:
    p = piece.replace("▁", " ").strip() or piece
    if word is None:
        return f"{p}[?]"
    w = word.strip()
    return p if p.casefold() == w.casefold() else f"{p}[{w}]"


# ---------------------------------------------------------------- blind labelling (part A)
PART_A_CONDS = ("EMBER", "CS", "CSGS")


def export():
    """Top-10-by-rho features of every part-A run, condition / variant / seed hidden, shuffled within a concept,
    exact duplicates (same 20 displayed tokens) merged. Batches hold whole concepts."""
    df = pd.read_parquet(OUT / "features.parquet")
    A = df[(df.group.isin(["abstract", "entity"])) & df.cond.isin(PART_A_CONDS) & (df["rank"] <= TOP_RHO)].copy()
    A["disp"] = A.top_disp.apply(lambda x: ", ".join(x))
    rng = random.Random(1234)
    key, items_by_concept = {}, {}
    for c, g in A.groupby("concept"):
        uniq = sorted(set(g.disp))
        rng.shuffle(uniq)
        ids = {d: f"{slug(c)[:4].upper()}{i:03d}" for i, d in enumerate(uniq)}
        items_by_concept[c] = [(ids[d], d) for d in uniq]
        for r in g.itertuples():
            key[f"{r.variant}|{r.cond}|{r.seed}|{r.feature}"] = ids[r.disp]
    LABELS.mkdir(parents=True, exist_ok=True)
    (LABELS / "hidden_key.json").write_text(json.dumps(key))
    concepts = sorted(items_by_concept, key=lambda c: -len(items_by_concept[c]))
    n_batches = 8
    batches = [[] for _ in range(n_batches)]
    for c in concepts:                                   # greedy: largest concept to the lightest batch
        b = min(range(n_batches), key=lambda i: sum(len(items_by_concept[x]) for x in batches[i]))
        batches[b].append(c)
    for i, cs in enumerate(batches):
        lines = []
        for c in cs:
            lines.append(f"### CONCEPT: {c}")
            if c in PARENTS:
                lines.append(f"Parent (broader) topics: {'; '.join(PARENTS[c])}")
            for iid, d in items_by_concept[c]:
                lines.append(f"{iid} | {d}")
            lines.append("")
        (LABELS / f"batch_{i}.txt").write_text("\n".join(lines))
        print(f"batch_{i}: {cs} ({sum(len(items_by_concept[c]) for c in cs)} items)")
    # second-rater subset: 12% of every concept's items, same format, for inter-rater agreement
    rng2 = random.Random(99)
    lines = []
    for c in sorted(items_by_concept):
        its = items_by_concept[c]
        sub = rng2.sample(its, max(5, round(0.12 * len(its))))
        lines.append(f"### CONCEPT: {c}")
        if c in PARENTS:
            lines.append(f"Parent (broader) topics: {'; '.join(PARENTS[c])}")
        lines += [f"{iid} | {d}" for iid, d in sub] + [""]
    (LABELS / "agreement.txt").write_text("\n".join(lines))
    print(f"{sum(len(v) for v in items_by_concept.values())} unique items, {len(key)} feature slots")


def load_labels() -> dict:
    """{item id: (label, topic)} from labels/labels_batch_*.json and labels/labels_B_batch_*.json ({id: [P|B|N, topic]})."""
    out = {}
    for p in sorted(LABELS.glob("labels_batch_*.json")) + sorted(LABELS.glob("labels_B_batch_*.json")):
        for k, v in json.loads(p.read_text()).items():
            out[k] = (v[0], v[1] if len(v) > 1 else "") if isinstance(v, list) else (v, "")
    return out


def export_B():
    """Part B: top-10-by-rho features of every contamination run (adjacent f > 0 and unrelated control, all fitted
    conditions; f = 0 runs not labelled in part A), hidden and shuffled within a concept like export(). Displays already
    labelled in part A (same concept, same 20 tokens) reuse that label. IDs get a 'B' prefix."""
    df = pd.read_parquet(OUT / "features.parquet")
    keyA = json.loads((LABELS / "hidden_key.json").read_text())
    df["slot"] = df.variant + "|" + df.cond + "|" + df.seed.astype(str) + "|" + df.feature.astype(str)
    B = df[df.concept.isin(list(ADJACENT)) & ~df.variant.str.contains("__S") & (df["rank"] <= TOP_RHO)].copy()
    A = df[df.slot.isin(list(keyA))]
    B["disp"] = B.top_disp.apply(lambda x: ", ".join(x))
    A_disp = {(r.concept, ", ".join(r.top_disp)): keyA[r.slot] for r in A.itertuples()}
    rng = random.Random(4321)
    key, lines_by_c = {}, {}
    for c, g in B.groupby("concept"):
        new = sorted({d for d in g.disp if (c, d) not in A_disp})
        rng.shuffle(new)
        ids = {d: f"B{slug(c)[:4].upper()}{i:04d}" for i, d in enumerate(new)}
        lines_by_c[c] = [(ids[d], d) for d in new]
        for r in g.itertuples():
            key[r.slot] = A_disp.get((c, r.disp)) or ids[r.disp]
    (LABELS / "hidden_key_B.json").write_text(json.dumps(key))
    per = 420
    i = 0
    for c in sorted(lines_by_c):
        its = lines_by_c[c]
        for k in range(0, len(its), per):
            lines = [f"### CONCEPT: {c}"]
            if c in PARENTS:
                lines.append(f"Parent (broader) topics: {'; '.join(PARENTS[c])}")
            lines += [f"{iid} | {d}" for iid, d in its[k:k + per]]
            (LABELS / f"B_batch_{i}.txt").write_text("\n".join(lines) + "\n")
            print(f"B_batch_{i}: {c} ({len(its[k:k + per])} items)")
            i += 1
    print(f"{sum(len(v) for v in lines_by_c.values())} new items; {len(key)} slots")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("features", "export", "export_B"))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.what == "features":
        features(a.only)
    else:
        {"export": export, "export_B": export_B}[a.what]()


if __name__ == "__main__":
    main()
