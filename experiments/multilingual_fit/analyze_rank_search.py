#!/usr/bin/env python3
"""Compare the ranks of an EMBER (embedding-track) rank search (run_rank_search.sbatch).

Reads, for every rank<R>/seed<S> found under ROOT/<model>/:
    pickles/.../embedding/embedding.pkl                F (d_model x K), G (|V'| x K)
    csvs/.../embedding/stats_embed.csv                 per-feature ratio_abs
    interpretations/.../embedding/from_activation.csv  Gemini descriptions + verdicts
    interpretations/.../embedding/potential_features.csv  kept features
and reports one row per (rank, seed):
    rel_err      ||A - F G^T||_F / ||A||_F with A = W_E[V'].T (fit quality; falls as rank grows)
    candidates   features with ratio_abs >= --ratio-thresh (what the judge sees)
    member       candidates the judge calls members (any confidence)
    kept         is_member with confidence >= 0.85 = the features EMBER erases with
    edit_total   concept-only tokens EMBER would edit: in V', nonzero G on a kept feature
                 (same rule as ember/erasure/embed_edit.py)
    edit_<L>     those tokens by language: a token counts for every language whose concept
                 sentences contain it
kept = NaN means candidates exist but the judge has not run for that rank yet.

Writes ROOT/<model>/rank_search/<concept>/summary.csv (the rows above) and features.csv
(every candidate: ratio, verdict, description, top tokens, and share of its concept-token |G|
mass per language; a token in k languages gives 1/k to each).

Usage:
    .venv/bin/python experiments/multilingual_fit/analyze_rank_search.py \
        --root mf_outputs/rank_search/multi_language/en+ru+de+es+fr --concept "Harry Potter"
"""
import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/morg/NLP_2526b/semyong/.cache/huggingface")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ember.erasure.features import load_embedding_payload  # noqa: E402
from ember.local_datasets import ConceptDataset  # noqa: E402
from ember.utils import _safe_concept, _safe_model_name, build_token_label_codes  # noqa: E402
from experiments.crosslingual_concept_axis import load_embedding_matrix  # noqa: E402


def token_languages(concept, langs, data_dir, tok):
    """{token id: languages whose concept sentences contain it}, and the pipeline's
    concept / neutral / both code per token (build_token_label_codes on the full data set)."""
    special = set(tok.all_special_ids)
    origin = {}
    for L in langs:
        ds = ConceptDataset(concept, languages=[L], multilingual_data_dir=data_dir)
        for sent, lbl in ds.data:
            if lbl != concept:
                continue
            for t in tok(sent, add_special_tokens=False)["input_ids"]:
                if t not in special:
                    origin.setdefault(int(t), set()).add(L)
    full = ConceptDataset(concept, neutral_sample_seed=42, languages=langs, multilingual_data_dir=data_dir)
    return origin, build_token_label_codes(concept, full, tok)


def read_csv(path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else None


def is_true(s):
    return s.astype(str).str.strip().str.lower() == "true"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="OUTDIR/multi_language/<langs> of the rank search (OUTDIR for an English-only run)")
    ap.add_argument("--langs", default="en+ru+de+es+fr")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "experiments" / "multilingual_fit" / "data_v2")
    ap.add_argument("--concept", default="World War II")
    ap.add_argument("--model-name", default="google/gemma-2-2b-it")
    ap.add_argument("--ratio-thresh", type=float, default=2.0)
    ap.add_argument("--confidence-thresh", type=float, default=0.85)
    ap.add_argument("--out", type=Path, default=None, help="default: ROOT/<model>/rank_search/<concept>")
    args = ap.parse_args()

    langs = [x for x in args.langs.split("+") if x]
    cs = _safe_concept(args.concept)
    base = args.root / _safe_model_name(args.model_name)
    runs = sorted((int(m[1]), int(m[2]), p) for p in base.glob(f"pickles/rank*/seed*/{cs}/embedding/embedding.pkl")
                  if (m := re.search(r"rank(\d+)/seed(\d+)/", str(p))))
    if not runs:
        sys.exit(f"no embedding.pkl under {base}/pickles/rank*/seed*/{cs}/embedding/")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    E = load_embedding_matrix(args.model_name)
    origin, code = token_languages(args.concept, langs, args.data_dir, tok)

    summary, feats = [], []
    for rank, seed, pkl in runs:
        F, G, vprime = load_embedding_payload(pkl)
        # the fit ran on TransformerLens' W_E, which for Gemma is the raw embedding * sqrt(d_model)
        A = E.index_select(0, torch.tensor(vprime)).float().T * (E.shape[1] ** 0.5 if "gemma" in args.model_name.lower() else 1.0)
        rel_err = float(torch.linalg.norm(A - F.float() @ G.float().T) / torch.linalg.norm(A))
        G = G.float().numpy()
        vcode = np.array([code.get(t, 0) for t in vprime])
        vlangs = [origin.get(t, set()) for t in vprime]

        rel = Path(f"rank{rank}") / f"seed{seed}" / cs / "embedding"
        stats = pd.read_csv(base / "csvs" / rel / "stats_embed.csv")
        cand = stats[pd.to_numeric(stats["ratio_abs"], errors="coerce") >= args.ratio_thresh]
        act = read_csv(base / "interpretations" / rel / "from_activation.csv")
        judged = act is not None or cand.empty
        verdict = act.set_index("feature") if act is not None else pd.DataFrame()

        kept_ids = []
        for _, s in cand.iterrows():
            f = int(s["feature"])
            g = np.abs(G[:, f])
            row = {"rank": rank, "seed": seed, "feature": f, "ratio_abs": s["ratio_abs"],
                   "num_concept": s["num_concept"], "num_neutral": s["num_neutral"]}
            if f in verdict.index:
                v = verdict.loc[f]
                member = is_true(pd.Series([v["is_member"]])).iloc[0]
                conf = pd.to_numeric(v["confidence"], errors="coerce")
                row.update(is_member=member, confidence=conf, description=v["description"],
                           kept=bool(member and conf >= args.confidence_thresh))
                if row["kept"]:
                    kept_ids.append(f)
            # language mix of the feature's concept-only tokens
            m = (g > 0) & (vcode == 1)
            tot = max(float(g[m].sum()), 1e-12)
            for L in langs:
                row[f"share_{L}"] = sum(g[i] / len(vlangs[i]) for i in np.flatnonzero(m) if L in vlangs[i]) / tot
            top = np.argsort(-g)[:15]
            row["top_tokens"] = " | ".join(tok.decode([vprime[i]]) for i in top if g[i] > 0)
            feats.append(row)

        row = {"rank": rank, "seed": seed, "V'": len(vprime), "rel_err": rel_err,
               "candidates": len(cand), "member": np.nan, "kept": np.nan}
        if judged:
            row["member"] = int(is_true(act["is_member"]).sum()) if act is not None else 0
            row["kept"] = len(kept_ids)
        elig = (vcode == 1) & (np.abs(G[:, kept_ids]).max(1) > 0) if kept_ids else np.zeros(len(vprime), bool)
        row["edit_total"] = int(elig.sum())
        for L in langs:
            row[f"edit_{L}"] = int(sum(L in vlangs[i] for i in np.flatnonzero(elig)))
        summary.append(row)

    out = args.out or base / "rank_search" / cs
    out.mkdir(parents=True, exist_ok=True)
    sm, ft = pd.DataFrame(summary), pd.DataFrame(feats)
    sm.to_csv(out / "summary.csv", index=False)
    ft.to_csv(out / "features.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 90)
    print(f"\n=== {args.concept}, langs={args.langs}: one row per (rank, seed) ===")
    print(sm.round(4).to_string(index=False))
    if sm["seed"].nunique() > 1:
        print("\n=== mean over seeds ===")
        print(sm.drop(columns="seed").groupby("rank").mean().round(3).to_string())
    if "kept" in ft and ft["kept"].fillna(False).astype(bool).any():
        k = ft[ft["kept"].fillna(False).astype(bool)]
        cols = ["rank", "seed", "feature", "ratio_abs", "confidence"] + [f"share_{L}" for L in langs] + ["description"]
        print("\n=== kept features (share_<L> = share of the feature's concept-token mass in language L) ===")
        print(k[cols].round(2).to_string(index=False))
    print(f"\nsaved {out / 'summary.csv'} and {out / 'features.csv'}")


if __name__ == "__main__":
    main()
