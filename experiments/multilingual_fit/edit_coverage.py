#!/usr/bin/env python3
"""How much of each language does an EMBER edit reach? (CPU only)

The edited tokens of a feature set are reproduced exactly as ember/erasure/embed_edit.py
selects them: tokens of V' labelled concept-only (in concept sentences, never in neutral
ones) with a non-zero G coefficient on a kept feature (potential_features.csv, ratio >= 2).

Two coverage measures per evaluation language L:
  q_cov  share of the 50 QA_test questions in L (data/qa_lang, the questions of the
         Part 1 leak test) whose question or correct answer contains an edited token;
  v_cov  share of the occurrences of L's own concept-only tokens in L's concept sentences
         (English: data/, others: data_v2/) that are edited tokens.

Part A: the Part 1 edit (the paper's English features, rank 100) for all 18 concepts,
joined with the leak measured in Part 1 (results/.../ember_fixed/summary.csv); also checks
that the edited-token counts equal Part 1's n_tokens_edited.
Part B: the rank sweeps in mf_outputs/rank_search: every single-language fit, arm B (union of
the five single-language edits at the same rank) and arm A (joint fit).

Output: <out>/part1_edit.csv, <out>/rank_sweep.csv (default mf_outputs/rank_search/coverage).

Usage:
    .venv/bin/python -I experiments/multilingual_fit/edit_coverage.py
"""
import argparse
import ast
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ember.erasure.features import load_embedding_payload  # noqa: E402
from ember.local_datasets import ConceptDataset  # noqa: E402
from ember.utils import build_token_label_codes, _safe_concept  # noqa: E402
from crosslingual.data import load_mc_items  # noqa: E402

LANGS = ["en", "ru", "de", "es", "fr"]
CONCEPTS = ["Ancient Rome", "Artificial intelligence", "Baseball", "COVID-19 pandemic", "Cannabis",
            "Culture of Greece", "Gambling", "Golf", "Gun", "Halloween", "Harry Potter", "Heroin",
            "Nazism", "Pornography", "Republic of Ireland", "Uranium", "Valentine's Day", "World War II"]
RANKS = [100, 200, 300, 400, 600, 800]
MODEL = "google_gemma-2-2b-it"
TOK = None


def tok_ids(text):
    return TOK(" " + text.strip(), add_special_tokens=False)["input_ids"]


@lru_cache(maxsize=None)
def labels(concept, langs, data_dir):
    ds = ConceptDataset(concept_name=concept, neutral_sample_seed=42, languages=list(langs),
                        multilingual_data_dir=data_dir)
    return build_token_label_codes(concept, ds, TOK), ds


def edited(base, concept, rank, langs, data_dir):
    """Edited token ids of one fit, or None if the fit does not exist."""
    rel = Path(MODEL)
    sub = Path(f"rank{rank}") / "seed42" / _safe_concept(concept) / "embedding"
    pkl = base / rel / "pickles" / sub / "embedding.pkl"
    pot = base / rel / "interpretations" / sub / "potential_features.csv"
    if not pkl.exists() or not pot.exists():
        return None
    df = pd.read_csv(pot) if pot.stat().st_size > 1 else pd.DataFrame()
    if df.empty:
        return set()
    ids = sorted({int(f) for f, m in zip(df["feature"], pd.to_numeric(df["metric_score"], errors="coerce"))
                  if m >= 2.0})
    if not ids:
        return set()
    _, G, vprime = load_embedding_payload(pkl)
    g = np.abs(G.float().numpy()[:, ids]).max(1)
    code, _ = labels(concept, tuple(langs), str(data_dir))
    return {t for t, v in zip(vprime, g) if v > 0 and code.get(t) == 1}


@lru_cache(maxsize=None)
def questions(concept, lang):
    return [(set(tok_ids(it["question"])), set(tok_ids(it["correct_text"])))
            for it in load_mc_items(concept, lang, "QA_test")]


@lru_cache(maxsize=None)
def own_concept_occurrences(concept, lang, data_dir):
    code, ds = labels(concept, (lang,), data_dir)
    occ = []
    for s, lbl in ds:
        if lbl == concept:
            occ += [t for t in TOK(s, add_special_tokens=False)["input_ids"] if code.get(t) == 1]
    return np.array(occ)


def coverage(E, concept, data_dir):
    out = {}
    for L in LANGS:
        qs = questions(concept, L)
        out[f"q_cov_{L}"] = float(np.mean([bool((q | a) & E) for q, a in qs])) if qs else np.nan
        occ = own_concept_occurrences(concept, L, data_dir)
        out[f"v_cov_{L}"] = float(np.isin(occ, list(E)).mean()) if len(occ) and E else 0.0
    return out


def main():
    global TOK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "experiments/multilingual_fit/data_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "mf_outputs/rank_search/coverage")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    args.out.mkdir(parents=True, exist_ok=True)
    dd = str(args.data_dir)

    # ---- Part A: the Part 1 edit, all 18 concepts
    summ = pd.read_csv(ROOT / "results/crosslingual/google_gemma-2-2b-it/ember_fixed/summary.csv")
    rows = []
    for c in CONCEPTS:
        E = edited(ROOT / "mf_outputs", c, 100, ["en"], dd)
        ej = json.loads((ROOT / "results/crosslingual/google_gemma-2-2b-it/ember_fixed" / _safe_concept(c)
                         / "edit.json").read_text())
        info = ej["edit_info"] if isinstance(ej["edit_info"], dict) else ast.literal_eval(ej["edit_info"])
        r = {"concept": c, "n_edited": len(E), "part1_n_tokens_edited": info["n_tokens_edited"],
             **coverage(E, c, dd)}
        for L in LANGS:
            s = summ[(summ.concept == c) & (summ.language == L)]
            r[f"leak_{L}"] = float(s.leak.iloc[0]) if len(s) and bool(s.reliable.iloc[0]) else np.nan
            r[f"retention_{L}"] = float(s.retention.iloc[0]) if len(s) else np.nan
        rows.append(r)
        print(f"[part1] {c}: {len(E)} tokens (Part 1: {info['n_tokens_edited']})", flush=True)
    a = pd.DataFrame(rows)
    a.to_csv(args.out / "part1_edit.csv", index=False)

    # ---- Part B: rank sweeps
    rs = ROOT / "mf_outputs/rank_search"
    fits = {L: (rs if L == "en" else rs / "multi_language" / L, [L]) for L in LANGS}
    fits["joint"] = (rs / "multi_language" / "+".join(LANGS), LANGS)
    rows = []
    for c in ["World War II", "Cannabis", "Gun"]:
        for rank in RANKS:
            sets = {name: edited(base, c, rank, langs, dd) for name, (base, langs) in fits.items()}
            avail = {k: v for k, v in sets.items() if v is not None}
            if all(L in avail for L in LANGS):
                avail["armB_union"] = set().union(*(avail[L] for L in LANGS))
            for name, E in avail.items():
                rows.append({"concept": c, "rank": rank, "edit": name, "n_edited": len(E), **coverage(E, c, dd)})
            print(f"[sweep] {c} rank {rank}: {sorted(avail)}", flush=True)
    pd.DataFrame(rows).to_csv(args.out / "rank_sweep.csv", index=False)
    print(f"saved {args.out}/part1_edit.csv and rank_sweep.csv")


if __name__ == "__main__":
    main()
