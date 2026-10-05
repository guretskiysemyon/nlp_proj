#!/usr/bin/env python3
"""Part 2, idea 1: multilingual re-fit of EMBER's embedding-track factorization.

Re-runs the SAME factorization the English pipeline uses (ember.train_mf_features,
embedding track: SparseMatrixFactorization on E[V'].T, rank 100, g_sparsity 0.01,
fit_with_ridge, build_token_label_codes, compute_embedding_stats) but with a
bigger V' / label set. Variants:

    en      English concept + neutral sentences only      (control = today's pipeline, refit)
    en+ru   + Russian concept sentences + Russian neutral sentences
    en+all  + ru, de, es, fr concept + neutral sentences
    +xn     (suffix) foreign neutral set also includes the other concepts' foreign articles

Foreign sentences come from experiments/multilingual_fit/fetch_wiki_sentences.py.
No generation, no judge, no model forward pass: only the embedding matrix.

Concept features are chosen WITHOUT the Gemini membership filter the stored English
features went through (that filter can't be used here): top-k features by ratio_abs among
features with >= MIN_CONCEPT_TOKENS non-zero concept tokens, identically for every variant.
The stored (Gemini-filtered) English subspace is reported as a reference row ("stored_en").

Evaluation reuses the cross-lingual PoC's measures (experiments/crosslingual_concept_axis.py):
    A       AUC of concept words (mean-pooled pieces) vs ordinary / other-concept words of the
            same language, scored by the fraction of the word's embedding inside the subspace
    qfrac   share of the foreign-built concept direction a_L (mean concept words - mean ordinary
            words of L) that lies inside the subspace (chance ~ k/2304)
    mix     for each selected feature, where its token mass sits: English-only tokens,
            foreign-only tokens, or shared tokens -> shared feature vs language-specific feature

Usage:
    .venv/bin/python experiments/multilingual_fit/multilingual_fit.py --stage fit --seeds 42
    .venv/bin/python experiments/multilingual_fit/multilingual_fit.py --stage eval
"""
import argparse
import contextlib
import difflib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ember.erasure import features  # noqa: E402
from ember.utils import (SparseMatrixFactorization, build_token_label_codes,  # noqa: E402
                         compute_embedding_stats, fit_with_ridge, set_seed)
from experiments.crosslingual_concept_axis import (auc_parts, energy, load_embedding_matrix,  # noqa: E402
                                                   qa_texts_by_concept, unit, words_from_text)

HERE = Path(__file__).parent
DATA, FITS, OUT = HERE / "data", HERE / "fits", HERE / "results"
WORDS = {k: v for k, v in json.load(open(ROOT / "experiments" / "crosslingual_words.json")).items()
         if not k.startswith("_")}
VARIANTS = {"en": [], "en+ru": ["ru"], "en+all": ["ru", "de", "es", "fr"],
            # "+xn": the foreign neutral set ALSO contains the other fetched concepts' foreign articles
            # (bigger, same-genre neutral text; diagnostic for generic inflected words looking concept-only)
            "en+ru+xn": ["ru"], "en+all+xn": ["ru", "de", "es", "fr"]}
EVAL_LANGS = ["en", "ru", "de", "es", "fr"]
MIN_CONCEPT_TOKENS = 10
G_SPARSITY = 0.01  # same default as train_mf_features.py


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


# --------------------------------------------------------------------------- #
# Fit                                                                          #
# --------------------------------------------------------------------------- #

def load_sets(concept: str, langs, xn: bool = False):
    en_c = next(e for e in json.load(open(ROOT / "data" / "concept_sentences.json"))
                if e["concept"] == concept)["sentences"]
    en_n = [o["sentence"] for o in json.load(open(ROOT / "data" / "neutral_sentences.json"))]
    sets = {"en": {"concept": [s for s in en_c if s], "neutral": en_n}}
    for L in langs:
        ce = {e["concept"]: e["sentences"] for e in json.load(open(DATA / f"concept_sentences_{L}.json"))}
        nl = [o["sentence"] for o in json.load(open(DATA / f"neutral_sentences_{L}.json"))]
        if xn:
            nl = nl + [x for c2, sents in ce.items() if c2 != concept for x in sents]
        sets[L] = {"concept": ce.get(concept, []), "neutral": nl}
    return sets


def fit_one(concept, variant, seed, E, tok, device, rank, max_iter):
    path = FITS / f"{safe(concept)}__{variant}__seed{seed}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    sets = load_sets(concept, VARIANTS[variant], xn=variant.endswith("+xn"))
    pairs, origin = [], {}
    special = set(tok.all_special_ids)
    for L, s in sets.items():
        pairs += [(x, concept) for x in s["concept"]] + [(x, "Neutral") for x in s["neutral"]]
        for sent in s["concept"] + s["neutral"]:
            for t in tok(sent, add_special_tokens=False)["input_ids"]:
                if t not in special:
                    origin.setdefault(int(t), set()).add(L)
    token_to_code = build_token_label_codes(concept, pairs, tok)
    vprime = sorted(origin)
    A = E.index_select(0, torch.tensor(vprime)).float().T.contiguous().to(device)

    set_seed(seed)
    nmf = SparseMatrixFactorization(rank, fitting_device=device, g_sparsity=G_SPARSITY)
    buf, t0 = io.StringIO(), time.time()
    with contextlib.redirect_stdout(buf):  # fit() prints every 50 iterations
        fit_with_ridge(nmf, A, max_iter)
    secs = time.time() - t0
    stop = re.findall(r"Stopping early at iter (\d+)", buf.getvalue())
    G, F = nmf.G_.detach().cpu(), nmf.F_.detach().cpu()
    stats = compute_embedding_stats(G, vprime, token_to_code, concept)

    # language group of every token in V': 0 = English only, 1 = foreign only, 2 = both
    langs_of = [origin[t] for t in vprime]
    grp = np.array([0 if s == {"en"} else 1 if "en" not in s else 2 for s in langs_of], dtype=np.int8)
    which_foreign = [(sorted(s - {"en"}) or [""])[0] if "en" not in s else "" for s in langs_of]
    fit = {"F": F.numpy(), "G": G.numpy(), "vprime": vprime, "code": np.array([token_to_code.get(t, 0) for t in vprime]),
           "grp": grp, "foreign_lang": which_foreign, "stats": stats.to_dict("list"),
           "n_sentences": {L: {k: len(v) for k, v in s.items()} for L, s in sets.items()},
           "fit_seconds": secs, "stopped_at_iter": int(stop[0]) if stop else None, "variant": variant,
           "concept": concept, "seed": seed}
    FITS.mkdir(exist_ok=True)
    torch.save(fit, path)
    print(f"[fit] {concept:13s} {variant:7s} seed={seed} |V'|={len(vprime):6d} "
          f"time={secs:5.1f}s stop_iter={fit['stopped_at_iter']}", flush=True)
    return fit


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #

class Words:
    """Word vectors + per-language backgrounds, built once (same definitions as the PoC, mean pooling)."""

    def __init__(self, E, tok, concepts):
        self.E, self.tok, self.cache, self.pieces = E, tok, {}, {}
        self.W, self.ids = {}, {}
        for c in concepts:
            for L in EVAL_LANGS:
                en = WORDS[c]["en"]
                seen, rows = set(), []
                for i, w in enumerate(WORDS[c][L]):
                    if w is None or w in seen or self.vec(w) is None:
                        continue
                    seen.add(w)
                    r = difflib.SequenceMatcher(None, w.lower(), en[i].lower()).ratio()
                    kind = "same" if r >= 0.99 else "similar" if r >= 0.7 else "distinct"
                    if L != "en" and kind != "distinct":  # = the PoC's --only-distinct
                        continue
                    rows.append(w)
                self.W[(c, L)] = rows
        per = {L: {k: words_from_text(v) for k, v in qa_texts_by_concept(L).items()} for L in EVAL_LANGS}
        self.bg_all = {L: sorted(w for w in set().union(*per[L].values()) if self.vec(w) is not None)
                       for L in EVAL_LANGS}
        self.mu = {L: self.mat(self.bg_all[L]).mean(0) for L in EVAL_LANGS}
        self.bg_nat, self.bg_oth = {}, {}
        for L in EVAL_LANGS:
            for c in concepts:
                others = set().union(*[v for k, v in per[L].items() if k != c])
                self.bg_nat[(L, c)] = sorted(w for w in others - set(self.W[(c, L)]) if self.vec(w) is not None)
                self.bg_oth[(L, c)] = sorted({w for c2 in concepts if c2 != c for w in self.W[(c2, L)]}
                                             - set(self.W[(c, L)]))

    def piece_ids(self, w):
        if w not in self.pieces:
            self.pieces[w] = self.tok.encode(" " + w, add_special_tokens=False)
        return self.pieces[w]

    def vec(self, w):
        if w not in self.cache:
            ids = self.piece_ids(w)
            self.cache[w] = self.E[torch.as_tensor(ids)].float().mean(0) if ids else None
        return self.cache[w]

    def mat(self, ws):
        return torch.stack([self.vec(w) for w in ws if self.vec(w) is not None])


def select_features(fit, k):
    st = pd.DataFrame(fit["stats"])
    st = st[st["num_concept"] >= MIN_CONCEPT_TOKENS].sort_values("ratio_abs", ascending=False)
    return st["feature"].head(k).astype(int).tolist()


def mix_rows(fit, feats):
    G = np.abs(fit["G"])
    rows = []
    for f in feats:
        g, code, grp = G[:, f], fit["code"], fit["grp"]
        nz = g > 1e-12
        def mass(mask):
            return float(g[nz & mask].sum())
        m = {n: mass(grp == i) for n, i in (("en", 0), ("foreign", 1), ("shared", 2))}
        tot = max(sum(m.values()), 1e-12)
        mc = {n: mass((grp == i) & (code == 1)) for n, i in (("en", 0), ("foreign", 1))}
        totc = max(mc["en"] + mc["foreign"], 1e-12)
        rows.append({"feature": f, "ratio_abs": float(pd.DataFrame(fit["stats"]).set_index("feature").loc[f, "ratio_abs"]),
                     "mass_en": m["en"] / tot, "mass_foreign": m["foreign"] / tot, "mass_shared": m["shared"] / tot,
                     "concept_mass_en": mc["en"] / totc, "concept_mass_foreign": mc["foreign"] / totc,
                     "balance": min(mc["en"], mc["foreign"]) / totc})  # 0 = one language only, 0.5 = even split
    return rows


def coverage_rows(fit, feats, concept, wd, E, label):
    """What the actual EMBER edit would touch (ember/erasure/embed_edit.py): a token is edited iff it is
    concept-only, in V', and has non-zero G on a selected feature. Per language: share of words with >= 1
    piece edited ("any") / every piece edited ("all"), for the concept's words vs other concepts' words
    (specificity) vs ordinary words (collateral). Also the edit size ||c_i|| / ||e_i|| (delta = 1)."""
    G, F = fit["G"][:, feats], fit["F"][:, feats]
    elig = (fit["code"] == 1) & (np.abs(G).max(1) > 0)
    T = {t for t, e in zip(fit["vprime"], elig) if e}
    rel = np.linalg.norm(G @ F.T, axis=1) / np.linalg.norm(E[torch.as_tensor(fit["vprime"])].float().numpy(), axis=1)
    out = {"n_edited": int(elig.sum())}
    for g, name in ((0, "en"), (1, "foreign")):
        m = elig & (fit["grp"] == g)
        out[f"n_edited_{name}"] = int(m.sum())
        out[f"rel_edit_{name}"] = float(rel[m].mean()) if m.any() else np.nan
    rows = []
    for L in EVAL_LANGS:
        row = {"concept": concept, "lang": L, **label, **out}
        for tag, ws in (("concept", wd.W[(concept, L)]), ("oth", wd.bg_oth[(L, concept)]), ("ord", wd.bg_nat[(L, concept)])):
            ids = [wd.piece_ids(w) for w in ws]
            row[f"any_{tag}"] = float(np.mean([any(t in T for t in p) for p in ids])) if ids else np.nan
            row[f"all_{tag}"] = float(np.mean([all(t in T for t in p) for p in ids])) if ids else np.nan
        rows.append(row)
    return rows


def eval_subspace(Q, concept, vset, wd, label):
    """Measure A and qfrac for one subspace, every language. Returns list of row dicts."""
    rows = []
    for L in EVAL_LANGS:
        ws = wd.W[(concept, L)]
        if len(ws) < 5:
            continue
        X = wd.mat(ws)
        row = {"concept": concept, "lang": L, "n": len(ws), "k": Q.shape[1], **label}
        for tag, bg in (("nat", wd.bg_nat[(L, concept)]), ("oth", wd.bg_oth[(L, concept)])):
            row[f"A_{tag}"] = float(auc_parts(energy(X, Q), energy(wd.mat(bg), Q)).mean())
            if vset is not None:  # seen / unseen = every piece of the word is in V' (the fit's vocabulary)
                unseen = np.array([not set(wd.piece_ids(w)) <= vset for w in ws])
                row["n_unseen"] = int(unseen.sum())
                if unseen.sum() >= 5:
                    parts = auc_parts(energy(X, Q), energy(wd.mat(bg), Q))
                    row[f"A_{tag}_unseen"] = float(parts[unseen].mean())
        a_L = unit(X.mean(0) - wd.mu[L])
        row["qfrac"] = float((Q.T @ a_L).pow(2).sum())
        rows.append(row)
    return rows


def evaluate(args, E, tok):
    concepts = args.concepts
    wd = Words(E, tok, concepts)
    rows, mixes, covs = [], [], []
    for c in concepts:
        # reference: the stored, Gemini-filtered English subspace (what EMBER actually erases with)
        pkl, pot = features._embedding_paths(args.model_name, c, args.rank, 42)
        F_dir, _, _ = features.load_embedding_payload(pkl)
        feat = features.select_embed_feature_ids(pd.read_csv(pot), ratio_thresh=2.0)
        Q, _ = torch.linalg.qr(F_dir[:, feat].float())
        rows += eval_subspace(Q, c, None, wd, {"variant": "stored_en", "seed": 42, "topk": f"gem{len(feat)}"})
        for variant in VARIANTS:
            for seed in args.seeds:
                path = FITS / f"{safe(c)}__{variant}__seed{seed}.pt"
                if not path.exists():
                    print(f"[missing] {path.name}"); continue
                fit = torch.load(path, weights_only=False)
                vset = set(fit["vprime"])
                for k in args.topk:
                    feats = select_features(fit, k)
                    Q, _ = torch.linalg.qr(torch.as_tensor(fit["F"][:, feats]).float())
                    rows += eval_subspace(Q, c, vset, wd, {"variant": variant, "seed": seed, "topk": k})
                    covs += coverage_rows(fit, feats, c, wd, E, {"variant": variant, "seed": seed, "topk": k})
                    if k == args.headline_k and variant != "en":
                        mixes += [{"concept": c, "variant": variant, "seed": seed, **r} for r in mix_rows(fit, feats)]
    OUT.mkdir(exist_ok=True)
    df, mx, cv = pd.DataFrame(rows), pd.DataFrame(mixes), pd.DataFrame(covs)
    df.to_csv(OUT / "measures.csv", index=False)
    mx.to_csv(OUT / "feature_mix.csv", index=False)
    cv.to_csv(OUT / "coverage.csv", index=False)
    return df, mx, cv


def report_coverage(cv, headline_k):
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    c = cv[cv["topk"] == headline_k]
    print(f"\n=== EDIT COVERAGE (what the actual EMBER edit touches), top-{headline_k} features, mean over concepts x seeds ===")
    print("    any_* = share of words with >=1 piece edited; all_* = every piece edited;")
    print("    concept = the concept's own words, oth = other concepts' words (collateral), ord = ordinary words (collateral)")
    for L in ("ru", "de", "es", "fr", "en"):
        s = c[c["lang"] == L].groupby("variant")[["any_concept", "all_concept", "any_oth", "any_ord"]].mean()
        s = s.loc[[v for v in VARIANTS if v in s.index]]
        if L != "en" and set(s.index) == {"en"}:
            continue
        print(f"\n  language = {L}")
        print(s.round(3).to_string())
    print("\n--- per concept, Russian: any_concept / any_oth (en -> en+ru -> en+all) ---")
    r = c[c["lang"] == "ru"].groupby(["concept", "variant"])[["any_concept", "any_oth"]].mean().unstack("variant").round(2)
    print(r.to_string())
    e = c.drop_duplicates(["concept", "variant", "seed"]).groupby("variant")[
        ["n_edited", "n_edited_en", "n_edited_foreign", "rel_edit_en", "rel_edit_foreign"]].mean()
    print("\n--- tokens edited per concept (mean) and edit size ||c_i||/||e_i|| at delta=1 ---")
    print(e.loc[[v for v in VARIANTS if v in e.index]].round(3).to_string())


def report(df, mx, headline_k):
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    h = df[(df["topk"].astype(str) == str(headline_k)) | (df["variant"] == "stored_en")].copy()
    h["variant"] = h["variant"].astype(str)
    order = ["stored_en", "en", "en+ru", "en+all", "en+ru+xn", "en+all+xn"]
    for lang, metric in (("ru", "A_oth"), ("ru", "A_nat"), ("en", "A_oth"), ("ru", "qfrac"), ("en", "qfrac")):
        sub = h[h["lang"] == lang]
        p = sub.groupby(["concept", "variant"])[metric].mean().unstack("variant")[[v for v in order if v in set(sub["variant"])]]
        p.loc["MEAN"] = p.mean()
        print(f"\n=== {metric} for {lang.upper()} concept words, top-{headline_k} features (mean over seeds) ===")
        print(p.round(3).to_string())
    for lang in ("de", "es", "fr"):
        sub = h[(h["lang"] == lang) & h["variant"].isin(["en", "en+all", "en+all+xn"])]
        if len(sub):
            p = sub.groupby(["concept", "variant"])["A_oth"].mean().unstack("variant")
            p.loc["MEAN"] = p.mean()
            print(f"\n=== A_oth for {lang.upper()} (distinct words only), en vs en+all ===")
            print(p.round(3).to_string())
    un = h[(h["lang"] == "ru") & h["variant"].isin(list(VARIANTS)) & h["A_oth_unseen"].notna()]
    if len(un):
        print("\n=== RU: held-out words (>=1 piece not in the fit's V'): A_oth_unseen / n_unseen ===")
        print(un.groupby(["concept", "variant"]).agg(A_oth_unseen=("A_oth_unseen", "mean"), n_unseen=("n_unseen", "mean")).round(3).to_string())
    if len(mx):
        print(f"\n=== Feature mix of the top-{headline_k} features (share of concept-token mass; balance 0 = one language, 0.5 = even) ===")
        g = mx.groupby(["variant", "concept"]).agg(en=("concept_mass_en", "mean"), foreign=("concept_mass_foreign", "mean"),
                                                    balance=("balance", "mean"),
                                                    n_shared_feats=("balance", lambda s: int((s >= 0.2).sum())),
                                                    n_feats=("balance", "size"))
        print(g.round(3).to_string())
        print("\nfeatures with balance >= 0.2 (shared across languages):",
              {v: f"{int((d['balance'] >= 0.2).sum())}/{len(d)}" for v, d in mx.groupby('variant')})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["fit", "eval", "all"], default="all")
    ap.add_argument("--model-name", default="google/gemma-2-2b-it")
    ap.add_argument("--concepts", nargs="*", default=["Gun", "Heroin", "Gambling", "Ancient Rome", "World War II"])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--rank", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=20_000)
    ap.add_argument("--topk", type=int, nargs="*", default=[1, 3, 5])
    ap.add_argument("--headline-k", type=int, default=3)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    E = load_embedding_matrix(args.model_name)
    if args.stage in ("fit", "all"):
        for seed in args.seeds:
            for c in args.concepts:
                for v in args.variants:
                    fit_one(c, v, seed, E, tok, args.device, args.rank, args.max_iter)
    if args.stage in ("eval", "all"):
        df, mx, cv = evaluate(args, E, tok)
        report(df, mx, args.headline_k)
        report_coverage(cv, args.headline_k)
        print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
