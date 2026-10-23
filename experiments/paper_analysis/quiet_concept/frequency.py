"""The frequency hypothesis ("concept tokens were seen less in training, so they carry little variance").

    python -m experiments.paper_analysis.quiet_concept.frequency counts      # Wikipedia token counts (cached)
    python -m experiments.paper_analysis.quiet_concept.frequency analyse

counts: articles of the cached wikimedia/wikipedia 20231101.en shard 40 (the only shard in the HF cache), in a seeded
    random order, tokenized with Gemma's tokenizer (add_special_tokens=False) until N_TOKENS tokens -> counts per id.
    Frequency proxy f_t = (count_t + 0.5) / total tokens, reported as log10 per million.
analyse: per concept, for concept-only (C) and neutral-only (N) tokens of V*:
    lf     log10 frequency per million
    nr     ||e_t||^2 (raw, x sqrt(d) embeddings)              nc   ||e_t - mu_B||^2
    mah    ||W (e_t - mu_B)||^2 (whitened energy)             cs   ||x_t||^2 (CS energy)
    top10  sum_{j<=10} (q_j^T (e_t - mu_B))^2 (energy along the 10 top background principal components), and its
           fraction of nc (frac10); same for 50 (top50 / frac50)
    Per concept: median of each over C and over N; paired Wilcoxon over concepts; within-concept Spearman of lf with
    each metric over the C u N tokens; pooled token-level Spearman over the union of V* tokens (each id once).
    Whole vocabulary: Spearman of lf vs nr / nc / mah / frac10 over all non-special ids with a nonzero count.
Outputs: frequency/wiki_counts.npz, frequency/per_concept.csv, frequency/tokens.csv, frequency/summary.json.
"""
from __future__ import annotations

import argparse
import glob
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats

from experiments.paper_analysis.quiet_concept.common import OUT, concepts, dump, matrices, whitener

N_TOKENS = 6_000_000
SEED = 0
SHARD = ("/home/morg/NLP_2526b/oripetel/.cache/huggingface/hub/datasets--wikimedia--wikipedia/snapshots/*/"
         "20231101.en/train-00040-of-00041.parquet")
FDIR = OUT / "frequency"
METRICS = ["nr", "nc", "mah", "cs", "top10", "frac10", "top50", "frac50", "mu_proj", "mu_share"]


def build_counts():
    import pyarrow.parquet as pq
    from experiments.varwidth.prep import get_tokenizer
    tok = get_tokenizer()
    pf = pq.ParquetFile(glob.glob(SHARD)[0])
    texts = pf.read(columns=["text"]).column("text").to_pylist()
    order = list(range(len(texts)))
    random.Random(SEED).shuffle(order)
    counts = np.zeros(len(tok), dtype=np.int64)
    total, n_art = 0, 0
    B = 256
    for i in range(0, len(order), B):
        batch = [texts[j] for j in order[i:i + B]]
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            np.add.at(counts, np.asarray(ids, dtype=np.int64), 1)
            total += len(ids)
            n_art += 1
        if total >= N_TOKENS:
            break
    FDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FDIR / "wiki_counts.npz", counts=counts, total=total, n_articles=n_art)
    print(f"{total} tokens from {n_art} articles; {int((counts > 0).sum())} distinct ids")


def load_counts():
    z = np.load(FDIR / "wiki_counts.npz")
    return z["counts"], int(z["total"])


def logf(counts, total, ids):
    return np.log10((counts[ids] + 0.5) / total * 1e6)


def token_metrics(E: torch.Tensor, X: torch.Tensor | None = None) -> dict:
    """E: (n, d) raw rows (float64). Returns per-token metric arrays."""
    wh = whitener()
    C = E - wh["mu"]
    U = C @ wh["W"]
    Q = wh["Q"].flip(1)                                    # background PCs, largest first
    P10, P50 = C @ Q[:, :10], C @ Q[:, :50]
    nc = C.pow(2).sum(1)
    out = {"nr": E.pow(2).sum(1), "nc": nc, "mah": U.pow(2).sum(1), "top10": P10.pow(2).sum(1),
           "top50": P50.pow(2).sum(1)}
    out["frac10"], out["frac50"] = out["top10"] / nc, out["top50"] / nc
    mu_hat = wh["mu"] / wh["mu"].norm()
    out["mu_proj"] = E @ mu_hat                             # signed component along mu_B / ||mu_B||
    out["mu_share"] = out["mu_proj"].pow(2) / out["nr"]     # share of ||e||^2 along mu^_B
    if X is not None:
        out["cs"] = X.pow(2).sum(1)
    return {k: v.numpy() for k, v in out.items()}


def analyse():
    counts, total = load_counts()
    per, toks = [], []
    for c in concepts("all"):
        M = matrices(c)
        ids = np.asarray(M["vstar_ids"])
        codes = M["codes"]
        tm = token_metrics(M["raw"].T, M["cs"].T)
        lf = logf(counts, total, ids)
        df = pd.DataFrame({"concept": c, "id": ids, "code": codes, "lf": lf, "zero": counts[ids] == 0, **tm})
        toks.append(df)
        sub = df[df.code.isin([0, 1])]
        row = {"concept": c, "n_C": int((codes == 1).sum()), "n_N": int((codes == 0).sum()),
               "zero_C": float(df[df.code == 1].zero.mean()), "zero_N": float(df[df.code == 0].zero.mean())}
        for k in ["lf"] + METRICS:
            row[f"{k}_C"] = float(df[df.code == 1][k].median())
            row[f"{k}_N"] = float(df[df.code == 0][k].median())
            if k != "lf":
                row[f"rho_lf_{k}"] = float(stats.spearmanr(sub.lf, sub[k]).statistic)
        # does rarity explain the C-N difference? logistic-free check: C-N gap in each metric within lf bins
        bins = pd.qcut(sub.lf, 5, duplicates="drop")
        for k in ("nr", "nc", "mah", "frac10", "mu_share"):
            g = sub.groupby([bins, "code"], observed=True)[k].median().unstack()
            row[f"{k}_gap_within_lf_bins"] = float(np.nanmedian((g[1] - g[0]) / g[0]))
        per.append(row)
        print(f"{c:28s} lf C {row['lf_C']:.2f} N {row['lf_N']:.2f} | nr C {row['nr_C']:.0f} N {row['nr_N']:.0f} | "
              f"mah C {row['mah_C']:.0f} N {row['mah_N']:.0f} | frac10 C {row['frac10_C']:.3f} N {row['frac10_N']:.3f}",
              flush=True)
    P = pd.DataFrame(per)
    T = pd.concat(toks)
    FDIR.mkdir(parents=True, exist_ok=True)
    P.to_csv(FDIR / "per_concept.csv", index=False)
    T.to_csv(FDIR / "tokens.csv", index=False)
    summ = {"n_concepts": len(P), "wiki_tokens": total}
    for k in ["lf"] + METRICS:
        d = P[f"{k}_C"] - P[f"{k}_N"]
        w = stats.wilcoxon(P[f"{k}_C"], P[f"{k}_N"])
        summ[k] = {"median_C": float(P[f"{k}_C"].median()), "median_N": float(P[f"{k}_N"].median()),
                   "n_concepts_C_gt_N": int((d > 0).sum()), "wilcoxon_p": float(w.pvalue),
                   "median_ratio_C_over_N": float((P[f"{k}_C"] / P[f"{k}_N"]).median()) if k != "lf" else None}
        if k != "lf":
            summ[k]["within_concept_spearman_lf_median"] = float(P[f"rho_lf_{k}"].median())
    for k in ("nr", "nc", "mah", "frac10", "mu_share"):
        summ[k]["C_minus_N_rel_gap_within_lf_quintiles_median"] = float(P[f"{k}_gap_within_lf_bins"].median())
    # pooled over unique ids (neutral tokens recur across concepts: keep each id once, C if it is C anywhere)
    U = T.sort_values("code", ascending=False).drop_duplicates("id")
    U = U[U.code.isin([0, 1])]
    summ["pooled"] = {k: float(stats.spearmanr(U.lf, U[k]).statistic)
                      for k in ("nr", "nc", "mah", "frac10", "top10", "mu_proj", "mu_share")}
    summ["pooled"]["n_ids"] = int(len(U))
    # whole vocabulary
    from experiments.varwidth.prep import get_tokenizer, load_input_embeddings
    tok = get_tokenizer()
    E = load_input_embeddings()
    special = set(tok.all_special_ids)
    ids = np.array([i for i in range(E.shape[0]) if counts[i] > 0 and i not in special])
    parts = [token_metrics(E[torch.as_tensor(ids[i:i + 10000])].double()) for i in range(0, len(ids), 10000)]
    tm = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del E
    lf = logf(counts, total, ids)
    summ["vocab"] = {"n_ids": int(len(ids)), **{k: float(stats.spearmanr(lf, tm[k]).statistic)
                                                 for k in ("nr", "nc", "mah", "frac10", "top10", "mu_proj",
                                                           "mu_share")}}
    V = pd.DataFrame({"id": ids, "lf": lf, **{k: tm[k] for k in ("nr", "nc", "mah", "frac10", "mu_proj", "mu_share")}})
    V["lf_bin"] = pd.cut(V.lf, [-np.inf, -1, 0, 1, 2, 3, np.inf])
    summ["vocab"]["by_lf_bin"] = V.groupby("lf_bin", observed=True)[["nr", "nc", "mah", "frac10", "mu_share"]].median() \
        .reset_index().astype({"lf_bin": str}).to_dict(orient="records")
    summ["vocab"]["count_by_lf_bin"] = V.groupby("lf_bin", observed=True).size().astype(int).to_dict()
    summ["vocab"]["count_by_lf_bin"] = {str(k): int(v) for k, v in summ["vocab"]["count_by_lf_bin"].items()}
    V.sample(min(len(V), 20000), random_state=0).to_csv(FDIR / "vocab_sample.csv", index=False)
    dump(summ, FDIR / "summary.json")
    print(pd.Series({k: v for k, v in summ.items() if k in ("pooled", "vocab")}).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["counts", "analyse"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    build_counts() if a.what == "counts" else analyse()


if __name__ == "__main__":
    main()
