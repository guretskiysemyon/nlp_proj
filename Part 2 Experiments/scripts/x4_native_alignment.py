#!/usr/bin/env python
"""X4: do each language's native concept words lie along the method's concept directions? (CPU, no model)

Part 1's concept-axis probe used hand-translated word lists; this uses only native data. Per (concept, language):
word tokens ('▁' + >= 3 letters) that occur only in that language's native concept sentences (concept-only) vs only
in its native neutral sentences (data/ for en, experiments/multilingual_fit/data_v2 for de/es/fr/ru). Score of a
token = share of its embedding's energy inside the span of the method's selected feature directions (in embedding
space, where the edit acts). AUC = how well that score separates concept from neutral tokens (0.5 = chance).

  all      every concept-only word token of the language
  native   only tokens that never occur in the English concept or neutral sentences (no shared surface forms)
  null     the same AUC for k random NON-selected features of the same fit (mean of 20 draws), k = #selected

Output: Part 2 Experiments/x4_native_alignment/{auc.csv, report.md}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


MIN_COUNT = 3


def codes(concept, lang):
    """(tid -> code, tid -> occurrences in the concept sentences, tid -> occurrences in the neutral sentences)."""
    from collections import Counter
    from ember.local_datasets import ConceptDataset
    from ember.utils import build_token_label_codes
    ds = ConceptDataset(concept_name=concept, neutral_sample_seed=42, languages=[lang],
                        multilingual_data_dir=p2x.DATA_V2)
    cc, nc = Counter(), Counter()
    for s, lbl in ds.data:
        (cc if lbl == concept else nc).update(p2x.tokenizer()(s, add_special_tokens=False)["input_ids"])
    return build_token_label_codes(concept, ds, p2x.tokenizer()), cc, nc


def energy(E, ids, D):
    Q, _ = torch.linalg.qr(D)
    X = E[torch.tensor(ids)]
    return ((X @ Q).pow(2).sum(1) / X.pow(2).sum(1)).numpy()


def auc(E, pos, neg, D):
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    s = energy(E, pos + neg, D)
    return roc_auc_score([1] * len(pos) + [0] * len(neg), s)


def main():
    out = p2x.outdir("x4_native_alignment")
    E = p2x.embeddings()
    from ember.erasure import features
    rng = np.random.default_rng(0)
    rows = []
    for c in p2x.CONCEPTS:
        en_all = set(codes(c, "en")[0])
        for L in p2x.LANGS:
            cd, cc, nc = codes(c, L)
            ids = list(cd)
            pieces = dict(zip(ids, p2x.tok_str(ids)))
            # characteristic words: >= MIN_COUNT occurrences on one side, none on the other
            pos = [t for t in ids if cd[t] == 1 and cc[t] >= MIN_COUNT and p2x.is_word_token(pieces[t])]
            neg = [t for t in ids if cd[t] == 0 and nc[t] >= MIN_COUNT and p2x.is_word_token(pieces[t])]
            pos_n = [t for t in pos if t not in en_all]
            neg_n = [t for t in neg if t not in en_all]
            for m in p2x.METHODS:
                D = p2x.directions(c, m)
                r = {"concept": c, "language": L, "method": m, "n_concept_words": len(pos),
                     "n_native_only": len(pos_n)}
                if D is None:
                    rows.append({**r, "auc_all": None, "auc_native": None, "auc_null_all": None})
                    continue
                features.MF_OUTPUTS_ROOT = p2x.METHODS[m]
                F, _, _ = features.load_embedding_payload(features._embedding_paths(p2x.C.MODEL_NAME, c, p2x.C.RANK,
                                                                                    p2x.C.SEED)[0])
                F = torch.as_tensor(F).double()
                sel = set(features.select_embed_feature_ids(pd.read_csv(features._embedding_paths(
                    p2x.C.MODEL_NAME, c, p2x.C.RANK, p2x.C.SEED)[1]), ratio_thresh=2.0))
                others = [i for i in range(F.shape[1]) if i not in sel]
                null = [auc(E, pos, neg, F[:, rng.choice(others, D.shape[1], replace=False)]) for _ in range(20)]
                top = []
                if pos_n:
                    sc = energy(E, pos_n + neg_n, D)
                    order = np.argsort(-sc)[:12]
                    allp = pos_n + neg_n
                    top = [f"{pieces[allp[i]]}{'*' if i < len(pos_n) else ''}" for i in order]
                rows.append({**r, "auc_all": auc(E, pos, neg, D), "auc_native": auc(E, pos_n, neg_n, D),
                             "auc_null_all": float(np.nanmean(null)), "top_native_by_energy": " ".join(top)})
            print(c, L, len(pos), len(pos_n), flush=True)
    R = pd.DataFrame(rows).round(3)
    R.to_csv(out / "auc.csv", index=False)
    piv = R.pivot_table(index=["concept", "method"], columns="language", values="auc_native").round(2).reset_index()
    piv_all = R.pivot_table(index=["concept", "method"], columns="language", values="auc_all").round(2).reset_index()
    L_ = ["# X4: native concept words vs the method's concept directions (AUC, 0.5 = chance)", "",
          "Concept-only vs neutral-only word tokens of each language's native Wikipedia sentences; score = share of "
          "the token's embedding energy in the span of the selected feature directions. `native`: tokens that never "
          "occur in the English sentences (no shared surface form). Null (random other features of the same fit) "
          f"in auc.csv. Words: >= {MIN_COUNT} occurrences in that language's concept sentences and none in its neutral "
          "ones (and vice versa).", "", "## Native-only tokens", "", p2x.md_table(piv), "", "## All concept-only word tokens", "",
          p2x.md_table(piv_all), "", "## Null: random other features of the same fit (all tokens)", "",
          p2x.md_table(R.pivot_table(index=["concept", "method"], columns="language", values="auc_null_all")
                       .round(2).reset_index()), "",
          "## Top native words by energy in the concept directions (* = concept word)", ""]
    for r in R.itertuples():
        if isinstance(r.top_native_by_energy, str) and r.top_native_by_energy and r.language != "en":
            L_.append(f"- {r.concept} / {r.method} / {r.language}: {r.top_native_by_energy}")
    (out / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
