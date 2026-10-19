#!/usr/bin/env python
"""X2: collateral exposure. How often do the edited tokens occur in ordinary text? (CPU, tokenizer only)

Ordinary text = the 300,000 English Wikipedia sentences of the whitening background
(experiments/whitening/data/wiki_sentences.jsonl, 4 random sentences per article, seeded). An edited token changes
the model's input wherever it occurs, so an edit whose tokens occur in many ordinary sentences should cost more
specificity. Per concept and method: share of background sentences with at least one edited token, edited-token
occurrences per 10,000 tokens, and the 15 edited tokens with the most occurrences.

Output: Part 2 Experiments/x2_collateral/{exposure.csv, top_tokens.csv, report.md}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    out = p2x.outdir("x2_collateral")
    path = p2x.ROOT / "experiments" / "whitening" / "data" / "wiki_sentences.jsonl"
    sents = [json.loads(l)["sentence"] for l in path.open(encoding="utf-8")]
    tok = p2x.tokenizer()
    ids, lens = [], []
    for i in range(0, len(sents), 5000):
        for seq in tok(sents[i:i + 5000], add_special_tokens=False)["input_ids"]:
            ids.extend(seq)
            lens.append(len(seq))
    flat = np.asarray(ids, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    nz = np.asarray(lens) > 0
    counts = np.bincount(flat, minlength=len(tok))
    print(f"{len(sents)} sentences, {len(flat)} tokens", flush=True)
    rows, top = [], []
    for c in p2x.CONCEPTS:
        for m in p2x.METHODS:
            tids, _ = p2x.edit(c, m)
            if not tids:
                rows.append({"concept": c, "method": m, "n_edited": 0, "sent_share_%": 0.0, "per_10k_tokens": 0.0})
                continue
            mark = np.zeros(len(tok), dtype=np.int8)
            mark[tids] = 1
            hit = np.zeros(len(sents), dtype=bool)
            hit[nz] = np.maximum.reduceat(mark[flat], starts[nz]) > 0
            rows.append({"concept": c, "method": m, "n_edited": len(tids),
                         "sent_share_%": round(100 * float(hit.mean()), 3),
                         "per_10k_tokens": round(1e4 * float(counts[tids].sum()) / len(flat), 2)})
            order = sorted(tids, key=lambda t: -counts[t])[:15]
            top += [{"concept": c, "method": m, "token": p, "count": int(counts[t])}
                    for t, p in zip(order, p2x.tok_str(order))]
    R, T = pd.DataFrame(rows), pd.DataFrame(top)
    R.to_csv(out / "exposure.csv", index=False)
    T.to_csv(out / "top_tokens.csv", index=False)
    L = ["# X2: collateral exposure of the edited tokens in ordinary English Wikipedia text", "",
         f"Background: {len(sents):,} sentences, {len(flat):,} tokens. `sent_share_%`: sentences with >= 1 edited "
         "token; `per_10k_tokens`: edited-token occurrences per 10,000 tokens.", "", p2x.md_table(R), "",
         "## Most frequent edited tokens", ""]
    for (c, m), g in T.groupby(["concept", "method"], sort=False):
        L.append(f"- **{c} / {m}**: " + ", ".join(f"`{r.token}` {r.count}" for r in g.itertuples()))
    (out / "report.md").write_text("\n".join(L) + "\n")
    print(p2x.md_table(R))


if __name__ == "__main__":
    main()
