#!/usr/bin/env python
"""Pick the fixed Global-MMLU ids (50 validation + 200 test) and write the per-language files.

Source: CohereLabs/Global-MMLU, ``test`` split (the ``dev`` split is only the 285
five-shot exemplars and has no cultural-sensitivity labels). Pool: items annotated as
culturally agnostic (``cultural_sensitivity_label == "CA"``), with four distinct,
non-empty options in every language, the same answer letter in every language, and an
English question of at most MAX_Q_CHARS characters (keeps the Thai NLLB translation of
reading passages out). Items are drawn round-robin over the 57 subjects (subject
order and within-subject order shuffled with SEED); the first 50 drawn are validation
(50 different subjects), the next 200 test (every subject 3-4 times). Validation and
test ids are disjoint and identical in every language.

Writes results/ml/data/mmlu/{ids.json, <lang>.json} for en, de, es, fr, it, pt, hi.
Thai is produced by build.py from the NLLB translation of the English items.

    python -m experiments.ml.evaldata.select_mmlu
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from experiments.ml.evaldata.common import MMLU_DIR

SEED = 0
N_VAL, N_TEST = 50, 200
MAX_Q_CHARS = 1000
GMMLU_LANGS = ("en", "de", "es", "fr", "it", "pt", "hi")
OPT_COLS = ("option_a", "option_b", "option_c", "option_d")


def snapshot() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download("CohereLabs/Global-MMLU", repo_type="dataset",
                                  allow_patterns=[f"{l}/test-*" for l in GMMLU_LANGS]))


def main() -> None:
    root = snapshot()
    dfs = {l: pd.read_parquet(next((root / l).glob("test-*.parquet"))).set_index("sample_id")
           for l in GMMLU_LANGS}
    en = dfs["en"]
    ok = (en.cultural_sensitivity_label == "CA") & (en.question.str.len() <= MAX_Q_CHARS)
    for l, df in dfs.items():
        df = df.reindex(en.index)
        opts = df[list(OPT_COLS)].fillna("").astype(str).apply(lambda r: [o.strip() for o in r], axis=1)
        ok &= opts.apply(lambda o: all(o) and len(set(o)) == 4)
        ok &= df.question.fillna("").str.strip().str.len() > 0
        ok &= df.answer.str.strip().str.upper() == en.answer.str.strip().str.upper()
    pool = en[ok]
    print(f"pool: {len(pool)} of {len(en)} items, {pool.subject.nunique()} subjects")

    rng = random.Random(SEED)
    subjects = sorted(pool.subject.unique())
    rng.shuffle(subjects)
    queues = {}
    for s in subjects:
        ids = sorted(pool.index[pool.subject == s])
        rng.shuffle(ids)
        queues[s] = ids
    order = []
    while len(order) < N_VAL + N_TEST and any(queues.values()):
        for s in subjects:
            if queues[s] and len(order) < N_VAL + N_TEST:
                order.append(queues[s].pop())
    ids = {"validation": order[:N_VAL], "test": order[N_VAL:N_VAL + N_TEST]}
    meta = {"source": "CohereLabs/Global-MMLU (test split)", "seed": SEED,
            "pool": "cultural_sensitivity_label == CA, 4 distinct options in all of " + ",".join(GMMLU_LANGS)
                    + f", same answer in all, en question <= {MAX_Q_CHARS} chars",
            "pool_size": int(len(pool)), **ids}
    MMLU_DIR.mkdir(parents=True, exist_ok=True)
    (MMLU_DIR / "ids.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    for l, df in dfs.items():
        out = {}
        for split, sel in ids.items():
            rows = []
            for sid in sel:
                r = df.loc[sid]
                rows.append({"id": sid, "subject": r.subject, "question": r.question.strip(),
                             "options": {L: str(r[c]).strip() for L, c in zip("ABCD", OPT_COLS)},
                             "answer": r.answer.strip().upper()})
            out[split] = rows
        (MMLU_DIR / f"{l}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {l}.json: " + ", ".join(f"{k}={len(v)}" for k, v in out.items()))


if __name__ == "__main__":
    main()
