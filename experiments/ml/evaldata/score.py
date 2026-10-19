#!/usr/bin/env python
"""Quality columns for every cached (English, translation) segment; resumable, one file per column.

    <comet venv>/bin/python -I experiments/ml/evaldata/score.py --engine nllb ctx [madlad] --metrics labse [chrf_bt kiwi]

Columns (work/score_<metric>_<engine>_<lang>.jsonl, rows {"src", "tr", "value"}):
  labse    cosine of sentence-transformers/LaBSE embeddings of source and translation (ungated)
  chrf_bt  chrF (sacrebleu, 0-1) of the NLLB round-trip back-translation against the English source
           (needs work/bt_nllb_<lang>.jsonl from translate.py; nllb only)
  kiwi     COMET-Kiwi, Unbabel/wmt22-cometkiwi-da (gated: needs our HF_TOKEN to have accepted the
           license), same call as experiments/multilingual_fit/score_translations.py
Adding a column later never needs a retranslation: it only reads the translation caches.

Runs in the comet environment (/home/morg/NLP_2526b/semyong/.venv-comet: unbabel-comet,
sentence-transformers, sacrebleu; unbabel-comet pins protobuf < 5, so not in the project .venv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ml.evaldata.common import (ENGINES, METRICS, MT_LANGS, candidates, WORK, append_jsonl, load_jsonl,  # noqa: E402
                                            score_path, scores)

CHUNK = 4096


def todo_pairs(metric: str, engine: str, lang: str):
    done = scores(metric, engine, lang)
    if metric == "chrf_bt":
        rows = load_jsonl(WORK / f"bt_{engine}_{lang}.jsonl")
        return list({(r["src"], r["tr"]): r["bt"] for r in rows if (r["src"], r["tr"]) not in done}.items())
    pairs = [(s, t) for s, ts in candidates(engine, lang).items() for t in ts]
    return [p for p in pairs if p not in done]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", nargs="+", choices=ENGINES, required=True)
    ap.add_argument("--metrics", nargs="+", choices=METRICS, default=["labse"])
    ap.add_argument("--langs", nargs="+", default=list(MT_LANGS))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for metric in args.metrics:
      for engine in args.engine:
        if metric == "chrf_bt" and engine != "nllb":
            continue
        todo = {l: todo_pairs(metric, engine, l) for l in args.langs}
        print(f"{metric}/{engine}: " + ", ".join(f"{l}={len(t)}" for l, t in todo.items()), flush=True)
        if not any(todo.values()):
            continue
        if metric == "labse":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/LaBSE", device=dev)

            def score(chunk):
                e = model.encode([s for s, _ in chunk], batch_size=args.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)
                t = model.encode([t for _, t in chunk], batch_size=args.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)
                return (e * t).sum(1).tolist()
        elif metric == "kiwi":
            from comet import download_model, load_from_checkpoint
            model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))

            def score(chunk):
                out = model.predict([{"src": s, "mt": t} for s, t in chunk], batch_size=args.batch_size,
                                    gpus=1 if dev == "cuda" else 0, progress_bar=False)
                return list(out.scores)
        else:
            from sacrebleu.metrics import CHRF
            chrf = CHRF()

            def score(chunk):
                return [chrf.sentence_score(bt, [src]).score / 100 for (src, _), bt in chunk]

        for lang, pairs in todo.items():
            for i in range(0, len(pairs), CHUNK):
                chunk = pairs[i:i + CHUNK]
                vals = score(chunk)
                keys = [p[0] for p in chunk] if metric == "chrf_bt" else chunk
                append_jsonl(score_path(metric, engine, lang),
                             [{"src": s, "tr": t, "value": float(v)} for (s, t), v in zip(keys, vals)])
                print(f"  {metric}/{lang}: {min(i + CHUNK, len(pairs))}/{len(pairs)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
