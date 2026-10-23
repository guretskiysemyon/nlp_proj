#!/usr/bin/env python
"""LaBSE column for the new concepts' translation caches (results/ml/data/qa_new/work/tr), as score.py does for
qa/ and qa_v2/. Runs in the comet environment (sentence-transformers):

    /home/morg/NLP_2526b/semyong/.venv-comet/bin/python -I experiments/ml/evaldata/qa_new_score.py \
        --engine nllb ctx [nllb_nbest madlad_nbest madlad_ctx] [--langs de fr ...]

Rows: work/tr/score_labse_<engine>_<lang>.jsonl {"src", "tr", "value"} (cosine of normalized LaBSE embeddings).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import experiments.ml.evaldata.common as C  # noqa: E402
from experiments.ml.evaldata.qa_new_common import TR_LANGS, WORK  # noqa: E402

C.WORK = WORK / "tr"
ENGINES = ("nllb", "ctx", "madlad", "nllb_nbest", "madlad_nbest", "madlad_ctx")
CHUNK = 4096


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", nargs="+", choices=ENGINES, required=True)
    ap.add_argument("--langs", nargs="+", default=list(TR_LANGS))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    for engine in args.engine:
        for lang in args.langs:
            done = C.scores("labse", engine, lang, reload=True)
            pairs = [(s, t) for s, ts in C.candidates(engine, lang).items() for t in ts if (s, t) not in done]
            print(f"labse/{engine}/{lang}: {len(pairs)} to score", flush=True)
            if not pairs:
                continue
            if model is None:
                model = SentenceTransformer("sentence-transformers/LaBSE", device=dev)
            for i in range(0, len(pairs), CHUNK):
                chunk = pairs[i:i + CHUNK]
                e = model.encode([s for s, _ in chunk], batch_size=args.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)
                t = model.encode([t for _, t in chunk], batch_size=args.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)
                C.append_jsonl(C.score_path("labse", engine, lang),
                               [{"src": s, "tr": tr, "value": float(v)} for (s, tr), v in zip(chunk, (e * t).sum(1))])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
