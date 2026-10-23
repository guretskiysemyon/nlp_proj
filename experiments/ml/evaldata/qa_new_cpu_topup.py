#!/usr/bin/env python
"""CPU top-up of stage-1 NLLB rows for segments added after the GPU stage-1 runs started (the 3 English items replaced
by qa_new_gen patch). Same model and decoding as translate.Engine (NLLB-200 3.3B, beam 4, max_new_tokens rule,
digit normalisation), fp32 on CPU. Each cache file gets ONE os.write (atomic O_APPEND) so a concurrent GPU writer of the
same file cannot interleave lines.

    python -m experiments.ml.evaldata.qa_new_cpu_topup --langs de fr hi es th it pt [--ctx-langs de fr hi es th]
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from experiments.ml.evaldata import qa_new_translate as T
import experiments.ml.evaldata.common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--ctx-langs", nargs="*", default=[])
    args = ap.parse_args()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", torch_dtype=torch.float32).eval()
    segs, lists = T.segments(), T.option_lists()
    for lang in sorted(set(args.langs) | set(args.ctx_langs)):
        for name, todo in (("nllb", segs), ("ctx", lists)):
            if (name == "nllb" and lang not in args.langs) or (name == "ctx" and lang not in args.ctx_langs):
                continue
            path = T.TR_WORK / f"{name}_{lang}.jsonl"
            done = C.cache_map(path)
            miss = [s for s in todo if s not in done]
            if len(miss) > 100:
                raise SystemExit(f"{name}/{lang}: {len(miss)} missing, not a top-up")
            rows = []
            for i in range(0, len(miss), 8):
                b = miss[i:i + 8]
                enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=512)
                n_in = int(enc["attention_mask"].sum(1).max())
                with torch.no_grad():
                    out = model.generate(**enc, num_beams=4, max_new_tokens=min(512, 2 * n_in + 16),
                                         forced_bos_token_id=tok.convert_tokens_to_ids(C.NLLB_CODE[lang]))
                rows += [{"src": s, "tr": C.normalize_digits(t.strip())}
                         for s, t in zip(b, tok.batch_decode(out, skip_special_tokens=True))]
            if rows:
                fd = os.open(str(path), os.O_WRONLY | os.O_APPEND)
                try:
                    os.write(fd, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
                finally:
                    os.close(fd)
            print(f"{name}/{lang}: +{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
