#!/usr/bin/env python
"""WP2 step 2 input: an independent semantic embedding of every feature (no Gemma, no Gemini).

    .venv/bin/python -m experiments.paper_analysis.feature_geometry.embed_text

Encoder: sentence-transformers/all-MiniLM-L6-v2 loaded with plain transformers, mean pooling over non-pad tokens,
l2-normalized (the model card's recipe). Texts per feature (key "concept|cfg|seed|f"):
    w20  ", ".join(first 20 normalized source words)       (words.json, every feature)
    w10  ", ".join(first 10)                                (sensitivity)
    desc stage-1 Gemini description from judge.json        (features.parquet `desc`, only rho >= 2 features)
Writes text_emb.npz {keys_w (n,), w20 (n, 384), w10 (n, 384), keys_d (m,), desc (m, 384)}.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from experiments.paper_analysis.feature_geometry.common import OUT

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def encoder():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()

    @torch.inference_mode()
    def enc(texts: list, bs: int = 128) -> np.ndarray:
        uniq = sorted(dict.fromkeys(texts), key=len)        # length-sorted batches: little padding
        out = {}
        for n, i in enumerate(range(0, len(uniq), bs)):
            b = uniq[i:i + bs]
            t = tok(b, padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = model(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
            out.update(zip(b, v.numpy()))
            if n % 50 == 0:
                print(f"  {i + len(b)}/{len(uniq)}", flush=True)
        return np.stack([out[x] for x in texts]).astype(np.float32)
    return enc


def main():
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "text_emb.npz"))
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    words = json.loads((OUT / "words.json").read_text())
    df = pd.read_parquet(OUT / "features.parquet")
    enc = encoder()
    keys = list(words)
    w20 = enc([", ".join(words[k]) if words[k] else "(none)" for k in keys])
    w10 = enc([", ".join(words[k][:10]) if words[k] else "(none)" for k in keys])
    d = df[df.desc.notna()]
    kd = [f"{r.concept}|{r.cfg}|{r.seed}|{r.feature}" for r in d.itertuples()]
    de = enc(d.desc.tolist())
    tmp = a.out + ".tmp.npz"
    np.savez_compressed(tmp, keys_w=np.array(keys), w20=w20, w10=w10, keys_d=np.array(kd), desc=de)
    os.replace(tmp, a.out)
    print(f"{len(keys)} word lists, {len(kd)} descriptions -> {a.out}")


if __name__ == "__main__":
    main()
