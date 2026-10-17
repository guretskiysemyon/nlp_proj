"""Shared helpers for the whitening experiments: model registry, embedding-row loading (from safetensors only,
never the full model), token-id filters, chunked background statistics, logging/json helpers.

Background statistics are computed in float64 in row chunks (sum and X^T X), so the full-vocabulary background
never needs an (N, d) float64 copy in RAM. The returned dict has the same keys as sharpen.fit_background
(mu, lam, Q, n), with lam clamped at 0 and ascending like torch.linalg.eigh.
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE / "data"
LOGS = HERE / "logs"
REPORTS = HERE / "reports"
for _p in (str(HERE), str(ROOT), str(HERE.parent / "resample_sentences"), str(HERE.parent / "multilingual_hp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sharpen import fit_background  # noqa: E402,F401  (re-exported)

MODELS = {
    "gemma": {"name": "google/gemma-2-2b-it", "d": 2304, "scale": math.sqrt(2304), "rank": 100,
              "min_types": 30_000},
    "llama": {"name": "meta-llama/Llama-3.1-8B-Instruct", "d": 4096, "scale": 1.0, "rank": 200,
              "min_types": 40_000},
}
CONCEPTS = ["Harry Potter", "Ancient Rome", "Valentine's Day"]
SEED = 42


def slug(concept: str) -> str:
    return re.sub(r"\W+", "_", concept).strip("_")


def load_env():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


def model_dir(model: str) -> Path:
    p = DATA / model
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def write_json(path: Path, obj, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent))
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=2)
def tokenizer(model: str):
    load_env()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODELS[model]["name"], token=os.environ.get("HF_TOKEN"))


@lru_cache(maxsize=2)
def excluded_ids(model: str) -> frozenset:
    """Special, unused/reserved, and byte-fallback token ids (never part of any background or V*)."""
    tok = tokenizer(model)
    ex = set(tok.all_special_ids)
    ex |= {i for i, a in tok.added_tokens_decoder.items() if a.special}
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    for i, p in enumerate(pieces):
        if p is None:
            ex.add(i)
        elif re.fullmatch(r"<unused\d+>|<0x[0-9A-Fa-f]{2}>|<\|reserved_special_token_\d+\|>|<mask>|<2mass>|\[@BOS@\]"
                          r"|\[toxicity=0\]", p):
            ex.add(i)
    if model == "llama":
        ex |= set(range(128000, len(tok)))
    return frozenset(ex)


def decode(model: str, ids) -> list[str]:
    return tokenizer(model).convert_ids_to_tokens([int(i) for i in ids])


def _embed_file(model: str) -> Path:
    from huggingface_hub import snapshot_download
    load_env()
    snap = Path(snapshot_download(MODELS[model]["name"], allow_patterns=["*.json"], local_files_only=True))
    idx = json.loads((snap / "model.safetensors.index.json").read_text())
    return snap / idx["weight_map"]["model.embed_tokens.weight"]


class Embeddings:
    """bf16 embedding matrix loaded from the safetensors shard (no model). rows(ids) -> float32 (n, d),
    already multiplied by the TransformerLens scale for Gemma (sqrt(d_model))."""

    def __init__(self, model: str):
        from safetensors import safe_open
        self.model = model
        t0 = time.time()
        with safe_open(str(_embed_file(model)), framework="pt") as f:
            self.E = f.get_tensor("model.embed_tokens.weight")  # bf16 (V, d)
        self.scale = MODELS[model]["scale"]
        log(f"[emb {model}] loaded {tuple(self.E.shape)} {self.E.dtype} in {time.time() - t0:.1f}s")

    @property
    def V(self):
        return self.E.shape[0]

    def rows(self, ids, dtype=torch.float32) -> torch.Tensor:
        idx = torch.as_tensor(list(ids) if not torch.is_tensor(ids) else ids, dtype=torch.long)
        return self.E.index_select(0, idx).to(dtype) * self.scale

    def free(self):
        del self.E
        gc.collect()


def chunked_stats(emb: Embeddings, ids, chunk: int = 8192) -> dict:
    """Same output as sharpen.fit_background(emb.rows(ids)) but streamed in float64 chunks."""
    ids = list(ids)
    d = emb.E.shape[1]
    s = torch.zeros(d, dtype=torch.float64)
    xtx = torch.zeros(d, d, dtype=torch.float64)
    # two-pass for accuracy: mean first, then centred second moment
    for i in range(0, len(ids), chunk):
        s += emb.rows(ids[i:i + chunk], torch.float64).sum(0)
    n = len(ids)
    mu = s / n
    for i in range(0, len(ids), chunk):
        X = emb.rows(ids[i:i + chunk], torch.float64) - mu
        xtx += X.T @ X
    cov = xtx / n
    lam, Q = torch.linalg.eigh(cov)
    return {"mu": mu, "lam": lam.clamp_min(0), "Q": Q, "n": n}


def top_eigvecs(stats: dict, k: int) -> torch.Tensor:
    return stats["Q"][:, -k:]  # eigh is ascending


def latin_filter(model: str, ids) -> list[int]:
    pat = re.compile(r"[▁Ġ ]?[A-Za-z0-9.,;:!?'\"()\-]+")
    pieces = decode(model, ids)
    return [int(i) for i, p in zip(ids, pieces) if p is not None and pat.fullmatch(p)]


def principal_angle_sines_top(sa: dict, sb: dict, k: int) -> torch.Tensor:
    from sharpen import principal_angle_sines
    return principal_angle_sines(top_eigvecs(sa, k), top_eigvecs(sb, k))
