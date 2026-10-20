"""Shared constants and helpers for the multilingual Llama experiment (experiments/ml/PLAN.md).

    E = embeddings()                         # (vocab, 4096) float32, Llama input embeddings (TL W_E = HF embed_tokens)
    ts = token_set(sentences, labels, langs) # V*, per-token codes (1 concept-only, 0 neutral-only, 2 both)
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "ml"
DATA = RESULTS / "data"
RUNS = RESULTS / "runs"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LANGS = ("en", "de", "fr", "it", "pt", "hi", "es", "th")
APPROACHES = ("A1", "A2", "A3")
RANK, SPARSITY, RIDGE, SEED = 200, 0.01, 1e-4, 42
TAU, TOP_N, CONFIDENCE, TOP_K_TOKENS = 2.0, 10, 0.85, 20
EJ_T, EJ_WORDS = 5, 20
DELTA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
MIN_FRAC = 0.9


def slug(concept: str) -> str:
    return re.sub(r"\W+", "_", concept).strip("_")


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


def concepts():
    """The concept set chosen by $ML_CONCEPTS: "orig" (default, EMBER's 18, data/mc_questions.json), "new" (the 20
    of data/concept_sentences_extra.json, PREREG_NEW.md) or "all" (orig then new)."""
    import os
    orig = list(json.loads((ROOT / "data" / "mc_questions.json").read_text(encoding="utf-8")))
    new = [e["concept"] for e in json.loads((ROOT / "data" / "concept_sentences_extra.json").read_text(encoding="utf-8"))]
    return {"orig": orig, "new": new, "all": orig + new}[os.environ.get("ML_CONCEPTS", "orig")]


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def model_dir() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"]))


@lru_cache(maxsize=1)
def embeddings() -> torch.Tensor:
    """Llama input embeddings, loaded from the safetensors shard only (fast; no model)."""
    from safetensors import safe_open
    d = model_dir()
    index = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(d / index["model.embed_tokens.weight"]), framework="pt") as f:
        return f.get_tensor("model.embed_tokens.weight").float()


def special_ids() -> set:
    tok = tokenizer()
    return {t for t in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) if t is not None} | \
        set(tok.all_special_ids)


def token_set(sentences, labels, langs=None) -> dict:
    """V* as in EMBER: unique non-special token ids of the sentences (TransformerLens tokenization: BOS + tokens).
    codes[t] = 1 if t occurs only in concept sentences, 0 only in neutral ones, 2 in both."""
    tok = tokenizer()
    special = special_ids()
    in_c, in_n = set(), set()
    for s, lab in zip(sentences, labels):
        ids = set(tok(s, add_special_tokens=False)["input_ids"]) - special
        (in_c if lab == "concept" else in_n).update(ids)
    vstar = sorted(in_c | in_n)
    codes = np.array([2 if (t in in_c and t in in_n) else 1 if t in in_c else 0 for t in vstar], dtype=np.int8)
    return {"vstar_ids": vstar, "codes": codes, "token_to_code": dict(zip(vstar, codes.tolist())),
            "sentences": list(sentences), "labels": list(labels), "langs": list(langs) if langs else None}


def run_dir(concept: str, approach: str, lang: str | None = None) -> Path:
    d = RUNS / slug(concept) / approach
    return d / lang if lang else d
