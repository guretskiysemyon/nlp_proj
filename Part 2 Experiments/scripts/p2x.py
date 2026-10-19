"""Shared helpers for the Part 2 CINDER vs EMBER analyses (Part 2 Experiments/).

Both methods' edits are reproduced exactly as ember/erasure/embed_edit.py applies them (cinder.edit_size.edit_vectors):
EMBER from mf_outputs/ (Part 1, the paper's features), CINDER from mf_outputs/cinder/ (Ori's L123, seed 42).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Part 2 Experiments"
for p in (str(ROOT), str(ROOT / "experiments" / "multilingual_fit")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cinder import common as C  # noqa: E402

CONCEPTS = C.CONCEPTS
LANGS = ["en", "de", "es", "fr", "ru"]
METHODS = {"EMBER": C.EMBER_MF_ROOT, "CINDER": C.MF_ROOT}
# extrapolated variants (x5b_build_extrapolated.py): same features and English edit + native foreign tokens
METHODS_X = {"EMBER+X": ROOT / "mf_outputs" / "extrap_EMBER", "CINDER+X": ROOT / "mf_outputs" / "extrap_CINDER"}
ALL_METHODS = {**METHODS, **METHODS_X}
COLORS = {"EMBER": "#2a78d6", "CINDER": "#eb6834",      # categorical slots 1-4 (dataviz default palette)
          "EMBER+X": "#1baf7a", "CINDER+X": "#eda100"}
DATA_V2 = ROOT / "experiments" / "multilingual_fit" / "data_v2"


def outdir(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.MODEL_NAME)


@lru_cache(maxsize=1)
def embeddings():
    """Scaled input embeddings (x sqrt(d), the space the edit works in), float64."""
    from experiments.varwidth.prep import load_input_embeddings
    return load_input_embeddings().double()


@lru_cache(maxsize=None)
def edit(concept: str, method: str):
    """(token ids, c_t rows (n, d) float64) of the method's edit at delta = 1."""
    from cinder.edit_size import edit_vectors
    return edit_vectors(concept, METHODS[method])


@lru_cache(maxsize=None)
def directions(concept: str, method: str):
    """Selected feature directions in embedding space, (d, k') float64, or None."""
    import pandas as pd
    import torch
    from ember.erasure import features
    features.MF_OUTPUTS_ROOT = METHODS[method]
    pkl, pot = features._embedding_paths(C.MODEL_NAME, concept, C.RANK, C.SEED)
    df = pd.read_csv(pot)
    if df.empty:
        return None
    F, _, _ = features.load_embedding_payload(pkl)
    ids = features.select_embed_feature_ids(df, ratio_thresh=2.0)
    return torch.as_tensor(F[:, ids]).double()


def tok_str(ids):
    return tokenizer().convert_ids_to_tokens([int(i) for i in ids])


def is_word_token(piece: str) -> bool:
    """A whole-word-start token: '▁' + at least 3 letters (any script)."""
    return piece.startswith("▁") and len(piece) >= 4 and piece[1:].isalpha()


def text_ids(text: str) -> set:
    return set(tokenizer()(" " + text.strip(), add_special_tokens=False)["input_ids"])


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def md_table(df) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and v != v) else str(v) for v in r) + " |")
    return "\n".join(lines)
