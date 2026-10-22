"""WP9 (ml_geometry): language identity in the multilingual fits (A1/A2/A3) and the selection cost.

Reads the multilingual study (branch orip-ml, worktree ../nlp_proj_ml) READ-ONLY: fits, selections, judge caches,
lexicons, logs. Nothing is written there and none of its code is imported (its helpers are re-implemented here,
so no __pycache__ lands in that tree). Outputs go to results/paper_analysis/ml_geometry/.

Run with (Thai word splitting needs pythainlp from the ML study's extra library dir):
    env HF_HOME=/home/morg/NLP_2526b/oripetel/.cache/huggingface HF_HUB_OFFLINE=1 \
        PYTHONPATH=/home/morg/NLP_2526b/oripetel/.cache/pylib_ml .venv/bin/python -m experiments.paper_analysis.ml_geometry.<module>
"""
from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/morg/NLP_2526b/oripetel/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_PYLIB = "/home/morg/NLP_2526b/oripetel/.cache/pylib_ml"
if _PYLIB not in sys.path:
    sys.path.append(_PYLIB)

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ML = ROOT.parent / "nlp_proj_ml"                       # read-only
MLRES = ML / "results" / "ml"
OUT = ROOT / "results" / "paper_analysis" / "ml_geometry"
CACHE = OUT / "cache"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LANGS = ("en", "de", "fr", "it", "pt", "hi", "es", "th")
SCRIPTS = ("lat", "deva", "thai", "other")
SEEDS = (42, 1, 2)
TAU, TOP_N, EJ_T, EJ_WORDS = 2.0, 10, 5, 20
DOM = 0.8                                              # "language-dominated": >= 80% of a feature's |Y| mass in one language

COLORS = {"A1": "#7F7F7F", "A2": "#7F7F7F", "A3": "#D55E00", "A3z": "#E69F00", "CS": "#0072B2", "other": "#009E73"}


def slug(concept: str) -> str:
    return re.sub(r"\W+", "_", concept).strip("_")


def concepts() -> list:
    orig = list(json.loads((ML / "data" / "mc_questions.json").read_text(encoding="utf-8")))
    new = [e["concept"] for e in json.loads((ML / "data" / "concept_sentences_extra.json").read_text(encoding="utf-8"))]
    return orig + new


def is_new(concept: str) -> bool:
    return concept in concepts()[18:]


def fit_dir(concept: str, approach: str, seed: int = 42, lang: str | None = None) -> Path:
    d = (MLRES / "runs" if seed == 42 else MLRES / "seeds" / f"fs{seed}") / slug(concept) / approach
    return d / lang if lang else d


def load_full(concept: str) -> dict:
    return json.loads((MLRES / "data" / "full" / f"{slug(concept)}.json").read_text())


def flatten(sets: dict, langs=LANGS):
    """Same order as experiments/ml/fit._flatten: per language, concept then neutral sentences."""
    sents, labels, ls = [], [], []
    for L in langs:
        for side in ("concept", "neutral"):
            for r in sets[L][side]:
                sents.append(r["text"] if isinstance(r, dict) else r)
                labels.append(side)
                ls.append(L)
    return sents, labels, ls


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


@lru_cache(maxsize=1)
def special_ids() -> frozenset:
    tok = tokenizer()
    return frozenset({t for t in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) if t is not None}
                     | set(tok.all_special_ids))


def tok_str(t: int) -> str:
    """As experiments/ml/select.tok_str."""
    tok = tokenizer()
    s = tok.decode([int(t)])
    return s if s.strip() and "�" not in s else tok.convert_ids_to_tokens(int(t))


_DEVA = re.compile(r"[ऀ-ॿ]")
_THAI = re.compile(r"[฀-๿]")
_LAT = re.compile(r"[A-Za-zÀ-ɏ]")


def script_of(t: int) -> str:
    s = tokenizer().decode([int(t)])
    if _DEVA.search(s):
        return "deva"
    if _THAI.search(s):
        return "thai"
    if _LAT.search(s):
        return "lat"
    return "other"


@lru_cache(maxsize=1)
def embeddings():
    """Llama input embeddings (vocab, 4096) float32 from the safetensors shard."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"]))
    index = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(d / index["model.embed_tokens.weight"]), framework="pt") as f:
        return f.get_tensor("model.embed_tokens.weight").float()


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                               default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
