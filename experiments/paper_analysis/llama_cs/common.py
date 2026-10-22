"""WP8 (llama_cs): single-language (English) contrastive sharpening (CS) on Llama-3.1-8B-Instruct.

Shared helpers: paths, the two models' input embeddings (loaded from the safetensors shard only), page targets
(token types of the concept's English Wikipedia page), V* (300 concept + 300 neutral sentences, as
experiments/varwidth/prep.py), background statistics, and the CS transform for any (eta, gamma, target).

Row convention as in experiments/whitening/sharpen.py: embeddings are rows; X = (E - mu_B) W S_eta^(gamma/2).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/morg/NLP_2526b/oripetel/.cache/huggingface")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "paper_analysis" / "llama_cs"
DATA = OUT / "data"
FITS = OUT / "fits"
WH = ROOT / "experiments" / "whitening"
for _p in (str(WH), str(ROOT / "experiments" / "resample_sentences"), str(ROOT / "experiments" / "multilingual_hp")):
    if _p not in sys.path:
        sys.path.append(_p)

MODELS = {
    "llama": {"name": "meta-llama/Llama-3.1-8B-Instruct", "d": 4096, "scale": 1.0, "rank": 200},
    "gemma": {"name": "google/gemma-2-2b-it", "d": 2304, "scale": math.sqrt(2304), "rank": 100},
}
EPS_REL = 1e-4
CONCEPTS18 = ['Culture of Greece', 'Golf', 'Republic of Ireland', 'Ancient Rome', 'Baseball', 'Uranium', 'Harry Potter',
              'Cannabis', 'Gambling', 'Gun', 'Pornography', 'COVID-19 pandemic', "Valentine's Day", 'World War II',
              'Nazism', 'Heroin', 'Halloween', 'Artificial intelligence']
FIT_CONCEPTS = ["Harry Potter", "Ancient Rome", "Valentine's Day", "Baseball"]
ML_ROOT = ROOT.parent / "nlp_proj_ml"
COLORS = {"ember": "#7F7F7F", "cs": "#0072B2", "cinder": "#D55E00", "cinder_z": "#E69F00", "other": "#009E73"}


def slug(concept: str) -> str:
    return re.sub(r"\W+", "_", concept).strip("_")


def write_json(path, obj, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- tokenizers / embeddings
@lru_cache(maxsize=2)
def tokenizer(model: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[model]["name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@lru_cache(maxsize=2)
def excluded_ids(model: str) -> frozenset:
    """Special / unused / reserved / byte-fallback ids (experiments/whitening/common.py's rule)."""
    import common as wc  # experiments/whitening/common.py
    wc.tokenizer.cache_clear()
    return frozenset(wc.excluded_ids(model))


@lru_cache(maxsize=2)
def embeddings(model: str) -> torch.Tensor:
    """Input embedding matrix (V, d) as float32, x sqrt(d) for Gemma (TransformerLens convention); shard only."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    snap = Path(snapshot_download(MODELS[model]["name"], allow_patterns=["*.json"], local_files_only=True))
    idx = snap / "model.safetensors.index.json"
    shard = json.loads(idx.read_text())["weight_map"]["model.embed_tokens.weight"] if idx.exists() \
        else "model.safetensors"
    with safe_open(str(snap / shard), framework="pt") as f:
        E = f.get_tensor("model.embed_tokens.weight").float()
    if MODELS[model]["scale"] != 1.0:
        E.mul_(MODELS[model]["scale"])
    return E


# ---------------------------------------------------------------- page targets
def page_sentences(concept: str) -> list[str]:
    """English Wikipedia page sentences: experiments/whitening/data/pages/<slug>.json (the source of the Gemma targets);
    Ancient Rome (no stored page in this checkout) from the multilingual study's raw page html
    (nlp_proj_ml/results/ml/data/raw/Ancient_Rome/en.json), split with the same sentence splitter."""
    p = WH / "data" / "pages" / f"{slug(concept)}.json"
    if p.exists():
        return read_json(p)["sentences"]
    cache = DATA / "pages" / f"{slug(concept)}.json"
    if cache.exists():
        return read_json(cache)["sentences"]
    from fetch_en_sentences import _ParagraphText, split_sentences
    raw = read_json(ML_ROOT / "results" / "ml" / "data" / "raw" / slug(concept) / "en.json")
    pt = _ParagraphText()
    pt.feed(raw["html"])
    sents = split_sentences(pt.paragraphs)
    write_json(cache, {"concept": concept, "title": raw.get("title"), "revid": raw.get("revid"),
                       "source": "nlp_proj_ml raw en.json", "sentences": sents})
    return sents


def page_target(model: str, concept: str) -> list[int]:
    """Sorted distinct token ids of the page (add_special_tokens=False) minus excluded ids."""
    cache = DATA / "targets" / model / f"{slug(concept)}.json"
    if cache.exists():
        return read_json(cache)
    tok, ex = tokenizer(model), excluded_ids(model)
    ids = set()
    for seq in tok(page_sentences(concept), add_special_tokens=False)["input_ids"]:
        ids.update(seq)
    out = sorted(int(i) for i in ids if i not in ex)
    write_json(cache, out, indent=None)
    return out


def ml_pooled_target(concept: str) -> list[int]:
    """Llama token types of the 300 concept sentences in all 8 languages of the multilingual study (its CS target)."""
    cache = DATA / "targets" / "llama" / f"{slug(concept)}__ml8.json"
    if cache.exists():
        return read_json(cache)
    full = read_json(ML_ROOT / "results" / "ml" / "data" / "full" / f"{slug(concept)}.json")
    tok, ex = tokenizer("llama"), excluded_ids("llama")
    ids = set()
    for L, sets in full.items():
        sents = [r["text"] if isinstance(r, dict) else r for r in sets["concept"]]
        for seq in tok(sents, add_special_tokens=False)["input_ids"]:
            ids.update(seq)
    out = sorted(int(i) for i in ids if i not in ex)
    write_json(cache, out, indent=None)
    return out


# ---------------------------------------------------------------- V*
def vstar(model: str, concept: str) -> dict:
    """V* exactly as experiments/varwidth/prep.py (ConceptDataset, neutral seed 42, 300 + 300 sentences, TL-style
    left-padded tokenization, specials removed), with `model`'s tokenizer. codes: 1 C-only, 0 N-only, 2 both."""
    cache = DATA / "vstar" / model / f"{slug(concept)}.json"
    if cache.exists():
        d = read_json(cache)
        d["codes"] = np.asarray(d["codes"])
        d["token_to_code"] = {int(k): v for k, v in d["token_to_code"].items()}
        return d
    from experiments.varwidth.prep import build_dataset, build_vstar
    tok = tokenizer(model)
    ds = build_dataset(concept)
    vs, t2c, special = build_vstar(concept, ds, tok)
    codes = [int(t2c.get(t, 0)) for t in vs]
    d = {"concept": concept, "model": model, "vstar_ids": vs, "codes": codes,
         "token_to_code": {int(k): int(v) for k, v in t2c.items()},
         "concept_sents": [s for s, lbl in ds.data if lbl == concept],
         "all_sents": [s for s, _ in ds.data]}
    write_json(cache, d, indent=None)
    d["codes"] = np.asarray(codes)
    d["token_to_code"] = {int(k): v for k, v in d["token_to_code"].items()}
    return d


# ---------------------------------------------------------------- background + CS
@lru_cache(maxsize=2)
def background(model: str) -> dict:
    """English Wikipedia token-type background (mu, lam, Q, n). Gemma: experiments/whitening/data/gemma (existing);
    Llama: results/paper_analysis/llama_cs/data/bg_wiki_llama_stats.pt (built by llama_cs.background on the same
    sentences)."""
    p = (WH / "data" / "gemma" / "bg_wiki_stats.pt") if model == "gemma" else DATA / "bg_wiki_llama_stats.pt"
    st = torch.load(p, map_location="cpu", weights_only=False)
    return {k: (v.double() if torch.is_tensor(v) else v) for k, v in st.items()}


@lru_cache(maxsize=2)
def whitener(model: str):
    from sharpen import BackgroundWhitener
    return BackgroundWhitener(background(model), EPS_REL)


def whiten_rows(model: str, ids) -> torch.Tensor:
    E = embeddings(model)
    return whitener(model).whiten(E.index_select(0, torch.as_tensor(list(ids), dtype=torch.long)).double())


def sharpener_eig(U_target: torch.Tensor):
    """Eigen-decomposition of the whitened target second moment M = U^T U / m (float64): (lam desc, P)."""
    U = U_target.double()
    M = U.T @ U / U.shape[0]
    lam, P = torch.linalg.eigh(M)
    return lam.flip(0).clamp_min(0), P.flip(1)


def cs_matrix(lam: torch.Tensor, P: torch.Tensor, eta: float, gamma: float):
    """(S_eta^{gamma/2}, S_eta^{-gamma/2}) from M's eigen-decomposition; g = (1 - eta) lam + eta."""
    g = (1 - eta) * lam + eta
    Sp = (P * g.pow(gamma / 2)) @ P.T
    Sm = (P * g.pow(-gamma / 2)) @ P.T
    return Sp, Sm


def lexicon(concept: str):
    from experiments.varwidth.j3_words import concept_lists
    return concept_lists("v1").get(concept)
