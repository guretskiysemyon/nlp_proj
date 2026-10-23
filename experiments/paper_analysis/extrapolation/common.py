"""Shared paths, concept list and loaders for WP5 (extrapolation to tokens outside the 300 sampled sentences).

Layout
    experiments/paper_analysis/extrapolation/   code (this package)
    results/paper_analysis/extrapolation/       outputs (REPORT.md, figures, tables, per-concept token sets)

Dictionaries compared (saved fits, seeds 42-46):
    EMBER   L0  = fixed_refit-False_plain   raw embeddings, RS-1%, F_C = EMBER's two-stage judge (TJ)
    CS      L1  = fixed_refit-False         CS matrix, RS-1%,      F_C = TJ or lexicon@5 (HJ)
    CINDER  L12 = rn_refit-False            CS matrix, GS,         F_C = HJ   (the paper's CINDER selection)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/morg/NLP_2526b/oripetel/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from experiments.varwidth.common import ROOT, run_dir, slug  # noqa: E402

OUT = ROOT / "results" / "paper_analysis" / "extrapolation"
TOKDIR = OUT / "tokens"
PAGES = ROOT / "experiments" / "whitening" / "data" / "pages"
DUMP = ROOT / "experiments" / "resample_sentences" / "data" / "wiki_dump_pages.jsonl"
PARQUET = Path("/home/morg/NLP_2526b/oripetel/.cache/huggingface/hub/datasets--wikimedia--wikipedia/snapshots/"
               "b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en/train-00040-of-00041.parquet")
SEEDS = (42, 43, 44, 45, 46)
DICTS = {"EMBER": ("L0", "TJ"), "CS": ("L1", "HJ"), "CINDER": ("L12", "HJ")}
# benchmark concepts (fixed before any scoring): >= 80 unsampled page sentences and a non-empty F_C for both
# EMBER (L0, TJ) and CINDER (L12, HJ) in at least one of seeds 42-46 (tokens.log + layer caches)
CONCEPTS = ["Republic of Ireland", "Ancient Rome", "Baseball", "Harry Potter", "Cannabis", "Gun", "Pornography",
            "Valentine's Day", "Heroin", "Halloween", "Artificial intelligence", "COVID-19 pandemic", "Tuberculosis",
            "Buddhism", "Judaism", "Robot", "Surfing"]
COLORS = {"EMBER": "#7F7F7F", "CS": "#0072B2", "CINDER": "#D55E00", "CINDER-z": "#E69F00", "other": "#009E73"}


def all_concepts() -> list:
    from experiments.varwidth.judge_report import concepts
    return concepts()


def page_sentences(concept: str) -> tuple[list, dict]:
    """All sentences of the concept's page as split for the CS target (experiments/whitening/data/pages)."""
    p = PAGES / f"{slug(concept)}.json"
    if not p.exists():
        p = ROOT / "experiments" / "resample_sentences" / "data" / "wiki_pages" / f"{slug(concept)}.json"
    d = json.loads(p.read_text())
    return list(d["sentences"]), {"title": d.get("title"), "revid": d.get("revid"), "source": str(p)}


def load_fit(concept: str, cfg: str, seed: int) -> dict:
    import torch

    from experiments.varwidth.layer_eval import CONFIGS
    d = run_dir(concept, CONFIGS[cfg], seed)
    return torch.load(d / "fit.pt", map_location="cpu", weights_only=False)


def concept_features(concept: str, cfg: str, seed: int, judge: str) -> list:
    """F_C (feature indices) of a saved run under a judge (layer_eval.fc on the run's layer_cache.json)."""
    from experiments.varwidth.layer_eval import fc, load
    cache = load(concept, cfg, seed)
    if cache is None:
        return []
    return [int(c["feature"]) for c in fc(cache, judge)]


def write_json(path: Path, obj, indent=1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=indent, ensure_ascii=False,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)
