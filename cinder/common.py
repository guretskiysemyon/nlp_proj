"""Paths and settings shared by the CINDER scripts.

CINDER features are written in EMBER's own file layout under ``mf_outputs/cinder/`` so that
EMBER's edit code (ember/erasure/embed_edit.py) applies them unchanged. ``use_cinder_features()``
points ember.erasure.features at that folder; nothing in ember/ is edited.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "google/gemma-2-2b-it"
MODEL_SAFE = "google_gemma-2-2b-it"
RANK, SEED = 100, 42                       # EMBER Part 1: rank 100, seed 42
RUN_REL = Path(MODEL_SAFE) / f"rank{RANK}" / f"seed{SEED}"

CONCEPTS = ["Cannabis", "Gun", "Nazism", "Republic of Ireland", "Culture of Greece", "Golf"]

# CINDER = Ori's ladder rung L123 (experiments/varwidth/layer_eval.py): L1 contrastive sharpening (whitened matrix,
# page target), L2 GS-norm ("rn", refit=False, floor 0.2% / cap 3%), L3 HJ = TJ or LO@5 on the top-20 features by rho.
CINDER_METHOD, CINDER_MATRIX = "rn", "whitened"

MF_ROOT = ROOT / "mf_outputs" / "cinder"                       # EMBER-format features (git-ignored)
EMBER_MF_ROOT = ROOT / "mf_outputs"                             # Part 1 EMBER features
OUT = ROOT / "judge_rerun" / "Cinder"                           # tracked results, as judge_rerun/Ember
FEATURES_OUT = OUT / "features"                                 # per concept: judge decisions, summary
GRID_OUT = OUT / "grid"                                         # per concept: grid_rescored.csv, records, completions
PICK_OUT = OUT / "pick"                                         # per concept: delta_metrics, best_delta
GRID_JSON = OUT / "delta_grid.json"                             # written by cinder.edit_size

EMBER_DIR = ROOT / "judge_rerun" / "Ember"
BASELINE_DIR = EMBER_DIR / "pass1_outputs" / RUN_REL / "_baseline"   # pristine model: same for both methods
EMBER_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]          # configs/ember_gemma.yaml

JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
JUDGE_ID = "qwen2.5-32b-instruct"
MIN_MMLU, MIN_ALPACA, TOP_K, ALPACA_N = 0.90, 0.9, 3, 50
ALPACA_GEN_N = 56   # 7 batches of 8: the first 50 are batched exactly as in Part 1's 150-item run


def safe(concept: str) -> str:
    from ember.utils import _safe_concept
    return _safe_concept(concept)


def use_cinder_features() -> None:
    """Make ember.erasure.features read mf_outputs/cinder/ instead of mf_outputs/."""
    from ember.erasure import features
    features.MF_OUTPUTS_ROOT = MF_ROOT


def potential_csv(concept: str) -> Path:
    """The TWJ-accepted CINDER features of a concept (what the edit uses)."""
    return (MF_ROOT / MODEL_SAFE / "interpretations" / f"rank{RANK}" / f"seed{SEED}" / safe(concept) / "embedding"
            / "potential_features.csv")


def load_grid() -> list[float]:
    if not GRID_JSON.exists():
        raise FileNotFoundError(f"{GRID_JSON} missing: run python -m cinder.edit_size first")
    return [float(d) for d in json.loads(GRID_JSON.read_text())["grid"]]


def write_json(path: Path, obj, indent=1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)
