"""WP7 (circularity): shared constants, paths and the sharpening transform for arbitrary targets.

The CS matrix for a target T (a set of token types) is built exactly as experiments/varwidth/prep.py
whitening_transform builds the default one, only with T, eta and gamma as arguments:
    W from the wiki background (experiments/whitening/data/gemma/bg_wiki_stats.pt, eps_rel from defaults.json)
    U_T = (E_T - mu_B) W ;  S = (1 - eta) U_T^T U_T / |T| + eta I ;  X = (E_V* - mu_B) W S^(gamma/2)
and A = X^T (d x n) is factorized with experiments/varwidth/core.fit_varwidth.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/morg/NLP_2526b/oripetel/.cache/huggingface")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "paper_analysis" / "circularity"
TEXTS = OUT / "texts"            # fetched page sentences + target id lists
FITS = OUT / "fits"              # one .pt per (concept, condition, seed)
FIGS = OUT / "figures"
WH_DIR = ROOT / "experiments" / "whitening" / "data" / "gemma"
PREP_DIR = ROOT / "results" / "varwidth" / "prep"

PAPER_CONCEPTS = ["Culture of Greece", "Golf", "Republic of Ireland", "Ancient Rome", "Baseball", "Uranium",
                  "Harry Potter", "Cannabis", "Gambling", "Gun", "Pornography", "Valentine's Day", "World War II",
                  "Nazism", "Heroin", "Halloween", "Artificial intelligence", "COVID-19 pandemic"]
TUNING_TRIO = ["Harry Potter", "Ancient Rome", "Valentine's Day"]
SEEDS = [42, 43, 44]

# Wrong-concept control: a fixed derangement of the 18 concepts into clearly unrelated pairs (chosen by hand before
# any fit; Harry Potter -> Golf as in the brief; Halloween is not paired with Harry Potter or Ireland).
WRONG = {"Harry Potter": "Golf", "Golf": "Nazism", "Nazism": "Valentine's Day", "Valentine's Day": "Uranium",
         "Uranium": "Pornography", "Pornography": "Ancient Rome", "Ancient Rome": "COVID-19 pandemic",
         "COVID-19 pandemic": "Baseball", "Baseball": "Heroin", "Heroin": "Culture of Greece",
         "Culture of Greece": "Artificial intelligence", "Artificial intelligence": "Halloween", "Halloween": "Gun",
         "Gun": "Cannabis", "Cannabis": "World War II", "World War II": "Gambling", "Gambling": "Republic of Ireland",
         "Republic of Ireland": "Harry Potter"}
assert sorted(WRONG) == sorted(PAPER_CONCEPTS) and sorted(WRONG.values()) == sorted(PAPER_CONCEPTS)
assert all(k != v for k, v in WRONG.items())

TARGETS = ["page", "disjoint", "related", "leakfree", "wrong", "pagesub"]
DEFAULT_ETA, DEFAULT_GAMMA = 0.05, 1.0
ETA_GRID, GAMMA_GRID = [0.05, 0.2, 0.5], [0.5, 1.0, 2.0]


def slug(concept: str) -> str:
    return re.sub(r"\W+", "_", concept).strip("_")


def cond_name(target: str, eta: float = DEFAULT_ETA, gamma: float = DEFAULT_GAMMA, method: str = "fixed") -> str:
    """'raw' or '<target>_eta<eta>_g<gamma>', with '_gs' appended for the GS (rn) fits."""
    base = "raw" if target == "raw" else f"{target}_eta{eta:g}_g{gamma:g}"
    return base + ("_gs" if method == "rn" else "")


def fit_path(concept: str, cond: str, seed: int) -> Path:
    return FITS / slug(concept) / cond / f"seed{seed}.pt"


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj, indent=1):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(p)


def load_prep(concept: str, matrix: str = "plain") -> dict:
    return torch.load(PREP_DIR / f"{slug(concept)}_{matrix}_S300.pt", weights_only=False, map_location="cpu")


def whitening_module():
    """experiments/whitening/common.py (tokenizer, excluded_ids) without clobbering experiments.varwidth.common."""
    import importlib.util
    wdir = ROOT / "experiments" / "whitening"
    for p in (wdir, ROOT / "experiments" / "resample_sentences", ROOT / "experiments" / "multilingual_hp"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    spec = importlib.util.spec_from_file_location("whitening_common", wdir / "common.py")
    wc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wc)
    return wc


def sharpen_module():
    wdir = ROOT / "experiments" / "whitening"
    if str(wdir) not in sys.path:
        sys.path.append(str(wdir))
    import sharpen
    return sharpen


class TransformBuilder:
    """Holds the background whitener (shared by all targets) and the full embedding matrix (x sqrt(d), float32)."""

    def __init__(self, E: torch.Tensor | None = None):
        sh = sharpen_module()
        defaults = read_json(WH_DIR / "defaults.json")
        self.eps_rel = defaults["eps_rel"]
        stats = torch.load(WH_DIR / "bg_wiki_stats.pt", map_location="cpu")
        stats = {k: (v.double() if torch.is_tensor(v) else v) for k, v in stats.items()}
        self.bw = sh.BackgroundWhitener(stats, self.eps_rel)
        if E is None:
            from experiments.varwidth.prep import load_input_embeddings
            E = load_input_embeddings()
        self.E = E
        self._sh = sh

    def X(self, E_vstar: torch.Tensor, target_ids, eta: float, gamma: float) -> tuple[torch.Tensor, dict]:
        """A = X^T (d, n) float32 for the given target types; also returns small diagnostics."""
        U_v = self.bw.whiten(E_vstar)
        if target_ids is None:          # whitening only (not used in WP7, kept for completeness)
            return U_v.float().T.contiguous(), {}
        ids = torch.tensor(sorted(set(int(t) for t in target_ids)), dtype=torch.long)
        U_t = self.bw.whiten(self.E.index_select(0, ids).double())
        S = self._sh.Sharpener(U_t, eta, gamma)
        X = S.apply(U_v)
        g = S.g
        diag = {"n_target": int(len(ids)), "top_enrichment": [round(float(x), 3) for x in g[:5]],
                "n_enriched_gt1": int((g > 1).sum())}
        return X.float().T.contiguous(), diag
