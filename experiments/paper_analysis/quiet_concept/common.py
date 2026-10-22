"""WP1 quiet_concept: shared loaders (prep matrices, fits, whitening pieces, rho, lexicon hits).

Matrices (all d x n, columns = tokens of V*, Gemma embeddings x sqrt(d)):
    raw  A_raw = E_V*^T                                   (EMBER, prep "plain")
    ctr  A_ctr = (E_V* - mu_B)^T                          (centring only; built here)
    rawp A_rawp = (I - mu^ mu^T) E_V*^T, mu^ = mu_B/||mu_B|| (raw with the mu_B direction projected out)
    wo   A_wo  = ((E_V* - mu_B) W)^T                      (whitening only, gamma = 0; built here from bg stats)
    cs   A_cs  = ((E_V* - mu_B) W S_eta^{1/2})^T          (CS, prep "whitened")
Fit tags: raw = fixed_refit-False_plain (L0, EMBER), cs = fixed_refit-False (L1), gs = rn_refit-False (L12),
wo = fixed_refit-False_g0 and ctr = fixed_refit-False_ctr (this WP, results/paper_analysis/quiet_concept/fits/).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from experiments.varwidth.common import RESULTS as VW_RESULTS
from experiments.varwidth.common import ROOT, run_dir, slug

OUT = ROOT / "results" / "paper_analysis" / "quiet_concept"
FIT_DIR = OUT / "fits"
WH_DIR = ROOT / "experiments" / "whitening" / "data" / "gemma"
SEEDS = list(range(42, 47))
WO_SEEDS = [42, 43, 44]
TAGS = {"raw": "fixed_refit-False_plain", "cs": "fixed_refit-False", "gs": "rn_refit-False",
        "gs_raw": "rn_refit-False_plain", "wo": "fixed_refit-False_g0", "ctr": "fixed_refit-False_ctr",
        "rawp": "fixed_refit-False_rawp"}
COLORS = {"raw": "#7F7F7F", "ctr": "#B0B0B0", "cs": "#0072B2", "gs": "#D55E00", "wo": "#009E73"}
LABELS = {"raw": "raw (EMBER)", "ctr": "centred", "wo": "whiten only", "cs": "CS", "gs": "CS+GS"}
OWN_CONDS = ("wo", "ctr", "rawp")                     # fits produced by this WP


def concepts(which: str = "all") -> list[str]:
    main = [e["concept"] for e in json.loads((ROOT / "data" / "concept_sentences.json").read_text())]
    extra = [e["concept"] for e in json.loads((ROOT / "data" / "concept_sentences_extra.json").read_text())]
    main = list(dict.fromkeys(main))
    extra = [c for c in dict.fromkeys(extra) if c not in main]
    return {"paper": main, "extra": extra, "all": main + extra}[which]


def load_prep(concept: str, matrix: str) -> dict:
    from experiments.varwidth.prep import prep_path
    return torch.load(prep_path(concept, matrix), map_location="cpu", weights_only=False)


@lru_cache(maxsize=1)
def whitener() -> dict:
    """mu_B, W, W_inv (float64) exactly as experiments/whitening/sharpen.BackgroundWhitener with defaults.json."""
    stats = torch.load(WH_DIR / "bg_wiki_stats.pt", map_location="cpu")
    eps_rel = json.loads((WH_DIR / "defaults.json").read_text())["eps_rel"]
    lam, Q = stats["lam"].double().clamp_min(0), stats["Q"].double()
    eps = eps_rel * lam.mean()
    W = (Q * (lam + eps).pow(-0.5)) @ Q.T
    W_inv = (Q * (lam + eps).pow(0.5)) @ Q.T
    return {"mu": stats["mu"].double(), "lam": lam, "Q": Q, "eps": float(eps), "W": W, "W_inv": W_inv,
            "n_bg": int(stats["n"])}


def matrices(concept: str) -> dict:
    """{'raw','wo','cs'} -> A (d, n) float64, plus codes, vstar_ids, tokens, T (d, d, row conv.), mu_B."""
    P = load_prep(concept, "plain")
    Wp = load_prep(concept, "whitened")
    assert P["vstar_ids"] == Wp["vstar_ids"]
    wh = whitener()
    E = P["A"].double().T                                     # (n, d)
    U = (E - wh["mu"]) @ wh["W"]
    mu_hat = wh["mu"] / wh["mu"].norm()
    out = {"raw": P["A"].double(), "ctr": (E - wh["mu"]).T.contiguous(), "wo": U.T.contiguous(),
           "rawp": (E - (E @ mu_hat)[:, None] * mu_hat[None, :]).T.contiguous(),
           "cs": Wp["A"].double(),
           "codes": np.asarray(P["codes"]), "vstar_ids": list(P["vstar_ids"]), "tokens": P["tokens"],
           "token_to_code": P["token_to_code"], "T": Wp["T_matrix"].double(), "mu_B": Wp["mu_B"].double(),
           "T_inv": Wp["T_inv"].double(), "eta": Wp["eta"], "n_target": Wp["n_target"]}
    return out


def fit_path(concept: str, cond: str, seed: int) -> Path:
    if cond in OWN_CONDS:
        return FIT_DIR / slug(concept) / TAGS[cond] / f"seed{seed}" / "fit.pt"
    return run_dir(concept, TAGS[cond], seed) / "fit.pt"


def load_fit(concept: str, cond: str, seed: int) -> dict | None:
    p = fit_path(concept, cond, seed)
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def load_summary(concept: str, cond: str, seed: int) -> dict | None:
    if cond in OWN_CONDS:
        return None
    p = run_dir(concept, TAGS[cond], seed) / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def rho(Y: torch.Tensor, codes: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """EMBER's ratio_abs per feature: mean |Y| over concept-only tokens / (mean |Y| over neutral-only + 1e-8)."""
    Ya = Y.abs().double().numpy()
    c, nn = codes == 1, codes == 0
    return Ya[:, c].mean(1) / (Ya[:, nn].mean(1) + eps)


def rho_rank(r: np.ndarray) -> np.ndarray:
    """1-based rank of every feature by descending rho (ties by index, as a stable sort)."""
    order = np.argsort(-r, kind="stable")
    rk = np.empty_like(order)
    rk[order] = np.arange(1, len(r) + 1)
    return rk


# ---------------------------------------------------------------- lexicon (judge-free concept feature)
@lru_cache(maxsize=None)
def lexicon(concept: str) -> tuple[str, ...]:
    from experiments.varwidth.j3_words import concept_lists
    return tuple(concept_lists("v1")[concept])


_COUNTS: dict = {}


def word_counts(concept: str) -> dict:
    """tid -> Counter of source words in S_C u S_N (ember.judge_words.token_word_counts)."""
    if concept not in _COUNTS:
        from ember.judge_words import token_word_counts
        from experiments.varwidth.prep import build_dataset, get_tokenizer
        _COUNTS.clear()
        _COUNTS[concept] = token_word_counts(get_tokenizer(), [s for s, _ in build_dataset(concept).data])
    return _COUNTS[concept]


def feature_words_all(Y: torch.Tensor, vstar_ids: list, counts: dict, n: int = 20) -> list[list[str]]:
    """First n distinct normalized source words of each row of Y in |y| order (j3_words.gen_features recipe)."""
    from experiments.varwidth import overlap_judge as oj
    from experiments.varwidth.judge_compare import row_order
    from experiments.varwidth.prep import get_tokenizer
    tok = get_tokenizer()
    tok_str = lambda t: tok.decode([t])  # noqa: E731
    return [oj.feature_words([vstar_ids[j] for j in row_order(Y, f)], counts, n=n, tok_str=tok_str)
            for f in range(Y.shape[0])]


def lexicon_hits(fwords: list[list[str]], lex) -> np.ndarray:
    s = set(lex)
    return np.array([sum(w in s for w in fw) for fw in fwords])


def best_concept_feature(hits: np.ndarray, rk: np.ndarray) -> int:
    """argmax lexicon hits; ties -> the better (smaller) rho rank."""
    cand = np.flatnonzero(hits == hits.max())
    return int(cand[np.argmin(rk[cand])])


def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
