"""rho filter + 2-stage Gemini judge, identical to cells 3.2.b-3.2.e (and the 3.2.f table) of
low_ram_demands/ancient_rome_whitened_varnnz.ipynb, with the Gemini calls run concurrently.

    sel = select_features(Y, prep, judge_path)          # {"F_C", "candidates", "stats", "cand"}
    widest = describe_features(Y, prep, features, judge_path)

The judge cache (judge_path) has the notebook's format {"descriptions": {f: str}, "verdicts": {f: [is_member, conf]}}
and is saved after every call; use_cache=False ignores (and then overwrites) whatever is in it.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.varwidth.prep import MODEL_NAME, TokenizerOnly, get_tokenizer

TAU = 2.0
CONFIDENCE = 0.85
TOP_K_TOKENS = 20
GEMINI_MODEL = "models/gemini-3.5-flash-lite"
WORKERS = 8
LABEL_TAG = {1: "C", 0: "N", 2: "B"}


# ---------------------------------------------------------------- cache
def load_judge(path: Path | None, use_cache: bool = True) -> tuple[dict, dict]:
    if path is None or not use_cache or not Path(path).exists():
        return {}, {}
    j = json.loads(Path(path).read_text())
    return ({int(f): d for f, d in j.get("descriptions", {}).items()},
            {int(f): tuple(v) for f, v in j.get("verdicts", {}).items()})


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


class JudgeCache:
    def __init__(self, path: Path | None, use_cache: bool = True):
        self.path = Path(path) if path is not None else None
        self.descriptions, self.verdicts = load_judge(self.path, use_cache)
        self.lock = threading.Lock()

    def save(self):
        if self.path is not None:
            save_json(self.path, {"descriptions": self.descriptions, "verdicts": self.verdicts})

    def put_desc(self, f: int, d: str):
        with self.lock:
            self.descriptions[f] = d
            self.save()

    def put_verdict(self, f: int, v):
        with self.lock:
            self.verdicts[f] = tuple(v)
            self.save()


# ---------------------------------------------------------------- Gemini client with retries, one per thread
class RetryingGemini:
    """SimpleGeminiClient (no throttle, as the notebook) with exponential-backoff retries on 429 / 5xx / transient errors."""

    def __init__(self, model_name: str = GEMINI_MODEL, max_attempts: int = 8):
        self.model_name, self.max_attempts = model_name, max_attempts
        self._local = threading.local()

    def _inner(self):
        c = getattr(self._local, "c", None)
        if c is None:
            from ember.interpret_features import SimpleGeminiClient
            c = SimpleGeminiClient(model_name=self.model_name, max_retries=1)
            self._local.c = c
        return c

    def generate(self, prompt: str) -> str:
        for attempt in range(self.max_attempts):
            try:
                return self._inner().generate(prompt)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "PerDay" in msg or attempt == self.max_attempts - 1:
                    raise
                m = re.search(r"retry in ([\d.]+)s", msg)
                wait = float(m.group(1)) + 1 if m else min(60.0, 2.0 * 2 ** attempt)
                time.sleep(wait * (1 + 0.25 * random.random()))
        raise RuntimeError("unreachable")


_CLIENT = None


def get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = RetryingGemini()
    return _CLIENT


# ---------------------------------------------------------------- 3.2.b / 3.2.c
def feature_tokens(Y: torch.Tensor, prep: dict) -> pd.DataFrame:
    """collect_feature_rows_for_embeddings (3.2.c): per feature, nonzero tokens sorted by |Y| (max 200) + labels."""
    from ember.utils import _safe_model_name, collect_feature_rows_for_embeddings
    concept, vstar_ids = prep["concept"], prep["vstar_ids"]
    token_label_map = {tid: "both" if c == 2 else concept if c == 1 else "Neutral"
                       for tid, c in prep["token_to_code"].items()}
    rows = collect_feature_rows_for_embeddings(Y.T, vstar_ids, token_label_map, TokenizerOnly(get_tokenizer()),
                                               _safe_model_name(MODEL_NAME), concept, Y.shape[0])
    return pd.DataFrame(rows).set_index("feature")[["activating_tokens", "labels"]]


def rho_candidates(Y: torch.Tensor, prep: dict, tau: float = TAU):
    from ember.utils import compute_embedding_stats
    Y = Y.detach().float().cpu()
    stats = compute_embedding_stats(Y.T, prep["vstar_ids"], prep["token_to_code"], prep["concept"]) \
        .sort_values("ratio_abs", ascending=False)
    candidates = stats[stats["ratio_abs"] >= tau]
    feat_tokens = feature_tokens(Y, prep)
    cand = candidates[["feature", "ratio_abs"]].join(feat_tokens, on="feature").reset_index(drop=True)
    cand["judge_tokens"] = cand["activating_tokens"].apply(lambda toks: toks[:TOP_K_TOKENS])
    return stats, cand, feat_tokens


# ---------------------------------------------------------------- 3.2.d / 3.2.e
def _run_pool(fn, items, workers):
    items = list(items)
    if not items:
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        for _ in ex.map(fn, items):
            pass


def judge(concept: str, feat_to_tokens: dict, cache: JudgeCache, workers: int = WORKERS, classify: bool = True):
    from ember.interpret_features import classify_membership_with_desc, get_description
    client = get_client()

    def one(f):
        if f not in cache.descriptions:
            cache.put_desc(f, get_description(client, feat_to_tokens[f]))
        if classify and f not in cache.verdicts:
            cache.put_verdict(f, classify_membership_with_desc(client, concept, cache.descriptions[f]))

    _run_pool(one, sorted(feat_to_tokens), workers)


def select_features(Y: torch.Tensor, prep: dict, judge_path: Path | None, use_cache: bool = True,
                    workers: int = WORKERS, tau: float = TAU, confidence: float = CONFIDENCE) -> dict:
    t0 = time.time()
    stats, cand, feat_tokens = rho_candidates(Y, prep, tau)
    cache = JudgeCache(judge_path, use_cache)
    feats = cand["feature"].astype(int).tolist()
    judge(prep["concept"], {f: t for f, t in zip(feats, cand["judge_tokens"])}, cache, workers)
    cand["description"] = cand["feature"].map(lambda f: cache.descriptions[int(f)])
    cand["is_member"] = cand["feature"].map(lambda f: cache.verdicts[int(f)][0])
    cand["confidence"] = pd.to_numeric(cand["feature"].map(lambda f: cache.verdicts[int(f)][1]),
                                       errors="coerce").fillna(0.0)
    cand["selected"] = (cand["is_member"] == True) & (cand["confidence"] >= confidence)  # noqa: E712
    F_C = sorted(cand.loc[cand["selected"], "feature"].astype(int).tolist())
    candidates = [{"feature": int(r.feature), "ratio_abs": float(r.ratio_abs), "description": r.description,
                   "is_member": None if r.is_member is None else bool(r.is_member),
                   "confidence": float(r.confidence), "selected": bool(r.selected)} for r in cand.itertuples()]
    return {"F_C": F_C, "candidates": candidates, "stats": stats, "cand": cand, "feat_tokens": feat_tokens,
            "elapsed": time.time() - t0}


def describe_features(Y: torch.Tensor, prep: dict, features, judge_path: Path | None, use_cache: bool = True,
                      workers: int = WORKERS, feat_tokens: pd.DataFrame | None = None) -> dict:
    """Stage-1 descriptions only (first TOP_K_TOKENS tokens, as the judge sees them). Returns {feature: description}."""
    if feat_tokens is None:
        feat_tokens = feature_tokens(Y.detach().float().cpu(), prep)
    features = [int(f) for f in features]
    cache = JudgeCache(judge_path, use_cache)
    judge(prep["concept"], {f: feat_tokens.loc[f, "activating_tokens"][:TOP_K_TOKENS] for f in features}, cache,
          workers, classify=False)
    return {f: cache.descriptions[f] for f in features}


# ---------------------------------------------------------------- 3.2.f
def concept_token_counts(Y: torch.Tensor, prep: dict, F_C) -> dict:
    codes = np.asarray(prep["codes"])
    all_cols = sorted({j for f in F_C for j in torch.nonzero(Y[f] != 0).flatten().tolist()})
    return {"n_distinct_tokens": len(all_cols), "C": int(sum(codes[j] == 1 for j in all_cols)),
            "N": int(sum(codes[j] == 0 for j in all_cols)), "B": int(sum(codes[j] == 2 for j in all_cols)),
            "tokens_per_feature": {str(int(f)): int((Y[f] != 0).sum()) for f in F_C}}


def wide_table(Y: torch.Tensor, prep: dict, F_C) -> pd.DataFrame:
    """One column per F_C feature: every token with nonzero Y, sorted by |Y|: 'token [C/N/B] |y|' (cell 51)."""
    codes, tokens = np.asarray(prep["codes"]), prep["tokens"]
    columns = {}
    for feat in F_C:
        mags = Y[feat].abs()
        nz_idx = torch.nonzero(mags > 0).flatten()
        nz_idx = nz_idx[torch.argsort(mags[nz_idx], descending=True)].tolist()
        columns[f"feature {feat}"] = pd.Series([f"{tokens[j]} [{LABEL_TAG[int(codes[j])]}] {mags[j]:.2f}"
                                                for j in nz_idx])
    wide = pd.DataFrame(columns).fillna("")
    wide.index = range(1, len(wide) + 1)
    return wide


def wide_table_md(Y: torch.Tensor, prep: dict, F_C, title: str = "") -> str:
    cnt = concept_token_counts(Y, prep, F_C)
    head = (f"{title}\n\n" if title else "") + (f"{len(F_C)} concept features, {cnt['n_distinct_tokens']} distinct "
                                                f"tokens (C={cnt['C']}, N={cnt['N']}, B={cnt['B']})\n\n")
    if not F_C:
        return head + "(F_C is empty)\n"
    wide = wide_table(Y, prep, F_C)
    esc = lambda s: str(s).replace("|", "\\|")  # noqa: E731
    cols = list(wide.columns)
    lines = ["| # | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for i, row in wide.iterrows():
        lines.append(f"| {i} | " + " | ".join(esc(row[c]) for c in cols) + " |")
    return head + "\n".join(lines) + "\n"
