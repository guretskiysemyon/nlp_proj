#!/usr/bin/env python3
"""Builds the multilingual background-whitened + target-sharpened notebooks (Llama, 8 languages, any concept):

    ember_three_steps_multilingual_whitened_llama.ipynb          from ember_three_steps_gambling_multilingual_llama.ipynb
    low_ram_demands/multilingual_whitened_llama.ipynb            from low_ram_demands/gambling_multilingual_llama.ipynb

The concept notebooks are concept-generic apart from text, so the Gambling one is the template. On top of it:
    - config: CONCEPT for any multilingual concept, BACKGROUND / TARGET / GAMMA (+ ETA, EPS_REL from
      experiments/whitening/data/llama/defaults_ml.json), EXT_PER_LANG / EXT_TOPUP, JUDGE_FORMAT;
      caches under runs/whiten_ml/<model>/<concept>/<neutral>_rep<r>/...
    - 3.1.c' whitening + sharpening with a multilingual background and target, rank / leak diagnostics, cPCA
    - 3.1.d on X, 3.1.d' reference fit on the raw matrix (reuses the standard notebook's smf.pt when it matches)
    - 3.2 judge input via ember.judge_words (JUDGE_FORMAT), 3.2.e' selection on the raw fit (F_C0)
    - analysis cells in X-space, 3.2.h extended set (300 per language) through the same transform
    - 3.2.w matching of the two fits + RUN_DIR/summary.json, 3.3 edit mapped back with T.unmap
Outputs are cleared. Re-running the script rebuilds both notebooks from the templates.

Usage: python scripts/build_multilingual_whitened_nb.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import nb_patch as P  # noqa: E402

# Templates are read at a fixed commit (before scripts/patch_notebooks_ml.py changed them), so the build is reproducible.
TEMPLATE_REV = "8d5603e"
TEMPLATES = {
    "ember_three_steps_gambling_multilingual_llama.ipynb": ROOT / "ember_three_steps_multilingual_whitened_llama.ipynb",
    "low_ram_demands/gambling_multilingual_llama.ipynb": ROOT / "low_ram_demands/multilingual_whitened_llama.ipynb",
}


def load_template(rel: str) -> dict:
    import json
    import subprocess
    out = subprocess.run(["git", "show", f"{TEMPLATE_REV}:{rel}"], cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)

# --------------------------------------------------------------------------------------------- cell texts
TITLE = r"""# EMBER end to end, multilingual (8 languages), background-whitened + target-sharpened, Llama-3.1-8B-Instruct

Multilingual counterpart of `ember_three_steps_whitened_llama.ipynb` (method: `Instructions.txt`, code:
`experiments/whitening/sharpen.py`), built on the multilingual concept notebooks (`ember_three_steps_<concept>_multilingual_llama.ipynb`).
$V^*$ is unchanged: up to 40 concept + 40 translated neutral sentences per language (en, es, fr, de, it, pt, hi, th). Step 3.1 factorizes
$X = (E_{V^*} - \mathbf 1\boldsymbol\mu_B^\top)\,W\,S_\eta^{\gamma/2}$ instead of the raw rows.

**How background and target are generalized to 8 languages** (`docs/Part 2/multilingual_whitening.md`):

- **Background** (`BACKGROUND = "wiki_ml"`): token types of random Wikipedia sentences, *the same number of sentences in each of the
  8 languages*, pooled and counted once (`experiments/whitening/background_ml.py`). That mirrors how $V^*$ is sampled. An English-only
  background (`"wiki_en"`) gives Devanagari / Thai / accented-Latin directions tiny variance, so $W\propto\Lambda^{-1/2}$ would blow them up
  and sharpening would amplify *language* rather than concept; the cPCA diagnostic in 3.1.c′ compares the two.
- **Target** (`TARGET = "ext"`): the token types of $S_C^{ext}$, the same 300-sentence-per-language concept corpus 3.2.h uses (concept page,
  short pages filled with Gemini translations), pooled over languages. The 40-per-language $T_C$ alone has ~2.4–3.0k types < $d$ = 4096
  (rank-deficient, every unused direction pinned at $\eta$); the pooled corpus has 7–13k. The target only enters through a $d\times d$
  matrix, so its size costs nothing in the factorization.
- **Leak control** (`TARGET = "ext_minus_vstar"`): the same corpus minus every $V^*$ token. The `ext` target contains every $T_C$ token,
  so concept tokens get stretched by construction; in English the leak-free target lost the gains (Llama Harry Potter AUC 0.87 → 0.61).
  Always read an `ext` run together with its `ext_minus_vstar` run. `"tc"` (the spec's literal $T_C$) is kept for completeness.

**Other switches.** `JUDGE_FORMAT`: what the Stage-1 judge sees per token (`"tokens"`: bare tokens as in the paper; `"words"`: token →
its source words grouped by language, alphabetically; `"top_word"`: token → most frequent word (lang), the standard multilingual format).
`EXT_PER_LANG` / `EXT_TOPUP`: the 3.2.h extended set (300 sentences per language and side; `EXT_TOPUP = None` keeps only native page
sentences, `EXT_PER_LANG = None` the original whole-page sets).

**Runs.** Caches in `RUN_DIR = runs/whiten_ml/llama/<concept>/<neutral>_rep<r>/<BACKGROUND>_<TARGET>_<ext>_g<GAMMA>/` (`smf.pt`,
`judge_<format>.json`, `mc.json`, `gen.json`, `summary.json`); the raw reference fit and its judge in `ORIG_DIR` (`.../original/`), shared by
all configurations. `scripts/run_overnight.py --param CONCEPT=... --stop-before` with the Step-3.3 marker runs one configuration headless (`experiments/whitening/queue_ml.json`).

**Requirements:** `HF_TOKEN` with access to `meta-llama/Llama-3.1-8B-Instruct`, a Gemini API key, the multilingual background
(`experiments/whitening/data/llama/bg_wiki_ml_stats.pt`) and the extended sets (`experiments/multilingual_hp/extended_sets.py --kind capped`).

| Step | What happens | Code |
|---|---|---|
| 3.1.c′ | $T$ = background whitening $W$ + target sharpening $S_\eta^{\gamma/2}$; rank / leak / cPCA diagnostics | `sharpen.py` |
| 3.1.d / d′ | Sparse MF on $X$ / on the raw $A_0$ | `SparseMatrixFactorization` |
| 3.2 | $\rho\ge\tau$ + two-stage judge on both fits → $F_C$, $F_{C0}$ | `compute_embedding_stats`, `ember/judge_words.py` |
| 3.2.h | extended set (300 / language) through the same $T$, $Z$ fixed | `extended_sets.py` |
| 3.3 | $\mathbf c_t = T^{-1}\tilde{\mathbf c}_t$, edit the raw embeddings | `T.unmap` |
"""

CONFIG = r'''import gc
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()  # HF_TOKEN / GEMINI_API_KEY from .env, if present
pd.set_option("display.max_colwidth", 200)

# ---- what to erase ------------------------------------------------------
CONCEPT = "Harry Potter"        # Harry Potter | Gambling | Gun | Halloween | Pornography | Valentine's Day
sys.path.insert(0, "experiments/multilingual_hp")
from splitting import CONCEPT_TITLES, LANGS, concept_paths, concept_slug, neutral_paths
NEUTRAL_SOURCE = "translated"   # the translated neutral sets exist for every concept
NEUTRAL_REP = 0                 # 0, 1 or 2: shared neutral replicate
SMOKE = False                   # True: quick end-to-end pass (short fit, 3 judged features, tiny evals)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"     # gated: needs HF_TOKEN with access
SEED = 42

# ---- Step 3.1: Sparse MF (App. A.1) -------------------------------------
RANK = 200 if "llama" in MODEL_NAME.lower() else 100   # k
SPARSITY = 0.01          # s
RIDGE = 1e-4             # lambda
MAX_ITER = 20_000        # T
PATIENCE = 50            # as in the standard multilingual notebooks, so their fit is the reference 3.1.d'

# ---- Step 3.2: feature selection (App. A.3) -----------------------------
TAU = 2.0
CONFIDENCE = 0.85
TOP_K_TOKENS = 20
GEMINI_MODEL = "models/gemini-3.5-flash-lite"   # repo default; the paper used Gemini-2.5-Flash-Lite
JUDGE_FORMAT = "words"   # Stage-1 judge input: "tokens" (bare tokens, paper) | "words" (token → its source words by language,
                         # alphabetical) | "top_word" (token → most frequent word (lang), the standard multilingual notebooks)

# ---- 3.2.h extended set (fixed Z) ----------------------------------------
EXT_PER_LANG = 300       # sentences per language and side in S_C^ext / S_N^ext; None = whole page (the original sets)
EXT_TOPUP = "translated" # "translated": pages shorter than EXT_PER_LANG are filled with Gemini translations; None: native only

# ---- Step 3.3: erasure (App. C.3) ---------------------------------------
DELTA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
DELTA = 10.0             # Gemma value, not tuned for Llama; on Llama efficacy saturates around 50-100 (Fig. 11), see 3.4

# ---- background whitening + target sharpening (Instructions.txt, experiments/whitening/sharpen.py) ----
sys.path.insert(0, "experiments/whitening")
from sharpen import BackgroundWhitener, Sharpener, Transform, principal_angle_sines

BACKGROUND = "wiki_ml"   # "wiki_ml" (8-language Wikipedia types) | "wiki_en" (English Wikipedia types) | "full" (whole vocabulary)
TARGET = "ext"           # "ext" (types of S_C^ext, 8 languages) | "ext_minus_vstar" (minus V*: leak-free control) | "tc" (T_C)
GAMMA = 1.0              # sharpening strength; 0 = background whitening only
MODEL_SLUG = "llama"     # selects experiments/whitening/data/<MODEL_SLUG>
RUN_BG_COMPARE = True    # 3.1.c': also show the cPCA directions under the English-only background

# ---- overrides from scripts/run_overnight.py --param NAME=VALUE (set in an injected first cell) ----
globals().update(globals().get("_OVERRIDES", {}))
assert CONCEPT in CONCEPT_TITLES, f"{CONCEPT!r} is not one of {list(CONCEPT_TITLES)}"
assert NEUTRAL_SOURCE == "translated" and NEUTRAL_REP in (0, 1, 2), (NEUTRAL_SOURCE, NEUTRAL_REP)
assert BACKGROUND in ("wiki_ml", "wiki_en", "full"), BACKGROUND
assert TARGET in ("ext", "ext_minus_vstar", "tc"), TARGET
assert JUDGE_FORMAT in ("tokens", "words", "top_word"), JUDGE_FORMAT
assert EXT_TOPUP in ("translated", None) and (EXT_PER_LANG is None or EXT_PER_LANG > 0), (EXT_PER_LANG, EXT_TOPUP)
assert GAMMA >= 0, GAMMA
TAG = concept_slug(CONCEPT).upper()   # short concept label in printouts

WH_DIR = Path("experiments/whitening/data") / MODEL_SLUG
for _name in ("defaults_ml.json", "defaults.json"):   # defaults_ml: eps for wiki_ml (eps_select.py); else the English defaults
    if (WH_DIR / _name).exists():
        defaults = json.loads((WH_DIR / _name).read_text())
        print(f"whitening defaults from {WH_DIR / _name}: eps_rel={defaults['eps_rel']} eta={defaults['eta']}")
        break
else:
    defaults = {"eps_rel": 1e-4, "eta": 0.2}
    print(f"no defaults in {WH_DIR}: built-in {defaults}")
ETA = globals().get("_OVERRIDES", {}).get("ETA", defaults["eta"])              # target shrinkage
EPS_REL = globals().get("_OVERRIDES", {}).get("EPS_REL", defaults["eps_rel"])  # eps = EPS_REL * mean(background eigenvalues)

if SMOKE:
    MAX_ITER, PATIENCE = 50, 5

# ---- run directories: resume caches ----
_smoke = "_smoke" if SMOKE else ""
EXT_TAG = "whole" if EXT_PER_LANG is None else f"n{EXT_PER_LANG}" + ("_tr" if EXT_TOPUP == "translated" else "")
BASE_DIR = Path("runs/whiten_ml") / MODEL_SLUG / concept_slug(CONCEPT) / f"{NEUTRAL_SOURCE}_rep{NEUTRAL_REP}"
RUN_DIR = BASE_DIR / f"{BACKGROUND}_{TARGET}_{EXT_TAG}_g{GAMMA:g}{_smoke}"
ORIG_DIR = BASE_DIR / f"original{_smoke}"   # raw reference fit + its judge, shared by every configuration
RUN_DIR.mkdir(parents=True, exist_ok=True)
ORIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")   # write then rename, so a crash never leaves a half file
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


def invalidate(directory: Path, names) -> None:
    """Move stale cache files aside (<name>.stale) so they are recomputed."""
    for name in names:
        p = directory / name
        if p.exists():
            p.replace(p.with_name(p.name + ".stale"))
            print(f"  moved stale {p} -> {p.name}.stale")


assert os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"), "Llama-3.1 is gated: set HF_TOKEN (e.g. in .env)"
print(f"concept={CONCEPT!r} | model={MODEL_NAME} | k={RANK} | languages={list(LANGS)} | neutral={NEUTRAL_SOURCE} rep {NEUTRAL_REP}")
print(f"background={BACKGROUND} | target={TARGET} | gamma={GAMMA:g} | eta={ETA:g} | eps_rel={EPS_REL:g} | ext={EXT_TAG} | "
      f"judge={JUDGE_FORMAT} | smoke={SMOKE}\nrun dir={RUN_DIR} | original-fit dir={ORIG_DIR}")
'''

WHITEN_MD = r"""### 3.1.c′ Background whitening and target sharpening (multilingual)

Instructions steps 1–4 in row form (`sharpen.py`): $U = (E_{V^*} - \mathbf 1\boldsymbol\mu_B^\top)W$, $S_\eta = (1-\eta)\tfrac1m U_C^\top U_C + \eta I$,
$X = U S_\eta^{\gamma/2}$, combined map `T` (`T.forward`, `T.unmap`, `T.matrix`).

1. **Background** from `WH_DIR/bg_<name>_stats.pt`: `wiki_ml` = 8-language Wikipedia types (same number of sentences per language),
   `wiki_en` = English Wikipedia types, `full` = whole vocabulary.
2. **Target** rows (`TARGET`), whitened with the same $W$:
   `ext` = types of the extended concept corpus $S_C^{ext}$ (up to `EXT_PER_LANG` sentences per language, pooled);
   `ext_minus_vstar` = the same minus every $V^*$ token (leak-free); `tc` = $T_C$, the concept-only types of $V^*$.
3. **Rank / leak table** for all three target options: $m$ types, $m/d$, rank of the target second moment, how many of the $d$
   directions sit exactly at the $\eta$ floor (rank deficiency), the share of $T_C$ inside the target (the leak), and the target's
   language mix. The configured target is then used for $S_\eta$.
4. **cPCA**: the top 5 enrichment directions, with the whitened background types projected on each (top 15; `*` = in the target;
   language tag by script). With `RUN_BG_COMPARE`, the same under the English-only background: language/script clusters there
   are the confound a multilingual background removes.

From here on `A` is $X^\top$; `E_vstar_X` = transformed rows (kept after 3.3.a), `A0` = the raw data matrix."""

WHITEN_CODE = r'''# Instructions steps 1-4 for the multilingual V*: background W, multilingual target S_eta, X = (E - mu_B) W S_eta^(gamma/2).
from collections import Counter

from extended_sets import build_concept_n, build_neutral_n, ext_paths

E_vstar_orig = E_vstar                                    # raw rows, for the reference fit 3.1.d'
A0 = E_vstar.T.contiguous().to(DEVICE)                    # (d, |V*|) raw
BG_FILES = {"wiki_ml": "bg_wiki_ml_stats.pt", "wiki_en": "bg_wiki_stats.pt", "full": "bg_full_stats.pt"}
BG_TYPES = {"wiki_ml": "bg_wiki_ml_types.json", "wiki_en": "bg_wiki_types.json", "full": "bg_full_types.json"}


def load_whitener(name: str) -> BackgroundWhitener:
    path = WH_DIR / BG_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: build it with experiments/whitening/background{'_ml' if name == 'wiki_ml' else ''}.py")
    st = torch.load(path, map_location="cpu")
    assert st["Q"].shape == (D_MODEL, D_MODEL), f"{path}: Q is {tuple(st['Q'].shape)}, expected d = {D_MODEL}"
    return BackgroundWhitener({k: (v.double() if torch.is_tensor(v) else v) for k, v in st.items()}, EPS_REL), st


bw, stats_bg = load_whitener(BACKGROUND)
lam_desc = stats_bg["lam"].double().clamp_min(0).sort(descending=True).values
shrink = lam_desc / (lam_desc + bw.eps)
print(f"background {BACKGROUND}: n = {stats_bg.get('n', '?')} types, d = {D_MODEL}, eps = {EPS_REL:g} x mean(lam) = {bw.eps:.3e}; "
      f"max |lam/(lam+eps) - 1| over the top 95% of directions = {(shrink[:math.ceil(0.95 * len(shrink))] - 1).abs().max():.2e}")

# ---- the extended concept corpus (the target and 3.2.h use the same sentences) ----
if EXT_PER_LANG is not None:   # cached under the concept's extended dir; no network calls here
    build_concept_n(CONCEPT, EXT_PER_LANG, EXT_TOPUP)
    build_neutral_n(CONCEPT, NEUTRAL_REP, EXT_PER_LANG, EXT_TOPUP)
EXT_FILES = ext_paths(CONCEPT, NEUTRAL_REP, EXT_PER_LANG, EXT_TOPUP)
target_corpus = json.loads(EXT_FILES["concept"].read_text())
target_tok_langs = {}
for r in target_corpus:
    for t in tokenizer(r["sentence"], add_special_tokens=False)["input_ids"]:
        target_tok_langs.setdefault(int(t), set()).add(r["lang"])
_codes31 = build_token_label_codes(CONCEPT, dataset, tokenizer)
tc_ids = {t for t in vstar_ids if _codes31.get(t, 0) == 1}
ext_types = set(target_tok_langs) - set(special)
TARGET_OPTIONS = {"ext": sorted(ext_types), "ext_minus_vstar": sorted(ext_types - set(vstar_ids)), "tc": sorted(tc_ids)}
print(f"S_C^ext ({EXT_FILES['concept'].name}): {len(target_corpus)} sentences, "
      f"{sum(r.get('source') == 'translated' for r in target_corpus)} translated top-up")


def whitened_rows(ids) -> torch.Tensor:
    return bw.whiten(E.index_select(0, torch.tensor(list(ids), device=E.device)).detach().double().cpu())


# ---- rank / leak table for every target option ----
rank_rows = []
for name, ids in TARGET_OPTIONS.items():
    if not ids:
        rank_rows.append({"target": name, "m": 0})
        continue
    U_t = whitened_rows(ids)
    ev = torch.linalg.eigvalsh(U_t.T @ U_t / U_t.shape[0])                       # eigenvalues of the target second moment
    rank_s = int((ev > ev.max() * 1e-10).sum())
    by_lang = Counter(l for t in ids for l in target_tok_langs.get(t, ()))
    tot = sum(by_lang.values()) or 1
    rank_rows.append({"target": name, "m": len(ids), "m / d": len(ids) / D_MODEL, "rank(S)": rank_s,
                      "dirs at eta floor": D_MODEL - rank_s, "T_C inside": f"{len(set(ids) & tc_ids)}/{len(tc_ids)}",
                      **{f"{l} share": by_lang.get(l, 0) / tot for l in LANGS}})
    del U_t
rank_df = pd.DataFrame(rank_rows).set_index("target")
print("\ntarget options (rank(S) < d means those directions are pinned at g = eta; 'T_C inside' > 0 is the label leak):")
display(rank_df.style.format({c: "{:.1%}" for c in rank_df.columns if c.endswith("share")} | {"m / d": "{:.2f}"}))

# ---- the configured target, sharpening, transform ----
target_ids = TARGET_OPTIONS[TARGET]
assert target_ids, f"target {TARGET!r} is empty"
U_target = whitened_rows(target_ids)
sh = Sharpener(U_target, ETA, GAMMA) if GAMMA > 0 else None
sh_diag = sh if sh is not None else Sharpener(U_target, ETA, 1.0)   # gamma = 0: diagnostics only, never applied
T = Transform(bw, sh)


def transform(rows: torch.Tensor) -> torch.Tensor:
    """Rows (n, d) in the original space -> X-space: (rows - mu_B) W S_eta^(gamma/2), float32."""
    return T.forward(rows.detach().cpu()).float()


E_X = transform(E_vstar)                                  # (|V*|, d)
E_vstar_X = E_X                                           # survives 3.3.a; the analysis cells use it
A = E_X.T.contiguous().to(DEVICE)                         # (d, |V*|): 3.1.d fits on this
m_tilde = U_target.mean(0)
m_X = m_tilde if sh is None else m_tilde @ sh.Sp          # target mean in X-space
I_d = torch.eye(D_MODEL, dtype=torch.float64)
print(f"\ntarget {TARGET}: m = {len(target_ids)}  inverse check max |M M^-1 - I| = "
      f"{(T.matrix @ (bw.W_inv if sh is None else sh.Sm @ bw.W_inv) - I_d).abs().max():.2e}")
del I_d
print(f"enrichment ratios g (eta = {ETA:g}): top 10 = {', '.join(f'{v:.2f}' for v in sh_diag.g[:10].tolist())}; "
      f"g > 1: {int((sh_diag.g > 1).sum())} / {D_MODEL}; at the eta floor: {int((sh_diag.g <= ETA * (1 + 1e-6)).sum())}")

# ---- cPCA: background types on the top enrichment directions, with a script tag ----
SCRIPT_TAGS = [("hi", re.compile(r"[\u0900-\u097F]")), ("th", re.compile(r"[\u0E00-\u0E7F]")),
               ("lat", re.compile(r"[A-Za-z\u00C0-\u024F]"))]


def script_tag(tid: int) -> str:
    s = tokenizer.decode([tid])
    return next((tag for tag, pat in SCRIPT_TAGS if pat.search(s)), "other")


def cpca_table(name: str, n_dir: int = 5, n_tok: int = 15) -> pd.DataFrame:
    """Top background types of `name` on the top n_dir cPCA directions, the target whitened with that background."""
    bw_n = bw if name == BACKGROUND else load_whitener(name)[0]
    U_t = bw_n.whiten(E.index_select(0, torch.tensor(target_ids, device=E.device)).detach().double().cpu())
    sh_n = sh_diag if name == BACKGROUND else Sharpener(U_t, ETA, 1.0)
    P5 = sh_n.P[:, :n_dir]
    signs = torch.sign(U_t.mean(0) @ P5)
    signs[signs == 0] = 1.0
    M5 = bw_n.W @ (P5 * signs)
    ids = [int(t) for t in json.loads((WH_DIR / BG_TYPES[name]).read_text()) if 0 <= int(t) < VOCAB_SIZE]
    scores = torch.cat([(E.index_select(0, torch.tensor(ids[i:i + 20_000], device=E.device)).detach().double().cpu()
                         - bw_n.mu) @ M5 for i in range(0, len(ids), 20_000)])
    tset = set(target_ids)
    out = {}
    for j in range(n_dir):
        top = torch.argsort(scores[:, j], descending=True)[:n_tok].tolist()
        out[f"p{j + 1} (g={sh_n.g[j]:.2f})"] = [f"{tokenizer.decode([ids[i]])!r} [{script_tag(ids[i])}]{' *' if ids[i] in tset else ''}"
                                               for i in top]
    df = pd.DataFrame(out)
    df.index += 1
    return df


for name in [BACKGROUND] + (["wiki_en"] if RUN_BG_COMPARE and BACKGROUND != "wiki_en" else []):
    if (WH_DIR / BG_TYPES[name]).exists():
        print(f"\ncPCA under background {name} (target {TARGET}; [hi]/[th]/[lat] = script, * = in the target):")
        display(cpca_table(name))
print(f"\nA (transformed): {tuple(A.shape)}   A0 (raw): {tuple(A0.shape)}")
'''

FIT_CODE = r'''# Resume: RUN_DIR/smf.pt is loaded when its configuration matches; otherwise it (and the caches built on it) are moved aside.
SMF_PATH = RUN_DIR / "smf.pt"
fit_meta = {"model": MODEL_NAME, "vstar_ids": vstar_ids, "background": BACKGROUND, "target": TARGET, "ext": EXT_TAG,
            "gamma": GAMMA, "eta": ETA, "eps_rel": EPS_REL, "rank": RANK, "sparsity": SPARSITY, "ridge": RIDGE,
            "max_iter": MAX_ITER, "patience": PATIENCE, "seed": SEED, "n_target": len(target_ids)}
ck = torch.load(SMF_PATH, weights_only=False) if SMF_PATH.exists() else None
if ck is not None and ck.get("meta") != fit_meta:
    print(f"{SMF_PATH} was made with another configuration: re-fitting")
    invalidate(RUN_DIR, ["smf.pt", "mc.json", "gen.json", "summary.json"] + [p.name for p in RUN_DIR.glob("judge*.json")])
    ck = None
if ck is not None:
    Z, Y, ridge_used, t_fit = ck["Z"], ck["Y"], ck["ridge_used"], {"elapsed": ck["elapsed"]}
    print(f"loaded {SMF_PATH}")
else:
    set_seed(SEED)
    smf = SparseMatrixFactorization(RANK, fitting_device=DEVICE, g_sparsity=SPARSITY)
    with Timer() as t_fit:
        ridge_used = fit_with_ridge(smf, A, MAX_ITER, patience=PATIENCE, base_reg=RIDGE)
    Z = smf.F_.detach().float().cpu()      # (d, k)      X-space features z~_i (columns)
    Y = smf.G_.detach().float().cpu().T    # (k, |V*|)   sparse coefficients
    torch.save({"Z": Z, "Y": Y, "ridge_used": ridge_used, "elapsed": t_fit["elapsed"], "meta": fit_meta}, SMF_PATH)
print(f"\nfitted in {t_fit['elapsed']:.1f}s (lambda={ridge_used:g})  Z: {tuple(Z.shape)}  Y: {tuple(Y.shape)}")
'''

FIT0_MD = r"""### 3.1.d′ Reference fit on the original (untransformed) matrix

Same class, settings and seed on the raw $A_0 = E_{V^*}^\top$: this *is* the standard multilingual notebook's fit, so its
`runs/<concept>/translated_rep<r>/smf.pt` is reused when it exists and was fitted on the same $V^*$. `Z0`, `Y0` are only compared
against (3.1.e/f, 3.2.e′, 3.2.w); Step 3.3 does not use them. Cached in `ORIG_DIR/smf0.pt`, shared by every configuration."""

FIT0_CODE = r'''SMF0_PATH = ORIG_DIR / "smf0.pt"
fit0_meta = {"model": MODEL_NAME, "vstar_ids": vstar_ids, "rank": RANK, "sparsity": SPARSITY, "ridge": RIDGE,
             "max_iter": MAX_ITER, "patience": PATIENCE, "seed": SEED}
ck0 = torch.load(SMF0_PATH, weights_only=False) if SMF0_PATH.exists() else None
if ck0 is not None and ck0.get("meta") != fit0_meta:
    print(f"{SMF0_PATH} was made with another configuration: re-fitting")
    invalidate(ORIG_DIR, ["smf0.pt"] + [p.name for p in ORIG_DIR.glob("judge0*.json")])
    ck0 = None
STD_SMF = Path("runs") / concept_slug(CONCEPT) / f"{NEUTRAL_SOURCE}_rep{NEUTRAL_REP}" / "smf.pt"   # standard notebook's fit
if ck0 is None and not SMOKE and STD_SMF.exists():
    _std = torch.load(STD_SMF, weights_only=False)
    if _std.get("vstar_ids") == vstar_ids:
        ck0 = {"Z": _std["Z"], "Y": _std["Y"], "ridge_used": _std["ridge_used"], "elapsed": _std["elapsed"], "meta": fit0_meta}
        torch.save(ck0, SMF0_PATH)
        print(f"reused the standard multilingual fit {STD_SMF}")
if ck0 is not None:
    Z0, Y0, ridge_used0, t_fit0 = ck0["Z"], ck0["Y"], ck0["ridge_used"], {"elapsed": ck0["elapsed"]}
    print(f"loaded {SMF0_PATH}")
else:
    set_seed(SEED)
    smf0 = SparseMatrixFactorization(RANK, fitting_device=DEVICE, g_sparsity=SPARSITY)
    with Timer() as t_fit0:
        ridge_used0 = fit_with_ridge(smf0, A0, MAX_ITER, patience=PATIENCE, base_reg=RIDGE)
    Z0 = smf0.F_.detach().float().cpu()      # (d, k)      raw-space features
    Y0 = smf0.G_.detach().float().cpu().T    # (k, |V*|)   sparse coefficients
    torch.save({"Z": Z0, "Y": Y0, "ridge_used": ridge_used0, "elapsed": t_fit0["elapsed"], "meta": fit0_meta}, SMF0_PATH)
print(f"\nfitted in {t_fit0['elapsed']:.1f}s (lambda={ridge_used0:g})  Z0: {tuple(Z0.shape)}  Y0: {tuple(Y0.shape)}")
'''

ERR_CODE = r'''A_cpu, A0_cpu = A.cpu(), A0.cpu()
rel_err = (torch.norm(A_cpu - Z @ Y) / torch.norm(A_cpu)).item()        # transformed fit, X-space
rel_err0 = (torch.norm(A0_cpu - Z0 @ Y0) / torch.norm(A0_cpu)).item()   # original fit, raw space
nz = Y.abs() > 0
nz0 = Y0.abs() > 0
for name, err, mask in [("transformed", rel_err, nz), ("original", rel_err0, nz0)]:
    tokens_per_feature, features_per_token = mask.sum(1), mask.sum(0)
    print(f"[{name}] relative reconstruction error ||A - ZY||_F / ||A||_F = {err:.4f}")
    print(f"[{name}] tokens per feature : min={tokens_per_feature.min().item()}  max={tokens_per_feature.max().item()}")
    print(f"[{name}] features per token : mean={features_per_token.float().mean():.2f}  "
          f"max={features_per_token.max().item()}  uncovered tokens={int((features_per_token == 0).sum())}\n")

# Instructions step 6 (optional check): |cos| of each X-space factor with the target mean m_X.
cos_m = F.cosine_similarity(Z.double(), m_X.unsqueeze(1), dim=0).abs()       # (k,)
cos_m_order = torch.argsort(cos_m, descending=True).tolist()
print(f"|cos(z~_i, m_X)|: median={cos_m.median():.3f}  max={cos_m.max():.3f}  "
      f"(random-direction scale ~ {math.sqrt(2 / (math.pi * D_MODEL)):.3f})")
'''

SVD_CODE = r'''for name, A_mat, err, mask, Z_mat in [("transformed", A_cpu, rel_err, nz, Z), ("original", A0_cpu, rel_err0, nz0, Z0)]:
    U, S, Vh = torch.linalg.svd(A_mat, full_matrices=False)
    A_svd = U[:, :RANK] @ torch.diag(S[:RANK]) @ Vh[:RANK]
    svd_rel_err = (torch.norm(A_mat - A_svd) / torch.norm(A_mat)).item()
    print(f"[{name}] rank-{RANK} truncated SVD : relative error = {svd_rel_err:.4f}")
    print(f"[{name}] rank-{RANK} Sparse MF     : relative error = {err:.4f}")
    tail = torch.sqrt(torch.flip(torch.cumsum(torch.flip(S ** 2, [0]), 0), [0]) / (S ** 2).sum())
    matching_rank = int((tail <= err).nonzero()[0]) if (tail <= err).any() else len(S)
    print(f"[{name}] SVD reaches the Sparse MF error ({err:.4f}) at rank {matching_rank}\n")
del U, S, Vh, A_svd
'''

WORDS_MD = r"""### 3.2.a′ Token → source words and language

Hindi and Thai Llama tokens are often byte pieces or sub-syllables, which mean nothing on their own. `ember.judge_words.token_word_counts`
records, for every token, every (language, word) it was cut from in $S_C\cup S_N$ (tokenizer offsets; whitespace words, pythainlp words
for Thai). `tok_word[t] = (word, lang)` is the most frequent pair, `tok_str[t]` a readable form of the token. What the judge sees is set
by `JUDGE_FORMAT` (3.2.c)."""

WORDS_CODE = r'''from ember.judge_words import PUNCT, desc_sys_prompt, judge_items, token_word_counts, word_spans

word_counts = token_word_counts(tokenizer, [s for s, _ in dataset], [sent_lang[s] for s, _ in dataset])  # tid -> Counter{(lang, word)}
tok_word = {t: c.most_common(1)[0][0][::-1] for t, c in word_counts.items()}                            # tid -> (word, lang)


def readable(tid: int) -> str:
    s = tokenizer.decode([tid]).strip()
    return s if s.strip("�") else tokenizer.convert_ids_to_tokens(tid)


tok_str = {t: readable(t) for t in vstar_ids}
missing = [t for t in vstar_ids if t not in tok_word]
print(f"{len(tok_word)} tokens mapped to source words; {len(missing)} V* tokens without one")
pd.DataFrame([
    {"lang": lang, "token": tok_str[t], "judge sees (words)": judge_items([t], tok_str, word_counts, "words", True)[0]}
    for lang in LANGS
    for t in [t for t in vstar_ids if t in tok_word and tok_word[t][1] == lang][:4]
])
'''

JUDGE_TOKENS_FN = r'''def judge_tokens_for(Ymat: torch.Tensor, f: int, n: int = TOP_K_TOKENS) -> list:
    """Top-n support tokens of feature f of Ymat by |Y|, formatted for the judge by JUDGE_FORMAT (ember/judge_words.py)."""
    mags = Ymat[f].abs()
    top = torch.argsort(mags, descending=True)[:n]
    top = top[mags[top] > 0].tolist()
    return judge_items([vstar_ids[j] for j in top], tok_str, word_counts, JUDGE_FORMAT, multilingual=True)


def ml_judge_tokens(f: int, n: int = TOP_K_TOKENS) -> list:
    return judge_tokens_for(Y, f, n)


DESC_SYS = desc_sys_prompt(JUDGE_FORMAT, multilingual=True)   # Stage-1 system prompt for this format
'''

STAGE1_CODE = r'''print(f"---- Stage 1 prompt (JUDGE_FORMAT = {JUDGE_FORMAT!r}) ----")
print(DESC_SYS.strip(), "\n\nTokens: {tokens}\nExplanation of feature behavior:")

todo = cand[~cand["feature"].astype(int).isin(descriptions)]
for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Stage 1: describe"):
    descriptions[int(row["feature"])] = get_description(client, row["judge_tokens"], sys_prompt=DESC_SYS)
    save_judge()
cand["description"] = cand["feature"].map(descriptions)
cand[["feature", "ratio_abs", "description"]]
'''

JUDGE0_MD = r"""### 3.2.e′ Same selection on the original fit → $F_{C0}$

$\rho\ge\tau$ and the same two-stage judge (same format, prompts and thresholds) on `Y0`, cached in `ORIG_DIR/judge0_<format>.json`."""

JUDGE0_CODE = r'''JUDGE0_PATH = ORIG_DIR / f"judge0_{JUDGE_FORMAT}.json"
_judge0 = load_json(JUDGE0_PATH, {})
descriptions0 = {int(f): d for f, d in _judge0.get("descriptions", {}).items()}
verdicts0 = {int(f): tuple(v) for f, v in _judge0.get("verdicts", {}).items()}


def save_judge0() -> None:
    save_json(JUDGE0_PATH, {"descriptions": descriptions0, "verdicts": verdicts0})


stats0 = compute_embedding_stats(Y0.T, vstar_ids, token_to_code, CONCEPT).sort_values("ratio_abs", ascending=False)
cand0 = stats0[stats0["ratio_abs"] >= TAU][["feature", "ratio_abs"]].reset_index(drop=True)
if SMOKE:
    cand0 = cand0.head(3)
for _, row in tqdm(cand0.iterrows(), total=len(cand0), desc="original fit: judge"):
    f = int(row["feature"])
    if f not in descriptions0:
        descriptions0[f] = get_description(client, judge_tokens_for(Y0, f), sys_prompt=DESC_SYS)
        save_judge0()
    if f not in verdicts0:
        verdicts0[f] = classify_membership_with_desc(client, CONCEPT, descriptions0[f])
        save_judge0()
cand0["description"] = cand0["feature"].map(lambda f: descriptions0[int(f)])
cand0["is_member"] = cand0["feature"].map(lambda f: verdicts0[int(f)][0])
cand0["confidence"] = pd.to_numeric(cand0["feature"].map(lambda f: verdicts0[int(f)][1]), errors="coerce").fillna(0.0)
cand0["selected"] = (cand0["is_member"] == True) & (cand0["confidence"] >= CONFIDENCE)
F_C0 = sorted(cand0.loc[cand0["selected"], "feature"].astype(int).tolist())
print(f"original fit: {len(cand0)} candidates pass tau = {TAU}  ->  F_C0 = {F_C0}   (transformed fit: F_C = {F_C})")
print(f"rho >= tau: transformed {int((stats['ratio_abs'] >= TAU).sum())} / {RANK}, original {int((stats0['ratio_abs'] >= TAU).sum())} / {RANK}"
      f"   (nearly all features passing tau in the transformed fit is the leak symptom; compare with TARGET='ext_minus_vstar')")
print("|cos(z~_i, m_X)| for i in F_C: " + (", ".join(
    f"{f}: {cos_m[f]:.3f} (rank {cos_m_order.index(f) + 1}/{RANK})" for f in F_C) or "F_C is empty"))
cand0[["feature", "ratio_abs", "is_member", "confidence", "selected", "description"]]
'''

MATCH_MD = r"""### 3.2.w Original vs transformed: matched-feature overlap, and the run summary

Supports $S^0_i=\{t: Y^0_{i,t}\neq0\}$ and $S_j=\{t: Y_{j,t}\neq0\}$, Jaccard $J_{ij}$, one-to-one matching with the largest total Jaccard
(Hungarian). `T.unmap` maps each transformed direction to the raw space for $|\cos(\mathbf z^0, T^{-1}\tilde{\mathbf z})|$.
`write_summary()` writes `RUN_DIR/summary.json` (configuration, errors, $\rho_{\max}$, #$\{\rho\ge\tau\}$, $F_C$, $F_{C0}$, rank / leak table,
edit size, MC results once 3.3 has them)."""

MATCH_CODE = r'''from scipy.optimize import linear_sum_assignment

S0 = (Y0 != 0).numpy()
S1 = (Y != 0).numpy()
inter = S0.astype(np.int64) @ S1.T.astype(np.int64)
union = S0.sum(1)[:, None] + S1.sum(1)[None, :] - inter
J = inter / np.maximum(union, 1)
rows0, cols1 = linear_sum_assignment(-J)
match0 = dict(zip(rows0.tolist(), cols1.tolist()))
match1 = {j: i for i, j in match0.items()}
Z_back = T.unmap(Z.T).T.float()
cos_map = F.cosine_similarity(Z0[:, rows0], Z_back[:, cols1], dim=0).abs().numpy()
mj = J[rows0, cols1]
print("matched Jaccard quantiles: " + "  ".join(f"q{int(q * 100)}={np.quantile(mj, q):.3f}" for q in (0, .25, .5, .75, 1)))
for f in F_C:
    i = match1[f]
    print(f"transformed concept feature {f} <-> original {i} (in F_C0: {i in F_C0})  J={J[i, f]:.3f}  "
          f"|cos(z0, unmap(z~))|={F.cosine_similarity(Z0[:, i], Z_back[:, f], dim=0).abs():.3f}")
for f0 in F_C0:
    j = match0[f0]
    print(f"original concept feature {f0} <-> transformed {j} (in F_C: {j in F_C})  J={J[f0, j]:.3f}")


def write_summary(**extra) -> dict:
    """Write RUN_DIR/summary.json from the current state; `extra` adds or overrides fields (e.g. mc)."""
    eligible_cols = ((Y[F_C].abs() > 0).any(dim=0).numpy() & (codes == 1)) if F_C else np.zeros(len(vstar_ids), bool)
    summary = {
        "config": {"model": MODEL_NAME, "concept": CONCEPT, "background": BACKGROUND, "target": TARGET, "ext": EXT_TAG,
                   "gamma": GAMMA, "eta": ETA, "eps_rel": EPS_REL, "eps": float(bw.eps), "rank": RANK, "smoke": SMOKE,
                   "neutral_rep": NEUTRAL_REP, "judge_format": JUDGE_FORMAT, "n_target": len(target_ids)},
        "n_vstar": len(vstar_ids),
        "target_options": rank_df.reset_index().to_dict(orient="records"),
        "rel_err": {"original": rel_err0, "transformed": rel_err},
        "rho_max": float(stats["ratio_abs"].max()), "rho_max_original": float(stats0["ratio_abs"].max()),
        "n_tau": int((stats["ratio_abs"] >= TAU).sum()), "n_tau_original": int((stats0["ratio_abs"] >= TAU).sum()),
        "F_C": F_C, "F_C0": F_C0,
        "F_C_matched": [{"feature": f, "partner": match1[f], "jaccard": float(J[match1[f], f]),
                         "partner_in_F_C0": match1[f] in F_C0, "cos_m": float(cos_m[f])} for f in F_C],
        "matched_jaccard_median": float(np.median(mj)),
        "n_edit": int(eligible_cols.sum()), "delta": DELTA, "mc": None,
        **extra,
    }
    save_json(RUN_DIR / "summary.json", summary)
    print(f"wrote {RUN_DIR / 'summary.json'}: F_C={F_C} F_C0={F_C0} n_edit={summary['n_edit']} n_tau={summary['n_tau']} "
          f"rel_err={rel_err:.4f} (orig {rel_err0:.4f}) rho_max={summary['rho_max']:.2f}  mc={summary['mc']}")
    return summary


summary = write_summary()
'''

EDIT_OLD = "C = (Z_C @ Y_C[:, edit_cols]).T                        # (n_edit, d): c_t per edited token\n"
EDIT_NEW = r'''C_X = (Z_C @ Y_C[:, edit_cols]).T                      # (n_edit, d): c~_t, rows in X-space
C = T.unmap(C_X).float()                               # (n_edit, d): c_t in the original space; the mean cancels
if len(edit_ids):  # transforming the edited raw row = editing the transformed row
    gap = (T.forward(E_vstar_tl[edit_cols] - C) - (T.forward(E_vstar_tl[edit_cols]) - C_X.double())).abs().max().item()
    print(f"max |T.forward(e - c) - (T.forward(e) - c_X)| = {gap:.2e}   (scale of c_X: {C_X.abs().max().item():.2e})")
'''

QUESTIONS_HP = '''QUESTIONS_EN = [
    "Who is Harry Potter?",
    "What is Hogwarts?",
    "Who is Lord Voldemort?",
]
QUESTIONS_ML = [
    "¿Quién es Harry Potter?",   # es
    "Qui est Dumbledore ?",   # fr
    "Wer ist Hermine Granger?",   # de
    "Chi è Harry Potter?",   # it
    "Quem é Harry Potter?",   # pt
    "हैरी पॉटर कौन है?",   # hi
    "แฮร์รี่ พอตเตอร์ คือใคร",   # th
]
QUESTIONS_SIM = [
    "Who wrote The Lord of the Rings?",
    "What is Narnia?",
]'''


def concept_questions() -> str:
    out = {"Harry Potter": QUESTIONS_HP}
    for c, slug in [("Gambling", "gambling"), ("Gun", "gun"), ("Halloween", "halloween"), ("Pornography", "pornography"),
                    ("Valentine's Day", "valentines_day")]:
        nb = load_template(f"ember_three_steps_{slug}_multilingual_llama.ipynb")
        s = P.src(nb["cells"][P.find(nb, "QUESTIONS_EN = [")])
        out[c] = s[s.index("QUESTIONS_EN = ["):].strip()
    body = "\n".join(f"    {c!r}: '''\n{q}\n''',".replace("\n'''", "\n'''") for c, q in out.items())
    return ("# English questions about the concept, one per other language, and similar-domain questions (specificity)\n"
            "QUESTION_SETS = {\n" + body + "\n}\n"
            "exec(QUESTION_SETS[CONCEPT])   # defines QUESTIONS_EN, QUESTIONS_ML, QUESTIONS_SIM\n"
            "print(QUESTIONS_EN, QUESTIONS_ML, QUESTIONS_SIM, sep='\\n')\n")


# --------------------------------------------------------------------------------------------- build
def build(template: str, out: Path) -> None:
    nb = load_template(template)
    P.clear_outputs(nb)

    # generic text: "Gambling" -> the configured concept
    for i, c in enumerate(nb["cells"]):
        s = P.src(c)
        if "QUESTIONS_EN = [" in s or "CONCEPT = 'Gambling'" in s:
            continue
        s2 = s.replace("Gambling", "concept").replace("gambling", "concept").replace("casino", "concept token")
        if s2 != s:
            P.set_source(nb, i, s2)

    P.set_source(nb, P.find(nb, "# EMBER end to end, multilingual concept"), TITLE)
    P.set_source(nb, P.find(nb, "CONCEPT = 'Gambling'"), CONFIG)

    # 3.1.b markdown: no concept-specific counts
    i = P.find(nb, "### 3.1.b Build $S_C \\cup S_N$")
    P.set_source(nb, i, re.sub(r"- \$S_C\$: up to 40 Wikipedia.*?\n  \$S_N\$", "- $S_C$: up to 40 Wikipedia concept sentences per language "
                                                                               "(`concept_paths(CONCEPT)`).\n  $S_N$", P.src(nb["cells"][i]), flags=re.S))

    # 3.1.c' after the A = E_vstar^T cell
    i = P.find(nb, "A = E_vstar.T.contiguous().to(DEVICE)", "code")
    P.insert_after(nb, i, [P.markdown(WHITEN_MD), P.code(WHITEN_CODE)])

    # 3.1.d on X, 3.1.d' raw
    i = P.find(nb, 'SMF_PATH = RUN_DIR / "smf.pt"')
    P.set_source(nb, i, FIT_CODE)
    P.insert_after(nb, i, [P.markdown(FIT0_MD), P.code(FIT0_CODE)])
    P.set_source(nb, P.find(nb, "rel_err = (torch.norm(A_cpu - Z @ Y)", "code"), ERR_CODE)
    P.set_source(nb, P.find(nb, "U, S, Vh = torch.linalg.svd(A_cpu", "code"), SVD_CODE)

    # 3.2.a' and the judge
    i = P.find(nb, "### 3.2.a′ Token → source word")
    P.set_source(nb, i, WORDS_MD)
    P.set_source(nb, P.find(nb, "word_counts = {}  # tid -> Counter of (word, lang)\nfor s, _ in tqdm(dataset"), WORDS_CODE)
    i = P.find(nb, "def ml_judge_tokens(f: int")
    s = P.src(nb["cells"][i])
    s = s[:s.index("def ml_judge_tokens")] + JUDGE_TOKENS_FN + "\n\n" + s[s.index("cand = candidates["):]
    P.set_source(nb, i, s)
    i = P.find(nb, 'JUDGE_PATH = RUN_DIR / "judge.json"')
    P.replace_in(nb, i, 'JUDGE_PATH = RUN_DIR / "judge.json"', 'JUDGE_PATH = RUN_DIR / f"judge_{JUDGE_FORMAT}.json"')
    P.replace_in(nb, i, "# Delete judge.json together with smf.pt when re-fitting Step 3.1.",
                 "# One cache per JUDGE_FORMAT; moved aside with smf.pt when Step 3.1 is re-fitted.")
    P.set_source(nb, P.find(nb, "ML_TOKENS_DESC_SYS = "), STAGE1_CODE)
    i = P.find(nb, "### 3.2.d LLM judge, stage 1")
    P.set_source(nb, i, re.sub(r"The prompt is the repo's.*", "The system prompt depends on `JUDGE_FORMAT` "
                               "(`ember.judge_words.desc_sys_prompt`) and is passed to `get_description` explicitly.",
                               P.src(nb["cells"][i]), flags=re.S))
    i = P.find(nb, "descriptions[f] = get_description(client, ml_judge_tokens(f))")
    P.replace_in(nb, i, "get_description(client, ml_judge_tokens(f))", "get_description(client, ml_judge_tokens(f), sys_prompt=DESC_SYS)")
    i = P.find(nb, 'F_C = sorted(cand.loc[cand["selected"], "feature"]', "code")
    P.insert_after(nb, i, [P.markdown(JUDGE0_MD), P.code(JUDGE0_CODE)])

    # analysis in X-space (Z lives there)
    for i in P.find_all(nb, '(E_vstar_tl if "E_vstar_tl" in globals() else E_vstar)'):
        s = P.src(nb["cells"][i]).replace('(E_vstar_tl if "E_vstar_tl" in globals() else E_vstar)', "E_vstar_X")
        P.set_source(nb, i, s.replace("# E_vstar is deleted in 3.3.a", "# X-space rows (the space Z lives in); kept after 3.3.a"))

    # 3.2.h: capped sets through the same transform
    i = P.find(nb, 'CONCEPT_EXT_PATH = concept_paths(CONCEPT)["extended"] / "concept_ext.json"')
    P.replace_in(nb, i, 'CONCEPT_EXT_PATH = concept_paths(CONCEPT)["extended"] / "concept_ext.json"\n'
                        'NEUTRAL_EXT_PATH = neutral_paths(CONCEPT, NEUTRAL_REP)["ext"]\n',
                 'CONCEPT_EXT_PATH, NEUTRAL_EXT_PATH = EXT_FILES["concept"], EXT_FILES["neutral"]   # built in 3.1.c′ (EXT_PER_LANG, EXT_TOPUP)\n')
    P.replace_in(nb, i, "A_ext = E_ext.T.contiguous().to(DEVICE)",
                 "X_ext = transform(E_ext)                                                 # the 3.1.c' transform, unchanged\n"
                 "A_ext = X_ext.T.contiguous().to(DEVICE)")
    P.replace_in(nb, i, 'print(f"S_C^ext = {len(ext_concept)}',
                 '_tr = Counter(r["lang"] for r in ext_concept_rows if r.get("source") == "translated")\n'
                 'print(f"translated top-up in S_C^ext: {dict(_tr)} ({sum(_tr.values())} of {len(ext_concept_rows)})")\n'
                 'print(f"S_C^ext = {len(ext_concept)}')
    i = P.find(nb, "### 3.2.h Extended vocabulary")
    P.set_source(nb, i, """### 3.2.h Extended vocabulary: fixed-Z refit on a larger token set, through the same transform

Keeps $Z$ from 3.1 fixed and refits only a sparse $Y'$ on a larger multilingual token set (files from `extended_sets.ext_paths`):

- $S_C^{ext}$: per language the 40 $V^*$ concept sentences plus a seeded sample of the concept page, up to `EXT_PER_LANG` (300);
  pages shorter than that are filled with Gemini translations of the other languages' pages (`EXT_TOPUP`, rows marked `source`);
- $S_N^{ext}$: per language the $V^*$ neutral sentences plus translated neutral sentences, as many as the concept side.

The rows go through the 3.1.c′ transform $T$ unchanged (nothing re-estimated). The target of 3.1.c′ is built from the same $S_C^{ext}$,
so with `TARGET = "ext"` the extended concept tokens shaped $T$; `ext_minus_vstar` keeps that, `tc` does not.
With `SMOKE`, each side keeps its first 30 sentences per language.""")

    # 3.2.w + summary before Step 3.3, with the overnight stop marker
    i = P.find(nb, "# Step 3.3: Erasing concept-related features", "markdown")
    P.set_source(nb, i, "<!-- STEP-3.3-START -->\n" + P.src(nb["cells"][i]) + """
**Whitened/sharpened variant.** $Z, Y$ live in X-space; the concept part $\\tilde{\\mathbf c}_t = \\sum_{i\\in F_C} Y_{i,t}\\tilde{\\mathbf z}_i$ is
mapped back with $\\mathbf c_t = \\tilde{\\mathbf c}_t S_\\eta^{-\\gamma/2}W^{-1}$ (`T.unmap`; the mean cancels), so subtracting $\\delta\\mathbf c_t$
from the raw row is exactly subtracting $\\delta\\tilde{\\mathbf c}_t$ in X-space. $\\delta$ needs retuning (3.3.g, 3.4).
""")
    P.insert_before(nb, i, [P.markdown(MATCH_MD), P.code(MATCH_CODE)])

    # 3.3.a: free the transformed matrices too
    i = P.find(nb, "del act_generator, tl_model, E, E_vstar, A, A_cpu")
    P.replace_in(nb, i, "del act_generator, tl_model, E, E_vstar, A, A_cpu",
                 "del act_generator, tl_model, E, E_vstar, E_vstar_orig, E_X, A, A_cpu, A0, A0_cpu")
    P.replace_in(nb, i, 'globals().pop("smf", None)  # absent when 3.1.d loaded smf.pt',
                 'for _name in ("smf", "smf0"):  # absent when 3.1.d / 3.1.d\' loaded their caches\n        globals().pop(_name, None)')
    # 3.3.b: edit mapped back
    i = P.find(nb, EDIT_OLD.strip())
    P.replace_in(nb, i, EDIT_OLD, EDIT_NEW)
    # summary with MC results
    i = P.find(nb, "mc_test = compare_mc()")
    P.set_source(nb, i, "mc_test = compare_mc()\nsummary = write_summary(n_edit=len(edit_ids), mc=mc_test.to_dict())\nmc_test\n")

    # questions for any concept
    P.set_source(nb, P.find(nb, "QUESTIONS_EN = ["), concept_questions())
    i = P.find(nb, "### 3.3.d′ Questions about")
    P.set_source(nb, i, P.src(nb["cells"][i]).replace("Questions about concept", "Questions about the concept"))

    out.parent.mkdir(parents=True, exist_ok=True)
    P.save(nb, out)
    print(f"wrote {out.relative_to(ROOT)} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    for t, o in TEMPLATES.items():
        build(t, o)
