# EMBER — Architecture & Agent Handbook

> **Read this first.** It is written for an autonomous agent (or a new human) that has
> never seen this repo. It explains what the project is, how the code is organised,
> how a run flows end to end, what the on-disk contracts are, which invariants must not
> be broken, and where to plug new work in.
>
> Everything below was derived by reading the source. Where a statement is a *plan* and
> not yet implemented, it is marked **[planned]**.

---

## 1. What this repository is

Two things live here at once:

1. **The EMBER research codebase** — the reference implementation of
   *"Don't Forget Your Embeddings: Robust Knowledge Erasure via Precise Editing of
   Embeddings"* (Suslik, Shafran, Geva; arXiv 2606.03695). EMBER = **EMB**edding
   **ER**asure: it removes a concept from an LLM by subtracting the concept's
   contribution from the **token-embedding matrix**, and can be stacked on top of
   existing unlearning methods (SNMF, RMU, CRISP, PISCES) to make them more robust
   to relearning.

2. **A course project (`nlp_proj`) extending EMBER cross-lingually.** The research
   question: *does erased concept knowledge re-emerge when prompts are translated
   into other languages, and does that differ between embedding-based and MLP-based
   erasure?* The plan lives in `archive_gemini/docs/crosslingual_plan.md`; translated evaluation
   data already sits in `data/qa_lang/`. **The code changes for this are not written
   yet** — see §16.

Supported models: `google/gemma-2-2b-it` (rank 100) and
`meta-llama/Llama-3.1-8B-Instruct` (rank 200). Anything else needs new layer-range
constants (§9) and a new PISCES feature file.

---

## 2. Ten-second mental model

```
              ┌──────────────────────── OFFLINE, per concept ───────────────────────┐
  English     │  train_mf_features.py        interpret_features.py                  │
  Wikipedia ──┼─►  factorize W_E  (embedding track) ──► Gemini names each feature ──┼─► mf_outputs/
  sentences   │    factorize MLP acts (MLP track)      ──► potential_features.csv   │   (features +
              └─────────────────────────────────────────────────────────────────────┘    which ones
                                                                                          are "the concept")
              ┌──────────────────────── ONLINE, per run ────────────────────────────┐
  YAML config │  run_erasure.py ─► pipeline.run(cfg)                                │
  + CLI    ───┼─►  baselines ─► EMBER δ ─► method HP grid ─► validate ─► final test ─┼─► results/
              │                                              └─► relearning         │   (CSV rows)
              └─────────────────────────────────────────────────────────────────────┘
```

Three sentences:

- **A "feature"** is a column of a sparse matrix factorization. The *embedding track*
  factorizes the token-embedding matrix; the *MLP track* factorizes MLP activations.
- **Erasure** = pick the concept-related feature columns, then subtract/project them out
  of the model's weights, scaled by a hyper-parameter δ.
- **Everything else in the repo is a harness**: enumerate a grid of δ / layer ranges /
  learning rates, evaluate each cell (does the concept die? does the model survive?),
  keep the best cell, stress-test it by trying to relearn the concept.

---

## 3. Repository layout

```
nlp_proj/
├── ember/                        ← the only Python package (installed as `ember-erasure`)
│   ├── run_erasure.py            ← ENTRY POINT: full pipeline (grid → validate → test)
│   ├── run_eval.py               ← ENTRY POINT: final-test stage only
│   ├── train_mf_features.py      ← ENTRY POINT: build factorizations  → mf_outputs/
│   ├── interpret_features.py     ← ENTRY POINT: Gemini names features → potential_features.csv
│   ├── data.py                   ← downloads prebuilt features from HF hub
│   ├── local_datasets.py         ← ALL dataset loading (QA, MMLU/Alpaca indices, concept sentences)
│   ├── utils.py                  ← SparseMatrixFactorization + path/seed/stat helpers
│   ├── timing.py                 ← `Timer` context manager
│   ├── erasure/                  ← the run harness
│   │   ├── pipeline.py           ← ★ the driver: 5 stages, resume logic, timing.json
│   │   ├── config.py             ← ★ RunConfig dataclasses + YAML loader + CLI
│   │   ├── io.py                 ← ★ output paths, CSV append/resume, column schema
│   │   ├── eval.py               ← thin adapter onto ember/evals
│   │   ├── features.py           ← feature loading/selection + `ConceptContext` cache
│   │   ├── embed_edit.py         ← the EMBER edit itself (embedding weights)
│   │   ├── mlp_edit.py           ← MLP up_proj/down_proj projection-out
│   │   ├── relearning.py         ← post-erasure fine-tuning stress test
│   │   ├── model_loader.py       ← HF + TransformerLens loading
│   │   ├── log.py                ← `[method|concept|stage]` structured logging
│   │   └── methods/              ← ★ one plug-in per erasure method
│   │       ├── base.py           ←   abstract `Method` + `REGISTRY`
│   │       ├── ember.py  snmf.py  rmu.py  crisp.py  pisces.py
│   └── evals/                    ← measurement, model-agnostic
│       ├── orchestrator.py       ← ★ `evaluate_model_for_mode` — the whole metric flow
│       ├── baselines.py          ← pristine-model scores + on-disk cache
│       ├── model_wrap.py         ← `WrappedHFModel` (chat template, BOS, batching)
│       ├── mc.py open_qa.py mmlu.py alpaca.py   ← the four eval sets
│       ├── gemini.py             ← Gemini-as-judge client (rate limit, retries, cost log)
│       ├── harmonic.py schema.py
├── configs/                      ← 18 YAML files: <method>[_ember]_<model>.yaml
├── data/                         ← all inputs that are not model weights (§11)
├── docs/
│   ├── progress.md               ← main progress log (start here)
│   ├── architecture/ARCHITECTURE.md  ← this file
│   ├── references/               ← EMBER paper, project proposal (PDFs)
│   ├── judge/                    ← Gemini → Qwen judge swap: plan + agreement-study results
│   ├── Part 1/                   ← cross-lingual leakage: per-method HP searches, leakage runs
│   └── Part 2/                   ← multilingual extension: ideas and early findings
├── external/                     ← git submodules: snmf, CRISP, wmdp, PISCES
├── *.sbatch                      ← SLURM job scripts for the university cluster
├── demo.ipynb                    ← walkthrough: erasing Harry Potter from Gemma
├── pyproject.toml / requirements.txt / uv.lock
└── mf_outputs/  results/         ← GENERATED, gitignored (absent in this checkout)
```

**Anything with a ★ is load-bearing.** Read those five files before changing behaviour.

---

## 4. The three-phase workflow

| Phase | Command | Reads | Writes | Needs |
|---|---|---|---|---|
| **A. Factorize** | `python -m ember.train_mf_features` | `data/concept_sentences.json`, `data/neutral_sentences.json` | `mf_outputs/**/pickles/`, `mf_outputs/**/csvs/` | GPU, `HF_TOKEN` |
| **B. Interpret** | `python -m ember.interpret_features` | phase A's CSVs | `mf_outputs/**/interpretations/potential_features.csv` | `GEMINI_API_KEY` |
| **C. Erase + eval** | `python -m ember.run_erasure --config … --concepts …` | phase B's output + `data/*.json` | `results/**/*.csv` | GPU, `GEMINI_API_KEY` |

Phases A+B are **optional**: `features_source: hf` (the default) makes phase C download
prebuilt features from the `ClSu/ember-features` HF dataset (`ember/data.py`), scoped to
`(model, rank, seed, concepts)`. Set `features_source: local` to use whatever is already
in `mf_outputs/` and never contact the hub.

Phase A is *seeded and reproducible*. Phase B calls an LLM and may select a slightly
different feature set between runs — **this is the one non-deterministic step**.

---

## 5. Phase A — `train_mf_features.py`

Two independent tracks per (concept, rank):

**Embedding track** (all EMBER needs, `--skip-mlp` to run only this)

1. Tokenize every concept + neutral sentence → `V′` = the set of unique non-special
   token ids that occur.
2. `A = E[V′].T`, shape `(d_model, |V′|)`.
3. `SparseMatrixFactorization` (in `ember/utils.py`, subclass of `NMFSemiNMF` from the
   `snmf` submodule) → `A ≈ F @ G.T` with
   - `F_` : `(d_model, K)` — **dense, signed** erasure directions in model space
   - `G_` : `(|V′|, K)` — **WTA-sparse, signed** per-token activation scores
     (`g_sparsity=0.01` ⇒ each feature keeps its top 1% of tokens)
   Alternating closed-form least squares with ridge, early stopping on Frobenius loss.
4. Emits `embedding.pkl` (`{"nmf": …, "vprime_token_ids": [...]}`),
   `token_features.csv` (top activating tokens + their concept/neutral/both labels,
   plus tokens obtained by projecting `F` columns through `W_U`), and
   `stats_embed.csv` (per-feature `ratio_abs = mean|G| over concept tokens ÷ over
   neutral tokens` — the score used to select features later).

**MLP track** (needed only for the SNMF erasure method)

Per layer: `NMFSemiNMF` over that layer's MLP activations (`sparsity=0.01` on `F_`),
one `layer{N}.pkl` per layer plus `token_features.csv` /
`stats_concept_vs_neutral.csv`.

Both tracks skip work that already exists on disk (checked by pickle + CSV presence).

---

## 6. Phase B — `interpret_features.py`

For every feature whose `ratio_abs ≥ --ratio-thresh` (default 2.0):

1. Send its top-20 activating tokens to Gemini → a one-sentence description.
2. Ask Gemini `{"is_member": bool, "confidence": 0..1}` for "is this feature *the
   concept*?" (prompt explicitly demands the concept be *dominant*, not merely present).
3. Keep rows with `is_member == true AND confidence ≥ --confidence-thresh` (0.85).

Outputs per (concept, track): `from_activation.csv`, `from_projection.csv` (MLP only),
and the union → **`potential_features.csv`**, which is the file every erasure method
reads. Threaded (5 workers), resumable (`--resume`, on by default), atomic CSV writes.

---

## 7. Phase C — the erasure pipeline (`ember/erasure/pipeline.py`)

`run_erasure.py` is a 40-line wrapper: load `.env` → `parse_args()` → configure logging
→ optionally `ensure_features()` → `pipeline.run(cfg)`. All the logic is in `pipeline.run`.

### The five stages

```
Step 0  load model + compute/read BASELINES for train_<mode> (+ test_mc, test_open)
Step 1  resolve the EMBER δ per concept        (_resolve_ember_step)
Step 2  run the method's HP GRID               (_run_method_grid)     → hps.csv, top_hps.csv
Step 3  VALIDATE the top-K with Alpaca         (_run_validate_topk)   → top_hps_valid.csv
Step 4  FINAL TEST the single best HP          (_run_final_test)      → final_test_{mc,open}.csv
        └─ then RELEARNING, patched back into those CSVs
        finally: timing.json in the run root
```

**Step 0 — baselines.** The pristine model's score on every eval set. Every metric in
this project is a *fraction of baseline*, so this is mandatory. Cached as JSON in
`data/baselines/<model_safe>/baseline_<set>.json`; recomputed only if a required
concept is missing from `meta.per_concept`.

**Step 1 — the EMBER δ.** For non-`ember` methods with `ember_step.enabled: true`, the
pipeline needs one best embedding δ per concept *before* the method grid runs. Lookup
order:

1. `method == "ember"` → no-op (the grid *is* the δ search), returns `{concept: 0.0}`.
2. `ember_step.enabled == false` → `{concept: 0.0}` (plain method, no `_ef` suffix).
3. Cache hit at `results/ember/<model>/rank<R>/seed<S>/train_<mode>/best_embed.csv` → load.
4. Otherwise run the EMBER grid *inline* for the missing concepts, then load.

This is why a `*_ember_*.yaml` run implicitly produces `results/ember/…` output too.

**Step 2 — the method grid.** For each concept, for each HP cell:
restore the model → `method.apply(...)` → `eval_mod.evaluate_model(...)` →
append one row to `hps.csv` → after the concept, write `top_hps.csv` (top-K by
`harmonic`). Grid-stage eval is cheap: **no Alpaca**, and it early-stops a cell that
fails `min_mmlu` or exceeds `max_qa_acc`.

**Step 3 — validate.** Re-evaluate only the top-K rows, this time *with* Alpaca,
reusing the MMLU/QA/SimDom numbers already in `top_hps.csv` via `precomputed_metrics`.
Ranks by `harmonic_alpaca`. Skipped for methods where `writes_topk_csv()` is False
(i.e. `ember`).

**Step 4 — final test.** Pick the best HP (`top_hps_valid.csv` by `harmonic_alpaca` if
present, else `method.pick_best_hp_row(hps.csv)`), apply it once, then evaluate on
**both** `test_mc` and `test_open` with full Alpaca and no early-stop gates. Then run
relearning once and patch `relearning_qa_mc` / `relearning_qa_open` into the two CSVs.

### Resume semantics

The pipeline is designed to be killed and restarted (SLURM `studentkillable` partition).

- **Grid / validate**: `io.load_done_set(csv, key_cols)` reads the CSV back and skips
  any HP tuple already present. Key columns are per-method (`Method.hp_key_columns()`).
  Type casting on read is done by prefix/suffix rules in `io._cast_key_cell` — a key
  column named `layer_*`, `k_features*`, `n_tokens_edited`, or ending `_lo/_hi/_step/_id`
  is an `int`, `setting_name` is a `str`, everything else is a `float`.
- **Final test**: `_classify_concept_state` returns `done` / `partial` / `fresh`.
  `partial` means the metric rows exist but relearning didn't finish → only relearning
  is re-run. `fresh` drops any existing rows for that concept first.
- `--overwrite` forces everything to be treated as fresh.

### Timing

`Timer` wraps each cell; `_STAGE_TIMES[concept][stage]` accumulates and is merged into
`<run_root>/timing.json` at the end (`grid`, `validate`, `final_test`, `relearning`).
Every CSV row also carries its own `wall_time_s`.

---

## 8. Core abstractions

### 8.1 `RunConfig` (`erasure/config.py`)

A tree of dataclasses. **YAML supplies everything invariant for a (method, model) pair;
the CLI supplies only what varies per invocation.**

```
RunConfig
├── method            "snmf" | "rmu" | "crisp" | "ember" | "pisces"
├── model_name, cache_dir, rank, seed, topk, overwrite
├── run_tests_after_train : bool     # run step 4?
├── features_source   "hf" (download) | "local"
├── ember_step   : EMBERStepConfig   # enabled + deltas  → the "_ef" variant
├── selection    : SelectionConfig   # ratio_thresh / coverage_thresh / neurons_thresh
├── eval         : EvalConfig        # alpaca, min_mmlu, max_qa_acc, alpaca_batch_size
├── relearning   : RelearningConfig  # enabled, max_paragraphs, csv_path
├── snmf / rmu / crisp / ember / pisces : per-method grid configs
└── concepts, train_eval             # CLI ONLY — never in YAML
```

- `load_yaml` **rejects unknown keys** at every level (a YAML typo is a hard error, not
  a silent default). Good.
- `validate()` enforces: known method, known `train_eval`, non-empty concepts, positive
  rank, valid `features_source`, at most one of `neurons_thresh`/`coverage_thresh`, and
  force-disables `ember_step` when `method == "ember"`.
- CLI: `--config` (required), `--concepts` (required, nargs +), `--train-eval {mc,open}`,
  `--seed`, `--rank`, `--features-source`, `--overwrite`.

### 8.2 The `Method` plug-in protocol (`erasure/methods/base.py`)

This is the extension seam. `pipeline.py` contains **zero** method-specific branching
(one exception: the string comparison `method.name != "ember"` when deciding the `_ef`
directory suffix).

| Member | Required | Purpose |
|---|---|---|
| `name` | ✔ | matches `RunConfig.method`; registry key |
| `requires_full_reload` | | `True` ⇒ pipeline never snapshots; the method reloads the model itself (RMU) |
| `enumerate_hps(cfg)` | ✔ | yields one dict per grid cell |
| `hp_key_columns()` | ✔ | subset that uniquely identifies a cell (resume key) |
| `hp_columns()` | ✔ | all HP + info columns this method writes |
| `apply(model, tok, hp, concept, cfg)` | ✔ | mutate the model; return *info columns* |
| `snapshot()` / `restore()` | | undo `apply` between cells (default no-op) |
| `on_concept_start/end`, `before_relearning`, `after_concept_grid` | | lifecycle hooks |
| `grid_eval_kwargs(cfg)` | | eval settings for the **grid** stage only |
| `pick_best_hp_row(df)` | | default = top by `harmonic` |
| `writes_topk_csv()` | | default `True`; `ember` returns `False` |
| `skip_eval_for_hp(hp)` | | emit a zero-metric row without evaluating (EMBER's δ=0) |
| `get_model_to_eval()` | | return a *different* model object to evaluate (RMU, CRISP) |

Registration is import-time: each module ends with `register(XMethod())`, and
`methods/__init__.py` imports all five, so `base.get("snmf")` works after
`import ember.erasure.methods`.

### 8.3 The column schema (`erasure/io.py`)

Every CSV in `results/` is built from the same concatenation:

```
ID_COLUMNS + EMBED_COLUMNS + <method HP columns> + <metric columns for the mode> [+ ALPACA] + wall_time_s
```

- `ID_COLUMNS` = `model, concept, method, rank, seed, embed_step_enabled`
- `EMBED_COLUMNS` = `delta_embed, k_features_embed, n_tokens_edited` (present for
  **every** method — that is the whole point of the `_ef` design)
- metric sets differ between `train_mc` / `train_open` / `test_*` (MC adds `*_invalid`
  columns; test adds Alpaca + a `relearning_qa_*` column)
- `full_columns_for()` builds grid/test headers; `validate_columns_for()` always
  includes Alpaca.

`append_csv_row()` reindexes each row to that column list (missing → NaN) and appends,
writing the header only on first write. **Never write a results CSV by hand — always go
through `io.append_csv_row` with the right column list**, or resume will break.

### 8.4 Logging & timing

`erasure/log.py` prefixes every line with `[method|concept|stage]` via a filter over a
module-level `_CONTEXT` dict; `log.stage(...)` and `log.concept(...)` are context
managers that set and restore fields. This is what makes multi-hour SLURM `.out` files
greppable — keep using it instead of `print` inside the pipeline.

---

## 9. The five erasure methods

| Method | What it edits | Snapshot strategy | Aux state | Layer constants |
|---|---|---|---|---|
| **ember** | input embedding `W_E` | cheap (clone `W_E`) | none | — |
| **snmf** | MLP `up_proj` rows / `down_proj` cols | clone `W_E` + all MLP weights | `ConceptContext` cache | `GEMMA/LLAMA_LAYER_RANGES_IN/OUT` in `methods/snmf.py` |
| **rmu** | full fine-tune (gradient) | **none** — reloads the model per cell | frozen ref model + working model | `GEMMA/LLAMA_UPDATE_SETTINGS`, `FIXED_PARAM_IDS=[6]` |
| **crisp** | LoRA adapter over SAE features | none — `unload_lora()` per cell | one `CRISP` object per layer-range, SAE cache | `GEMMA/LLAMA_LAYER_RANGES` in `methods/crisp.py` |
| **pisces** | MLP `W_out` in TransformerLens, synced to HF `down_proj` | clone `W_E` + `down_proj` only | a resident TL model | — (feature JSON instead) |

### 9.1 EMBER (`embed_edit.py` + `methods/ember.py`) — the paper's contribution

For every **eligible** token *i*:

```
c_i = F′[:, k′] @ G′[i, k′]        # the factorization's concept contribution to token i
e_i ← e_i − δ · c_i
```

- `k′` = feature columns from `potential_features.csv` with `metric_score ≥ ratio_thresh`
  (falls back to *all* features if the threshold leaves none).
- **Eligible** = the token is labelled exactly `== concept_name` in
  `csvs/…/embedding/token_features.csv` **AND** it is in `V′` **AND** it has a nonzero
  `G` on at least one selected feature. Tokens labelled `both` (appear in concept *and*
  neutral sentences) are deliberately **not** edited.
- **Gemma scale trap:** HF stores Gemma's `W_E` unscaled and multiplies by
  `sqrt(d_model)` at runtime, but the factorization was fit in the *scaled* space.
  `_embedding_scale()` therefore multiplies on read and divides on write. Llama = 1.0.
- Grid = a list of δ. `skip_eval_for_hp` makes δ=0 a free zero-metric row.
  `grid_eval_kwargs` is **overridden**: EMBER always evaluates Alpaca and gates on
  `min_mmlu=0.90, min_alpaca=0.9, max_qa_acc=None` — hard-coded, *not* read from
  `eval:` in the YAML (the configs say so in a comment).
- `after_concept_grid` appends the winner to the shared `best_embed.csv`, which is what
  step 1 of every other method reads.
- `concept_edit_tokens()` is a read-only helper returning `(token, ‖c_i‖)` for every
  edited token — the basis for the planned token-overlap analysis.

### 9.2 SNMF (`mlp_edit.py` + `methods/snmf.py`)

Projects each concept feature direction out of the MLP weights, restricted to the
neurons in that feature's support mask:

```
out:  down_proj[:, idx] −= δ · f ⊗ (f · W_cols)     # f normalised
in :  up_proj[idx, :]   −= δ · (W_rows · f) ⊗ f
```

Grid = `in_deltas × ranges_in × out_deltas × ranges_out` (a `δ=0` side collapses to one
`(0.0, -1, -1)` no-op cell). `w_mode` ∈ `in`/`out`/`both` decides which sides run.
Optional feature-set narrowing: `neurons_thresh` (keep neurons covered by ≥N features)
or `coverage_thresh` (keep the top-k neurons covering that fraction of normalised mass;
the expensive per-neuron mass arrays are cached as `.npz` under `mf_outputs/**/cache/`).

`ConceptContext` caches the potential-features DataFrame and, per
`(layer_lo, layer_hi, w_mode)`, the materialised feature tensors — so each SNMF pickle
is unpickled at most once per range per concept.

### 9.3 RMU, CRISP, PISCES — the baselines

- **RMU** (`external/wmdp`): representation-misdirection fine-tuning. `requires_full_reload
  = True`; every cell loads a fresh model from disk (48 GB VRAM for Llama), applies the
  embed edit to *both* the frozen reference and the working copy, then calls
  `run_rmu(...)`. Grid = `lr × alpha × steering × update_setting`.
- **CRISP** (`external/CRISP`): SAE feature identification + LoRA unlearning. Manages its
  own cache — one `CRISP` object per `(concept, layer_range)`, SAE checkpoints downloaded
  once into `external/CRISP/crisp/{gemma,llama}_sae_cache/`, and only the LoRA is retrained
  between cells. `data_config=None` is passed deliberately so `crisp` and `crisp_ef` runs
  don't collide on the same processed-features filename.
- **PISCES** (`external/PISCES`): edits SAE features in TransformerLens space via an
  `unlearn_concept` context manager, then `copy_tl_to_hf_weights` mirrors every
  `blocks[L].mlp.W_out` into HF `down_proj.weight` (they are transposes) so the normal
  HF eval path works unchanged. `linscale=True` for Gemma, `False` for Llama.
  `before_relearning` frees the TL model so the optimizer fits in memory.

---

## 10. The evaluation subsystem (`ember/evals/`)

### 10.1 The four eval sets

| Set | Loader | Scoring | Cost |
|---|---|---|---|
| **QA** (the concept) | `load_mc_qa_items` / `load_open_qa_examples` | MC: parse a letter · Open: Gemini judge | 50 items/concept |
| **SimDom** (a similar domain) | same, `subset="SimdomQA"` | same | 50 items/concept |
| **MMLU** | `load_mmlu_indices` + `cais/mmlu` (cached to `data/mmlu_cache.json`) | letter parse, batched | 1000 test items |
| **Alpaca** | `tatsu-lab/alpaca_eval` via `hf_hub_download` | Gemini scores instruction-relevance and fluency 0–2 | 150 gens + 300 judge calls |

### 10.2 The metric definitions (`orchestrator.evaluate_model_for_mode`)

Everything is a **fraction of the pristine model's score**, chance-corrected for MC:

```
mmlu_frac   = clip01( (mmlu_acc   − .25) / (mmlu_base   − .25) )
qa_frac     = clip01( (qa_acc     − .25) / (qa_base     − .25) )    # MC
qa_frac     = min(1, qa_acc / qa_base)                              # open
simdom_frac = same shape as qa_frac

efficacy        = 1 − qa_frac                       ← concept gone?
specificity     = HM(mmlu_frac, simdom_frac)        ← model intact?
harmonic        = HM(efficacy, specificity)         ← grid ranking key
coherence       = HM(alp_instr_frac, alp_flu_frac)  ← still fluent?
harmonic_alpaca = HM(efficacy, specificity, coherence)  ← validate/test ranking key
```

`harmonic_mean` returns **0.0 if any input is ≤ 0 or non-finite** — so one zeroed
component zeroes the whole aggregate. That is intentional (it is a hard gate), but it
means a NaN anywhere silently becomes a 0 score.

### 10.3 Execution order and early exits

```
MMLU ──► if mmlu_frac < min_mmlu:  zero out qa/simdom/efficacy/specificity/harmonic, RETURN
QA   ──► if qa_frac  > max_qa_acc: zero out simdom/efficacy/specificity/harmonic, RETURN   (open only)
SimDom
harmonic
Alpaca (optional) ──► if min_alpaca set and either frac < it: harmonic_alpaca := 0
```

These gates make the grid affordable — a cell that has already lobotomised the model
never pays for Alpaca. They also mean **grid-stage rows are not comparable as raw
measurements**; a `0.0` may mean "not measured".

### 10.4 `WrappedHFModel` (`evals/model_wrap.py`) — read before touching prompts

Handles the two models' chat templates and, critically, **double-BOS avoidance**:

- Gemma-2: `apply_chat_template` output has its leading `<bos>` **string** stripped
  (`s[5:]`), then tokenized with `add_special_tokens=True` so exactly one BOS token
  is prepended.
- Llama-3: template kept intact (it contains `<|begin_of_text|>`), tokenized with
  `add_special_tokens=False`.

Also forces `padding_side = "left"` (decoder-only batched generation) and strips the
prompt header out of the decoded string per model family. MC generation is deliberately
**unbatched** (one question at a time) — batching with left-padding can perturb greedy
decoding and would invalidate the cached baselines.

### 10.5 The Gemini judge (`evals/gemini.py`)

Process-wide rate limiter, retry-with-backoff, a hard abort after 20 cumulative quota
errors, and an append-only cost log at `gemini_usage_log.jsonl` (priced in NIS for
`gemini-flash-lite`). `GeminiBadFinishError` (SAFETY / RECITATION) is retried 3× by the
callers, then **the item defaults to incorrect / score 0** — note this bias when
evaluating sensitive concepts.

### 10.6 Relearning (`erasure/relearning.py`)

The paper's headline stress test. Full-parameter fine-tuning of the *erased* model on
concept paragraphs (`data/relearn_paragraphs.json`), default **2 epochs, lr 1e-5,
batch 1, max_len 128**, 8-bit AdamW, gradient checkpointing on. Evaluates after each
epoch into `relearning.csv`; the final `qa_acc` on `test_mc` and `test_open` is patched
back into the final-test CSVs. **Lower is better.** The model is mutated in place, so
the caller must restore it afterwards (the pipeline does).

---

## 11. On-disk contracts

### 11.1 `data/` — inputs (checked in)

| File | Shape | Used by |
|---|---|---|
| `concept_sentences.json` | `[{concept, sentences:[…]}]` | phase A, RMU/CRISP forget set |
| `neutral_sentences.json` | `[{sentence}]` | phase A, RMU/CRISP retain set |
| `mc_questions.json` | `{concept: {"QA_train"/"QA_test"/"SimdomQA_train"/"SimdomQA_test": [{q, options[4+], correct_answer}]}}` | MC eval |
| `open_questions.json` | same keys, items `{q, a}` | open eval |
| `mmlu_{train,test}_indices.json` | `[int]` into `cais/mmlu` "all"/test | MMLU eval |
| `alpaca_{train,test}_indices.json` | `[int]` into `alpaca_eval.json` | Alpaca eval |
| `coherency_prompts.json` | `{concept: [≥20 prompts]}` | CRISP only |
| `relearn_paragraphs.json` | `{concept: {"RelearnParagraphs": […]}}` | relearning |
| `pisces_concept_features_{gemma,llama}.json` | `[{name, features:[{layer,id,neg?}]}]` | PISCES |
| `qa_lang/translated/{de,es,fr,ru}/{mc,open}_questions.json` | mirrors the English files | **[planned]** cross-lingual eval — *not yet read by any code* |

Generated + gitignored under `data/`: `baselines/`, `baseline_samples/`, `mmlu_cache.json`.

### 11.2 `mf_outputs/` — features (generated, gitignored)

```
mf_outputs/<model_safe>/
  pickles/rank<R>/seed<S>/<concept_safe>/embedding/embedding.pkl
  pickles/rank<R>/seed<S>/<concept_safe>/mlp/layer<N>.pkl
  csvs/…/embedding/{token_features.csv, stats_embed.csv}
  csvs/…/mlp/{token_features.csv, stats_concept_vs_neutral.csv}
  interpretations/…/{embedding,mlp}/{from_activation.csv, from_projection.csv, potential_features.csv}
  cache/rank<R>/seed<S>/<concept_safe>/mlp/mass_norm/*.npz     # coverage-filter cache
  timing.json
```

Built by `utils.get_pipeline_path(outdir, model_safe, file_type, rank, concept_safe,
track, filename, seed)`. **Note the ordering quirk:** `file_type` comes *before* `rank`,
so the tree is `pickles/rank…` not `rank…/pickles`.

### 11.3 `results/` — outputs (generated, gitignored)

```
results/<method>[_ef]/<model_safe>/rank<R>/seed<S>/
  timing.json
  train_<mode>/
    best_embed.csv                 # ember runs only: winning δ per concept
    <concept_safe>/
      hps.csv                      # one row per grid cell visited
      top_hps.csv                  # top-K by harmonic
      top_hps_valid.csv            # top-K re-scored with Alpaca
  test_<mode>/
    final_test_mc.csv              # rolled up across concepts
    final_test_open.csv
    <concept_safe>/relearning.csv  # per-epoch trace
```

`<mode>` is the `--train-eval` value (`mc` or `open`). The **`_ef` suffix** marks
"method + EMBER as a pre-step" (`io.method_dir_name`); plain `ember` never gets it.

### 11.4 Name sanitisation

- `_safe_model_name`: `/` → `_` (`google/gemma-2-2b-it` → `google_gemma-2-2b-it`).
  ⚠️ `evals/baselines.py` has its own `_model_safe_name` that additionally replaces
  `:`, `@`, and spaces — **two near-duplicate functions**, keep them in sync.
- `_safe_concept`: non-word chars → `_`, collapsed, stripped
  (`Harry Potter` → `Harry_Potter`, `COVID-19 pandemic` → `COVID-19_pandemic`).
- The **unsanitised** concept name is what lives in JSON keys, CSV `concept` cells and
  `--concepts` arguments. Sanitised names appear only in paths.

---

## 12. Config reference

`configs/<method>_<model>.yaml` = the method alone.
`configs/<method>_ember_<model>.yaml` = the same method with the EMBER pre-step
(`ember_step.enabled: true`) → writes to `results/<method>_ef/…`.
`<model>` ∈ `{gemma, llama}`. 18 files = 5 methods × 2 models × (plain / `_ef`),
minus the `ember_ember_*` combinations that don't exist.

Model-specific defaults worth knowing: **rank 100 for Gemma, 200 for Llama**;
`selection.ratio_thresh: 2.0` everywhere; `eval.min_mmlu: 0.7`, `eval.max_qa_acc: 0.6`
for non-EMBER methods; `relearning.enabled: true` with 100 paragraphs.

---

## 13. Invariants an agent must not break

1. **Never change the shape of an existing results CSV** without also updating
   `io.*_COLUMNS` and accepting that old runs can no longer resume against it.
2. **Key columns must round-trip through CSV.** If you add a key column, make sure
   `io._cast_key_cell` / `pipeline._hp_key` cast it to the same type on read as on
   write, or resume silently re-runs everything.
3. **Baselines are keyed by set name only** (`baseline_<set>.json`) — they carry no
   language, no concept subset. Anything that changes what a "set" *is* must change
   the cache key, or you will silently compare against the wrong denominator. This is
   the single most important constraint for the cross-lingual work.
4. **Don't re-tune per condition when the question is "does it leak".** The design
   deliberately tunes in English/MC and evaluates the *same* edited model elsewhere.
5. **Feature selection is by `metric_score`/`ratio_abs ≥ ratio_thresh`**, with a
   documented fallback to *all* features when the threshold empties the set. If you
   tighten thresholds, check the logs for `ratio_thresh=… left 0 features`.
6. **`method.apply()` must be idempotent given a preceding `restore()`.** The pipeline
   restores → applies → evaluates for every single cell. A method that accumulates
   state across cells produces silently wrong numbers.
7. **Keep `pipeline.py` method-agnostic.** New behaviour goes in a `Method` hook, not
   in an `if cfg.method == …` branch.
8. **Use `log.info` inside the pipeline**, not `print` — the `[method|concept|stage]`
   prefix is how multi-hour SLURM logs are read. (`evals/` and the phase A/B scripts
   still use `print`; that's the existing split.)

---

## 14. Extension recipes

### Add a new erasure method

1. `ember/erasure/methods/<name>.py`: subclass `Method`, implement `enumerate_hps`,
   `hp_key_columns`, `hp_columns`, `apply`; add `snapshot`/`restore` if you mutate
   weights cheaply, or set `requires_full_reload = True`. End with `register(XMethod())`.
2. Import it in `methods/__init__.py`.
3. `config.py`: add `METHODS += ("<name>",)` and an `XGridConfig` dataclass + a field on
   `RunConfig` + an entry in `load_yaml`'s `sub_classes`.
4. `io.py`: add `X_HP_COLUMNS` and an entry in `hp_columns_for`.
5. Add `configs/<name>_{gemma,llama}.yaml` and `configs/<name>_ember_{gemma,llama}.yaml`.

Nothing in `pipeline.py` changes.

### Add a new metric

Compute it in `evals/orchestrator.evaluate_model_for_mode`, then add the key to the
relevant list in `io.py` (`TRAIN_MC_METRICS` / `TRAIN_OPEN_METRICS` / `ALPACA_METRICS`).
`normalize_metrics_for_mode` reindexes to those lists, so a metric not listed there is
**silently dropped**.

### Add a language axis (the current project) **[planned]**

`archive_gemini/docs/crosslingual_plan.md` is the authoritative spec. In short: thread a `lang="en"`
parameter through four files — `local_datasets.py` (two loaders → `data/qa_lang/
translated/<lang>/`), `evals/baselines.py` (cache key becomes
`baseline_<set>__<lang>.json` for non-English, and MMLU/Alpaca stay `lang="en"`),
`evals/orchestrator.py` (pass through to the set loaders), `erasure/eval.py`
(pass-through) — plus a new `ember/run_crosslingual.py` shaped like `run_eval.py` that
loads the model once, computes per-language pristine baselines, applies the
English-tuned edit once, and loops languages with `eval_mmlu=False, eval_alpaca=False,
precomputed_metrics=<english>, min_mmlu=None, max_qa_acc=None`.

Everything defaults to `"en"` so existing results and caches stay valid.

---

## 15. Gotchas / traps

| # | Trap |
|---|---|
| 1 | **The submodules in `external/` are empty in this checkout** (`git submodule status` shows all four unstaged). `ember/utils.py` imports `factorization.seminmf` from `external/snmf`, so **`import ember.erasure.*` fails outright** without `git submodule update --init --recursive`. This laptop copy is edit-only; runs happen on the cluster. |
| 2 | **`mf_outputs/` and `results/` don't exist here** and are gitignored. Any local test that touches them will fail unless features are downloaded first (`features_source: hf`). |
| 3 | **Gemma embedding scale.** Editing `W_E` without the `sqrt(d_model)` round-trip in `embed_edit._embedding_scale` produces a wrong-magnitude edit that still "runs". |
| 4 | **Gemma `lm_head` untying.** `model_loader.load_hf_model` clones `lm_head.weight` and sets `tie_word_embeddings = False` for Gemma-2 — otherwise editing the input embedding also corrupts the output projection. Don't remove this. |
| 5 | **EMBER ignores the `eval:` block during its grid.** Thresholds are hard-coded in `EMBERMethod.grid_eval_kwargs` (`min_mmlu=0.90`, `min_alpaca=0.9`). Only `alpaca_batch_size` from the YAML has an effect there. |
| 6 | **Zero metrics ≠ measured zero.** Early-stop rows and `skip_eval_for_hp` rows both write literal `0.0` into `efficacy/specificity/harmonic`. |
| 7 | **`harmonic_mean` collapses to 0 on any non-finite or ≤0 input.** A single NaN metric silently produces a 0 aggregate. |
| 8 | **Two `_safe_model_name` implementations** (`ember/utils.py` and `evals/baselines.py::_model_safe_name`) with different rules. Diverging them relocates the baseline cache. |
| 9 | **`prepare_mc_items` raises `AssertionError` if `correct_answer ∉ options`.** The translated data has at least two items with a stray leading space that will crash a cross-lingual run (see `crosslingual_plan.md` §6). |
| 10 | **`config._instantiate` doesn't recurse into nested dataclasses** under `from __future__ import annotations` (field types are strings, so `is_dataclass(f.type)` is False). It works today only because every sub-config is flat and `load_yaml` dispatches the top level explicitly. Don't add a dataclass *inside* a sub-config. |
| 11 | **`run_eval.py` calls private pipeline helpers** (`_compute_all_baselines`, `_resolve_ember_step`, `_run_final_test`). Renaming those breaks it — it's an undeclared dependency. |
| 12 | **Grid-stage rows lack Alpaca columns** for every method except `ember` (`_ALPACA_IN_GRID_METHODS`). Don't rank grid rows by `harmonic_alpaca`. |
| 13 | **Gemini refusals default to "incorrect" / score 0.** On sensitive concepts (Nazism, Pornography) this inflates apparent erasure success. Audit refusal rates. |
| 14 | **`TORCHDYNAMO_DISABLE=1`** is set in `run_erasure.py` before importing torch-adjacent code. Keep it. |
| 15 | **RMU reloads the full model per grid cell** — Llama needs ≥48 GB VRAM and the grid is `3 lr × 4 alpha × 4 steering × 3 settings = 144` cells per concept. Budget accordingly. |

---

## 16. State of this working copy (as of the last commit)

- Branch `main`, three commits: `Initial commit - EMBER` → `after failing on relearning,
  before changing run eval to divide between nodes` → `translated files`.
- **Untracked:** `docs/` (both `crosslingual_plan.md` and this file).
- `external/*` submodules **not initialised** here.
- `docs/references/Main Article.pdf` is the EMBER paper.
- `baseline_mgpu_871643.out` is a leftover SLURM log.
- Cluster working directory (from the `.sbatch` files):
  `/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure`, partition
  `studentkillable`, launched with `uv run python -m ember.…`.
- Cross-lingual work status: **data translated (de/es/fr/ru), plan written, no code
  written.** `data/qa_lang/` is not referenced anywhere in `ember/` yet.

### Running it

```bash
# one-time
git submodule update --init --recursive
pip install -r requirements.txt && pip install -e .
cp .env.example .env        # HF_TOKEN (gated models), GEMINI_API_KEY (judge + interpretation)

# full pipeline
python -m ember.run_erasure --config configs/snmf_ember_gemma.yaml \
    --concepts "Harry Potter" --train-eval mc

# final-test stage only (reuses an existing grid)
python -m ember.run_eval --config configs/ember_gemma.yaml \
    --concepts "Harry Potter" --train-eval mc
```

Environment variables that matter: `HF_TOKEN`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY` /
`GEMINI_API_TOKEN`), `HF_HOME`, `EMBER_DEVICE` (`cpu`/`cuda`, else `device_map="auto"`),
`GEMINI_TEMPERATURE`, `CRISP_SAE_CACHE`, `PYTORCH_CUDA_ALLOC_CONF`.

---

## 17. Where to look first, by task

| You want to… | Start at |
|---|---|
| understand a run end to end | `ember/erasure/pipeline.py` (top docstring + `run()`) |
| understand what a metric means | `ember/evals/orchestrator.py` |
| add/change a hyper-parameter | `ember/erasure/config.py` + the relevant `configs/*.yaml` |
| add an erasure method | `ember/erasure/methods/base.py`, then copy `ember.py` |
| change what gets written | `ember/erasure/io.py` |
| understand the EMBER edit itself | `ember/erasure/embed_edit.py` |
| debug "why did this cell score 0" | early-stop gates in `orchestrator.py`; `skip_eval_for_hp` |
| debug "why did it re-run everything" | `io.load_done_set` + `Method.hp_key_columns` |
| work on the cross-lingual extension | `archive_gemini/docs/crosslingual_plan.md`, then §14 above |
| see a worked example | `demo.ipynb` (Harry Potter, Gemma, EMBER + SNMF) |
