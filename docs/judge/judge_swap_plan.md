# Plan — replace the Gemini judge with a free open-weight judge

**Status:** design, no code written.
**Companion docs:** [`architecture/ARCHITECTURE.md`](../architecture/ARCHITECTURE.md) (§10 evaluation
subsystem, §13 invariants), [`progress.md`](../progress.md) (what has already been run).

---

## 0. Prerequisite answer: is the erasure real, or on the fly?

**It is a real weight edit — but only ever to the in-memory model. Nothing is ever
written back to disk as a checkpoint.**

| Method | What is mutated | Where |
|---|---|---|
| `ember` | `model.get_input_embeddings().weight.data.index_copy_(...)` | `ember/erasure/embed_edit.py::_erase_factored` |
| `snmf` | `mlp.up_proj.weight.data` rows / `mlp.down_proj.weight.data` cols, `index_copy_` | `ember/erasure/mlp_edit.py` |
| `pisces` | TransformerLens `blocks[L].mlp.W_out`, mirrored into HF `down_proj.weight` | `ember/erasure/methods/pisces.py` |
| `rmu` | full-parameter gradient fine-tune | `external/wmdp` via `methods/rmu.py` |
| `crisp` | a LoRA adapter attached to the model | `methods/crisp.py` |

So:

- There is **no forward hook, no runtime masking, no logit post-processing**. The tensors
  themselves change, under `torch.no_grad()`.
- `grep -rn "save_pretrained\|torch.save" ember/ crosslingual/` returns **nothing**. The
  edited model exists only for the life of the process.
- The pipeline's unit of work is `restore() → apply(hp) → evaluate()` per grid cell
  (`ARCHITECTURE.md` §13.6). `snapshot()`/`restore()` keep a CPU clone of just the edited
  tensors (`W_E`, or `W_E` + the MLP weights) and copy them back between cells. `rmu`
  opts out with `requires_full_reload = True` and reloads from the HF cache instead.
- What *is* persisted is the **recipe**, not the weights: the winning δ / layer ranges in
  `results/**/hps.csv`, `top_hps.csv`, `best_embed.csv`. Any later stage (final test,
  relearning, the cross-lingual runs) reloads a pristine model and re-applies the edit
  from those numbers.

That is also why the relearning stress test is meaningful: it fine-tunes the genuinely
edited weights, which is only possible because the edit is in the parameters and not a
decoding-time filter.

**Consequence for this change:** swapping the judge touches *nothing* in the erasure path.
The judge is downstream measurement only.

---

## 1. Where the judge lives today

The judge is used in **three** measurements and **one** offline phase. MC and MMLU are
scored by letter-parsing and never touch a judge.

| Consumer | Judge call | Volume | Affects |
|---|---|---|---|
| Open QA (`qa`, `simdom` in `*_open` modes) | `judge_open_qa` | ~50 items/concept/set | `qa_frac`, `simdom_frac` → `efficacy`, `specificity`, `harmonic` |
| Alpaca coherence | `score_alpaca_instruct` + `score_alpaca_fluency` | 150 items × 2 = 300 calls | `coherence` → `harmonic_alpaca` (the validate + final-test ranking key) |
| Pristine baselines for the two above | same | once per set | every denominator |
| **Phase B** feature interpretation | `SimpleGeminiClient.generate` | ~1 call pair per candidate feature | **which features get erased at all** |

### 1.1 Exact call sites

```
ember/evals/gemini.py                  the client (GeminiEvaluator + 2 exception types + 3 prompts)
ember/evals/orchestrator.py:73         GeminiEvaluator(token_stats=gem_stats)   ← per evaluate_model_for_mode call
ember/evals/baselines.py:174           GeminiEvaluator()                        ← _compute_alpaca_baseline
ember/evals/baselines.py:208           GeminiEvaluator()                        ← _compute_qa_open
ember/evals/open_qa.py:11,16,52        type hint + retry on GeminiBadFinishError
ember/evals/alpaca.py:25,42,82         type hint + retry on GeminiBadFinishError
crosslingual/baseline.py:38,134,174    Optional[GeminiEvaluator] / GeminiEvaluator()
crosslingual/evaluate.py:27,85,123     type hint + retry
crosslingual/run.py:237-245            _LazyEvaluator (built on first use so MC-only runs need no key)
ember/interpret_features.py:103-130    SimpleGeminiClient — a SEPARATE, simpler client (Phase B)
```

### 1.2 The interface that has to be reimplemented

Narrow — this is the good news. Four members total:

```python
class Judge(Protocol):
    token_stats: GeminiTokenStats | None            # set by orchestrator.py
    def judge_open_qa(self, question: str, answer: str, attempted: str) -> bool: ...
    def score_alpaca_instruct(self, instruction: str, completion: str) -> int: ...  # 0..2
    def score_alpaca_fluency(self, completion: str) -> int: ...                     # 0..2
```

Plus two exception types the callers branch on:

- `GeminiBadFinishError` → caller retries 3× (open QA) or 3× with progressive truncation
  `None → 150 → 50` chars (Alpaca), then **defaults the item to incorrect / 0**.
- `GeminiQuotaExceededError` → aborts the whole run.

### 1.3 What does *not* have to change

- **No CSV schema change.** `orchestrator.py:236-239` writes `gemini_prompt_tokens`,
  `gemini_output_tokens`, `gemini_total_tokens`, `gemini_calls` into `metrics`, but
  `grep -n gemini ember/erasure/io.py` is empty — `normalize_metrics_for_mode` reindexes to
  `TRAIN_*_METRICS` / `ALPACA_METRICS` and silently drops them. So `io.py`, resume keys and
  every existing results CSV are untouched. (Invariant §13.1 is safe.)
- **No erasure-path change** (§0).
- **No MC / MMLU change.**

---

## 2. Design decisions

### 2.1 Which model — and the "not the model under erasure" constraint

The request is explicit: the judge must not be the model being erased. The erasure targets
are `google/gemma-2-2b-it` and `meta-llama/Llama-3.1-8B-Instruct`.

There is a stronger reason than self-preference bias to be careful here: **the judge must
be a controlled variable across arms.** The cross-lingual result compares the Gemma arm's
numbers against each other, and the project is also set up to run Llama. If the judge were
"Llama when erasing Gemma, Gemma when erasing Llama", no cross-model comparison would be
interpretable. So pick **one judge family that is neither Gemma nor Llama** and use it
everywhere.

**Recommendation: a Qwen instruct model** (e.g. `Qwen2.5-72B-Instruct` / `Qwen3-32B` class).
It is neither erasure target, it is open-weight, it is served free by several providers, and
it is genuinely multilingual — which matters, because `crosslingual/evaluate.py` feeds the
judge German/Spanish/French/Russian questions and answers under an English rubric. Mistral
is a reasonable second choice. **Gemma is specifically excluded** even though it is free on
the Google AI Studio API, because it is erasure target #1.

### 2.2 Hosted free API vs. local weights

| | Hosted free tier (OpenAI-compatible) | Local `transformers` judge |
|---|---|---|
| Change needed | new client class using `requests` (already installed; `openai` is **not**) | new client + a second model resident |
| Fits the 30-thread callers | yes, unchanged | no — needs a lock + batching |
| GPU cost | none | competes for the same 11 GB RTX 2080 |
| Reproducible | no (provider may rotate the served checkpoint) | yes, pinned |
| Rate limits | yes, and they are the main engineering problem | none |
| Judge quality on ru/de/es/fr | good (70B-class available free) | a model small enough to fit is weak |

**Recommendation: hosted free tier as the default path**, because the GPU budget does not
allow a good local judge. The arithmetic: Gemma-2-2b at bf16 is ~5.2 GB, and
`progress.md` §6.2 records that Alpaca generation already OOMed on this GPU at
`alpaca_batch_size: 50` — the fix was to drop it to 8. There is not 5 GB of headroom for a
4-bit 7B judge. A 4-bit 3B judge would fit but is exactly the case the table's last row
warns about, and the Llama-3.1-8B arm has no headroom at all.

Candidate providers, all OpenAI-compatible `/chat/completions`: **Groq**, **Cerebras**,
**OpenRouter** (`:free` model suffixes), **HF Inference Providers**, **Mistral**'s free
tier. ⚠️ **Verify current model IDs and free-tier limits at implementation time** — these
change frequently and are not worth guessing. Pick based on which one's free RPM can feed
30 threads; if none can, the limiter in §3.2 handles it.

Keep the local path available as an **optional backend** (§4, Phase 5). It is the only way
to get a fully reproducible, key-free, quota-free judge, and it is the right answer if the
project ever gets a bigger GPU.

### 2.3 The thing most likely to produce wrong numbers

**Every metric in this project is a fraction of a pristine baseline, and for open QA and
Alpaca the judge is part of the measuring instrument.** Baselines are cached at
`data/baselines/<model_safe>/baseline_<set>.json`, keyed by set name only —
`ARCHITECTURE.md` §13.3 calls this out as the single most important constraint.

If the judge changes and the cache key does not, then:

```
coherence = HM( alpaca_instr(NEW judge) / alpaca_instr_base(GEMINI),  ... )
```

— a numerator and denominator produced by two different instruments. Silent, plausible-
looking, wrong. Same hazard in `data/crosslingual_baselines/<model_safe>/<concept>_<lang>.json`.

**The judge identity must enter the baseline cache key**, exactly the way the cross-lingual
work put `lang` into it. And only for the judge-dependent sets:

| Set | Judge-dependent? | Key |
|---|---|---|
| `mmlu_train`, `mmlu_test` | no | **unchanged** — existing cache stays valid |
| `qa_*_mc`, `simdom_*_mc` | no | **unchanged** |
| `qa_*_open`, `simdom_*_open` | yes | `baseline_<set>__<judge_id>.json` |
| `alpaca_train`, `alpaca_test` | yes | `baseline_<set>__<judge_id>.json` |

Keeping the MC/MMLU paths byte-identical means the 8 concepts already finished
(`progress.md` §3) do not need their expensive MMLU baselines recomputed.

`<judge_id>` = a short slug of the judge, e.g. `qwen2p5-72b-instruct`. Suffixing rather than
renaming means **the existing Gemini baselines keep working** if you point the config back
at Gemini — no data is orphaned, nothing needs migrating. Legacy Gemini files keep their
un-suffixed names via an explicit back-compat alias.

### 2.4 Blast radius on work already done

From `progress.md`: 8 concepts × 4 configs × 5 languages are complete, all at
`--train-eval mc` / `--qa-kinds mc`.

- **Open QA**: not used in those runs at all (§6.4 — non-English open accuracy was 0.04–0.24,
  too low to be a denominator). A judge swap changes nothing already recorded.
- **Alpaca coherence**: *is* used, in the validate and final-test stages of Phase 1. Those
  `harmonic_alpaca` values were Gemini-judged. Under a new judge they must be recomputed to
  be comparable, which means the validate stage re-runs (15 rows/concept/arm, not the
  144-cell grid — the grid stage has no Alpaca for `snmf`; see `ARCHITECTURE.md` gotcha 12).
  **EMBER is the exception**: its grid *does* include Alpaca and gates on `min_alpaca=0.9`,
  so a judge swap can in principle change EMBER's chosen δ, and δ feeds `snmf_ember`
  through `best_embed.csv`. Decide deliberately whether to re-run the EMBER grids or freeze
  the current δ values as an experimental constant.
- **Phase B (feature interpretation)**: changing this judge changes *which features are
  erased*, which invalidates every downstream result for that concept. **Do not bundle it.**
  Keep `interpret_features.py` on Gemini by default; make its client swappable but
  opt-in via `--judge`, and note that flipping it requires regenerating
  `potential_features.csv` and re-running from the grid up.

### 2.5 Refusal bias — a chance to fix an existing problem

`ARCHITECTURE.md` gotcha 13: a Gemini refusal (SAFETY/RECITATION) defaults the item to
**incorrect / score 0**, which *inflates* apparent erasure success on sensitive concepts
(Nazism, Pornography are in the concept set). An open-weight judge without a
RECITATION filter should refuse far less — a real methodological improvement. Make it
measurable: count refusals/parse failures per run and log the rate, so "the judge failed"
can never again be silently indistinguishable from "the model was wrong".

---

## 3. The design

### 3.1 New module layout

```
ember/evals/judge/
├── __init__.py        get_judge(cfg) factory + re-exports
├── base.py            Judge ABC: the 3 methods + shared prompt constants
│                      + JudgeBadFinishError, JudgeQuotaExceededError
│                      + the response parsers (_parse_open_correct, _parse_rating)
│                      + judge_id() → the cache-key slug
├── gemini.py          GeminiJudge  (today's ember/evals/gemini.py, reparented)
├── openai_compat.py   OpenAICompatJudge — Groq / Cerebras / OpenRouter / HF / Mistral
└── local_hf.py        LocalHFJudge   (Phase 5, optional)
```

`ember/evals/gemini.py` stays as a **shim** re-exporting `GeminiEvaluator` and the two error
names, so nothing breaks on the first commit and `crosslingual/` keeps importing.

Move the three prompt strings (`OPEN_QA_PROMPT`, `ALPACA_INSTRUCT_PROMPT`,
`ALPACA_FLUENCY_PROMPT`) and both parsers into `base.py` **verbatim**. Identical prompts are
what make a judge-vs-judge agreement study (§6) meaningful.

### 3.2 `OpenAICompatJudge`

- One `requests.Session`, POST to `{base_url}/chat/completions`, body
  `{model, messages:[{role:"user",content:prompt}], temperature, max_tokens}`.
  No new dependency (`requests` 2.34 and `httpx` 0.28 are both already in `.venv`).
- **Keep the module-level rate limiter and quota counter as they are.** They are already
  process-wide (`_RATE_LOCK`, `_REQUEST_TIMES`, `_QUOTA_ERROR_COUNT`) and already handle
  exactly the free-tier problem; just make `RATE_LIMIT_PER_MIN` configurable rather than the
  hard-coded 3000, and add a token-per-minute bucket if the chosen provider limits TPM
  (several do, and it binds before RPM on 200-token Alpaca completions).
- Map HTTP 429 / `rate_limit` bodies onto the existing `_is_quota_error` path so the
  `MAX_QUOTA_ERRORS = 20` abort still works. Honour `Retry-After` when present instead of
  the fixed `30 + U(0,5)` sleep.
- Refusals and truncation → raise `JudgeBadFinishError` so the callers' existing
  3×-retry-then-default logic applies unchanged. Map `finish_reason == "length"` and any
  provider refusal onto it.
- Reuse `_append_usage` for the cost log. **Set both prices to 0.0** for a free provider and
  keep logging token counts — the volume number stays useful for rate-limit budgeting even
  when the cost is zero. Do not keep charging NIS for a free model.
- Drop the pre-flight `count_tokens` call (Gemini-specific, and it doubles the request
  count). Take token counts from the response's `usage` block, which every
  OpenAI-compatible provider returns.

### 3.3 Config surface

The judge is chosen by **environment variable**, not YAML. Reason: every one of the 18
config files would otherwise need the same block, and `config.load_yaml` rejects unknown
keys, so a partial rollout would hard-error. Env vars also match how `HF_TOKEN`,
`GEMINI_API_KEY`, `EMBER_DEVICE` and `GEMINI_TEMPERATURE` already work, and `.sbatch`
scripts already `source .env`.

```bash
EMBER_JUDGE=openai_compat            # gemini | openai_compat | local_hf   (default: gemini)
EMBER_JUDGE_MODEL=<provider model id>
EMBER_JUDGE_BASE_URL=https://.../openai/v1
EMBER_JUDGE_API_KEY=...
EMBER_JUDGE_RPM=30                   # free-tier requests/min
EMBER_JUDGE_TPM=                     # optional tokens/min bucket
EMBER_JUDGE_ID=qwen2p5-72b-instruct  # cache-key slug; derived from the model id if unset
EMBER_JUDGE_MAX_WORKERS=8            # caps the 30-thread pools in open_qa/alpaca
```

**Default stays `gemini`** so every existing command reproduces today's behaviour byte for
byte until the env var is set. Add all of these to `.env.example` with comments.

### 3.4 The one construction problem worth flagging

`orchestrator.py:73` builds a **fresh judge per `evaluate_model_for_mode` call**, i.e. per
grid cell. That is free for an HTTP client but pathological for `local_hf` (a model reload
per cell). Make `get_judge()` cache the instance per config at module level and give the
per-call `GeminiTokenStats` its own setter, rather than relying on construction to carry it.
Guard `token_stats` mutation with a lock, since it is written from 30 threads.

### 3.5 Thread count

`open_qa.py` and `alpaca.py` both hard-code `max_workers=30`, sized for Gemini's 3000 RPM.
A free tier will be far below that, and 30 threads against a 30-RPM limit means 29 of them
sit in the limiter's `time.sleep`. Read the pool size from `EMBER_JUDGE_MAX_WORKERS`
(default 30, so nothing changes for Gemini).

---

## 4. Implementation phases

Each phase is independently testable and leaves `main` working.

### Phase 1 — extract the abstraction (no behaviour change)

1. Create `ember/evals/judge/base.py`: the `Judge` ABC, the three prompts, both parsers,
   `JudgeBadFinishError` / `JudgeQuotaExceededError`, the rate limiter, `_append_usage`,
   and `judge_id()`.
2. Move `GeminiEvaluator` → `judge/gemini.py` as `GeminiJudge(Judge)`, keeping
   `GeminiEvaluator = GeminiJudge` as an alias.
3. `judge/__init__.py`: `get_judge()` reading `EMBER_JUDGE` (only `gemini` registered yet),
   module-level instance cache.
4. Turn `ember/evals/gemini.py` into a shim re-exporting the old names, so
   `crosslingual/` and `ember/evals/*` keep importing unchanged.
5. **Verify:** re-run one finished cell and diff the CSV row against the recorded one. It
   must be identical.

### Phase 2 — the judge-aware baseline cache key (the correctness fix)

1. `evals/baselines.py`: add `judge_id: Optional[str] = None` to
   `compute_or_read_baseline`. Build the filename as
   `baseline_{set_name}__{judge_id}.json` **only when** `judge_id` is set *and* the set is
   judge-dependent (`set_name.startswith("alpaca_")` or `set_name.endswith("_open")`);
   otherwise keep `baseline_{set_name}.json`.
2. Back-compat: when `judge_id` is the Gemini slug, fall back to the un-suffixed path if the
   suffixed one is absent, so the existing `baseline_alpaca_*.json` and
   `baseline_qa_test_open.json` files on disk keep being used.
3. Thread `judge_id` from `get_judge()` through `pipeline._compute_all_baselines` and
   `run_eval.py`.
4. Mirror the same suffix rule in `crosslingual/baseline.py::baseline_path` —
   `<concept_safe>_<lang>__<judge_id>.json`, with the same fall-back for the 40 existing
   files. Note that the cross-lingual baselines cover both MC and open in one JSON, so the
   suffix has to apply whenever `open` is in `qa_kinds`; the MC-only runs that produced
   today's files are judge-free and can keep the un-suffixed name.
5. **Verify:** with `EMBER_JUDGE` unset, the resolved paths are exactly today's paths.

### Phase 3 — `OpenAICompatJudge`

1. Write `judge/openai_compat.py` per §3.2; register it in `get_judge()`.
2. Add a standalone smoke script (`scripts/judge_smoke.py`) that runs ~10 fixed
   (question, gold, attempted) triples and ~10 Alpaca completions through the configured
   judge and prints the parsed outputs. **No GPU, no model load.** This is where the
   provider's model IDs and limits get confirmed for real.
3. Handle the boring failures explicitly: empty `choices`, a `content` of `None`,
   markdown-fenced output, and reasoning models that emit a `<think>` block before
   `Rating: [[2]]` — the existing `_parse_rating` fallback `re.search(r"(\d+)")` would
   happily return a digit from the reasoning text. **Tighten the fallback**: strip any
   `<think>...</think>` span first, and prefer the *last* match rather than the first.
4. Make `EMBER_JUDGE_MAX_WORKERS` actually drive the two thread pools.
5. **Verify:** `scripts/judge_smoke.py` gives sane, parseable answers on all 20 items with
   zero parse failures.

### Phase 4 — the guard against judging with the erased model

In `get_judge()`, compare the judge model id against the model under erasure at *family*
level (substring match on `gemma` / `llama` / `qwen` / `mistral`, lower-cased) and **raise**
on a collision, with `EMBER_JUDGE_ALLOW_SELF=1` as an explicit escape hatch for someone who
deliberately wants that ablation. The model under erasure is already in `RunConfig.model_name`;
pass it into the factory.

Also log one line per run — judge backend, model id, `judge_id` slug, erased model — so
every SLURM `.out` records which instrument produced its numbers. Put it through
`erasure/log.py`, not `print` (invariant §13.8).

### Phase 5 — `LocalHFJudge` (optional, and only if a bigger GPU appears)

`judge/local_hf.py` loading a pinned instruct model with `transformers`. Requirements:
serialize all calls behind one lock (or batch them), accept a `device` so it can be pushed
to CPU when the GPU is full, and set `EMBER_JUDGE_MAX_WORKERS=1`. Pin the revision so
`judge_id` really identifies the instrument.

A better companion to this, and useful on its own: a **two-pass / re-judge mode**. The eval
already returns full `records` containing `question` / `gold_answer` / `model_answer` and
`instruction` / `completion`. Dumping those and judging them in a second, GPU-free process
would (a) let the judge use the whole GPU, and (b) allow **re-judging finished runs with a
new judge for free**. `results/cross_lang_test/<lang>/<Concept>/<config>__<wrapper>.json`
(288 files) and `data/baseline_samples/samples_*.json` already have exactly this shape —
see §6.

---

## 5. Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Mixed-instrument metrics** (new-judge numerator ÷ Gemini denominator). Silent and plausible. | Phase 2 cache key. Do it before Phase 3 ships. |
| 2 | Free-tier rate limit turns a 300-call Alpaca pass into hours. | Reuse the existing limiter; cap worker count; measure on one concept before launching the sweep. |
| 3 | Provider rotates the checkpoint behind a model alias → the instrument drifts mid-experiment. | Record the response's `model` field in the usage log; pin an explicit version when offered; prefer one provider for the whole campaign. |
| 4 | A weaker judge is noisier, especially on **ru/de** content under an English rubric. | §6 agreement study, per language. Prefer a 70B-class multilingual judge. |
| 5 | A reasoning/`<think>` model breaks `_parse_rating`'s digit fallback. | Phase 3 step 3: strip think-spans, prefer the last match, count parse failures. |
| 6 | Bundling Phase B (`interpret_features.py`) silently changes *which features are erased*. | Keep it on Gemini by default; separate opt-in flag; re-run from the grid up if flipped. |
| 7 | Free-tier keys leak into the repo. | `.env` only (already gitignored); `.env.example` gets the names with empty values. |
| 8 | A new judge with fewer safety refusals shifts the sensitive concepts (Nazism, Pornography) relative to the rest — a *real* change, not a bug. | Log the refusal rate under both judges and report it; this is an improvement over gotcha 13, but it must be stated in the write-up. |

---

## 6. Validation — the cheap agreement study, first

**Do this before spending any GPU time.** The material is already on disk and needs no model:

| Source | Contents | n |
|---|---|---|
| `data/baseline_samples/samples_alpaca_{train,test}_google_gemma-2-2b-it.json` | `instruction`, `completion`, Gemini's `instruct_score`, `fluency_score` | 2 files |
| `data/baseline_samples/samples_qa_{test}_open_*.json`, `samples_simdom_test_open_*.json` | `question`, `gold_answer`, `model_answer`, Gemini's `is_correct` | 2 files |
| `results/cross_lang_test/<lang>/<Concept>/<config>__<wrapper>.json` | per-question records in all 5 languages | 288 files |

Written at `sample_fraction=1.0`, so these are the **full** record sets, not subsamples.

Procedure:

1. Re-judge those records with the candidate judge, same prompts (§3.1).
2. Report, overall and **per language**:
   - open QA: raw agreement + Cohen's κ against Gemini's `is_correct`;
   - Alpaca: mean-score delta and exact-match rate on the 0/1/2 scale, per axis;
   - parse-failure and refusal rate for each judge.
3. Eyeball every disagreement in a 20-item sample. A judge that is *differently* wrong is
   fine; a judge that is *systematically lenient* shifts `coherence` and therefore the
   chosen hyper-parameters.

Gate: **κ ≥ 0.7 on open QA and a mean Alpaca delta within ±0.15** before adopting. Below
that, try another model rather than proceeding.

Then, and only then:

4. Recompute the judge-dependent baselines under the new `judge_id` (Alpaca train+test; open
   QA only if you intend to run open).
5. Re-run **one** concept's validate stage end to end and compare `harmonic_alpaca` and the
   selected HP row against the Gemini-judged one. If the *selected row* changes, say so
   explicitly in `progress.md` — that is a result, not a bug.
6. Decide on the EMBER grids (§2.4): re-run, or freeze the current δ as a constant.

---

## 7. What to write where

| File | Change |
|---|---|
| `ember/evals/judge/{__init__,base,gemini,openai_compat}.py` | new |
| `ember/evals/gemini.py` | becomes a re-export shim |
| `ember/evals/baselines.py` | `judge_id` param + suffixed path + Gemini fall-back |
| `ember/evals/orchestrator.py` | `get_judge()` instead of `GeminiEvaluator()`; pass `judge_id` to baselines |
| `ember/evals/open_qa.py`, `alpaca.py` | type hints → `Judge`; `max_workers` from env |
| `ember/erasure/pipeline.py`, `ember/run_eval.py` | thread `judge_id` into baseline calls |
| `crosslingual/baseline.py`, `run.py`, `evaluate.py` | `get_judge()`; suffixed baseline path |
| `ember/interpret_features.py` | swappable client, **default unchanged** |
| `scripts/judge_smoke.py`, `scripts/judge_agreement.py` | new, GPU-free |
| `.env.example` | the `EMBER_JUDGE_*` block |
| `docs/architecture/ARCHITECTURE.md` | §10.5 rewritten; §13.3 extended to cover judge identity; gotcha 13 updated |
| `docs/progress.md` | record which judge produced which rows |

Untouched: `ember/erasure/io.py` (no schema change), every `configs/*.yaml`, the whole
erasure path.
