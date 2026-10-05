# Last step log

**Goal:** Smoke-test `google/gemma-4-12B-it` (already cached locally) as a candidate
judge for Gemma-2-2b-it evaluation, per `judge_rerun/smoke_gemma_judge.py`, and see
how it compares to the already-adopted Qwen2.5-32B-Instruct judge (`progress.md`
§3.1). First attempt (job 978320) crashed: the model's `config.json` reports
`model_type: "gemma4_unified_audio"` (a multimodal audio+text model, not plain
text), and the project's pinned `transformers==4.53.3` (July 2025) predates that
tokenizer schema (`extra_special_tokens` as a list, not a dict) and can't even
load the tokenizer.

**Plan:** build an isolated venv (not touching the project's `.venv`/`uv.lock`)
with a newer `transformers`/`tokenizers`, keeping torch/accelerate/bitsandbytes
pinned to the project's existing versions, then rerun the smoke test through it.

**NOTE — conflicting edit found 2026-10-05:** between this agent's read and
write of this file, its content changed to a version claiming transformers
5.18.0 "is noise" and PyPI's real ceiling is 5.9.0, and that no release
contains the Gemma4 classes. **That claim is incorrect** — directly
re-verified below with fresh commands (`pip index versions transformers`,
actual imports) in this same session, right after finding that edit. Keeping
the real, re-verified status here. If another session/agent is also working
this file concurrently, that's worth the user's attention — two writers on
one file will keep stomping on each other.

**Status:** step 1 fully done. `gemma4_unified` is registered directly in
`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` under transformers 5.18.0 (confirmed via
the registry dict, not a full CPU materialization — building a 12B+ omni
model's random-init weights on CPU timed out at 50s and isn't a
representative test anyway; real loading happens 4-bit on GPU in the sbatch
job). So `AutoModelForCausalLM.from_pretrained(...)` — exactly what
`qwen_judge.py` already does — should work unmodified; no loader code
changes needed. Proceeding to step 2.

## Steps

1. [x] Create an isolated venv (`judge_rerun/.venv_gemma4/`, not the project's
   `.venv`/`uv.lock`). Installed: torch==2.7.1, bitsandbytes==0.49.2,
   sentencepiece==0.2.0, accelerate (latest compatible),
   google-generativeai==0.8.5 (for `ember.evals.gemini`'s prompt constants),
   and `transformers==5.18.0` (see finding below for why — nothing in the
   4.x line works).
   - Re-verified directly, just now: `pip index versions transformers` lists
     5.18.0 as the real latest on PyPI (full version list obtained, 5.18.0 at
     the top, "LATEST: 5.18.0"). It is installed in this venv right now.
   - `transformers/models/` in this installed copy contains `gemma4`,
     `gemma4_assistant`, `gemma4_unified`, `gemma4_unified_assistant` folders.
     `from transformers import Gemma4UnifiedForConditionalGeneration` succeeds.
   - `AutoTokenizer.from_pretrained('google/gemma-4-12B-it')` succeeds under
     5.18.0 → resolves to `GemmaTokenizer`, vocab_size 262144. This is what
     failed under the project's pinned 4.53.3 (and still fails under 4.57.6,
     the latest 4.x release) with `AttributeError: 'list' object has no
     attribute 'keys'` — the tokenizer config's `extra_special_tokens` schema
     needs transformers 5.x, full stop.
   - **Second finding, independent of the version problem:** Gemma-4-12B-it's
     chat template emits a `<|channel>thought<channel|>` block before the
     real answer — it's a reasoning model. This is exactly the hazard
     `judge_swap_plan.md` risk #5 flags for `<think>`-style models: the
     rating-parser's digit fallback can grab a number from the reasoning
     text instead of the real `Rating: [[n]]`. Must check for this
     specifically once real generations exist.
   - **Third finding:** `config.json` shows `architectures:
     ["Gemma4UnifiedForConditionalGeneration"]`, `model_type: "gemma4_unified"`,
     with nested `text_config` / `vision_config` / `audio_config` blocks —
     this is a genuinely omni-modal (text+image+audio+video) checkpoint, not
     a text-only instruct LLM the way Qwen2.5-32B-Instruct is. `qwen_judge.py`
     (which `smoke_gemma_judge.py` reuses) assumes plain `AutoModelForCausalLM`
     + a text chat template — checking now whether that mapping still works
     for this config, or whether the omni class needs a different loader
     before step 2 can run as-is.
2. [x] Submitted `judge_rerun/run_smoke_gemma.sbatch` (edited to call
   `judge_rerun/.venv_gemma4/bin/python -m judge_rerun.smoke_gemma_judge`
   directly, with `PYTHONPATH` set to the repo root, instead of `uv run`) —
   job 978398, `studentkillable`, 1x RTX 2080 Ti. Checkpoint on disk is 23GB
   (bf16-ish for the full omni model); 4-bit should land around ~6GB, same
   ballpark as Qwen2.5-14B which already fits on one of these cards per
   `progress.md` §3.3, so one GPU should be enough.
3. [ ] Confirm the job's output: does `google/gemma-4-12B-it` load, follow the
   `Rating: [[n]]` format (watching for the reasoning-block parse hazard
   above), and produce parseable scores on the sample baseline items — and
   how those compare to Gemini's archived scores / Qwen2.5-32B's agreement
   numbers.

## A second, independent job is also running this (user confirmed, 2026-10-05)

Job **978371** (`judge_rerun/run_gemma4_judge_eval.sbatch`, not created by this
agent — already on disk when this session started) is running the real,
rigorous version of this same question in parallel on the same node (s-004,
8 GPUs, no contention): the full 150-item agreement study against Gemini's
archived scores, `pass2_judge_baseline` on the real pristine completions, and
`judge_sensitivity` (clean vs. scrambled/mismatched text). It uses its own
separate venv (`/home/morg/NLP_2526b/semyong/.venvs/gemma4_judge`), distinct
from this session's `judge_rerun/.venv_gemma4`. This is almost certainly the
source of the earlier conflicting edit to this file.

**Decision (user, 2026-10-05): keep both running for now.** Check this
session's smoke test (978398) first; decide what to do with 978371
afterward based on those results. Not cancelling either job.

## Job 978398 result: timed out, zero output — time limit too short, not a bug

978398 was killed at the 20-minute `--time` limit with **no output at all**,
not even the very first `print(... flush=True)` right after imports (before
any model loading). Isolated the imports (`torch`, `ember.evals.gemini`,
`judge_rerun.qwen_judge`) and timed them individually: ~11s total — but that
test ran on `c-010` (this interactive session's node), which already has a
warm page cache from everything run here so far this conversation. The
actual job ran on `s-004`, reading this brand-new venv's site-packages
*cold* over networked home storage, likely while job 978371 was
simultaneously hammering the same storage with its 23GB checkpoint read
(its own log showed highly variable per-shard load times, up to 70s/shard,
consistent with I/O contention). Conclusion: not an import bug or a loader
bug, just an underestimated time budget for a cold cache + shared I/O.

**Action:** bumped `--time` in `run_smoke_gemma.sbatch` from 20min → 2h,
resubmitted as **job 978439**.

## Job 978439 result: model loaded fine, crashed on a cosmetic print line

Confirmed the cold-cache theory: this time all 677 weight shards loaded in
**~8 seconds** (cache now warm from the first attempt) — model loading is
not the bottleneck once the cache is warm. It crashed one line later:
`smoke_gemma_judge.py:75` assumed `judge.model.hf_device_map` always exists
(true for Qwen, which `device_map="auto"` shards across 2-3 GPUs). Gemma-4
fit on a single RTX 2080 Ti, so accelerate never set that attribute on the
model object → `AttributeError`. Not a model-loading or judge-correctness
problem, just a diagnostic print written assuming multi-GPU sharding.

**Action:** made the `hf_device_map` print defensive (`getattr(...,
None)`), resubmitted as **job 978466**.

## Job 978466 result: runs cleanly, format is perfect, calibration is not

Loaded in 19s warm-cache, 7.68GB weights / 8.77GB peak — fits comfortably on
one 11GB 2080 Ti. **10/10 strict `Rating: [[n]]` format on both axes, 0 parse
failures.** No sign of the `<|channel>thought<channel|>` reasoning block
(risk flagged in step 1) leaking into the rating.

**But item-level agreement with Gemini is poor on this sample.** These 10
items were deliberately the rarest-score cells (to test discrimination, not
just confirm easy agreement):

| axis | gemini scores | gemma-4-12b scores |
|---|---|---|
| instruct | 1,1,2,2,2,2,2,2,2,2 | 2,2,2,2,2,2,2,2,2,2 |
| fluency | 1,2,0,0,1,1,1,2,2,2 | 2,2,2,0,2,2,2,2,2,2 |

Gemma-4 gave a 2 on every item but one, regardless of Gemini's score —
including one fluency item (a bare regex string) where its own written
explanation says *"it does not possess 'fluency' in a linguistic sense"*
and then rates it `[[2]]` anyway. n=10 and deliberately skewed toward hard
cases, so this isn't a real agreement statistic (unlike Qwen2.5-32B's
92.7%/64.0% exact-match on the full random 150-item set, `progress.md`
§3.1) — but the pattern (near-universal ceiling score, including a
self-contradicting case) is a real signal, not sampling noise.

**Status: smoke test complete.** Mechanically Gemma-4-12B-it works fine as a
judge (loads, fits, formats correctly). Whether it's *usable* as a judge is
now a calibration question, not a plumbing question — that's what job
978371's full 150-item agreement study + sensitivity check will quantify.
Decision on 978371 (and on Gemma-4 as a candidate at all, on top of the
self-family exclusion concern from the original design discussion) pending
user review of those results.
