# Progress log

**Last updated:** 2026-10-05
**Status:** Judge model picked via a rigorous agreement study (Qwen2.5-32B-Instruct
clears the adoption gate; 14B does not, on fluency). Pass 1 (generation, all 18
concepts) and the pristine baseline pass are both done. Pass 2 proper (judge the
8-delta grid, pick the best δ per concept) is written and rolling out concept by
concept — 3 of 18 in process as of this update. Live per-concept status:
[`docs/ember_search.md`](ember_search.md).

This file tracks *where things stand right now* — what's done, what's next, why.
It is not a results dump: numbers and detailed write-ups live in the files linked
below, so this page stays short enough to read in one sitting.

---

## 1. History — the Gemini-era run

The project originally ran its cross-lingual erasure experiment (8 of 18
concepts, embedding + MLP erasure, evaluated in 5 languages) using **Gemini**
as the judge for open-ended QA correctness and AlpacaEval coherence scoring —
the same judge the paper itself used.

That entire run (results CSVs, baselines, the per-question records, Gemini's
usage log, and the write-up of what it found) has been moved to
[`archive_gemini/`](../archive_gemini/) at the repo root, untouched. The full
execution record for that era — what was run, what broke, the actual numbers —
is at [`archive_gemini/docs/progress_gemini.md`](../archive_gemini/docs/progress_gemini.md),
and the results summary at
[`archive_gemini/docs/crosslingual_results_summary.md`](../archive_gemini/docs/crosslingual_results_summary.md).
Nothing from that era is deleted — it's kept as a reference point to compare
the new judge's numbers against, and the agreement study below re-judges its
exact saved completions for that reason.

## 2. What we're doing now, and why

Gemini costs real money at the volume this project needs (tens of thousands of
judge calls for a full grid search across 18 concepts), so we're switching to a
**free, local** judge. One constraint: the judge must not be the same model
family as whatever we're erasing (currently Gemma-2-2b-it, later Llama), so it
stays an independent measuring instrument.

Full reasoning and the detailed design are in
[`docs/judge_swap_plan.md`](judge_swap_plan.md). In short:

- **Why decoupled generation + judging ("Pass 1" / "Pass 2"):** the model
  being erased and the judge can't both sit in GPU memory at once on this
  hardware (11 GB RTX 2080 Ti / 12 GB Titan Xp per card). So the plan splits
  every evaluation into (1) generate the model's answers and save them to
  disk, no judge involved, then (2) separately load a judge and score what's
  saved. This also means Pass 2 gets the *entire* GPU budget for the judge,
  since the model under erasure isn't loaded at all at that point.
- **Scope for now:** Gemma only (Llama needs a bigger GPU budget — parked for
  later). One erasure method first — EMBER's own δ (strength) search, since
  that's the one place the judge's score directly decides the result. All 18
  concepts.

## 3. Where things stand

**Done:**
- Archived the Gemini-era results/baselines (see §1).
- Downloaded fresh, complete features for all 18 concepts (Gemma, both
  erasure tracks) from the public features dataset.
- **Pass 1** (`judge_rerun/pass1_ember_delta.py`): for all 18 concepts, ran
  EMBER's 8 candidate deltas, scored MMLU/concept-QA/SimDom-QA (letter-graded,
  no judge), and saved the 150 AlpacaEval completions per delta per concept,
  unscored. Output under `judge_rerun/pass1_outputs/<model>/rank<rank>/seed<seed>/<Concept>/`.
- **Pristine baseline pass** (`judge_rerun/pass1_baseline.py`): the same
  machinery with no edit applied. Computes MMLU (global, 50 items) and the
  150 Alpaca completions once, then loops over all 18 concepts for QA/SimDom
  accuracy (concept-specific item sets). Output under `.../_baseline/`.
  Pristine `mmlu_acc=0.52` matches the archived Gemini-era baseline exactly —
  a good reproducibility check before trusting anything built on top of it.
- **Judge model chosen — see §3.1 for the full writeup.** Qwen2.5-32B-Instruct,
  4-bit (bitsandbytes NF4), confirmed via a rigorous same-completions agreement
  study against Gemini's archived scores.
- The pristine baseline's Alpaca completions are judged (both 14B and 32B, for
  comparison) — `baseline_alpaca_scores__<judge_id>.json` under `.../_baseline/`.
- **Pass 2 proper** (`judge_rerun/pass2_delta_search.py`): judges all 8 deltas'
  saved completions for one concept, computes `harmonic_alpaca`, and picks the
  best δ (`io.pick_best_embed_delta`, same formula the original pipeline uses
  — see §3.2). Per-delta *and* per-item checkpointing, so a timeout or
  preemption only loses the one delta in flight — resubmitting the identical
  `sbatch` command continues from there (see §3.3). Submitted per concept via
  `judge_rerun/run_pass2_delta_search_anygpu.sbatch --concept "<Concept>"`; no
  all-concepts submit script yet (unlike Pass 1's `submit_pass1_all.sh`), so
  each concept is kicked off by hand. **Rollout status, live:**
  [`docs/ember_search.md`](ember_search.md).

**Not done yet:**
- The other 15 of 18 concepts' Pass 2 runs haven't been started.
- A `submit_pass2_all.sh` analogous to Pass 1's doesn't exist yet — worth
  writing once the per-concept workflow (and the multi-user GPU-budget split,
  §3.3) is confirmed stable.
- Nothing beyond EMBER's δ search — SNMF, SNMF+EMBER, validate/final-test
  stages, relearning, and the cross-lingual re-evaluation are all later
  steps, once this first piece is proven end-to-end.
- Cleanup done: `judge_rerun/smoke_qwen_judge*.py` and their `.sbatch` files
  were one-off hardware-validation scripts (do 4-bit Qwen + this GPU even
  work together), superseded by `qwen_judge.py` as the real reusable module —
  deleted 2026-10-06. Also deleted `judge_rerun/run_pass2_delta_search.sbatch`
  (pinned `--gres=gpu:geforce_rtx_2080:3`, which only exists on 2 of the 5
  `studentkillable` nodes and queued behind busy 2080 jobs even when titan
  nodes sat idle); `run_pass2_delta_search_anygpu.sbatch` (`--gres=gpu:3`,
  no model pinned) replaces it and is confirmed working on titan GPUs.

### 3.1 Judge model: which one, and why

**Constraint (judge_swap_plan.md §2.1):** must not be the same family as
either erasure target (Gemma, Llama), and should be genuinely multilingual
for the later cross-lingual re-judging. Qwen2.5-Instruct fits both.

**The real question was size: 14B vs 32B**, both run locally in 4-bit NF4
via bitsandbytes (free, no API key, no rate limit — the local-judge path the
plan had originally deprioritized for lack of GPU headroom, which turned out
to be less of a problem than expected once Pass 2 gets the whole card with no
model-under-erasure loaded). Evaluated with the plan's own §6 agreement
study: re-judge Gemini's *exact* already-scored 150 Alpaca-train completions
(`archive_gemini/data/baseline_samples/samples_alpaca_train_*.json`) with
Qwen, same prompts, compare item-by-item. This is more rigorous than
comparing mean scores across *different* completions (no confound from
generation nondeterminism) — `judge_rerun/agreement_study.py`.

| | Qwen2.5-14B-Instruct | **Qwen2.5-32B-Instruct** | gate (plan §6) |
|---|---|---|---|
| instruct mean_delta | -0.100 | **-0.073** | within ±0.15 |
| instruct exact-match | 91.3% | **92.7%** | "healthy" |
| instruct MAE | 0.100 | **0.073** | — |
| fluency mean_delta | **+0.200 (fails)** | **+0.040** | within ±0.15 |
| fluency exact-match | 61.3% | 64.0% | "healthy" |
| fluency MAE | 0.413 | 0.373 | — |
| fluency prompt used | Qwen-strengthened (see below) | original, verbatim Gemini | — |
| parse failures | 0/150, 0/150 | 0/150, 0/150 | — |

**32B passes the gate on both axes, with the original unmodified prompts.
14B fails on fluency even after a prompt fix**, so 32B is the pick for Pass 2.
Full per-item confusion matrices: `judge_rerun/agreement_study_train.json`
(14B) and `judge_rerun/agreement_study_train__qwen2.5-32b-instruct.json` (32B).

**Caveat worth remembering:** exact-match on fluency is ~62-64% and MAE is
~0.37-0.41 for *both* models — a real per-item scatter that happens to cancel
out in the mean. The gate is about the mean (which is what `harmonic_alpaca`
actually consumes — a mean score across 150 items, not an item-level
pass/fail), so this doesn't block adoption, but it means don't over-trust any
*single* item's score from either judge.

**What the 14B run surfaced along the way (kept here since it explains why
32B's prompt is the plain original, not a tuned one):** the verbatim
`ALPACA_FLUENCY_PROMPT` tells the judge once, in passing, to ignore a
cut-off final sentence (every Pass 1 completion is generated at
`max_new_tokens=200`, so this is common, not an edge case). Qwen2.5-14B
didn't reliably follow that — its own reasoning repeatedly penalized
completions for "ending abruptly" despite the instruction, dragging mean
fluency down (`mean_fluency_score=1.15` vs Gemini's archived `1.613`, from
`baseline_alpaca_scores.json`, the very first baseline-only check before the
rigorous agreement study existed). A more forceful, Qwen-specific restatement
of the same rule (`ALPACA_FLUENCY_PROMPT_QWEN` in `qwen_judge.py`) brought
14B's mean fluency to 1.85 — overshot the other way, still outside the gate.
**32B needed no such tuning**: it follows the plain original instruction
correctly, which is the more trustworthy signal (a prompt that needs
model-specific surgery to behave is a smell).

### 3.2 The delta-selection formula Pass 2 has to replicate

From `ember/erasure/methods/ember.py::grid_eval_kwargs` and
`ember/evals/orchestrator.py` (traced, not reinvented):

```
mmlu_frac = clip((mmlu_acc − 0.25) / (mmlu_base − 0.25), 0, 1)
qa_frac   = clip((qa_acc   − 0.25) / (qa_base   − 0.25), 0, 1)
sim_frac  = clip((simdom_acc − 0.25) / (simdom_base − 0.25), 0, 1)
efficacy    = 1 − qa_frac
specificity = harmonic_mean(mmlu_frac, sim_frac)
harmonic    = harmonic_mean(efficacy, specificity)

alp_instr_frac = min(1, mean_instruct_score / alp_instr_base)   ← needs Qwen-judged baseline
alp_flu_frac   = min(1, mean_fluency_score  / alp_flu_base)
coherence       = harmonic_mean(alp_instr_frac, alp_flu_frac)
harmonic_alpaca = harmonic_mean(efficacy, specificity, coherence)

gate: if mmlu_frac < 0.90 → zero everything for that row
gate: if alp_instr_frac < 0.9 or alp_flu_frac < 0.9 → harmonic_alpaca = 0

best δ = row with max harmonic_alpaca (ties → smaller δ);
         if every row is gated to 0 → fall back to δ=0.0 (no edit beats editing)
```

`mmlu_base`/`qa_base`/`sim_base` (per-concept, letter-parsed, judge-independent)
are already cached in `archive_gemini/data/baselines/google_gemma-2-2b-it/` —
safe to reuse as-is. `alp_instr_base`/`alp_flu_base` could **not** reuse the
archived Gemini numbers (mixing judge instruments silently corrupts the
fraction — `judge_swap_plan.md` §2.3), which is why the pristine baseline's
Alpaca completions were judged fresh with Qwen (§3 above).

### 3.3 Operational findings worth not re-discovering

- **bitsandbytes/Triton/python3-dev gotcha:** `bitsandbytes.nn` eagerly
  imports Triton-based SwitchBack modules (unused for plain NF4 4-bit) that
  JIT-compile a C extension on first import. Needs `python3-dev` headers,
  present on the login node but **not** on this cluster's compute nodes →
  crashes before the judge even loads. Fixed by stubbing that submodule out
  before import (`qwen_judge.py::_stub_out_bnb_triton`).
- **GPU headroom:** a single 11 GB RTX 2080 Ti leaves only ~0.6 GB free after
  loading Qwen2.5-14B in 4-bit — any batched call risks OOM. Fix: shard across
  2 GPUs (`device_map="auto"`, no manual `max_memory` tuning needed) for 14B,
  3 GPUs for 32B (~16-18 GB of weights alone). `qwen_judge.py` also has an
  adaptive batch-halving retry on `CUDA OutOfMemoryError` as a safety net.
- **Page cache, not "re-downloading":** model load time varies wildly by
  node — 18s warm vs. ~68 min cold for the 32B model — because Linux's page
  cache for the weight files is per-node, and each compute node
  (`s-002..s-006`) is independent. This is a **one-time, per-node** tax, not
  a repeated one: it does not multiply with the number of concepts judged in
  a job that keeps the model loaded across concepts. It also doesn't explain
  the bigger problem below — even a perfectly warm cache barely moves a
  multi-hour judging job.
- **The real bottleneck is GPU compute time, not I/O.** Measured throughput:
  Qwen2.5-14B ≈ 12-13s/item, Qwen2.5-32B ≈ 28-32s/item (batched). One
  concept's full 8-delta Alpaca grid is 2,400 judge calls → **~8.3h (14B) or
  ~19.3h (32B) of judging per concept**, independent of caching. The
  `studentkillable` partition caps a single job at 24h wall time
  (`MaxTime=1-00:00:00`), and jobs on it can be preempted
  (`PreemptMode=REQUEUE`) — so running several concepts sequentially in one
  job (to amortize one model load across them) needs checkpointing *across*
  concepts/deltas, not just within a judging pass, or it needs to be split
  into more, smaller per-concept jobs. Full grid volume, corrected (the
  pristine baseline is global, not per-concept, so it's not ×18):
  `18 concepts × 8 deltas × 150 items × 2 axes = 43,200 calls` (+300 for the
  baseline, already done).
- **Running Pass 2 across two cluster accounts to split the GPU budget:**
  `judge_rerun/run_pass2_delta_search_anygpu.sbatch` hardcodes `UV_CACHE_DIR` and
  `HF_HOME` to paths under the original user's home dir, which are
  read-only for a second user — a collaborator's `sbatch` fails at `uv`
  cache init (`Permission denied`) before the job even starts. Fix is either
  (a) the second user exports their own `UV_CACHE_DIR`/`HF_HOME` before
  calling `sbatch` (Slurm exports the submitting shell's environment by
  default, so the script's `${VAR:-default}` picks that up) — simplest, but
  costs that user a fresh ~62GB download of the Qwen2.5-32B-Instruct judge
  weights; or (b) `chmod -R o+rwX` the original user's `.cache/uv` and
  `.cache/huggingface` to share the already-downloaded weights — this
  filesystem has no ACL support (`setfacl` → "Operation not supported"), so
  there's no tighter-scoped way to share than opening to everyone with
  filesystem access, same breadth as the repo's own `chmod 777`.
- **Per-user GPU cap on `studentkillable`:** the partition's QoS
  (`gpustudentkill`) limits each user to `MaxTRESPerUser=gres/gpu=12`
  running at once, with no separate cap on job count. Since each Pass 2 job
  requests 3 GPUs, that's **up to 4 concurrent Pass 2 jobs per user** —
  submitting more just queues them `PENDING` until GPUs free up, rather than
  failing. Two users each running near that cap can have ~8 concepts'
  grids in flight simultaneously. Each concept should go to only one user's
  job at a time — two jobs judging the same concept would race on the same
  checkpoint files.

## 4. Code map (new code only)

| Path | What |
|---|---|
| `judge_rerun/pass1_ember_delta.py` | Pass 1: generate + letter-graded scoring, 8 deltas, one concept |
| `judge_rerun/pass1_baseline.py` | pristine-model counterpart of the above: no edit, MMLU+Alpaca once, QA/SimDom per concept |
| `judge_rerun/run_pass1.sbatch`, `run_pass1_baseline.sbatch` | SLURM wrappers for the two above |
| `judge_rerun/submit_pass1_all.sh` | submits one Pass 1 job per concept, all 18 |
| `judge_rerun/qwen_judge.py` | reusable `QwenJudge`: batched generation, OOM-halving retry, resume-from-checkpoint, the Qwen-tuned fluency prompt |
| `judge_rerun/pass2_judge_baseline.py` | judges the pristine baseline's Alpaca completions with Qwen (any size via `--judge-model`) |
| `judge_rerun/pass2_delta_search.py` | Pass 2 proper: judges one concept's 8-delta grid, picks the best δ (see §3.2) |
| `judge_rerun/run_pass2_delta_search_anygpu.sbatch` | SLURM wrapper for the above, one concept per job, 3 GPUs (any model), 24h |
| `judge_rerun/agreement_study.py` | the §3.1 agreement study: re-judges Gemini's exact archived completions, item-by-item comparison |
| `judge_rerun/run_judge_baseline.sbatch`, `run_judge_baseline_32b.sbatch`, `run_agreement_study.sbatch`, `run_agreement_study_32b.sbatch` | SLURM wrappers for the two scripts above, at both judge sizes |
| `judge_rerun/pass1_outputs/` | Pass 1 + baseline output: `grid_raw.csv` / `baseline_raw.csv`, saved completions, judged scores, per concept |
| `judge_rerun/pass2_outputs/` | Pass 2 output: per-concept `delta_metrics__<judge_id>.csv` and `best_delta__<judge_id>.json` |
| `docs/judge_swap_plan.md` | the full plan and reasoning behind all of the above |
| `docs/ember_search.md` | live, hand-updated per-concept Pass 2 status (who's running what job on which concept) |

Everything under `ember/` (the paper's own reference implementation) and
`crosslingual/` (the original language-leakage experiment code) is untouched —
`judge_rerun/` only imports from them, never edits them.
