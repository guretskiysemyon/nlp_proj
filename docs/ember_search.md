# EMBER Pass 2 delta search — status by concept

Tracks `judge_rerun/pass2_delta_search.py` progress (judge the candidate
deltas' saved Alpaca completions, pick the best delta) for each of the 18
concepts. Updated 2026-10-08.

## Delta-selection policy (updated 2026-10-08)

Judging all 8 deltas on the full 150 saved Alpaca completions is the cost
driver (~2.5h/delta on the rtx_2080 nodes). The policy is now two-staged to
cut that ~8x, without changing any of the underlying formulas:

1. **Fast screen (judge-free).** `grid_raw.csv`'s mmlu/qa/simdom accuracies
   already give every delta's `efficacy`/`specificity`/`harmonic` with zero
   judge calls. Rank the deltas that pass `min_mmlu` by that fast `harmonic`
   and keep only the top 3.
2. **Judge, cheaply.** Qwen-judge only those 3 deltas, and only their first
   50 saved Alpaca completions (`idx < 50` — the saved completion order is
   fixed across every delta/concept via `load_alpaca_indices("train")`, so
   "first 50" is the same 50 questions everywhere).
3. **Pick.** Same `harmonic_alpaca` formula and `io.pick_best_embed_delta`
   tie-break as before, just computed from 3×50 instead of 8×150.

This is implemented as `--top-k`/`--alpaca-n` flags on
`pass2_delta_search.py` (defaults 3/50; pass `--top-k 8 --alpaca-n 150` to
recover the exact original behavior), so it's what new `sbatch` submissions
use by default now. For a concept Pass 2 had **already fully judged** under
the old (8×150) policy, `judge_rerun/pass2_recompute_topk.py` re-derives the
new-policy pick from the cached per-item judge scores already on disk — no
GPU, no judge calls, and it never overwrites the original
`_rows__`/`alpaca_judged__`/`best_delta__` files (writes
`best_delta_topk__{judge}.json` / `delta_metrics_topk__{judge}.csv`
alongside them instead). Sanity check so far: Harry Potter's top-3/n=50 pick
(delta=10.0) matches its original full-grid pick exactly.

**SNMF**: the same top-3-by-fast-metric + 50-question-judge policy should
apply there too once it gets an analogous decoupled pass1/pass2 judge split
— SNMF's grid search currently still runs inline through
`ember/run_erasure.py`/`ember/erasure/methods/snmf.py`, not through
`judge_rerun`, so there's no script to point `--top-k`/`--alpaca-n` at yet.
That's a separate, not-yet-started task.

| Concept | Status | Person |
|---|---| --- |
| Ancient Rome | Finished | Mary |
| Artificial intelligence | Finished | Mary |
| Baseball | Finished | Mary |
| COVID-19 pandemic | Finished | Mary |
| Cannabis | Finished | Mary |
| Culture of Greece | 999298 | Mary |
| Gambling | Finished | Ori |
| Golf | - |  |
| Gun | Finished | Ori |
| Halloween | in run | Ori |
| Harry Potter | Finished | Semyon |
| Heroin | - |  |
| Nazism | - |  |
| Pornography | - |  |
| Republic of Ireland | Finished | Ori |
| Uranium | Finished | Ori |
| Valentine's Day | Finished | Ori |
| World War II | - | | sbatch judge_rerun/run_pass2_delta_search_anygpu.sbatch --concept "Culture of Greece" "Golf" "Heroin" "Nazism" "Pornography" "World War II"
Submitted batch job 999320

## How to submit

Submit one concept (fast screen to top-3, then judges only those 3 on their
first 50 Alpaca questions by default; checkpoints per delta, so a timeout or
preemption just needs the identical command resubmitted to continue):

```
sbatch judge_rerun/run_pass2_delta_search_anygpu.sbatch --concept "<Concept Name>"
```

e.g.:

```
sbatch judge_rerun/run_pass2_delta_search_anygpu.sbatch --concept "World War II"
```

To run the original full 8-delta/150-question grid instead, add
`--top-k 8 --alpaca-n 150`.

**Batching several concepts into one job.** `--concept` takes multiple
values and runs them back-to-back in the same process, loading the Qwen
judge only once. The judge load itself is cheap (~20-30s on the rtx_2080
nodes, confirmed from the Halloween run's log) -- the real cost this avoids
is SLURM's queue wait for a free `s-004`/`s-005` slot, which `sacct` has
shown running 15-18h when both pinned nodes are busy. A concept already
finished (its `best_delta__*.json` exists) is skipped immediately with no
GPU touched, so a timed-out multi-concept job can be resubmitted with the
identical command and will only resume the unfinished ones:

```
sbatch judge_rerun/run_pass2_delta_search_anygpu.sbatch \
    --concept "Culture of Greece" "Golf" "Heroin" "Nazism" "Pornography" "World War II"
```

At ~3h/concept under the default top-3/n=50 policy, all 6 remaining
concepts fit comfortably inside the sbatch script's existing
`--time=24:00:00` limit in a single job.

Despite its name, `run_pass2_delta_search_anygpu.sbatch` is pinned to
`--gres=gpu:geforce_rtx_2080:3`, i.e. only `s-004`/`s-005`. Jobs may queue
while the titan nodes (`s-002`/`s-003`/`s-006`) are idle — that's intended.
Unpinned runs on titan (jobs 981330/981331/981347) did not work: on TITAN Xp
the Qwen-32B judge ran ~1190 s/batch vs ~30 s on 2080 Ti (likely the bf16
4-bit compute dtype, which Pascal lacks), and on `s-002` CUDA failed to
initialize so the jobs silently ran on CPU. All three hit the 24h limit without
finishing a single delta. An fp16 variant for titan is to be tested separately.


## multi language add on
run eg.
python -m ember.train_mf_features --concepts "World War II" --languages en+ru+de+es+fr