# EMBER Pass 2 delta search — status by concept

Tracks `judge_rerun/pass2_delta_search.py` progress (judge the candidate
deltas' saved Alpaca completions, pick the best delta) for each of the 18
concepts. Updated 2026-10-09.

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
`_rows__`/`alpaca_judged__`/`best_delta__` files (it writes
`best_delta_topk__{judge}.json` / `delta_metrics_topk__{judge}.csv`
alongside them instead). It has been run for all 12 such concepts (2026-10-09)
and the results promoted into `judge_rerun/Ember/pass2_outputs/`; 10 of the 12
picks equal the old full-grid pick (e.g. Harry Potter, delta=10.0), see the
table below for the two that differ.

**SNMF**: the same top-3 + 50-question policy applies there. Pass 1
(`judge_rerun/pass1_snmf_top3.py`) is done for all 18 concepts, and Pass 2
(`judge_rerun/pass2_snmf_judge.py`) for 15 (2026-10-10). Per-concept status is
in `docs/snmf_search.md`.

Final results for all 18 concepts: `judge_rerun/Ember/pass2_outputs/` — start with
`google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv` (layout and policy: `judge_rerun/Ember/README.md`).
"Best δ" below is the winner under the top-3 / n=50 policy.

| Concept | Status | Best δ | Person |
|---|---|---|---|
| Ancient Rome | Finished (re-derived from full grid) | 200 | Mary |
| Artificial intelligence | Finished (re-derived from full grid) | 100 | Mary |
| Baseball | Finished (re-derived from full grid) | 100 (old full-grid pick: 200) | Mary |
| COVID-19 pandemic | Finished (re-derived from full grid) | 50 | Mary |
| Cannabis | Finished (re-derived from full grid) | 200 | Mary |
| Culture of Greece | Finished | 200 | Mary |
| Gambling | Finished (re-derived from full grid) | 100 | Ori |
| Golf | Finished | 10 | Ori |
| Gun | Finished (re-derived from full grid) | 100 | Ori |
| Halloween | Finished (re-derived from full grid) | 10 (old full-grid pick: 5) | Ori |
| Harry Potter | Finished (re-derived from full grid) | 10 | Semyon |
| Heroin | Finished | 100 | Ori |
| Nazism | Finished | 100 | Ori |
| Pornography | Finished | 5 | Ori |
| Republic of Ireland | Finished (re-derived from full grid) | 100 | Ori |
| Uranium | Finished (re-derived from full grid) | 100 | Ori |
| Valentine's Day | Finished (re-derived from full grid) | 200 | Ori |
| World War II | Finished | 5 | Ori |

**Provenance (2026-10-09).** Six concepts were judged directly under top-3/n=50: Culture of Greece (Mary, job 999320),
World War II (job 1001318), and Golf / Heroin / Nazism / Pornography (Ori; jobs 1001317 / 1000902 / 1000903 / 1000904).
The other twelve had already been judged on the old 8×150 grid, so their top-3/n=50 result was re-derived from the cached
per-item scores with `pass2_recompute_topk.py` (Halloween's was the last one missing). Ten of those twelve picks equal the old
full-grid pick; **Baseball (200 → 100)** and **Halloween (5 → 10)** differ, and both are near-ties (identical judge-free
score, Alpaca term within 0.001).

**Permissions (resolved 2026-10-09).** The standard output dirs for Golf / Heroin / Nazism / Pornography were owned by
`semyong` with mode 755, so nobody else could write there: Golf failed on this twice (Mary's job 993617, then her multi-concept job 999320, which
had finished Culture of Greece and died at Golf with a `PermissionError` before reaching the rest), and Ori's four
concepts had to be run with `--out-dir …/pass2_outputs_ori`. Everything is now consolidated under
`judge_rerun/Ember/` (a world-writable tree), so that workaround is no longer needed; the old side folders
(`pass2_outputs_ori`, `pass2_outputs_topk`) and the full-grid originals are kept in `judge_rerun/Ember/archive/`.
(The original `judge_rerun/pass2_outputs_ori/` is owned by `oripetel`, so only Ori can delete it.)

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

Inputs and outputs default to `judge_rerun/Ember/pass1_outputs/` (read) and
`judge_rerun/Ember/pass2_outputs/` (written); that tree is world-writable, so
no `--out-dir` is needed. A concept whose `best_delta__*.json` already exists
there is skipped — add `--overwrite` (or use another `--out-dir`) to redo it.

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

At ~3h/concept under the default top-3/n=50 policy, about 6 concepts fit
comfortably inside the sbatch script's existing `--time=24:00:00` limit in a
single job (the example above is the batch Mary submitted on 2026-10-08).

Despite its name, `run_pass2_delta_search_anygpu.sbatch` is pinned to
`--gres=gpu:geforce_rtx_2080:3`, i.e. only `s-004`/`s-005`. Jobs may queue
while the titan nodes (`s-002`/`s-003`/`s-006`) are idle — that's intended.
Unpinned runs on titan (jobs 981330/981331/981347) did not work: on TITAN Xp
the Qwen-32B judge ran ~1190 s/batch vs ~30 s on 2080 Ti (likely the bf16
4-bit compute dtype, which Pascal lacks), and on `s-002` CUDA failed to
initialize so the jobs silently ran on CPU. All three hit the 24h limit without
finishing a single delta. An fp16 variant for titan was to be tested separately;
the Titan diagnostic/benchmark scripts were removed from `judge_rerun/` on 2026-10-09
(still in git history, commit `7f86c2e`).


## multi language add on
run eg.
sbatch run_train_mf_features.sbatch --concepts "World War II" --languages en+ru+de+es+fr


batch job 1010008