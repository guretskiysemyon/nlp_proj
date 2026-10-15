# Ember — delta search results (Gemma-2-2b-it, rank 100, seed 42)

Everything the EMBER delta search produced lives in the two folders below.
`archive/` is history only — nothing in it is needed to use the results.

| Folder | What it is |
|---|---|
| `pass1_outputs/` | Pass 1 (generation + letter-graded scores). Per concept: `grid_raw.csv` and `alpaca_completions/delta_*.json` (8 deltas x 150 answers). `_baseline/` = the pristine, unedited model. |
| `pass2_outputs/` | **Final** Pass 2: the best delta per concept, all 18 concepts, one policy. |
| `archive/` | Superseded runs, kept untouched (see the last section). |

## Where the answer is

`pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv` — one row per concept.
Each concept folder next to it (`<Concept>/`) holds exactly three files, judge = Qwen2.5-32B-Instruct:

- `best_delta__qwen2.5-32b-instruct.json` — the winning delta (`best_delta_embed`)
- `delta_metrics__qwen2.5-32b-instruct.csv` — the 8-row table behind that choice
- `alpaca_judged__qwen2.5-32b-instruct.json` — the per-question judge scores (3 deltas x 50 questions)

## The policy every result follows (top-3 x 50)

1. Rank the 8 deltas that pass the MMLU gate by the judge-free `harmonic` (from `grid_raw.csv`); keep the top 3.
2. Judge only those 3, and only on the first 50 saved Alpaca questions (`idx` 0-49; identical questions for every delta and concept).
3. Pick the max `harmonic_alpaca` (`ember.erasure.io.pick_best_embed_delta`, ties to the smaller delta).

Details and formulas: `docs/Part 1/ember_search.md`, `docs/progress.md` §3.2.

## How the 18 results were produced (`method` column in the summary)

- `judged` (6): Culture of Greece, World War II, Golf, Heroin, Nazism, Pornography — judged directly under top-3 x 50.
- `recomputed` (12): the rest had already been judged on the old full grid (8 deltas x 150 questions). Their top-3 x 50 result was
  re-derived from those cached scores by `judge_rerun/pass2_recompute_topk.py` — no new judge calls. The folder's
  `alpaca_judged__` file is the exact subset (top-3 deltas, questions 0-49) that produced the numbers in its CSV.
- Two picks differ from the old full-grid pick: **Baseball 200 -> 100** and **Halloween 5 -> 10**. Both are near-ties
  (identical judge-free score; the Alpaca term differs by under 0.001), so treat either delta as defensible.
  `full_grid_best_delta` / `same_as_full_grid` in the summary show the old pick for the 12 recomputed concepts.

## Running more

`pass2_delta_search.py`, `pass1_ember_delta.py`, `pass1_baseline.py` and `pass2_judge_baseline.py` now default to
`pass1_outputs/` and `pass2_outputs/` in this folder, so no `--out-dir` is needed. This tree is world-writable so
every teammate can write here. A concept whose `best_delta__*.json` already exists is skipped.

## archive/

| Folder | Was | Contains |
|---|---|---|
| `pass2_outputs_original/` | `judge_rerun/pass2_outputs/` | the old full 8x150 results (12 concepts), the first direct top-3 x 50 runs, and stale partial checkpoints |
| `pass2_outputs_topk_recompute/` | `judge_rerun/pass2_outputs_topk/` | raw output of `pass2_recompute_topk.py` (`*_topk__*` files) |
| `pass2_outputs_ori/` | `judge_rerun/pass2_outputs_ori/` | Ori's side-folder runs (Golf, Heroin, Nazism, Pornography). This is a verified copy: the original is owned by `oripetel` (mode 755), so only Ori can delete it |
| `pass1_outputs_superseded/` | `_baseline/` files of `pass1_outputs/` | baseline scores from judges that were evaluated and not adopted (Qwen2.5-14B, Gemma-4-12B) |
| `slurm_logs/` | `pass2_delta_*.out` in the repo root | Slurm logs of the pass-2 jobs |

Note: new Slurm logs are still written to the directory you run `sbatch` from (the `#SBATCH --output` line).
