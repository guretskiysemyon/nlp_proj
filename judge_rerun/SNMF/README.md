# SNMF — HP search results (Gemma-2-2b-it, rank 100, seed 42)

Everything the SNMF hyperparameter search produced lives in the two folders below.
`archive/` is history only — nothing in it is needed to use the results.
Finished for all 18 concepts on 2026-10-10.

| Folder | What it is |
|---|---|
| `pass1_outputs/` | Pass 1 (no judge). Per concept: `top3_hps.json` (the 3 shortlisted grid cells with their judge-free scores) and `alpaca_completions/cell_0..2.json` (50 Alpaca answers each). Ancient Rome also has `screen/` (the loop screen, see below). |
| `pass2_outputs/` | Pass 2 with the original 4-token MC scoring: the best HP cell per concept, all 18 concepts, judged by Qwen2.5-32B-Instruct. |
| `rescore_mc/` | 2026-10-15 MC scoring repair (`judge_rerun/rescore_snmf_grid.py`): per concept `records/cell_<row>.json` (per-question answers), `grid_rescored.csv` (re-scored cells, with the old grid values and the determinism control) and `compare.json` (old vs new top 3). |
| `pass2_rescored/` | **Final** picks after the repair (`judge_rerun/pass2_snmf_rescored.py`): `best_hps_summary.csv` plus per concept `best_hp__…json`, `cell_metrics__…csv`; COVID-19 also `alpaca_completions/row_128.json` and its Qwen scores. Same cells as `pass2_outputs/` for all 18 concepts. |
| `archive/` | Superseded runs and Slurm logs (see the last section). |

The 144-cell grid each pick comes from is not in this folder:
`results/snmf/google_gemma-2-2b-it/rank100/seed42/train_mc/<Concept>/hps.csv`
(10 concepts, run fresh), or the same path under `archive_gemini/` (8 concepts,
reused from the Gemini era; the grid is judge-free, so that is safe). The
`grid_source` column of the summary says which.

## Where the answer is

`pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_hps_summary.csv` — one row per concept:
the picked cell, the full HP set, efficacy / specificity / harmonic, the two Alpaca
fractions, coherence and `harmonic_alpaca`. Each concept folder next to it holds:

- `best_hp__qwen2.5-32b-instruct.json` — **the pick**: `best_hp` (the HP dict to apply), `best_cell`, `best_metrics`, the baselines used
- `cell_metrics__qwen2.5-32b-instruct.csv` — the 3-row table behind the pick
- `alpaca_judged__qwen2.5-32b-instruct.json` — per-question judge scores (3 cells x 50 questions)
- `_rows__qwen2.5-32b-instruct.json` — per-cell checkpoint (used for resume)

`best_hps_summary.csv` was assembled from those files on 2026-10-10 (no script
writes it, same as EMBER's `best_deltas_summary.csv`). Rebuild it by hand if a pick ever changes.

## The policy every result follows

1. Rank the 144 grid cells by the judge-free `harmonic` (MC efficacy x specificity); keep the top 3.
2. Generate the first 50 Alpaca answers for each (the same 50 questions EMBER uses) and judge them with Qwen2.5-32B.
3. Pick the max `harmonic_alpaca`. **No Alpaca gate** (EMBER zeroes cells below 0.9; SNMF's original validate stage never did).

The Alpaca denominators are the same Qwen-judged pristine baseline EMBER used,
so SNMF and EMBER scores are directly comparable. Details: `docs/Part 1/snmf_search.md`.

**One exception, Ancient Rome (`selection = loop_screen` in the summary).** Its
top 3 cells all broke generation ("Roman Roman Roman …" in 45–49 of 50 answers).
For this concept only, `pass1_snmf_screen.py` walked down the ranking, skipping
duplicate cells and cells with more than 5 of 50 looping answers, and kept the
first 3 healthy ones (grid ranks 9, 11, 16; 16 rows walked). The walk is in
`pass1_outputs/.../Ancient_Rome/screen/screen_report.csv`.

## Running more

`pass1_snmf_top3.py`, `pass1_snmf_screen.py` and `pass2_snmf_judge.py` default to
`pass1_outputs/` and `pass2_outputs/` in this folder, so no `--out-dir` is needed.
This tree is world-writable so every teammate can write here. A concept whose
`best_hp__*.json` already exists is skipped; a half-judged one resumes.

## archive/

| Folder | Was | Contains |
|---|---|---|
| `ancient_rome_unscreened_top3/pass1/`, `.../pass2/` | `Ancient_Rome/_unscreened_top3/` in both output folders | Ancient Rome's first, broken top-3 pick (job 1004452): kept for the article's "what went wrong" |
| `slurm_logs/` | `pass1_snmf_*.out`, `pass2_snmf_*.out` in the repo root | Slurm logs of every SNMF job (1003727 – 1006227) |

Note: new Slurm logs are still written to the directory you run `sbatch` from (the `#SBATCH --output` line).
