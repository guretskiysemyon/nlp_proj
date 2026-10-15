# SNMF+EMBER — HP search results (Gemma-2-2b-it, rank 100, seed 42)

The paper's combined method (`snmf_ember`): each concept's best EMBER δ applied
to the embeddings, plus an SNMF MLP edit whose 144-cell grid was searched again
on top of it. Finished for all 18 concepts on 2026-10-13.
Status, results table and comparison with EMBER / SNMF alone: `docs/Part 1/snmf_ember_search.md`.

| Folder | What it is |
|---|---|
| `pass1_outputs/` | Pass 1 (no judge). Per concept: `top3_hps.json` (the 3 shortlisted grid cells with their judge-free scores, plus `ember_delta_embed`) and `alpaca_completions/cell_0..2.json` (50 Alpaca answers each). |
| `pass2_outputs/` | **Final** Pass 2: the best cell per concept, all 18 concepts, judged by Qwen2.5-32B-Instruct. |
| `archive/slurm_logs/` | Slurm logs of every job (Pass 1: 1007155–1007165, Pass 2: 1007174–1007177). |

The 144-cell grid each pick comes from is not in this folder:
`results/snmf_ef/google_gemma-2-2b-it/rank100/seed42/train_mc/<Concept>/hps.csv`
(10 concepts, run fresh), or the same path under `archive_gemini/` (8 concepts,
reused from the Gemini era; run with exactly the Qwen-era EMBER δ, checked).
The `grid_source` column of the summary says which.

## Where the answer is

`pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_hps_summary.csv`: one row per concept with
the picked cell, `delta_embed` (the EMBER δ), the full SNMF HP set, efficacy /
specificity / harmonic, the two Alpaca fractions, coherence and `harmonic_alpaca`.
Each concept folder next to it holds:

- `best_hp__qwen2.5-32b-instruct.json`: **the pick**. `best_hp` (the HP dict to apply, including `delta_embed`), `best_cell`, `best_metrics`, the baselines used
- `cell_metrics__qwen2.5-32b-instruct.csv`: the 3-row table behind the pick
- `alpaca_judged__qwen2.5-32b-instruct.json`: per-question judge scores (3 cells x 50 questions)
- `_rows__qwen2.5-32b-instruct.json`: per-cell checkpoint (used for resume)

`best_hps_summary.csv` was assembled from those files on 2026-10-14 (no script
writes it, as for SNMF and EMBER). Rebuild it by hand if a pick ever changes.

## The policy every result follows

Exactly the SNMF policy (`judge_rerun/SNMF/README.md`), on the combined grid:

1. Fix the EMBER δ to the concept's EMBER pick (`judge_rerun/Ember/pass2_outputs/.../best_deltas_summary.csv`).
2. Rank the 144 SNMF cells (run with that δ) by the judge-free `harmonic`; keep the top 3.
3. Generate the first 50 Alpaca answers for each and judge them with Qwen2.5-32B.
4. Pick the max `harmonic_alpaca`. **No Alpaca gate.**

The Alpaca denominators are the same Qwen-judged pristine baseline as EMBER and
SNMF, so the three methods' scores are directly comparable. No concept needed
the loop screen: no cell of the 54 has more than 3 looping answers out of 50.

## Running more

`pass1_snmf_top3.py`, `pass1_snmf_screen.py` and `pass2_snmf_judge.py` write here
when given `--with-ember`. This tree is world-writable so every teammate can
write here. A concept whose `best_hp__*.json` already exists is skipped; a
half-judged one resumes.

Note: new Slurm logs are still written to the directory you run `sbatch` from (the `#SBATCH --output` line).
