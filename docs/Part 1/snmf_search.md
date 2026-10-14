# SNMF HP search — status by concept

Tracks the two-pass SNMF hyperparameter search for each of the 18 concepts
(Gemma-2-2b-it, rank 100, seed 42). Updated 2026-10-10.

**Finished for all 18 concepts (2026-10-10).** Every concept has a complete
144-cell grid, a Pass 1 shortlist and a Qwen-judged Pass 2 pick. Ancient Rome's
first pick was a broken model; it was redone with a loop screen (see "Ancient
Rome" below) and now has a healthy pick. Nothing is running.

**Where the results are:** `judge_rerun/SNMF/` (layout, policy and archive:
[`judge_rerun/SNMF/README.md`](../../judge_rerun/SNMF/README.md)). Start with
`judge_rerun/SNMF/pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_hps_summary.csv`,
one row per concept with the full HP set to apply.

## What the two passes do

**Pass 1** (`judge_rerun/pass1_snmf_top3.py`, no judge). Takes SNMF's 144-cell
HP grid (scored without a judge: `mmlu`/`qa`/`simdom` → `harmonic`), keeps the
**top 3 cells** by `harmonic`, and generates the **first 50 Alpaca
completions** for each (the same fixed 50 questions the EMBER search uses).
The grid is reused from the archived Gemini-era run when one exists (it is
judge-free, so that is safe); otherwise it is run fresh.
Output: `judge_rerun/SNMF/pass1_outputs/google_gemma-2-2b-it/rank100/seed42/<Concept>/`
(`top3_hps.json`, `alpaca_completions/cell_0..2.json`). A fresh grid also
leaves `results/snmf/google_gemma-2-2b-it/rank100/seed42/train_mc/<Concept>/hps.csv`
(all 144 rows) and `top_hps.csv`.

**Pass 2** (`judge_rerun/pass2_snmf_judge.py`, Qwen2.5-32B-Instruct judge).
Judges the 3 × 50 completions, combines the scores with each cell's grid
efficacy/specificity into `coherence` and `harmonic_alpaca`, and picks the cell
with the highest `harmonic_alpaca`. The Alpaca denominators are the same
Qwen-judged pristine baseline EMBER's Pass 2 used, so SNMF and EMBER scores are
directly comparable.
Output: `judge_rerun/SNMF/pass2_outputs/google_gemma-2-2b-it/rank100/seed42/<Concept>/`:

- `best_hp__qwen2.5-32b-instruct.json` — **the answer**: `best_hp` (the full HP
  dict to apply), `best_cell`, `best_metrics`, the baselines used
- `cell_metrics__qwen2.5-32b-instruct.csv` — the 3-row table behind the pick
- `alpaca_judged__qwen2.5-32b-instruct.json` — per-question judge scores (3 cells × 50)
- `_rows__qwen2.5-32b-instruct.json` — per-cell checkpoint (used for resume)

**No Alpaca gate for SNMF (by design).** EMBER zeroes `harmonic_alpaca` when
either Alpaca fraction is below 0.9. SNMF's original validate stage never set
that gate, so `pass2_snmf_judge.py` defaults `--min-alpaca` to `None`. A
picked SNMF cell can therefore have Alpaca fractions below 0.9 (Golf 0.87,
Pornography 0.85), and in the worst case a broken model (Ancient Rome's first
pick, below).

## Completeness check (2026-10-10)

Checked on disk for every concept:

- **Grid:** `hps.csv` has 144 rows, 144 distinct HP combinations, and no
  missing `mmlu_acc` / `qa_acc` / `simdom_acc` / `efficacy` / `specificity` /
  `harmonic`. All 18 grids cover the same 144 combinations. 10 grids are live
  under `results/snmf/...`, 8 reused from `archive_gemini/results/snmf/...`.
- **Pass 1:** `top3_hps.json` and 3 × 50 completions for all 18.
- **Pass 2:** `best_hp__*.json` with `complete: true` and 3 of 3 cells scored,
  judge `qwen2.5-32b-instruct`, for all 18.
- **Loops:** no picked cell has more than 1 looping Alpaca answer (rule below).

## Status

All jobs ended `COMPLETED`, exit 0, with no tracebacks and no preemptions. The
judge's answer was unparseable twice in the whole search (Uranium cell 0 and
Valentine's Day cell 0, 1 of 50 each). HP notation: `δ @ layers lo–hi (k features)`,
`in` = MLP `up_proj`, `out` = `down_proj`; `w_mode=both`,
`feature_source=all` for every concept. "min frac" = the lower of
`alp_instr_frac` / `alp_flu_frac` (bold: below 0.9).

| Concept | Pass 2 job | Best cell | δ_in | δ_out | efficacy | specificity | coherence | harmonic_alpaca | min frac |
|---|---|---|---|---|---|---|---|---|---|
| Ancient Rome (loop screen) | 1006121 + 1006227 | 0 | 7 @ 0–8 (k=13) | 1 @ 9–17 (k=4) | 0.618 | 0.934 | 0.948 | 0.802 | 0.90 |
| Artificial intelligence | 1004452 | 0 | 4 @ 0–25 (k=51) | 4 @ 0–8 (k=36) | 0.691 | 0.957 | 0.993 | 0.857 | 0.99 |
| Baseball | 1004452 | 0 | 1 @ 0–12 (k=33) | 10 @ 0–8 (k=29) | 0.702 | 0.771 | 0.998 | 0.806 | 1.00 |
| COVID-19 pandemic | 1004452 | 1 | 7 @ 0–8 (k=31) | 4 @ 9–17 (k=13) | 0.788 | 0.772 | 0.980 | 0.837 | 0.97 |
| Cannabis | 1004453 | 0 | 4 @ 0–8 (k=25) | 4 @ 0–8 (k=25) | 0.508 | 0.788 | 0.970 | 0.703 | 0.94 |
| Culture of Greece | 1004453 | 2 | 1 @ 0–25 (k=14) | 10 @ 0–8 (k=10) | 0.694 | 0.919 | 0.975 | 0.844 | 0.97 |
| Gambling | 1004453 | 1 | 10 @ 0–12 (k=26) | 7 @ 9–17 (k=6) | 0.863 | 0.808 | 0.939 | 0.867 | 0.93 |
| Golf | 1004453 | 2 | 4 @ 0–25 (k=35) | 7 @ 13–25 (k=10) | 0.590 | 0.887 | 0.879 | 0.758 | **0.87** |
| Gun | 1004454 | 0 | 4 @ 0–8 (k=36) | 4 @ 9–17 (k=4) | 0.852 | 0.806 | 0.955 | 0.867 | 0.91 |
| Halloween | 1004454 | 1 | 10 @ 0–12 (k=16) | 7 @ 13–25 (k=8) | 0.606 | 0.873 | 0.986 | 0.787 | 0.98 |
| Harry Potter | 1004454 | 0 | 7 @ 0–12 (k=16) | 1 @ 0–8 (k=15) | 0.877 | 0.919 | 0.998 | 0.929 | 1.00 |
| Heroin | 1004454 | 0 | 7 @ 0–25 (k=32) | 10 @ 9–17 (k=3) | 0.667 | 0.958 | 0.972 | 0.840 | 0.94 |
| Nazism | 1004455 | 0 | 10 @ 0–12 (k=25) | 1 @ 9–17 (k=9) | 0.980 | 0.608 | 0.942 | 0.805 | 0.91 |
| Pornography | 1006221 | 0 | 4 @ 0–8 (k=23) | 10 @ 9–17 (k=7) | 0.816 | 0.947 | 0.887 | 0.880 | **0.85** |
| Republic of Ireland | 1006221 | 2 | 1 @ 0–25 (k=9) | 10 @ 0–8 (k=5) | 0.558 | 0.826 | 0.952 | 0.740 | 0.95 |
| Uranium | 1004455 | 2 | 10 @ 0–25 (k=36) | 1 @ 0–8 (k=17) | 0.513 | 0.844 | 0.948 | 0.716 | 0.90 |
| Valentine's Day | 1006221 | 1 | 1 @ 0–12 (k=14) | 10 @ 0–8 (k=13) | 0.829 | 0.955 | 0.955 | 0.909 | 0.91 |
| World War II | 1004455 | 1 | 10 @ 0–25 (k=15) | 1 @ 13–25 (k=3) | 0.873 | 0.685 | 0.988 | 0.829 | 0.98 |

Notes on the last three concepts (job 1006221, run 2026-10-10 on `s-004`, 9 h):

- **Pornography**: the pick's instruction fraction is 0.85, below EMBER's 0.9
  gate. Allowed under SNMF's no-gate policy (like Golf), but worth a footnote.
- **Republic of Ireland**: cells 0 and 1 are the same edit (all 50 answers
  word-for-word identical, `layer_hi_in` 8 vs 12 again), so only 2 distinct
  edits were judged. The judge gave them identical scores, a free determinism
  check. All three cells tie on the grid score; cell 2 wins by 0.0002, so the
  three picks are interchangeable.
- **Valentine's Day**: all three cells have the same grid score; the judge
  decided. The pristine model only scores 0.46 on its English `QA_test`, so
  cross-lingual numbers for this concept will be noisy.

**Pass 1 provenance** (all finished): archived grid reused for Baseball,
Culture of Greece, Gambling, Golf, Halloween, Harry Potter, Uranium, World War II
(jobs 999645, 1001938). Fresh 144-cell grid for Heroin (999634); Ancient Rome,
Artificial intelligence, COVID-19 pandemic (1003727); Cannabis, Gun, Nazism
(1003728); Pornography, Republic of Ireland, Valentine's Day (1004399). Mary's
first try at the last three (1003855) died with a `PermissionError` on
`mf_outputs/.../cache/rank100/seed42/Pornography`, and 1004399 replaced it.
Ancient Rome's loop screen: 1006121.

## Ancient Rome: broken top-3 pick, redone with a loop screen

### What went wrong

All three top cells broke free-text generation: **45, 45 and 49 of 50** Alpaca
answers collapse into "Roman Roman Roman …" (typical: *"0.14 × $47.50 = $6.77
Roman Roman Roman …"*). The judge scored them accordingly (fluency 0.14, 0.14,
0.0 vs 1.69 for the unedited model). "Best cell 0" was only the least-broken of
three broken cells.

Why the top 3 were all broken: the judge-free `harmonic` ranks the grid by
multiple-choice tests only (concept QA, SimDom, MMLU), and those cannot see
broken free text. Ancient Rome's top 8 grid rows all use a strong output edit
(δ_out 7–10) in layers 13–25, the half of the model closest to the next-word
choice. That edit suppresses the concept on MC questions and breaks generation.
The SNMF edit `W ← W − δ·f(fᵀW)` with δ > 1 does not just remove the concept
direction, it flips it and multiplies it by (δ − 1). Across all 45 cells judged
in Pass 2, the 10 cells with δ_out ≥ 7 in layers 13–25 averaged 14.5 looping
answers and fluency 0.99, against 0.5 and 1.63 for the other 35.

Also, the grid rows come in **identical pairs**: `layer_hi_in` 8 vs 12 is the
same edit for Ancient Rome, because layers 9–12 hold no concept features. Cells 0
and 1 gave word-for-word identical answers (50/50), so the "top 3" held only 2
distinct edits. Culture of Greece and Republic of Ireland have the same
duplication, and their picks are healthy.

This is deterministic (fixed features, greedy decoding), so re-running the same
cells would give the same broken answers. The 144-cell grid itself is fine and
was not re-run; only the selection from it changed.

### The fix: loop screen (Ancient Rome only)

`judge_rerun/pass1_snmf_screen.py` replaces Pass 1's top-3 selection **for
this concept only**. The other 17 concepts keep their Pass 1 / Pass 2 results
as they are.

1. Rank the full grid (`results/snmf/.../Ancient_Rome/hps.csv`) exactly as
   Pass 1 does (`io.topk_per_concept`, `primary="harmonic"`, no extra filter).
2. Walk down the ranking, generating the first 50 Alpaca answers per cell
   (Gemma, greedy, zero-shot — the same generation as Pass 1).
3. Skip a cell if it is a **duplicate** (all 50 answers word-for-word
   identical to a higher-ranked cell's, i.e. the same edit) or if it is
   **looping**. An answer loops if one word containing a letter or digit is
   repeated 8+ times in a row (punctuation-only tokens such as ASCII-art
   `|||` are ignored). A cell loops if more than 5 of its 50 answers (>10%) loop.
4. **Stop as soon as 3 distinct, non-looping cells are found** (safety limit:
   40 grid rows). Every row below the third one ranks lower and could never
   replace it, so generating further would change nothing.
5. Judge those 3 with the unchanged Pass 2 (Qwen2.5-32B, same baselines, same
   `harmonic_alpaca` pick).

Generation and judging are both greedy (no sampling), so the screen is fully
reproducible. A copy of the edit always gives the same 50 answers, which is
why duplicates are detected from the answers themselves. That is safer than
comparing HP numbers, since feature selection per layer range is not
predictable from the table.

The rule was checked against every answer saved so far. The unedited model
has 0/150 looping answers. Of the 54 top-3 cells generated for the 18 concepts,
only Ancient Rome's first 3 (45, 45, 49) and Golf's cell 0 (14) exceed 5; every
other cell has 0–3. So **applying the same screen to the other 17 concepts
would change no pick**: Golf's cell 0 was not picked anyway.

### Outcome (jobs 1006121 + 1006227, 2026-10-10)

The screen walked **16 grid rows** (45 min) and kept ranks 9, 11 and 16:

| Grid ranks | Edit | Looping answers | Result |
|---|---|---|---|
| 1–8, 12–15 | δ_out 7–10 @ 13–25 (various δ_in) | 43–49 / 50 | looping (or a duplicate of one) |
| **9** | δ_in 7 @ 0–8, δ_out 1 @ 9–17 | 0 | kept → **picked** (cell 0) |
| 10 | same edit as rank 9 | 0 | duplicate |
| 11 | δ_in 7 @ 0–25, δ_out 10 @ 13–25 | 1 | kept (cell 1) |
| 16 | δ_in 7 @ 0–8, δ_out 1 @ 13–25 | 0 | kept (cell 2) |

Pass 2 picked cell 0 (grid rank 9): efficacy 0.618, specificity 0.934,
coherence 0.948, `harmonic_alpaca` 0.802 (cells 1 and 2: 0.790 and 0.786).
As predicted, efficacy is lower than the broken pick's 0.800, but fluency is
normal (Alpaca fractions 1.00 / 0.90, against 0.08 before). The full walk is in
`judge_rerun/SNMF/pass1_outputs/.../Ancient_Rome/screen/screen_report.csv`
(every row: rank, HP, scores, duplicate-of, looping answers, selected) and
`screen/alpaca_completions/row_NNN.json`. The first, broken results are kept in
`judge_rerun/SNMF/archive/ancient_rome_unscreened_top3/`.

### Note for the article

> For SNMF, the judge-free grid score (MC efficacy and specificity) is used to
> shortlist 3 of 144 hyperparameter cells for Alpaca judging. For one concept
> (Ancient Rome) all three shortlisted cells produced degenerate text: 45–49 of
> 50 Alpaca answers collapsed into repeating the concept word ("Roman Roman …").
> They used a strong output-side edit (δ_out 7–10) in layers 13–25, which
> suppresses the concept on multiple-choice questions but breaks free
> generation. For this concept only, we went down the grid ranking, generating
> Alpaca answers for each cell, and skipped cells that were duplicates
> (identical answers to a higher-ranked cell) or that looped (an answer loops if
> a word repeats ≥8 times in a row; a cell is excluded if >10% of its answers
> loop), until 3 distinct non-looping cells were found (16 rows checked). Those 3
> were judged as usual. The selected edit (grid rank 9) has efficacy 0.62 and
> coherence 0.95. The unedited model has no looping answers, and applying
> the same screen to the other 17 concepts excludes no selected cell.

## Re-running

Nothing needs re-running. If a concept ever has to be redone, delete its
`best_hp__*.json` (a finished concept is skipped) and submit:

```
sbatch judge_rerun/run_pass1_snmf_top3.sbatch --concept "<Concept>"   # 1 GPU; ~2h10m fresh grid, ~9 min with an existing grid
sbatch judge_rerun/run_pass1_snmf_screen.sbatch --concept "<Concept>" # loop-screen variant instead of top3
sbatch judge_rerun/run_pass2_snmf.sbatch --concept "<Concept>" ...    # 3x RTX 2080 Ti; ~50 min per cell, ~2.5 h per concept
```

Pass 2 asks for 3× RTX 2080 Ti (`s-004`/`s-005` only), 32G RAM, 6 CPUs on
`studentkillable`. A half-judged concept resumes, so after a preemption
resubmit the identical command. Put each concept in only one job at a time.
The Qwen-32B load alone took 40–53 min on a cold node in the last runs.

Slurm logs are written to the repo root (the directory `sbatch` is run from).
All SNMF logs so far were moved to `judge_rerun/SNMF/archive/slurm_logs/`.
