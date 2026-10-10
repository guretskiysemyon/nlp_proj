# SNMF HP search — status by concept

Tracks the two-pass SNMF hyperparameter search for each of the 18 concepts
(Gemma-2-2b-it, rank 100, seed 42). Updated 2026-10-10.

**Pass 1 is done for all 18 concepts. Pass 2 is done for 14 concepts with a
usable pick; Pornography, Republic of Ireland and Valentine's Day are not
submitted yet (Ori will run them).** Ancient Rome's first pick was a broken
model. It is being redone with a loop screen (see "Ancient Rome" below), and
its old result is kept in `_unscreened_top3/`.

## What the two passes do

**Pass 1** (`judge_rerun/pass1_snmf_top3.py`, no judge). Takes SNMF's 144-cell
HP grid (scored without a judge: `mmlu`/`qa`/`simdom` → `harmonic`), keeps the
**top 3 cells** by `harmonic`, and generates the **first 50 Alpaca
completions** for each (the same fixed 50 questions the EMBER search uses).
The grid is reused from the archived Gemini-era run when one exists (it is
judge-free, so that is safe); otherwise it is run fresh.
Output: `judge_rerun/pass1_outputs_snmf/google_gemma-2-2b-it/rank100/seed42/<Concept>/`
(`top3_hps.json`, `alpaca_completions/cell_0..2.json`). A fresh grid also
leaves `results/snmf/google_gemma-2-2b-it/rank100/seed42/train_mc/<Concept>/hps.csv`
(all 144 rows) and `top_hps.csv`.

**Pass 2** (`judge_rerun/pass2_snmf_judge.py`, Qwen2.5-32B-Instruct judge).
Judges the 3 × 50 completions, combines the scores with each cell's grid
efficacy/specificity into `coherence` and `harmonic_alpaca`, and picks the cell
with the highest `harmonic_alpaca`. The Alpaca denominators are the same
Qwen-judged pristine baseline EMBER's Pass 2 used, so SNMF and EMBER scores are
directly comparable.
Output: `judge_rerun/pass2_outputs_snmf/google_gemma-2-2b-it/rank100/seed42/<Concept>/`:

- `best_hp__qwen2.5-32b-instruct.json` — **the answer**: `best_hp` (the full HP
  dict to apply), `best_cell`, `best_metrics`, the baselines used
- `cell_metrics__qwen2.5-32b-instruct.csv` — the 3-row table behind the pick
- `alpaca_judged__qwen2.5-32b-instruct.json` — per-question judge scores (3 cells × 50)
- `_rows__qwen2.5-32b-instruct.json` — per-cell checkpoint (used for resume)

**No Alpaca gate for SNMF (by design).** EMBER zeroes `harmonic_alpaca` when
either Alpaca fraction is below 0.9. SNMF's original validate stage never set
that gate, so `pass2_snmf_judge.py` defaults `--min-alpaca` to `None`. A
picked SNMF cell can therefore have Alpaca fractions below 0.9 (Golf: 0.87),
and in the worst case a broken model (Ancient Rome, below).

## Status

All Pass 1 entries are finished. Pass 2 jobs 1004452–1004455 (2026-10-09/10,
`s-004`/`s-005`, 3× RTX 2080 Ti each) all ended `COMPLETED`, exit 0. They had
no tracebacks and no preemptions, and the judge's answer was unparseable only
once (Uranium cell 0, 1/50). HP notation: `δ @ layers lo–hi (k features)`,
`in` = MLP `up_proj`, `out` = `down_proj`; `w_mode=both`,
`feature_source=all` for every concept. "min frac" = the lower of
`alp_instr_frac` / `alp_flu_frac`.

| Concept | Pass 2 | Job | Best cell | δ_in | δ_out | efficacy | specificity | coherence | harmonic_alpaca | min frac |
|---|---|---|---|---|---|---|---|---|---|---|
| Ancient Rome | **Redo with loop screen (not submitted yet)** — old pick below, unusable | 1004452 | 0 | 7 @ 0–8 (k=13) | 7 @ 13–25 (k=6) | 0.800 | 0.879 | **0.135** | **0.307** | **0.08** |
| Artificial intelligence | Finished | 1004452 | 0 | 4 @ 0–25 (k=51) | 4 @ 0–8 (k=36) | 0.691 | 0.957 | 0.993 | 0.857 | 0.99 |
| Baseball | Finished | 1004452 | 0 | 1 @ 0–12 (k=33) | 10 @ 0–8 (k=29) | 0.702 | 0.771 | 0.998 | 0.806 | 1.00 |
| COVID-19 pandemic | Finished | 1004452 | 1 | 7 @ 0–8 (k=31) | 4 @ 9–17 (k=13) | 0.788 | 0.772 | 0.980 | 0.837 | 0.97 |
| Cannabis | Finished | 1004453 | 0 | 4 @ 0–8 (k=25) | 4 @ 0–8 (k=25) | 0.508 | 0.788 | 0.970 | 0.703 | 0.94 |
| Culture of Greece | Finished | 1004453 | 2 | 1 @ 0–25 (k=14) | 10 @ 0–8 (k=10) | 0.694 | 0.919 | 0.975 | 0.844 | 0.97 |
| Gambling | Finished | 1004453 | 1 | 10 @ 0–12 (k=26) | 7 @ 9–17 (k=6) | 0.863 | 0.808 | 0.939 | 0.867 | 0.93 |
| Golf | Finished | 1004453 | 2 | 4 @ 0–25 (k=35) | 7 @ 13–25 (k=10) | 0.590 | 0.887 | 0.879 | 0.758 | 0.87 |
| Gun | Finished | 1004454 | 0 | 4 @ 0–8 (k=36) | 4 @ 9–17 (k=4) | 0.852 | 0.806 | 0.955 | 0.867 | 0.91 |
| Halloween | Finished | 1004454 | 1 | 10 @ 0–12 (k=16) | 7 @ 13–25 (k=8) | 0.606 | 0.873 | 0.986 | 0.787 | 0.98 |
| Harry Potter | Finished | 1004454 | 0 | 7 @ 0–12 (k=16) | 1 @ 0–8 (k=15) | 0.877 | 0.919 | 0.998 | 0.929 | 1.00 |
| Heroin | Finished | 1004454 | 0 | 7 @ 0–25 (k=32) | 10 @ 9–17 (k=3) | 0.667 | 0.958 | 0.972 | 0.840 | 0.94 |
| Nazism | Finished | 1004455 | 0 | 10 @ 0–12 (k=25) | 1 @ 9–17 (k=9) | 0.980 | 0.608 | 0.942 | 0.805 | 0.91 |
| Pornography | **Not submitted (Ori will run it)** | — | | | | | | | | |
| Republic of Ireland | **Not submitted (Ori will run it)** | — | | | | | | | | |
| Uranium | Finished | 1004455 | 2 | 10 @ 0–25 (k=36) | 1 @ 0–8 (k=17) | 0.513 | 0.844 | 0.948 | 0.716 | 0.90 |
| Valentine's Day | **Not submitted (Ori will run it)** | — | | | | | | | | |
| World War II | Finished | 1004455 | 1 | 10 @ 0–25 (k=15) | 1 @ 13–25 (k=3) | 0.873 | 0.685 | 0.988 | 0.829 | 0.98 |

Cannabis, Culture of Greece, Gambling, Ancient Rome and Gun were finished by
earlier attempts and skipped by these jobs. Golf and Artificial intelligence
resumed from a checkpoint with 1 of 3 cells already scored.

**Pass 1 provenance** (all finished): archived grid reused for Baseball,
Culture of Greece, Gambling, Golf, Halloween, Harry Potter, Uranium, World War II
(jobs 999645, 1001938). Fresh 144-cell grid for Heroin (999634); Ancient Rome,
Artificial intelligence, COVID-19 pandemic (1003727); Cannabis, Gun, Nazism
(1003728); Pornography, Republic of Ireland, Valentine's Day (1004399). Mary's
first try at the last three (1003855) died with a `PermissionError` on
`mf_outputs/.../cache/rank100/seed42/Pornography`, and 1004399 replaced it.

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
distinct edits. Culture of Greece has the same duplication, and its pick is healthy.

This is deterministic (fixed features, greedy decoding), so re-running the same
cells would give the same broken answers. The 144-cell grid itself is fine and
is not re-run; only the selection from it changes.

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
has 0/150 looping answers. Of the 54 cells generated for the 18 concepts,
only Ancient Rome's 3 (45, 45, 49) and Golf's cell 0 (14) exceed 5; every
other cell has 0–3. So **applying the same screen to the other 17 concepts
would change no pick**: Golf's cell 0 was not picked anyway. The screen is
therefore a fix for one failure case, not a change to the method's results.

Expected outcome: the strong late-layer cells (grid ranks 1–8) should be
screened out, and the winner should be a milder edit around rank 9+
(e.g. δ_in 7 @ 0–8, δ_out 1 @ 9–17, harmonic 0.744). That means lower efficacy
(~0.6 instead of 0.8) but normal fluency.

**Status: script written and tested without a GPU (including the stopping
rule, on fake saved answers); not submitted yet.** The old unscreened results
were moved, not deleted, to
`pass1_outputs_snmf/.../Ancient_Rome/_unscreened_top3/` and
`pass2_outputs_snmf/.../Ancient_Rome/_unscreened_top3/`.

```
# 1) screen (1x RTX 2080, ~1 h: ~3 min per grid row; expected to stop around row 15-20)
sbatch judge_rerun/run_pass1_snmf_screen.sbatch --concept "Ancient Rome"
# 2) after it finishes: judge the 3 survivors (3x RTX 2080, ~2.5 h)
sbatch judge_rerun/run_pass2_snmf.sbatch --concept "Ancient Rome"
```

Outputs of step 1: `pass1_outputs_snmf/.../Ancient_Rome/screen/screen_report.csv`
(every grid row walked: rank, HP, scores, duplicate-of, looping answers,
selected) and `screen/alpaca_completions/row_NNN.json`, plus the usual
`top3_hps.json` (with `"selection": "loop_screen"`) and
`alpaca_completions/cell_0..2.json` for Pass 2. Step 1 resumes from saved rows
if it's interrupted.

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
> loop), until 3 distinct non-looping cells were found (N rows checked). Those 3
> were judged as usual. The unedited model has no looping answers, and applying
> the same screen to the other 17 concepts excludes no selected cell.

Fill in N (`rows_walked` in `top3_hps.json`), the final pick and its scores
once the two jobs finish.

## How to submit Pass 2 for the remaining three

```
sbatch judge_rerun/run_pass2_snmf.sbatch --concept "Pornography" "Republic of Ireland" "Valentine's Day"
```

Asks for 3× RTX 2080 Ti (`s-004`/`s-005` only), 32G RAM, 6 CPUs on
`studentkillable`. ~50 min per cell → ~2.5 h per concept, so ~7.5 h for the
three (24 h limit). Splitting them into three single-concept jobs finishes
faster if nodes are free. A finished concept is skipped and a half-judged one
resumes, so after a preemption resubmit the identical command. Put each
concept in only one job at a time. Pass 1 equivalent, if ever needed:
`sbatch judge_rerun/run_pass1_snmf_top3.sbatch --concept "<Concept>" ...`
(1 GPU; ~2h10m per fresh grid, ~9 min with an archived grid).

Slurm logs (`pass1_snmf_top3_*.out`, `pass2_snmf_*.out`) are written to the
repo root (the directory `sbatch` is run from). They have not been moved into
an archive folder yet.
