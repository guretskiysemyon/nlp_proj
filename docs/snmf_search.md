# SNMF Pass 1 — status by concept

Tracks `judge_rerun/pass1_snmf_top3.py` progress for each of the 18 concepts.
Updated 2026-10-09.

## What Pass 1 does for SNMF

For each concept it takes SNMF's 144-cell hyperparameter grid (scored with no
judge: `mmlu`/`qa`/`simdom` → `harmonic`), keeps the **top 3 cells**, and
generates — but does not judge — the **first 50 Alpaca completions** for each of
those 3 cells (the same fixed 50 questions the EMBER search uses). The grid is
reused from the archived Gemini-era run when one exists (it is judge-free, so
that is safe); otherwise it is run fresh. Judging these completions is Pass 2,
which is **not written yet** for SNMF.

Output: `judge_rerun/pass1_outputs_snmf/google_gemma-2-2b-it/rank100/seed42/<Concept>/`
(`top3_hps.json` and `alpaca_completions/cell_0..2.json`). A fresh grid also
leaves `results/snmf/google_gemma-2-2b-it/rank100/seed42/train_mc/<Concept>/hps.csv`
(all 144 rows) and `top_hps.csv`.

## Status: 9 finished, 6 submitted, 3 not submitted yet

| Concept | Pass 1 status | Grid | Job | Person |
|---|---|---|---|---|
| Ancient Rome | Submitted | fresh 144-cell | 1003727 | Semyon |
| Artificial intelligence | Submitted | fresh 144-cell | 1003727 | Semyon |
| Baseball | Finished | archived (reused) | 1001938 | Semyon |
| COVID-19 pandemic | Submitted | fresh 144-cell | 1003727 | Semyon |
| Cannabis | Submitted | fresh 144-cell | 1003728 | Semyon |
| Culture of Greece | Finished | archived (reused) | 1001938 | Semyon |
| Gambling | Finished | archived (reused) | 1001938 | Semyon |
| Golf | Finished | archived (reused) | 999645 | Semyon |
| Gun | Submitted | fresh 144-cell | 1003728 | Semyon |
| Halloween | Finished | archived (reused) | 1001938 | Semyon |
| Harry Potter | Finished | archived (reused) | 1001938 | Semyon |
| Heroin | Finished | fresh 144-cell | 999634 | Semyon |
| Nazism | Submitted | fresh 144-cell | 1003728 | Semyon |
| Pornography | **Not submitted yet** | fresh 144-cell | — | — |
| Republic of Ireland | **Not submitted yet** | fresh 144-cell | — | — |
| Uranium | Finished | archived (reused) | 1001938 | Semyon |
| Valentine's Day | **Not submitted yet** | fresh 144-cell | — | — |
| World War II | Finished | archived (reused) | 1001938 | Semyon |

"Finished" means `top3_hps.json` and the three `cell_*.json` files exist.

**Jobs in flight** (both started 2026-10-09 ≈09:51 on `s-004`, 1 GPU each):
- 1003727 — Ancient Rome, Artificial intelligence, COVID-19 pandemic
- 1003728 — Cannabis, Gun, Nazism

A fresh grid takes about 2h10m per concept (Heroin: 144 cells at ~55 s each,
plus ~9 min of completions), plus a one-time ~17 min model load per job, so each
3-concept job should take about 7 h. A concept with an archived grid takes only
~9 min.

**Not submitted yet:** Pornography, Republic of Ireland, Valentine's Day — meant
to go together as the third job.

## How to submit

```
sbatch judge_rerun/run_pass1_snmf_top3.sbatch --concept "Pornography" "Republic of Ireland" "Valentine's Day"
```

`--concept` takes several values and runs them back-to-back with one model load.
A concept that is already finished is skipped, and an interrupted fresh grid
resumes from its saved rows, so after a timeout or preemption just resubmit the
identical command. Put each concept in only one job at a time (two jobs would
write the same grid file).
