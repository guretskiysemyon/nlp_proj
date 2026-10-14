# SNMF+EMBER HP search — status

The paper's combined method (`snmf_ember`, results folder `snmf_ef`), redone
with the Qwen judge (Gemma-2-2b-it, rank 100, seed 42). Started 2026-10-10.

**Finished for all 18 concepts (2026-10-13).** Every concept has a complete
144-cell grid (run with exactly its EMBER δ), a Pass 1 shortlist and a
Qwen-judged Pass 2 pick. No pick loops, so no concept needed the loop screen.
Nothing is running. Next step: the `snmf_ember` cross-lingual run
([`crosslingual_next.md`](crosslingual_next.md) §5).

**Where the results are:** `judge_rerun/SNMF_EMBER/` (layout and policy:
[`judge_rerun/SNMF_EMBER/README.md`](../../judge_rerun/SNMF_EMBER/README.md)).
Start with
`judge_rerun/SNMF_EMBER/pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_hps_summary.csv`,
one row per concept with the full HP set to apply (EMBER δ + SNMF cell).

## What it is

The same two-pass search as SNMF ([`snmf_search.md`](snmf_search.md)), with
one difference: each concept's **best EMBER δ** (the Qwen-judged pick in
`judge_rerun/Ember/pass2_outputs/.../best_deltas_summary.csv`) is applied under
every SNMF cell, and the 144-cell SNMF grid is **searched again** on top of it.
This is what `ember/erasure/pipeline.py` does when `ember_step.enabled` is true
(`configs/snmf_ember_gemma.yaml`). The EMBER δ is reused; the SNMF settings are
not (the best SNMF cell can change once the embeddings are edited).

Same rules as SNMF: top 3 cells by the judge-free `harmonic`, 50 Alpaca answers
each, Qwen2.5-32B judge, pick by `harmonic_alpaca`, no Alpaca gate. The Alpaca
denominators are the same Qwen-judged pristine baseline as EMBER and SNMF, so
the three methods' scores are directly comparable.

## Results

HP notation as in `snmf_search.md`: `δ @ layers lo–hi (k features)`, `in` =
MLP `up_proj`, `out` = `down_proj`; `w_mode=both`, `feature_source=all` for
every concept. δ_emb = the EMBER δ (same as the EMBER pick for every concept).
"min frac" = the lower of `alp_instr_frac` / `alp_flu_frac` (bold: below 0.9).
"Loops" = looping Alpaca answers out of 50 for cells 0/1/2 (rule in
`snmf_search.md`: a word repeated 8+ times in a row; a cell fails at >5).

| Concept | Grid | Best cell | δ_emb | δ_in | δ_out | efficacy | specificity | coherence | harmonic_alpaca | min frac | Loops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ancient Rome | fresh | 0 | 200 | 1 @ 0–8 (k=13) | 4 @ 0–8 (k=13) | 0.909 | 0.952 | 0.998 | 0.952 | 1.00 | 0/0/0 |
| Artificial intelligence | fresh | 0 | 100 | 4 @ 0–8 (k=36) | 4 @ 0–8 (k=36) | 0.836 | 0.910 | 0.935 | 0.892 | 0.92 | 0/3/0 |
| Baseball | archived | 0 | 100 | 4 @ 0–25 (k=47) | 7 @ 13–25 (k=14) | 0.982 | 0.838 | 0.915 | 0.908 | **0.88** | 0/0/0 |
| COVID-19 pandemic | fresh | 0 | 50 | 4 @ 0–25 (k=54) | 4 @ 0–8 (k=31) | 0.970 | 0.818 | 0.953 | 0.908 | 0.93 | 0/0/0 |
| Cannabis | fresh | 1 | 200 | 10 @ 0–12 (k=28) | 1 @ 0–8 (k=25) | 0.780 | 0.826 | 0.991 | 0.857 | 0.99 | 0/0/0 |
| Culture of Greece | archived | 0 | 200 | 1 @ 0–8 (k=10) | 1 @ 0–8 (k=10) | 1.000 | 0.953 | 1.000 | 0.984 | 1.00 | 0/0/0 |
| Gambling | archived | 2 | 100 | 1 @ 0–25 (k=36) | 7 @ 0–8 (k=23) | 0.980 | 0.985 | 0.998 | 0.988 | 1.00 | 0/0/0 |
| Golf | archived | 1 | 10 | 1 @ 0–8 (k=24) | 1 @ 13–25 (k=10) | 0.623 | 0.962 | 0.998 | 0.823 | 1.00 | 0/0/0 |
| Gun | fresh | 1 | 100 | 1 @ 0–25 (k=55) | 7 @ 9–17 (k=4) | 0.918 | 0.902 | 0.959 | 0.926 | 0.92 | 0/0/0 |
| Halloween | archived | 1 | 10 | 10 @ 0–25 (k=24) | 1 @ 0–8 (k=14) | 0.727 | 0.940 | 0.993 | 0.871 | 0.99 | 0/0/0 |
| Harry Potter | archived | 0 | 10 | 10 @ 0–8 (k=15) | 1 @ 0–8 (k=15) | 0.842 | 0.883 | 0.998 | 0.903 | 1.00 | 0/0/0 |
| Heroin | fresh | 2 | 100 | 10 @ 0–25 (k=32) | 4 @ 9–17 (k=3) | 0.872 | 0.864 | 0.990 | 0.905 | 0.98 | 0/0/0 |
| Nazism | fresh | 1 | 100 | 1 @ 0–12 (k=25) | 10 @ 13–25 (k=13) | 0.735 | 0.891 | 0.907 | 0.837 | 0.90 | 2/1/0 |
| Pornography | fresh | 0 | 5 | 4 @ 0–8 (k=23) | 10 @ 9–17 (k=7) | 0.939 | 0.947 | 0.881 | 0.921 | **0.85** | 0/0/0 |
| Republic of Ireland | fresh | 0 | 100 | 10 @ 0–25 (k=9) | 10 @ 9–17 (k=2) | 0.837 | 0.933 | 0.974 | 0.911 | 0.97 | 0/0/0 |
| Uranium | archived | 1 | 100 | 4 @ 0–12 (k=20) | 10 @ 13–25 (k=16) | 1.000 | 1.000 | 0.906 | 0.967 | **0.89** | 1/0/1 |
| Valentine's Day | fresh | 2 | 200 | 1 @ 0–12 (k=14) | 10 @ 0–8 (k=13) | 1.000 | 0.869 | 0.909 | 0.923 | **0.87** | 0/0/0 |
| World War II | archived | 0 | 5 | 10 @ 0–25 (k=15) | 1 @ 0–8 (k=11) | 0.836 | 0.891 | 0.975 | 0.897 | 0.97 | 0/0/0 |

### Compared with EMBER and SNMF alone

Same judge, same baselines, same `QA_train` / MMLU (50) / SimDom items used for selection.
Mean over the 18 concepts (median in brackets):

| Method | efficacy | specificity | coherence | harmonic_alpaca |
|---|---|---|---|---|
| EMBER | 0.520 (0.440) | 0.968 (0.965) | 0.992 (0.998) | 0.735 (0.693) |
| SNMF | 0.724 (0.698) | 0.848 (0.858) | 0.959 (0.962) | 0.821 (0.833) |
| **SNMF+EMBER** | **0.877** (0.890) | 0.909 (0.906) | 0.960 (0.974) | **0.909** (0.908) |

- The combined edit has the highest `harmonic_alpaca` for **17 of 18**
  concepts. The exception is **Harry Potter** (0.903 vs SNMF's 0.929). SNMF's
  own best cell is cell 1 of the combined top 3, but with δ_emb = 10 added its
  concept-QA accuracy is 0.36 instead of 0.32 (efficacy 0.81 vs 0.88), so the
  EMBER step did not help here. That is 2 questions of 50, within noise.
- **Efficacy** is higher than SNMF for 15 of 18 and higher than EMBER for all 18.
  Three concepts reach efficacy 1.00: concept-QA accuracy at or below chance
  (Culture of Greece 0.24, Uranium 0.24, Valentine's Day 0.22, chance 0.25).
- **Specificity** is *higher* than SNMF on average (0.909 vs 0.848; lower for
  only 5 concepts) and lower than EMBER for 15. Mean MMLU of the picks is 0.524
  (pristine 0.52, 50 items).
- **Coherence** matches SNMF (0.960 vs 0.959) and is below EMBER (0.992).
- **Caveat:** these are the scores the picks were *selected* on (`QA_train`,
  50 items, best of 144 cells, on top of a δ chosen on the same items). They
  are optimistic, and more so for the combined method, which searches the most.
  The held-out comparison is `QA_test` in the cross-lingual run (English column).

### Notes

- **Alpaca fraction below 0.9** for 4 picks (no gate, by design, as for SNMF):
  Pornography (instruction 0.85), Valentine's Day (instruction 0.87), Baseball
  (fluency 0.88), Uranium (fluency 0.89). SNMF had 2 (Golf 0.87, Pornography 0.85).
  None of them loops.
- **Ancient Rome loops no more.** Plain SNMF's top 3 all looped ("Roman Roman
  …", 45–49 of 50) and needed the loop screen. With δ_emb = 200 applied, the
  top 3 are all milder edits (output δ ≤ 4, layers 0–8) with 0 looping answers.
- **Strong late output edits are not always broken.** Nazism and Uranium pick
  δ_out 10 @ 13–25, the pattern that broke Ancient Rome under plain SNMF, but
  they have 1 and 0 looping answers.
- **Duplicates:** Ancient Rome and Culture of Greece cells 0 and 1 are the same
  edit (all 50 answers word-for-word identical, `layer_hi_in` 8 vs 12), so only
  2 distinct edits were judged for them. The judge gave each pair the same
  score.
- **Uranium:** cells 0 and 1 both have `harmonic` 1.00 (efficacy and
  specificity both at the cap); the judge decided (0.951 vs 0.967).
- **Judge parse failures:** 3 in the whole search, 1 of 50 each (Halloween
  cell 2, Uranium cell 2, Valentine's Day cell 1), none in a picked cell.

## Completeness check (2026-10-14)

Checked on disk for every concept:

- **Grid:** `hps.csv` has 144 rows, 144 distinct HP combinations, no missing
  `mmlu_acc` / `qa_acc` / `simdom_acc` / `efficacy` / `specificity` /
  `harmonic`, and every row has `delta_embed` equal to the concept's EMBER δ.
  10 grids under `results/snmf_ef/...`, 8 under `archive_gemini/results/snmf_ef/...`.
- **Pass 1:** `top3_hps.json` and 3 × 50 completions for all 18.
- **Pass 2:** `best_hp__*.json` with `complete: true` and 3 of 3 cells scored,
  judge `qwen2.5-32b-instruct`, for all 18. Each pick's `delta_embed` equals the
  EMBER pick.
- **Loops:** the most looping answers in any of the 54 cells is 3 of 50
  (Artificial intelligence cell 1, not picked); no picked cell has more than 1.

## Code

The SNMF scripts take `--with-ember` (2026-10-10):

- `judge_rerun/pass1_snmf_top3.py --with-ember`: loads `configs/snmf_ember_gemma.yaml`,
  runs/reuses grids under `results/snmf_ef/` (archived: `archive_gemini/results/snmf_ef/`),
  writes `judge_rerun/SNMF_EMBER/pass1_outputs/`. Refuses a grid whose rows do not
  all carry exactly the concept's EMBER δ. `top3_hps.json` records `ember_delta_embed`.
- `judge_rerun/pass1_snmf_screen.py --with-ember`: the loop screen, same switch
  (not needed in this search).
- `judge_rerun/pass2_snmf_judge.py --with-ember`: reads/writes `judge_rerun/SNMF_EMBER/`.
  `best_hp` now always includes `delta_embed` (0.0 for plain SNMF).

## Grids

8 concepts reuse the Gemini-era combined grid (144 cells, run with exactly the
Qwen-era EMBER δ): Harry Potter (10), World War II (5), Halloween (10),
Uranium (100), Golf (10), Baseball (100), Gambling (100), Culture of Greece (200).
The other 10 run a fresh grid: Ancient Rome (200), Artificial intelligence (100),
COVID-19 pandemic (50), Cannabis (200), Gun (100), Heroin (100), Nazism (100),
Pornography (5), Republic of Ireland (100), Valentine's Day (200).

## Jobs

| Pass | Job | Concepts | Outcome |
|---|---|---|---|
| 1 | 1007155 | the 8 archived-grid concepts (answers only) | done 2026-10-11, 1h11m |
| 1 | 1007156–1007165 | one fresh-grid concept each (order as listed above) | done 2026-10-11, 1.5–2.3 h each |
| 2 | 1007174 | Harry Potter, World War II, Halloween, Uranium | `COMPLETED` 2026-10-13 22:46, 9 preemptions |
| 2 | 1007175 | Golf, Baseball, Culture of Greece, Gambling | `COMPLETED` 2026-10-13 17:14, 6 preemptions |
| 2 | 1007176 | Ancient Rome, AI, COVID-19, Cannabis, Gun | `COMPLETED` 2026-10-13 23:33, 7 preemptions |
| 2 | 1007177 | Heroin, Nazism, Pornography, Republic of Ireland, Valentine's Day | `COMPLETED` 2026-10-13 20:41, 6 preemptions |

Pass 2 first started 2026-10-11 01:22 and finished 2026-10-13 23:33. It was
preempted 28 times in total (often during the 1–3 h cold load of Qwen-32B);
every requeue resumed from the per-cell / per-item checkpoints, with no
tracebacks and no OOM. Judging itself took 43 h over the 54 cells (~48 min per
cell). Slurm logs: `judge_rerun/SNMF_EMBER/archive/slurm_logs/`.

## Re-running

Nothing needs re-running. If a concept ever has to be redone, delete its
`best_hp__*.json` (a finished concept is skipped) and submit with `--with-ember`:

```
sbatch judge_rerun/run_pass1_snmf_top3.sbatch --with-ember --concept "<Concept>"
sbatch judge_rerun/run_pass2_snmf.sbatch --with-ember --concept "<Concept>" ...
```

Same resources and resume rules as SNMF (`snmf_search.md` "Re-running").
