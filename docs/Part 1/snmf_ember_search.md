# SNMF+EMBER HP search — status

The paper's combined method (`snmf_ember`, results folder `snmf_ef`), redone
with the Qwen judge (Gemma-2-2b-it, rank 100, seed 42). Started 2026-10-10.

**Finished for all 18 concepts (2026-10-13).** Every concept has a complete
144-cell grid (run with exactly its EMBER δ), a Pass 1 shortlist and a
Qwen-judged Pass 2 pick. No pick loops, so no concept needed the loop screen.
**The `snmf_ember` cross-lingual run is done too (2026-10-14)**, job 1040077,
all 18 concepts × 5 languages, every edit check exact. It erases English the
most of the three methods, but it leaks (mean non-English Δ +0.17, Russian
+0.37). See "Cross-lingual run" below and
[`crosslingual_next.md`](crosslingual_next.md) §2.5.

> **Final picks after the MC scoring repair (2026-10-16):**
> `judge_rerun/SNMF_EMBER/pass2_rescored/google_gemma-2-2b-it/rank100/seed42/best_hps_summary.csv`.
> The original grid scoring cut answers off after 4 tokens ("The answer is **C**"
> counted as wrong); this method was the worst affected (15% invalid at the
> picks). After the repair **7 of 18 picks changed** and mean `harmonic_alpaca`
> fell from 0.909 to 0.888 (efficacy 0.877 → 0.834). See "Repair" below. The
> Results table further down shows the **pre-repair** picks. The cross-lingual
> run is redone too (2026-10-16): mean leak +0.17 → **+0.05 [−0.02, +0.11]**,
> Russian +0.23 [+0.08, +0.38]; English retention 0.39
> ([`crosslingual_next.md`](crosslingual_next.md) §2.6). The "Cross-lingual
> run" section below is the pre-repair run.

**Where the results are:** `judge_rerun/SNMF_EMBER/` (layout and policy:
[`judge_rerun/SNMF_EMBER/README.md`](../../judge_rerun/SNMF_EMBER/README.md)).
Final: `judge_rerun/SNMF_EMBER/pass2_rescored/…/best_hps_summary.csv`. Pre-repair:
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

## Cross-lingual run (2026-10-14, done)

The picks above go into the same judge-free cross-lingual MC test as `ember` and
`snmf` ([`crosslingual_next.md`](crosslingual_next.md) §1–§4). Each pick is
applied once and never re-tuned per language. Gemma alone then answers
`QA_test` and `SimdomQA_test` (50 MC questions each) in en/de/es/fr/ru, plus
English MMLU on the 1000-question test split. No judge is used.

- **Code:** `crosslingual/run_edited.py` takes `--methods snmf_ember`. It reads
  the picks from `judge_rerun/SNMF_EMBER/`, loads `configs/snmf_ember_gemma.yaml`
  and applies the pick through the `snmf` method with `delta_embed` set. That is
  how Pass 1 applied it, and snapshot/restore resets both the embeddings and the
  MLP. Before running, it refuses a pick whose `delta_embed` differs from the
  δ Pass 1 ran with. The edit check compares English `QA_train`, `SimdomQA_train`,
  50-question MMLU and the feature counts (`k_features_embed`, `n_tokens_edited`,
  `k_features_mlp_in/out`) against the search. A dry run resolved all 18 picks,
  and they match the Results table.
- **Job:** 1040077, all 18 concepts, 1× RTX 2080 Ti on `s-004`, 16G.
  `COMPLETED` 2026-10-14 20:28 after 2h22m, no preemption, no errors
  (7–10 min per concept). Command:
  `sbatch crosslingual/run_edited.sbatch --methods snmf_ember`. Finished concepts
  are skipped, so after a preemption resubmit the same command.
- **Output:** `results/crosslingual/google_gemma-2-2b-it/snmf_ember/<Concept>/<lang>.json`,
  `edit.json` (HP, edit check, MMLU) and `summary.csv` (90 rows = 18 × 5). Log:
  `cl_edited_1040077.out` in the repo root.
- **Edit check:** all 18 passed (`features_match` and `reproduces_search`
  true, Δ 0.0 on `QA_train`, `SimdomQA_train` and 50-q MMLU).
- **Results** (mean over concepts, 88 of 90 cells pass the screen, same 2
  dropped as for `ember` / `snmf`; full table and caveats in
  [`crosslingual_next.md`](crosslingual_next.md) §2.5):

  | | `ember` | `snmf` | `snmf_ember` |
  |---|---|---|---|
  | English `QA_test` acc (pristine 0.73) | 0.50 | 0.47 | **0.38** |
  | Retention r_en | 0.52 | 0.45 | **0.27** |
  | Mean leak Δ over de/es/fr/ru | +0.24 [+0.16, +0.32] | +0.01 [−0.07, +0.09] | +0.17 [+0.08, +0.27] |
  | Leak Δ ru | +0.41 | +0.05 | +0.37 |
  | `SimdomQA_test` retention, en / mean non-en | 0.96 / 0.98 | 0.73 / 0.64 | 0.81 / 0.74 |
  | English MMLU, 1000-q test (pristine 0.553) | 0.548 | 0.506 | 0.511 |
  | `QA_test` invalid rate, en / mean non-en | 0.09 / 0.05 | 0.00 / 0.00 | **0.16 / 0.08** |

  In short: it erases English best, but adding the EMBER embedding edit brings
  the leak back (most of it into Russian). It also has the most invalid
  answers, so part of its erasure is broken output.

## Repair (2026-10-15/16, done)

**Status: done (2026-10-16, 12:00).** All 24 jobs completed, no errors.
Re-score 50 min – 3 h 36 per job (~80 s per cell: SNMF+EMBER answers are
wordier), generation 16–23 min per group, Qwen judging 5 h 41 – 8 h 34 per
group. Results below.

The final EMBER δ are in
`judge_rerun/Ember/pass2_rescored/google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv`
([`ember_search.md`](ember_search.md), "MC scoring repair → Results"). Every
grid here was run with the concept's old EMBER δ. Same procedure as the SNMF
repair ([`snmf_search.md`](snmf_search.md), "MC scoring repair check"), with
three modes:

| Mode | Concepts | Cells re-scored |
|---|---|---|
| **Fresh grid** (EMBER δ changed: the old grid is a different edit) | Baseball (100 → 200), Gambling (100 → 10), Gun (100 → 5), Uranium (100 → 200) | all 144 at the new δ |
| **Partial** (δ unchanged: only cells with an invalid answer) | Ancient Rome 135, Artificial intelligence 141, COVID-19 130, Cannabis 125, Culture of Greece 135, Harry Potter 124, Heroin 138, Nazism 110, Republic of Ireland 138, Valentine's Day 144, Golf 19, Halloween 18, World War II 26 | as listed |
| **Full** (δ unchanged, but the systematic one-question Pornography difference would mix runs) | Pornography | all 144 |

- **Selection, unchanged:** all cells scored on the closed questions only
  (QA / SimDom / MMLU, no judge), `min_mmlu` gate, top 3 by the judge-free
  `harmonic`; then the first 50 Alpaca answers of the top 3, Qwen2.5-32B,
  pick by `harmonic_alpaca`. A cell still in the old top 3 (same δ, same SNMF
  cell) reuses its saved Alpaca answers (`pass1_outputs/…/cell_*.json`) and its
  Qwen scores; only cells new to the top 3 get answers generated (plus the loop
  check) and judged. The 4 fresh concepts reuse nothing (old answers are at the
  old δ), so at least 12 cells are judged.
- **Code:** `--with-ember` on `judge_rerun/rescore_snmf_grid.py` and
  `judge_rerun/pass2_snmf_rescored.py`: `configs/snmf_ember_gemma.yaml`,
  `results/snmf_ef/` grids, `judge_rerun/SNMF_EMBER/` folders, EMBER δ from the
  repaired picks. A concept whose grid δ differs from the repaired δ becomes a
  fresh grid automatically (`control_applicable = False`: no old run at that δ).
  As in the grid, a cell is scored on MMLU first and on QA / SimDom only if it
  passes the MMLU gate. **CPU check before submitting:** the scoring formula
  reproduces every row of all 18 SNMF+EMBER grids (max error 1e-16) and the
  ranking reproduces Pass 1's top 3 for all 18.
- **Output:** `judge_rerun/SNMF_EMBER/rescore_mc/` (re-scored cells,
  `compare.json`) and `judge_rerun/SNMF_EMBER/pass2_rescored/` (new Alpaca
  answers, Qwen scores, final picks).

**Jobs (24, chained with SLURM dependencies):**

| Stage | Jobs | Notes |
|---|---|---|
| Re-score (Gemma, 1 GPU each) | 1042781–1042796: one per concept, except Golf + Halloween + World War II together (1042796) | 12 run at once (per-user GPU cap) |
| Alpaca generation (Gemma, 1 GPU) | 1042797 (group 1), 1042799 (2), 1042801 (3), 1042803 (4) | each starts after all re-score jobs of its group succeed (`afterok`) |
| Qwen judging (3 GPUs) | 1042798 (1), 1042800 (2), 1042802 (3), 1042804 (4) | after its group's generation; Qwen loaded once per job, only if a concept needs it |

Groups (one fresh concept each, to spread the judging): 1 = Baseball, Ancient
Rome, Artificial intelligence, Golf, Halloween; 2 = Gambling, COVID-19,
Cannabis, World War II; 3 = Gun, Culture of Greece, Harry Potter, Heroin;
4 = Uranium, Nazism, Pornography, Republic of Ireland, Valentine's Day.

**To check by hand after the re-score:** near-ties at the 3rd place between a
re-scored and an old cell in a partial concept (re-score that concept in full if
so), and `best_hps_summary.csv`, which the 4 judging jobs write concurrently
(rebuild it from the per-concept `best_hp__*.json`). Then the cross-lingual
`snmf_ember` run again, with the fixed scoring (`crosslingual/run_edited.py`
must first read the repaired picks).

### Repair results

**Checks.** The control reproduced the October grid for every fully measured
cell except single-answer flips (Artificial intelligence 5, Heroin 3, Nazism 3,
COVID-19 1, Cannabis 1) and Pornography's systematic one-question difference
(85 cells; it was re-scored on all 144, so its ranking is one run). The failures
in Golf, Halloween and World War II (the concepts with the fewest re-scored
cells, where mixing runs matters most) were all in MMLU-gated cells, which
cannot reach a top 3. No near-tie between a re-scored and an old cell at any
3rd place. The fresh grids have no control (no old run at their δ).

**Top 3.** Unchanged for Golf, Halloween, Harry Potter and World War II (no
Alpaca, no judging). New cells: 3 each for the 4 fresh concepts, Ancient Rome
and Culture of Greece; 1–2 for the others. Every new cell passed the loop check
except Artificial intelligence row 83 (8 of 50 answers looping), which was
rejected; row 63 moved up. 27 new cells judged in all.

**Picks (7 changed):**

| Concept | EMBER δ | SNMF cell (δ_in@layers / δ_out@layers), old → new | efficacy | specificity | coherence | harmonic_alpaca old → new |
|---|---|---|---|---|---|---|
| Ancient Rome | 200 | 1@0-8 / 4@0-8 → **7@0-25 / 7@9-17** | 1.000 | 0.879 | 0.946 | 0.952 → 0.939 |
| Artificial intelligence | 100 | 4@0-8 / 4@0-8 | 0.764 | 0.942 | 0.935 | 0.892 → 0.872 |
| Baseball | 100 → **200** | 4@0-25 / 7@13-25 → **1@0-8 / 10@0-8** | 0.912 | 0.771 | 0.985 | 0.908 → 0.880 |
| COVID-19 pandemic | 50 | 4@0-25 / 4@0-8 | 0.970 | 0.818 | 0.953 | 0.908 → 0.908 |
| Cannabis | 200 | 10@0-12 / 1@0-8 | 0.610 | 0.826 | 0.991 | 0.857 → 0.778 |
| Culture of Greece | 200 | 1@0-8 / 1@0-8 → **4@0-25 / 10@13-25** | 1.000 | 0.919 | 0.985 | 0.984 → 0.967 |
| Gambling | 100 → **10** | 1@0-25 / 7@0-8 → **4@0-8 / 7@0-8** | 0.863 | 0.852 | 0.980 | 0.988 → 0.895 |
| Golf | 10 | 1@0-8 / 1@13-25 | 0.623 | 0.962 | 0.998 | 0.823 → 0.823 |
| Gun | 100 → **5** | 1@0-25 / 7@9-17 → **4@0-12 / 4@9-17** | 0.852 | 0.879 | 0.972 | 0.926 → 0.898 |
| Halloween | 10 | 10@0-25 / 1@0-8 | 0.727 | 0.940 | 0.993 | 0.871 → 0.871 |
| Harry Potter | 10 | 10@0-8 / 1@0-8 | 0.842 | 0.883 | 0.998 | 0.903 → 0.903 |
| Heroin | 100 | 10@0-25 / 4@9-17 | 0.821 | 0.864 | 0.990 | 0.905 → 0.886 |
| Nazism | 100 | 1@0-12 / 10@13-25 | 0.694 | 0.891 | 0.907 | 0.837 → 0.818 |
| Pornography | 5 | 4@0-8 / 10@9-17 | 0.898 | 0.947 | 0.881 | 0.921 → 0.908 |
| Republic of Ireland | 100 | 10@0-25 / 10@9-17 | 0.791 | 0.933 | 0.974 | 0.911 → 0.892 |
| Uranium | 100 → **200** | 4@0-12 / 10@13-25 → **7@0-25 / 4@0-8** | 0.923 | 0.912 | 0.972 | 0.967 → 0.935 |
| Valentine's Day | 200 | 1@0-12 / 10@0-8 → **1@0-12 / 7@0-8** | 0.878 | 0.938 | 0.938 | 0.923 → 0.917 |
| World War II | 5 | 10@0-25 / 1@0-8 | 0.836 | 0.891 | 0.975 | 0.897 → 0.897 |
| **Mean** | | | **0.834** | **0.892** | **0.965** | **0.909 → 0.888** |

SNMF+EMBER is still the best of the three methods on the selection score (EMBER
0.667, SNMF 0.820 after the repair), but by less than reported before.

The grids saved only per-cell totals, not per-question answers, so a cell is
re-scored in full.

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
