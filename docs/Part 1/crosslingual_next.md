# Cross-lingual leakage, round 2 — plan and status

**Status (2026-10-15):** **all three methods are done** for all 18 concepts ×
5 languages: `ember` and `snmf` (`crosslingual/run_edited.py`, jobs 1007121
pilot + 1007145 / 1007146, 2026-10-11) and `snmf_ember` (job 1040077,
2026-10-14). Every edit check passed. First-look results: §2.3 (`ember`,
`snmf`), §2.5 (`snmf_ember`); `snmf` leak bootstrap and its collateral damage
against the Gemini-era picks: §2.4. The pristine baselines were done before
(job 1006226). **Still to do:** the proper analysis (§6).

> **⚠ `ember` and `snmf_ember` numbers are provisional (2026-10-15).** "Invalid"
> answers are mostly not broken output. They are "The answer is **C**" cut off
> by the 4-token budget before the letter, and they count as wrong. That
> inflates erasure, and English more than the other languages (so Δ too). The
> picks themselves also come from the affected grid scoring. Both runs will be
> redone with the fixed scoring after the HP searches are repaired:
> [`ember_search.md`](ember_search.md), "MC scoring repair". `snmf` and the
> pristine baseline are not affected (0.3% invalid, lowercase letters).

[`crosslingual_plan.md`](../../archive_gemini/docs/crosslingual_plan.md) is still the
design reference (metric, screening rule, F1–F6 principles), and
[`archive_gemini/docs/crosslingual_results_summary.md`](../../archive_gemini/docs/crosslingual_results_summary.md)
holds the Gemini-era results that this round redoes.

## 1. The experiment

Erase a concept with the edit that was tuned **in English**, then ask the
concept's held-out multiple-choice questions (`QA_test`, 50 per concept) in
**en, de, es, fr, ru**. If the concept is still answerable in language L but
not in English, the erasure leaked.

Per (concept, method, language):

```
retention r_L = clip((acc_L − 0.25) / (pristine_acc_L − 0.25), 0, 1)   # per-language denominator
leak      Δ_L = r_L − r_en          # > 0: the concept survives better in L than in English
```

A (concept, language) cell is screened out when the pristine model scores
below 0.40 there (it never knew the concept in L). The edit is applied once and
never re-tuned per language. `SimdomQA_test` (50 similar-domain questions per
concept, also in all 5 languages) measures in-language specificity. Details:
`crosslingual_plan.md` §3–§4.

## 2. What is ready

| Input | Where | State |
|---|---|---|
| EMBER best δ, 18 concepts (Qwen judge) | `judge_rerun/Ember/pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv` | done |
| SNMF best HP, 18 concepts (Qwen judge) | `judge_rerun/SNMF/pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_hps_summary.csv` (full dict per concept in `<Concept>/best_hp__qwen2.5-32b-instruct.json`) | done ([`snmf_search.md`](snmf_search.md)) |
| Translated QA, 18 concepts × 5 languages, MC + open | `data/qa_lang/` | repaired and validated (`scripts/validate_qa_lang.py` → PASS) |
| MC evaluation code | `crosslingual/` (`data.py`, `prompts.py`, `mc_eval.py`, `run_baseline.py`, `run_edited.py`) | **new, works**; used for the baselines and the edited runs |
| Pristine baselines, 18 × 5, `QA_test` + `SimdomQA_test` | `results/crosslingual/google_gemma-2-2b-it/baseline/<Concept>/<lang>.json` + `summary.csv` | **done** (job 1006226, 42 min, 1 GPU) |
| Edited runs, `ember` + `snmf`, 18 × 5 | `results/crosslingual/google_gemma-2-2b-it/{ember,snmf}/<Concept>/<lang>.json` + `edit.json` (HP, edit check, MMLU), `summary.csv` per method | **done** (2026-10-11; pilot 1007121, then 1007145 / 1007146, ~2 h each on 1 GPU) |
| SNMF+EMBER best HP, 18 concepts (Qwen judge) | `judge_rerun/SNMF_EMBER/pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_hps_summary.csv` | done ([`snmf_ember_search.md`](snmf_ember_search.md)) |
| Edited run, `snmf_ember`, 18 × 5 | `results/crosslingual/google_gemma-2-2b-it/snmf_ember/` (same layout) | **done** (2026-10-14, job 1040077, 2h22m on 1 GPU) |
| Gemini-era results, 8 concepts × 3 methods, MC | `archive_gemini/results/cross_lang_test/` | reference only (see §2.2) |

The old Gemini-era runner (`run.py`, `best_hp.py`, `baseline.py`,
`evaluate.py`) was moved to `archive_gemini/crosslingual/` and is not used.

### 2.1 How the new package asks the questions

- **Prompt fully in the target language** (the `native` wrapper): instruction
  lines, question and options are all in L; only the answer letters stay Latin
  A–D, because the parser reads them back. The English prompt is byte-identical
  to the HP search's `ember.evals.mc.build_mc_prompt`. (`crosslingual/prompts.py`)
- **Language-aligned option shuffle**: the order is seeded by (concept, set,
  question index, seed) only, so question *i* has the same correct letter in
  every language (checked for all 90 cells). The model's letter-position bias
  is then identical across languages and cannot show up as a leak.
  (`crosslingual/data.py`)
- **Greedy decoding, `max_new_tokens=4`, first A–D letter parsed**, one question
  at a time, as in `ember.evals.mc.evaluate_mc_generation`. No judge anywhere.
- Every cell JSON keeps the per-question records (prompt options, raw output,
  parsed letter), so Δ can be paired by question.

### 2.2 Baseline results

Pristine `QA_test` accuracy (50 questions per cell, `*` = below the 0.40 cutoff):

| Concept | en | de | es | fr | ru |
|---|---|---|---|---|---|
| Ancient Rome | 0.80 | 0.58 | 0.68 | 0.64 | 0.48 |
| Artificial intelligence | 0.78 | 0.78 | 0.76 | 0.78 | 0.68 |
| Baseball | 0.66 | 0.62 | 0.60 | 0.56 | 0.48 |
| COVID-19 pandemic | 0.64 | 0.48 | 0.54 | 0.52 | 0.44 |
| Cannabis | 0.94 | 0.88 | 0.90 | 0.80 | 0.66 |
| Culture of Greece | 0.62 | 0.52 | 0.62 | 0.54 | 0.52 |
| Gambling | 0.84 | 0.80 | 0.76 | 0.70 | 0.58 |
| Golf | 0.80 | 0.76 | 0.72 | 0.72 | 0.50 |
| Gun | 0.82 | 0.66 | 0.64 | 0.62 | 0.66 |
| Halloween | 0.60 | 0.48 | 0.48 | 0.48 | 0.48 |
| Harry Potter | 0.80 | 0.60 | 0.58 | 0.50 | 0.36* |
| Heroin | 0.68 | 0.52 | 0.46 | 0.46 | 0.50 |
| Nazism | 0.74 | 0.68 | 0.72 | 0.66 | 0.50 |
| Pornography | 0.82 | 0.84 | 0.76 | 0.76 | 0.70 |
| Republic of Ireland | 0.74 | 0.66 | 0.54 | 0.58 | 0.48 |
| Uranium | 0.62 | 0.56 | 0.50 | 0.52 | 0.46 |
| Valentine's Day | 0.46 | 0.44 | 0.44 | 0.42 | 0.32* |
| World War II | 0.74 | 0.62 | 0.62 | 0.62 | 0.48 |
| **Mean** | **0.73** | **0.64** | **0.63** | **0.60** | **0.52** |

- 88 of 90 cells pass the 0.40 screen. Invalid (unparseable) answers are low;
  the worst is COVID-19 in one language at 0.14.
- **Valentine's Day** is weak everywhere, including English (0.46). With
  pristine − 0.25 ≈ 0.2 as the retention denominator, its Δ will be very noisy.
- **These do not reproduce the Gemini-era baselines, by design.** For the 8
  shared concepts, 36 of 40 cells differ (by up to 0.14). The questions are
  identical, but the old code used a different option shuffle: only 86 of the
  2,000 questions got the same option order, and the model gave the same answer
  on all 86. The other questions were asked with the options in a different
  order. The model has a letter preference (it picks C/D more than A/B), so on
  uncertain questions the order decides: 176 went right → wrong and 160
  wrong → right, which mostly cancels. So the differences come from the option
  order, not from a bug. Neither shuffle is "right"; the new one is better for
  Δ (§2.1). Consequence: the old plan's check "`ember` must reproduce the
  archived `qa_acc` exactly" no longer applies.

### 2.3 Edited results, `ember` and `snmf` (first look)

Plain means over concepts, from the two `summary.csv` files; 88 of 90 cells
pass the screen (same 2 as §2.2). Not yet the §6 analysis (no bootstrap).
± is the SE over concepts.

| | `ember` | `snmf` |
|---|---|---|
| English `QA_test` acc (pristine 0.73) | 0.50 | 0.47 |
| English retention r_en | 0.52 | 0.45 |
| leak Δ de | +0.17 ± 0.05 | −0.00 ± 0.04 |
| leak Δ es | +0.22 ± 0.06 | +0.00 ± 0.06 |
| leak Δ fr | +0.19 ± 0.05 | −0.02 ± 0.05 |
| leak Δ ru | **+0.41 ± 0.06** (r_ru 0.94) | +0.05 ± 0.06 |
| mean Δ over de/es/fr/ru | +0.24 | +0.01 |
| concepts with mean non-English Δ > 0 | 16 / 18 | 9 / 18 |
| `SimdomQA_test` acc, mean of de/es/fr/ru (pristine 0.74) | 0.73 | 0.57 |
| `SimdomQA_test` acc, en (pristine 0.89) | 0.87 | 0.73 |
| English MMLU, 1000-q test split (pristine 0.553) | 0.548 | 0.506 |
| `QA_test` invalid rate, en / other languages | 0.09 / 0.01–0.07 | 0.00 / 0.00–0.01 |

- **`ember` leaks; `snmf` does not.** The embedding edit erases English but
  much less elsewhere, and barely touches Russian (retention 0.94, close to
  the pristine model). SNMF erases about evenly in all five languages. Same
  direction as the Gemini-era run (`ember` Δ +0.25 there), now on 18 concepts.
- **SNMF pays in specificity.** Its similar-domain accuracy drops by ~0.15–0.17
  in every language and English MMLU by ~0.05; `ember` loses ~0.01–0.03 on
  similar-domain accuracy and ~0.005 on MMLU.
  SNMF's even erasure is partly a broader, less targeted edit. This is not
  caused by the new SNMF selection (§2.4).
- **`ember` produces unparseable answers** on some concepts, which count as
  wrong and so flatter its erasure: Cannabis es 0.36 / fr 0.30 / en 0.22,
  Gun en 0.26, Uranium en 0.26 / de 0.20, Gambling 0.16–0.20, COVID-19 de 0.18.
  `snmf`'s worst cell is 0.12 (Republic of Ireland de). Check these before
  reading per-concept Δ (§7).
- Per-concept non-English Δ is noisy, as expected (§7): `ember` ranges from
  −0.09 (COVID-19) to +0.53 (Gun); `snmf` from −0.38 (Republic of Ireland) to
  +0.36 (Halloween).
- **Edit check**: all 36 (concept, method) edits reproduced the search's
  English `qa_acc`, `simdom_acc` and `mmlu_acc` exactly (Δ 0.0, features match),
  logged in each `edit.json`.

### 2.4 `snmf`: is the leak zero, and is its damage new? (2026-10-11)

**Leak, bootstrapped over concepts** (10,000 resamples of the concepts, numpy
seed 0, 95% interval; same cells as §2.3):

| | de | es | fr | ru | mean of the 4 |
|---|---|---|---|---|---|
| `snmf` | −0.00 [−0.08, +0.07] | +0.00 [−0.10, +0.11] | −0.02 [−0.12, +0.08] | +0.05 [−0.06, +0.16] | +0.01 [−0.07, +0.09] |
| `ember` | +0.17 [+0.07, +0.27] | +0.22 [+0.11, +0.33] | +0.19 [+0.09, +0.29] | +0.41 [+0.28, +0.52] | +0.24 [+0.16, +0.32] |

- **`snmf` has no detectable leak.** Every interval contains 0, and about half
  the concepts go each way (de 9/18, es 9/18, fr 8/18, ru 9/16 positive). A mean
  leak above ~0.09 is ruled out. Every `ember` interval excludes 0.
- Per concept, `snmf`'s mean non-English Δ runs from −0.38 (Republic of
  Ireland: erased *more* outside English) to +0.36 (Halloween); the SD of one
  cell's Δ is 0.21. The three largest (Halloween +0.36, Uranium +0.27,
  Valentine's Day +0.24) have low pristine accuracy, so small denominators.

**Collateral damage.** The retention formula applied to `SimdomQA_test`
(clipped, mean over concepts), next to the concept's own retention:

| | de | en | es | fr | ru |
|---|---|---|---|---|---|
| `snmf` `QA_test` retention | 0.44 | 0.45 | 0.45 | 0.43 | 0.51 |
| `snmf` `SimdomQA_test` retention | 0.62 | 0.73 | 0.65 | 0.68 | 0.59 |
| `ember` `SimdomQA_test` retention | 0.98 | 0.96 | 0.98 | 0.98 | 1.00 |

`snmf` hits the concept harder than its neighbours in every language, but
also removes 30–40% of the neighbours. English MMLU (1000-question test split,
English only, not part of Δ) is the same story: `snmf` 0.506 on average
(0.442–0.541; worst Valentine's Day 0.44, Gambling 0.45) against pristine
0.553, `ember` 0.548 (0.537–0.553).

**The damage is not caused by the new SNMF selection.** The 144-cell grid is
unchanged (the 8 Gemini-era concepts reuse the archived grid); only the pick
from it changed: old = top 15 by `harmonic`, Alpaca judged by Gemini, max
`harmonic_alpaca`; new = top 3 × 50 Alpaca answers judged by Qwen2.5-32B, same
pick rule ([`snmf_search.md`](snmf_search.md)). For the 8 shared concepts, grid
metrics (`QA_train` etc.) of the two picks:

| Concept | Old pick (grid rank) | New pick (grid rank) | SimDom frac old → new | MMLU (50 q) old → new |
|---|---|---|---|---|
| Baseball | 1@0–12 / 10@0–8 (2) | same | 0.63 | 0.54 |
| Culture of Greece | 1@0–12 / 10@0–8 (2) | 1@0–25 / 10@0–8 (1), same scores = same edit | 0.85 | 0.54 |
| Gambling | 7@0–12 / 4@0–8 (1) | 10@0–12 / 7@9–17 (2) | 0.94 → 0.77 | 0.46 → 0.48 |
| Golf | 10@0–12 / 10@9–17 (2) | 4@0–25 / 7@13–25 (3) | 0.49 → 0.80 | 0.48 → 0.56 |
| Halloween | 10@0–25 / 7@13–25 (1) | 10@0–12 / 7@13–25 (2) | 0.77 → 0.77 | 0.50 → 0.52 |
| Harry Potter | 7@0–12 / 1@0–8 (1) | same | 0.85 | 0.54 |
| Uranium | 10@0–25 / 1@0–8 (2) | same | 0.73 | 0.56 |
| World War II | 10@0–25 / 1@13–25 (2) | same | 0.52 | 0.54 |
| **Mean** | | | **0.72 → 0.74** | **0.520 → 0.535** |

(HP: `δ_in @ layers / δ_out @ layers`. Pristine MMLU on these 50 questions: 0.52.)

- **5 of 8 picks are the same edit.** Of the 3 that changed, Gambling's new
  pick is more damaging, Golf's much less, Halloween's about equal. On average
  the new picks are slightly *less* damaging.
- **The Gemini-era run had the same damage** on the cross-lingual test (same
  8 concepts, `SimdomQA_test` retention):

  | | en | de | es | fr | ru |
  |---|---|---|---|---|---|
  | Gemini-era `snmf` | 0.65 | 0.60 | 0.58 | 0.58 | 0.59 |
  | Round 2 `snmf` | 0.71 | 0.64 | 0.66 | 0.68 | 0.55 |

- **Its MMLU damage was hidden, not absent.** The archived summary's "MMLU
  retained 0.97" for `snmf` comes from the §4.2 bug: `mmlu_acc_en` is 0.541 for
  all 8 concepts, which is exactly what round 2 measures for Harry Potter's
  (identical) edit. Edits identical to the old picks score World War II 0.49
  and Uranium 0.50 on the 1000-question split. The grid's 50-question MMLU
  (binomial SE ≈ 0.07) cannot see a ~0.05 drop, so neither search could
  select against it.
- **Same edit, different harness, different Δ.** For the 5 identical edits the
  per-concept leak still moved between the runs (Uranium +0.02 → +0.27,
  Culture of Greece +0.10 → −0.20), from the option shuffle and prompt alone
  (§2.2). Over the 8 concepts: Gemini-era +0.10, round 2 +0.05. Both small.

Conclusion: the broad damage is a property of SNMF on this grid, not of the
new selection. Reducing it would take a different selection objective (a
larger MMLU set, or a SimDom constraint), not a different judge or top-k.

### 2.5 Edited results, `snmf_ember` (first look, 2026-10-15)

Job 1040077 (2026-10-14, 2h22m, no preemption). All 18 edit checks exact
(Δ 0.0 on `QA_train`, `SimdomQA_train`, 50-q MMLU; feature counts match).
Same cells as §2.3 (88 of 90 pass the screen). ± is the SE over concepts;
brackets are the 95% bootstrap over concepts (10,000 resamples,
`np.random.default_rng(0)`; recomputing `ember` / `snmf` this way moves their
§2.4 bounds by at most 0.01).

| | `ember` | `snmf` | `snmf_ember` |
|---|---|---|---|
| English `QA_test` acc (pristine 0.73) | 0.50 | 0.47 | **0.38** |
| English retention r_en | 0.52 | 0.45 | **0.27** |
| Retention de / es / fr / ru | 0.69 / 0.74 / 0.71 / 0.94 | 0.44 / 0.45 / 0.43 / 0.51 | 0.38 / 0.39 / 0.36 / 0.65 |
| leak Δ de | +0.17 ± 0.05 | −0.00 ± 0.04 | +0.11 ± 0.06 [−0.01, +0.23] |
| leak Δ es | +0.22 ± 0.06 | +0.00 ± 0.06 | +0.12 ± 0.05 [+0.02, +0.23] |
| leak Δ fr | +0.19 ± 0.05 | −0.02 ± 0.05 | +0.09 ± 0.05 [+0.00, +0.19] |
| leak Δ ru | +0.41 ± 0.06 | +0.05 ± 0.06 | **+0.37 ± 0.09** [+0.20, +0.54] |
| mean Δ over de/es/fr/ru | +0.24 [+0.16, +0.32] | +0.01 [−0.07, +0.09] | **+0.17** [+0.08, +0.27] |
| concepts with mean non-English Δ > 0 | 16 / 18 | 9 / 18 | 14 / 18 |
| `SimdomQA_test` acc, mean of de/es/fr/ru (pristine 0.74) | 0.73 | 0.57 | 0.61 |
| `SimdomQA_test` acc, en (pristine 0.89) | 0.87 | 0.73 | 0.78 |
| `SimdomQA_test` retention, en / de / es / fr / ru | 0.96 / 0.98 / 0.98 / 0.98 / 1.00 | 0.73 / 0.62 / 0.65 / 0.68 / 0.59 | 0.81 / 0.69 / 0.75 / 0.77 / 0.76 |
| English MMLU, 1000-q test (pristine 0.553) | 0.548 | 0.506 | 0.511 (0.435–0.549) |
| `QA_test` invalid rate, en / de / es / fr / ru | 0.09 / 0.05 / 0.07 / 0.06 / 0.01 | 0.00 / 0.01 / 0.00 / 0.00 / 0.00 | **0.16** / 0.11 / 0.10 / 0.10 / 0.02 |

- **Strongest English erasure of the three** (English `QA_test` 0.73 → 0.38,
  r_en 0.27), in line with its top selection score
  ([`snmf_ember_search.md`](snmf_ember_search.md)).
- **But it leaks.** Mean non-English Δ +0.17, interval excludes 0; 14 of 18
  concepts positive. Most of it is Russian (+0.37, retention 0.65), which is the
  same pattern as `ember` (+0.41). de/es/fr are smaller (+0.09 to +0.12) and
  de's interval touches 0. Adding the EMBER embedding edit on top of SNMF
  brings back an `ember`-like leak: the embedding part erases English (and
  Latin-script languages) more than Russian. Same ordering as the Gemini-era
  run (`snmf_ember` +0.22 vs `ember` +0.25 there).
- **Specificity sits between the two**: `SimdomQA_test` retention 0.69–0.81
  (vs `snmf` 0.59–0.73, `ember` 0.96–1.00); English MMLU 0.511, about the same
  as `snmf` (worst Valentine's Day 0.435, COVID-19 0.440).
- **Most invalid answers of the three.** English 0.16 on average, with Gun en
  0.42, Uranium en 0.36 / de 0.52, Gambling en 0.34, Cannabis en 0.32 / es 0.36,
  Valentine's Day de 0.56 / es 0.46 / fr 0.48, COVID-19 ru 0.32. Invalid answers
  count as wrong, so part of the erasure is broken output. English is hit more
  than de/es/fr/ru, which flatters r_en and so inflates Δ. This needs the
  invalid-dropped re-score from §6 before Δ is read per concept.
- **Floor effects.** Three concepts land at or below chance in English
  (COVID-19 0.22, Uranium 0.20, Valentine's Day 0.18 → r_en = 0 after the
  clip), so their Δ cannot be negative (Uranium ru +1.00). Per-concept Δ runs
  from −0.18 (Pornography) to +0.57 (Gun); one cell's SD is 0.29.

Per-concept table: [`part1_results.md`](part1_results.md) §5.3.

## 3. Scope (decided)

| Choice | Decision | Why |
|---|---|---|
| Format | **MC only** | judge-free, cheap, and the Gemini-era headline was MC too. Open QA is a later step (§6) |
| Concepts | **all 18** | the Gemini run had n=8 and its method differences landed at t≈2.0, just short of significance (\|t\|≈2.4 needed). One cell's Δ has noise ±0.30, so only averages over concepts mean anything. n=18 cuts the across-concept SE by ~1.5× |
| Methods | **`pristine`, `ember`, `snmf`, `snmf_ember` (all done)** | `snmf_ember` used its own HP search; see §5 |
| Languages | en, de, es, fr, ru | 2 cells screened out (Harry Potter/ru, Valentine's Day/ru) |
| Wrapper | **`native` only for now** | the Gemini run found the wrapper effect negligible (mean −0.014). The `english` wrapper (English instructions, translated question) is not implemented in the new package; add it to `prompts.py` if wanted |
| Test split | `QA_test` + `SimdomQA_test` | `QA_train` picked the HPs, so it would flatter English |

**Cost.** The baseline took ~28 s per (concept, language) cell on one
RTX 2080 Ti (both sets). An edited model is the same 5 cells plus applying
the edit and one English MMLU pass: a few minutes per (concept, method), so
**~2–3 GPU-hours for all 36 edited models**. One 1-GPU job (or a few) is enough;
no Qwen, no 3-GPU pinning. Actual: ~7 min per (concept, method) including the
edit check, ~2 h per method for 17 concepts, ~4.5 GPU-hours in all.

## 4. The edited-model runner (done)

**Built and run (2026-10-10/11)** as `crosslingual/run_edited.py` +
`run_edited.sbatch` (1 RTX 2080 Ti; with no arguments it runs `ember` + `snmf`
× 18 concepts). The design below is what it implements; the module docstring
has the details.

New CLI next to `run_baseline.py` that reuses `mc_eval.evaluate_cell` unchanged, so edited and
pristine runs see the same items, shuffle, prompts and decoding. Per
(concept, method): load the model, apply the edit, evaluate all 5 languages,
restore or reload, next.

### 4.1 Reading the picks

- `ember`: `{"delta_embed": best_delta_embed}` from `best_deltas_summary.csv`.
  `ember.hp_columns()` (`io.EMBED_COLUMNS`) also lists `k_features_embed` and
  `n_tokens_edited`, but `pass1_ember_delta.py` records those from the
  edit's returned info rather than passing them in. Confirmed: `delta_embed`
  alone reproduces the search's scores and feature counts (edit check below).
- `snmf`: `best_hp` from `<Concept>/best_hp__qwen2.5-32b-instruct.json`, plus
  `delta_embed = 0.0`, exactly as `judge_rerun/pass1_snmf_top3.py::_build_hp`
  applies it. `neurons_thresh` is in `io.SNMF_HP_COLUMNS` but not in the JSON
  (it is empty in the grid CSV too), so pass it the same way `_build_hp` does.
- Load the YAML from `configs/ember_gemma.yaml` / `snmf_gemma.yaml`, since
  `method.apply` reads other config fields from it.
- **Check the edit before trusting it:** the English `QA_test` accuracy is not
  comparable to the search's (different split), but the edited model's
  `QA_train` English accuracy (or its MMLU) should match the grid row's
  `qa_acc` / `mmlu_acc` for the same HP. That replaces the old
  "reproduce the archive" check. **Result:** all 36 edits matched the search
  exactly on `QA_train`, `SimdomQA_train` and the 50 MMLU-train questions
  (Δ 0.0), recorded in each `<method>/<Concept>/edit.json`.

Write the resolved HP into every output cell, so each result says which edit
produced it.

### 4.2 MMLU: one per edited model, keyed by concept

The Gemini-era `baseline.py` cached an edited model's MMLU under
`_mmlu_by_config/<config>/` with **no concept in the path**, so the first
concept's MMLU was reused for every later concept (`mmlu_acc_en` is identical
for all 8 concepts within each method in that run). Its specificity /
harmonic / "MMLU retained" columns are wrong; its retention and Δ (the headline)
do not use MMLU and are unaffected. The new runner must compute MMLU per
(concept, method), using the same MMLU set as the HP search (pristine
`mmlu_acc = 0.52`).

As built: the 50-question MMLU-train set (pristine 0.52) is used only inside
the edit check. The reported English MMLU is the 1000-question **test** split,
computed per (concept, method); pristine 0.553, in `baseline/_mmlu.json`.

### 4.3 Output

`results/crosslingual/google_gemma-2-2b-it/<method>/<Concept>/<lang>.json`, the
same cell format as `baseline/` (plus the HP and MMLU), and a `summary.csv`
per method. Skip existing cells, so a preempted job resumes.

As built: each concept folder also has `edit.json` (HP, edit check, MMLU).
`summary.csv` has one row per (concept, language) with `reliable`,
`QA_test_acc` / `_pristine`, `retention`, `leak`, invalid rates, SimDom
accuracy, `mmlu_test_acc_en` and the HP.

## 5. `snmf_ember` — options

**Decided 2026-10-10: option 2, the proper re-run** (EMBER δ reused, SNMF grid
searched again on top of it). **Done:** search finished 2026-10-13
([`snmf_ember_search.md`](snmf_ember_search.md)); `snmf_ember` added to
`run_edited.py` (`METHODS`, reading `judge_rerun/SNMF_EMBER/`, config
`configs/snmf_ember_gemma.yaml`) and the 18 × 5 test run 2026-10-14
(job 1040077). Results: §2.5.

The paper's headline config. The Gemini run found it leaks about as much as
`ember` (Δ +0.22 vs +0.25) while erasing English harder.

1. **Skip it in round 2** (recommended). `ember` vs `snmf` is already the core
   contrast the question asks about (embedding edit vs MLP edit).
2. **Proper re-run**: the SNMF grid with each concept's EMBER δ applied
   (`configs/snmf_ember_gemma.yaml`, results under `snmf_ef/`), through the
   same Pass 1 / Pass 2 split. ~2h10m per concept for the fresh 144-cell grid
   (1 GPU), plus ~2.5 h of Qwen judging (3 GPUs). That is ~80 GPU-hours for
   18 concepts, a separate project step.
3. **Composed shortcut**: apply the best SNMF HP + the best EMBER δ without
   tuning them together. Cheap, but it is not the paper's method, so it
   must be labelled as such.

## 6. After the edited runs

- **Analysis**: Δ per method and language, averaged over concepts with a
  bootstrap over concepts (the Gemini summary §5 has the code). Report how many
  cells were screened out, and `invalid_rate` per language next to retention.
  Plain means for `ember` / `snmf` are in §2.3, the per-language bootstrap of Δ
  in §2.4; the paired `ember` − `snmf` difference per (concept, language) is
  still to do (also `snmf_ember` − `ember` and `snmf_ember` − `snmf`; plain
  means for `snmf_ember` in §2.5), and so is a "targeted leak" that discounts
  the SimDom drop. Also check whether the high-invalid cells of `ember` (§2.3)
  and especially `snmf_ember` (§2.5, English invalid 0.16) change Δ, e.g. by
  re-scoring with invalid answers dropped.
- **Open QA** with the decoupled design: generate answers (Gemma, 1 GPU), then
  judge them (Qwen2.5-32B, 3 GPUs). Add the multilingual line to the judge
  prompt (`crosslingual_plan.md` §5.7). Hand-check ~30 judged items per
  language (Semyon can check Russian), and watch for refusals on sensitive
  concepts (Nazism, Heroin, Pornography, Gun, Cannabis).
- **Token overlap → leak** (`crosslingual_plan.md` §8): the "money figure",
  CPU only. With 18 concepts the Gemini-era correlations (r −0.68 / −0.91 at n=8)
  can actually be tested.
- **Cross-lingual relearning** (`crosslingual_plan.md` §10).
- Part 2 ideas (zero-shot foreign edit directions, layer sweep):
  [`Part 2/ideas_and_discoveries.md`](../Part%202/ideas_and_discoveries.md).

## 7. Risks to watch

- **A broken model looks like successful erasure.** If generation collapses,
  MC answers stop parsing and count as wrong. Always report `invalid_rate` per
  language next to retention. SNMF picks have no Alpaca gate (Pornography 0.85,
  Golf 0.87); Ancient Rome's broken pick was replaced (`snmf_search.md`).
- **Near-chance baselines**: the 0.40 screen handles Russian; Valentine's Day is
  near-chance-ish in every language (§2.2).
- **Noise**: n=50 per cell, and reshuffling the options alone moves a cell by
  up to ±0.14 (§2.2). Report averages over concepts with SEs or a bootstrap,
  never single cells.
- **Specificity is mixed** (English MMLU + in-language SimDom). Say so.

## 8. Decisions

Resolved: all 18 concepts; Ancient Rome SNMF fixed (loop screen); SNMF Pass 2
finished for all concepts before the cross-lingual runs; `snmf_ember` = proper
re-run (§5, search done 2026-10-13); `ember` + `snmf` cross-lingual runs done
(2026-10-11), `snmf_ember` done (2026-10-14).

Still open: add the `english` wrapper check, or report `native` only?

## 9. Implementation order

1. ~~Pristine baselines~~ — done (job 1006226).
2. ~~`crosslingual/run_edited.py` + sbatch (§4)~~ — done.
3. ~~Pilot: Harry Potter × `ember` + `snmf`~~ — done (job 1007121; edit check
   exact, English `QA_test` drops 0.80 → 0.60 / 0.58).
4. ~~Remaining 17 concepts~~ — done (jobs 1007145 `ember`, 1007146 `snmf`,
   2026-10-11). First look: §2.3.
5. ~~`snmf_ember`: wait for its search, add it to `run_edited.py`, run 18 × 5~~ —
   done (job 1040077, 2026-10-14). First look: §2.5.
6. Analysis (§6).
