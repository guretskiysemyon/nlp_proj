# Part 1 — results

All Part 1 results in one place: the judge choice, the three HP searches
(EMBER, SNMF, SNMF+EMBER) and the cross-lingual leakage test. Model
Gemma-2-2b-it, rank 100, seed 42, 18 concepts. Snapshot of 2026-10-15.

Detailed write-ups (how each number was made, job IDs, re-running):
[`ember_search.md`](ember_search.md), [`snmf_search.md`](snmf_search.md),
[`snmf_ember_search.md`](snmf_ember_search.md),
[`crosslingual_next.md`](crosslingual_next.md), and
[`../progress.md`](../progress.md) §3.1 for the judge.

> **⚠ Partly provisional (2026-10-15).** The MC scoring cut answers off after
> 4 tokens and counted "The answer is **C**" as wrong. **Repaired:** the EMBER
> and SNMF HP searches (§4 shows the repaired numbers; EMBER: 4 picks changed,
> SNMF: none). **Not yet repaired:** everything SNMF+EMBER, and the
> cross-lingual `ember` / `snmf_ember` numbers (§1, §5), which still use the old
> picks and the old scoring and are inflated by it, SNMF+EMBER most. Repair:
> [`ember_search.md`](ember_search.md), "MC scoring repair".

**Not done yet in Part 1:** the paired method differences (and re-scoring
with invalid answers dropped), open-QA cross-lingual judging, the final-test
stage and relearning.

---

## 1. Headline

| | EMBER | SNMF | SNMF+EMBER |
|---|---|---|---|
| Selection score `harmonic_alpaca` (mean of 18, `QA_train`) | 0.667 | 0.820 | **0.909** ² |
| Efficacy | 0.430 | 0.721 | **0.877** ² |
| Specificity | **0.970** | 0.848 | 0.909 ² |
| Coherence | **0.992** | 0.959 | 0.960 ² |
| English `QA_test` retention r_en (held out; lower = stronger erasure) | 0.52 ² | 0.45 | **0.27** ² |
| Cross-lingual leak, mean Δ over de/es/fr/ru (`QA_test`) | +0.24 [+0.16, +0.32] ² | **+0.01** [−0.07, +0.09] | +0.17 [+0.08, +0.27] ² |
| English MMLU, 1000-q test (pristine 0.553) | **0.548** ² | 0.506 | 0.511 ² |

² Not yet repaired: old 4-token scoring and/or old picks. EMBER's HP-search rows
are repaired; its cross-lingual rows still use the old picks (4 differ) and scoring.

1. **SNMF+EMBER is the best edit on the English selection score** for 17 of 18
   concepts (Harry Potter is the exception, within noise). These scores are
   measured on the items used for selection, so they are optimistic, and they
   are not yet repaired (SNMF+EMBER is the method most inflated by the cut-off
   answers). After the repair, **EMBER erases less than first reported**: mean
   efficacy 0.43, not 0.52 (§4.2).
2. **EMBER leaks across languages.** Tuned in English, it erases English but
   much less in other languages, and almost not at all in Russian (retention
   0.94). All four per-language bootstrap intervals exclude 0.
3. **SNMF does not leak** (every interval contains 0), but it damages more:
   similar-domain accuracy drops ~0.15 in every language and English MMLU drops
   ~0.05. The damage comes from SNMF on this grid, not from our new selection
   procedure (§5.5).
4. **SNMF+EMBER erases English the most on held-out questions, but leaks
   again.** Adding the EMBER embedding edit on top of SNMF brings back an
   EMBER-like leak, mostly into Russian (+0.37). Its specificity is between
   the two, and it has the most invalid answers (English 0.16), so part of its
   erasure is broken output (§5.4).

---

## 2. Judge: Qwen2.5-32B-Instruct

Gemini (the paper's judge) was replaced with a free local judge from a
different model family than the erased model. Candidates were scored by
re-judging Gemini's 150 saved Alpaca completions item by item. Gate: mean
difference from Gemini within ±0.15.

| | Qwen2.5-14B | **Qwen2.5-32B** | Gemma-4-12B |
|---|---|---|---|
| instruct mean Δ vs Gemini | −0.100 | **−0.073** | +0.013 |
| instruct exact match | 91.3% | **92.7%** | — |
| fluency mean Δ vs Gemini | +0.200 (fails) | **+0.040** | +0.367 (fails) |
| fluency exact match | 61.3% | 64.0% | — |
| parse failures | 0/300 | 0/300 | — |

- 32B passes both axes with Gemini's original prompts. 14B fails fluency even
  with a Qwen-specific prompt fix. Gemma-4-12B fails fluency and is the same
  family as the erased model.
- Per-item fluency agreement is only ~64% for every judge, but the errors
  cancel in the mean, which is what `harmonic_alpaca` uses. Do not trust
  single-item scores.
- Pristine Alpaca baseline with this judge (150 items): instruct **1.927**,
  fluency **1.687**. All three searches use these as denominators, so their
  scores are directly comparable.
- Pristine reference values: MMLU 0.52 (50-item selection set, same as the
  Gemini-era baseline), 0.553 (1000-item test split).

Files: `docs/judge/judge_selection/`.

---

## 3. Selection procedure (all three searches)

```
efficacy        = 1 − clip((qa_acc − 0.25) / (qa_base − 0.25), 0, 1)
specificity     = hmean(mmlu_frac, simdom_frac)
coherence       = hmean(min(1, instr / instr_base), min(1, flu / flu_base))
harmonic        = hmean(efficacy, specificity)                 # judge-free
harmonic_alpaca = hmean(efficacy, specificity, coherence)      # final pick
```

- **Two passes**: Pass 1 generates without a judge and takes the top 3
  candidates by the judge-free `harmonic`. Pass 2 judges the first 50 Alpaca
  answers of each candidate with Qwen-32B and picks the highest
  `harmonic_alpaca`.
- **EMBER**: 8 candidate δ, MMLU gate (`mmlu_frac ≥ 0.9`), Alpaca gate (both
  fractions ≥ 0.9).
- **SNMF / SNMF+EMBER**: a 144-cell HP grid and **no Alpaca gate** (matching the
  original pipeline). SNMF+EMBER fixes the concept's EMBER δ and re-searches
  the SNMF grid on top.
- All selection metrics are measured on `QA_train`, a 50-item MMLU set and
  `SimdomQA_train`.

---

## 4. HP search results (English, selection items)

### 4.1 Per concept, `harmonic_alpaca`

EMBER and SNMF after the MC scoring repair; SNMF+EMBER not yet repaired (²).

| Concept | EMBER δ | EMBER | SNMF | SNMF+EMBER |
|---|---|---|---|---|
| Ancient Rome | 200 | 0.816 | 0.802 ¹ | **0.952** |
| Artificial intelligence | 100 | 0.442 | 0.857 | **0.892** |
| Baseball | **200** | 0.428 | 0.806 | **0.908** |
| COVID-19 pandemic | 50 | 0.630 | 0.837 | **0.908** |
| Cannabis | 200 | 0.568 | 0.703 | **0.857** |
| Culture of Greece | 200 | 0.938 | 0.844 | **0.984** |
| Gambling | **10** | 0.720 | 0.867 | **0.988** |
| Golf | 10 | 0.672 | 0.758 | **0.823** |
| Gun | **5** | 0.663 | 0.867 | **0.926** |
| Halloween | 10 | 0.659 | 0.787 | **0.871** |
| Harry Potter | 10 | 0.767 | **0.929** | 0.903 |
| Heroin | 100 | 0.795 | 0.840 | **0.905** |
| Nazism | 100 | 0.772 | 0.805 | **0.837** |
| Pornography | 5 | 0.592 | 0.864 | **0.921** |
| Republic of Ireland | 100 | 0.589 | 0.740 | **0.911** |
| Uranium | **200** | 0.621 | 0.716 | **0.967** |
| Valentine's Day | 200 | 0.684 | 0.909 | **0.923** |
| World War II | 5 | 0.659 | 0.829 | **0.897** |
| **Mean** | | **0.667** | **0.820** | **0.909** ² |

¹ After the loop screen (§4.4).

Mean over the 18 concepts (median in brackets):

| Method | efficacy | specificity | coherence | harmonic_alpaca |
|---|---|---|---|---|
| EMBER | 0.430 (0.403) | 0.970 (0.965) | 0.992 (0.998) | 0.667 (0.661) |
| SNMF | 0.721 (0.698) | 0.848 (0.858) | 0.959 (0.962) | 0.820 (0.833) |
| SNMF+EMBER ² | **0.877** (0.890) | 0.909 (0.906) | 0.960 (0.974) | **0.909** (0.908) |

### 4.2 EMBER

After the MC scoring repair (final). Pre-repair values: [`ember_search.md`](ember_search.md), "MC scoring repair → Results".

| Concept | δ | efficacy | specificity | coherence | harmonic_alpaca |
|---|---|---|---|---|---|
| Ancient Rome | 200 | 0.618 | 0.947 | 0.998 | 0.816 |
| Artificial intelligence | 100 | 0.211 | 1.000 | 0.966 | 0.442 |
| Baseball | **200** (was 100) | 0.203 | 0.925 | 0.992 | 0.428 |
| COVID-19 pandemic | 50 | 0.364 | 0.986 | 0.998 | 0.630 |
| Cannabis | 200 | 0.305 | 1.000 | 0.992 | 0.568 |
| Culture of Greece | 200 | 0.857 | 0.969 | 1.000 | 0.938 |
| Gambling | **10** (was 100) | 0.471 | 0.962 | 0.998 | 0.720 |
| Golf | 10 | 0.407 | 1.000 | 0.992 | 0.672 |
| Gun | **5** (was 100) | 0.407 | 0.935 | 1.000 | 0.663 |
| Halloween | 10 | 0.400 | 0.962 | 0.990 | 0.659 |
| Harry Potter | 10 | 0.526 | 1.000 | 0.990 | 0.767 |
| Heroin | 100 | 0.564 | 1.000 | 0.998 | 0.795 |
| Nazism | 100 | 0.549 | 0.940 | 0.998 | 0.772 |
| Pornography | 5 | 0.327 | 1.000 | 0.998 | 0.592 |
| Republic of Ireland | 100 | 0.326 | 0.984 | 0.998 | 0.589 |
| Uranium | **200** (was 100) | 0.359 | 0.962 | 0.998 | 0.621 |
| Valentine's Day | 200 | 0.439 | 0.934 | 0.962 | 0.684 |
| World War II | 5 | 0.400 | 0.956 | 0.992 | 0.659 |

- Six concepts were judged directly with the top-3 × 50 policy. For the other
  12, the pick was re-derived from cached full-grid judge scores (8 δ × 150).
  Ten of those 12 match the old full-grid pick. The two that changed then,
  Baseball (200 → 100) and Halloween (5 → 10), are near-ties. (Both before the
  MC scoring repair; the repair moved Baseball back to 200.)
- EMBER barely hurts the model (specificity 0.97, coherence 0.99) but erases
  well under half the concept on average (efficacy 0.43; 0.52 before the
  repair, when cut-off "The answer is **C**" answers counted as erased).
- The repair changed 4 picks: Baseball 100 → 200, Gambling 100 → 10, Gun
  100 → 5, Uranium 100 → 200. Uranium fell most (efficacy 0.82 → 0.36): most
  of its "erasure" at δ = 100 was cut-off answers, 10 of 14 of them correct.
- EMBER's Alpaca answers hardly depend on δ (48–50 of 50 identical across δ),
  so coherence is nearly constant and the pick is decided by efficacy and
  specificity.

Files: `judge_rerun/Ember/pass2_rescored/.../best_deltas_summary.csv` (final); pre-repair `judge_rerun/Ember/pass2_outputs/`.

### 4.3 SNMF

HP notation: `δ @ layers lo–hi (k features)`. `in` is MLP `up_proj`, `out` is
`down_proj`. Bold min frac is below 0.9.

| Concept | δ_in | δ_out | efficacy | specificity | coherence | harmonic_alpaca | min Alpaca frac |
|---|---|---|---|---|---|---|---|
| Ancient Rome ¹ | 7 @ 0–8 (k=13) | 1 @ 9–17 (k=4) | 0.618 | 0.934 | 0.948 | 0.802 | 0.90 |
| Artificial intelligence | 4 @ 0–25 (k=51) | 4 @ 0–8 (k=36) | 0.691 | 0.957 | 0.993 | 0.857 | 0.99 |
| Baseball | 1 @ 0–12 (k=33) | 10 @ 0–8 (k=29) | 0.702 | 0.771 | 0.998 | 0.806 | 1.00 |
| COVID-19 pandemic | 7 @ 0–8 (k=31) | 4 @ 9–17 (k=13) | 0.788 | 0.772 | 0.980 | 0.837 | 0.97 |
| Cannabis | 4 @ 0–8 (k=25) | 4 @ 0–8 (k=25) | 0.508 | 0.788 | 0.970 | 0.703 | 0.94 |
| Culture of Greece | 1 @ 0–25 (k=14) | 10 @ 0–8 (k=10) | 0.694 | 0.919 | 0.975 | 0.844 | 0.97 |
| Gambling | 10 @ 0–12 (k=26) | 7 @ 9–17 (k=6) | 0.863 | 0.808 | 0.939 | 0.867 | 0.93 |
| Golf | 4 @ 0–25 (k=35) | 7 @ 13–25 (k=10) | 0.590 | 0.887 | 0.879 | 0.758 | **0.87** |
| Gun | 4 @ 0–8 (k=36) | 4 @ 9–17 (k=4) | 0.852 | 0.806 | 0.955 | 0.867 | 0.91 |
| Halloween | 10 @ 0–12 (k=16) | 7 @ 13–25 (k=8) | 0.606 | 0.873 | 0.986 | 0.787 | 0.98 |
| Harry Potter | 7 @ 0–12 (k=16) | 1 @ 0–8 (k=15) | 0.877 | 0.919 | 0.998 | 0.929 | 1.00 |
| Heroin | 7 @ 0–25 (k=32) | 10 @ 9–17 (k=3) | 0.667 | 0.958 | 0.972 | 0.840 | 0.94 |
| Nazism | 10 @ 0–12 (k=25) | 1 @ 9–17 (k=9) | 0.980 | 0.608 | 0.942 | 0.805 | 0.91 |
| Pornography | 4 @ 0–8 (k=23) | 10 @ 9–17 (k=7) | 0.776 | 0.947 | 0.887 | 0.864 | **0.85** |
| Republic of Ireland | 1 @ 0–25 (k=9) | 10 @ 0–8 (k=5) | 0.558 | 0.826 | 0.952 | 0.740 | 0.95 |
| Uranium | 10 @ 0–25 (k=36) | 1 @ 0–8 (k=17) | 0.513 | 0.844 | 0.948 | 0.716 | 0.90 |
| Valentine's Day | 1 @ 0–12 (k=14) | 10 @ 0–8 (k=13) | 0.829 | 0.955 | 0.955 | 0.909 | 0.91 |
| World War II | 10 @ 0–25 (k=15) | 1 @ 13–25 (k=3) | 0.873 | 0.685 | 0.988 | 0.829 | 0.98 |

Files: `judge_rerun/SNMF/pass2_rescored/.../best_hps_summary.csv` (after the MC scoring repair: same 18 cells as `judge_rerun/SNMF/pass2_outputs/`; only Pornography's numbers moved, by one `QA_train` question that answers differently on today's nodes, see [`snmf_search.md`](snmf_search.md)).

### 4.4 Ancient Rome: broken SNMF picks and the loop screen

- All three of Ancient Rome's original top SNMF cells broke free-text
  generation. 45, 45 and 49 of 50 Alpaca answers collapsed into "Roman Roman
  Roman …" (judge fluency 0.14 / 0.14 / 0.0, pristine 1.69).
- The cause is the selection score. The judge-free `harmonic` uses
  multiple-choice tests only, and those cannot see broken free text. The top 8
  grid rows all used a strong late output edit (δ_out 7–10 in layers 13–25).
  Across all 45 judged SNMF cells, the 10 with that pattern averaged 14.5
  looping answers and fluency 0.99. The other 35 averaged 0.5 looping answers
  and fluency 1.63.
- **Fix, for this concept only:** walk down the grid ranking and skip
  duplicates and looping cells. An answer loops when a word repeats ≥8 times in
  a row. A cell is skipped when >5 of its 50 answers loop. The walk stopped at
  the first 3 healthy cells (16 rows checked). The new pick is grid rank 9:
  efficacy 0.62, coherence 0.95, `harmonic_alpaca` 0.80.
- The unedited model has 0/150 looping answers. Applying the same screen to the
  other 17 concepts would change no pick.

### 4.5 SNMF+EMBER

> **Not yet repaired.** These numbers use the old 4-token scoring (this method
> has the most cut-off answers: 15% at the picks) and, for Baseball, Gambling,
> Gun and Uranium, the old EMBER δ. Plan: [`snmf_ember_search.md`](snmf_ember_search.md), "Repair".

| Concept | δ_emb | δ_in | δ_out | efficacy | specificity | coherence | harmonic_alpaca | min frac |
|---|---|---|---|---|---|---|---|---|
| Ancient Rome | 200 | 1 @ 0–8 (k=13) | 4 @ 0–8 (k=13) | 0.909 | 0.952 | 0.998 | 0.952 | 1.00 |
| Artificial intelligence | 100 | 4 @ 0–8 (k=36) | 4 @ 0–8 (k=36) | 0.836 | 0.910 | 0.935 | 0.892 | 0.92 |
| Baseball | 100 | 4 @ 0–25 (k=47) | 7 @ 13–25 (k=14) | 0.982 | 0.838 | 0.915 | 0.908 | **0.88** |
| COVID-19 pandemic | 50 | 4 @ 0–25 (k=54) | 4 @ 0–8 (k=31) | 0.970 | 0.818 | 0.953 | 0.908 | 0.93 |
| Cannabis | 200 | 10 @ 0–12 (k=28) | 1 @ 0–8 (k=25) | 0.780 | 0.826 | 0.991 | 0.857 | 0.99 |
| Culture of Greece | 200 | 1 @ 0–8 (k=10) | 1 @ 0–8 (k=10) | 1.000 | 0.953 | 1.000 | 0.984 | 1.00 |
| Gambling | 100 | 1 @ 0–25 (k=36) | 7 @ 0–8 (k=23) | 0.980 | 0.985 | 0.998 | 0.988 | 1.00 |
| Golf | 10 | 1 @ 0–8 (k=24) | 1 @ 13–25 (k=10) | 0.623 | 0.962 | 0.998 | 0.823 | 1.00 |
| Gun | 100 | 1 @ 0–25 (k=55) | 7 @ 9–17 (k=4) | 0.918 | 0.902 | 0.959 | 0.926 | 0.92 |
| Halloween | 10 | 10 @ 0–25 (k=24) | 1 @ 0–8 (k=14) | 0.727 | 0.940 | 0.993 | 0.871 | 0.99 |
| Harry Potter | 10 | 10 @ 0–8 (k=15) | 1 @ 0–8 (k=15) | 0.842 | 0.883 | 0.998 | 0.903 | 1.00 |
| Heroin | 100 | 10 @ 0–25 (k=32) | 4 @ 9–17 (k=3) | 0.872 | 0.864 | 0.990 | 0.905 | 0.98 |
| Nazism | 100 | 1 @ 0–12 (k=25) | 10 @ 13–25 (k=13) | 0.735 | 0.891 | 0.907 | 0.837 | 0.90 |
| Pornography | 5 | 4 @ 0–8 (k=23) | 10 @ 9–17 (k=7) | 0.939 | 0.947 | 0.881 | 0.921 | **0.85** |
| Republic of Ireland | 100 | 10 @ 0–25 (k=9) | 10 @ 9–17 (k=2) | 0.837 | 0.933 | 0.974 | 0.911 | 0.97 |
| Uranium | 100 | 4 @ 0–12 (k=20) | 10 @ 13–25 (k=16) | 1.000 | 1.000 | 0.906 | 0.967 | **0.89** |
| Valentine's Day | 200 | 1 @ 0–12 (k=14) | 10 @ 0–8 (k=13) | 1.000 | 0.869 | 0.909 | 0.923 | **0.87** |
| World War II | 5 | 10 @ 0–25 (k=15) | 1 @ 0–8 (k=11) | 0.836 | 0.891 | 0.975 | 0.897 | 0.97 |

- Efficacy beats SNMF for 15 of 18 concepts and EMBER for all 18. Three
  concepts reach efficacy 1.00 (concept-QA at or below chance: Culture of
  Greece, Uranium, Valentine's Day).
- Specificity is higher than SNMF on average (0.909 vs 0.848) and lower than
  EMBER for 15 concepts. Mean 50-item MMLU of the picks is 0.524 (pristine 0.52).
- No pick loops: at most 1 looping answer in a picked cell, at most 3 in any
  cell. Ancient Rome no longer needs the screen. With δ_emb = 200, its top 3
  are mild edits with 0 looping answers.
- Four picks have an Alpaca fraction below 0.9 (no gate, by design).
- **Caveat:** these are selection scores (best of 144 cells, on top of a δ
  picked on the same items). They are optimistic, most of all for this method,
  which searches the most. The held-out comparison will be the English
  `QA_test` column of its cross-lingual run.

Files: `judge_rerun/SNMF_EMBER/pass2_outputs/.../best_hps_summary.csv`.

---

## 5. Cross-lingual leakage (English-tuned edit, asked in 5 languages)

**Setup.** Apply each concept's English-tuned pick once, never re-tuned per
language. Ask the held-out multiple-choice `QA_test` (50 questions) and
`SimdomQA_test` (50) in en/de/es/fr/ru. The prompt is entirely in the target
language. The option order is the same in every language. Greedy decoding, no
judge.

```
retention r_L = clip((acc_L − 0.25) / (pristine_acc_L − 0.25), 0, 1)
leak      Δ_L = r_L − r_en        # > 0: the concept survives better in L than in English
```

A (concept, language) cell is dropped when the pristine model scores below
0.40 there. That drops 2 of 90 cells (Harry Potter/ru, Valentine's Day/ru).

Every one of the 54 (concept, method) edits reproduced its search scores
exactly (`QA_train`, `SimdomQA_train`, 50-item MMLU).

### 5.1 Pristine baseline, `QA_test` accuracy

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

`*` below the 0.40 cutoff. Pristine `SimdomQA_test`: en 0.89, other
languages 0.74 on average. Pristine English MMLU (1000-q test): 0.553.

### 5.2 Summary, all three methods

Mean over concepts. 95% bootstrap interval over concepts (10,000 resamples) in
brackets.

| | `ember` | `snmf` | `snmf_ember` |
|---|---|---|---|
| English `QA_test` acc (pristine 0.73) | 0.50 | 0.47 | **0.38** |
| Retention r_en | 0.52 | 0.45 | **0.27** |
| Retention de / es / fr / ru | 0.69 / 0.74 / 0.71 / **0.94** | 0.44 / 0.45 / 0.43 / 0.51 | 0.38 / 0.39 / 0.36 / 0.65 |
| Leak Δ de | +0.17 [+0.07, +0.27] | −0.00 [−0.08, +0.07] | +0.11 [−0.01, +0.23] |
| Leak Δ es | +0.22 [+0.11, +0.33] | +0.00 [−0.10, +0.11] | +0.12 [+0.02, +0.23] |
| Leak Δ fr | +0.19 [+0.09, +0.29] | −0.02 [−0.12, +0.08] | +0.09 [+0.00, +0.19] |
| Leak Δ ru | **+0.41** [+0.28, +0.52] | +0.05 [−0.06, +0.16] | +0.37 [+0.20, +0.54] |
| **Mean Δ over de/es/fr/ru** | **+0.24** [+0.16, +0.32] | **+0.01** [−0.07, +0.09] | **+0.17** [+0.08, +0.27] |
| Concepts with mean non-English Δ > 0 | 16 / 18 | 9 / 18 | 14 / 18 |
| `SimdomQA_test` acc, en (pristine 0.89) | 0.87 | 0.73 | 0.78 |
| `SimdomQA_test` acc, mean non-en (pristine 0.74) | 0.72 | 0.57 | 0.61 |
| `SimdomQA_test` retention, en / de / es / fr / ru | 0.96 / 0.98 / 0.98 / 0.98 / 1.00 | 0.73 / 0.62 / 0.65 / 0.68 / 0.59 | 0.81 / 0.69 / 0.75 / 0.77 / 0.76 |
| English MMLU, 1000-q test (pristine 0.553) | 0.548 (0.537–0.553) | 0.506 (0.442–0.541) | 0.511 (0.435–0.549) |
| `QA_test` invalid-answer rate, en | 0.09 | 0.00 | **0.16** |
| `QA_test` invalid-answer rate, de / es / fr / ru | 0.05 / 0.07 / 0.06 / 0.01 | 0.01 / 0.00 / 0.00 / 0.00 | 0.11 / 0.10 / 0.10 / 0.02 |

### 5.3 Per concept

`r_en` is English retention. Δ is the leak per language. "SimDom" is
`SimdomQA_test` accuracy (en / mean of the 4 others). "inv" is the `QA_test`
invalid-answer rate (en / worst other language). "—" marks a dropped cell.

**`ember`**

| Concept | en acc | r_en | Δ de | Δ es | Δ fr | Δ ru | mean Δ | SimDom en / other | MMLU | inv en / max |
|---|---|---|---|---|---|---|---|---|---|---|
| Ancient Rome | 0.44 | 0.35 | +0.53 | +0.33 | +0.35 | +0.65 | +0.47 | 0.78 / 0.72 | 0.545 | 0.08 / 0.06 |
| Artificial intelligence | 0.62 | 0.70 | +0.26 | +0.18 | +0.15 | +0.26 | +0.21 | 0.94 / 0.90 | 0.537 | 0.14 / 0.12 |
| Baseball | 0.48 | 0.56 | +0.22 | +0.44 | +0.05 | +0.44 | +0.29 | 0.78 / 0.66 | 0.548 | 0.10 / 0.00 |
| COVID-19 pandemic | 0.46 | 0.54 | −0.23 | −0.02 | +0.17 | −0.28 | −0.09 | 0.80 / 0.78 | 0.548 | 0.08 / 0.18 |
| Cannabis | 0.46 | 0.30 | +0.06 | −0.01 | −0.10 | +0.55 | +0.12 | 0.80 / 0.57 | 0.552 | 0.22 / 0.36 |
| Culture of Greece | 0.50 | 0.68 | −0.05 | +0.27 | +0.19 | +0.32 | +0.18 | 0.84 / 0.68 | 0.547 | 0.12 / 0.00 |
| Gambling | 0.54 | 0.49 | +0.36 | +0.04 | +0.11 | +0.51 | +0.25 | 0.98 / 0.83 | 0.547 | 0.16 / 0.20 |
| Golf | 0.66 | 0.75 | +0.10 | −0.00 | −0.13 | +0.25 | +0.06 | 0.84 / 0.62 | 0.551 | 0.00 / 0.00 |
| Gun | 0.46 | 0.37 | +0.63 | +0.48 | +0.36 | +0.63 | +0.53 | 0.78 / 0.57 | 0.545 | 0.26 / 0.06 |
| Halloween | 0.38 | 0.37 | +0.11 | +0.45 | +0.45 | +0.63 | +0.41 | 0.94 / 0.66 | 0.546 | 0.00 / 0.00 |
| Harry Potter | 0.60 | 0.64 | +0.14 | −0.18 | −0.04 | — | −0.03 | 0.90 / 0.66 | 0.553 | 0.04 / 0.10 |
| Heroin | 0.46 | 0.49 | +0.51 | +0.51 | +0.13 | +0.51 | +0.42 | 0.98 / 0.90 | 0.550 | 0.08 / 0.02 |
| Nazism | 0.52 | 0.55 | −0.06 | +0.19 | +0.16 | +0.45 | +0.18 | 0.90 / 0.78 | 0.551 | 0.06 / 0.12 |
| Pornography | 0.64 | 0.68 | +0.08 | +0.08 | +0.08 | +0.27 | +0.13 | 0.96 / 0.86 | 0.553 | 0.00 / 0.02 |
| Republic of Ireland | 0.56 | 0.63 | +0.12 | +0.16 | +0.19 | +0.37 | +0.21 | 0.94 / 0.87 | 0.553 | 0.08 / 0.06 |
| Uranium | 0.32 | 0.19 | +0.10 | +0.25 | +0.59 | +0.81 | +0.44 | 0.74 / 0.64 | 0.549 | 0.26 / 0.20 |
| Valentine's Day | 0.30 | 0.24 | +0.13 | +0.76 | +0.64 | — | +0.51 | 0.82 / 0.66 | 0.546 | 0.00 / 0.06 |
| World War II | 0.68 | 0.88 | −0.04 | +0.01 | +0.01 | +0.12 | +0.03 | 0.90 / 0.66 | 0.552 | 0.00 / 0.08 |

**`snmf`**

| Concept | en acc | r_en | Δ de | Δ es | Δ fr | Δ ru | mean Δ | SimDom en / other | MMLU | inv en / max |
|---|---|---|---|---|---|---|---|---|---|---|
| Ancient Rome | 0.48 | 0.42 | −0.15 | +0.02 | +0.02 | −0.03 | −0.03 | 0.74 / 0.59 | 0.537 | 0.00 / 0.02 |
| Artificial intelligence | 0.54 | 0.55 | +0.00 | −0.14 | +0.04 | −0.15 | −0.06 | 0.92 / 0.72 | 0.522 | 0.00 / 0.00 |
| Baseball | 0.46 | 0.51 | −0.16 | +0.14 | −0.22 | +0.23 | −0.00 | 0.40 / 0.40 | 0.527 | 0.00 / 0.00 |
| COVID-19 pandemic | 0.44 | 0.49 | +0.25 | −0.18 | −0.30 | +0.09 | −0.03 | 0.72 / 0.55 | 0.494 | 0.00 / 0.00 |
| Cannabis | 0.68 | 0.62 | −0.10 | −0.15 | −0.10 | +0.13 | −0.05 | 0.38 / 0.38 | 0.523 | 0.00 / 0.00 |
| Culture of Greece | 0.50 | 0.68 | −0.19 | −0.27 | −0.23 | −0.12 | −0.20 | 0.86 / 0.53 | 0.537 | 0.00 / 0.00 |
| Gambling | 0.50 | 0.42 | −0.11 | −0.25 | −0.00 | −0.21 | −0.14 | 0.84 / 0.63 | 0.450 | 0.00 / 0.00 |
| Golf | 0.58 | 0.60 | +0.09 | −0.07 | −0.15 | +0.40 | +0.07 | 0.68 / 0.51 | 0.535 | 0.00 / 0.00 |
| Gun | 0.46 | 0.37 | +0.09 | +0.22 | −0.13 | −0.15 | +0.01 | 0.52 / 0.38 | 0.476 | 0.00 / 0.00 |
| Halloween | 0.36 | 0.31 | +0.08 | +0.60 | +0.42 | +0.34 | +0.36 | 0.76 / 0.49 | 0.538 | 0.00 / 0.00 |
| Harry Potter | 0.58 | 0.60 | +0.06 | +0.04 | −0.08 | — | +0.00 | 0.84 / 0.55 | 0.541 | 0.00 / 0.06 |
| Heroin | 0.34 | 0.21 | +0.05 | +0.03 | +0.12 | +0.07 | +0.07 | 0.96 / 0.86 | 0.489 | 0.00 / 0.00 |
| Nazism | 0.42 | 0.35 | +0.14 | −0.07 | +0.02 | +0.09 | +0.05 | 0.68 / 0.55 | 0.502 | 0.00 / 0.04 |
| Pornography | 0.54 | 0.51 | −0.12 | −0.10 | +0.06 | −0.22 | −0.09 | 0.98 / 0.78 | 0.484 | 0.00 / 0.00 |
| Republic of Ireland | 0.60 | 0.71 | −0.40 | −0.40 | −0.38 | −0.32 | −0.38 | 0.80 / 0.64 | 0.515 | 0.00 / 0.12 |
| Uranium | 0.38 | 0.35 | +0.26 | +0.33 | +0.20 | +0.27 | +0.27 | 0.58 / 0.58 | 0.501 | 0.00 / 0.00 |
| Valentine's Day | 0.28 | 0.14 | +0.23 | +0.12 | +0.39 | — | +0.24 | 0.86 / 0.54 | 0.442 | 0.00 / 0.00 |
| World War II | 0.36 | 0.22 | −0.09 | +0.18 | −0.09 | +0.34 | +0.09 | 0.66 / 0.50 | 0.490 | 0.00 / 0.00 |

**`snmf_ember`**

| Concept | en acc | r_en | Δ de | Δ es | Δ fr | Δ ru | mean Δ | SimDom en / other | MMLU | inv en / max |
|---|---|---|---|---|---|---|---|---|---|---|
| Ancient Rome | 0.36 | 0.20 | +0.44 | +0.20 | +0.44 | +0.63 | +0.42 | 0.74 / 0.70 | 0.536 | 0.16 / 0.08 |
| Artificial intelligence | 0.38 | 0.25 | +0.11 | +0.13 | +0.11 | −0.04 | +0.08 | 0.88 / 0.72 | 0.502 | 0.22 / 0.16 |
| Baseball | 0.28 | 0.07 | +0.44 | +0.36 | +0.15 | +0.93 | +0.47 | 0.60 / 0.53 | 0.534 | 0.16 / 0.00 |
| COVID-19 pandemic | 0.22 | 0.00 | +0.22 | +0.03 | +0.00 | +0.00 | +0.06 | 0.62 / 0.47 | 0.440 | 0.18 / 0.32 |
| Cannabis | 0.40 | 0.22 | −0.04 | −0.11 | −0.20 | +0.44 | +0.02 | 0.54 / 0.34 | 0.499 | 0.32 / 0.36 |
| Culture of Greece | 0.48 | 0.62 | −0.07 | +0.22 | +0.17 | +0.38 | +0.18 | 0.84 / 0.66 | 0.544 | 0.14 / 0.02 |
| Gambling | 0.30 | 0.08 | +0.37 | +0.21 | +0.34 | +0.73 | +0.41 | 0.98 / 0.81 | 0.524 | 0.34 / 0.20 |
| Golf | 0.58 | 0.60 | +0.09 | +0.06 | +0.02 | +0.40 | +0.14 | 0.84 / 0.61 | 0.549 | 0.00 / 0.00 |
| Gun | 0.34 | 0.16 | +0.65 | +0.53 | +0.52 | +0.60 | +0.57 | 0.70 / 0.48 | 0.501 | 0.42 / 0.10 |
| Halloween | 0.28 | 0.09 | +0.04 | +0.65 | +0.31 | +0.57 | +0.39 | 0.84 / 0.52 | 0.542 | 0.00 / 0.00 |
| Harry Potter | 0.44 | 0.35 | +0.25 | −0.19 | +0.01 | — | +0.03 | 0.70 / 0.51 | 0.539 | 0.08 / 0.14 |
| Heroin | 0.34 | 0.21 | +0.05 | +0.12 | −0.21 | −0.21 | −0.06 | 0.92 / 0.77 | 0.482 | 0.14 / 0.12 |
| Nazism | 0.52 | 0.55 | −0.25 | +0.02 | +0.11 | +0.21 | +0.02 | 0.88 / 0.74 | 0.524 | 0.06 / 0.12 |
| Pornography | 0.52 | 0.47 | −0.25 | −0.14 | −0.14 | −0.18 | −0.18 | 0.98 / 0.78 | 0.483 | 0.00 / 0.02 |
| Republic of Ireland | 0.40 | 0.31 | +0.21 | −0.13 | +0.03 | +0.26 | +0.09 | 0.84 / 0.72 | 0.540 | 0.08 / 0.08 |
| Uranium | 0.20 | 0.00 | +0.00 | +0.28 | +0.11 | +1.00 | +0.35 | 0.54 / 0.60 | 0.510 | 0.36 / 0.52 |
| Valentine's Day | 0.18 | 0.00 | +0.00 | +0.00 | +0.00 | — | +0.00 | 0.76 / 0.51 | 0.435 | 0.22 / 0.56 |
| World War II | 0.60 | 0.71 | −0.31 | −0.04 | −0.09 | +0.20 | −0.06 | 0.76 / 0.60 | 0.520 | 0.00 / 0.00 |

Files: `results/crosslingual/google_gemma-2-2b-it/{baseline,ember,snmf,snmf_ember}/summary.csv`.

### 5.4 Findings

- **`ember` leaks, worst into Russian.** Retention in Russian is 0.94, close
  to the unedited model. All 4 per-language intervals exclude 0. 16 of 18
  concepts leak on average. The Gemini-era run found the same direction on 8
  concepts (Δ +0.25).
- **`snmf` erases about evenly across languages.** Every interval contains 0,
  so a mean leak above ~0.09 is ruled out. About half the concepts go each way.
  The largest positive leaks (Halloween +0.36, Uranium +0.27, Valentine's Day
  +0.24) come from concepts with low pristine accuracy, where the retention
  denominator is small.
- **`snmf` costs more specificity.** It also removes 30–40% of
  similar-domain knowledge in every language (`SimdomQA_test` retention
  0.59–0.73), against 0.96–1.00 for `ember`. English MMLU drops 0.553 → 0.506
  (worst: Valentine's Day 0.44, Gambling 0.45). Part of SNMF's even erasure is
  a broader, less targeted edit.
- **`ember`'s erasure is partly invalid answers.** Unparseable answers count
  as wrong. Examples: Cannabis es 0.36 / fr 0.30 / en 0.22, Gun en 0.26,
  Uranium en 0.26 / de 0.20, Gambling 0.16–0.20. `snmf`'s worst cell is 0.12.
- **`snmf_ember` erases English the most, but leaks.** English `QA_test`
  drops 0.73 → 0.38 (r_en 0.27, against 0.52 / 0.45). Mean non-English Δ is
  +0.17 and its interval excludes 0; 14 of 18 concepts are positive. Most of
  it is Russian (+0.37, retention 0.65). de/es/fr are smaller (+0.09 to +0.12;
  de's interval touches 0). So the embedding edit brings back an `ember`-like
  leak on top of SNMF. The Gemini-era run had the same ordering (`snmf_ember`
  +0.22, `ember` +0.25).
- **`snmf_ember` specificity is between the two.** `SimdomQA_test` retention
  0.69–0.81 (vs `snmf` 0.59–0.73); English MMLU 0.511, about `snmf`'s level
  (worst: Valentine's Day 0.435, COVID-19 0.440).
- **`snmf_ember` has the most invalid answers.** English 0.16 on average (Gun
  0.42, Uranium 0.36, Gambling 0.34, Cannabis 0.32); some non-English cells
  reach 0.5 (Valentine's Day de 0.56, Uranium de 0.52). English is hit harder
  than the other languages, which lowers r_en and so inflates Δ. Three
  concepts are at or below chance in English (COVID-19, Uranium, Valentine's
  Day: r_en = 0), so their Δ cannot be negative. Both effects need the
  invalid-dropped re-score before per-concept Δ is trusted.

### 5.5 Is SNMF's damage caused by the new selection? No.

The 144-cell grid is the same as in the Gemini era. Only the pick changed: Qwen
top-3 × 50 now, Gemini top-15 before. For the 8 concepts both runs share:

- 5 of 8 picks are the same edit. Of the other 3, Gambling's new pick is more
  damaging, Golf's much less, and Halloween's about the same. On average the
  new picks are slightly *less* damaging (SimDom frac 0.72 → 0.74, 50-q MMLU
  0.520 → 0.535).
- The Gemini-era `snmf` had the same `SimdomQA_test` damage (retention
  0.58–0.65; round 2: 0.55–0.71).
- Its MMLU damage was hidden by a caching bug. The old runner reused the first
  concept's MMLU for every concept, so "MMLU retained 0.97" was wrong. The
  50-question selection MMLU (SE ≈ 0.07) cannot see a ~0.05 drop, so neither
  search could select against it.
- The same edit gives a different per-concept Δ in the two harnesses (e.g.
  Uranium +0.02 → +0.27) because of the option shuffle and prompt. The 8-concept
  mean is small in both (+0.10 Gemini-era, +0.05 now).

Conclusion: the broad damage is a property of SNMF on this grid. Reducing it
would need a different selection objective (a larger MMLU set or a SimDom
constraint), not a different judge or top-k.

---

## 6. Caveats

- **Noise.** n = 50 per cell. Reshuffling the options alone moves a cell by up
  to ±0.14, and one cell's Δ has SD ≈ 0.21. Only averages over concepts mean
  anything.
- **The pristine baselines differ from the Gemini era by design.** The new
  option shuffle keeps the correct letter the same across languages, and the
  model prefers C/D. For the 8 shared concepts, 36 of 40 cells moved, by up to
  0.14.
- **HP-search scores are selection scores** (`QA_train`), so they are
  optimistic. `QA_test` is the held-out measure.
- **No Alpaca gate for SNMF / SNMF+EMBER.** Some picks are below 0.9 on an
  Alpaca fraction (SNMF: Golf, Pornography; SNMF+EMBER: Pornography,
  Valentine's Day, Baseball, Uranium). None loops.
- **Valentine's Day** is near chance in every language (English pristine 0.46),
  so its numbers are unreliable.
- **The specificity measure is mixed**: English MMLU plus in-language SimDom.
- **Invalid answers count as wrong**, so they look like erasure. Most of them
  were not broken output but "The answer is **C**" cut off by the 4-token
  budget. Fixed for the EMBER and SNMF HP searches (§4); still in the
  SNMF+EMBER search and the cross-lingual `ember` / `snmf_ember` numbers
  (English invalid 0.09 / 0.16); `snmf` is near 0.
- **Runs are not bit-identical.** Re-running an edit weeks later changed single
  answers, and one Pornography `QA_train` question systematically. Cells are
  compared only within one run where it matters ([`ember_search.md`](ember_search.md), "Determinism").
