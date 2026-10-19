# Edit brief: bring `report/main.tex` in line with the corrected evaluation

## Context

The draft (`report/main.tex`) was written while a scoring bug was being fixed. The
original multiple-choice scorer generated only 4 tokens, so answers like "The answer
is **C**" were cut off before the letter and counted as wrong. It also rejected
lowercase letters and misread one answer when an option ended in "model".

That fix is now complete:
- the scorer allows 32 tokens and uses a fixed parser;
- all three hyperparameter searches were re-scored and the edits re-selected;
- the cross-lingual evaluation was rerun with the new edits.

The draft still reports the pre-repair numbers, and in several places it says the
re-selection is "in progress".

Your job is to update the text, tables and captions so that every number and claim
matches the final results below.

**Rules**
1. Use only the numbers in this brief. Do not recompute, estimate or round them
   differently. Where a number in the paper is not listed here, leave it unchanged.
2. Do not just swap numbers where a claim's *meaning* changed (section 1). Rewrite
   those sentences.
3. Keep the paper's style: short declarative sentences, the existing macros
   (`\qatrain`, `\qatest`, `\simtest`), 95% bootstrap intervals in `[$+a$, $+b$]` form,
   and signed leaks (`$+0.19$`).
4. Make the before/after explicit once, in §4.3 "A scoring artifact and its effect".
   Elsewhere, report only the final numbers.

All statistics: 18 concepts (16 for Russian), Gemma-2-2B-it. Each concept's mean leak
first averages over its reliable languages (as the Table 2 caption already says).
Intervals are 95% bootstrap over concepts (10,000 resamples), and *t* is a paired
t-test over concepts.

---

## 1. What survived and what changed

**Survived (update the numbers, keep the claim)**
- **EMBER leaks.**
  - Mean leak +0.19 [+0.11, +0.27], down from +0.24.
  - Every per-language interval still excludes 0.
  - Russian is still the extreme: +0.33, retention 0.96.
- **SNMF does not leak** (+0.01 [−0.07, +0.08]) and still causes the most collateral
  damage.
- **EMBER leaks more than SNMF**, paired: +0.18 [+0.09, +0.28], t = 3.76, down from
  +0.23 with t = 4.87.
- **SNMF+EMBER has the best English selection score** on 17 of 18 concepts (Harry
  Potter is the exception).
- **The token-coverage mechanism (§4.4) and the multilingual-fitting results (§4.5)**
  do not depend on the scorer.

**Changed (rewrite the claim)**
- **SNMF+EMBER no longer "brings the leak back".**
  - Its mean leak is +0.04 [−0.02, +0.11]: the interval includes 0.
  - German, Spanish and French are each indistinguishable from 0.
  - **Only Russian still leaks**: +0.23 [+0.08, +0.38].
  - Paired against SNMF, the difference is +0.04 [−0.03, +0.11] with t = 1.01, so not
    significant. The draft had +0.15 [+0.04, +0.27].
- **EMBER and SNMF+EMBER are now clearly different.**
  - EMBER leaks more: +0.15 [+0.06, +0.24], t = 3.10. The draft said "not clearly
    different, +0.08 [−0.01, +0.17]".
  - The exception is Russian, where the two do not differ: +0.10 [−0.06, +0.25].
- **"SNMF+EMBER gives the strongest English erasure" needs qualifying.**
  - Its English retention r_en is 0.39, the lowest on average.
  - It is *not* significantly below SNMF's 0.45: paired −0.06 [−0.17, +0.05]. Before
    the repair this difference was significant (−0.18 [−0.28, −0.06]).
  - It is clearly below EMBER's 0.61.
- **The draft's robustness check was not enough.** Dropping unparseable answers gave
  SNMF+EMBER +0.11 with an interval excluding 0. Proper re-selection and re-scoring
  give +0.04 with an interval including 0. Do **not** keep the claims "our leakage
  conclusions are robust", "the direction of every conclusion is robust" or "the
  conclusions hold when unparseable answers are excluded".
- **EMBER erases less than first reported.**
  - Efficacy is 0.43, not 0.52.
  - Held-out r_en is 0.61, not 0.52.
  - In English, EMBER now erases significantly less than SNMF: r_en paired +0.16
    [+0.08, +0.25].

---

## 2. Final numbers

### 2.1 Table 1 (`tab:english`): selection on \qatrain, means over 18 concepts

| Method | Effic. | Spec. | Coher. | Score |
|---|---|---|---|---|
| EMBER | 0.43 | **0.97** | **0.99** | 0.67 |
| SNMF | 0.72 | 0.85 | 0.96 | 0.82 |
| SNMF+EMBER | **0.83** | 0.89 | 0.97 | **0.89** |

Held-out English retention r_en (\qatest): EMBER 0.61, SNMF 0.45, SNMF+EMBER **0.39**.

Caption: delete "EMBER and SNMF+EMBER efficacy is also inflated by the scoring
artifact of \S\ref{sec:artifact}." Replace it with: all scores use the corrected
scorer.

### 2.2 Table 2 (`tab:crossling`): replace every value

| Row | EMBER | SNMF | SNMF+EMBER |
|---|---|---|---|
| English \qatest accuracy (unedited 0.73) | 0.55 | 0.47 | **0.44** |
| Retention r_L: en / de / es / fr / ru | 0.61 / 0.71 / 0.81 / 0.76 / 0.96 | 0.45 / 0.44 / 0.45 / 0.43 / 0.51 | 0.39 / 0.37 / 0.40 / 0.35 / 0.64 |
| Leak Δ_de | +0.10 [+0.01, +0.20] | −0.01 [−0.08, +0.06] | −0.02 [−0.09, +0.05] |
| Leak Δ_es | +0.19 [+0.10, +0.29] | +0.00 [−0.10, +0.11] | +0.01 [−0.07, +0.11] |
| Leak Δ_fr | +0.15 [+0.06, +0.24] | −0.02 [−0.11, +0.07] | −0.04 [−0.11, +0.03] |
| Leak Δ_ru | +0.33 [+0.22, +0.42] | +0.05 [−0.06, +0.15] | +0.23 [+0.08, +0.38] |
| **Mean leak over de/es/fr/ru** | **+0.19** [+0.11, +0.27] | **+0.01** [−0.07, +0.08] | **+0.04** [−0.02, +0.11] |
| Concepts with mean leak > 0 | 16 / 18 | 9 / 18 | 11 / 18 |
| ~~Mean leak, unparseable answers dropped~~ | **delete this row** | | |
| \simtest retention: en / de / es / fr / ru | 0.95 / 0.98 / 0.98 / 0.98 / 1.00 | 0.73 / 0.63 / 0.65 / 0.68 / 0.59 | 0.76 / 0.67 / 0.68 / 0.73 / 0.67 |
| English MMLU, 1000 questions (unedited 0.553) | **0.547** | 0.506 | 0.507 |
| Unparseable answers on \qatest: en / mean of others | 0.01 / 0.01 | 0.00 / 0.00 | 0.02 / 0.02 |

The project notes (`docs/Part 1/part1_results.md`) list SNMF+EMBER's mean leak as
+0.05 and EMBER's lower bound as +0.12. They average per-language means; the paper
averages per concept first. **Use the values above.**

### 2.3 Paired method differences: new paragraph text for §4.2

All values below are leak differences, A − B, paired by concept. The final column is
the number of concepts where A − B > 0.

| Comparison | Mean leak | de | es | fr | ru | t (mean leak) | A > B |
|---|---|---|---|---|---|---|---|
| EMBER − SNMF | **+0.18** [+0.09, +0.28] | +0.11 [−0.02, +0.24] | +0.19 [+0.07, +0.31] | +0.17 [+0.09, +0.26] | +0.28 [+0.13, +0.42] | 3.76 (p = 0.002) | 13/18 |
| SNMF+EMBER − SNMF | **+0.04** [−0.03, +0.11] | −0.01 [−0.11, +0.09] | +0.01 [−0.06, +0.08] | −0.02 [−0.11, +0.07] | +0.18 [+0.04, +0.32] | 1.01 (p = 0.33) | 11/18 |
| EMBER − SNMF+EMBER | **+0.15** [+0.06, +0.24] | +0.12 [−0.00, +0.23] | +0.18 [+0.08, +0.29] | +0.19 [+0.11, +0.27] | +0.10 [−0.06, +0.25] | 3.10 (p = 0.006) | 14/18 |

English retention r_en, A − B, paired:
- EMBER − SNMF: +0.16 [+0.08, +0.25]
- SNMF+EMBER − SNMF: −0.06 [−0.17, +0.05], p = 0.27
- EMBER − SNMF+EMBER: +0.23 [+0.15, +0.31]

**Caution:** the per-language tests are not corrected for multiple comparisons.
SNMF+EMBER's Russian results are borderline under a 4-language Bonferroni threshold
of 0.0125:
- its Russian leak has p = 0.012;
- its difference from SNMF in Russian has p = 0.027.

Report them with their intervals, and do not call them strong.

### 2.4 Other inline numbers

| Where | Old | New |
|---|---|---|
| §4.2 "The embedding edit leaks": Russian retention | 0.94 | 0.96 |
| same: Russian above mean of de/es/fr | 15 of 16 concepts, +0.22 [+0.13, +0.31] | 15 of 16 concepts, +0.19 [+0.12, +0.26] |
| same: concepts that leak on average | 16 of 18 | 16 of 18 (unchanged) |
| same: "intervals that exclude zero for all four languages" | | still true |
| §4.2 SNMF paragraph: similar-domain loss | SNMF 27–41%, EMBER 0–4% | SNMF 27–41%, EMBER 0–5% |
| §4.2 SNMF paragraph: MMLU 0.553 → 0.506 | | unchanged |
| §4.1 text "removes only about half of the concept" | | efficacy is 0.43, so "less than half" |

### 2.5 Selection changes, for the before/after in §4.3

**EMBER**
- 4 of 18 selected δ changed: Baseball 100→200, Gambling 100→10, Gun 100→5,
  Uranium 100→200.
- Mean selection score 0.735 → 0.667; efficacy 0.52 → 0.43.
- Already in the draft, and still correct: re-scoring the *old* selected edits alone
  raised \qatrain accuracy from 0.49 to 0.54 and lowered efficacy from 0.52 to 0.42
  (Uranium 0.82 → 0.31).
- Replace "for 8 of 18 concepts the selected δ changes or a new candidate has to be
  judged" with the 4 actual changes.

**SNMF+EMBER**
- 7 of 18 edits changed: Ancient Rome, Baseball, Culture of Greece, Gambling, Gun,
  Uranium, Valentine's Day.
- The 4 concepts with a new EMBER δ got a fresh 144-cell SNMF grid.
- Mean selection score 0.909 → 0.888; efficacy 0.88 → 0.83.

**SNMF**
- No edit changed. Mean score 0.821 → 0.820: Pornography 0.880 → 0.864, because
  re-scored cells changed its specificity.

**Cross-lingual rerun**
- Unchanged edits: only the answers that were still unparseable were regenerated.
  Greedy decoding means their first 4 tokens must reproduce the saved answer; this
  matched for all but one answer per method.
- Changed edits: every question was rerun.
- All 54 edits reproduced their corrected English selection scores exactly.
- Unparseable \qatest answers in English fell from 9% to 0.6% (EMBER) and from 16% to
  1.8% (SNMF+EMBER).
- Averaged over concepts, no language now exceeds 2.2%. The remaining unparseable
  answers are mostly empty outputs, not cut-off answers.

### 2.6 Appendix Table `tab:perconcept`: replace all rows

Columns: EMBER δ | Unedited en | selection score E / S / S+E | r_en E / S / S+E | mean
leak E / S / S+E. A "(was X)" marks a changed EMBER δ; mention the changes in the
caption, but do not print "(was …)" in the table.

```
Ancient Rome            & 200            & 0.80 & 0.82 & 0.80 & 0.94 & 0.35 & 0.42 & 0.31 & +0.48 & -0.03 & +0.16
Artificial intelligence & 100            & 0.80 & 0.44 & 0.86 & 0.87 & 0.82 & 0.56 & 0.45 & +0.13 & -0.08 & -0.08
Baseball                & 200 (was 100)  & 0.66 & 0.43 & 0.81 & 0.88 & 0.76 & 0.51 & 0.51 & +0.09 & -0.00 & +0.02
COVID-19 pandemic       & 50             & 0.64 & 0.63 & 0.84 & 0.91 & 0.69 & 0.49 & 0.00 & -0.10 & -0.07 & +0.11
Cannabis                & 200            & 0.94 & 0.57 & 0.70 & 0.78 & 0.39 & 0.62 & 0.33 & +0.14 & -0.05 & +0.03
Culture of Greece       & 200            & 0.62 & 0.94 & 0.84 & 0.97 & 0.73 & 0.68 & 0.62 & +0.13 & -0.20 & -0.03
Gambling                & 10 (was 100)   & 0.84 & 0.72 & 0.87 & 0.89 & 0.66 & 0.42 & 0.46 & +0.17 & -0.14 & -0.02
Golf                    & 10             & 0.80 & 0.67 & 0.76 & 0.82 & 0.75 & 0.60 & 0.60 & +0.06 & +0.07 & +0.14
Gun                     & 5 (was 100)    & 0.82 & 0.66 & 0.87 & 0.90 & 0.54 & 0.37 & 0.54 & +0.42 & +0.01 & +0.01
Halloween               & 10             & 0.60 & 0.66 & 0.79 & 0.87 & 0.37 & 0.31 & 0.09 & +0.41 & +0.36 & +0.39
Harry Potter†           & 10             & 0.80 & 0.77 & 0.93 & 0.90 & 0.67 & 0.60 & 0.35 & -0.04 & +0.00 & +0.11
Heroin                  & 100            & 0.68 & 0.79 & 0.84 & 0.89 & 0.58 & 0.21 & 0.35 & +0.35 & +0.07 & -0.17
Nazism                  & 100            & 0.74 & 0.77 & 0.80 & 0.82 & 0.63 & 0.35 & 0.63 & +0.14 & +0.05 & -0.03
Pornography             & 5              & 0.82 & 0.59 & 0.86 & 0.91 & 0.68 & 0.51 & 0.47 & +0.14 & -0.10 & -0.16
Republic of Ireland     & 100            & 0.74 & 0.59 & 0.74 & 0.89 & 0.67 & 0.71 & 0.43 & +0.23 & -0.36 & +0.04
Uranium                 & 200 (was 100)  & 0.62 & 0.62 & 0.72 & 0.93 & 0.62 & 0.35 & 0.08 & +0.12 & +0.27 & +0.20
Valentine's Day†        & 200            & 0.46 & 0.68 & 0.91 & 0.92 & 0.24 & 0.14 & 0.00 & +0.51 & +0.24 & +0.11
World War II            & 5              & 0.74 & 0.66 & 0.83 & 0.90 & 0.88 & 0.22 & 0.71 & +0.03 & +0.08 & -0.07
```

Formatting:
- Write `-0.00` as `$0.00$` and `+0.00` as `$0.00$`, as the draft does.
- Keep the † footnote: Russian excluded.
- The caption's note that concepts with r_en = 0 can only show positive leak still
  applies: COVID-19 and Valentine's Day under S+E.
- In the appendix text, "SNMF's largest positive leaks (Halloween, Uranium,
  Valentine's Day)" still holds.

---

## 3. Edits by location

Line numbers refer to the current `main.tex`.

1. **Abstract, l. 45–48.**
   - Replace the numbers: EMBER "0.24 … 0.41 in Russian" → 0.19 and 0.33.
   - Rewrite the SNMF+EMBER sentence: adding EMBER to SNMF does not bring the leak back
     into German, Spanish or French; a smaller leak remains only into Russian (+0.23).
   - Delete "The conclusions hold when unparseable answers are excluded."
   - In l. 48, the scoring artifact inflated EMBER's efficacy *and* the apparent
     leak of SNMF+EMBER. Say that all edits were re-selected with the corrected scorer.
2. **Introduction, contributions, l. 75–80.**
   - l. 75: +0.24 [+0.15, +0.32] → +0.19 [+0.11, +0.27]; Russian retention 0.94 → 0.96.
   - l. 76: SNMF (+0.01, [−0.07, +0.08]); "30–40%" can stay.
   - l. 77: rewrite. SNMF+EMBER has the lowest English retention on average, but not
     significantly below SNMF. Its mean leak (+0.04) is not distinguishable from 0;
     it leaks only into Russian.
   - l. 80: rewrite. All three searches were re-selected with the corrected scorer
     (4 EMBER and 7 SNMF+EMBER edits changed) and the cross-lingual evaluation was
     rerun. EMBER's leak survives, smaller. The apparent SNMF+EMBER leak into
     Latin-script languages was an artifact. **Remove "robust".**
3. **§3.4 Cross-lingual evaluation (around l. 174–176).** Add one sentence: all
   multiple-choice scoring uses a 32-token budget and the corrected parser
   (\S\ref{sec:artifact}).
4. **Table 1 + §4.1 text (l. 184–205).** Apply 2.1 and the 2.4 "about half" fix.
   "The held-out English retention follows the same order" stays true (0.61 > 0.45 >
   0.39). Optionally add that the SNMF vs SNMF+EMBER gap is not significant.
5. **Table 2 (l. 210–236).** Apply 2.2. The caption can stay; it already says the mean
   leak averages each concept first.
6. **§4.2 paragraphs (l. 240–256).**
   - Apply 2.4.
   - Rewrite "The combination leaks again." (l. 252–256): new heading, e.g. "The
     combination leaks only into Russian." Use the numbers in 2.2/2.3:
     - SNMF+EMBER's de/es/fr leaks are near 0, and its Russian leak is +0.23;
     - EMBER − SNMF is +0.18, t = 3.76;
     - SNMF+EMBER − SNMF is +0.04, not significant;
     - EMBER − SNMF+EMBER is +0.15, t = 3.10, but not in Russian (+0.10 [−0.06, +0.25]).
   - **Delete** l. 256 ("An earlier run … found the same ordering (leaks of +0.25,
     +0.10 and +0.22)"). That run used the 4-token scorer, and the corrected ordering
     (EMBER > SNMF+EMBER ≈ SNMF) no longer matches it.
7. **§4.3 Scoring artifact (l. 274–286).**
   - Keep the description of the bug and Figure 2; that figure is from the re-scored
     grid and is still valid.
   - Replace l. 278–280 ("for 8 of 18 … in progress … original scorer") with the final
     re-selection from 2.5.
   - Replace the paragraph "Does the artifact explain the leak?" (l. 282–286) with the
     before/after, using the before → after values:

     | Quantity | Before | After |
     |---|---|---|
     | EMBER mean leak | +0.24 | +0.19 |
     | EMBER Russian leak | +0.41 | +0.33 |
     | SNMF+EMBER mean leak | +0.16 [+0.07, +0.26] | +0.04 [−0.02, +0.11] |
     | SNMF+EMBER Russian leak | +0.37 | +0.23 |
     | EMBER − SNMF+EMBER | +0.08 (n.s.) | +0.15 (t = 3.10) |

     State plainly which conclusions survived and which changed (section 1).
   - Say that the earlier shortcut was insufficient: dropping unparseable answers
     gave +0.11 for SNMF+EMBER, but re-selection and re-scoring give +0.04. The cut-off
     answers were concentrated in English, which inflated both English erasure and
     the leak. Remove "robust".
8. **§4.4 mechanism, l. 297.** The 8-concept correlations (r = −0.68, −0.91, +0.20)
   come from the old run with the 4-token scorer. Either delete the sentence or mark it
   explicitly as "(earlier run, original scorer)". Do not present it as support for
   SNMF+EMBER leaking. Leave the rest of §4.4 unchanged.
9. **Discussion, l. 336–339.** The EMBER argument stands. Optionally add one
   descriptive sentence: in the combination, a leak remains only into Russian, the
   language the embedding edit reaches least. Do not add mechanism claims beyond this.
10. **Limitations (iv), l. 345.**
    - Replace with: all results use the corrected scorer. On average at most 2.2% of
      answers per language remain unparseable (mostly empty output), and they count
      as wrong.
    - Optional new item: one re-selected SNMF+EMBER edit (Gambling) costs a lot of
      general ability (1000-question MMLU 0.450 vs 0.524 for the previous edit),
      because the 50-question MMLU gate used in selection is coarse.
    - Optional: per-language tests are not corrected for multiple comparisons.
11. **Conclusion, l. 355.** Rewrite "combining the two inherits the embedding edit's
    leak". The combination removes the leak into German, Spanish and French, but a
    smaller leak into Russian remains.
12. **Appendix table and caption (l. 379–415).** Apply 2.6. In the caption, note that
    the EMBER δ changed for Baseball, Gambling, Gun and Uranium after re-selection.

**Do not change:** §2, §3.1–3.3, the judge appendix, the degenerate-SNMF appendix,
the data appendix, §4.5 (Q2), Figure 2, the footnote on the Pornography control
(l. 275), "2 of 90 pairs" (l. 173), "All 54 edits reproduced …" (l. 176), and the AI
disclosure section.

## 4. Not a text edit: Figure 1 is outdated

`figures/retention_by_language.pdf` was drawn from the pre-repair results. No
plotting script is in the repo. Regenerate it with these values: retention r_L, mean
over concepts, with 95% bootstrap interval.

| | en | de | es | fr | ru |
|---|---|---|---|---|---|
| EMBER | 0.61 [0.53, 0.69] | 0.71 [0.61, 0.81] | 0.81 [0.73, 0.87] | 0.76 [0.69, 0.82] | 0.96 [0.89, 1.00] |
| SNMF | 0.45 [0.38, 0.52] | 0.44 [0.37, 0.51] | 0.45 [0.37, 0.54] | 0.43 [0.36, 0.49] | 0.51 [0.41, 0.61] |
| SNMF+EMBER | 0.39 [0.28, 0.48] | 0.37 [0.27, 0.46] | 0.40 [0.31, 0.49] | 0.35 [0.25, 0.44] | 0.64 [0.49, 0.78] |

The caption can stay as is.

## Sources

- **Paired differences:** produced by `report/analysis/paired_fixed.py`, which writes
  `report/analysis/paired_fixed.csv`. Its "original" column reproduces the draft's
  numbers to ±0.01, which confirms the method.
- **Per-method values and selection metrics:**
  - `results/crosslingual/google_gemma-2-2b-it/<method>_fixed/summary.csv`
  - `judge_rerun/{Ember,SNMF,SNMF_EMBER}/pass2_rescored/…/best_*_summary.csv`
  - `docs/Part 1/part1_results.md`
