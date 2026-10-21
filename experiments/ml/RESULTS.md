# Multilingual concept erasure on Llama-3.1-8B-Instruct: results

Complete run, 2026-10-20. 18 concepts × 4 approaches, 8 languages (en, de, fr, it, pt, hi, es, th).
Plan and fixed hyperparameters: [PLAN.md](PLAN.md). Observations and caveats: [OBSERVATIONS.md](OBSERVATIONS.md).
Full per-unit tables: `results/ml/REPORT_tables.md` (`python -m experiments.ml.report`).

## The approaches

| | Selection (which tokens, which features) | Edit direction |
|---|---|---|
| **A1** per-language EMBER | EMBER on each language's full set (8 fits, 8 judge passes); tokens merged, edited once, by Wikipedia page priority | each language's Z |
| **A2** multilingual EMBER | one fit on an N = 30 + 30 sentence sample per language; Z fixed, Y solved for the full multilingual set | Z (original space) |
| **A3** multilingual CINDER | A2 with CS (8-language background whitening + sharpening), GS-norm global top-K, TWJ judge | whitened Z, mapped back |
| **A3z** CINDER, original-space edit | same as A3 | Z re-solved on the raw embeddings with A3's Y fixed |

A3z was added after A3's results were seen (PLAN amendment, [diagnose_a3](../../results/ml/diagnose_a3/DIAGNOSIS.md)).

**Score.** Per language, H_L = HM(efficacy, HM(MMLU, similar-domain), fluency), each normalized to the unedited
model. The unit's score is the mean over eligible languages (unedited concept accuracy ≥ 0.40). Gates: MMLU and
fluency ≥ 0.9 of baseline in every language, otherwise 0. δ is chosen on the validation split; test is reported.

## Main table (test score)

"—" means no feature was selected, so there is no edit (score 0). An asterisk means a gate failed on test (score 0),
with the ungated score in parentheses.

| Concept | A1 | A2 | A3 | A3z |
|---|---|---|---|---|
| Cannabis | 0.42 | — | 0.59 | **0.87** |
| Harry Potter | **0.80** | 0.67 | 0.24 | 0.34 |
| COVID-19 pandemic | 0.74 | 0.48 | 0.38 | **0.75** |
| Pornography | 0.67 | **0.74** | 0.31 | 0* (0.76) |
| Baseball | 0.51 | 0.47 | 0.34 | **0.67** |
| Uranium | **0.59** | — | 0.15 | 0.18 |
| Gun | 0.21 | 0.40 | 0.21 | **0.54** |
| Culture of Greece | 0.31 | 0.28 | 0.22 | **0.53** |
| Gambling | **0.53** | 0.49 | 0.20 | 0.40 |
| Ancient Rome | **0.41** | — | — | — |
| Nazism | — | — | 0.10 | **0.34** |
| Golf | **0.33** | — | — | — |
| Republic of Ireland | **0.33** | — | — | — |
| Artificial intelligence | 0.16 | 0* (0.48) | **0.32** | 0* (0.52) |
| World War II | **0.27** | — | — | — |
| Halloween, Heroin, Valentine's Day | — | — | — | — |

## Summary

| | A1 | A2 | A3 | A3z |
|---|---|---|---|---|
| Concepts with an edit (of 18) | **14** | 8 | 11 | 11 |
| Mean, all 18 concepts (no edit = 0) | **0.350** | 0.196 | 0.170 | 0.257 |
| Mean, 8 concepts all four edit | **0.492** | 0.442 | 0.278 | 0.404 |
| Same, ungated | 0.492 | 0.501 | 0.278 | **0.564** |
| Best on the 8 shared concepts (strict / ungated) | 2 / 2 | 1 / 0 | 1 / 0 | **4 / 6** |
| Test-split gate failures | 0 | 1 | 0 | 2 |
| Mean, cleaned it/pt/hi/th questions (all 18) | 0.344 | 0.194 | 0.166 | 0.253 |

Mean test efficacy per language (edited units only):

| | en | de | fr | it | pt | hi | es | th |
|---|---|---|---|---|---|---|---|---|
| A1 | 0.48 | 0.32 | 0.31 | 0.40 | 0.34 | 0.06 | 0.30 | 0.09 |
| A2 | **0.72** | 0.28 | 0.29 | 0.43 | 0.36 | 0.08 | 0.40 | **0.19** |
| A3 | 0.21 | 0.16 | 0.13 | 0.15 | 0.13 | 0.02 | 0.16 | 0.09 |
| A3z | 0.65 | **0.41** | **0.44** | **0.53** | **0.47** | 0.08 | **0.44** | 0.15 |

Selection cost per concept (mean):

| | Factorizations | Rows factorized | ALS iterations | Judged candidates (2 Gemini calls each) |
|---|---|---|---|---|
| A1 | 8 | 30,793 | 3,136 | 80 |
| A2, A3 | 1 | 6,540 (+ one solve on ~19,100) | ~356 | 10 |
| A3z | A3's fit + one k × k solve | | | as A3 |

Evaluation, 1–4 GPU-hours per unit, costs the same for every approach. Total Gemini spend for the study: 2.25 NIS.

## Conclusions

1. **The multilingual pipeline works.** A2, with one fit on a 30-sentence-per-language sample plus a fixed-Z solve
   for the full set, comes within 0.05 of per-language EMBER on the concepts both edit (0.44 vs 0.49; 0.50 vs 0.49
   ungated). It needs 1/8 of the factorizations and judge calls. It erases more in English and Thai, and is weaker in
   German and French.
2. **A1's overall lead is coverage.** Any of A1's 8 per-language fits can find a feature, so A1 edits 14 concepts and
   A2 only 8. Over all 18 concepts, that gap decides the mean.
3. **CINDER's selection is better; its first edit direction was not.** A3 trails, because the edit, mapped back from
   the whitened space, points into directions layer 0 barely reads (read gain 1.1–1.7× random vs 2.6–7.6× for A2).
   A3z keeps A3's selection and builds the edit in the original space. That alone raises A3 on 9 of 11 concepts
   (mean 0.28 → 0.42 on the 11). A3z is the best approach on 4 of the 8 shared concepts (6 of 8 ungated), has the
   highest ungated mean on them (0.56), and the highest non-English efficacy in every language except hi and th.
   It also edits concepts A2 cannot (Cannabis 0.87, Nazism 0.34, Uranium).
4. **The take-away for CINDER: whiten to select, edit in the original space.** The diagnosis and the fix are both
   cheap: one least-squares solve, no new factorization or judge calls.
5. **A3z's strength comes with gate risk.** Its grids climb steeply up to the gate, and twice the δ chosen on 50
   validation items failed a gate on test (Pornography: fr/es MMLU 0.85; AI: en MMLU 0.87). A safer δ rule, such
   as one step below the best or a margin on the gate, would likely turn these into its best scores. That's not
   tested here.

## Caveats

- **Hindi** (and mostly Thai) cannot be erased by any approach. EMBER edits only concept-only tokens, and Llama splits
  Hindi into pieces shared with everyday text (OBSERVATIONS §1).
- **Selection noise.** Fit-seed and sample changes move the edited-token sets as much as each other (Jaccard about
  0.2), for all approaches (OBSERVATIONS §5). Single concepts are one draw, and the means are what count. One seed
  per unit.
- **A3z is post hoc.** It was defined after A3's results were seen. It is reported next to A3, not instead of it.
- **4-bit model.** The transformer runs in NF4 (Titan Xp); the edited embeddings are fp16. Absolute numbers need a
  bf16/fp16 check.
- **Cleaned translations** (it/pt/hi/th) change scores by about ±0.03 (Republic of Ireland A1: 0.33 → 0.24), and
  no conclusion changes.
- **Protocol deviations from EMBER's code:** a perplexity fluency gate instead of Alpaca, the δ sweep stops at the
  first gated δ, and the eligibility cutoff is computed from the split's own baseline (PLAN, OBSERVATIONS).

## Held-out confirmation: 20 new concepts, 38 pooled (2026-10-21)

Pre-registered in [PREREG_NEW.md](PREREG_NEW.md) before any data for the new concepts existed. The protocol is
identical to the 18-concept run. The 18 enter the pooled numbers with their v2 (cleaned-question) scores. Full
tables are in `results/ml/PREREG_RESULTS.md` (`python -m experiments.ml.analyze_prereg`) and
`results/ml/seeds/COVERAGE.md`.

Test score per new concept ("—" = no edit; \* = gated on test):

| Concept | A1 | A2 | A3 | A3z |
|---|---|---|---|---|
| Robot | 0.57 | **0.80** | 0.35 | 0.70 |
| Surfing | 0.50 | — | 0.31 | **0.76** |
| Doctor of Philosophy | 0.18 | — | 0.38 | **0.75** |
| Buddhism | 0.02 | — | 0.10 | **0.67** |
| Prime number | 0† | — | 0.16 | **0.52** |
| Cricket | **0.56** | — | — | — |
| American Revolutionary War | **0.56** | — | — | — |
| Existentialism | **0.47** | — | — | — |
| Plutonium | — | — | **0.41** | 0.31 |
| Tourism | 0.29 | 0\* (0.63) | 0.12 | **0.32** |
| Judaism | 0.07 | — | **0.29** | 0\* |
| Tuberculosis | — | — | **0.12** | 0.11 |
| Cloud | 0\* | — | 0.04 | **0.08** |
| Karate | 0† | — | — | — |
| French Revolution, The Matrix, Louvre, Sigmund Freud, Giraffe, Flush toilet | — | — | — | — |

† A1 found features, but its edit fails a gate already at δ = 0.5 on validation, so no δ is admissible (score 0).
Exact values for every unit are in `results/ml/REPORT_tables.md`.

| | A1 | A2 | A3 | A3z |
|---|---|---|---|---|
| New 20: concepts with an edit | **12** | 2 | 10 | 10 |
| New 20: mean strict / ungated | 0.162 / 0.170 | 0.040 / 0.071 | 0.114 / 0.114 | **0.210 / 0.243** |
| All 38: concepts with an edit | **26** | 10 | 21 | 21 |
| All 38: mean strict / ungated | **0.248** / 0.252 | 0.113 / 0.143 | 0.139 / 0.139 | 0.231 / **0.282** |
| Coverage over fit seeds 42 / 1 / 2 (union) | 26 / 24 / 28 (31) | 10 / 9 / 12 (13) | 21 / 17 / 20 (29) | as A3 |

Pre-registered hypotheses, all 38 concepts, strict score; mean difference [bootstrap 95% CI], wins–losses,
Wilcoxon p (ungated in parentheses):

| | n | Difference | Wins–losses | p | Verdict |
|---|---|---|---|---|---|
| **H1:** A2 − A1, concepts both edit | 10 | −0.051 [−0.156, +0.057] (+0.063) | 3–7 | 0.38 | **Partly supported.** The gap is about 0.05 and the CI includes 0, but A2 edits only 10 concepts. On the new 20 it cannot be tested (n = 2). |
| **H2:** A3z − A3, concepts A3 edits | 21 | **+0.166 [+0.054, +0.268]** (+0.259) | 16–5 (19–2) | **0.007** (<0.001) | **Confirmed.** It also holds on the new 20 alone: +0.19 [+0.03, +0.35], 7–3, p = 0.06 (ungated +0.26, p = 0.014). |
| **H3:** A3z − A2, concepts all four edit | 10 | −0.005 [−0.22, +0.17] (+0.011) | 5–4 | 0.73 | **Not supported:** no difference where both edit. |
| A3z − A2, concepts either edits | 21 | **+0.213 [+0.053, +0.366]** (+0.252) | 15–4 | **0.011** (0.004) | **A3z is better, through coverage.** On the new 20: +0.34, 8–1, p = 0.012. |
| A3z − A1, concepts either edits | 29 | −0.023 [−0.156, +0.110] (+0.039) | 15–13 | 0.78 | **Tie.** |

**What the confirmation shows.**

1. **The A3z fix generalizes.** On unseen concepts, the original-space edit beats the mapped-back one, and over all
   38 concepts the effect is large and significant. This is the most solid result of the study.
2. **CINDER's advantage is coverage, not per-concept strength.** Where A2 and A3z both edit, they are equal.
   A2's selection collapses on unseen concepts: it edits 2 of 20, against 10 for CINDER. Across 3 fit seeds A2 edits
   9–12 of 38 concepts and CINDER 17–21, so the gap is not a seed accident. Counting coverage, A3z clearly beats
   multilingual EMBER (p ≈ 0.01).
3. **A3z matches per-language EMBER** at 1/8 of its selection cost. The means are within 0.02 strict over 38
   concepts, and A3z is ahead ungated (0.28 vs 0.25). A1 still edits more concepts (26 vs 21). Taking the union of
   CINDER's selections over 3 fit seeds (29 concepts) would close most of that gap; this union has not been
   evaluated.
4. **Gates that pass on validation and fail on test** occur for every approach. There are 6 cases (A1 Cloud, A2
   AI and Tourism, A3z AI, Pornography and Judaism), so we report the ungated score alongside the strict one.

**Caveats specific to the new concepts.**
- The questions were generated with Gemini 3.5 Flash-Lite and verified automatically by a blind verifier, without
  human review.
- German, French and Spanish were machine-translated (for the 18 they came from existing sets).
- The single-language held-out text mentions some new concepts in 2–7 of 100 sentences.
- The usage log was NUL-padded by concurrent NFS writes. The records were recovered, and the total is about
  17 NIS, possibly slightly more.

See `results/ml/data/qa_new/MANIFEST.md` and `results/ml/data/SAMPLE.md`.

## Assessment (final, after the held-out confirmation)

**Established** (38 concepts, pre-registered tests on 20 unseen concepts):
- **Whiten to select, edit in the original space.** A3z beats A3 by +0.17 strict (16–5, p = 0.007), and the gain
  also holds on the unseen concepts alone. The cause is mechanistic: A3's mapped-back edit points into directions
  layer 0 barely reads. Confidence: high.
- **CINDER's selection finds concept features where multilingual EMBER does not.** A2 edits 9–12 of 38 concepts
  across 3 fit seeds, CINDER 17–21. On the unseen concepts A2 edits 2 of 20, CINDER 10. Counting coverage,
  A3z beats A2 by +0.21 (p ≈ 0.01). Confidence: high.
- **A single multilingual fit with fixed CINDER matches per-language EMBER** (0.23 vs 0.25 strict, 0.28 vs 0.25
  ungated, 15–13, p = 0.78) at 1/8 of its selection cost. This is a tie, not a win. Confidence: moderate.
- **Limits of embedding-only erasure:**
  - Hindi (and mostly Thai) cannot be erased, because almost no Hindi token is concept-only;
  - per-concept selection is unstable across fit seeds for every approach;
  - δ chosen on 50 validation questions sometimes fails a gate on test (6 units).

**Not established:**
- **A3z vs A2 per concept:** where both edit, they are equal (H3 not supported, n = 10).
- **A2 within 0.05 of A1 (H1):** consistent with the data but untestable on the new concepts (n = 2).
- **Scores across seeds:** only selection was replicated; every score is one draw.
- **The new concepts' questions:** verified automatically, not by a person.

## Suggested next steps (not run)

1. A δ rule with a margin below the gate, especially for A3z. This needs no new fits, only re-scoring.
2. Seed-robust selection: union or vote over 3 fit seeds. The selections exist (results/ml/seeds/); the union
   reaches 29 concepts for CINDER vs 31 for A1. It still needs evaluation, about half a day of GPU time and no Gemini.
3. The A3z idea on the single-language CINDER ladder (Gemma).
4. An fp16 confirmation on 2–3 RTX 2080s for 4–5 concepts.
5. Noise and mean-token baselines, as in the EMBER paper.
