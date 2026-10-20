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

## Suggested next steps (not run)

1. A δ rule with a margin below the gate, especially for A3z. This needs no new fits, only re-scoring.
2. Seed-robust selection: union or vote over 3 fit seeds. This also attacks A2/A3z coverage.
3. The A3z idea on the single-language CINDER ladder (Gemma).
4. An fp16 confirmation on 2–3 RTX 2080s for 4–5 concepts.
5. Noise and mean-token baselines, as in the EMBER paper.
