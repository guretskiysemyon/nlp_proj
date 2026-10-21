# Pre-registration: the multilingual comparison on 20 new concepts

Written 2026-10-20, before any data for these concepts was built and before any fit, selection or evaluation.
The goal is a held-out confirmation of the findings on EMBER's 18 concepts ([RESULTS.md](RESULTS.md)), under
**the same protocol**, so that the two sets can be aggregated (38 concepts).

## Concepts

The 20 concepts of `data/concept_sentences_extra.json` (`ML_CONCEPTS=new`): French Revolution, Tuberculosis,
Cricket, Buddhism, Plutonium, Judaism, Karate, Existentialism, Prime number, The Matrix, Louvre, American
Revolutionary War, Robot, Sigmund Freud, Giraffe, Surfing, Flush toilet, Cloud, Doctor of Philosophy, Tourism.
None of them was used to design or tune any of A1/A2/A3/A3z. They were used before only for single-language
judge labelling (CINDER ladder), not for erasure.

## Frozen protocol (identical to the 18-concept run)

- **Model and setup:** Llama-3.1-8B-Instruct, NF4 transformer, fp16 embeddings, Titan Xp; 8 languages
  (en de fr it pt hi es th).
- **Data:**
  - concept sentences: up to 300 per language, native Wikipedia first, NLLB translations to fill;
  - neutral sentences: as in `experiments/ml/data`;
  - fitting sample: N = 30 + 30 sentences per language (data seed 0);
  - background and held-out text: unchanged.
- **Approaches:** A1, A2, A3 and A3z, as in PLAN.md and its A3z amendment, with fit seed 42.
  - Fit settings: k = 200, s = 1%, λ = 1e-4; τ = 2, top 10 candidates.
  - Judging: TJ at confidence ≥ 0.85; EJ@5 with a 20-word lexicon per language (A3 and A3z).
- **Evaluation:**
  - per language: concept MC (QA_train/QA_test, 50 + 50 items), similar-domain MC (50 + 50), Global-MMLU (50 val, 200 test), fluency on held-out native text;
  - the same H_L, score, eligibility (unedited accuracy ≥ 0.40 on the split's own baseline), gates (MMLU and fluency ≥ 0.9 in every language), δ grid {0.5 … 200} with the stop at the first gated δ, and δ selection on the validation split;
  - test reported strict (0 if a gate fails on test), with the ungated score alongside.
- **Questions:** English MC built with Gemini in EMBER's format, from the concept's Wikipedia page, with an adjacent
  similar-domain topic. They are checked automatically for answerability, a single correct option and no leakage
  between splits. Translation into 7 languages uses the v2 pipeline (NLLB and MADLAD n-best, LaBSE ranking, script
  and number checks, de-duplication), the version used for the 18 concepts' v2 re-score.
  - For aggregation, the 18 concepts enter with their v2 scores (`rescore_v2.json`).

**Not changed, by decision.** The δ rule stays as it is, even though A3z failed a gate on test twice. A margin rule
would apply to about two thirds of the units (29 of 44 edited units on the 18 concepts have the chosen δ next to the
gated δ). That would cost about 60 extra GPU-hours to apply to both sets, and changing the rule would break
aggregation. It is left as future work.

## Hypotheses and analysis (decided now)

Unit score = strict test score; "no edit" = 0. The shared set is the concepts that all four approaches edit.

- **H1 (pipeline):** on concepts both edit, A2's mean is within 0.05 of A1's.
- **H2 (fix):** A3z > A3 on a majority of the concepts A3 edits (paired), with a higher mean.
- **H3 (CINDER):** A3z ≥ A2 on the shared set, in mean strict and ungated score. This is reported either way.
- **Coverage:** concepts edited per approach (A1 vs A2 vs A3z).

Each is reported on the 20 new concepts alone (the confirmation) and on all 38. The tests are:
- paired sign tests and Wilcoxon signed-rank tests across concepts;
- bootstrap 95% confidence intervals of mean differences, over concepts, with 10,000 resamples.

## Budgets

- **Gemini:** 15 NIS allocated for this extension, on the shared log `results/ml/gemini_usage.jsonl`; the guard is raised to 30 NIS in total.
- **GPU:** at most 6 SLURM GPUs, shared fairly with the single-language CINDER session (cl_* jobs), plus the local c-001 GPUs.

## Amendment 1 (2026-10-21, before any seed-union evaluation): seed-union selection

**Why.** In the 3-seed coverage study (results/ml/seeds/COVERAGE.md, selection only), the concepts edited over any
of fit seeds 42, 1 and 2 are 31 (A1) and 29 (A3/A3z), against 26 and 21 at seed 42 alone. This amendment tests
whether pooling selections over seeds changes the comparison between fixed CINDER and per-language EMBER. Both
approaches get the same treatment, so neither benefits from extra draws alone.

**New approaches** (only the fit seed differs between draws; data sample, settings and judging are unchanged):
- **A1U:** per seed s ∈ {42, 1, 2}, A1's merged edit for that seed, using the existing edit-once rule by page
  priority. The three per-seed edits are then pooled.
- **A3zU:** per seed, A3's selection with Z re-solved on the original embeddings (the A3z construction). The three
  per-seed edits are then pooled.
- **Pooling rule:** a token edited by any seed is edited. Its edit vector is the mean of the c_t from the seeds
  that edit it.
- **Evaluation:** the frozen protocol, unchanged (δ grid and stop, δ on validation, strict and ungated test score,
  gates, v2 questions for the 18 original concepts).
  - When only seed 42 edits a concept, the pooled edit equals the seed-42 edit. The existing A1 or A3z result is
    reused, since evaluation is deterministic.
  - When no seed edits a concept, it scores 0.

**Hypotheses** (all 38 concepts; unit score = strict test score, no edit = 0; ungated reported alongside):
- **H4 (main):** A3zU vs A1U over all 38 concepts, mean difference. Reported either way, with a paired
  Wilcoxon test, a sign test and a bootstrap 95% CI. "A3zU ≥ A1U" is supported if the CI's lower bound is above
  −0.05.
- **H5:** pooling helps both approaches. A1U − A1 and A3zU − A3z, over all 38 concepts.
- **Coverage:** concepts with an edit under A1U and A3zU.
- Also reported: these results on the 20 new concepts alone, and on concepts both pooled approaches edit.

**Budget.** No Gemini, since all selections exist. GPUs: the 3 local c-001 GPUs from now on, plus SLURM GPUs once
the single-language session releases them (about 02:00), up to the 12-GPU per-user cap as agreed with it.
