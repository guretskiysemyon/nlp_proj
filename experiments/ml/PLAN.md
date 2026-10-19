# Multilingual erasure on Llama-3.1-8B-Instruct: plan

2026-10-18 · branch `orip-ml` (worktree `../nlp_proj_ml`). Outputs only under `experiments/ml/` and `results/ml/`.

## Question

Which pipeline erases a concept best in all 8 languages that Llama-3.1 officially supports
(en, de, fr, it, pt, hi, es, th)?

| ID | Approach | Factorization | Coefficients used for the edit | Selection |
| --- | --- | --- | --- | --- |
| A1 | **Per-language EMBER** | One EMBER fit per language, on that language's full set | Each language's own Y | TJ per language |
| A2 | **ML-EMBER + extrapolation** | One RS fit on a small multilingual sample | Z fixed, row-sparse Y solved for the full multilingual set | TJ on the full Y |
| A3 | **ML-CINDER** | A2 with CS (8-language background) and GS (rn) | Z fixed, GS-sparse Y on the full set, mapped back with T⁻¹ | TWJ = TJ ∨ EJ@5 |

## Data (the "database"), all under `results/ml/data/`

- **Concepts:** the 18 paper concepts (QA exists for all of them).
- **Concept pages:** native Wikipedia page per (concept, language) via interlanguage links; word count recorded per page.
- **Full set** per (concept, language): up to 300 concept sentences from the native page. If the page is shorter,
  top up with NLLB translations of English page sentences, flagged `translated`. Matched native neutral sentences
  (Wikipedia dump pages of that language, no concept-title words), as many as the concept side.
- **Sample** (A2, A3): N concept + N neutral sentences per language, drawn from the full set (seed 0).
  N is fixed for all concepts (see "Sample size").
- **Evaluation**, per language:
  - concept MC `QA_train` / `QA_test` and similar-domain `SimdomQA_train` / `SimdomQA_test` (50 each, from
    `data/mc_questions.json`); de/es/fr from `data/qa_lang`; it/pt/hi/th translated with NLLB-200 3.3B and scored
    with COMET-Kiwi; native-language prompt scaffold;
  - MMLU: Global-MMLU (same item ids in every language; Thai translated from English with NLLB), 50 validation +
    200 test items;
  - fluency text: 100 held-out native neutral sentences per language (disjoint from every neutral set).
- **Gemini** is used only for feature judging and lexicons (no translations): own log
  `results/ml/gemini_usage.jsonl`, hard stop 15 NIS (`GEMINI_BUDGET_NIS=15`, prices 0.25 / 2.50 USD per 1M).

## Sample size

The sampled matrix E_V\* (|V\*| × 4096) must have full column rank 4096 but stay small. Rule: the smallest N in
{20, 30, 40, 50} such that, for every concept, |V\*| ≥ 1.25 · 4096 and the numerical rank of E_V\* is 4096. The
saved 8-language runs give |V\*| ≈ 7,000 at N = 40. Result (results/ml/data/SAMPLE.md): N = 30 (min |V\*| 6,260, Baseball; rank 4096 for every concept); N = 20 fails for 9 concepts.

## Edit rule for A1 (edit once)

A token edited by several language fits is edited once, with the edit c_t from the fit of the language whose concept
page has the most words. Native pages rank above pages that were topped up with translations.

## Evaluation and δ selection (no Alpaca)

- Per language L: efficacy_L = 1 − qa_frac_L, specificity_L = HM(mmlu_frac_L, simdom_frac_L),
  fluency_L = min(1, ppl_base,L / ppl_edit,L) on the held-out native text; normalizations as in EMBER.
- H_L = HM(efficacy_L, specificity_L, fluency_L). Selection score = mean over the 8 languages of H_L.
- Languages whose unedited QA_train accuracy is below 0.40 (Part 1 cutoff; chance is 0.25) are excluded from the selection score and from reported efficacy; their gates still apply. (Added 2026-10-19 before any grid row was scored, after the baselines showed e.g. Valentine's Day th = 0.26.)
- Gates (per language, as EMBER's MMLU and Alpaca gates): mmlu_frac_L ≥ 0.9 and fluency_L ≥ 0.9 for every L,
  otherwise the δ scores 0.
- Grid δ ∈ {0.5, 1, 2, 5, 10, 50, 100, 200} on the train split; one δ per (concept, approach) for all languages.
  Ties go to the smaller δ; if all are 0, no edit. The test split is scored at the chosen δ.
- Also reported: the leak profile (efficacy per language) and the number of edited tokens per language.

## Fixed hyperparameters

k = 200 (Llama), s = 1%, λ = 1e-4, EMBER's ALS; candidates = features with ρ ≥ 2, top 10 by ρ (all approaches);
TJ = gemini-3.5-flash-lite 2-stage judge, accept at confidence ≥ 0.85, thinking off; EJ@5 = ≥ 5 exact matches
between the feature's first 20 normalized source words and the concept's lexicon (20 words per language, one
Gemini call per concept). CS: η = 0.05, γ = 1, ε = 1e-4 relative; background = token types of equal numbers of
native Wikipedia sentences in the 8 languages; target = token types of the 8 concept pages. GS: rn, floor 0.2%,
cap 3%. Seed 42.

## Compute

Llama in fp16 over 2 Titan Xp, or NF4 on 1 (decided by `bench.py`). At most 6 GPUs at a time (shared cap with
the single-language agent), ≤ 4 CPUs per task, nodes s-004 and s-005 excluded, job names `ml_*`.

## Known caveats (fixed in advance)

- CS target overlaps V\* (see CINDER_improvements.md, L1); kept as the default, the erasure metric is label-free.
- MC translations for it/pt/hi/th are machine translations (NLLB), checked by COMET-Kiwi only.
- Labels for A2/A3 come from the multilingual full set, so a token that is a concept token in one language and a
  neutral token in another is "both" and is not edited; A1 labels per language.

## Amendment 2026-10-19: A3z (approved by the user)

A3's edits, mapped back from the whitened space, point into directions layer 0 barely reads (read gain 1.1–1.7×
random vs 2.6–7.6× for A2; results/ml/diagnose_a3/DIAGNOSIS.md), so efficacy plateaus near 0.1 and no gate fires.
A3z keeps everything A3 uses to decide what to edit (whitened fit, GS-norm masks, TWJ selection, edited tokens) and
builds the edit in the original space: Y fixed, Z_o = ridge solve against the original embeddings,
c_t = Z_o[:, F_C] y_t (experiments/ml/a3z.py). Same evaluation, gates, δ grid and test protocol as A1–A3.
Reported as a fourth approach, next to (not instead of) A3. Defined after A3's results were seen.
