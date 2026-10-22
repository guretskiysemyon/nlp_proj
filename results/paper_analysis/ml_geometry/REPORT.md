<!-- Saved by the lead from the WP agent's hand-back (a hook blocked the agent's own write); text unchanged. -->

# WP9 `ml_geometry`: language identity in the multilingual fits, and the exact selection cost

Llama-3.1-8B-Instruct, 8 languages, 38 concepts, fit seeds 42 / 1 / 2. Read-only on the multilingual study (worktree `../nlp_proj_ml`). CPU only, zero Gemini.

## Question

> 1. Why does a single multilingual EMBER fit (A2) so rarely find a concept feature? Hypothesis: in a multilingual embedding matrix, language and script identity are the loudest directions, so sparse MF spends its features on them and the concept becomes even quieter than in English. [...] Also: the share of the multilingual matrix's energy along the top directions that separate languages, raw vs CS (enrichment 6.82 with an English background vs 2.82 with an 8-language background; find and reproduce it).
> 2. Are CINDER's multilingual concept features shared across languages?
> 3. Exact cost table, per concept and in total, per approach. [...] State clearly which part is saved.

## Method

- **Token language labels.** Every V\* token comes from the 8-language full set (300 concept + 300 neutral sentences per language).
  - p(L | t) is the token's occurrence distribution over languages, normalised by each language's token count (`tokens.py`).
  - A hard label needs p ≥ 0.8. The other 39% of sample tokens are "shared".
- **Per feature** (`features.py`). All 200 features × 38 concepts × 3 seeds for A2 and A3, plus the 8 per-language A1 fits:
  - q_L, the language mass: Σ_t |Y_it| p(L|t) / Σ_t |Y_it|. The max-language share is max q_L; **language-dominated** means ≥ 0.8.
  - Script mass: Latin, Devanagari, Thai.
  - ρ (EMBER's ratio_abs) and its rank.
  - **Candidate**: ρ ≥ 2 and top 10 by ρ, the study's judged set.
  - **EJ hits**: matches of the first 20 normalised source words against the 8-language lexicon (`results/ml/lexicons.json`). This is select.py's rule exactly; it reproduces the saved A3 `ej.feature_words`. **EJ5** means ≥ 5 hits.
  - **TJ**: the cached Gemini verdicts.
  - Null: random supports of the same widths.
- **Coverage 2×2.** Fit (raw A2 vs CS+GS A3) × judge (TJ vs TWJ = TJ ∨ EJ@5), on the saved fits.
  - A concept is covered if one of its 10 candidates is accepted.
  - The as-run cells reproduce COVERAGE.md exactly.
- **Blind labels.** The 46 EJ-only accepts were labelled P (specific), B (broader) or N (unrelated). Approach and seed were hidden and order shuffled (`blind/`).
- **Energy** (`energy.py`). On the N = 30 + 30 sample matrix that A2/A3 factorise, in raw, whitened-only and full CS form (as `experiments/ml/fit.cs_transform`):
  - between-language share of the centred energy;
  - concept-vs-neutral share;
  - energy in the 7-dim language-centroid span;
  - η² of language and concept on the top 10 PCs.
  - cPCA enrichment g under the 8-language background vs an English Wikipedia background. The English one is WP8's `results/paper_analysis/llama_cs/data/bg_wiki_llama_stats.pt`, because the original Llama `wiki_en` stats no longer exist.
  - Purity: mean max-language share of the 50 V\* tokens projecting highest on each of the top 5 directions.
- **Cost** (`cost.py`):
  - fit_summary, selection and judge files for every (concept, seed, approach);
  - `gemini_usage.jsonl` with NULs replaced by newlines. It has no purpose tag, so per-call means come from judge-like and lexicon-like records, and predicted counts are reconciled per time window;
  - `[worker] done … in N min` lines for evaluation time.
- **Statistics.** Concept is the unit (mean or sum over seeds). Wilcoxon signed-rank, 10k bootstrap 95% CI.

## Backed findings

### Q1. A2's low coverage comes mostly from the judge, partly from language features

1. **Language is loud and the concept is quiet** (raw multilingual matrix, 38 concepts).
   - Language means: 1.61% of centred energy (1.56–1.65%). Concept-vs-neutral mean: 0.047%. Ratio 34× (median).
   - The language-centroid span holds 2.3% of the energy, 13.6× a random 7-dim subspace.
   - Top 10 PCs: η²_lang 0.25, η²_concept 0.004.
   - The concept is quieter than in English: 0.138% in the English-only matrix vs 0.047% here (2.9×, 38/38, p < 10⁻¹¹).
   - CS halves the language share (1.61% → 0.79%, 38/38, p < 10⁻¹¹) but does not raise the concept mean (0.030%). Language is still 26× louder.
2. **The enrichment check reproduces, but "the clusters disappear" is too strong.**
   - g = 8.16 (7.36–9.00) with the English background vs 2.55 (2.16–5.17) with the 8-language one (38/38, p < 10⁻⁷).
   - HP: 7.91 vs 2.28 (the earlier notebook had 6.82 vs 2.82).
   - Top-5 purity stays high: 0.93 vs 0.88. HP's top directions under the 8-language background are fr/it/de/it/de function words.
   - So the language directions become about 3× weaker; they are not removed.
3. **The raw fit spends features, and more of its judged slots, on single languages.**
   - Language-dominated, all features: A2 8.6% vs A3 4.9% (38/38, p < 10⁻⁷). Null: 0.0%. Median max share 0.28 vs 0.24 (null 0.20).
   - Among candidates: A2 13.7% (1.4 per fit; German 86, English 57) vs **0%** for A3 (36–0, p < 10⁻⁶).
   - In A2 fits that contain an EJ5 feature, the fits where it misses the top 10 have more language-dominated candidates (1.83 vs 1.13; Mann–Whitney p = 0.013, 24 vs 54 fits). This is correlational.
   - Over the whole A2 fit, the language-dominated features are mostly Thai (608), German (426) and Hindi (383).
4. **The raw fit still builds and ranks concept features.**
   - An EJ5 feature exists in A2 25/26/27 of 38 fits (seeds 42/1/2) vs A3 28/23/25.
   - It is in the top 10 in A2 17/20/17 vs A3 16/9/14.
   - Median ρ rank of the best EJ5 feature: 5 for A2 vs 9.5 for A3 (in A3, 183/200 features pass ρ ≥ 2).
5. **The loss is at EMBER's judge.** TJ acceptance by lexicon hits in the top 20:

   | | 1–2 | 3–4 | 5–7 | 8+ |
   |---|---|---|---|---|
   | A1 | 4% | 30% | 51% | 96% |
   | A2 | 1% | 5% | 41% | 70% |
   | A3 | 8% | 30% | 48% | 93% |

   - The judge-free composition of the top-20 tokens barely differs between A2 and A3 at ≥ 3 hits: concept-only share 0.70 vs 0.68, English share 0.33 vs 0.27.
   - Typical rejections:
     - Cannabis: marijuana, cannabis, alcohol, tobacco, THC, described as "psychoactive substances";
     - Tuberculosis: infection, antibiotic, vaccine, described as "infectious diseases".
6. **Coverage, fit × judge** (seeds 42/1/2, union over seeds in parentheses):

   | | TJ | TWJ |
   |---|---|---|
   | raw (A2) | **10/9/12** (13), as run | 18/20/18 (25) |
   | CS+GS (A3) | 16/16/16 (26) | **21/17/20** (29), as run |

   Per concept, counting seeds covered (0–3):
   - Under TJ, CS beats raw: +0.45 [+0.24, +0.68], 15–1, p < 0.001.
   - **Under TWJ, CS and raw are tied:** +0.05 [−0.21, +0.32], 11–8, p = 0.73.
   - On the raw fit, TWJ beats TJ: +0.66, 15–0, p < 0.001.
   - As run, A3 beats A2: +0.71, 18–1, p < 0.001.
7. **Blind labels of the EJ-only accepts:** A2 9 P / 20 B / 0 N; A3 6 P / 11 B / 0 N.

**Answer to Q1.**
- Language and script directions are about 30× louder than the concept mean, and the concept is about 3× quieter than in English.
- The raw fit gives 14% of its judged slots to single-language features; CINDER gives 0%.
- But the raw fit contains and ranks concept features about as often as CINDER does. The as-run gap (10 vs 21 concepts) comes mainly from TJ rejecting raw multilingual concept features.
- CS+GS fixes this under TJ (+6 concepts), and the lexicon judge fixes it on the raw fit (+8). The two act as substitutes (+9 together).
- The claim should read "CS+GS or the two-way judge", not "CS finds features the raw fit lacks".

### Q2. A3's concept features are more spread across languages, but much of the extra spread is noise

Selected features (F_C), all seeds:

| | n | Median max share | Entropy (bits) | ≥ 2 scripts | 3 scripts | Devanagari+Thai mass | hi/th lexicon word in first 50 |
|---|---|---|---|---|---|---|---|
| A1 | 316 | 0.44 | 2.26 | 15% | 0% | 3.3% | 10% |
| A2 | 36 | 0.43 | 2.38 | 0% | 0% | 1.2% | 22% |
| A3 | 109 | 0.30 | 2.71 | 34% | 13% | 7.3% | 34% |

- Paired, on the 13 concepts both select: A3 − A2 max share = −0.09 [−0.14, −0.04], 10–3, p = 0.008.
- Own-language mass in A1 fits: English 0.72, Hindi 0.16, Thai 0.30. The Hindi and Thai fits are mostly Latin pieces.
- Much of A3's Devanagari and Thai mass is generic words (एक, है, สามารถ). Some is real concept vocabulary: युद्ध, प्रक्षेप्य, धातु for Gun; विषमलैंगिक for Pornography.
- Cross-language sharing mostly goes through shared Latin tokens: pistol / pistole / pistola, revolution / rivoluzione / revolução. Rome/Roma/रोम-style alignment across scripts is rare.

### Q3. Exact selection cost (38 concepts; per concept, mean (min–max))

| Seed 42, per concept | A1 | A2 | A3 | A3z |
|---|---|---|---|---|
| Factorizations | 8 | 1 | 1 | 0 |
| Rows | 30,598 (28,405–32,584) | 6,474 (6,150–6,984) + solve on 18,925 | same as A2 | — |
| ALS iterations | 3,119 | 342 | 365 | — |
| Rows × iterations | 11.0 M | 2.2 M (1/5.0) | 2.4 M (1/4.7) | — |
| Fit wall time, GPU | 44 s (37–98) | 7 s (5–14) | 40 s (33–53; includes the CS eigendecomposition) | +0.4 s solve |
| Fit wall time, CPU (seeds 1, 2) | 39–40 s | 42 s | 76 s | — |
| Judged candidates | 80 | 10 | 10 | as A3 |
| Judge calls | 160 | 20 | 20 | as A3 |
| Lexicon calls | 0 | 0 | 1 (cached across seeds) | as A3 |
| Tokens in / out | 30,420 / 2,933 | 3,802 / 367 | 4,031 / 1,478 | as A3 |
| Gemini cost | 0.055 NIS | 0.0069 NIS | 0.0174 NIS | as A3 |
| Evaluation, GPU min per edited unit | 150 (59–298), n = 26 | 190 (46–280), n = 12 | 211 (46–241), n = 22 | 191 (111–252), n = 21 |

**Totals.**
- Seed 42: A1 304 factorizations, 6,080 calls, 2.10 NIS; A2 38 factorizations, 760 calls, 0.26 NIS; A3 38 factorizations, 760 + 38 lexicon calls, 0.66 NIS.
- Each extra seed costs A1 2.10 NIS and A3 0.26 NIS.
- Evaluation: A1 68 h, A2 34 h, A3 71 h, A3z 66 h.

**Per call.**
- Judge: 190 / 18.3 tokens, 0.000345 NIS (n = 25,037).
- Lexicon: 228 / 1,112 tokens, 0.0105 NIS (n = 40). One lexicon call costs about as much as 30 judge calls.

**Reconciliation with the usage log.**
- Oct 18, 20h: 3,728 judge-like records vs 3,600 predicted, and 20 lexicon-like vs 18.
- Oct 20, 16h: 4,000 vs 4,000 and 20 vs 20.
- Seeds 1 + 2: 15,139 vs 15,200. The 61 missing fit the NUL corruption (123 NUL bytes, 27 unparseable fragments).
- Study total: 17.11 NIS over 25,948 records. Of this, 8.04 NIS is question generation, 0.75 NIS the stability study, and 8.32 NIS selection.

**What is saved.**
- Judge calls: exactly 1/8.
- Factorization work: about 1/5.
- Gemini cost: 31% of A1's, or 12.5% once the lexicon is cached.
- A3's GPU fit wall time is not lower, because the CS transform costs about as much as A1's 8 fits.
- **Evaluation is not saved.** It is the same protocol per unit, 2.5–3.5 GPU-h, about 250× the fit stage (which is under a minute per concept). The 1/8 is a saving in judge and API cost, not in total compute.

## Speculations (untested)

- **Why TJ rejects raw features at equal lexicon evidence.** Raw features put the concept next to its English neighbours, and the judge describes the neighbourhood. CS features carry the concept name with multilingual variants. Test: re-describe a matched A2/A3 set with and without the neighbour tokens (about 0.1 NIS).
- **Raw+TWJ edits may cost more specificity.** Test: evaluate A2 with EJ@5 on the roughly 8 concepts it newly covers (about 25 GPU-h).
- **A top-20 candidate list would help A2 more than A3**, by recovering EJ5 features displaced by language features. Test: EJ only, judge-free.

## Paper paragraph

Why does a single multilingual EMBER fit so rarely yield a concept feature? In the multilingual embedding matrix, language identity is loud: language means carry 34× the energy of the concept-vs-neutral mean, and the concept is 3× quieter than in an English-only matrix. The raw fit spends 14% of its judged candidates on single-language features (German and English function words), against none for CINDER. Yet the raw fit contains a lexicon-matched concept feature about as often as the CINDER fit (25–27 vs 23–28 of 38 concepts per seed), and ranks it among the judged candidates as often. The loss happens at EMBER's judge, which accepts 5% of raw multilingual candidates with 3–4 lexicon words in their top 20, against 30% for CINDER's and for per-language EMBER's. Either part of CINDER's selection therefore closes most of the gap: with EMBER's judge, CS+GS raises coverage from 10 to 16 concepts (seed 42); with the two-way judge, the raw fit reaches 18 and CINDER 21, and over three seeds the difference is not significant. The single fit cuts judge calls by exactly 8× (20 vs 160 per concept) and factorization work by about 5×. Evaluation, at 2.5–3.5 GPU-hours per concept and approach, is unchanged.

## Appendix text

**Definitions.**
- p(L|t) = (c_L(t)/N_L) / Σ_L′ (c_L′(t)/N_L′).
- q_L = Σ_t |Y_it| p(L|t) / Σ_t |Y_it|. Language-dominated means max q_L ≥ 0.8.
- EJ is select.py's rule applied to every feature.
- Candidates are ρ ≥ 2, top 10. Every fit judged exactly 10 in all seeds.
- TJ = is_member and confidence ≥ 0.85. TWJ = TJ ∨ EJ@5.

**Energy settings.**
- Sample V\*: 6,150–6,984 rows.
- CS: X = (E − μ_B) W S_η^{1/2}, η = 0.05, γ = 1, ε = 10⁻⁴ × mean background eigenvalue.
- Background: 5,000 native sentences per language. Target: the 8 concept pages.
- Shared tokens (max p < 0.8) are excluded from the language measures; code-2 tokens from the concept measures.
- cPCA: svd_lowrank with q = 25, g = (1−η)σ²/m + η.

**Cost sources.**
- A1's fit_summary "secs" is cumulative over its 8 languages (t0 is set before the loop in `experiments/ml/fit.run`). The A1 total is the sum of the last value of each monotone run.
- Wall time includes tokenizing, the CS transform and the extrapolation solve.
- Seed 42 ran on a local GPU of c-001. Seeds 1 and 2 ran on CPU (A1 on SLURM CPU nodes, A2/A3 on c-001).
- Prices: 0.25 / 2.50 USD per 1M tokens, about 3.70 NIS/USD.
- Evaluation time: the last `[worker] done` line per unit; 5 duplicates were dropped.

**Per-concept data.**
- The 38-row table (language-dominated fractions and the 2×2 coverage, in seeds of 3) is in `per_concept.csv`.
- Examples:
  - Tuberculosis, Doctor of Philosophy, Buddhism, Prime number and French Revolution are covered by raw+TWJ but not by raw+TJ.
  - Golf, Ancient Rome, Uranium, Heroin and Existentialism are covered only by the CS fits.

**Caveats.**
- The lexicons are Gemini-written.
- One rater for the blind labels.
- The non-as-run coverage cells are counterfactual: no edit was evaluated for them.
- The English background differs from the original check's.

## Figures

`fig_ml_language.pdf` / `.png` (6.8 in; data in `fig_ml_language_data.json`).

*Caption:* Language identity and concept selection in a single multilingual fit (Llama-3.1-8B, 38 concepts, 3 seeds).
- (a) Per-feature max-language share for multilingual EMBER (grey) and CINDER (orange). Ticks mark the selected concept features. The random-support median is 0.20, with none above 0.8.
- (b) Concepts covered, by fit and by judge (TJ hollow, TWJ filled). Bars are means and dots are seeds; the outer bars are the approaches as run.
- (c) TJ acceptance by the number of lexicon words in the top 20, for A1, A2 and A3.

## Sources

- Code in `experiments/paper_analysis/ml_geometry/`: `common`, `tokens`, `features`, `top20`, `energy`, `cost`, `report`.
- Outputs in `results/paper_analysis/ml_geometry/`: `features.csv.gz`, `fits.csv`, `top20.csv`, `energy.json`, `energy_per_concept.csv`, `cost_per_unit.csv`, `cost_totals.csv`, `cost.json`, `eval_minutes.csv`, `coverage_2x2.csv`, `per_concept.csv`, `summary.json`, `blind/`, `cache/`.
- Inputs, read-only: `../nlp_proj_ml/results/ml/{runs,seeds,data,lexicons.json,gemini_usage.jsonl,logs}`.
