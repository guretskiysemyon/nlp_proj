<!-- Saved by the lead from the WP agent's hand-back (a hook blocked the agent's own write); text unchanged. -->

# WP3 (widths): is the GS width a real property of the concept?

## Question

From the motivation document, on Global Sparsity and the fixed budget:

> "To make sure these are the right suited number of tokens (and not just a random budget) I can make the following comparisons/tests:
> - Run multiple times, replace seeds, and see how the amount of tokens allocated for this direction settles. Check if there is also connection to the serial number of the feature – shouldn't be.
> - Obtain the dense matrix back from the sparse one. From the iterative algorithm, just fix Z and look for dense Y'. We should get Y'=Y in the non-zero locations. Then we rank the rest of the tokens: should we have 'extended the tail' or cut it sooner.
> - For each setting separately (EMBER and CINDER) compare the z feature direction obtained by the 'sparse-row' and the 'fixed-budget' and check their similarity."

We also test the claims that "GS needs CS" and that GS width does not track concept breadth (Spearman ≈ 0.05).

## Method

- **Fits.** We use saved fits only and train nothing new. There are 38 concepts × seeds 42–46 for each of: EMBER = raw RS-1% (L0), CS RS-1% (L1), CS GS (L12, CINDER's fit), raw GS (`rn_refit-False_plain`), and CS RS-2% (`fixed_sparsity-0.02_refit-False`, a control for co-adaptation). That is 950 fits. Gemini spend was zero; all computation ran on CPU.
- **Concept feature.** No judge is used. The concept feature is the top-20-by-ρ candidate with the most lexicon matches (EJ count, `concept_words_v1`), ties broken by ρ. This is the rule of the existing judge-free table in `layer_report.py`. To compare width rules on *the same direction*, the **anchor** is CINDER's (L12) concept feature. In every other fit of the same seed we take its best |cos| match. Raw fits are compared in the raw space, with the CS directions mapped back by z̃ T⁻¹.
- **Dense re-solve.** Y' = (ZᵀZ + λI)⁻¹ZᵀA with the saved Z, λ = 1e-4, in float64. Tokens are ranked by the dense share |y'_ij| / ‖y'_i‖.
- **Token labels along the ranking.**
  - C: concept-only (occurs in S_C and never in S_N).
  - LEX: the token's source word is one of the 20 lexicon words.
  - **X20, page-enriched (primary, strict):** the token occurs in S_C and ln[p_page / p_wiki] ≥ ln 20. Here p_page is its frequency in the 300 concept sentences. p_wiki is its frequency in 150k articles (162M tokens) of the cached `wikimedia/wikipedia` shard 40/41, with add-0.5 smoothing.
- **Natural tail T.** Fixed before looking at any width-vs-tail statistic.
  - Primary rule: T is the first rank m at which fewer than half of the next 20 tokens are X20.
  - Robustness grid: 72 definitions. These are labels C / X5 / X20 / X100 × windows 10/20/30 × thresholds 0.3–0.7, plus a one-change-point Bernoulli MLE over the top 200/300/400 ranks.
- **Units and tests.** The concept is the unit: seed means, n = 38 unless stated. We report Spearman correlations with bootstrap 95% CIs (10k resamples over concepts), and paired Wilcoxon signed-rank tests on per-concept differences.

## Backed findings

All numbers come from `results.json`, produced by `analyze.py`.

**1. Seeds and serial number.**
- **Widths across concepts and seeds (CS GS).** Concept-mean widths span 21–114 tokens. Across seeds the median SD is 7.7 tokens and the median range is 17. 66% of the width variance lies between concepts (ICC(1) = 0.66; Kruskal–Wallis over concepts p = 1e-12).
- **Most seed spread comes from identification switches.** Sometimes a different feature is picked, for example Baseball's "pitch" feature instead of its "bat" feature, or a non-concept feature. Within the largest group of seeds whose concept features match each other (|cos| > 0.9), the median SD is **4.0 tokens** and the median range is 9. All 5 seeds fall in one group for 18 of 38 concepts.
- **Raw GS is less stable.** Its median SD is 9.5 tokens and its ICC is 0.49.
- **Width does not depend on the feature index.** Pooled over 190 runs × 100 features, Spearman(width, index) is −0.004 (p = 0.59). Within single runs, 4.7% of runs are significant at 5%, as the null predicts. The index of the concept feature is uniform over deciles (χ² p = 0.66), and it does not predict the concept feature's width (ρ = 0.05, p = 0.48, n = 190 runs).
- **The slot is set by the random init.**
  - At the same seed, the L1 match of CINDER's concept feature sits at the same index in 68% of runs (RS-2%: 53%).
  - Across seeds this happens in 0.9% of runs (chance: 1%).
  - So the same init puts the same concept in the same slot under either width rule, and the rule still changes its width.
  - On raw GS, the within-concept index–width correlation is −0.17 [−0.31, −0.01]. This is one of several tests, so we treat it as chance.

**2. Dense re-solve.**
- **Y' = Y on the support in all 950 fits.** The maximum relative error is 2.2e-6. Every row's support is exactly the top-|S| of its dense row. This holds by construction for refit = False.
- **Every GS support is a single global threshold.** In all 380 GS fits, every row that is neither at the floor nor at the cap keeps exactly the tokens with share > τ, for one τ per fit (median τ = 0.037 with CS, 0.044 raw). No concept feature sits at the floor or the cap.
- **EMBER's 1% cut is usually too early in the CS space.** On the CS RS-1% fit, the median natural tail is **T = 69 tokens**, against the RS width m_RS = 47. T / m_RS has median 1.54 (IQR 0.90–2.17). RS-1% cuts **25/38 concepts short** (T > 1.2 m_RS), pads 8 (T < 0.8 m_RS), and is about right for 5.
- **In the raw space it is short and padded equally often.** On EMBER's own raw fit, the median T is 46, with 15 cut short, 15 padded and 8 about right.
- **The tail is a property of the direction, not of the fit's budget.** Doubling the fit's width (RS-2%, +46.5 tokens) moves T by only +3.2 tokens (median; p = 0.02). Spearman(T_RS1%, T_RS2%) = 0.96, and Spearman(T_RS1%, T_GS) = 0.94.
- **GS moves the width in the right direction.** Of the 28 concepts with T > m_RS, GS widens 26. Of the 10 with T ≤ m_RS, GS does not widen 7 (Fisher p = 0.0003).
- **GS width tracks the tail; RS width does not.**
  - Spearman(GS width, T) = **0.67 [0.42, 0.84]**, against 0.27 [−0.10, 0.57] for the RS width.
  - Mean absolute error is **30.2 vs 42.4 tokens**: a paired gain of 12.2 [5.3, 19.3], Wilcoxon p = 0.0009, with GS closer for 28/38 concepts.
  - The level-free measure agrees. The SD over concepts of ln((w+1)/(T+1)) is 0.79 for GS and 1.01 for RS (difference CI [0.07, 0.34]).
  - The result holds with T from GS's own fit (ρ = 0.76), from the RS-2% fit (0.62), and from their mean (0.73).
- **GS compresses widths toward the middle.** The Theil–Sen slope of GS width on T is 0.30 [0.18, 0.41]. GS overshoots the lowest-T third of concepts by +24 tokens and undershoots the highest-T third by −44 (medians). Examples: Buddhism 100 vs 182, Judaism 75 vs 154, Sigmund Freud 62 vs 137.
- **Robustness over the 72 tail definitions.**
  - ρ > 0 in 70/72 definitions; p < 0.05 in 55/72.
  - We then drop the degenerate definitions: X100, where T collapses to ≈0, and the lenient ones with median T > 200, which run into the base rate. In the 45 that remain, ρ = 0.37–0.71 with p < 0.05 in all 45, the GS MAE is lower in 41/45, and the log-ratio SD is lower in 45/45.
  - The MAE advantage flips only for strict rules whose median T is about 40, near the RS width (X20 at θ = 0.7). The rank correlation survives there too.
- **Sensitivity to the anchor.** Restricted to anchors that pass EJ@5 (25 concepts, 119 runs): ρ = 0.54 [0.13, 0.82] and MAE 30.7 vs 41.5 (p = 0.006). Restricted to anchors hand-labelled P (25 concepts, 109 runs): ρ = 0.70 and MAE 32 vs 46 (p = 0.003).

**3. Breadth.**
- **GS width ignores token-count breadth proxies.** Its Spearman with:
  - concept-only tokens in V*: 0.05 [−0.30, 0.40] (this reproduces the earlier ≈ 0.05);
  - page-enriched tokens in V*: 0.21 [−0.14, 0.54];
  - page token types: 0.05;
  - page sentences: −0.02 (n = 37);
  - lexicon words present in S_C: −0.31 [−0.56, −0.00].
- **It does track the natural tail:** 0.67 [0.42, 0.84].
- **The old breadth measure is |V*| in disguise.** "Concept-only tokens in V*" has exactly the same ranks as |V*|, because S_N is the same for every concept. So the old breadth measure is a vocabulary-size count, and RS-1% width (= ⌈0.01|V*|⌉) correlates with it at 0.99 trivially.
- **The tail itself is only loosely tied to the counts.** Natural tail vs page-enriched count: 0.46; vs concept-only count: 0.25.

**4. RS vs GS directions (same seed, same space).**
- **GS changes the token set, not the direction.**
  - CINDER's concept feature vs its RS-1% match: |cos| median **0.973** (mean 0.944 [0.92, 0.96]); > 0.9 in 86% of runs.
  - Seed noise for comparison: RS-1% across seeds 0.975, GS across seeds 0.961.
  - Paired GS-vs-RS minus GS seed noise: +0.007 [−0.01, 0.02], p = 0.18. So GS does not change the direction beyond seed noise.
  - Token Jaccard is 0.60 [0.55, 0.65], against 0.71–0.72 seed noise. The overlap coefficient is 0.91: the smaller support lies almost inside the larger.
  - The two fits identify the same feature independently in 79% of runs.
- **Raw space, for reference.**
  - EMBER's concept feature vs the CS RS-1% feature mapped back: |cos| median 0.75, 0/190 runs above 0.9, against EMBER seed noise of 0.95; Jaccard 0.48. CS changes the direction; GS does not.
  - Raw RS vs raw GS: |cos| 0.97, Jaccard 0.63.

**5. GS needs CS (38 concepts).**
- **Mean neutral-only share of the concept feature:** raw RS (EMBER) 12.1%, raw GS 10.1%, CS RS 6.3%, CS GS 6.2%.
  - CS GS is below raw RS in 37/38 concepts (p = 2e-11).
  - Raw GS is *below* raw RS (−2.0 pts [−3.3, −0.9], p = 0.0008, 28/38 concepts). This is mostly because raw GS narrows the concept feature: 43 tokens on average, against 64 with CS.
- **The tokens raw GS adds are neutral.** In the part of the support beyond m_RS, raw GS's extension is 16.4% neutral, against 7.2% in its first m_RS ranks (27 concepts, p = 0.0009). For CS GS the figures are 8.2% vs 3.9% (35 concepts).
- **In the raw space GS gives no error gain.** Raw GS width tracks the raw natural tail only weakly: ρ = 0.44 [0.12, 0.69], MAE 21.5 vs 23.9 for RS (p = 0.39).
- **Verdict.** "GS needs CS" holds as "GS adapts width usefully only after CS". The 3-concept form "raw GS features are more neutral than RS" does not generalize. Per concept, at the current 3% cap (raw GS vs raw RS):

  | Concept | Raw GS neutral share | Raw RS neutral share |
  | --- | --- | --- |
  | Ancient Rome | 26% | 33% |
  | Harry Potter | 13% | 12% |
  | Valentine's Day | 7% | 15% |

**Contradictions with the brief.**
- **Widths.** The current (3% cap) fits give Baseball **50 ± 4** tokens under GS, not 15. The concept-mean range is 21–114. The low end, The Matrix, is a "matrix/grid" feature with only 2 lexicon matches, not the concept.
- **Seed stability.** "±4 across seeds" holds only within seeds whose concept feature is identified consistently. Over all seeds the median SD is 7.7.
- **Ancient Rome.** One seed (43) picks a 27-token borderline feature. The other four give 92–109 tokens.

## Speculations (with tests)

- **Why GS tracks the tail.** GS keeps share > τ. The share drops below τ at about the point where the tokens stop being page-enriched: at T the share is 0.77·τ (median; IQR 0.63–1.25). In other words, the label transition coincides with a knee in the magnitude. *Test:* fit a two-segment model to log-share vs rank, and correlate the knee position with T.
- **Why broad concepts are undershot.** The row-norm score penalizes a feature whose energy spreads over many tokens: at most 1/τ² entries can exceed τ, and the broad dense tail inflates ‖y'_i‖. *Test:* regress (w_GS − T) on the share of ‖y'_i‖² outside the top 600 (`energy_top600` in features.csv), and rerun GS with the score normalized by the top-k norm instead of the full norm.
- **Why raw tails are short.** Raw features mix concept tokens with high-variance generic ones (WP brief), so page-enriched precision decays early (median T 46 vs 69 with CS). *Test:* compare the C / X20 precision curves of the raw and CS anchors for the same concept direction, already in `curves.pkl`.

## Paper paragraph

Under row sparsity every feature keeps 1% of the tokens, so the width is a fixed budget. Under global sparsity the width is an output of the fit. To test whether that output reflects the concept, we fix the learned dictionary Z and re-solve Y densely. This reproduces the sparse code exactly on its support and ranks every other token by its share of the concept feature. Along this ranking the fraction of page-enriched tokens stays high and then drops. We call the rank where it falls below one half the feature's natural tail. The natural tail is a property of the concept direction: doubling the fit's budget moves it by only three tokens, and its rank order is nearly unchanged (Spearman 0.96). On the contrastively sharpened matrix, EMBER's 1% cut stops before the natural tail for 25 of 38 concepts. Global sparsity widens 26 of the 28 features that are cut short, and its width tracks the tail (Spearman 0.67 [0.42, 0.84] vs 0.27 for the fixed width; mean error 30 vs 42 tokens, p < 0.001). The tracking is robust across 45 non-degenerate tail definitions, although GS compresses the range, cutting the broadest concepts short. The width is stable across seeds once the same feature is identified (median SD 4 tokens), and it is unrelated to the feature's index. Global sparsity leaves the direction itself unchanged within seed noise (|cos| 0.97), whereas sharpening changes it (0.75). On the raw embedding matrix, the width that global sparsity adds is mostly neutral tokens and gives no gain in error.

## Appendix text

**Setup.** Gemma-2-2B-it input embeddings (×√d). V* = token types of 300 concept and 300 neutral sentences. k = 100, λ = 1e-4, refit off (Y = Y_dense ⊙ S, as in EMBER), unit-norm Y rows. RS-1%: ⌈0.01 n⌉ per row (43–52 tokens). GS: K = k⌈0.01 n⌉ entries chosen globally by |y_ij|/‖y_i‖₂, with floor ⌈0.002 n⌉ and cap ⌈0.03 n⌉. CS: whitening against Wikipedia token types, then sharpening toward the concept page (η = 0.05, γ = 1). Five seeds (42–46) per fit. Each seed fixes the random N(0,1) initialization of Y, and that initialization is identical across width rules.

**Dense re-solve.** Y' = (ZᵀZ + λI)⁻¹ZᵀA with the saved Z, A the fit's matrix (raw Eᵀ or the CS Xᵀ), float64.
- Check 1: ‖(Y' − Y) ⊙ S‖ / ‖Y‖ per row (max 2.2e-6 over 95,000 rows).
- Check 2: each row's support equals its top-|S| by |y'|, which holds for all rows.
- Check 3, GS only: max off-support share ≤ min on-support share over all rows strictly between floor and cap (all 380 fits).

**Labels.**
- **C, concept-only:** occurs in S_C, never in S_N. We checked that this equals the prep codes for 100% of tokens.
- **Xθ, page-enriched:** c_C ≥ 1 and ln[(c_C + 0.5)/N_C] − ln[(c_W + 0.5)/N_W] ≥ ln θ. Here c_C is the count in the 300 concept sentences (N_C tokens), and c_W the count in 150,000 random articles of `wikimedia/wikipedia` 20231101.en shard 40 (N_W = 162M tokens; titles equal to a study concept dropped). θ = 20 is primary; 5 and 100 are robustness checks.
- **LEX:** normalize(most frequent source word) is in the concept's 20-word lexicon (`concept_words_v1`), the same normalization as EJ.

**Natural tail.**
- Window rule: T = min{m ≥ 0 : mean(x_{m+1..m+w}) < θ}, with x the label sequence of the dense ranking.
- Change-point rule: argmax_T of the two-segment Bernoulli log-likelihood over the first N ranks, with p_head > p_tail.
- Primary: X20, w = 20, θ = 0.5. Per-concept T is the mean over the 5 seeds of the anchor direction.
- RS classes: cut short if T > 1.2 m_RS, padded if T < 0.8 m_RS, otherwise about right.

**Concept feature and matching.**
- Concept feature: the top-20-by-ρ feature with the most exact matches between its first 20 normalized source words and the lexicon, ties broken by ρ.
- Anchor: the L12 concept feature.
- Matches: argmax |cos| over the other fit's 100 columns of Z, in the CS space for CS–CS pairs and in the raw space otherwise (z̃ T⁻¹).
- Seed-noise baselines: the same matching across seeds within one rule (20 ordered seed pairs per concept).

**Statistics.**
- Concept-level means over seeds, n = 38 unless stated.
- Spearman with a pairs bootstrap (10,000 resamples) 95% CI.
- Paired Wilcoxon signed-rank on per-concept differences, with a bootstrap CI of the mean difference.
- Fisher exact test for the direction table.
- ICC(1) from a one-way ANOVA of widths on concept.
- Theil–Sen slope with 95% CI.

**Full tables.**
- `tables.md`: per concept — m_RS, GS width ± SD, RS-2% width, T for each fit, RS class, breadth proxies.
- `natural_tail_per_concept.csv`, `tail_robustness.csv` (all 72 definitions × 3 reference fits).
- `seed_widths_*.csv`, `seed_widths_consistent_L12.csv`, `directions_per_concept.csv`, `gs_needs_cs_per_concept.csv`.
- `features.csv` (one row per fit × seed × own/anchor feature), `runs.csv`, `cross_match.csv`, `concepts.csv`.

**Caveats.**
1. **Frequent concept words fail X20.** Words such as independence, revolution, war or matrix are common in general Wikipedia, so they are not page-enriched. T is then too small for concepts like American Revolutionary War (7) and French Revolution (8). Their anchors also carry few lexicon matches.
2. **The judge-free anchor is sometimes not the concept.** Examples: The Matrix (a matrix/grid feature), Karate (Japanese names), Republic of Ireland in 2 seeds. The EJ@5 and hand-P subsets address this.
3. **The labels use the same page as CS.** The page-enriched label uses S_C, the same page CS sharpens toward. This cannot favour GS over RS, because both widths are scored on the same CS fits. It may still inflate the level of T.
4. **T comes from co-adapted fits.** Its dependence on the fit's budget is small (+3 tokens for +46 tokens of budget), but it is not zero (p = 0.02).

## Figures

All figures are saved as PDF and PNG, with the plotted data as CSV next to each.

- `fig_precision_ranking.pdf` (6.8 in). Fraction of page-enriched tokens among the next 20 ranks of the dense re-solve, along the dense ranking of the concept direction, for Baseball, Harry Potter, Ancient Rome and Culture of Greece. Blue: CS RS-1% fit. Orange: CS GS fit (CINDER). Both are means over 5 seeds. Markers: grey dashed line, RS-1% cut; orange dash-dot line, GS width (seed mean); blue triangle, natural tail of the RS-1% fit (seed mean of per-run tails); dotted line, base rate in V*.
- `fig_tail_vs_width.pdf` (3.3 in). Natural tail T (RS-1% fit, primary rule) vs the width of the concept feature, one point per concept (n = 38). Orange: GS width ± 1 SD over seeds. Grey: RS-1% width. Line: identity. Spearman 0.67 vs 0.27.
- `fig_gs_needs_cs.pdf` (3.3 in).
  - Left: neutral-only share of each fit's concept feature, per concept (dots) with the median (bar).
  - Right, GS fits only: neutral-only share of the first m_RS ranks of the support ("head") vs the ranks beyond m_RS ("extra"). One line per concept; black is the median.

## Sources

- **Code** in `experiments/paper_analysis/widths/`:
  - `wikifreq.py`: Wikipedia token counts → `wiki_counts.npz`.
  - `collect.py`: labels, dense re-solves, anchors and matches → `cache/<slug>.pkl`, 49 MB.
  - `analyze.py`: all statistics → `results.json` and the CSVs.
  - `figures.py`: the figures.
  - `common.py`: shared constants and helpers.
  - Run order: wikifreq → collect → analyze → figures, all as `.venv/bin/python -m experiments.paper_analysis.widths.<module>`.
- **Inputs:**
  - `results/varwidth/<Slug>/<tag>/seed4[2-6]/{fit.pt, layer_cache.json}` for the tags `fixed_refit-False_plain`, `fixed_refit-False`, `rn_refit-False`, `rn_refit-False_plain` and `fixed_sparsity-0.02_refit-False`.
  - `results/varwidth/prep/<Slug>_{plain,whitened}_S300.pt`.
  - `experiments/varwidth/j3_data/concept_words_v1.json` and `labels*.json` (via `layer_report.labels()`).
  - `data/concept_sentences*.json` and `experiments/whitening/data/pages/*.json`.
  - The cached shard `wikimedia/wikipedia` 20231101.en `train-00040-of-00041.parquet`.
