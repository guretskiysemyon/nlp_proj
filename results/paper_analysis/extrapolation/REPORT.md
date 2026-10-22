<!-- Saved by the lead from the WP agent's hand-back (a hook blocked the agent's own write); text unchanged. -->

# WP5: Extrapolating concept features to unsampled page tokens

## Question

The motivation document's section "Extrapolation" raises four points:
> EMBER edits only tokens of the 300 sampled concept sentences; pages have 500–1000 sentences, and the multilingual pipeline samples only 30–40 sentences per language. "That's where this sparse extrapolation may succeed big time – we depend on it!" · "Naïve extrapolation by projecting on the Harry Potter feature direction … mixed results … mainly mixed with the 'magic' concept … maybe we can project on these 2 dimensions … such that the magic part will go the magic direction." · "Sparse extrapolation – finding a sparse Y' … while fixing Z – seems to bring a few good tokens, but also many indirect/unrelated tokens." · Hypothesis: unsampled tokens are rarer and noisier; "the 1% … increased proportionally from 48 to 59 … we may want to decrease and re-tune the 1%."

## Method

**Concepts (n = 17, fixed before scoring).** The rule was: at least 80 page sentences not among the 300 sampled ones, and a non-empty F_C for both EMBER and CINDER in at least one of seeds 42–46. The concepts are Republic of Ireland, Ancient Rome, Baseball, Harry Potter, Cannabis, Gun, Pornography, Valentine's Day, Heroin, Halloween, Artificial intelligence, COVID-19 pandemic, Tuberculosis, Buddhism, Judaism, Robot and Surfing. Pages average 467 sentences, of which 80–439 (mean 225) were not sampled.

**Token sets.** The page is the revision used for the CS target. New tokens are the page token types not in V\*, plus new neutral token types. The neutral types come from one random sentence per page of a 1000-page Wikipedia sample (`wiki_dump_pages.jsonl`, the notebook's source). Every new token is grouped as C\* (concept text only), B\* (in both) or N\* (neutral only).

Two neutral ratios were run:
- **Balanced** (the notebook's rule): add neutral sentences until #N\* ≥ #C\*. This gives N\*/C\* = 1.01, about 118 neutral sentences.
- **Natural**: one neutral sentence per new page sentence, i.e. EMBER's 1:1 sentence ratio. This gives N\*/C\* = 2.19 (range 1.5–5.5).

On average there are 734 C\* tokens per concept, and the extended set is 1.31×|V\*|. C\* tokens are the only new tokens EMBER's rule could edit, so they are the primary evaluation population.

**Dictionaries.** All are saved fits with Z fixed:
- **EMBER**: L0, F_C under TJ.
- **CS**: L1 (CS + RS-1%), F_C under HJ.
- **CINDER**: L12 (CS + GS), F_C under HJ.

Runs with a non-empty F_C: EMBER 66, CS 80, CINDER 83 (of 85). New rows go through each run's own transform x = (e − μ_B)T. The dense fixed-Z solve reproduces every saved support exactly (support recovery 1.000).

**Scores** (each is the max over f ∈ F_C, with f oriented so its saved support sums positive):
- **proj**: ⟨a_t, z_f⟩/‖z_f‖, the naive projection.
- **cos**: the same score divided by ‖a_t‖.
- **coef**: y_ft of the joint dense solve (ZᵀZ + λI)⁻¹Zᵀa_t, which is what EMBER's fixed-Z sparse re-solve ranks by.
- **share**: y_ft|y_ft| / ‖y_·t‖², the "channelling" score.

**Admission rules:**
- **frac**: the fit's own rule re-applied over V\*_ext. This is RS-1% for EMBER and CS, and GS over V\*_ext (K = 1%, floor 0.2%, cap 3%) for CINDER.
- **width**: the saved absolute width is held.
- **cutoff**: admit if |y_ft| ≥ the smallest in-sample |y| on the saved support.
- **argmax**: admit if f is the token's largest |y|.
- **share50**: admit if share ≥ 0.5.

**Ground truth: blind S/A/U labels.** S is concept-specific, A adjacent, U unrelated (`labels/rubric.txt`).
- **What labellers saw:** only the concept, the token, its source words and a ±8-word context, shuffled within concept.
- **Labellers:** 12 Claude subagents, zero Gemini.
- **Coverage:** every new page token for 7 concepts (the "FULL" set). For the other 10, every token in any run's top 20 by any score, every admitted token, and 50 random tokens.
- **In-sample reference:** 80 random tokens per concept from the union of saved F_C edit sets.
- **N\* tokens** are U by construction.
- **Reliability:** a second labeller on 300 items agreed 0.90 (κ = 0.82; S vs rest κ = 0.85).

**Judge-free proxies on all C\* tokens:**
- Exclusivity: log₁₀(page frequency / Wikipedia frequency), from 42M tokens of a cached Wikipedia shard.
- Lexicon: `concept_words_v1`, matched on source words.

**Statistics.** The concept is the unit (mean over its runs). Paired comparisons use Wilcoxon signed-rank tests with bootstrap 95% CIs over concepts.

## Backed findings (balanced ratio unless stated)

1. **Extrapolated tokens are mostly not concept-specific, under every rule.**
   - Among C\* (random order gives P(S) = 0.065), top-10 precision is 0.18–0.27; counting S or A it is 0.49–0.69.
   - AUC for S over all C\* is only 0.48–0.57, so the feature separates only the top few tokens.
   - The fixed-Z re-solve (frac) admits 15–22 tokens per run at S-precision 0.22–0.25 (S or A: 0.61–0.65). About 37% of admitted tokens are unrelated: U page tokens or neutral tokens.
   - Harry Potter, seed 42: EMBER admits Tolkien (A), Scholastic (A), Beowulf (U) and five neutral tokens (Pradesh, Orioles, Quercus …). CS and CINDER admit hog(wartsia), gry(ffindori), (Dig)gory (S), Tolkien (A) and Beowulf (U).
   - Recall (6 fully labelled concepts): frac recovers 4% of the S tokens among unsampled page tokens. Even the top 100 by coefficient recovers only 35–40%.

2. **Naive projection on the raw dictionary ranks rarity; the joint solve fixes this, and in CS space the two agree.**
   - EMBER, projection → coefficient: P@10 over C\* goes 0.18 → 0.24 (+0.065, CI [+0.001, +0.135], p = 0.08, 8/4 concepts). Over all new tokens it goes 0.14 → 0.22 (+0.083, CI [+0.024, +0.149], p = 0.033). AUC goes 0.48 → 0.55 (n = 6).
   - Raw projection has Spearman 0.70 (CI [0.68, 0.72]) with Wikipedia rarity. The figure is 0.25 for the joint coefficient and 0.14–0.17 for CS scores (CINDER − EMBER = −0.54, p = 1.5e-5).
   - Raw projection fills 38% (balanced) and 57% (natural) of its top 10 with random neutral tokens.
   - In CS space the concept feature is nearly orthogonal to the dictionary: its largest |cos| is 0.28–0.30, against 0.64 raw (p = 1.5e-5). So projection and coefficient agree (Spearman 0.85 vs 0.34), and their P@10 is equal (+0.005, n.s.).

3. **"Channelling" fails.**
   - Code share vs coefficient on P@10: +0.013 for EMBER (n.s.), −0.027 for CS (p = 0.08), −0.007 for CINDER.
   - No score separates concept-specific from adjacent tokens: AUC(S vs A) is 0.40–0.48 for every score and dictionary (n = 6). The A share of the top 10 stays at 0.39–0.43 for CS dictionaries.
   - Dominant-feature (argmax) admits 21–39 tokens per run at precision 0.15–0.18, with a U share of 0.52–0.63 (p ≤ 0.015 vs frac). It also drops 22–29% of the saved support.
   - share ≥ 0.5 admits only 3–5 tokens (precision 0.25–0.37). It would reject 51–61% of the saved support itself.

4. **Width: fraction, fixed width and cut-off give the same precision.**
   - RS-1% over V\*_ext widens features; for Harry Potter the width goes 48 → 59 (n_ext = 5,831, as in the notebook).
   - S-precision of frac, width and cutoff differs by ≤ 0.03 in every dictionary (all n.s.).
   - Holding the absolute width halves admissions but pushes out 13–15% of the in-sample support.
   - The cut-off keeps the whole support and admits 80% of what frac admits.
   - The natural ratio doubles EMBER's neutral admissions (2.6 → 5.3 per run; precision over all admitted tokens 0.22 → 0.18). It leaves the C\* ranking unchanged (|Δ| ≤ 0.006).
   - A stricter cut-off (median in-sample coefficient) only reaches precision 0.27–0.34 (S or A ≈ 0.8), with 3–6 admissions per run.

5. **The frequency hypothesis is supported.**
   - 87% of new page tokens occur once on the page, against 52% of concept-only V\* tokens.
   - In-sample precision rises with page frequency: 0.28–0.30 at one occurrence, 0.59–0.71 at 10 or more.
   - Overall, in-sample support is more precise than admitted new page tokens: EMBER 0.45 vs 0.29, CS 0.39 vs 0.27, CINDER 0.37 vs 0.26 (p ≤ 0.001, n = 17).
   - At one occurrence the difference vanishes: −0.01 / +0.01 / +0.02 (p ≥ 0.5).
   - Weighting in-sample precision by the new tokens' frequency profile predicts their precision within 0.02–0.03 (n.s.).

6. **Dictionaries.**
   - With projection, CS and CINDER beat EMBER on P@10 over all new tokens (+0.09 to +0.11, p = 0.03–0.05). The gain comes mostly from rejecting neutral tokens; on C\* it is +0.07/+0.09, n.s.
   - With the coefficient there is no difference.
   - The 89 lexicon tokens among new C\* rank higher with CS dictionaries: lexicon AUC 0.91 vs 0.67 with projection (p = 3e-5, n = 16) and 0.86 vs 0.75 with the coefficient (p = 0.016).
   - CINDER admits more S tokens per run under frac (6.1 vs 3.9, p = 0.04) at equal precision, because its features are wider (80 vs 52 tokens).

7. **The proxies are not valid ground truth here.**
   - Exclusivity predicts S weakly (AUC 0.58): at one occurrence on the page it reduces to Wikipedia rarity. Raw projection scores AUC 0.82 on it precisely because both measure rarity.
   - The lexicon has only 5.2 hits per concept among C\*.

## Speculations (with tests)

- **S1. The raw concept feature carries a rarity direction**, which might also explain the norm puzzle. *Test:* compute the cosine of z_f with a rare-minus-frequent mean-embedding direction, raw vs CS, and correlate it with the ‖z_f‖ percentile.
- **S2. Refitting on all page tokens would add coverage but not precision for rare tokens** (expected ≈ 0.28). *Test:* refit on V\*_ext, match new features to the saved F_C by support overlap (no Gemini), and label the new tokens by page count.
- **S3. With 30–40 sentences, the unsampled tokens include frequent ones, which should extrapolate better**; a page-count floor (c ≥ 3) may help. *Test:* subsample S_C to 40 sentences, fit, extrapolate, and measure precision by page count.
- **S4. Labeller strictness varies by concept.** This affects levels, not within-concept comparisons. *Test:* double-label whole concepts.

## Paper paragraph

EMBER edits only tokens of its sampled sentences. We therefore tested whether the identified concept features extrapolate to token types that occur only in unsampled sentences of the concept's page. The benchmark covers 17 concepts, about 730 such tokens per concept, three dictionaries and five seeds, scored against blind token labels. Whatever the rule, about one in four extrapolated tokens is concept-specific (top-10 precision 0.18–0.27, against 0.065 at random), and about a third are unrelated. Re-solving the sparse codes with the dictionary fixed recovers only about 4% of the concept-specific unsampled tokens. On the raw-embedding dictionary, naive projection onto the concept feature mostly ranks token rarity (Spearman 0.70 with Wikipedia rarity), so the joint least-squares solve is needed (P@10 0.18 → 0.24). In the CS space the concept feature is nearly orthogonal to the rest of the dictionary (largest |cos| 0.28 vs 0.64), and projection and joint solve agree. The low precision is a property of rare tokens rather than of extrapolation itself: 87% of unsampled page tokens occur once on the page, and at one occurrence in-sample support tokens are no more precise than extrapolated ones (0.28–0.30 vs 0.26–0.31).

## Recommendation

- **Rule.** Rank by the fixed-Z joint solve, never by raw projection. Admit with the in-sample cut-off (|y_ft| ≥ the weakest saved-support coefficient) instead of re-applying the 1% fraction: precision is the same, the result does not depend on how much neutral text was added, and no support tokens are displaced. Do not hold the width fixed, and do not use argmax.
- **Dictionary.** CINDER is the safer basis: robust to the scoring choice, rejects neutral tokens, and ranks lexicon words higher.
- **What to expect.** About 25% S, 40% A and 35% U per extrapolated batch, covering a few percent of the missed specific tokens. Extrapolation is a supplement, not a step the method can depend on.
- **Multilingual pipeline.** With 30–40 sentences, extrapolation mostly adds singletons at about 0.25–0.30 precision. Prefer sampling more sentences, since selection cost is per feature, not per sentence. A page-frequency floor is worth testing (S3).

## Caveats

- The CS target contains every page token, including the new ones, so CS's rejection of neutral tokens is partly built in. The C\*-only metrics are not affected.
- In 16 of 1,832 run×score top-20 lists, 1–2 unlabelled C\* tokens were dropped.
- AUC rests on 6 concepts.
- EMBER contributes fewer runs (66).
- Gun's 300 sampled sentences come from another source (0/300 on the current page).
- Labels are LLM labels, blind to condition, with one labeller per concept.

## Appendix tables

The tables below and their CSV/JSON sources are in `results/paper_analysis/extrapolation/`: `summary.json`, `extra.json`, `runs_metrics.csv`, `admit.csv`. P@k counts S among C\*. "All new" means C\*, B\* and N\* ranked together.

**Ranking of new tokens (balanced).**

| dict | score | P@1 | P@5 | P@10 [CI] | P@20 | P(S+A)@10 | AUC (n=6) | P@10 all new | N\* in top 10 |
|---|---|---|---|---|---|---|---|---|---|
| EMBER | proj | .25 | .19 | .18 [.10,.27] | .15 | .49 | .48 | .14 | .38 |
| EMBER | cos | .26 | .21 | .19 [.10,.28] | .16 | .56 | .48 | .17 | .23 |
| EMBER | coef | .41 | .32 | .24 [.15,.34] | .22 | .63 | .55 | .22 | .20 |
| EMBER | share | .33 | .30 | .26 [.16,.35] | .23 | .64 | .55 | .22 | .23 |
| CS | proj | .41 | .29 | .27 [.16,.38] | .21 | .69 | .53 | .24 | .07 |
| CS | cos | .36 | .28 | .24 [.15,.34] | .20 | .69 | .53 | .21 | .15 |
| CS | coef | .41 | .33 | .27 [.17,.38] | .23 | .67 | .53 | .26 | .10 |
| CS | share | .28 | .29 | .25 [.15,.35] | .23 | .64 | .54 | .21 | .21 |
| CINDER | proj | .38 | .27 | .25 [.15,.36] | .21 | .68 | .56 | .23 | .07 |
| CINDER | cos | .39 | .28 | .23 [.13,.34] | .20 | .67 | .57 | .19 | .16 |
| CINDER | coef | .44 | .31 | .25 [.15,.37] | .23 | .67 | .57 | .24 | .09 |
| CINDER | share | .25 | .29 | .25 [.14,.37] | .23 | .64 | .57 | .22 | .19 |

Random-order baseline: 0.065 among C\* and 0.032 among all new tokens (balanced); 0.023 among all new tokens (natural).

**Admission rules (balanced; per run).**

| dict | rule | admitted | S | A | U page | N\* | prec S | prec S+A | edited C\* | support kept |
|---|---|---|---|---|---|---|---|---|---|---|
| EMBER | frac | 15.3 | 3.9 | 5.5 | 3.3 | 2.6 | .22 | .61 | 12.5 | .99 |
| EMBER | width | 7.1 | 1.9 | 3.0 | 1.3 | 0.8 | .20 | .66 | 6.2 | .87 |
| EMBER | cutoff | 12.4 | 3.1 | 4.5 | 2.9 | 1.8 | .20 | .63 | 10.4 | 1.00 |
| EMBER | argmax | 21.3 | 4.7 | 4.8 | 4.3 | 7.5 | .18 | .48 | 13.7 | .71 |
| EMBER | share50 | 3.3 | 1.5 | 1.5 | 0.2 | 0.2 | .37 | .83 | 3.1 | .40 |
| CS | frac | 15.4 | 4.2 | 6.1 | 3.8 | 1.4 | .25 | .65 | 13.9 | .99 |
| CS | width | 8.8 | 2.5 | 3.6 | 2.1 | 0.7 | .25 | .70 | 8.0 | .86 |
| CS | cutoff | 13.5 | 3.7 | 5.4 | 3.2 | 1.2 | .24 | .65 | 12.1 | 1.00 |
| CS | argmax | 38.5 | 8.4 | 8.9 | 9.7 | 11.4 | .15 | .37 | 26.4 | .78 |
| CS | share50 | 4.7 | 1.7 | 2.1 | 0.5 | 0.4 | .29 | .71 | 4.3 | .49 |
| CINDER | frac | 21.5 | 6.1 | 8.9 | 4.6 | 1.9 | .23 | .63 | 19.3 | 1.00 |
| CINDER | width | 11.8 | 3.6 | 5.4 | 2.0 | 0.8 | .24 | .69 | 10.9 | .85 |
| CINDER | cutoff | 17.6 | 5.0 | 7.8 | 3.3 | 1.4 | .23 | .65 | 15.9 | 1.00 |
| CINDER | argmax | 37.1 | 8.4 | 10.1 | 8.9 | 9.7 | .16 | .42 | 26.9 | .75 |
| CINDER | share50 | 4.5 | 1.6 | 2.1 | 0.4 | 0.4 | .25 | .73 | 4.1 | .39 |

**Natural ratio (per run: admitted / N\* / prec S).**
- frac: EMBER 19.4/5.3/.18, CS 18.4/2.6/.23, CINDER 24.5/3.4/.22.
- cutoff: EMBER 13.6/3.1/.18, CS 14.3/2.0/.23, CINDER 18.3/2.2/.21.
- argmax: EMBER 27.2/13.3/.16, CS 48.2/21.1/.12, CINDER 45.2/17.8/.13.

**Recall of S tokens among C\* (6 FULL concepts).**

| dict | frac | width | cutoff | argmax | share50 | coef top 10 | top 20 | top 50 | top 100 |
|---|---|---|---|---|---|---|---|---|---|
| EMBER | .04 | .00 | .00 | .01 | .00 | .10 | .14 | .30 | .36 |
| CS | .04 | .02 | .02 | .07 | .00 | .07 | .11 | .25 | .35 |
| CINDER | .04 | .02 | .03 | .07 | .00 | .06 | .11 | .24 | .40 |

**S share by page-count bin (concept means).**

| | 1 | 2 | 3–4 | 5–9 | 10+ |
|---|---|---|---|---|---|
| EMBER in-sample support | .30 | .36 | .41 | .41 | .71 |
| CINDER in-sample support | .28 | .35 | .31 | .34 | .59 |
| EMBER new admitted | .31 | .13 | .32 | .33 | — |
| CINDER new admitted | .26 | .16 | .41 | .46 | — |
| All new page tokens | .06 | .06 | .01 | .20 | — |

Number of concepts with new tokens per bin: 17, 11–12, 8–10, 5–6.

The per-concept table (token counts, base rates, P@10 per dictionary) is printed by `tables.py`.

**Settings.** Gemma-2-2b-it, k = 100, λ = 1e-4, η = 0.05, γ = 1. Neutral pool from `random.Random(7)`. Pool depth 20; 50 random tokens per pooled concept; 80 in-sample tokens. CPU only. No Gemini.

## Figures

All in `results/paper_analysis/extrapolation/figures/`, as PDF and PNG with the plotted data as JSON.

- `fig_precision_at_k`: P@k among C\* for the four scores. One panel per score; lines for EMBER, CS and CINDER; bootstrap 95% bands; random order shown dotted.
- `fig_precision_at_k_all` (appendix): the same over all new tokens, balanced ratio on top and natural below.
- `fig_precision_vs_pagefreq`: (a) page-frequency distribution of new vs concept-only V\* tokens; (b) S share by page frequency for in-sample support (solid) and new admitted tokens (dashed), EMBER and CINDER, against all new page tokens (dotted).
- `fig_admission` (appendix): admitted new tokens per run for each rule and dictionary, stacked S / A / U / N\*.

## Sources

**Code** (`experiments/paper_analysis/extrapolation/`): `common.py`, `wikifreq.py`, `tokens.py`, `embrows.py`, `score.py`, `run.py`, `pool.py`, `analyze.py`, `extra.py`, `cutoff_sweep.py`, `tables.py`, `figures.py`.

**Outputs** (`results/paper_analysis/extrapolation/`):
- `tokens/`, `runs/`
- `labels/`: `rubric.txt`, batches, `labels_*.tsv`, `key.json`
- CSVs and JSONs: `summary.json`, `extra.json`, `cutoff_sweep.json`, etc.

**Inputs:**
- Saved fits `results/varwidth/<Slug>/{fixed_refit-False_plain, fixed_refit-False, rn_refit-False}/seed4[2-6]` and their layer caches
- `results/varwidth/prep`
- `experiments/whitening/data/pages`
- `wiki_dump_pages.jsonl`
- `concept_words_v1.json`
