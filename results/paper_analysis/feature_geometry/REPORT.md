<!-- Saved by the lead from the WP agent's hand-back (a hook blocked the agent's own write); text unchanged. -->

# WP2: Feature geometry (orthogonality, cosines, the norm puzzle, concept splits)

## Questions (from the motivation document)

- "We obtain feature directions that are much more orthogonal (close to 90 degrees), where directions with similar concept are closer ... Original: direction isn't orthogonal, cos-similarity isn't interpretable. Contrastive: close to orthogonal, high cos-similarity for closer concepts (magic)."
- "I would expect the magnitude of the HP feature direction in the contrastive to be 'larger' than the original, but it got smaller. Oddly enough, in the original, this feature direction has the largest norm."
- Ancient Rome "may produce 2 concept feature directions, a split. Maybe it's artificial and should have been one."

## Method

- **Data.** Saved fits only: no new fits, no Gemini. 38 concepts × seeds 42–46 × {L0 = EMBER (raw, RS-1%), L1 = CS + RS-1%, L12 = CINDER (CS + GS)}, 570 runs, k = 100.
- **Statistics.** Concept = unit (mean over seeds). Paired Wilcoxon and bootstrap 95% CI (10,000 resamples) over concepts.
- **Sign and metrics.** Directions are sign-aligned by sign(Σ_t y_it); Y rows have 93–99% same-sign mass. Cosines are measured in four metrics:
  - fit: the space the fit ran in;
  - raw: CS directions mapped back, z = z̃ T⁻¹;
  - CS metric: raw directions mapped forward, z T;
  - raw ⊥ μ̂_B: raw space with the background-mean direction projected out of every z_i.
- **Judge-free concept feature.** The feature with the most lexicon hits among its first 20 source words (concept_words_v1), ties broken by ρ, kept if it has ≥ 3 hits. Found in 38 / 37 / 35 concepts for L0 / L1 / L12. As a robustness check I repeated with EMBER's F_C (summary.json) and hand labels P (labels*.json); the conclusions do not change.
- **Semantic relatedness** (independent of Gemma).
  - Primary: MiniLM (all-MiniLM-L6-v2, plain transformers, mean pooling) cosine between the two features' first-20 source-word lists, for all 4,950 pairs per run.
  - Secondary: the saved stage-1 Gemini descriptions. These exist only for features with ρ ≥ 2: on average 21 / 40 / 47 per run.
- **Token frequency.** 32.6M tokens from 30k random articles of the cached Wikipedia shard (20231101.en, file 40/41), Gemma tokenizer, log10 per million with +1 smoothing.
- **Norm decomposition.**
  - For near-disjoint supports, z_i ≈ z*_i = A y_iᵀ, and exactly log‖z*_i‖ = log Σ|y_it| + log n̄_i + log κ_i.
  - n̄ is the |y|-weighted norm of the support's vectors in the fit space. κ = ‖Σ y_t a_t‖ / Σ |y_t| ‖a_t‖ ∈ [0, 1] is the coherence.
  - corr(log‖z‖, log‖z*‖) = 0.95 / 0.91 / 0.83.
  - Mean split (lead's request): ‖z‖² = (z·μ̂_B)² + ‖z⊥‖² in raw space.
- **Splits.**
  - Per run I counted lexicon-matching features (≥ 5 hits = EJ@5, any ρ rank) and CINDER-judge accepted features (HJ = TJ ∨ EJ@5 among the top-20 by ρ).
  - I labelled 56 distinct split pairs blind (configs hidden, order shuffled):
    - SUB: two distinct sub-topics;
    - CUT: the same words divided between two features;
    - OFF: one list is off-topic.

## Backed findings

### F1. Orthogonality depends on the metric, not on CS directions per se.

Mean |cos| over pairs (concept means):

| metric | EMBER | CS + RS-1% | CINDER | EMBER → CINDER |
|---|---|---|---|---|
| fit space (each its own) | 0.164 | 0.052 | 0.051 | −0.114 [−0.117, −0.110], 38/38 lower, p = 7e-12 |
| raw embedding metric | 0.164 | 0.291 | 0.258 | +0.094 [0.089, 0.099], 38/38 higher |
| CS metric | 0.411 | 0.052 | 0.051 | −0.360 |
| raw, μ̂_B projected out | 0.079 | 0.086 | 0.079 | 0.000, p = 0.91 |

- **Fit-space summaries:** mutual coherence 0.82 (EMBER) vs 0.67 / 0.68 (CS); log10 Gram condition number 2.24 vs 1.52 / 1.69.
- **Reference values:**
  - random directions: 0.017;
  - random-support "features" (means of 48 random tokens): 0.69 raw, 0.38 CS;
  - random token pairs: 0.079 raw, 0.029 CS.
- **Each fit is near-orthogonal in its own metric.**
- **In the raw metric, CS directions are *less* orthogonal.** Mapping back multiplies by T⁻¹ ∝ Σ_B^{1/2}, which re-inflates the dominant background direction. Median cos(z, μ̂_B) is −0.53 for mapped-back CS directions and +0.29 for EMBER.
- **With that one direction removed, EMBER and CINDER are equally orthogonal in raw space.** EMBER's raw non-orthogonality is a shared mean component, not entanglement.

### F2. The concept feature's neighbour: an artefact of the mean direction under EMBER, a related topic under CS.

- **Angles (fit space):** nearest-neighbour angle of the concept feature is 50.6° (EMBER), 73.6° (CS + RS-1%) and 74.3° (CINDER). EMBER → CINDER: +22° [19.5, 24.7], 35/35 concepts. Same result with the TJ or hand-label identification.
- **EMBER's neighbour is a rare-token feature.**
  - Its within-run log-frequency z-score is −2.8.
  - It is the recurring "fulda, amien, ferdinando, wojciech, cerambycidae" feature from the shared neutral sentences.
  - Its relatedness to the concept feature is at chance: 59th percentile of MiniLM relatedness.
- **CS neighbours are related topics.**
  - They sit at the 94th (L1) / 92nd (L12) relatedness percentile; 34/37 concepts are higher than EMBER.
  - Examples: magic for HP, empire for Ancient Rome, computing for AI, smoking for Cannabis, religion for Judaism and Buddhism, injection for Heroin, costume for Halloween, art for Louvre.
- **Removing μ̂_B closes the gap.** Projecting μ̂_B out of the raw EMBER directions alone raises EMBER's angle from 50.6° to 74.0° (+23.4° [21.3, 25.5], 38/38), i.e. to the CS level.
- **HP:**
  - EMBER's neighbours are rare-name features at 35–42° (seed 43: Gryffindor and Slytherin together with Fulda, Amiens...).
  - Under CS, magic sits at 68–75° in the fit space and every other feature is beyond 82°. This reproduces the motivation's 68°.

### F3. "Cosines are interpretable" holds for the top of the distribution, not for the global rank correlation.

- **Global Spearman(cos, word-list relatedness)** over all pairs: EMBER 0.33, CS + RS-1% 0.29, CINDER 0.24. EMBER is higher than CINDER in 38/38 concepts.
  - Disjoint-support pairs only: 0.32 / 0.26 / 0.18.
  - Gemini descriptions: 0.24 / 0.22 / 0.15.
- **But EMBER's raw cosine is largely word frequency.**
  - Spearman(cos, pair mean log-frequency) = −0.34; frequent features share sign along μ̂_B.
  - MiniLM also groups lists by word type.
- **Controlling for the pair's frequency** (rank-partial Spearman): CS + RS-1% is higher (0.34 vs 0.25, 37/38 concepts) and CINDER is slightly higher (0.27, 28/38).
- **The decisive difference is at the top.** Among a run's top 0.5% pairs by cos:
  - EMBER: relatedness at the 51st percentile (chance);
  - CS + RS-1% / CINDER: 79th / 75th percentile (+24 points [21.5, 26.2], 38/38).
- **Nearest direction, all features:** relatedness at the 65th percentile (EMBER) vs 83rd / 80th (CS), 38/38.
- **Mapped back to raw space, CS cosines lose this:** global Spearman 0.05 / 0.02; top-0.5% pairs at the 59th / 54th percentile. The interpretability is a property of the CS metric.

### F4. Norm puzzle: EMBER's large concept norm is the background-mean (frequency) component.

- **The observation generalizes.**
  - Percentile of the concept feature's ‖z‖ among the 100 features: EMBER 82 (median 87; ≥ 90th in 14/38 concepts), CS + RS-1% 57, CINDER 61.
  - EMBER → CS + RS-1%: −26.6 points [−35.5, −18.0], 32/37 concepts lower, p = 6e-8.
  - HP under EMBER: rank 3–6 of 100 (95th–98th percentile, share 1.49–1.56%), *not* the largest. Under CS + RS-1%: rank 69–75 (25th–31st percentile, share 0.84–0.85%).
- **Mean split (EMBER, raw space).**
  - The concept feature's ‖z‖² is 45% along μ̂_B; the run median is 12%.
  - 97% (pooled) of its excess ‖z‖² over the run mean is the μ̂_B part.
  - With μ̂_B removed, its percentile falls from 82 to 63 (−19.5 [−24.6, −15.0], 37/38).
  - HP: μ̂_B share 0.67–0.80 (run median 0.12–0.14). The μ̂_B part exceeds the whole excess, and the percentile drops from 95–98 to 51–62.
- **Why concept features carry more μ: μ̂_B is the frequency axis of Gemma's input embeddings.**
  - μ̂_B has cos 0.82 with the top background PC; λ₁ = 349 = 4.5 λ₂ = 160 × the median λ.
  - Each concept's least-squares "rarity direction" has cos 0.88 with μ̂_B and 0.95 with PC1.
  - A token's coordinate β_t = e_t·μ_B/‖μ_B‖² falls from 1.10 (rarest decile) to 0.00 (most frequent decile) over 29.8k tokens (Spearman −0.64).
  - Raw token norms also fall with frequency (Spearman −0.37; 87 vs 80).
  - Concept features are built from rare tokens: support log-frequency z = −1.9 within run.
  - With same-sign y ≈ 1/√w they collect about √w‖μ_B‖, and they are coherent: κ z = +1.43, token norm z = +1.16, Σ|y| z = +0.19.
  - Within EMBER runs, coherence carries 64% of the variance of log‖z*‖.
- **CS removes the term and re-weights by frequency.**
  - Whitening against the type-level Wikipedia background divides PC1 by √λ₁ and centres the data at μ_B.
  - Rare tokens sit near the centre of the type cloud, so CS token norms grow with frequency (Spearman +0.33).
  - The CS concept feature has low token norm (z = −1.0), ordinary coherence (+0.29 / −0.36) and evenly spread weights (Σ|y| z = +0.9).
  - ‖z̃‖ is then mainly a frequency score: standardized β of support log-frequency 0.61, R² 0.34 for L1. For EMBER the raw token norm dominates (β 0.40).
- **Neither norm measures concept strength.** With unit-norm Y rows, ‖z‖ is the scale of the support's vectors in the fit metric.

### F5. Splits are rare, mostly genuine sub-topics, and GS does not merge them.

- **Rates (190 runs each):**
  - ≥ 2 lexicon-matching features: EMBER 4.2%, CS + RS-1% 11.6%, CINDER 7.4%.
  - ≥ 2 HJ-accepted features: 15.3 / 27.9 / 22.6%.
  - EMBER → CS + RS-1%: +0.13 per concept, p = 0.007. CS + RS-1% → CINDER: −0.05, p = 0.22.
- **Blind labels of the 56 distinct lexicon-matching pairs: 52 SUB, 2 CUT, 2 OFF.** Examples: batting vs pitching; research vs doctor/degree; tourism/hospitality vs travel; card games vs betting; sea vs surfing; porn vs sex; devil/ghost vs trick-or-treating.
- **Split-pair geometry:** median angle 63° (EMBER) vs 79° (CS); Jaccard ≈ 0.1.
- **GS keeps the parts apart.** The widest CINDER concept feature of the same seed covers only about 20–35% (median) of each member's support.
- **Ancient Rome.**
  - All 15 runs contain both a Rome/Roman feature and an empire/emperor/monarchy feature.
  - The second feature is about empire in general: empire, imperial, dynasty, colony, king, queen, duke, Qinhuangdao, Menelik.
  - EMBER's judge never accepts the empire feature (0/15); the existing hand label for this group is B.
  - Angles: EMBER 63–71°, CS + RS-1% 77–80°, CINDER 76–91°.
  - GS changes the Roman feature, not the split:
    - under GS the Roman feature grows to 92–109 tokens and absorbs Roman persons (Tiberius, Hadrian, Tacitus, Claudius, Augustus);
    - EMBER's fixed 48 slots instead pad it with other capitals (Milan, Paris, Ottawa, London, Sydney);
    - the empire feature stays separate (27–43 tokens).
  - The split is a real sub-concept (Rome vs empire as a category), not an artifact of the fixed width.

## Speculations (with tests)

- **S1. MiniLM's word-list similarity partly reflects word type** (names vs function words), which inflates EMBER's global Spearman.
  - *Test:* restrict to pairs of content-word features (both with median support log-frequency in the middle tertile), or use a topic-only relatedness such as WordNet or Wikipedia-category overlap.
- **S2. EMBER's "rare proper names" neighbour** (fulda, amien, ferdinando...) is the fit's best single feature for the large-β tail of the neutral sentences.
  - *Test:* refit EMBER on centred embeddings (E − μ_B, WP6's EMBER-c) and check whether this feature and the 35–60° neighbours disappear.
- **S3. CS + RS-1% shows more HJ splits than EMBER because more concept features are visible at all**, not because of more fragmentation.
  - *Test:* condition on runs with ≥ 1 accepted feature and compare lexicon coverage of the union. The conditional rates (0.21 / 0.37 / 0.31) suggest this does not fully explain it.

## Paper paragraph

Contrastive sharpening makes the dictionary near-orthogonal in the space where it is fitted (mean |cos| between directions 0.05 vs 0.16 for EMBER over 38 concepts × 5 seeds), but this is a property of the metric: mapped back to the embedding space, CINDER's directions are less orthogonal than EMBER's (0.26 vs 0.16), and after removing the single background-mean direction μ̂_B both reach the same value (0.08). That direction is the frequency axis of Gemma's input embeddings (a token's coordinate along μ_B falls from 1.1‖μ_B‖² for the rarest to 0 for the most frequent tokens), and it explains two observations on raw fits: concept features, built from rare tokens, have large norms (82nd percentile on average; 97% of the excess is the μ_B component), and their nearest neighbours are unrelated rare-token features at about 51°. In the CS metric the concept feature's nearest neighbour is a related topic at about 74° (e.g., magic for Harry Potter), and the most similar 0.5% of feature pairs are semantically related (79th percentile of an independent sentence-encoder similarity, vs chance for EMBER). Concepts rarely split over two accepted features (7–28% of runs), and when they do the parts are distinct sub-topics, such as Rome and empire for Ancient Rome, which global sparsity keeps separate.

## Appendix text

### Definitions

- **Directions.** z_i are the columns of Z (d = 2304, k = 100), sign-aligned by sign(Σ_t y_it).
- **Spaces.**
  - fit: raw for EMBER; X = (E − μ_B) W S_η^{1/2} for CS;
  - raw: z = z̃ T⁻¹, with T⁻¹ = S_η^{−1/2} W⁻¹;
  - CS metric: z T;
  - raw ⊥ μ̂_B: z − (z·μ̂_B) μ̂_B.
- **Orthogonality summaries.** Mutual coherence = max_{i≠j} |cos|. Condition number = that of the Gram matrix of unit directions.
- **Concept feature.** argmax over features of (lexicon hits in the first 20 normalized source words, then ρ), requiring ≥ 3 hits.
- **Nearest neighbour.** argmax_j |cos(z_c, z_j)|; angle = arccos of that value.
- **Relatedness.** MiniLM cosine of ", ".join(words).
- **Lift.** The within-run percentile of relatedness for the pairs in a within-run cos-percentile bin.
- **Rank-partial Spearman.** Both ranks are regressed on the ranks of the pair's mean and absolute difference of support log-frequency.
- **Norm percentile.** The share of the other 99 features with smaller ‖z‖. The mean split is computed on ‖z_raw‖². Decomposition terms are z-scored within run.
- **Rarity direction.** r ∝ Σ_t (mean logf − logf_t)(e_t − ē) over V*.
- **Splits.** EJ@5 = ≥ 5 lexicon hits. HJ = TJ ∨ EJ@5 within the top-20 by ρ (layer_cache).

### Settings

- Fits as saved: `fixed_refit-False_plain`, `fixed_refit-False`, `rn_refit-False`.
- CS: η = 0.05, γ = 1, wiki background (107,472 types, eps_rel = 1e-4), page target.
- Frequencies: 30,000 articles, 32.6M tokens.
- Bootstrap: 10,000 resamples over concepts.

### Additional numbers

- **Orthogonality** (per run in `orth_runs.csv`, ECDF data in `fig_cos_ecdf_data.csv`):
  - share of pairs with angle < 60°: 1.6% EMBER vs 0.1% CS in the fit space; 11% / 10% for CS in the raw metric;
  - background-whitening-only metric W: EMBER 0.76 vs CS 0.02.
- **Neighbour angles by identification** (lex / TJ / P): EMBER 50.6 / 50.3 / 51.8°; CINDER 74.3 / 73.4 / 73.7°.
- **Norm percentile by identification** (lex / TJ / P): EMBER 82 / 82 / 81; CINDER 61 / 58 / 58.
- **Concept feature at run-level ‖z‖ rank ≤ 10:** EMBER 38% of runs, CS + RS-1% 17%, CINDER 13%.
- **Regression of log‖z‖** (within-run standardized; naive SEs ≈ 0.007, and features within a run are not independent):

  | | β(raw token norm) | β(log-freq) | β(log ρ) | β(log width) | R² |
  |---|---|---|---|---|---|
  | EMBER | 0.40 | 0.13 | −0.20 | – | 0.22 |
  | CS + RS-1% | 0.06 | 0.61 | 0.08 | – | 0.34 |
  | CINDER | −0.01 | 0.56 | −0.13 | 0.21 | 0.37 |

- **Mechanism terms:** the three terms explain R² 0.90 / 0.85 / 0.70 of log‖z‖.
- **Split rates given ≥ 1 accepted feature (HJ):** 0.21 / 0.37 / 0.31.
- Ancient Rome per seed: `ancient_rome.md`. Per-concept norm percentiles: `fig_norm_percentile_data.csv`.

## Figures

- **`fig_cos_ecdf.pdf`** (6.8 in): ECDF of pairwise |cos| (seed 42, 38 concepts pooled) in (a) the space each fit ran in, (b) the raw metric, (c) the CS metric. Dotted line: random directions in d = 2304. Each method is near-orthogonal only in its own metric.
- **`fig_cos_vs_semantic.pdf`** (6.8 in): fit-space cos vs MiniLM relatedness of the two features' source words (seed 42, 188,100 pairs per panel). Hexbin density, with median and IQR per relatedness bin.
- **`fig_semantic_lift.pdf`** (3.3 in): relatedness percentile by within-run cos percentile. Solid: fit space; dotted: CS directions mapped to raw; bands: 95% CI over 38 concepts. Only CS's top pairs are semantic.
- **`fig_norm_percentile.pdf`** (3.3 in): percentile of the concept feature's ‖z‖ per concept (mean over seeds), EMBER vs CS + RS-1% vs CINDER.
- **`fig_norm_terms.pdf`** (3.3 in): the concept feature's within-run z-score of Σ|y|, token norm, coherence κ and ‖z‖ (mean over concepts, 95% CI).

Every figure has its data in a `*_data.csv` next to it.

## Sources

**Code** (`experiments/paper_analysis/feature_geometry/`):

| script | what it does | outputs |
|---|---|---|
| `token_freq.py` | Wikipedia token frequencies | `wiki_token_counts.npz` |
| `collect.py` | per-feature table, source words, pairwise cosines, token norms | `features.parquet`, `words.json`, `pairs/`, `tokens/` |
| `embed_text.py` | MiniLM embeddings | `text_emb.npz` |
| `orth.py` | F1, F2 | `orth_*.csv/json`, `concept_nn.csv` |
| `rarity.py` | mechanism for F2, F4 | `rarity_*` |
| `semantic.py`, `semantic_extra.py` | F3 | `semantic_*` |
| `norms.py` | F4; mean split under `mu_split` | `norm_*`, `norm_stats.json` |
| `splits.py` | F5 | `splits_*`, `split_pairs*.csv`, `split_items_blind.csv`, `split_items_key.csv`, `split_labels.json`, `ancient_rome.md` |

**Inputs:**
- `results/varwidth/<Slug>/<tag>/seed<N>/{fit.pt, summary.json, judge.json, layer_cache.json}`
- `results/varwidth/prep/*_S300.pt`
- `experiments/whitening/data/gemma/bg_wiki_stats.pt`
- `experiments/varwidth/j3_data/{concept_words_v1.json, labels*.json}`

Gemini spend: 0. GPUs: none.

**Caveats:**
- Frequencies come from one Wikipedia shard (recent article ids).
- The blind split labels are mine (a single rater).
- Lexicon identification finds no concept feature (no feature with ≥ 3 hits) for The Matrix under CS + RS-1% and CINDER, and for French Revolution and Nazism under CINDER.
- EMBER's raw HP feature is 3rd–6th largest by ‖z‖, not the largest as the motivation states.
