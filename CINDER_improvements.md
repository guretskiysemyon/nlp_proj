# CINDER: sharper, adaptive and verified concept features for embedding erasure

Oct 18, 2026 · @Ori

## Summary

EMBER (Suslik, Shafran and Geva) erases a concept by finding its directions in the embedding matrix and subtracting them. How well this works depends on whether the concept gets a feature of its own, and on whether the method can recognize that feature. We found that each of the three feature-finding steps of EMBER has a failure mode of its own, and that each one can be traced to a specific modeling choice. CINDER (Contrastive Informed Directional Embedding Removal) replaces each of these choices with a principled alternative and leaves the edit unchanged:

**sharpen the matrix, budget sparsity globally, verify both ways.**

| Layer | Step | What fails in EMBER | Our change |
| --- | --- | --- | --- |
| L1 | Matrix | The concept is a low-variance direction, so the factorization spends its features on generic high-variance topics (magic, writing, books). | **Contrastive Sharpening (CS):** whiten against background text, then sharpen toward the concept page. |
| L2 | Factorization | Every feature keeps exactly 1% of the tokens, whatever the concept's breadth. | **Global Sparsity (GS):** one nonzero budget for the whole matrix instead of one per feature. |
| L3 | Selection | The judge sees only a one-line summary, so it rejects correct features (Valentine's Day, WWII, Heroin). | **Two-Way Judge (TWJ):** add a top-down check that expands the concept into a lexicon and matches it to the feature's words. |

The layers build on each other, and this is part of the claim. GS does not help on the raw matrix, and it starts helping once CS makes the features clean. TWJ is evaluated on the features that CS+GS produce.

### Results at a glance

All numbers are from Gemma-2-2b-it, k = 100, 5 seeds per configuration unless noted. They measure the **features**. Erasure after the edit has not been measured for CINDER yet (see "Status and path to a paper").

| Claim | Evidence |
| --- | --- |
| CS turns a buried concept into the top feature. | Harry Potter feature: ρ rank 11 → **1** (ρ 3.9 → 125, 2.4× the runner-up). |
| CS makes the feature reproducible. | Ancient Rome, 20 seeds: a concept feature in 17 → **20** seeds (exactly one each), token Jaccard across seeds 0.26 → **0.69**, neutral tokens per feature 13.8 → **1.7**. |
| CS removes most of the neutral tokens. | 3 concepts × 5 seeds, EMBER's sparsity rule: pure seeds (≤ 5 neutral tokens) 4/15 → **15/15**, neutral share 15–25% → 1–9%. |
| GS lets width follow the concept without losing purity. | 18 concepts × 5 seeds, with CS: concept-feature width ranges over **15–113** tokens (EMBER: 43–52), with purity equal to EMBER's (76/90 vs 75/90 pure seeds). Doubling EMBER's fixed budget instead drops purity to 31/90. |
| GS needs CS. | On the raw matrix, GS widens features into neutral tokens (neutral share 11–44% vs 15–25% for EMBER's rule); with CS it does not (1–9%). |
| TWJ recovers concept features that the judge rejects, at almost no cost in precision. | 19 concepts × 5 seeds, hand-labeled: runs where a concept-related feature is accepted go from 61 → **82** of 91, precision 1.00 → 0.99 (feature on-topic or broader). |

**Contributions.**

1. **A diagnosis.** EMBER's feature-finding misses concepts for three separate reasons: variance (the matrix), allocation (the sparsity rule) and verification (the judge). Each reason has a measurable symptom.
2. **Three changes, one per reason, each grounded in a known technique.** Contrastive whitening (generalized eigenproblem / contrastive PCA), global top-K projection (iterative hard thresholding, BatchTopK), and two-sided verification (query expansion in retrieval).
3. **Evidence that the changes depend on each other.** GS helps only on the CS matrix, and TWJ fixes rejections that remain after CS+GS. This is why they have to be evaluated as layers, not as independent tricks.
4. **A path to multilingual erasure.** Each layer addresses a specific risk of fitting the matrix across languages (language-identity variance, many surface forms of one concept, sub-word tokens in other scripts).

### Why it matters

In our Part 1 reproduction on 18 concepts, EMBER erases only about half of the concept knowledge on average (efficacy 0.52), and the edit leaks across languages: erase in English, and the model still answers in other languages (mean non-English gain +0.24, Russian +0.41). Both numbers are provisional while the MC scoring repair runs (docs/progress.md). EMBER's edit can only remove what its features capture. Better features are therefore the most direct lever on both problems, and CINDER changes exactly that part of the pipeline.

## L1. Contrastive Sharpening (CS)

CS changes the matrix the factorization sees, so that the concept is a strong direction instead of a weak one.

**Idea.** Sparse MF minimizes squared error, so it spends its k features on the directions with the most variance. In E\_V\*, those are generic topics shared by many tokens: magic, writing, books, criticism. A concept like Harry Potter has little variance. It has few tokens, and those tokens were seen less in training, so their representations are poorer. CS first removes the variance that ordinary background text already explains, then amplifies what makes the concept page different. The concept then stands out in the geometry itself, and the factorization does not have to force it out by inflating Y entries.

**Steps.**

1. **Background statistics.** Compute the mean μ\_B and covariance Σ\_B of background token embeddings (Wikipedia text).
2. **Whiten.** Map every row e of E\_V\* to ẽ = W(e − μ\_B), with W = (Σ\_B + ηI)^(−1/2). Every background direction now has unit variance.
3. **Sharpen.** Compute the covariance Σ̃\_C of the concept-page tokens in whitened space, and map ê = (Σ̃\_C + ηI)^(γ/2) ẽ. Directions where the concept page varies more than the background are stretched.
4. **Factorize** the transformed matrix with sparse MF, exactly as in EMBER.
5. **Edit.** Map the selected feature directions back to the original space before subtracting them.

Defaults: η = 0.05 (relative ridge), γ = 1.

**Theory.** After whitening, the variance of a unit direction v in the concept tokens equals the concept-to-background variance ratio:

```latex
\frac{v^\top \Sigma_C \, v}{v^\top \Sigma_B \, v}
```

Maximizing it is the generalized eigenproblem Σ\_C v = λ Σ\_B v, the same objective as Fisher's discriminant and contrastive PCA. Sharpening with γ > 0 gives more weight to directions where this ratio is large. Whitening alone only takes away the background's advantage; sharpening is what puts the concept on top.

**Literature.**

- Contrastive PCA (Abid et al., 2018) finds directions enriched in a target dataset relative to a background dataset. CS uses the same contrast as a preprocessing step, not as the decomposition itself.
- Whitening is standard preprocessing before ICA and sparse coding (Hyvärinen and Oja, 2000).
- Word and sentence embeddings are dominated by a few common directions (Ethayarajh, 2019). Removing or whitening them improves representations (Mu and Viswanath, 2018; Su et al., 2021).
- LEACE (Belrose et al., 2023) whitens to make a concept removable by a linear projection.

**Results so far.**

| Evidence | EMBER | +CS |
| --- | --- | --- |
| Harry Potter feature, rank by ρ (ρ value) | 11th (3.9) | **1st** (125; 2.4× the 2nd, magic) |
| HP tokens among ranks 20–30 of that feature | 2 of 11 | **11 of 11** |
| SVD rank that matches the sparse MF error | 5 | 49 |
| Ancient Rome, 20 seeds: seeds with a concept feature | 17, two features in 4 of them | **20, exactly one each** |
| Ancient Rome: concept share of the feature | 0.72 ± 0.15 | **0.92 ± 0.04** |
| Ancient Rome: neutral tokens per feature | 13.8 | **1.7** |
| Ancient Rome: token Jaccard across seeds | 0.26 (no token in every seed) | **0.69** (22 tokens in every seed) |
| AR, HP, Valentine's Day × 5 seeds, EMBER's 1% rule: pure seeds | 4/15 | **15/15** |
| Same runs: neutral share of the concept feature | 15–25% | **1–9%** |
| Same runs: token Jaccard across seeds | 0.36–0.63 | **0.75–0.89** |

What the numbers show:

- **The concept goes from buried to dominant.** On Harry Potter, the concept feature moves from 11th to 1st by ρ, with a margin of 2.4× over the next feature. The matrix is also much richer: a rank-5 SVD matches EMBER's sparse MF error, and a rank-49 SVD is needed for CS's matrix. So the factorization now has many more distinct directions to choose among.
- **Sharpening is the decisive step.** Whitening alone found no Harry Potter feature, and adding sharpening recovered it.
- **The tokens are right.** Without CS, the HP feature's tail holds Nestlé, Lego, Trump and taxonomic fragments. With CS, it holds Gringotts, Hogsmeade, Quidditch, McGonagall and Hagrid.
- **The geometry is interpretable.** CS features are close to orthogonal: HP vs magic is 68°, and every other neighbour is above 82°.
- **The result is stable across seeds.** On Ancient Rome, EMBER's output depends on the seed (0, 1 or 2 concept features). With CS, every seed finds one feature, with a shared core of 22 tokens.

**Caveats.** The Ancient Rome sweep used random per-feature budgets for both matrices, so it compares stability, not width. The full comparison on all 18 concepts without CS (the first rung of the ladder) is running. With Llama (d = 4096 vs 2304 for Gemma), sharpening needs enough concept-page tokens for Σ̃\_C to be well conditioned, so η must grow or γ must shrink.

## L2. Global Sparsity (GS)

GS keeps EMBER's total sparsity of 1% but lets each feature choose its own width. A concept feature takes as many tokens as the data supports, instead of a fixed 1% (about 48 tokens).

**Idea.** EMBER's winner-takes-all step keeps the top 1% of tokens in every feature. We call this **Row Sparsity (RS)**. It gives every feature the same width, but concepts differ in breadth. Ancient Rome and Culture of Greece may need more than 1% of the tokens, and Baseball fewer. A broad concept that is cut short leaves part of the concept unedited, and a narrow one is padded with unrelated tokens that the edit then damages. GS replaces the per-feature budget with one budget for the whole matrix, so the width becomes an output of the fit, not a hyperparameter. Because the support is re-chosen at every iteration, the width also shapes the feature directions Z, not only the edit set.

**Steps** (inside the ALS loop, replacing the per-row WTA step):

1. Update Y with Z fixed (closed-form least squares, as in EMBER).
2. Score every entry by its share of its own row: s\_ij = |y\_ij| / ‖y\_i‖.
3. Keep the global top-K entries, with K = 1% of all entries of Y (the same total as EMBER).
4. Enforce a floor of 0.2% and a cap of 3% of the tokens per feature.
5. Rescale the rows and update Z, as in EMBER. Repeat until convergence.

**Theory.** EMBER constrains each row of Y separately, and GS constrains the matrix as a whole, with the same total:

```latex
\text{RS: } \|y_i\|_0 \le s\,n \;\; \forall i \qquad\qquad \text{GS: } \|Y\|_0 \le s\,k\,n
```

Keeping the global top-K entries is the exact projection onto the GS constraint, the same step as iterative hard thresholding. The scores must be comparable across rows, otherwise high-energy generic features take the whole budget. Dividing by the row norm makes the score scale-free, so every feature is asked the same question: is this token a meaningful share of the feature? The floor and cap keep a feature from dying or from absorbing the budget.

**Variants** (all with the same total budget unless noted):

| Name | Rule | Role |
| --- | --- | --- |
| RS-1% | Top 1% of tokens per feature | EMBER's rule |
| RS-2% | Top 2% per feature, twice the total | Control: more tokens for every feature, without reallocation |
| **GS-norm** | Global top-K on \|y\_ij\| / ‖y\_i‖ | Our method |
| GS-gain | Global top-K on \|y\_ij\| · ‖z\_i‖, the reconstruction gain of the entry | Ablation; also feeds generic high-energy features |
| GS-entropy | Each feature's width set by the effective support size exp(H) of its coefficients | Ablation |

**Literature.**

- BatchTopK SAEs (Bussmann et al., 2024) replace per-example top-k with a top-k across the batch, so the number of active latents adapts per example. GS makes the same move across features instead of examples.
- Top-k sparsity in autoencoders (Makhzani and Frey, 2014; Gao et al., 2025), and hard thresholding onto an L0 ball (Blumensath and Davies, 2009).
- L1-penalized coding (the Lasso; Tibshirani, 1996) also gives a variable support, but it is controlled by a penalty λ instead of a budget. We tried an L1 ALS variant: with CS, its Ancient Rome feature was never pure and the judge kept it in none of 5 seeds.

**Results so far.** All five rules on the CS matrix, 18 concepts × 5 seeds (90 runs per rule), judge-free metrics on the tracked concept feature. This sweep used a 5% cap; the current default is 3%.

| Metric | RS-1% (EMBER) | RS-2% | **GS-norm** | GS-gain | GS-entropy |
| --- | --- | --- | --- | --- | --- |
| Concept width, range over concepts (tokens) | 43–52 | 86–104 | **15–113** | 9–143 | 25–121 |
| Width change across seeds (mean, tokens) | ±0 | ±0 | **±4.1** | ±8.1 | ±7.9 |
| Pure seeds (≤ 5 neutral tokens) | 75/90 | 31/90 | **76/90** | 55/90 | 61/90 |
| Mean neutral share | 5.8% | 8.2% | **5.6%** | 8.0% | 8.2% |
| Mean token Jaccard across seeds | 0.719 | 0.726 | **0.735** | 0.717 | 0.733 |
| Mean worst-pair direction similarity across seeds | 0.838 | **0.872** | 0.866 | 0.817 | 0.808 |
| Seeds where the judge kept the concept | 66/90 | 69/90 | 63/90 | 63/90 | 67/90 |

What the numbers show:

- **GS-norm adapts width without paying for it.** It is the only rule that changes the width per concept (Baseball 15, Valentine's Day 51, Harry Potter 63, Ancient Rome 103, Culture of Greece 113 tokens) while keeping EMBER's purity and stability. The width is stable for a given concept (±4 tokens across seeds), so it reflects the concept, not the seed.
- **Simply giving more tokens does not work.** RS-2% widens every feature and drops the pure seeds from 75 to 31. GS-gain and GS-entropy also widen, but let neutral tokens in. Of the reallocation rules we tried, only the scale-free score keeps the features clean.
- **The data sets the concept width, not the guardrail.** With the 3% cap (116 runs over 24 concepts), no concept feature reached the cap (0 of 95; widest: Culture of Greece, 120 of 156). Generic features do reach it in most runs, which is what the cap is for.
- **GS depends on CS.** On the raw matrix (Ancient Rome, Harry Potter, Valentine's Day), GS-norm widens the concept feature into neutral tokens: neutral share 37%, 44% and 11%, against 19%, 25% and 15% for RS-1%. With CS, the same rule gives 2%, 1% and 9%, and 15 of 15 seeds are pure (4 of 15 on the raw matrix, for either rule). Reallocation only helps once the features it reallocates to are clean.
- **The judge is now the bottleneck.** The "judge kept" row is about the same for all rules (63–69 of 90), and the per-concept differences are mostly judge decisions on features that are already clean. For example, under GS-norm the Nazism feature is pure in 5 of 5 seeds and stable (worst-pair similarity 0.99), and the judge rejects it in all 5. Valentine's Day goes the other way (judge kept 2 → 5 of 5). This is what L3 addresses.

**Caveats.** The width tells us that the feature changed, not that the erasure improved. Whether a wider Ancient Rome feature erases more, and a narrower Baseball feature damages less, is measured only after the edit, which has not been run yet. The cross-ablation of GS without CS covers 3 concepts so far.

## L3. Two-Way Judge (TWJ)

TWJ accepts a feature if either of two independent checks accepts it: EMBER's judge, or a lexicon match that needs no LLM call per feature. It recovers concept features that EMBER's judge rejects.

**Idea.** EMBER's judge works bottom-up. It summarizes a feature's tokens in one sentence, then asks whether that summary is the concept, and anything the summary drops is lost. In our runs, "heroin" became "addictive drugs" and WWII became "20th-century totalitarian regimes", and both were rejected. TWJ adds a top-down path: expand the concept once into a lexicon of words specific to it, then count how many of the feature's words appear in it. The two paths look at one question from opposite sides: the feature is abstracted up to the concept, and the concept is expanded down to the feature.

**Steps.**

1. **Token Judge (TJ),** EMBER's judge, unchanged: stage 1 describes the top-20 tokens without the concept name, stage 2 decides membership from that description (accept at confidence ≥ 0.85).
2. **Concept lexicon,** once per concept: one LLM call lists 20 words you would expect in the concept's Wikipedia page, specific to the concept rather than its broader category.
3. **Feature words:** map each of the feature's top tokens (in |y| order) to its most frequent source word in S\_C ∪ S\_N ("ffindor" → Gryffindor), normalize, remove duplicates and keep the first 20.
4. **Expansion Judge (EJ@t):** accept if at least t lexicon words match a feature word exactly.
5. **TWJ = TJ ∨ EJ@5.**

**Theory.** The two paths fail in different ways. The summary path fails when the summary abstracts the concept away. The lexicon path fails when the feature is about the concept but uses words the lexicon missed, or when a broader feature shares many words with it. An OR of two tests with different blind spots raises recall, and the threshold t controls precision.

**Variants.**

| Name | Rule | Role |
| --- | --- | --- |
| TJ | EMBER's two-stage judge | Baseline |
| WJ | TJ, with each token shown as its source words | Word Judge; tests whether fragments like "ffindor" mislead the judge |
| WJ+share | WJ, plus a concept share of at least 0.5 | Stricter word judge |
| EJ@t | Lexicon match, at least t words | One check alone |
| **TWJ** | TJ ∨ EJ@5 | Our judge |
| TWJ-R | WJ+share ∨ EJ@4 | High-recall variant |

**Literature.**

- Automated feature interpretation with LLMs explains a feature, then scores the explanation (Bills et al., 2023; Paulo et al., 2024). TJ follows this pattern.
- Query expansion in retrieval expands the query before matching it to documents (query2doc, Wang et al., 2023; HyDE, Gao et al., 2023). EJ does the same for the concept.
- Verbalized LLM confidence is poorly calibrated (Xiong et al., 2024). This is why EMBER's 0.85 cutoff is fragile: all three wrong rejections below sat exactly at 0.85.

**Results so far.**

- **TJ's failures.** TJ wrongly rejected Valentine's Day, Heroin and WWII, each at confidence 0.85. It correctly rejected the Uranium candidate, a feature of inorganic compounds with no uranium token.
- **Labeled evaluation.** Every ρ candidate from the CS+GS-norm runs (23 concepts × 5 seeds, 7,105 candidates) was scored by every rule. The features any rule accepts were hand-labeled, once per recurring feature and without seeing the judges' decisions: **P** (specific to the concept), **B** (borderline: a broader category or an adjacent topic, e.g. generic nuclear physics for Uranium) or **N** (not the concept). The table excludes the 4 concepts used as examples in the lexicon prompt (Culture of Greece, Ancient Rome, Artificial intelligence, World War II), which leaves 19 concepts and 95 runs.

| Rule | Precision (P) | Recall (P) | Runs hit (P), of 72 | Precision (P+B) | Recall (P+B) | Runs hit (P+B), of 91 |
| --- | --- | --- | --- | --- | --- | --- |
| TJ (EMBER) | **0.85** | 0.69 | 60 | **1.00** | 0.41 | 61 |
| WJ | 0.56 | 0.81 | 61 | 0.77 | 0.55 | 66 |
| EJ@5 alone | 0.74 | 0.64 | 64 | 0.99 | 0.43 | 81 |
| **TWJ** (TJ ∨ EJ@5) | 0.71 | 0.77 | 67 | 0.99 | 0.54 | **82** |
| TWJ-R (WJ+share ∨ EJ@4) | 0.61 | **0.90** | **71** | 0.91 | **0.68** | **84** |

"Runs hit" counts runs where at least one accepted feature is a positive. Recall is relative to the positives that some rule accepted, since unlabeled candidates count as negatives.

What the numbers show:

- **TWJ finds the concept in many more runs.** Runs with an accepted on-topic or broader feature go from 61 to 82 of 91 (+21), and precision stays at 0.99: only 1 of 110 TWJ accepts is unrelated to the concept.
- **The lexicon check is strong on its own.** EJ@5 alone reaches 81 runs at precision 0.99, with no LLM call per feature, and it is deterministic.
- **Most of the added features are broader than the concept.** Under the strict labels (B counts as negative), TWJ adds 7 runs (60 → 67) and its precision drops from 0.85 to 0.71. So the lexicon often accepts a feature about the concept's wider area. That is a correct feature for some uses and too broad for others, and the threshold t is the knob between the two.
- **Showing words to the judge is not enough.** WJ raises recall but halves the strict precision. The lexicon match is what adds recall without adding unrelated features.

**Caveats.** The threshold t = 5 was chosen on this labeled set, and the labels are ours. The 15 concepts added after the prompt was fixed (Judaism, Karate, ..., Tourism) are the real held-out test: their fits are done, and their judging and labeling are next. TWJ's broader accepts will cost specificity in the edit, which only the erasure evaluation will show.

## Extension: multilingual erasure

A concept should be erased in every language the model speaks, not only in English. Our Part 1 results show that EMBER's English edit leaks to other languages (mean non-English gain +0.24, provisional), so this is a real gap. A naive multilingual EMBER already worked on Ancient Rome with Llama. The three layers are designed to carry that result to more concepts.

**Naive extension (ML-EMBER).** Build E\_V\* from sentences in 8 languages instead of one. To keep the matrix small enough to factorize, sample 40 sentences from the concept's Wikipedia page in each language (320 in total, close to EMBER's 300). Everything else is EMBER unchanged: factorize, filter by ρ, judge and edit.

**First result.** On Ancient Rome with Llama, ML-EMBER succeeded as a proof of concept. \[Add the metric: MC accuracy after the edit, per language.\] It has not yet been tested on other concepts.

**What can go wrong, and which layer addresses it.**

| Risk in ML-EMBER | Layer | How it helps |
| --- | --- | --- |
| Language identity is a large, high-variance direction. The factorization can spend features on "French tokens" or "Cyrillic tokens" instead of on concepts. | CS | Whitening against a background drawn from the same 8 languages removes the variance that language identity already explains, so the concept, shared across languages, is what remains. First check: against an English-only background, the top contrastive directions are es/pt/de/it language clusters (enrichment 6.82); against the 8-language background they mostly disappear (2.82). |
| Llama's larger dimension (4096) leaves too few concept-page tokens per language for a well-conditioned Σ̃\_C. | CS | Pooling concept-page tokens across the 8 languages gives the sharpening step enough tokens, so η does not have to grow. |
| One concept now has many surface forms (Rome, Roma, Rom, Рим). At a fixed 1% width, the concept feature is cut short or split by language or script. | GS | The feature can widen to cover every language's forms, with the total budget unchanged. |
| With 40 sentences per language, frequent concept tokens can be missed by the sample, and only sampled tokens are edited. | Sparse extrapolation | Fix Z and solve a sparse Y' for tokens outside the sample, so they can be edited too. |
| Sub-word tokens in other scripts are hard for the judge to read, and the lexicon is in English. | TWJ | WJ shows each token with its source words and their language. EJ needs a lexicon per language, or a matcher that counts translations. |

**Sparse extrapolation so far (single language).** On English Harry Potter, extending the factorization to unsampled tokens added a few correct tokens but also many indirect or unrelated ones. Adding as many neutral tokens as concept tokens improved it, but not enough yet. One likely reason: unsampled tokens are by definition the rarer words on the page, which are noisier (Tolkien, Narnia). Multilingual erasure depends on this step more than on any other, so it needs work before scaling.

**Literature.**

- Multilingual models encode language identity in a few dominant directions, separate from meaning (Chang et al., 2022). Removing each language's mean makes representations more language-neutral (Libovický et al., 2020). CS's background whitening generalizes this centering.

**Plan.**

1. Run ML-EMBER on more concepts, to see whether the Ancient Rome result holds.
2. Run ML-CINDER (ML-EMBER + CS + GS + TWJ) on the same concepts and compare, per language.

To confirm: the 8 languages, whether the neutral sentences are also multilingual, and the metric behind the Ancient Rome result.

## Status and path to a paper

**Where we stand.** Each layer has a clear mechanism, a link to established methods, and feature-level evidence on 18–23 concepts. The effects are large where the mechanism predicts them: CS takes the Harry Potter feature from 11th to 1st and removes most neutral tokens; GS widens or narrows features by up to 7× across concepts at no cost in purity; TWJ finds the concept in about a third more runs at the same precision. We also find that the changes interact (GS needs CS), which argues for presenting CINDER as one pipeline with a layered evaluation rather than three separate tricks.

**What a paper needs.** All of the evidence so far is about the features. The central claim of an erasure paper is about the edit, so the deciding experiment is still ahead:

1. **Erasure on all 18 concepts**, with EMBER's own protocol (efficacy, specificity, coherence) and the fixed MC scoring, for the cumulative ladder EMBER → +CS → +CS+GS → +CS+GS+TWJ.
2. **The two cross-ablations** on all 18 concepts: GS without CS (fits running), and TWJ without GS.
3. **The held-out judge test** on the 15 new concepts.
4. **Cross-lingual leakage** of the CINDER edit, using the Part 1 setup (erase in English, ask in de/es/fr/ru).

**Potential.** If the feature gains carry over to erasure, CINDER is a solid paper: a clear diagnosis of a recent method, three principled and cheap fixes, and a layered ablation that shows why each one is needed. The judge part (TWJ) is useful beyond erasure, for any pipeline that labels features with an LLM. The multilingual extension would make the contribution larger, since cross-lingual leakage is a known weakness of EMBER and CS is designed for exactly that setting. The main risk is that cleaner features do not translate into better erasure, or that TWJ's broader accepts cost specificity; the ladder above measures both.

## Names: experiments → paper

| In the experiments | In the paper |
| --- | --- |
| Contrastive PCA, whitened, CW (background whitening + concept sharpening) | Contrastive Sharpening (CS) |
| plain, non-whitened | EMBER's matrix (no CS) |
| fixed 1% / 2% per feature | Row Sparsity: RS-1% / RS-2% |
| varnnz, pooled, shared budget | Global Sparsity (GS) |
| rn, RNP | GS-norm |
| m1, CP, pooled\_contrib | GS-gain |
| m3, EW | GS-entropy |
| m4, L1 ALS | Lasso variant (dropped) |
| J\_tokens, J\_tok | Token Judge (TJ) |
| J\_words, CJ / W (CJ+S) | Word Judge (WJ) / WJ+share |
| J3, overlap judge, LO@t | Expansion Judge (EJ@t) |
| HJ = J\_tok ∨ J3@5 / HJ-R = W ∨ J3@4 | Two-Way Judge (TWJ) / TWJ-R |

## Open items

- [ ] Erasure results (efficacy, specificity, coherence) for the ladder on all 18 concepts.
- [ ] Finish the no-CS fits on all 18 concepts, and rerun the GS comparison at the 3% cap.
- [ ] Judge and label the 15 held-out concepts; report TWJ on them.
- [ ] Choose t for EJ on a split that is not the evaluation set (or with a cross-concept null: match each feature against another concept's lexicon).
- [ ] Confirm the CS transform against the code: the background set, whether Σ̃\_C is computed in whitened space, and the exact role of γ and η.
- [ ] Confirm how the CS edit is done: are the feature directions mapped back to the original embedding space before subtraction?
- [ ] Confirm the GS-norm score: is ‖y\_i‖ the norm of the dense row or of the kept entries?
- [ ] Add the ML-EMBER Ancient Rome metric.

## References

- Abid, Zhang, Bagaria and Zou. Exploring patterns enriched in a dataset with contrastive principal component analysis. Nature Communications, 2018.
- Belrose et al. LEACE: Perfect linear concept erasure in closed form. NeurIPS 2023.
- Bills et al. Language models can explain neurons in language models. OpenAI, 2023.
- Blumensath and Davies. Iterative hard thresholding for compressed sensing. Applied and Computational Harmonic Analysis, 2009.
- Bussmann, Leask and Nanda. BatchTopK sparse autoencoders. arXiv:2412.06410, 2024.
- Chang, Tu and Bergen. The geometry of multilingual language model representations. EMNLP 2022.
- Ethayarajh. How contextual are contextualized word representations? EMNLP 2019.
- Gao, Ma, Lin and Callan. Precise zero-shot dense retrieval without relevance labels (HyDE). ACL 2023.
- Gao et al. Scaling and evaluating sparse autoencoders. ICLR 2025.
- Hyvärinen and Oja. Independent component analysis: algorithms and applications. Neural Networks, 2000.
- Libovický, Rosa and Fraser. On the language neutrality of pre-trained multilingual representations. Findings of EMNLP 2020.
- Makhzani and Frey. k-Sparse autoencoders. ICLR 2014.
- Mu and Viswanath. All-but-the-top: simple and effective postprocessing for word representations. ICLR 2018.
- Paulo et al. Automatically interpreting millions of features in large language models. arXiv:2410.13928, 2024.
- Su et al. Whitening sentence representations for better semantics and faster retrieval. arXiv:2103.15316, 2021.
- Suslik, Shafran and Geva. Don't forget your embeddings: robust knowledge erasure via precise editing of embeddings (EMBER). arXiv:2606.03695, 2026.
- Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society B, 1996.
- Wang, Yang and Wei. Query2doc: query expansion with large language models. EMNLP 2023.
- Xiong et al. Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. ICLR 2024.
