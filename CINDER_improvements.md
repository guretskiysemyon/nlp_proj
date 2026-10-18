# CINDER: sharper, adaptive and verified concept features for embedding erasure

Oct 18, 2026 · @Ori

**How to read this.** Part 1 is the overview: the problem, the method, the main results and where we stand. Part 2 gives the details of each layer: exact steps, theory, literature, full tables and caveats. Part 3 holds the naming map, open items and references.

---

## Part 1. Overview

### The problem

EMBER (Suslik, Shafran and Geva) erases a concept in four steps. It builds a matrix from the embeddings of concept and neutral tokens, factorizes it into sparse features, selects the features that belong to the concept, and subtracts them from the concept tokens' embeddings.

The edit can only remove what the selected features capture. In our Part 1 reproduction on 18 concepts, EMBER removes only about half of the concept knowledge, and the edit leaks to other languages: erase in English, and the model still answers in Russian or German. (These numbers are provisional while the MC scoring repair runs.)

We traced the weak features to the first three steps. Each has its own failure:

- **The matrix.** A concept like Harry Potter is a weak direction in the embeddings. The factorization spends its features on strong generic topics (magic, writing, books) instead.
- **The factorization.** Every feature keeps exactly 1% of the tokens. A broad concept is cut short, and a narrow one is padded with unrelated tokens.
- **The selection.** An LLM judge decides from a one-sentence summary of the feature, and the summary often loses the concept. It rejected correct features for Valentine's Day, World War II and Heroin.

### The method

CINDER (Contrastive Informed Directional Embedding Removal) makes one change per step and keeps EMBER's edit as it is: **sharpen the matrix, budget sparsity globally, verify both ways.**

| Layer | Change | In one sentence |
| --- | --- | --- |
| L1 | **Contrastive Sharpening (CS)** | Transform the matrix so that what distinguishes the concept from ordinary text becomes the strongest direction. |
| L2 | **Global Sparsity (GS)** | Give the whole matrix one sparsity budget, so each feature takes as many tokens as the data supports. |
| L3 | **Two-Way Judge (TWJ)** | Next to the judge, check top-down whether the feature's words match a lexicon of the concept. |

The layers depend on each other. GS only helps once CS has made the features clean, and TWJ fixes the rejections that remain after both.

### Main results

All results so far measure the **features**, on Gemma-2-2b-it. Erasure after the edit has not been measured for CINDER yet.

- **CS brings the concept to the top.** The Harry Potter feature moves from 11th to 1st place. On three concepts, the runs whose concept feature is nearly free of neutral tokens go from 4 of 15 to 15 of 15.
- **GS lets each concept find its own width.** Across 18 concepts, concept features range from 15 to 113 tokens instead of a fixed 48, and they stay as clean as EMBER's. Giving every feature more tokens instead (a fixed 2%) cuts the clean runs from 75 to 31 of 90.
- **GS needs CS.** On the original matrix, the extra width fills with neutral tokens.
- **TWJ finds the concept far more often.** On 19 concepts, a correct or broader feature is accepted in 82 of 91 runs, against 61 for EMBER's judge. Only 1 of its 110 accepts was unrelated to the concept.

### What to keep in mind

- **Features, not erasure.** Cleaner features are the mechanism, and the erasure evaluation is the claim. It is the next experiment.
- **CS may be partly circular at the feature level.** The sharpening target is the concept's Wikipedia page, which contains the very tokens our purity metrics score. With a target that excludes those tokens, the CS gains mostly disappear. The erasure evaluation does not depend on these labels, so it will tell, and a disjoint target is needed for a paper.
- **TWJ's extra accepts are often broader than the concept.** With strict labels its gain shrinks from 21 to 7 runs. Its threshold was chosen on the same labeled set, and the 15 held-out concepts are not judged yet.

### Ladder and erasure results (Oct 18)

The cumulative ladder ran on 38 concepts (15 of them held out) at the feature level, and on 14 paper concepts with EMBER's erasure protocol. The analysis was pre-registered. Full numbers are in [results/varwidth/layers/SUMMARY.md](results/varwidth/layers/SUMMARY.md).

| Rung | Concept acc. ↓ | Efficacy | H |
| --- | --- | --- | --- |
| EMBER | 54.3 | 0.36 | 0.52 |
| +CS | 62.4 | 0.21 | 0.38 |
| +CS+GS | 62.1 | 0.22 | 0.39 |
| +CS+GS+TWJ (CINDER) | 59.7 | 0.28 | 0.49 |
| EMBER + TJ∨WJ+share (exploratory) | **50.0** | **0.47** | **0.65** |

- **CS hurts erasure** (Δ H −0.15). Its features absorb generic words that occur only in the concept sentences, such as get, win, heart and values. The feature-level purity gains above partly came from that label.
- **GS is neutral for erasure.**
- **The judge layer helps** (+0.10 on top of CS+GS, +0.12 on EMBER's own matrix, no concept worse). It rescues concepts where EMBER's judge selected nothing: Heroin, Uranium, Nazism, Halloween.
- **On the 15 held-out concepts, EJ@5 does not generalize.** TJ ∨ WJ+share does: precision 0.71, recall 0.89.
- **Fidelity checks pass.** Our EMBER matches the paper's Table 6, and the paper's own Alpaca-in-grid tuning gives the same result as our cheaper gate.
- **The pre-registered feature-level endpoint (strict clean-hit) showed no gain.**

What this means for the paper: the robust contribution is the judge diagnosis and fix. CS needs a stricter definition of concept tokens before it can help erasure. The judge-only configuration was chosen after seeing the results, so it needs confirmation on new concepts or seeds.

### Contributions and potential

1. **A diagnosis.** EMBER's features miss concepts for three separate reasons (variance, allocation and verification), and each one has a measurable symptom.
2. **Three principled fixes,** each grounded in an established technique: contrastive whitening, global top-K projection and query expansion.
3. **A layered design.** The fixes depend on each other, so we evaluate them as a cumulative ladder: EMBER → +CS → +CS+GS → +CS+GS+TWJ.
4. **A path to multilingual erasure,** where EMBER is weakest and each layer addresses a specific risk.

If the feature gains carry over to erasure, this is a solid paper: a clear diagnosis of a recent method, three cheap and principled fixes, and an ablation that shows why each is needed. TWJ is useful beyond erasure, for any pipeline that labels features with an LLM. The multilingual extension would make the contribution larger.

What the paper still needs:

1. Erasure on all 18 concepts for the full ladder, with EMBER's protocol and the fixed MC scoring.
2. CS with a sharpening target disjoint from the scored tokens.
3. The cross-ablations on all 18 concepts: GS without CS (running) and TWJ without GS.
4. TWJ on the 15 held-out concepts.
5. The three-way multilingual comparison: per-language EMBER, ML-EMBER with extrapolation, and ML-CINDER.

### Extension: multilingual erasure

EMBER is fit on English text, and its edit leaks to other languages. We will compare three ways to erase a concept in every language:

1. **Per-language EMBER (baseline).** Run the original EMBER separately in each language, and apply all the edits.
2. **ML-EMBER with extrapolation.** Fit one factorization on a small multilingual sample, then keep its features fixed and compute coefficients for the full multilingual token set, so that tokens outside the sample are edited too.
3. **ML-CINDER.** Approach 2 with the three CINDER layers.

Approach 2 replaces one factorization per language with a single cheap fit. Approach 3 adds the layers, each of which addresses a risk specific to fitting across languages. So far we have proof-of-concept fits on Llama for two concepts, and no multilingual erasure numbers yet (details in Part 2).

---

## Part 2. Details

### L1. Contrastive Sharpening (CS)

**Idea.** Sparse MF minimizes squared error, so it spends its features on the directions with the most variance. In EMBER's matrix those are generic topics shared by many tokens. A concept has little variance: it has few tokens, and they were seen less in training. CS first removes the variance that ordinary background text already explains, then stretches the directions where the concept page differs from the background. The concept then stands out in the geometry itself.

**Method.**

1. **Background.** Compute the mean μ\_B and covariance Σ\_B = QΛQᵀ of the embeddings of the token types in a large Wikipedia sample, each type counted once. This is done once per model.
2. **Whiten.** Map every row e of the matrix to ũ = W(e − μ\_B), with W = (Σ\_B + εI)^(−1/2). Every background direction now has unit variance. The small ridge ε = 10⁻⁴ · mean(Λ) damps only the smallest eigenvalues.
3. **Sharpen.** Whiten the token types of the concept's English Wikipedia page in the same way, giving rows ũ\_1, …, ũ\_m. Their second moment, shrunk toward the identity, is S\_η = (1 − η) · (1/m) Σ ũ\_j ũ\_jᵀ + ηI. Map every row to x = S\_η^(γ/2) ũ.
4. **Factorize** the transformed matrix with sparse MF, exactly as in EMBER.
5. **Edit.** Map each concept token's part back exactly, c\_t = T⁻¹ c̃\_t with T⁻¹ = W⁻¹ S\_η^(−γ/2), and subtract it from the original embedding. The mean cancels in a difference, so the map is exact, and the notebook checks this numerically. As in EMBER, only concept-only tokens are edited. The mapped-back c\_t has a different norm from EMBER's, so δ is tuned again for each configuration on the validation grid, and every erasure run first checks that the transformed model rows equal the factorized matrix columns, stopping otherwise ([erase.py](experiments/varwidth/erase.py)).

Defaults: η = 0.05 and γ = 1, chosen by a judge-free sweep on Harry Potter, Ancient Rome and Valentine's Day. η = 1 would mean whitening only, and γ sets how strongly the enriched directions are stretched.

**Theory.** After whitening, the second moment of the page tokens along a unit direction v is a concept-to-background ratio:

```latex
\frac{v^\top \left(\Sigma_C + \delta\delta^\top\right) v}{v^\top \Sigma_B \, v}, \qquad \delta = \mu_C - \mu_B
```

The numerator counts both how much the page varies along v and how far its mean is shifted from the background. Maximizing this ratio is a generalized eigenproblem, the same objective as Fisher's discriminant and contrastive PCA. Sharpening with γ > 0 gives more weight to the directions where the ratio is large. Whitening alone only removes the background's advantage; sharpening is what puts the concept on top.

**Literature.**

- Contrastive PCA (Abid et al., 2018) finds directions enriched in a target dataset relative to a background. CS uses the same contrast as preprocessing, not as the decomposition itself.
- Whitening is standard before ICA and sparse coding (Hyvärinen and Oja, 2000).
- Embeddings are dominated by a few common directions (Ethayarajh, 2019), and removing or whitening them improves representations (Mu and Viswanath, 2018; Su et al., 2021).
- LEACE (Belrose et al., 2023) whitens to make a concept removable by a linear projection.

**Results.**

| Evidence | EMBER | +CS |
| --- | --- | --- |
| Harry Potter feature: rank by ρ | 11th (ρ = 3.9) | **1st** (ρ = 125) |
| HP tokens among ranks 20–30 of that feature | 2 of 11 | **11 of 11** |
| SVD rank that matches the sparse MF error | 5 | 49 |
| Ancient Rome, 20 seeds: seeds with a concept feature | 17 | **20** |
| Ancient Rome: concept share of the feature | 0.72 ± 0.15 | **0.92 ± 0.04** |
| Ancient Rome: neutral tokens per feature | 13.8 | **1.7** |
| Ancient Rome: token overlap across seeds (Jaccard) | 0.26 | **0.69** |
| 3 concepts × 5 seeds: runs with ≤ 5 neutral tokens | 4/15 | **15/15** |
| Same runs: neutral share of the concept feature | 15–25% | **1–9%** |
| Same runs: token overlap across seeds | 0.36–0.63 | **0.75–0.89** |

The three concepts in the last rows are Ancient Rome, Harry Potter and Valentine's Day, all under EMBER's 1% sparsity rule.

What the numbers show:

- **The concept goes from buried to dominant.** The Harry Potter feature leads the next one (magic) by 2.4×. The matrix is also richer: matching the sparse MF error takes a rank-49 SVD with CS, against rank 5 without it.
- **The tokens are right.** Without CS, the HP feature's tail holds Nestlé, Lego and Trump. With CS, it holds Gringotts, Hogsmeade, Quidditch and Hagrid.
- **The result is stable.** Without CS, an Ancient Rome run finds 0, 1 or 2 concept features depending on the seed. With CS, every seed finds exactly one, and 22 tokens appear in all 20 seeds.
- **The features are distinct.** HP and magic are 68° apart, and every other neighbour is above 82°.
- **Sharpening is the decisive step.** Whitening alone is worse than the original matrix on all three concepts: it lowers both concept-token coverage and the concept-vs-neutral AUC. Adding sharpening raises both above the original.

**The target overlaps the scored tokens.** The sharpening target is the concept's page, so it contains the concept tokens that the purity metrics count. The η sweep includes a control target, the page minus every token of the matrix, which the code calls leak-free. With it, the gains mostly disappear:

| Concept | Original: ρ max / purity / AUC | CS, page | CS, page minus matrix tokens |
| --- | --- | --- | --- |
| Harry Potter | 26 / 0.88 / 0.64 | 125 / 0.94 / 0.70 | 14 / 0.67 / 0.64 |
| Ancient Rome | 11 / 0.88 / 0.65 | 49 / 0.88 / 0.68 | 9 / 0.79 / 0.62 |
| Valentine's Day | 45 / 0.94 / 0.63 | 50 / 0.90 / 0.67 | 7 / 0.73 / 0.58 |

Using the page is legitimate: EMBER's concept sentences come from the same page, and an eraser always has the concept's text. But part of the feature-level gain may be circular, because the transform stretches the directions of the same tokens that are then scored. Erasure is measured on held-out questions and does not depend on these labels, so it is the fair test. A paper should also use a disjoint target, such as other pages about the concept or a held-out half of the page.

**Other caveats.**

- The Ancient Rome sweep drew a random width for each feature in both matrices, so it compares stability, not width.
- η and γ were tuned on the same three concepts the comparisons use.
- The comparison without CS on all 18 concepts is running.
- With Llama (d = 4096, against 2304 for Gemma), the page may have too few tokens for a well-conditioned S\_η, so η must grow or γ must shrink.

### L2. Global Sparsity (GS)

**Idea.** EMBER keeps the top 1% of tokens in every feature, about 48 tokens. We call this **Row Sparsity (RS)**. Concepts differ in breadth: Culture of Greece may need more than 1%, and Baseball fewer. A concept that is cut short is left partly unedited, and one that is padded with unrelated tokens damages them in the edit. GS keeps EMBER's total budget but shares it across the whole matrix, so a feature's width becomes an output of the fit. The support is re-chosen at every iteration, so the width shapes the feature directions too, not only the edit set.

**Method** (inside the ALS loop, replacing EMBER's per-row step):

1. Solve for a dense Y with Z fixed, as in EMBER.
2. Score every entry by its share of its own row: s\_ij = |y\_ij| / ‖y\_i‖₂, using the whole dense row.
3. Keep the top K entries of the whole matrix, with K = 1% of all entries, the same total as EMBER.
4. Bound each feature's width between 0.2% and 3% of the tokens. Every row first keeps its top 0.2%. The other entries are then added in score order, skipping rows at the cap, until exactly K are kept.
5. Mask the dense Y with the kept entries, as in EMBER. Rescale the rows to unit norm, update Z, and repeat until convergence.

**Theory.** RS constrains each row of Y on its own. GS constrains the whole matrix, with the same total:

```latex
\text{RS: } \|y_i\|_0 \le s\,n \;\; \forall i \qquad\qquad \text{GS: } \|Y\|_0 \le s\,k\,n
```

Keeping the global top K is the exact projection onto the GS constraint, the same step as iterative hard thresholding. The scores must be comparable across rows, or high-energy generic features take the whole budget. Dividing by the row norm makes the score scale-free, so every feature is asked the same question: is this token a meaningful share of the feature? The floor and cap keep a feature from dying or from absorbing the budget.

**Variants.**

| Name | Rule | Role |
| --- | --- | --- |
| RS-1% | Top 1% of tokens per feature | EMBER's rule |
| RS-2% | Top 2% per feature (twice the total) | Control: more tokens, no reallocation |
| **GS-norm** | Global top K on \|y\_ij\| / ‖y\_i‖ | Our method |
| GS-gain | Global top K on \|y\_ij\| · ‖z\_i‖ | Ablation: favours high-energy features |
| GS-entropy | Width from the effective support size exp(H) of each row | Ablation |

**Literature.**

- BatchTopK SAEs (Bussmann et al., 2024) replace per-example top-k with a top-k across the batch. GS makes the same move across features instead of examples.
- Top-k sparsity in autoencoders (Makhzani and Frey, 2014; Gao et al., 2025) and hard thresholding onto an L0 ball (Blumensath and Davies, 2009).
- The Lasso (Tibshirani, 1996) also gives a variable support, controlled by a penalty instead of a budget. Our L1 variant failed: with CS, its Ancient Rome feature was never clean, and the judge kept it in none of 5 seeds.

**Results.** All five rules on the CS matrix, 18 concepts × 5 seeds (90 runs per rule), measured on the concept feature. This sweep used a 5% cap; the current default is 3%.

| Metric | RS-1% (EMBER) | RS-2% | **GS-norm** | GS-gain | GS-entropy |
| --- | --- | --- | --- | --- | --- |
| Concept width across concepts (tokens) | 43–52 | 86–104 | **15–113** | 9–143 | 25–121 |
| Width change across seeds (tokens) | ±0 | ±0 | **±4.1** | ±8.1 | ±7.9 |
| Runs with ≤ 5 neutral tokens | 75/90 | 31/90 | **76/90** | 55/90 | 61/90 |
| Mean neutral share | 5.8% | 8.2% | **5.6%** | 8.0% | 8.2% |
| Token overlap across seeds (Jaccard) | 0.719 | 0.726 | **0.735** | 0.717 | 0.733 |
| Direction similarity across seeds (worst pair) | 0.838 | **0.872** | 0.866 | 0.817 | 0.808 |
| Runs where the judge kept the concept | 66/90 | 69/90 | 63/90 | 63/90 | 67/90 |

What the numbers show:

- **GS-norm adapts width at no cost.** It is the only rule that changes the width per concept while staying as clean and stable as EMBER. Example widths: Baseball 15, Harry Potter 63, Ancient Rome 103, Culture of Greece 113 tokens. A concept's width moves by only about 4 tokens across seeds, so it reflects the concept, not the seed.
- **More tokens alone do not help.** RS-2% widens every feature and loses more than half of the clean runs. GS-gain and GS-entropy also let neutral tokens in.
- **The data sets the width, not the cap.** At the 3% cap (116 runs, 24 concepts), no concept feature reached the cap. The widest was Culture of Greece, at 120 of 156 tokens. Generic features do reach it, which is what the cap is for.
- **GS needs CS.** On the original matrix (Ancient Rome, Harry Potter, Valentine's Day), GS-norm's concept features are 11–44% neutral, against 15–25% for RS-1%. With CS they are 1–9% neutral.
- **The judge is now the bottleneck.** The judge keeps the concept in a similar number of runs under every rule, and the differences are mostly judge decisions on features that are already clean. Under GS-norm, the Nazism feature is clean and stable in all 5 seeds, and the judge rejects it every time. This is what L3 addresses.

**Caveats.** A different width shows that the feature changed, not that erasure improved; that needs the edit. The comparison without CS covers 3 concepts so far.

### L3. Two-Way Judge (TWJ)

**Idea.** EMBER's judge works bottom-up. It summarizes the feature's tokens in one sentence, then asks whether the summary is the concept, so anything the summary drops is lost. In our runs, "heroin" became "addictive drugs" and World War II became "20th-century totalitarian regimes", and both were rejected. TWJ adds a top-down path. It expands the concept once into a lexicon of specific words, and counts how many of the feature's words appear in it. The feature is abstracted up to the concept, and the concept is expanded down to the feature.

**Method.**

1. **Token Judge (TJ):** EMBER's judge, unchanged. Stage 1 describes the top-20 tokens without the concept name; stage 2 decides membership from the description, accepting at confidence ≥ 0.85.
2. **Concept lexicon,** once per concept: one LLM call lists 20 words specific to the concept's Wikipedia page, not to its broader category.
3. **Feature words:** map each of the feature's top tokens to its most frequent source word ("ffindor" → Gryffindor), normalize, remove duplicates, and keep the first 20.
4. **Expansion Judge (EJ@t):** accept if at least t lexicon words match a feature word exactly.
5. **TWJ = TJ ∨ EJ@5.**

**Theory.** The two paths fail in different ways. The summary path fails when the summary abstracts the concept away. The lexicon path fails when the feature uses words the lexicon missed, or when a broader feature shares many of its words. An OR of two tests with different blind spots raises recall, and the threshold t controls precision.

**Variants.**

| Name | Rule | Role |
| --- | --- | --- |
| TJ | EMBER's two-stage judge | Baseline |
| WJ | TJ, shown each token's source words | Tests whether fragments like "ffindor" mislead the judge |
| WJ+share | WJ, and concept share ≥ 0.5 | Stricter word judge |
| EJ@t | Lexicon match, at least t words | The new check alone |
| **TWJ** | TJ ∨ EJ@5 | Our judge |
| TWJ-R | WJ+share ∨ EJ@4 | High-recall variant |

**Literature.**

- LLM feature interpretation explains a feature, then scores the explanation (Bills et al., 2023; Paulo et al., 2024). TJ follows this pattern.
- Query expansion in retrieval expands the query before matching it (query2doc, Wang et al., 2023; HyDE, Gao et al., 2023). EJ does the same for the concept.
- Verbalized LLM confidence is poorly calibrated (Xiong et al., 2024). This makes EMBER's 0.85 cutoff fragile: all three wrong rejections sat exactly at 0.85.

**Results.**

TJ wrongly rejected Valentine's Day, Heroin and World War II, each at confidence 0.85. It correctly rejected a Uranium candidate about inorganic compounds, which had no uranium token.

For the labeled evaluation, every rule scored every candidate feature from the CS+GS-norm runs: 23 concepts × 5 seeds, 7,105 candidates. Every feature that some rule accepted was hand-labeled, without seeing the judges' decisions, as **P** (specific to the concept), **B** (broader or adjacent, such as generic nuclear physics for Uranium) or **N** (unrelated). The table leaves out the 4 concepts used as examples in the lexicon prompt, so it covers 19 concepts and 95 runs.

| Rule | Precision, P only | Recall, P only | Runs hit, P only (of 72) | Precision, P+B | Recall, P+B | Runs hit, P+B (of 91) |
| --- | --- | --- | --- | --- | --- | --- |
| TJ (EMBER) | **0.85** | 0.69 | 60 | **1.00** | 0.41 | 61 |
| WJ | 0.56 | 0.81 | 61 | 0.77 | 0.55 | 66 |
| EJ@5 alone | 0.74 | 0.64 | 64 | 0.99 | 0.43 | 81 |
| **TWJ** | 0.71 | 0.77 | 67 | 0.99 | 0.54 | **82** |
| TWJ-R | 0.61 | **0.90** | **71** | 0.91 | **0.68** | **84** |

A run is "hit" when at least one accepted feature is a positive. Recall counts only the positives that some rule accepted, because unlabeled candidates count as negatives.

What the numbers show:

- **TWJ finds the concept in many more runs,** 82 of 91 against 61, and only 1 of its 110 accepts is unrelated.
- **The lexicon check is strong on its own.** EJ@5 alone reaches 81 runs, needs no LLM call per feature, and is deterministic.
- **Most added features are broader than the concept.** With strict labels, TWJ adds 7 runs instead of 21, and its precision drops from 0.85 to 0.71. A broader feature suits some uses and is too broad for others; t is the knob between the two.
- **Showing words to the judge is not enough.** WJ raises recall but loses a third of the strict precision. The lexicon match adds recall without adding unrelated features.

**Caveats.** The threshold t = 5 was chosen on this labeled set, and the labels are ours. The real held-out test is the 15 concepts added after the prompt was fixed (Judaism, Karate, …, Tourism): their fits are done, and judging and labeling are next. Broader accepts may cost specificity in the edit, which only the erasure evaluation will show.

### Multilingual extension

**Motivation.** EMBER's English edit leaks to other languages. In Part 1, the model retained on average 0.24 more of its concept accuracy in German, Spanish, French and Russian than in English, and 0.41 more in Russian (retention relative to the unedited model; provisional).

**The three approaches.**

1. **Per-language EMBER (baseline).** For each language L, build EMBER's token set from concept and neutral sentences in L, factorize it, select the concept features and compute the edit. Apply the edits of all languages together. This costs one factorization and one judge pass per language. Tokens shared by several languages (names such as "Harry") may be edited by more than one fit, so we need a rule for combining their edits.
2. **ML-EMBER with extrapolation.**
   - *Sample.* Take 40–50 concept sentences and as many neutral sentences in each language. The sample must contain at least d distinct tokens (2304 for Gemma, 4096 for Llama), so that the sampled embedding matrix has full rank. It should also stay small enough for the factorization to be cheap. In the saved 8-language Llama runs, 40 + 40 sentences per language give about 7,000 tokens.
   - *Fit.* Factorize the sampled matrix with EMBER's row-sparse factorization to get the features Z.
   - *Extrapolate.* Keep Z fixed and solve a row-sparse Y for the full token set: every token of all the concept and neutral sentences in every language, not only the sampled ones.
   - *Select and edit.* Select the concept features on the full Y (ρ filter and judge), and edit every concept-only token of the full set.
3. **ML-CINDER.** Approach 2 with the three layers. CS whitens against a background with the same number of sentences in each language and sharpens toward the pooled concept pages. GS replaces row sparsity in both the fit and the extrapolation. TWJ shows the judge each token's source word and language, and matches the feature against a lexicon in each language.

**What exists already.**

| Piece | Status |
| --- | --- |
| 8-language token sets (en, es, fr, de, it, pt, hi, th), 40 + 40 sentences per language | Built for 6 concepts: Harry Potter, Gambling, Gun, Halloween, Pornography, Valentine's Day. |
| Approach 2's fit on Llama-3.1-8B (k = 200) | Run for 2 concepts. Harry Potter: one concept feature (ρ = 4.36; Voldemort, Dumbledore, Hogwarts, Hermione, …), 47 tokens edited. Valentine's Day: no concept feature, so no edit. |
| Approach 2's extrapolation (fixed Z, sparse Y on up to 300 concept sentences per language, plus matched neutral ones) | Implemented (§3.2.h of the multilingual notebooks). On Harry Potter, the concept feature grows from 72 to 191 tokens, 127 of them new and 86 used in more than one language. **Not yet used by the edit**, which still edits only the sampled tokens. |
| Multilingual CS | Notebook built, not run end to end. First check: against an English-only background, the top contrastive directions are language clusters (enrichment 6.82); against the 8-language background they mostly disappear (2.82). |
| Per-language evaluation | MC and open questions in en, de, es, fr and ru for all 18 concepts (Part 1, `data/qa_lang`). The 8-language notebooks evaluate English only. |
| Translated sentence sets | Concept and neutral sentences in ru, de, es and fr for all 18 concepts (`experiments/multilingual_fit/data_translated`). |

\[An earlier version of this doc said that a naive multilingual EMBER worked on Ancient Rome with Llama. No such run is in the repo; the Ancient Rome Llama notebooks are English-only. Add it here if it exists elsewhere.\]

**Gaps to close before the comparison.**

- **Languages.** The fits use 8 languages, but per-language evaluation exists for 5. Only en, de, es and fr are in both; ru has no 8-language fit data, and it, pt, hi and th have no evaluation data.
- **Model.** The multilingual fits run on Llama, while the Part 1 evaluation, baselines and cross-lingual results are on Gemma.
- **Concepts.** The 8-language data covers 6 concepts; the 5-language data and evaluation cover all 18.
- **Edit.** The extrapolated Y must feed the edit, or approach 2 is not what we test.

**Risks, and the layer that addresses each.**

| Risk | Layer | How it helps |
| --- | --- | --- |
| Language identity is a strong direction, so features may capture "French tokens" instead of concepts. | CS | Whitening against an 8-language background removes it. First check: against an English-only background, the top contrastive directions are language clusters (enrichment 6.82); against the 8-language background they mostly disappear (2.82). |
| Llama's larger dimension leaves too few page tokens per language for sharpening. | CS | Pooling the page tokens across the 8 languages gives enough. |
| One concept has many surface forms (Rome, Roma, Рим), so a fixed width cuts or splits the feature. | GS | The feature can widen to cover every language's forms. |
| Extrapolated coefficients for unsampled tokens are noisy, so the edit reaches unrelated tokens (see below). | GS, CS | GS lets an unsampled token join only the features it is a meaningful share of; CS makes the fixed features cleaner to begin with. |
| Tokens in other scripts are hard for the judge to read, and the lexicon is in English. | TWJ | WJ shows each token's source words and language; EJ needs a lexicon per language or a translation-aware matcher. |

**Extrapolation so far (English only).** On Harry Potter, extending the fit to unsampled tokens added a few correct tokens and many unrelated ones. Adding as many neutral tokens as concept tokens helped, but not enough. Unsampled tokens are the rarer words on the page, which are noisier (Tolkien, Narnia). Approaches 2 and 3 depend on this step most.

**Literature.** Multilingual models encode language identity in a few dominant directions, separate from meaning (Chang et al., 2022), and removing each language's mean makes representations more language-neutral (Libovický et al., 2020). CS's background whitening generalizes this centering.

---

## Part 3. Reference

### Names: experiments → paper

| In the experiments | In the paper |
| --- | --- |
| Contrastive PCA, whitened, CW | Contrastive Sharpening (CS) |
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

### Open items

- [ ] Erasure results (efficacy, specificity, coherence) for the ladder on all 18 concepts.
- [ ] CS with a target disjoint from the matrix tokens, on the features and on erasure.
- [ ] Finish the no-CS fits on all 18 concepts, and rerun the GS comparison at the 3% cap.
- [ ] Judge and label the 15 held-out concepts; report TWJ on them.
- [ ] Choose t for EJ on data other than the evaluation set, for example with a cross-concept null.
- [ ] Tune η and γ on concepts outside the evaluation set.
- [ ] Multilingual: choose the model, languages and concepts so that the fits and the per-language evaluation match.
- [ ] Multilingual: feed the extrapolated Y into the edit.
- [ ] Find the multilingual Ancient Rome run, or drop the claim.
- [x] Method details checked against the code: [sharpen.py](experiments/whitening/sharpen.py), [core.py](experiments/varwidth/core.py), and §3.3 of [ember\_three\_steps\_whitened.ipynb](ember_three_steps_whitened.ipynb).

### Sources of the numbers

- CS, Ancient Rome 20 seeds: [results/ancient\_rome\_seed\_sweep/experiments.md](results/ancient_rome_seed_sweep/experiments.md).
- CS, 3 concepts with and without CS: [results/varwidth/whitened\_vs\_plain.md](results/varwidth/whitened_vs_plain.md).
- CS, η sweep and leak-free control: [experiments/whitening/data/gemma/defaults.json](experiments/whitening/data/gemma/defaults.json).
- GS, 18 concepts: [results/varwidth/overall\_report\_cap0.05.md](results/varwidth/overall_report_cap0.05.md).
- TWJ, labeled evaluation: `python -m experiments.varwidth.j3_eval report` (labels in [experiments/varwidth/j3\_data/labels.json](experiments/varwidth/j3_data/labels.json)).
- Part 1 baselines: [docs/progress.md](docs/progress.md).

### References

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
