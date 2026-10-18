# CINDER: three simple changes to EMBER

Oct 18, 2026 · @Ori

## Storyline

We make three small changes to EMBER, one per step of its pipeline. Each change fixes a failure that only becomes visible once the previous one is in place: **sharpen the matrix, budget sparsity globally, verify both ways.**

EMBER (Suslik, Shafran and Geva) erases a concept in four steps. It builds the sub-embedding matrix E\_V\* from concept and neutral tokens, factorizes it with sparse MF, keeps the concept features (a ρ filter, then an LLM judge), and subtracts them from the concept tokens' embeddings. We change the first three steps and keep the edit as it is.

| Layer | Step | What fails in EMBER | Our change |
| --- | --- | --- | --- |
| L1 | Matrix | The concept is a low-variance direction, so the factorization spends its features on generic high-variance topics (magic, writing, books). | **Contrastive Sharpening (CS):** whiten against background text, then sharpen toward the concept page. |
| L2 | Factorization | Every feature keeps exactly 1% of the tokens. Broad concepts are cut short or split, and narrow ones are padded with noise. | **Global Sparsity (GS):** one nonzero budget for the whole matrix instead of one per feature. |
| L3 | Selection | The judge sees only a one-line summary, so it rejects correct features (Valentine's Day, WWII, Heroin). | **Two-Way Judge (TWJ):** add a top-down check that expands the concept into a lexicon and matches it to the feature's words. |

The layers depend on each other, and this is part of the claim. GS gives no gain on the raw matrix; it helps once CS makes the features clean. TWJ is evaluated on the features CS+GS produce. We test this with a cumulative ladder (EMBER → +CS → +CS+GS → +CS+GS+TWJ) and two cross-ablations: GS without CS, and TWJ without GS.

We call the whole pipeline **CINDER** (Contrastive Informed Directional Embedding Removal).

## L1. Contrastive Sharpening (CS)

CS changes the matrix the factorization sees, so that the concept is a strong direction instead of a weak one. On Harry Potter, it moves the concept feature from rank 11 by ρ to rank 1.

**Idea.** Sparse MF minimizes squared error, so it spends its k features on the directions with the most variance. In E\_V\*, those are generic topics shared by many tokens: magic, writing, books, criticism. A concept like Harry Potter has little variance. It has few tokens, and those tokens were seen less in training, so their representations are poorer. CS gives every topic a fair start. It removes the variance that ordinary background text already explains, then amplifies what makes the concept page different. The concept is then distilled by the geometry, and the factorization does not have to force it by inflating Y entries.

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

Maximizing it is the generalized eigenproblem Σ\_C v = λ Σ\_B v, the same objective as Fisher's discriminant and contrastive PCA. Sharpening with γ > 0 raises the weight of directions where this ratio is large. Whitening alone only removes the background's advantage; sharpening is what puts the concept on top.

**Literature.**

- Contrastive PCA (Abid et al., 2018) finds directions enriched in a target dataset relative to a background dataset. CS uses the same contrast as a preprocessing step, not as the decomposition itself.
- Whitening is standard preprocessing before ICA and sparse coding (Hyvärinen and Oja, 2000).
- Word and sentence embeddings are dominated by a few common directions (Ethayarajh, 2019). Removing or whitening them improves representations (Mu and Viswanath, 2018; Su et al., 2021).
- LEACE (Belrose et al., 2023) whitens to make a concept removable by a linear projection.

**Results so far.**

| Evidence | EMBER | +CS |
| --- | --- | --- |
| Harry Potter feature, rank by ρ (ρ value) | 11th (3.9) | 1st (125; 2.4× the 2nd, magic) |
| HP tokens among ranks 20–30 of that feature | 2 of 11 | 11 of 11 |
| SVD rank that matches the sparse MF error | 5 | 49 |
| Ancient Rome: seeds with a concept feature (20 seeds) | 17, two features in 4 of them | 20, exactly one each |
| Ancient Rome: concept share of the feature | 0.72 ± 0.15 | 0.92 ± 0.04 |
| Ancient Rome: neutral tokens per feature | 13.8 | 1.7 |
| Ancient Rome: token Jaccard across seeds | 0.26 | 0.69 (22 tokens in every seed) |
| Neutral share of selected features (3 concepts) | 19–59% | 2–9% |

What the numbers show:

- **Sharpening is the decisive step.** Whitening alone found no Harry Potter feature, and adding sharpening recovered it.
- **Cleaner token lists.** Without CS, the HP feature's tail holds Nestlé, Lego, Trump and taxonomic fragments. With CS, it holds Gringotts, Hogsmeade, Quidditch, McGonagall and Hagrid.
- **Interpretable geometry.** CS features are close to orthogonal: HP vs magic is 68°, and every other neighbour is above 82°.
- **Caveat.** The Ancient Rome sweep used random per-feature budgets for both matrices, so it compares stability, not width.

**Limits.** Sharpening needs enough concept-page tokens for Σ̃\_C to be well conditioned. With Llama (d = 4096 vs 2304 for Gemma), η must grow or γ must shrink. Erasure accuracy after the edit has not been measured yet.

## L2. Global Sparsity (GS)

GS keeps EMBER's total sparsity of 1% but lets each feature choose its own width. Concept features take as many tokens as the data supports, not a fixed 48.

**Idea.** EMBER's winner-takes-all step keeps the top 1% of tokens in every feature. We call this **Row Sparsity (RS)**. It gives every feature the same width, while concepts differ in breadth. Ancient Rome may need more tokens than 1% (Cleopatra, gladiator and Constantine are cut off), so it gets split into two features. Harry Potter may need fewer, so its feature is padded with unrelated tokens. GS replaces the per-feature budget with one budget for the whole matrix. The width becomes an output of the fit, not a hyperparameter. Because the support is re-chosen at every iteration, the width also shapes the feature directions Z, not only the edit set.

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

Keeping the global top-K entries is the exact projection onto the GS constraint, the same step as iterative hard thresholding. The scores must be comparable across rows, otherwise high-energy generic features take the whole budget. Dividing by the row norm makes the score scale-free, so every feature is asked the same question: is this token a meaningful share of the feature? The floor and cap keep a feature from dying or absorbing the budget.

**Variants** (all with the same total budget unless noted):

| Name | Rule | Role |
| --- | --- | --- |
| RS-1% | Top 1% of tokens per feature | EMBER's rule |
| RS-2% | Top 2% per feature, twice the total | Separates more tokens from better-allocated tokens |
| **GS-norm** | Global top-K on \|y\_ij\| / ‖y\_i‖ | Our method |
| GS-gain | Global top-K on \|y\_ij\| · ‖z\_i‖, the reconstruction gain of the entry | Ablation; also feeds generic high-energy features |
| GS-entropy | Each feature's width set by the effective support size exp(H) of its coefficients | Ablation |

**Literature.**

- BatchTopK SAEs (Bussmann et al., 2024) replace per-example top-k with a top-k across the batch, so the number of active latents adapts per example. GS makes the same move across features instead of examples.
- Top-k sparsity in autoencoders (Makhzani and Frey, 2014; Gao et al., 2025), and hard thresholding onto an L0 ball (Blumensath and Davies, 2009).
- L1-penalized coding (the Lasso; Tibshirani, 1996) also gives a variable support, but it is controlled by a penalty λ instead of a budget.

**Results so far.**

- **Width has to follow the data.** With random per-feature budgets, every concept feature's width equalled its drawn budget (41 of 41) and the reconstruction error did not move. Variable width alone is not enough; the allocation must come from the fit.
- **Under CS, GS-norm sets widths below the cap.** The 3% cap was never reached, so the widths come from the data, not from the guardrail.
- **GS depends on CS.** On the raw matrix, GS gives no clear gain. \[Numbers from the method comparison to be added.\]

## L3. Two-Way Judge (TWJ)

TWJ accepts a feature if either of two independent checks accepts it: EMBER's judge, or a lexicon match that needs no LLM call per feature. It recovers concept features that EMBER's judge rejects.

**Idea.** EMBER's judge works bottom-up. It summarizes a feature's tokens in one sentence, then asks whether that summary is the concept, and anything the summary drops is lost. In our runs, "heroin" became "addictive drugs" and WWII became "20th-century totalitarian regimes", and both were rejected. TWJ adds a top-down path: expand the concept once into a lexicon of words specific to it, then count how many of the feature's words appear in it. The two paths are mirror views of one question. The feature is abstracted up to the concept, and the concept is expanded down to the feature.

**Steps.**

1. **Token Judge (TJ),** EMBER's judge, unchanged: stage 1 describes the top-20 tokens without the concept name, stage 2 decides membership from that description.
2. **Concept lexicon,** once per concept: one LLM call lists words you would expect in the concept's Wikipedia page, specific to the concept rather than its broader category.
3. **Feature words:** map each of the feature's top tokens (in |y| order) to its most frequent source word in S\_C ∪ S\_N ("ffindor" → Gryffindor), normalize, remove duplicates and keep the first 10.
4. **Expansion Judge (EJ@t):** accept if at least t feature words match the lexicon exactly.
5. **TWJ = TJ ∨ EJ@5.**

**Theory.** The two paths fail in different ways. The summary path fails when the summary abstracts the concept away. The lexicon path fails when the feature is about the concept but uses words the lexicon missed, or when a broader feature shares many words with it. An OR of two tests with different blind spots raises recall. The threshold t keeps precision in check, and it is chosen with a cross-concept null: match each feature against another concept's lexicon and read off the false-accept rate.

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
- **Held-out concepts.** The 4 concepts used as examples in the lexicon prompt (Culture of Greece, Ancient Rome, Artificial intelligence, World War II) are excluded from every EJ and TWJ result.
- &#91;TWJ vs TJ precision and recall, and EJ on the fixed-width runs, to be added.\]

## Extension: multilingual erasure

A concept should be erased in every language the model speaks, not only in English. A naive multilingual EMBER already worked on Ancient Rome with Llama. The three layers are designed to carry that result to more concepts.

**Naive extension (ML-EMBER).** Build E\_V\* from sentences in 8 languages instead of one. To keep the matrix small enough to factorize, sample 40 sentences from the concept's Wikipedia page in each language (320 in total, close to EMBER's 300). Everything else is EMBER unchanged: factorize, filter by ρ, judge and edit.

**First result.** On Ancient Rome with Llama, ML-EMBER succeeded as a proof of concept. \[Add the metric: MC accuracy after the edit, per language.\] It has not yet been tested on other concepts.

**What can go wrong, and which layer addresses it.**

| Risk in ML-EMBER | Layer | How it helps |
| --- | --- | --- |
| Language identity is a large, high-variance direction. The factorization can spend features on "French tokens" or "Cyrillic tokens" instead of on concepts. | CS | Whitening against a background drawn from the same 8 languages removes the variance that language identity already explains, so the concept, shared across languages, is what remains. |
| Llama's larger dimension (4096) leaves too few concept-page tokens per language for a well-conditioned Σ̃\_C. | CS | Pooling concept-page tokens across the 8 languages gives the sharpening step enough tokens, so η does not have to grow. |
| One concept now has many surface forms (Rome, Roma, Rom, Рим). At a fixed 1% width, the concept feature is cut short or split by language or script. | GS | The feature can widen to cover every language's forms, with the total budget unchanged. |
| With 40 sentences per language, frequent concept tokens can be missed by the sample, and only sampled tokens are edited. | Sparse extrapolation | Fix Z and solve a sparse Y' for tokens outside the sample, so they can be edited too. |
| Sub-word tokens in other scripts are hard for the judge to read, and the lexicon is in English. | TWJ | WJ shows each token with its source words and their language. EJ needs a lexicon per language, or a matcher that counts translations. |

**Sparse extrapolation so far (single language).** On English Harry Potter, extending the factorization to unsampled tokens added a few correct tokens but also many indirect or unrelated ones. Adding as many neutral tokens as concept tokens improved it, but not enough yet. One likely reason: unsampled tokens are by definition the rarer words on the page, which are noisier (Tolkien, Narnia). This is the step multilingual erasure depends on most, so it needs work before scaling.

**Literature.**

- Multilingual models encode language identity in a few dominant directions, separate from meaning (Chang et al., 2022). Removing each language's mean makes representations more language-neutral (Libovický et al., 2020). CS's background whitening generalizes this centering.

**Plan.**

1. Run ML-EMBER on more concepts, to see whether the Ancient Rome result holds.
2. Run ML-CINDER (ML-EMBER + CS + GS + TWJ) on the same concepts and compare, per language.

To confirm: the 8 languages, whether the neutral sentences are also multilingual, and the metric behind the Ancient Rome result.

## Names: experiments → paper

| In the experiments | In the paper |
| --- | --- |
| Contrastive PCA, whitened, CW (background whitening + concept sharpening) | Contrastive Sharpening (CS) |
| fixed 1% / 2% per feature | Row Sparsity: RS-1% / RS-2% |
| varnnz, pooled, shared budget | Global Sparsity (GS) |
| rn, RNP | GS-norm |
| m1, CP, pooled\_contrib | GS-gain |
| m3, EW | GS-entropy |
| J\_tokens, J\_tok | Token Judge (TJ) |
| J\_words, CJ / CJ+S | Word Judge (WJ) / WJ+share |
| J3, overlap judge, LO@t | Expansion Judge (EJ@t) |
| HJ = TJ ∨ LO@5 / HJ-R | Two-Way Judge (TWJ) / TWJ-R |

## Open items

- [ ] Confirm the CS transform against the code: the background set, whether Σ̃\_C is computed in whitened space, and the exact role of γ and η.
- [ ] Confirm how the CS edit is done: are the feature directions mapped back to the original embedding space before subtraction?
- [ ] Confirm the GS-norm score: is ‖y\_i‖ the norm of the dense row or of the kept entries?
- [ ] Add the GS method comparison (RS-1%, RS-2%, GS-norm, GS-gain, GS-entropy) and the GS-without-CS ablation.
- [ ] Add the judge comparison, and EJ and TWJ on the fixed-width runs.
- [ ] Add erasure results (MC accuracy after the edit) for at least a subset of concepts.

## References

- Abid, Zhang, Bagaria and Zou. Exploring patterns enriched in a dataset with contrastive principal component analysis. Nature Communications, 2018.
- Chang, Tu and Bergen. The geometry of multilingual language model representations. EMNLP 2022.
- Belrose et al. LEACE: Perfect linear concept erasure in closed form. NeurIPS 2023.
- Bills et al. Language models can explain neurons in language models. OpenAI, 2023.
- Blumensath and Davies. Iterative hard thresholding for compressed sensing. Applied and Computational Harmonic Analysis, 2009.
- Bussmann, Leask and Nanda. BatchTopK sparse autoencoders. arXiv:2412.06410, 2024.
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
