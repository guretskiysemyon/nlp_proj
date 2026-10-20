# Better Features, More Tokens: Reducing the Cross-Lingual Leakage of Embedding Erasure

**Part 2 report: CINDER and token extrapolation on Gemma-2-2B-it**

Mary Arav, Ori Petel, Semyon Guretskiy (Tel Aviv University, NLP course 2025/26)

---

## 1. Methods

### 1.1 Starting point

Part 1 showed that an erasure edit built from English data leaks across languages. EMBER subtracts concept features from the embeddings of a few dozen English tokens, and after this edit the concept survives better in German, Spanish, French and especially Russian than in English (mean leak +0.19, Russian +0.33). Two causes are plausible. First, the edit may be weak: in our Part 1 reproduction EMBER removes less than half of the concept even in English (efficacy 0.43), and with the judge model now available to us the feature-selection step sometimes finds no concept feature at all. Second, the edit may simply not reach the tokens that foreign questions use. Part 2 tests both explanations. We ask whether a better feature-finding method (CINDER, Section 1.2) improves English erasure and reduces leakage, and whether the edit can be extended to foreign tokens without training on foreign text (extrapolation, Section 1.4). We also test the more direct alternative of fitting the features on foreign text (Section 4.1).

All edits keep EMBER's edit step. Features are found on English concept and neutral sentences, a feature is accepted by a judge, and the concept component of each selected token is subtracted from its embedding with strength δ (Part 1, Section 2.2). What differs between the methods below is how the features are found and which tokens are edited.

![[fig1_methods.png|720]]

**Figure 1.** The methods compared. EMBER (top) is the Part 1 method. CINDER (middle) changes the first three steps and keeps the edit. Extrapolation (bottom, "+X") is added to either method and changes only the set of edited tokens.

### 1.2 CINDER: contrastive, adaptive and verified features

CINDER (Contrastive Informed Directional Embedding Removal) was developed in our team as a repair of three weak points of EMBER's feature discovery. Its first three layers are listed below; the edit itself is unchanged.

**Contrastive Sharpening (CS).** Sparse matrix factorization spends its features on the directions with the most variance, and in the embeddings of concept and neutral tokens those are generic topics (for Harry Potter: magic, writing, books). A concept has few tokens, so its direction is weak. CS first whitens every token embedding against a large background of English Wikipedia, with mean $\mu_B$ and covariance $\Sigma_B$, so that ordinary directions have unit variance:

$$\tilde u = W(e-\mu_B),\qquad W=(\Sigma_B+\epsilon I)^{-1/2}.$$

It then stretches the directions in which the concept's own Wikipedia page differs from the background, with the second moment $S_\eta=(1-\eta)\frac1m\sum_j\tilde u_j\tilde u_j^\top+\eta I$ of the page tokens, giving $x=S_\eta^{\gamma/2}\tilde u$. The factorization is run on $x$. Each concept token's component is mapped back exactly to the embedding space before it is subtracted. We use the defaults $\eta=0.05$ and $\gamma=1$. The idea is the same as in contrastive PCA, which finds directions enriched in a target set relative to a background.

**Global Sparsity (GS).** EMBER keeps the top 1% of tokens in every feature, about 48 tokens. A broad concept such as Culture of Greece is cut short and a narrow one such as Baseball is padded with unrelated tokens. GS keeps EMBER's total budget (1% of all entries of the coefficient matrix) but shares it across the whole matrix. Each entry is scored by its share of its own feature, $|y_{ij}|/\lVert y_i\rVert_2$, the top entries of the whole matrix are kept, and every feature keeps between 0.2% and 3% of the tokens. The width of a feature thus becomes an output of the fit.

**Two-Way Judge (TWJ).** EMBER's judge works bottom-up: it summarizes the feature's top tokens in one sentence and decides whether the summary is the concept, so anything the summary abstracts away is lost (in our runs "heroin" became "addictive drugs" and World War II became "20th-century totalitarian regimes", and both were rejected). TWJ adds a top-down check. One LLM call expands the concept into a lexicon of 20 specific words, and a feature is also accepted if at least five of its own top 20 source words appear in the lexicon. The result is the OR of the two tests, TWJ = TJ ∨ EJ@5, where TJ is EMBER's judge unchanged.

### 1.3 Why we use CINDER

CINDER is a direct candidate answer to the first explanation in Section 1.1. Its three layers target three failure points of EMBER's feature discovery that we also observed in Part 1: concept features that are weak, features whose width does not fit the concept, and a judge that rejects correct features or finds none. At the feature level the CINDER authors report cleaner and more stable features: the Harry Potter feature moves from 11th to 1st place by concept ratio, concept-feature widths range from 15 to 113 tokens instead of a fixed 48, and a correct or broader feature is accepted in 82 of 91 runs against 61 for EMBER's judge. These are measurements of features, not of erasure.

The authors' own ablation on 14 concepts with EMBER's erasure protocol is less favorable. CS lowered erasure (it admits generic words that occur only in the concept sentences), GS was neutral, and the judge layer helped: it rescued concepts for which EMBER's judge selects nothing (Heroin, Uranium, Nazism, Halloween). Whether CINDER as a whole erases better, and whether better English erasure carries over to other languages, had not been tested end to end. We therefore evaluate CINDER as one method under Part 1's protocol, and we add one arm that isolates its judge (Section 1.5). CINDER is also an informative case for a second reason: its edit is mapped back from a whitened space, and its features are wider for some concepts, which changes how many tokens, and which tokens, are edited.

### 1.4 Extrapolation to native tokens (+X)

EMBER edits only tokens that occur in the English sentences used for the fit. A foreign question contains other tokens, and a Russian word is typically split into several short pieces (the name Гитлер becomes Г + ит + лер) that are not edited. Extrapolation extends the edit to such tokens without refitting the features and without any change to the English edit.

The features $Z\in\mathbb R^{d\times k}$ of the English fit are kept fixed. For every token $t$ that occurs in native German, Spanish, French or Russian concept and neutral sentences (Wikipedia, up to 300 sentences per concept and language) but not in the English token set, the coefficients on the fixed features are solved by ridge regression,

$$y_t=(Z^\top Z+\lambda I)^{-1}Z^\top e_t,$$

and then sparsified with the method's own rule (EMBER: the top 1% of tokens per feature; CINDER: the global budget in its whitened space, over the union of all tokens). A token is added to the edit if it is concept-only, that is, it occurs in the concept sentences of some language and in no neutral sentence of any of the five languages, and if it loads on a selected concept feature. The added tokens are edited with the same formula as the original ones, $e_t\leftarrow e_t-\delta\sum_{i\in F_C}y_{i,t}z_i$. The original English rows and the strength δ are kept exactly.

Extrapolation uses foreign native text to choose the added tokens, so it is no longer a pure query-only method; the features themselves stay English-only.

### 1.5 Arms

**Table 1.** The arms of the main comparison. All use the same model, English concept and neutral sentences, feature judge model, strength grid, selection rule and tests; only the method differs.

| Arm | Features | Judge | Edited tokens |
|---|---|---|---|
| EMBER | EMBER's factorization | EMBER's two-stage judge (TJ) | English tokens |
| CINDER | CS + GS factorization | TWJ = TJ ∨ EJ@5 | English tokens |
| EMBER+TWJ | EMBER's factorization | TWJ | English tokens |
| EMBER+X | as EMBER | as EMBER | English tokens + extrapolated foreign tokens |
| CINDER+X | as CINDER | as CINDER | English tokens + extrapolated foreign tokens |

EMBER+TWJ separates the contribution of CINDER's judge from that of its sharpening and sparsity, since it uses EMBER's factorization unchanged. It is built on the CPU from the same candidate features: it keeps EMBER's verdicts and adds a feature only where the lexicon test accepts one. The +X arms use the parent's features and strength.

---

## 2. Experimental setup

### 2.1 Model, concepts and data

We use Gemma-2-2B-it in bfloat16 with greedy decoding, rank $k=100$ and seed 42, and the data of Part 1: the 18 concepts of the EMBER paper, 50 + 50 four-option multiple-choice questions per concept for selection (`QA_train`) and held-out evaluation (`QA_test`), 50 + 50 questions on a similar topic (`SimdomQA`), MMLU for general knowledge, and the German, Spanish, French and Russian translations of the test questions. The English concept and neutral sentences (300 + 300 per concept) are those of the EMBER paper. The native sentences used by extrapolation (Section 1.4) and by the foreign-language fits (Section 4.1) are Wikipedia sentences from the concept's articles in each language, cleaned of back matter, with random sentences from the same language as the neutral set.

The paper's feature judge, Gemini-2.5-Flash-Lite, is no longer available, so every Part 2 arm uses gemini-3.5-flash-lite as the feature judge. With it EMBER keeps the same features as in Part 1 for 10 of the 18 concepts, different features for four, and none for four (Halloween, Heroin, Uranium and Valentine's Day), so EMBER makes no edit for these four. CINDER keeps a feature for 15 concepts and none for Nazism, Republic of Ireland and World War II.

We ran the comparison in two stages. The first stage used six concepts and Part 1's full protocol, including the Alpaca coherence judge in the selection of δ. The concepts, Cannabis, Gun, Nazism, Republic of Ireland, Culture of Greece and Golf, span strong, medium and weak EMBER erasure, have stable EMBER features under the new judge, and none was used to tune CINDER. The second stage, the fair comparison, covers all 18 concepts without the Alpaca term, so that all arms share one cheap pipeline.

### 2.2 Selection of the strength and the final test

CINDER's edit is about half the size of EMBER's at the same δ (the median edit norm relative to the embedding norm is 0.24 against 0.46, a ratio of 1.93), which lies inside the band of 0.5 to 2 that we had fixed in advance for reusing EMBER's grid. All arms therefore use EMBER's grid, δ ∈ {0.5, 1, 2, 5, 10, 50, 100, 200}, and one more value is scored if the best δ is at the edge.

In the fair comparison the strength is chosen in English, per concept and arm, by Part 1's judge-free screen: among the strengths whose MMLU fraction is at least 0.9, the one with the highest HM(efficacy, specificity) on `QA_train`, `SimdomQA_train` and the 50-question MMLU set, with ties broken towards the smaller δ. There is no Qwen coherence term here, so these picks are not directly comparable with Part 1's. Each +X arm keeps its parent's δ and has no search of its own. The edit is then applied once and evaluated on `QA_test` and `SimdomQA_test` in five languages and on the 1,000-question MMLU test split, with a check that the edited model reproduces its grid row. A concept without a kept feature gets no edit and is scored as the unedited model.

### 2.3 Mechanism analyses and controls

Three analyses test why the edit leaks and what extrapolation changes. The touched/untouched analysis splits the questions that the unedited model answers correctly into those whose text or correct answer contains an edited token and those that do not, and compares how many answers each group loses. The coverage analysis relates the share of questions with an edited token in a language to the leak in that language. The extrapolation controls apply real erasure to the six first-stage concepts on the train questions: variants of the added tokens (without tokens that occur in English; clean of noise) are compared with the full set, and two token-choice controls add the same number of tokens with the same edit norms, chosen by sentence frequency (no features) or at random from the same concept-only pool. The held-out test split is used only to confirm choices made on the train split.

---

## 3. Metrics

We use the metrics of Part 1 unchanged: the chance-corrected ratio $\phi(a;a_0)=\operatorname{clip}\big((a-0.25)/(a_0-0.25),0,1\big)$, efficacy $=1-\phi(\text{QA}_{\text{train}})$, specificity $=\operatorname{HM}(\phi(\text{MMLU}),\phi(\text{SimDom}))$, and the retention $r_L=\phi(\mathrm{acc}_L;\mathrm{acc}^{\text{unedited}}_L)$ and leak $\Delta_L=r_L-r_{\text{en}}$ on the held-out `QA_test`, with the same reliability screen (a concept–language pair is dropped when the unedited accuracy is below 0.40; two of 90 pairs, Harry Potter and Valentine's Day in Russian). The mean leak averages a concept's reliable foreign languages and then the concepts. A concept without an edit has retention 1 in every language and leak 0, and is included in the means (Table 4 also gives the means over the 11 concepts that all four main arms edit). Without the Alpaca judge, the selection score of Section 2.2 is $\operatorname{HM}(\text{efficacy},\text{specificity})$.

New in Part 2:

- **Question coverage.** For a language and an edit, the share of the 50 `QA_test` questions whose question or correct answer contains a token that the edit changes. The coverage gap is the English coverage minus the coverage in the language. We use coverage because the embedding edit can act on a question only through the tokens it contains.
- **Touched-question loss.** Among the questions that the unedited model answers correctly, the share the edit turns wrong, reported separately for touched and untouched questions. This isolates the effect of the edit on the questions it reaches from its effect on all others.
- **Collateral damage.** The change in `SimdomQA_test` accuracy (edited minus unedited, negative = damage) in each language, and MMLU on the 1,000-question test split, as in Part 1; for the added tokens also the share of ordinary Wikipedia sentences that contain an edited token (exposure).
- **Edit size.** The norm of a token's edit relative to the norm of its embedding, used to compare the strengths of different methods.

Method differences are tested pairwise over concepts. We report the mean paired difference with a 95% interval from a two-level bootstrap (resampling the concepts and, within each cell, the 50 questions, paired across arms), a Wilcoxon signed-rank test over concepts, and Holm's correction over the six primary comparisons. Coverage and leak are related by Spearman's rank correlation. As in Part 1, the unit of analysis is the concept, because the questions of one concept are not independent.

---

## 4. Results

### 4.1 Fitting the features on foreign text does not help the embedding track

The most direct way to cover foreign tokens is to find the features on foreign sentences. We tried two arms. A joint fit factorizes the English, Russian, German, Spanish and French sentences together (World War II and Cannabis). A per-language fit factorizes each language alone and takes the union of the edits, each token edited once (World War II, Cannabis and Gun).

**Table 2.** Embedding and MLP features kept for World War II (candidates → kept), rank 100, seed 42, judge gemini-3.5-flash-lite.

| Fit | Embedding | MLP |
|---|---|---|
| English (the paper's fit, re-judged) | 45 → 1 | 150 → 12 |
| Joint (en+ru+de+es+fr) | 24 → 0 | 53 → 3 |
| Russian alone | 20 → 0 | 55 → 0 |
| German alone | 15 → 0 | 47 → 4 |
| Spanish alone | 18 → 0 | 62 → 2 |
| French alone | 22 → 0 | 51 → 3 |

At rank 100 no fit outside English keeps an embedding feature for World War II. The joint fit keeps none at any of the ranks 100 to 800 we tried, and the German, Spanish and French fits keep one at some higher ranks, through tokens shared with English (Hitler, Mussolini, Stalin, Auschwitz), but the Russian fit never does. For Cannabis the joint fit finds one multilingual feature from rank 400 on, with only 4% of its concept mass in Russian. The reason is how the embedding track works. It factorizes the embedding rows of the distinct tokens that occur in the sentences, and the text only decides which tokens enter the matrix and how each is labeled (concept-only, neutral, or both). English concept words are whole tokens that occur only in concept sentences (Hitler, Nazi, Führer, cannabis, THC). Russian words are split into short pieces that also occur in unrelated text, so they are labeled "both" or neutral and carry no concept meaning of their own: Гитлер is Г + ит + лер, all three labeled "both", and 30% of the Russian tokens of World War II are labeled "both" against 13% of the English ones. No rank or amount of text can build a feature for Hitler out of Г, ит and лер. The English features also separate native foreign concept words only poorly (AUC 0.41–0.64, close to random directions of the same fit). The per-language arm raises the coverage of German, Spanish and French questions for World War II and Gun, and Russian only where whole-word tokens exist (Gun: оружие). On the three concepts it does not appreciably reduce the leak: its mean leak is +0.18 against +0.20 for EMBER and +0.13 for EMBER+X (Appendix A). We therefore keep the English features and extend only the token set.

### 4.2 CINDER in English: no better than EMBER

On the six first-stage concepts, CINDER finds a feature for four. Table 3 gives the edits and the held-out result with Part 1's protocol.

**Table 3.** First stage, six concepts, Part 1's protocol. Features / edited tokens / δ, English retention $r_{\text{en}}$ and mean leak (held-out `QA_test`; lower retention = more erased).

| Concept | EMBER edit | $r_{\text{en}}$ | leak | CINDER edit | $r_{\text{en}}$ | leak |
|---|---|---|---|---|---|---|
| Cannabis | 1 / 45 / 200 | 0.39 | +0.14 | 1 / 34 / 5 | 0.59 | +0.12 |
| Gun | 2 / 61 / 5 | 0.54 | +0.42 | 2 / 67 / 100 | 0.68 | +0.26 |
| Culture of Greece | 1 / 52 / 200 | 0.73 | +0.13 | 1 / 106 / 100 | 0.46 | +0.37 |
| Golf | 2 / 70 / 10 | 0.75 | +0.06 | 1 / 44 / 10 | 0.89 | −0.03 |
| Nazism | 1 / 41 / 100 | 0.63 | +0.14 | no feature | 1.00 | 0.00 |
| Republic of Ireland | 1 / 31 / 100 | 0.67 | +0.23 | no feature | 1.00 | 0.00 |
| Mean, the four CINDER edits | | 0.60 | +0.19 | | 0.66 | +0.18 |

CINDER erases English less than EMBER on average (retention 0.66 against 0.60 on the four concepts it edits). Its selection score is lower on all four (for example Cannabis 0.434 against 0.568, Culture of Greece 0.822 against 0.938), and it makes no edit for Nazism and Republic of Ireland. It erases English more only for Culture of Greece (0.46 against 0.73), and there it leaks more (+0.37 against +0.13). The mean leak is the same (+0.18 against +0.19). Equal strength gives the same picture. At the same relative edit size, CINDER does not erase English better on the train questions: the two are about equal for Gun, CINDER erases less for Cannabis and Golf, and for Culture of Greece it erases more at small edits and less at large ones. CINDER's wider features add neighbouring topics: the tokens that only CINDER edits include Roman, Rome and Italy for Culture of Greece, and baseball and ball for Golf. In ordinary Wikipedia text its Culture of Greece tokens occur in 2.8% of the sentences against 0.56% for EMBER's. The two edit directions agree on the tokens that both edit (median cosine 0.64–0.83).

![[fig2_english_retention.png|380]]

**Figure 2.** English retention of CINDER against EMBER for all 18 concepts of the fair comparison (lower = more erased; points on the diagonal are equal). Open circles: one of the two methods makes no edit, so its retention is 1.

### 4.3 The edit changes only the questions it reaches

Because the edit acts through tokens, a foreign question should lose its answer exactly when it contains an edited token. We tested this on the held-out questions at the selected δ.

![[fig3_touched_untouched.png|560]]

**Figure 3.** Share of the correct answers that the edit turns wrong, for questions that contain an edited token ("touched") and questions that do not, pooled over the first-stage concepts (EMBER six, CINDER four). Bars resting on fewer than 10 correct answers (Russian touched questions) are omitted.

The edit turns 37–46% of the correct answers wrong when the question contains an edited token (EMBER; CINDER 30–32%), and 0–5% when it does not, in every language (Figure 3). The same holds on the English train questions at every δ, where untouched questions do not change at all. The edit therefore removes words, not knowledge, and the cross-lingual leak is a coverage effect. This also explains why the leak depends on the language. Averaged over concepts, the edit reaches 74% of the English test questions (Part 1 EMBER), but only 38% of the German, 34% of the Spanish, 39% of the French and 4% of the Russian ones.

![[fig4_coverage_leak.png|420]]

**Figure 4.** Leak in a language against the coverage of its questions, for the 296 edited (concept, foreign language) cells of all five arms. Dark line: mean leak in coverage bins.

Over all edited cells the rank correlation between coverage and leak is −0.58, and it is between −0.35 and −0.77 within each arm (Figure 4). Where less than 5% of the questions contain an edited token the mean leak is +0.30, and above 40% coverage it is about +0.05. Concept examples agree: for Cannabis the German, Spanish and French coverage is 0.86–0.90 (the shared tokens Cannabis, THC and CBD) and the leak is about 0, whereas Russian coverage is 0.08 and the leak +0.51.

### 4.4 Extrapolation raises coverage and lowers the leak

Figure 5 shows the effect of extrapolation on coverage. Averaged over the concepts that each arm edits, the share of reached questions rises from 0.33 to 0.51 in German, from 0.32 to 0.59 in Spanish and from 0.34 to 0.60 in French for EMBER, and from 0.05 to 0.28 in Russian. CINDER gains similarly (Russian 0.06 to 0.37). For Gun, for example, the coverage of Spanish questions rises from 0.08 to 0.62 and that of Russian questions from 0.00 to 0.60. The added tokens are mostly right (Griechenland, griego, Grèce, Гре… for Culture of Greece; pistola, fusil, pistolet, оружие for Gun), with some noise (Appendix B).

![[fig5_extrapolation_coverage.png|480]]

**Figure 5.** Question coverage by language for each arm (mean over the concepts the arm edits).

Coverage translates into erasure, and the extra erasure sits where the added tokens reach. For foreign questions that contain only an added token, the share of correct answers that are lost rises from 0.5% to 36.7% for EMBER+X and from 0.5% to 23.7% for CINDER+X. Questions with an original token change by at most one point (44.2% to 45.3% and 29.4% to 29.3%), and questions with no edited token hardly change at all (0.6% to 1.1% and 0.1% to 0.6%).

![[fig6_extrapolation_mechanism.png|560]]

**Figure 6.** Foreign test questions answered wrong among those the unedited model answers correctly, with the parent edit alone and with extrapolation, by the kind of edited token the question contains.

On the six first-stage concepts and their train questions, real erasure with the added tokens at the same δ lowers the mean foreign retention from 0.81 to 0.68 and the leak from 0.18 to 0.10, and EMBER+X erases at least as much foreign knowledge as EMBER in all six concepts, at both strengths we tried. The largest gains are Culture of Greece in German (0.81 to 0.36) and Republic of Ireland in French (0.82 to 0.09); the smallest are Gun at its weak strength δ = 5 (0.99 to 0.96) and Golf. Two filters, dropping tokens that also occur in English and dropping noisy tokens, did not beat the plain extrapolated set (differences below one question).

Two controls test whether the tokens must be chosen by the features. Without features, a frequency rule that picks the concept-only tokens most frequent in the concept sentences reaches somewhat more foreign questions (coverage 0.69 against 0.61) but with 2–2.5 times the similar-domain collateral and 4–5 times the background exposure; the two sets share only 5–23% of their tokens, since the frequency rule picks subword fragments that are concept-only only because they are missing from 300 neutral sentences. Real erasure (Figure 7) shows that random tokens barely help (foreign retention 0.77 against 0.81 for EMBER), whereas the frequency rule erases as much as the feature-based tokens (0.65 against 0.68) at a higher collateral cost (similar-domain change −0.033 against −0.022 in the foreign languages). Token choice matters, the features are not the only way to make a good choice, and they buy less collateral damage. These controls use six concepts and the train questions, so we treat them as descriptive.

![[fig7_token_control.png|560]]

**Figure 7.** Real erasure at the same δ with EMBER's original edit plus 96 added tokens on average: feature-based (X), chosen by sentence frequency (F_own) or at random (R_own). (a) Mean foreign retention. (b) Change of foreign similar-domain accuracy. Bars: mean over six concepts; dots: single concepts (train questions).

### 4.5 Fair comparison on 18 concepts

Table 4 and Figures 8 and 9 give the held-out result for all arms on all 18 concepts.

**Table 4.** Held-out `QA_test` retention by language (mean over concepts; the unedited model is 1) and mean leak with a 95% bootstrap interval over concepts. Upper block: all 18 concepts, a concept without an edit counted as unedited (EMBER and EMBER+X make no edit for four, CINDER and CINDER+X for three). Lower block: the 11 concepts that EMBER, CINDER, EMBER+X and CINDER+X all edit.

| Arm | en | de | es | fr | ru | Mean leak |
|---|---|---|---|---|---|---|
| EMBER | 0.75 | 0.82 | 0.85 | 0.82 | 0.96 | +0.10 [+0.05, +0.16] |
| CINDER | 0.74 | 0.88 | 0.87 | 0.85 | 0.97 | +0.15 [+0.08, +0.21] |
| EMBER+TWJ | 0.67 | 0.77 | 0.81 | 0.80 | 0.96 | +0.15 [+0.10, +0.21] |
| EMBER+X | 0.74 | 0.75 | 0.75 | 0.77 | 0.91 | +0.05 [−0.00, +0.10] |
| CINDER+X | 0.73 | 0.85 | 0.80 | 0.80 | 0.92 | +0.11 [+0.05, +0.17] |
| *11 common concepts* | | | | | | |
| EMBER | 0.67 | 0.77 | 0.80 | 0.76 | 0.93 | +0.14 [+0.06, +0.21] |
| CINDER | 0.71 | 0.86 | 0.85 | 0.81 | 0.95 | +0.15 [+0.06, +0.24] |
| EMBER+X | 0.65 | 0.68 | 0.68 | 0.68 | 0.85 | +0.07 [−0.01, +0.15] |
| CINDER+X | 0.69 | 0.80 | 0.77 | 0.73 | 0.88 | +0.10 [+0.03, +0.20] |

![[fig8_retention_by_language.png|520]]

**Figure 8.** Concept retention by question language, mean over 18 concepts with 95% bootstrap intervals (16 concepts for Russian). Filled: parent methods; hollow: with extrapolation. Every edit was selected in English only.

![[fig9_paired_leak.png|520]]

**Figure 9.** Paired differences in mean leak between arms (first minus second), with the two-level 95% interval; labels give the number of concepts with lower and higher leak. Blue: the difference survives Holm's correction.

**CINDER against EMBER.** The two methods do not differ. English retention is 0.74 against 0.75 (paired difference −0.013, Wilcoxon p = 0.91; lower in 7 and higher in 11 of 18 concepts), and the mean leak differs by +0.04 with an interval that contains zero ([−0.06, +0.14], lower in 9 and higher in 9 concepts, Holm p = 1.0). On the 12 concepts that CINDER's authors neither used to tune it nor as lexicon-prompt examples the result is the same (leak +0.05, p = 0.42). The sharpening and the global sparsity therefore bring no measurable gain in either English erasure or leak.

**The judge is what helps.** EMBER+TWJ keeps the same edit as EMBER for 14 concepts and adds an edit for the four on which EMBER's judge selects nothing: Halloween, Heroin, Uranium and Valentine's Day. It lowers the mean English retention from 0.75 to 0.67 (all four added concepts are erased, with retention 0.54–0.81) and raises the mean leak by +0.05 ([−0.01, +0.12]), because the concepts that were not erased had zero leak. Compared with EMBER+TWJ, CINDER erases English a little less (+0.067, p = 0.08) and leaks the same (−0.01, [−0.09, +0.08]). CINDER's judge rescues the concepts for which EMBER's judge fails, and in our runs CINDER's factorization does not improve on EMBER's once the judge is the same.

**Extrapolation reduces the leak.** EMBER+X leaks less than EMBER by 0.059 ([−0.102, −0.023], lower in 12 and higher in 2 concepts, Wilcoxon p = 0.001, Holm p = 0.007); this is the only paired difference that survives the correction. CINDER+X leaks less than CINDER by 0.039 ([−0.088, −0.004], lower in 10 and higher in 4, p = 0.02), which does not survive the correction (Holm p = 0.10). The same holds when retention is not clipped at 0 and 1 (EMBER+X: −0.055, Holm p = 0.014; CINDER+X: −0.044, Holm p = 0.18). English retention is unchanged within noise (−0.011, p = 0.31). EMBER+X and CINDER+X do not differ (+0.06, [−0.04, +0.15], Holm p = 0.43); when only German, Spanish and French are used, CINDER+X leaks more (+0.07, p = 0.02), which does not survive the correction either.

**What remains.** Extrapolation closes only part of the gap. After EMBER+X, 77% of the foreign questions that the unedited model answers correctly are still answered correctly (German 360 of 480, Spanish 348 of 472, French 337 of 450, Russian 319 of 358), and about 60% of them contain no edited token. The failures are mostly answers that are numbers or dates (1955, 1969–1984, 17 марта) and names that are split into pieces (St·aling·rad, R·ube·us Hag·rid, Б·роне·бой·ный). Russian remains the hardest language: its retention stays 0.91–0.92 and its coverage rises only to 0.28–0.37. Per-concept values are in Appendix A.

### 4.6 Collateral damage

**Table 5.** Collateral damage of the selected edits, mean over all 18 concepts (concepts without an edit count as unedited). `SimdomQA_test` is the change in accuracy against the unedited model (negative = damage); MMLU on the 1,000-question test split (unedited 0.553).

| Arm | SimDom, English | SimDom, foreign mean | MMLU |
|---|---|---|---|
| EMBER | −0.024 | −0.006 | 0.550 |
| CINDER | −0.020 | −0.006 | 0.551 |
| EMBER+TWJ | −0.027 | −0.006 | 0.550 |
| EMBER+X | −0.033 | −0.015 | 0.550 |
| CINDER+X | −0.024 | −0.008 | 0.551 |

All arms leave MMLU at 0.550–0.551 against 0.553 and similar-domain accuracy within a few points of the unedited model. Extrapolation costs a little: for EMBER the similar-domain change more than doubles in the foreign languages (−0.006 to −0.015) and grows by about one point in English, while for CINDER it is within 0.005. On the six first-stage concepts and the train questions the same cost is −0.007 to −0.022 in the foreign languages.

---

## 5. Discussion

**The leak is a coverage problem, not a feature-quality problem.** In every analysis the edit acts only on the questions that contain an edited token, and how much of the concept survives in a language follows how many questions are reached. A better feature finder cannot change this, and CINDER indeed neither erases English better nor leaks less than EMBER. A plausible reason is that the sharpening aims at making concept features cleaner, which is a different quantity from how many of the tokens that a question uses are edited. This fits the authors' own erasure ablation, in which the sharpening lowered erasure, and it suggests that the part of CINDER that matters for erasure is its judge, which finds a feature where EMBER's finds none (Heroin, Uranium, Halloween, Valentine's Day).

**Extrapolation addresses the actual mechanism.** Adding native tokens raises coverage in the foreign languages (Russian from 0.05 to 0.28) and the extra erasure appears in the questions that the new tokens reach, so the reduction of the leak is explained by the mechanism and not by a general weakening of the edit. The effect is moderate (−0.06 for EMBER) because many questions are answered through tokens that no edit reaches, such as numbers, dates and names split into pieces, and because the added tokens are noisy. The token controls show that the choice of tokens matters and that the features are not the only good rule, but they buy lower collateral damage.

**Fitting on foreign text is the wrong lever for embeddings.** The embedding features are built from token rows and not from the text, so foreign text can only choose which rows enter the factorization, and the pieces of foreign words carry no concept meaning. Extrapolation keeps the English features, which are well defined, and uses foreign text only to decide which tokens load on them.

**Practical implication.** An embedding edit should be checked for question coverage in each target language, and a language with a different script needs explicit coverage of its tokens. Extrapolation is a cheap way to obtain it: it needs no new fit and no new judge calls, only native text.

### Limitations

1. **One model, one seed.** All results are for Gemma-2-2B-it with seed 42, and feature selection can vary with the seed.
2. **Concepts without an edit.** Whether a concept is edited depends on the judge. Means that include unedited concepts mix the quality of the edit with how often features are found; we therefore also report the 11 concepts that all four main arms edit.
3. **CINDER was tuned on a few concepts.** Its parameters (η, γ, the lexicon threshold) were chosen on three concepts and the lexicon prompt uses four example concepts; the 12-concept analysis excludes them. Its sharpening target is the concept's Wikipedia page, which also supplies some of the tokens that its feature-level metrics score.
4. **The judge is a different model than in the paper.** All arms use gemini-3.5-flash-lite, and EMBER+TWJ was added after the main comparison had been decided, so it is a post-hoc arm.
5. **Extrapolation is not query-only.** It uses native foreign text to choose tokens, and its δ was not re-tuned. Some added tokens are noise and some occur in English questions.
6. **The controls are small.** The extrapolation variants and token controls use six concepts and the train questions and are descriptive; the held-out test is used only to confirm the final arms.
7. **Multiple choice, machine-translated questions, 50 questions per cell**, as in Part 1, and no relearning test.
8. **Only the paired difference EMBER+X − EMBER survives correction for multiple comparisons.** The effects of CINDER+X and of the judge are suggestive.

---

## 6. Conclusion

An embedding edit acts only on the questions that contain an edited token, so an erasure fitted on English leaks into the languages whose questions use other tokens. A better feature finder, CINDER, does not change this: on 18 concepts it neither erases English better nor leaks less than EMBER, and the part of it that helps is its judge, which finds features for four concepts where EMBER's judge finds none. Fitting the features on foreign text does not help the embedding track, because the pieces of foreign words carry no concept meaning. Extending the edit to native tokens by extrapolation, with English features, raises the coverage of foreign questions and lowers EMBER's leak by 0.06 (Holm p = 0.007) at a small cost in similar-domain accuracy, while leaving English erasure unchanged. About three quarters of the foreign questions are still answered correctly after the extension, so complete cross-lingual erasure remains open.

---

## Appendix A. Per-concept results of the fair comparison

Notation as in Section 2. For the +X arms the number of edited tokens includes the parent's tokens, and the features and δ are the parent's.

**Table A1.** Selected edits: number of features / number of edited tokens / strength δ. "—": no feature kept, so no edit.

| Concept | EMBER | CINDER | EMBER+TWJ | EMBER+X (tokens) | CINDER+X (tokens) |
|---|---|---|---|---|---|
| Ancient Rome | 2 / 66 / 10 | 1 / 92 / 5 | 2 / 66 / 10 | 191 | 200 |
| Artificial intelligence | 1 / 39 / 50 | 1 / 31 / 2 | 1 / 39 / 50 | 149 | 109 |
| Baseball | 1 / 39 / 200 | 3 / 61 / 50 | 1 / 39 / 200 | 93 | 158 |
| COVID-19 pandemic | 1 / 42 / 50 | 1 / 63 / 2 | 1 / 42 / 50 | 138 | 158 |
| Cannabis | 1 / 45 / 200 | 1 / 34 / 5 | 1 / 45 / 200 | 120 | 70 |
| Culture of Greece | 1 / 52 / 200 | 1 / 106 / 100 | 1 / 52 / 200 | 151 | 231 |
| Gambling | 1 / 41 / 10 | 2 / 56 / 2 | 1 / 41 / 10 | 115 | 139 |
| Golf | 2 / 70 / 10 | 1 / 44 / 10 | 2 / 70 / 10 | 206 | 108 |
| Gun | 2 / 61 / 5 | 2 / 67 / 100 | 2 / 61 / 5 | 191 | 166 |
| Halloween | — | 1 / 76 / 10 | 1 / 41 / 5 | — | 178 |
| Harry Potter | 1 / 39 / 10 | 1 / 51 / 2 | 1 / 39 / 10 | 100 | 105 |
| Heroin | — | 1 / 86 / 50 | 1 / 46 / 100 | — | 187 |
| Nazism | 1 / 41 / 100 | — | 1 / 41 / 100 | 122 | — |
| Pornography | 1 / 43 / 5 | 1 / 69 / 5 | 1 / 43 / 5 | 143 | 167 |
| Republic of Ireland | 1 / 31 / 100 | — | 1 / 31 / 100 | 88 | — |
| Uranium | — | 1 / 52 / 200 | 1 / 38 / 10 | — | 120 |
| Valentine's Day | — | 1 / 52 / 50 | 1 / 38 / 2 | — | 146 |
| World War II | 1 / 36 / 5 | — | 1 / 36 / 5 | 101 | — |

**Table A2.** English retention / mean leak (over the reliable languages among de, es, fr, ru) on the held-out `QA_test`. Retention 1.00 and leak 0.00 for concepts without an edit.

| Concept | EMBER | CINDER | EMBER+TWJ | EMBER+X | CINDER+X |
|---|---|---|---|---|---|
| Ancient Rome | 0.78 / +0.16 | 0.82 / +0.06 | 0.78 / +0.16 | 0.82 / +0.07 | 0.78 / −0.01 |
| Artificial intelligence | 0.78 / +0.20 | 0.89 / +0.06 | 0.78 / +0.20 | 0.71 / +0.09 | 0.89 / +0.02 |
| Baseball | 0.71 / +0.17 | 0.61 / +0.26 | 0.71 / +0.17 | 0.66 / +0.13 | 0.56 / +0.15 |
| COVID-19 pandemic | 0.69 / −0.10 | 0.90 / −0.03 | 0.69 / −0.10 | 0.69 / −0.12 | 0.90 / −0.01 |
| Cannabis | 0.39 / +0.14 | 0.59 / +0.12 | 0.39 / +0.14 | 0.33 / +0.07 | 0.56 / +0.13 |
| Culture of Greece | 0.73 / +0.13 | 0.46 / +0.37 | 0.73 / +0.13 | 0.73 / −0.06 | 0.46 / +0.14 |
| Gambling | 0.66 / +0.23 | 0.49 / +0.42 | 0.66 / +0.23 | 0.59 / +0.23 | 0.42 / +0.48 |
| Golf | 0.74 / +0.06 | 0.89 / −0.03 | 0.74 / +0.06 | 0.64 / +0.09 | 0.82 / +0.03 |
| Gun | 0.54 / +0.42 | 0.68 / +0.26 | 0.54 / +0.42 | 0.54 / +0.32 | 0.65 / +0.18 |
| Halloween | 1.00 / 0.00 | 0.54 / +0.26 | 0.54 / +0.28 | 1.00 / 0.00 | 0.54 / +0.26 |
| Harry Potter | 0.67 / −0.04 | 0.78 / −0.05 | 0.67 / −0.04 | 0.67 / −0.14 | 0.78 / −0.07 |
| Heroin | 1.00 / 0.00 | 0.63 / +0.30 | 0.58 / +0.35 | 1.00 / 0.00 | 0.63 / +0.22 |
| Nazism | 0.63 / +0.14 | 1.00 / 0.00 | 0.63 / +0.14 | 0.71 / +0.02 | 1.00 / 0.00 |
| Pornography | 0.68 / +0.14 | 0.72 / +0.19 | 0.68 / +0.14 | 0.72 / +0.07 | 0.75 / +0.08 |
| Republic of Ireland | 0.67 / +0.23 | 1.00 / 0.00 | 0.67 / +0.23 | 0.67 / +0.07 | 1.00 / 0.00 |
| Uranium | 1.00 / 0.00 | 0.62 / +0.22 | 0.62 / +0.16 | 1.00 / 0.00 | 0.68 / +0.18 |
| Valentine's Day | 1.00 / 0.00 | 0.71 / +0.22 | 0.81 / +0.08 | 1.00 / 0.00 | 0.71 / +0.15 |
| World War II | 0.88 / +0.03 | 1.00 / 0.00 | 0.88 / +0.03 | 0.88 / −0.01 | 1.00 / 0.00 |

Single cells are noisy: each has 50 questions, so only the averages over concepts in Section 4.5 support conclusions. The per-language arm of Section 4.1 (union of per-language EMBER fits) was run for three concepts: leak +0.16 (Cannabis), +0.34 (Gun), +0.03 (World War II), against +0.14, +0.42, +0.03 for EMBER and +0.07, +0.32, −0.01 for EMBER+X.

## Appendix B. Tokens that extrapolation adds

For the 12 concepts outside the first stage, extrapolation adds 54–125 tokens per concept beyond the parent's (Table A1), 52–83% of them whole words and 85–94% of them never occurring in English text. Most are on topic (Casinos, cuarentena, mascarilla, Neutron, Plut, römischen), and some are clear noise (EMBER+X for Harry Potter: SHIELD, Nintendo, Wii, Prada, Hitler; for Pornography: Poverty, Punk, Emo). First added tokens of the first-stage concepts, EMBER+X:

| Concept | Added tokens (first 14) |
|---|---|
| Cannabis | ▁Canna cannabis ▁Hemp ▁weed nabis ▁vape ▁CBN ▁Endoc ▁edibles ▁кан ▁Kush CBC ▁Bong ioid |
| Gun | ▁Guns ▁pistola ▁rifles ▁pistolet ▁Weapons ▁gunner Pist ▁Weapon pist ▁disparos ▁disparo ▁fusil ▁оружие ▁Pist |
| Culture of Greece | ▁greek ▁griech ▁griega ▁griechischen ▁griego ▁Grèce ▁Griechenland ▁Griechen ▁grecque ▁Atenas ▁grec ▁Grecs ▁greco ▁Aristote |
| Republic of Ireland | Cork ▁Kilkenny ▁Sligo ▁Leinster Irlande ▁Wexford ▁Derry ▁Hiber ▁Munster ▁Waterford ▁GAA ▁Tipper ▁Dund ▁Yeats |
| Golf | ▁Swing Swing ▁golfing ▁Handic Hole ▁Hole ▁Penalty ▁golfe ▁Tee ▁Gol ▁penal Gol gol ▁caddy |

## Appendix C. Reproducibility

| What | Where |
|---|---|
| Fair comparison, all arms and concepts | `judge_rerun/Fair/report.csv`, `report.md`; code `cinder/fair.py` |
| CINDER features and edits | `judge_rerun/Cinder/`, code `cinder/`, `experiments/varwidth/`, `experiments/whitening/` |
| Touched/untouched, coverage, extrapolation, controls (X1–X13) | `Part 2 Experiments/` (one folder and `report.md` per analysis; scripts in `scripts/`) |
| Foreign-text fits and rank sweeps | `docs/Part 2/plan.md`, `docs/Part 2/multilingual_features.md` |
| Figures in this report | `report/Part2_report/make_figures.py` → `figures/` |

Settings: Gemma-2-2B-it, bfloat16, greedy decoding; rank 100, seed 42; feature judge gemini-3.5-flash-lite; δ grid 0.5–200; reliability cutoff 0.40; two-level bootstrap with 4,000 resamples and Holm over six comparisons for the paired tests; bootstrap `numpy.random.default_rng(0)`, 10,000 resamples for the intervals over concepts in Table 4 and Figure 8. The means in Table 4 use the reliable cells only and can differ by 0.01 from the unfiltered means in `judge_rerun/Fair/report.md`.

To regenerate the figures (needs matplotlib, numpy, pandas):

```bash
python report/Part2_report/make_figures.py
```
