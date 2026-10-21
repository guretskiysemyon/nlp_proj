# Better Features, More Tokens: Reducing the Cross-Lingual Leakage of Embedding Erasure

**Part 2 report: CINDER and token extrapolation on Gemma-2-2B-it**

Mary Arav, Ori Petel, Semyon Guretskiy (Tel Aviv University, NLP course 2025/26)

---

## 1. Methods

### 1.1 Research questions

Part 1 showed that an erasure tuned in English leaks across languages. After EMBER's edit, the concept survives better in German, Spanish, French and especially Russian than in English (mean leak +0.19, Russian +0.33). Two explanations are plausible. The edit may be too weak: EMBER removes less than half of the concept even in English (efficacy 0.43), and with the judge model now available to us its feature search sometimes finds no concept feature at all. Or the edit may simply not reach the tokens that foreign questions use. We ask three questions.

1. Does a better feature-finding method, CINDER, improve English erasure and reduce leakage?
2. Can the edit be extended to foreign tokens without training on foreign text (extrapolation)?
3. Does fitting the features on foreign text help?

All methods keep EMBER's edit step. Features are found on concept and neutral sentences, a judge accepts the features that belong to the concept, and the concept component of each selected token is subtracted from its embedding with strength δ (Part 1, Section 2.2). The methods differ in how the features are found and in which tokens are edited (Figure 1).

![[fig1_methods.png|720]]

**Figure 1.** The methods compared. EMBER (top) is the Part 1 method. CINDER (middle) changes the first three steps and keeps the edit. Extrapolation (bottom, "+X") is added to either method and changes only the set of edited tokens.

### 1.2 CINDER: contrastive, adaptive and verified features

CINDER (Contrastive Informed Directional Embedding Removal) was developed in our team as a repair of three weak points of EMBER's feature search that we also observe in our reproduction: concept features are weak because a concept has little variance in the embedding matrix, features have a fixed width that does not fit the concept, and the judge rejects correct features. Each of its three layers targets one of them, and the edit step is unchanged. We evaluate CINDER end to end against EMBER.

**Contrastive Sharpening (CS).** Sparse matrix factorization spends its features on the directions with the most variance, and in the embeddings of concept and neutral tokens those are generic topics (for Harry Potter: magic, writing, books). A concept has few tokens, so its direction is weak. CS first whitens every token embedding against a large background of English Wikipedia, with mean $\mu_B$ and covariance $\Sigma_B$, so that ordinary directions have unit variance:

$$\tilde u = W(e-\mu_B),\qquad W=(\Sigma_B+\epsilon I)^{-1/2}.$$

It then stretches the directions in which the concept's own Wikipedia page differs from the background, with the second moment $S_\eta=(1-\eta)\frac1m\sum_j\tilde u_j\tilde u_j^\top+\eta I$ of the page tokens, giving $x=S_\eta^{\gamma/2}\tilde u$. The factorization is run on $x$, and each concept token's component is mapped back exactly to the embedding space before it is subtracted. We use the defaults $\eta=0.05$ and $\gamma=1$. The idea is that of contrastive PCA, which finds directions enriched in a target set relative to a background.

**Global Sparsity (GS).** EMBER keeps the top 1% of tokens in every feature, about 48 tokens. A broad concept such as Culture of Greece is cut short and a narrow one such as Baseball is padded with unrelated tokens. GS keeps EMBER's total budget (1% of all entries of the coefficient matrix) but shares it across the whole matrix. Each entry is scored by its share of its own feature, $|y_{ij}|/\lVert y_i\rVert_2$, the top entries of the whole matrix are kept, and every feature keeps between 0.2% and 3% of the tokens. The width of a feature thus becomes an output of the fit.

**Two-Way Judge (TWJ).** EMBER's judge works bottom-up: it summarizes the feature's top tokens in one sentence and decides whether the summary is the concept, so anything the summary abstracts away is lost (in our runs "heroin" became "addictive drugs" and World War II became "20th-century totalitarian regimes", and both were rejected). TWJ adds a top-down check. One LLM call expands the concept into a lexicon of 20 specific words, and a feature is also accepted if at least five of its own top 20 source words appear in the lexicon. The result is the OR of the two tests, TWJ = TJ ∨ EJ@5, where TJ is EMBER's judge unchanged.

### 1.3 Extrapolation to native tokens (+X)

EMBER edits only tokens that occur in the English sentences used for the fit. A foreign question contains other tokens, and a Russian word is typically split into several short pieces (the name Гитлер becomes Г + ит + лер) that are not edited. Extrapolation extends the edit to such tokens without refitting the features and without any change to the English edit.

The features $Z\in\mathbb R^{d\times k}$ of the English fit are kept fixed. For every token $t$ that occurs in native German, Spanish, French or Russian concept and neutral sentences (Wikipedia, up to 300 sentences per concept and language) but not in the English token set, the coefficients on the fixed features are solved by ridge regression,

$$y_t=(Z^\top Z+\lambda I)^{-1}Z^\top e_t,$$

and then sparsified with the method's own rule (EMBER: the top 1% of tokens per feature; CINDER: the global budget in its whitened space, over the union of all tokens). A token is added to the edit if it is concept-only, that is, it occurs in the concept sentences of some language and in no neutral sentence of any of the five languages, and if it loads on a selected concept feature. The added tokens are edited with the same formula as the original ones, $e_t\leftarrow e_t-\delta\sum_{i\in F_C}y_{i,t}z_i$. The original English rows and the strength δ are kept exactly. Extrapolation uses foreign native text to choose the added tokens, so it is not a pure query-only method; the features themselves stay English-only.

### 1.4 Fitting the features on native text

The direct alternative to extrapolation is to find the features on foreign sentences. The per-language arm refits EMBER on the native sentences of each language alone (rank 100, the same judge) and takes the union of the five edits. A token that the English fit edits is edited exactly as in EMBER, and every other token is edited by the foreign fit in which its edit has the largest norm, so each token is edited once and EMBER's English edit is kept. The joint arm factorizes the sentences of all five languages together, at rank 500 because five languages need more features than one, and uses that fit's own features and edit. Both arms have their own strength search.

### 1.5 Arms

**Table 1.** The arms of the comparison. All use the same model, feature judge model, strength grid, selection rule and tests. The first five arms use English concept and neutral sentences for the features; the two native-text arms use native sentences.

| Arm | Features | Judge | Edited tokens |
|---|---|---|---|
| EMBER | EMBER's factorization | EMBER's two-stage judge (TJ) | English tokens |
| CINDER | CS + GS factorization | TWJ = TJ ∨ EJ@5 | English tokens |
| EMBER+TWJ | EMBER's factorization | TWJ | English tokens |
| EMBER+X | as EMBER | as EMBER | English tokens + extrapolated foreign tokens |
| CINDER+X | as CINDER | as CINDER | English tokens + extrapolated foreign tokens |
| Native per-language | five EMBER factorizations, one per language, on native sentences (rank 100) | EMBER's TJ | union of the five edits |
| Native joint | one EMBER factorization of all five languages (rank 500) | EMBER's TJ | the fit's own edit |

EMBER+TWJ separates the contribution of CINDER's judge from that of its sharpening and sparsity, since it uses EMBER's factorization unchanged. It is built from the same candidate features: it keeps EMBER's verdicts and adds a feature only where the lexicon test accepts one. The +X arms use the parent's features and strength.

---

## 2. Experimental setup

### 2.1 Model, data and feature judge

We use Gemma-2-2B-it in bfloat16 with greedy decoding, rank $k=100$ and seed 42, and the data of Part 1: the 18 concepts of the EMBER paper, 50 + 50 four-option multiple-choice questions per concept for selection (`QA_train`) and held-out evaluation (`QA_test`), 50 + 50 questions on a similar topic (`SimdomQA`), MMLU for general knowledge, and the German, Spanish, French and Russian translations of the test questions. The English concept and neutral sentences (300 + 300 per concept) are those of the EMBER paper. The native sentences used by extrapolation and by the native-text fits are Wikipedia sentences from the concept's articles in each language, cleaned of back matter, with random sentences from the same language as the neutral set. Every arm is evaluated on all 18 concepts.

The paper's feature judge, Gemini-2.5-Flash-Lite, is no longer available, so every arm uses gemini-3.5-flash-lite. With it EMBER keeps the same features as in Part 1 for 10 of the 18 concepts, different features for four, and none for four (Halloween, Heroin, Uranium and Valentine's Day), so EMBER makes no edit for these four. CINDER keeps a feature for 15 concepts and none for Nazism, Republic of Ireland and World War II.

### 2.2 Selection of the strength and the final test

CINDER's edit is about half the size of EMBER's at the same δ (the median edit norm relative to the embedding norm is 0.24 against 0.46, a ratio of 1.93), which lies inside the band of 0.5 to 2 within which EMBER's grid still covers CINDER's useful range. All arms therefore use EMBER's grid, δ ∈ {0.5, 1, 2, 5, 10, 50, 100, 200}, and one more value is scored if the best δ is at the edge.

The strength is chosen in English, per concept and arm, by Part 1's judge-free screen: among the strengths whose MMLU fraction is at least 0.9, the one with the highest HM(efficacy, specificity) on `QA_train`, `SimdomQA_train` and the 50-question MMLU set, with ties broken towards the smaller δ. Unlike in Part 1 there is no Qwen coherence term, so that every arm runs through one cheap pipeline; the picks are therefore not directly comparable with Part 1's. Each +X arm keeps its parent's δ and has no search of its own; the two native-text arms have their own search. The edit is then applied once and evaluated on `QA_test` and `SimdomQA_test` in five languages and on the 1,000-question MMLU test split, with a check that the edited model reproduces its grid row. A concept without a kept feature gets no edit and is scored as the unedited model.

### 2.3 Mechanism analyses and controls

Three analyses explain the main comparison, and all use the question-level records of the held-out test. The touched/untouched analysis splits the questions that the unedited model answers correctly into those whose text or correct answer contains an edited token and those that do not, and compares how many answers each group loses. The coverage analysis relates the share of questions with an edited token in a language to the leak in that language. The mechanism of extrapolation is analysed in the same way, separating the questions that contain only an added token from those that contain an original one.

The token-choice controls of Section 4.4 apply real erasure with variants of the added tokens. Each variant is a separate set of model edits and evaluations, so they are limited to six concepts (Cannabis, Gun, Nazism, Republic of Ireland, Culture of Greece and Golf) that span strong, medium and weak EMBER erasure, and they use the train questions. They are descriptive; the held-out test split is used only for the main comparison.

---

## 3. Metrics

We use the metrics of Part 1: the chance-corrected ratio $\phi(a;a_0)=\operatorname{clip}\big((a-0.25)/(a_0-0.25),0,1\big)$, efficacy $=1-\phi(\text{QA}_{\text{train}})$, specificity $=\operatorname{HM}(\phi(\text{MMLU}),\phi(\text{SimDom}))$, and on the held-out `QA_test` the retention $r_L=\phi(\mathrm{acc}_L;\mathrm{acc}^{\text{unedited}}_L)$ and the leak $\Delta_L=r_L-r_{\text{en}}$ with the same reliability screen (a concept–language pair is dropped when the unedited accuracy is below 0.40; two of 90 pairs, Harry Potter and Valentine's Day in Russian). The mean leak averages a concept's reliable foreign languages and then the concepts. A concept without an edit has retention 1 in every language and leak 0, and is included in the means; Table 2 also gives the means over the 11 concepts that all four main arms edit. Without the Alpaca judge, the selection score of Section 2.2 is $\operatorname{HM}(\text{efficacy},\text{specificity})$.

New in Part 2:

- **Foreign retention.** The mean of $r_L$ over a concept's reliable de, es, fr and ru cells (lower = more erased), so that the leak is the foreign retention minus $r_{\text{en}}$. We report it next to the leak as a co-primary measure, because the leak alone penalizes an arm that also erases English more: a lower English retention raises the leak.
- **Question coverage.** For a language and an edit, the share of the 50 `QA_test` questions whose question or correct answer contains a token that the edit changes. We use coverage because the embedding edit can act on a question only through the tokens it contains.
- **Touched-question loss.** Among the questions that the unedited model answers correctly, the share the edit turns wrong, reported separately for touched and untouched questions. This isolates the effect of the edit on the questions it reaches from its effect on all others.
- **Collateral damage.** The change in `SimdomQA_test` accuracy (edited minus unedited; negative = damage) in each language, and MMLU on the 1,000-question test split, as in Part 1. The accuracy change is also split by whether the similar-domain question contains an edited token. The exposure of an edit is the share of other concepts' sentences (English and native) that contain an edited token.
- **Edit size.** The norm of a token's edit relative to the norm of its embedding, used to compare the strengths of different methods.

Method differences are tested pairwise over concepts. We report the mean paired difference with a 95% interval from a two-level bootstrap (resampling the concepts and, within each cell, the 50 questions, paired across arms), a Wilcoxon signed-rank test over concepts, and Holm's correction within each block of eight comparisons (the leak and the foreign retention form one block each). Each comparison is repeated on the concepts that both arms of a pair edit, so that it is not decided by concepts on which one arm makes no edit. Coverage and leak are related by Spearman's rank correlation. As in Part 1, the unit of analysis is the concept, because the questions of one concept are not independent.

---

## 4. Results

### 4.1 Main comparison

Table 2 and Figures 2 and 3 compare all arms on all 18 concepts. Four results stand out. CINDER does not differ from EMBER. Its judge, however, finds edits for concepts where EMBER's finds none. Extrapolation lowers the foreign retention and the leak for both methods. Fitting the features per language erases the most, in every language including English, and at a higher cost.

**Table 2.** Held-out `QA_test` retention by language (mean over concepts; the unedited model is 1), foreign retention (mean of the four foreign languages) and mean leak with a 95% bootstrap interval over concepts. Upper block: all 18 concepts, a concept without an edit counted as unedited (EMBER and EMBER+X make no edit for four concepts, CINDER and CINDER+X for three, the native joint arm for eleven and the native per-language arm for one). Lower block: the 11 concepts that EMBER, CINDER, EMBER+X and CINDER+X all edit.

| Arm | en | de | es | fr | ru | Foreign | Mean leak |
|---|---|---|---|---|---|---|---|
| EMBER | 0.75 | 0.82 | 0.85 | 0.82 | 0.96 | 0.86 | +0.10 [+0.05, +0.16] |
| CINDER | 0.74 | 0.88 | 0.87 | 0.85 | 0.97 | 0.89 | +0.15 [+0.08, +0.21] |
| EMBER+TWJ | 0.67 | 0.77 | 0.81 | 0.80 | 0.96 | 0.83 | +0.15 [+0.10, +0.21] |
| EMBER+X | 0.74 | 0.75 | 0.75 | 0.77 | 0.91 | 0.79 | +0.05 [−0.00, +0.10] |
| CINDER+X | 0.73 | 0.85 | 0.80 | 0.80 | 0.92 | 0.84 | +0.11 [+0.05, +0.17] |
| Native per-language | 0.59 | 0.59 | 0.70 | 0.60 | 0.90 | 0.69 | +0.10 [+0.04, +0.16] |
| Native joint | 0.85 | 0.91 | 0.89 | 0.89 | 0.94 | 0.91 | +0.06 [+0.01, +0.11] |
| *11 common concepts* | | | | | | | |
| EMBER | 0.67 | 0.77 | 0.80 | 0.76 | 0.93 | 0.81 | +0.14 [+0.06, +0.21] |
| CINDER | 0.71 | 0.86 | 0.85 | 0.81 | 0.95 | 0.86 | +0.15 [+0.06, +0.24] |
| EMBER+X | 0.65 | 0.68 | 0.68 | 0.68 | 0.85 | 0.71 | +0.07 [−0.01, +0.15] |
| CINDER+X | 0.69 | 0.80 | 0.77 | 0.73 | 0.88 | 0.79 | +0.10 [+0.03, +0.20] |
| Native per-language | 0.58 | 0.56 | 0.68 | 0.54 | 0.84 | 0.65 | +0.07 [−0.01, +0.15] |
| Native joint | 0.83 | 0.90 | 0.87 | 0.86 | 0.91 | 0.89 | +0.06 [−0.00, +0.13] |

![[fig2_retention_by_language.png|520]]

**Figure 2.** Concept retention by question language for the four main arms, mean over 18 concepts with 95% bootstrap intervals (16 concepts for Russian). Filled: parent methods; hollow: with extrapolation. Every edit was selected in English only.

![[fig3_paired_differences.png|560]]

**Figure 3.** Paired differences between arms (first minus second) over the 18 concepts, with the two-level 95% interval, for (a) the mean leak and (b) the foreign retention. Blue: the difference survives Holm's correction within the block of eight comparisons.

**CINDER against EMBER.** The two methods do not differ. English retention is 0.74 against 0.75 (paired difference −0.013, Wilcoxon p = 0.91; lower in 7 and higher in 11 of 18 concepts). Foreign retention differs by +0.028 ([−0.037, +0.089]) and the mean leak by +0.041 ([−0.060, +0.137], lower in 9 and higher in 9 concepts), with Holm p = 1.0 for both. The result is the same on the 11 concepts that both methods edit (foreign retention +0.053, [−0.011, +0.123]; leak +0.013, [−0.096, +0.126]), so it is not an artefact of counting unedited concepts, and on the 12 concepts that were neither used to tune CINDER nor serve as examples in its lexicon prompt (leak +0.05, p = 0.42). The sharpening and the global sparsity bring no measurable gain in English erasure or in leak.

**The judge finds edits that EMBER misses.** EMBER+TWJ keeps the same edit as EMBER for 14 concepts and adds an edit for the four on which EMBER's judge selects nothing: Halloween, Heroin, Uranium and Valentine's Day. It lowers the mean English retention from 0.75 to 0.67 (the four added concepts are erased, with retention 0.54–0.81) and the foreign retention by 0.032 ([−0.081, −0.004]). It raises the mean leak by +0.048 ([−0.006, +0.119]), because the concepts that were not erased had zero leak. Compared with EMBER+TWJ, CINDER erases English a little less (+0.067, p = 0.08) and the foreign questions slightly less (foreign retention +0.060, [+0.014, +0.113], Wilcoxon p = 0.015, Holm p = 0.075), and it leaks the same (−0.007, [−0.094, +0.081]). Once the judge is the same, CINDER's factorization does not improve on EMBER's.

**Extrapolation lowers foreign retention and the leak.** EMBER+X erases the foreign questions more than EMBER (foreign retention −0.070, [−0.109, −0.040], lower in 14 and higher in none of 18 concepts, Wilcoxon p = 0.0001, Holm p = 0.0008) and leaks less (−0.059, [−0.102, −0.023], lower in 12 and higher in 2 concepts, p = 0.001, Holm p = 0.010). For CINDER the foreign retention falls by 0.050 ([−0.095, −0.022], lower in 12 and higher in 2 concepts, Holm p = 0.014) and the leak by 0.039 ([−0.088, −0.004], p = 0.02), which does not survive the correction (Holm p = 0.14). On the concepts that both a parent and its extrapolated arm edit the effects are larger, because arms without an edit contribute zero (EMBER+X: foreign retention −0.090, leak −0.076; CINDER+X: −0.060 and −0.047). English retention is unchanged within noise (−0.011, p = 0.31), and so is the English selection score (EMBER+X: p = 0.26; CINDER+X: +0.023, p = 0.01). EMBER+X and CINDER+X do not differ (foreign retention +0.048, [−0.048, +0.126]; leak +0.061, [−0.035, +0.150]; Holm p ≥ 0.5). Section 4.4 explains the effect.

**Native-text fits.** The per-language arm lowers the foreign retention more than any other arm (−0.174 against EMBER, [−0.266, −0.088], lower in 13 and higher in none of 18 concepts, Holm p = 0.0014), but it also erases English more, so its leak is unchanged (−0.008, [−0.087, +0.074]). Against EMBER+X its extra foreign erasure (−0.103, [−0.198, −0.009]) is not significant after correction (Holm p = 0.44). Section 4.5 gives the details.

### 4.2 CINDER's edit does not differ from EMBER's

The null result of Section 4.1 is not due to unedited concepts alone. Figure 4 compares the English retention of the two methods concept by concept. The points scatter on both sides of the diagonal: CINDER erases English more for some concepts (Culture of Greece 0.46 against 0.73, Gambling 0.49 against 0.66, Baseball 0.61 against 0.71) and less for others (Cannabis 0.59 against 0.39, Golf 0.89 against 0.74, Artificial intelligence 0.89 against 0.78), with no systematic direction. The concepts for which one method makes no edit lie on the right and top edges.

![[fig4_english_retention.png|380]]

**Figure 4.** English retention of CINDER against EMBER for all 18 concepts (lower = more erased; points on the diagonal are equal). Open circles: one of the two methods makes no edit, so its retention is 1.

At the same relative edit size, CINDER does not erase English better either (train questions, the four concepts that both methods edit): the two are about equal for Gun, CINDER erases less for Cannabis and Golf, and for Culture of Greece it erases more at small edits and less at large ones. CINDER's wider features also add neighbouring topics: the tokens that only CINDER edits include *Roman*, *Rome* and *Italy* for Culture of Greece, and *baseball* and *ball* for Golf. As a consequence its edited tokens occur more often in other concepts' English sentences, in 1.9% of them against 0.84% for EMBER (Table 5). A plausible reason that the sharper features do not improve erasure is that sharpening makes the features cleaner, which is a different quantity from how many of the tokens that a question uses are edited.

### 4.3 The edit changes only the questions it reaches

Because the edit acts through tokens, a foreign question should lose its answer exactly when it contains an edited token. We tested this on the held-out questions of every concept that a method edits, at the selected δ (Figure 5).

![[fig5_touched_untouched.png|560]]

**Figure 5.** Share of the correct answers that the edit turns wrong, for questions that contain an edited token ("touched") and questions that do not, over the concepts each method edits (EMBER 14, CINDER 15). The Russian touched bars rest on about 20 questions.

When a question contains an edited token, the edit turns 37–46% of the correct answers wrong for EMBER and 28–36% for CINDER, in every language. When it does not, it turns at most 1.7% wrong (EMBER) and 1.2% (CINDER). Pooled over the foreign languages, EMBER turns 44.2% of 530 touched questions wrong against 0.6% of 1,230 untouched ones, and CINDER 29.4% of 632 against 0.3% of 1,128. The edit removes the ability to read certain words, not the knowledge, and the cross-lingual leak is a coverage effect.

This also explains why the leak depends on the language. The share of questions that contain an edited token (the coverage) is 66% in English for EMBER, but only 33% in German, 32% in Spanish, 34% in French and 5% in Russian. For CINDER it is 80% in English and 41%, 39%, 43% and 6% in the foreign languages. Russian words are split into pieces that no edit touches, so almost no Russian question is touched and the concept survives there.

![[fig6_coverage_leak.png|420]]

**Figure 6.** Leak in a language against the coverage of its questions, for the 296 edited (concept, foreign language) cells of the five English-fitted arms. Dark line: mean leak in coverage bins.

Across all edited cells the rank correlation between coverage and leak is −0.58, and it lies between −0.35 and −0.77 within each arm (Figure 6). Where less than 5% of the questions contain an edited token (47 cells) the mean leak is +0.30, and where at least 40% do (148 cells) it is about +0.05. Cannabis shows both ends: German, Spanish and French share the tokens *Cannabis*, *THC* and *CBD* with English, their coverage is 0.86–0.90 and the leak is about 0, whereas Russian coverage is 0.08 and the leak +0.51.

### 4.4 Extrapolation raises coverage and lowers the leak

Extrapolation adds native tokens to the edit and raises coverage in every foreign language (Figure 7). For EMBER the share of reached questions rises from 0.33 to 0.51 in German, from 0.32 to 0.59 in Spanish, from 0.34 to 0.60 in French and from 0.05 to 0.28 in Russian; CINDER gains similarly (Russian 0.06 to 0.37). The added tokens are mostly right (*Griechenland*, *griego*, *Grèce*, *Гре…* for Culture of Greece; *pistola*, *fusil*, *pistolet*, *оружие* for Gun), with some noise (Appendix B).

![[fig7_extrapolation_coverage.png|480]]

**Figure 7.** Question coverage by language for each arm (mean over the concepts the arm edits).

The extra erasure sits where the added tokens reach. For foreign questions that contain only an added token, the share of correct answers that are lost rises from 0.5% to 36.7% for EMBER+X and from 0.5% to 23.7% for CINDER+X (Figure 8). Questions with an original token change by at most one point (44.2% to 45.3% and 29.4% to 29.3%), and questions with no edited token hardly change at all (0.6% to 1.1% and 0.1% to 0.6%). The reduction of the leak is therefore explained by the mechanism and not by a general weakening of the edit.

![[fig8_extrapolation_mechanism.png|560]]

**Figure 8.** Foreign test questions answered wrong among those the unedited model answers correctly, with the parent edit alone and with extrapolation, by the kind of edited token the question contains.

**Which added tokens matter.** Real erasure with the added tokens, at the same δ, lowers the mean foreign retention from 0.81 to 0.68 and the leak from 0.18 to 0.10 on the six control concepts (train questions), and EMBER+X erases at least as much foreign knowledge as EMBER in all six, at both strengths we tried. The largest gains are Culture of Greece in German (0.81 to 0.36) and Republic of Ireland in French (0.82 to 0.09); the smallest are Gun at its weak strength δ = 5 (0.99 to 0.96) and Golf. Two filters, dropping tokens that also occur in English and dropping noisy tokens, do not beat the plain extrapolated set (differences below one question).

Two controls test whether the tokens must be chosen by the features. A frequency rule that picks the concept-only tokens most frequent in the concept sentences, without any features, reaches somewhat more foreign questions (coverage 0.69 against 0.61) but with 2–2.5 times the similar-domain collateral and 4–5 times the background exposure; the two sets share only 5–23% of their tokens, because the frequency rule picks subword fragments that are concept-only only because they are missing from 300 neutral sentences. In real erasure (Figure 9), random tokens barely help (foreign retention 0.77 against 0.81 for EMBER), whereas the frequency rule erases as much as the feature-based tokens (0.65 against 0.68) at a higher collateral cost (similar-domain change −0.033 against −0.022 in the foreign languages). Token choice matters, the features are not the only way to make a good choice, and they buy less collateral damage.

![[fig9_token_control.png|560]]

**Figure 9.** Real erasure at the same δ with EMBER's original edit plus 96 added tokens on average: feature-based (X), chosen by sentence frequency (F_own) or at random (R_own). (a) Mean foreign retention. (b) Change of foreign similar-domain accuracy. Bars: mean over the six control concepts; dots: single concepts (train questions).

**What remains.** Extrapolation closes only part of the gap. After EMBER+X, 77% of the foreign questions that the unedited model answers correctly are still answered correctly (German 360 of 480, Spanish 348 of 472, French 337 of 450, Russian 319 of 358), and about 60% of them contain no edited token. The failures are mostly answers that are numbers or dates (*1955*, *1969–1984*, *17 марта*) and names that are split into pieces (*St·aling·rad*, *R·ube·us Hag·rid*, *Б·роне·бой·ный*), which no edit of whole-word rows reaches. Russian remains the hardest language: its retention stays 0.91–0.92 and its coverage rises only to 0.28–0.37. Per-concept values are in Appendix A.

### 4.5 Fitting the features on native text

Fitting on foreign text changes which tokens enter the factorization, but the embedding track does not fit the text itself. It factorizes the embedding rows of the distinct tokens that occur in the sentences, and the text only decides which tokens enter the matrix and how each is labeled (concept-only, neutral, or both). English concept words are whole tokens that occur only in concept sentences (*Hitler*, *Nazi*, *Führer*, *cannabis*, *THC*). Russian words are split into short pieces that also occur in unrelated text, so they are labeled "both" or neutral and carry no concept meaning of their own: *Гитлер* is Г + ит + лер, all three labeled "both", and for World War II 30% of the Russian tokens are labeled "both" against 13% of the English ones. No amount of text can build a feature for *Hitler* out of Г, ит and лер.

The features that the native fits keep reflect this. The per-language arm edits 17 of the 18 concepts (Heroin is the exception). The fits keep a feature for 14 concepts in English (the EMBER arm), 12 in German, 9 in Spanish and 9 in French, and for only 7 in Russian (Republic of Ireland, Baseball, Harry Potter, Pornography, Valentine's Day, Artificial intelligence and COVID-19); Cyrillic tokens enter an edit for six concepts (8–29 tokens). The edit has a median of 98 tokens against 42 for EMBER, and the foreign fits add 37–128 tokens to the English edit for 15 concepts (the three others are Heroin, without an edit, and Golf and World War II, where the foreign fits keep nothing beyond the English edit). The joint fit keeps a feature for only 7 concepts (Republic of Ireland, Baseball, Uranium, Cannabis, Gambling, Gun, COVID-19), edits 122–326 tokens in them (1–21 Cyrillic), and makes no edit for the other 11. Per-concept details are in Appendix A (Table A3).

**Table 3.** Native-text arms against EMBER and EMBER+X, 18 concepts. Concepts with an edit, median edited tokens (concepts with an edit), English retention, foreign retention, mean leak, change of `SimdomQA_test` accuracy in English and in the foreign languages, and MMLU (unedited 0.553).

| Arm | Edited | Tokens | $r_{\text{en}}$ | Foreign | Leak | SimDom en | SimDom foreign | MMLU |
|---|---|---|---|---|---|---|---|---|
| EMBER | 14 | 42 | 0.75 | 0.86 | +0.10 | −0.024 | −0.006 | 0.550 |
| EMBER+X | 14 | 130 | 0.74 | 0.79 | +0.05 | −0.033 | −0.015 | 0.550 |
| Native per-language | 17 | 98 | 0.59 | 0.69 | +0.10 | −0.038 | −0.029 | 0.548 |
| Native joint | 7 | 144 | 0.85 | 0.91 | +0.06 | −0.017 | −0.008 | 0.552 |

![[fig10_all_arms.png|720]]

**Figure 10.** (a) English against foreign retention for all arms, 18 concepts (points above the diagonal have a positive leak). (b) Retention by language for EMBER, EMBER+X and the two native-text arms, with 95% bootstrap intervals over concepts.

**Effect on erasure.** The per-language arm erases much more than any other arm, in English and in the foreign languages (Table 3, Figure 10): English retention 0.59 and foreign retention 0.69, against 0.75 and 0.86 for EMBER. Its foreign retention is 0.174 lower than EMBER's (Section 4.1; on the 14 concepts that both edit −0.156, Holm p = 0.014). It also erases English more, so its leak does not change. Against extrapolation it erases the foreign questions more, but the difference is not significant after correction, and its leak is higher (+0.051, [−0.027, +0.136]). Russian is erased about as much as with extrapolation (retention 0.90 against 0.91). The joint arm erases less than EMBER, because it edits only 7 concepts: its English retention is 0.85 and its foreign retention 0.91 (against EMBER, concept-level differences of +0.10, [+0.01, +0.18], and +0.05, [−0.02, +0.12]).

**Cost.** The per-language arm needs five factorizations and five judge passes per concept instead of one, and it edits more than twice as many tokens as EMBER. The extra erasure is paid for in collateral damage (Section 4.6): the similar-domain accuracy changes by −0.038 in English and −0.029 in the foreign languages (EMBER: −0.024 and −0.006), and for some concepts the damage is large (Baseball: similar-domain change −0.14 in English, MMLU 0.543, German retention 0.00).

### 4.6 Collateral damage

**Table 4.** Collateral damage of the selected edits, mean over all 18 concepts (concepts without an edit count as unedited). `SimdomQA_test` is the change in accuracy against the unedited model (negative = damage); MMLU on the 1,000-question test split (unedited 0.553).

| Arm | SimDom, English | SimDom, foreign mean | MMLU |
|---|---|---|---|
| EMBER | −0.024 | −0.006 | 0.550 |
| CINDER | −0.020 | −0.006 | 0.551 |
| EMBER+TWJ | −0.027 | −0.006 | 0.550 |
| EMBER+X | −0.033 | −0.015 | 0.550 |
| CINDER+X | −0.024 | −0.008 | 0.551 |
| Native per-language | −0.038 | −0.029 | 0.548 |
| Native joint | −0.017 | −0.008 | 0.552 |

All arms leave MMLU at 0.548–0.552 against 0.553, and similar-domain accuracy within a few points of the unedited model; the native per-language arm is the most damaging. Extrapolation costs a little: for EMBER the similar-domain change more than doubles in the foreign languages (−0.006 to −0.015) and grows by about one point in English, while for CINDER it is within 0.005.

The damage is confined to the questions that contain an edited token (Table 5). Of the foreign similar-domain questions that the unedited model answers correctly, those without an edited token lose at most 0.4% of their correct answers in any arm, whereas those with an edited token lose 16–43%. The cost of an edit is therefore its reach. EMBER+X touches 161 foreign similar-domain questions against 63 for EMBER (CINDER+X 255 against 67 for CINDER), and the per-language arm touches 262. The exposure of the edited tokens in ordinary text shows the same ordering: 0.5% of the other concepts' foreign sentences contain an edited token for EMBER and CINDER, 2.1% for EMBER+X, 2.5% for CINDER+X and 1.7% for the per-language arm, and in English 0.8% for EMBER, 1.4% for EMBER+X and 4.7% for the per-language arm. General ability in the foreign languages was not measured (MMLU is English only), so the foreign collateral damage is estimated from these two measures and not tested directly.

**Table 5.** Foreign `SimdomQA_test` questions that the unedited model answers correctly: number and share lost, by whether the question contains an edited token, and the share of other concepts' sentences that contain an edited token (exposure, mean over the concepts each arm edits).

| Arm | Touched: questions | lost | Untouched: questions | lost | Exposure, foreign | Exposure, English |
|---|---|---|---|---|---|---|
| EMBER | 63 | 42.9% | 1,994 | 0.1% | 0.48% | 0.84% |
| CINDER | 67 | 28.4% | 2,103 | 0.0% | 0.54% | 1.91% |
| EMBER+TWJ | 73 | 37.0% | 2,564 | 0.1% | 0.51% | 0.95% |
| EMBER+X | 161 | 40.4% | 1,896 | 0.3% | 2.12% | 1.41% |
| CINDER+X | 255 | 16.1% | 1,915 | 0.1% | 2.52% | 2.57% |
| Native per-language | 262 | 43.1% | 2,194 | 0.4% | 1.68% | 4.69% |

---

## 5. Discussion

**The leak is a coverage problem, not a feature-quality problem.** In every analysis the edit acts only on the questions that contain an edited token, and how much of the concept survives in a language follows how many questions are reached. A better feature finder cannot change this, and CINDER indeed neither erases English better nor leaks less than EMBER. The part of CINDER that matters for erasure is its judge, which finds a feature where EMBER's finds none (Heroin, Uranium, Halloween, Valentine's Day).

**Extrapolation addresses the mechanism.** Adding native tokens raises coverage in the foreign languages (Russian from 0.05 to 0.28), and the extra erasure appears in the questions that the new tokens reach. The effect is moderate (−0.06 in leak for EMBER) because many questions are answered through tokens that no edit reaches, such as numbers, dates and names split into pieces, and because some added tokens are noise. The token controls show that the choice of tokens matters and that the features are not the only good rule, but they buy lower collateral damage.

**Fitting on foreign text erases more, but at a cost, and hardly reaches Russian.** Foreign text can only choose which token rows enter the factorization, and the pieces of foreign words carry no concept meaning, so the Russian fit keeps a feature for only 7 concepts and the joint fit edits only 7. The per-language fits lower the foreign retention more than any other arm (0.69 against 0.86 for EMBER), but they also erase English more and edit more than twice as many tokens, so the leak does not change and the similar-domain damage in the foreign languages is about five times larger. Against extrapolation the extra foreign erasure is not significant after correction. Extrapolation keeps the English features, which are well defined, and uses foreign text only to decide which tokens load on them; it reaches part of the same effect with one fit, no new judge calls and a much smaller edit.

**Practical implication.** An embedding edit should be checked for question coverage in each target language, and a language with a different script needs explicit coverage of its tokens. Extrapolation is a cheap way to obtain it: it needs no new fit and no new judge calls, only native text. A per-language refit erases the most, but at several times the collateral damage and five times the feature-discovery cost.

### Limitations

1. **One model, one seed.** All results are for Gemma-2-2B-it with seed 42, and feature selection can vary with the seed.
2. **Concepts without an edit.** Whether a concept is edited depends on the judge. Means that include unedited concepts mix the quality of the edit with how often features are found; we therefore also report the 11 concepts that all four main arms edit.
3. **CINDER was tuned on a few concepts.** Its parameters (η, γ, the lexicon threshold) were chosen on three concepts and its lexicon prompt uses four example concepts; the 12-concept analysis excludes them. Its sharpening target is the concept's Wikipedia page, which also supplies some of the tokens that its feature-level metrics score.
4. **The judge differs from the paper's.** All arms use gemini-3.5-flash-lite. EMBER+TWJ is an exploratory arm that isolates CINDER's judge.
5. **Extrapolation and the native-text arms are not query-only.** They use native foreign text to choose tokens. The δ of the +X arms was not re-tuned, and some added tokens are noise and some occur in English questions.
6. **The token controls are small.** The token variants and controls use six concepts and the train questions and are descriptive.
7. **Multiple choice, machine-translated questions, 50 questions per cell**, as in Part 1, and no relearning test.
8. **Few differences survive correction for multiple comparisons.** For the foreign retention EMBER+X − EMBER, CINDER+X − CINDER and the native per-language arm − EMBER do; for the leak only EMBER+X − EMBER does. The effect of the judge and the differences between extrapolation and the native per-language arm are suggestive.
9. **The native-text arms mix two effects.** They have their own strength search and edit more tokens, so their results combine the effect of the features with that of a larger edit. They were run for one seed, the joint arm at one rank (500), and the per-language arm's English fit is EMBER's, so it inherits EMBER's four empty concepts.

---

## 6. Conclusion

An embedding edit acts only on the questions that contain an edited token, so an erasure fitted on English leaks into the languages whose questions use other tokens. A better feature finder, CINDER, does not change this: on 18 concepts it neither erases English better nor leaks less than EMBER, and the part of it that helps is its judge, which finds features for four concepts where EMBER's judge finds none. Refitting the features on foreign text finds a Russian feature for only 7 of the 18 concepts. Refitting per language does erase the foreign questions most (foreign retention 0.69 against 0.86 for EMBER), but it also erases English more, leaves the leak unchanged and costs several times the collateral damage. Extending the edit to native tokens by extrapolation, with English features, raises the coverage of foreign questions and lowers EMBER's foreign retention by 0.07 (Holm p = 0.0008) and its leak by 0.06 (Holm p = 0.010) at a small cost in similar-domain accuracy, while leaving English erasure unchanged. About three quarters of the foreign questions are still answered correctly after the extension, so complete cross-lingual erasure remains open.

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

Single cells are noisy: each has 50 questions, so only the averages over concepts in Section 4.5 support conclusions.

**Table A3.** Native-text arms per concept. Per-language arm: features kept by the en / de / es / fr / ru fits, edited tokens, δ; joint arm: features kept, edited tokens, δ. $r_{\text{en}}$ and foreign retention (mean over the reliable de/es/fr/ru cells) on the held-out `QA_test`; "—" = no edit (retention 1).

| Concept | Per-language: kept (en/de/es/fr/ru) | tokens / δ | $r_{\text{en}}$ | foreign | Joint: features / tokens / δ | $r_{\text{en}}$ | foreign |
|---|---|---|---|---|---|---|---|
| Ancient Rome | 2/2/0/1/0 | 144 / 200 | 0.27 | 0.43 | — | 1.00 | 1.00 |
| Artificial intelligence | 1/1/1/1/1 | 155 / 50 | 0.67 | 0.62 | — | 1.00 | 1.00 |
| Baseball | 1/1/0/0/2 | 107 / 200 | 0.46 | 0.35 | 1 / 122 / 5 | 0.61 | 0.88 |
| COVID-19 pandemic | 1/1/1/1/1 | 170 / 10 | 0.69 | 0.59 | 1 / 174 / 10 | 0.85 | 0.80 |
| Cannabis | 1/1/1/1/0 | 109 / 100 | 0.30 | 0.47 | 1 / 142 / 5 | 0.45 | 0.56 |
| Culture of Greece | 1/1/0/0/0 | 89 / 50 | 0.78 | 0.86 | — | 1.00 | 1.00 |
| Gambling | 1/1/1/0/0 | 99 / 10 | 0.59 | 0.79 | 1 / 144 / 200 | 0.69 | 0.70 |
| Golf | 2/0/0/0/0 | 70 / 10 | 0.74 | 0.80 | — | 1.00 | 1.00 |
| Gun | 2/1/2/0/0 | 157 / 5 | 0.58 | 0.92 | 3 / 326 / 5 | 0.51 | 0.81 |
| Halloween | 0/0/0/1/0 | 41 / 50 | 0.54 | 0.80 | — | 1.00 | 1.00 |
| Harry Potter | 1/1/1/1/1 | 98 / 10 | 0.60 | 0.56 | — | 1.00 | 1.00 |
| Heroin | 0/0/0/0/0 | — | 1.00 | 1.00 | — | 1.00 | 1.00 |
| Nazism | 1/0/1/1/0 | 92 / 100 | 0.51 | 0.63 | — | 1.00 | 1.00 |
| Pornography | 1/1/0/1/1 | 128 / 5 | 0.65 | 0.76 | — | 1.00 | 1.00 |
| Republic of Ireland | 1/1/0/1/1 | 85 / 200 | 0.31 | 0.60 | 1 / 150 / 10 | 0.67 | 0.78 |
| Uranium | 0/0/2/0/0 | 50 / 50 | 0.57 | 0.70 | 1 / 141 / 200 | 0.57 | 0.86 |
| Valentine's Day | 0/1/1/0/1 | 91 / 200 | 0.43 | 0.56 | — | 1.00 | 1.00 |
| World War II | 1/0/0/0/0 | 36 / 5 | 0.88 | 0.91 | — | 1.00 | 1.00 |

## Appendix B. Tokens that extrapolation adds

For the twelve concepts other than those in the table below, extrapolation adds 54–125 tokens per concept beyond the parent's (Table A1), 52–83% of them whole words and 85–94% of them never occurring in English text. Most are on topic (Casinos, cuarentena, mascarilla, Neutron, Plut, römischen), and some are clear noise (EMBER+X for Harry Potter: SHIELD, Nintendo, Wii, Prada, Hitler; for Pornography: Poverty, Punk, Emo). First added tokens of the six token-control concepts, EMBER+X:

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
| Touched/untouched, coverage, extrapolation, controls (X1–X14) | `Part 2 Experiments/` (one folder and `report.md` per analysis; scripts in `scripts/`); foreign retention, collateral by touched question and exposure: `x14_followup/` |
| Native-text arms (per-language and joint fits) | `cinder/ml_arms.py`, `judge_rerun/Fair/ml_sep/`, `ml_joint/`, `ml_build.json` |
| Foreign-text fits and rank sweeps | `docs/Part 2/plan.md`, `docs/Part 2/multilingual_features.md` |
| Figures in this report | `report/Part2_report/make_figures.py` → `figures/` |

Settings: Gemma-2-2B-it, bfloat16, greedy decoding; rank 100, seed 42; feature judge gemini-3.5-flash-lite; δ grid 0.5–200; reliability cutoff 0.40; two-level bootstrap with 4,000 resamples and Holm within blocks of eight comparisons for the paired tests; bootstrap `numpy.random.default_rng(0)`, 10,000 resamples for the intervals over concepts in Table 2 and Figure 2. The means in Table 2 use the reliable cells only and can differ by 0.01 from the unfiltered means in `judge_rerun/Fair/report.md`.

To regenerate the figures (needs matplotlib, numpy, pandas):

```bash
python report/Part2_report/make_figures.py
```
