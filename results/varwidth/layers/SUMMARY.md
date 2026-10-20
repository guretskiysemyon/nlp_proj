# CINDER layer ladder: summary of results (2026-10-18)

**Setup.**
- **Model and concepts:** Gemma-2-2b-it; 38 concepts (23 original + 15 held-out); 5 seeds each.
- **Judging:** judges see the top 20 features by ρ at every rung.
- **Pre-registration:** [PREREG.md](PREREG.md), amendments 1–2.
- **Erasure anchor:** the 14 non-prompt paper concepts, seed 42, EMBER's MC protocol. The δ grid is gated by MMLU, and Alpaca is scored at the chosen δ.
- **Full tables:** [ladder.md](ladder.md), [layers.md](layers.md), [erasure.md](erasure.md).
- **Gemini spend:** 16.4 NIS.

## Main results

### 1. Erasure (EMBER's own metric, 14 concepts)

| Rung | Con ↓ | Sim ↑ | MM ↑ | Efficacy | H |
|---|---|---|---|---|---|
| Unedited | 73.4 | 89.3 | 55.3 | — | — |
| EMBER | 54.3 | 86.4 | 55.2 | 0.36 | 0.52 |
| +CS | 62.4 | 88.6 | 55.3 | 0.21 | 0.38 |
| +CS+GS | 62.1 | 88.0 | 55.3 | 0.22 | 0.39 |
| +CS+GS+TWJ (CINDER, pre-registered) | 59.7 | 87.6 | 55.3 | 0.28 | 0.49 |
| EMBER+TWJ (exploratory) | **50.1** | 86.1 | 55.1 | **0.47** | **0.65** |
| EMBER + TJ∨WJ+share (exploratory) | 50.0 | 85.3 | 55.0 | 0.47 | 0.65 |

Paired over concepts:

| Comparison | Δ H | 95% CI | Concepts up / down |
|---|---|---|---|
| EMBER → +CS | −0.15 | [−0.32, +0.04] | 3 / 9 |
| +CS → +CS+GS | +0.02 | — | — |
| +CS+GS → CINDER | +0.10 | [0.00, +0.25] | 2 / 0 |
| EMBER → CINDER | −0.03 | [−0.24, +0.19] | — |
| EMBER → EMBER+TWJ | +0.13 | — | 2 / 0 |
| EMBER → EMBER + TJ∨WJ+share | +0.12 | [+0.01, +0.28] | 5 / 0 |

**Why the judges help EMBER:** they rescue concepts where EMBER's judge selected no feature, so EMBER made no edit at all. Heroin, Uranium, Nazism and Halloween go from H = 0 to between 0.15 and 0.92. No concept gets worse.

### 2. Feature level (pre-registered primary: strict clean-hit, 34 non-prompt concepts)

| Rung | Strict clean-hit |
|---|---|
| EMBER | 0.46 |
| +CS | 0.44 |
| +CS+GS | 0.41 |
| CINDER | 0.40 |

- **No significant change:** all Holm-corrected p = 1.
- **Finding metrics improve:**
  - runs with no feature selected: 48% → 28%;
  - runs that find a P feature: 0.51 → 0.59;
  - lenient clean: 0.51 → 0.71.
- **Strict clean falls** because the extra accepted features are often broader than the concept (B).

### 3. Per layer

- **CS:**
  - **Separability rises.** Concept-vs-neutral AUC goes up in 38/38 concepts (0.69 → 0.80), and judge-free hit@5 goes 0.56 → 0.62.
  - **Erasure gets worse.** CS features absorb generic tokens that our labels call concept-only, because they occur only in the concept sentences. Baseball's feature gains get, win, catch, gain; Valentine's Day's gains heart, values, dates, Easter. The mapped-back edit is also less aligned with each token's own embedding (cosine about 0.3 vs about 0.5), and efficacy stops rising beyond δ ≈ 5–10.
  - **Conclusion:** feature-level purity overstated CS. The "concept-only" label is too weak a definition of concept tokens.
- **GS (GS-norm):**
  - **Most stable.** It has the best seed stability of the width rules (Jaccard 0.675 vs 0.652 for RS-1%) and lower collateral than RS-2%, GS-gain or GS-entropy.
  - **Width doesn't track breadth.** The Spearman correlation of width with concept breadth is about 0.05.
  - **No effect on erasure.**
- **Judges:**
  - **TJ noise is tiny:** 0.4% of decisions flip on re-judging 230 candidates.
  - **On the original concepts,** TWJ = TJ ∨ EJ@5 raises recall: P 0.71 / R 0.79.
  - **On the 15 held-out concepts, EJ@5 does not generalize** (P 0.46, R 0.26), so TWJ falls to P 0.53 / R 0.56. TJ ∨ WJ+share does generalize: P 0.71, R 0.89, 49 of 75 runs hit, against 33 for TJ alone. In erasure, TWJ and TJ ∨ WJ+share perform the same.

### 4. Validity checks

- **Our EMBER reproduces the paper.** Con 48.4 / Sim 83.2 / MM 55.1, against the paper's Table 6: 52.1 / 84.6 / 53.6.
- **The cheaper gate is harmless.** The paper's Alpaca-in-grid tuning gives the same δ and result on Harry Potter (H 0.727 vs 0.726).
- **The proxies track erasure.** Held-out coverage and clean-hit correlate with erasure H at ρ ≈ 0.5–0.65 over 98 concept × rung points.

## Conclusions

1. **The judge is EMBER's bottleneck.** The token judge often rejects correct features, so there is no edit at all: 4 of 14 erasure concepts at seed 42, and 48% of runs at the feature level. Adding a second, complementary check fixes most of these, and this is the one layer that improves erasure: about +0.12 H with no concept getting worse. The judge that generalizes to held-out concepts is **TJ ∨ WJ+share**.
2. **Contrastive Sharpening, as built, should not be in the erasure pipeline.** It improves feature-level geometry but lowers erasure efficacy by about 0.15. The cause is that "concept-only" tokens include generic words, and the sharpened features pick these up.
3. **Global Sparsity is neutral for erasure.** It gives stability, not efficacy.
4. **Recommended configuration:** EMBER's matrix and widths with the improved judge, i.e. EMBER + TJ ∨ WJ+share. H 0.65 vs 0.52; Con 50.0 vs 54.3; specificity and coherence unchanged.

## Caveats

- **The judge-only rungs were chosen after seeing the results** (PREREG amendment 2). Confirm them on new concepts or seeds before claiming them.
- **Erasure covers 14 concepts at one seed.** The new concepts have no MC questions.
- **Labels are Claude's** (1,096 new groups, adjudicated). The user's 60-group check in human_labels.csv is still pending; strict P/B decisions affect only the label-based metrics, not erasure.
