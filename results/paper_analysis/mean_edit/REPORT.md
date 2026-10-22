<!-- Saved by the lead from the WP agent's hand-back (a hook blocked the agent's own write); text unchanged. -->
# WP6b mean_edit: does the background-mean term erase the concept, or only damage the word?

2026-10-22/23 · GPU erasure runs (3 TITAN Xp workers, ~16.5 GPU-h) and CPU follow-ups, no Gemini (0 NIS). Pre-registration: `PREREG.md`, written before any result; amendments 1–2 are logged there. Code is in `experiments/paper_analysis/mean_edit/`, outputs in `results/paper_analysis/mean_edit/`.

## Question

WP6 S1: *"Much of the 'stronger' uncentered edit is generic, not about the concept. Subtracting β μ_B makes a token atypical. Test: … a pure-mean edit e_t − β_t μ_B on CINDER's tokens (β from CINDER-z) … if it reproduces most of CINDER-z's efficacy gain at r ≥ 1 and its destruction labels, the gain is not concept-specific."* Also S2 (does read gain come from the layer-0 norm weights?) and S4 (does the linear read-strength proxy predict forward-pass sensitivity?).

**Short answer: S1 is supported (pre-registered rule), in both families.**
- **CINDER family.** A concept-free edit along the background mean, with CINDER-z's own per-token coefficients, erases more than CINDER-z at every r ≥ 2 (validation 0.60–0.76 vs 0.46–0.55). It does this by destroying the surrounding prompt: unrelated MMLU questions after the word fall below chance (0.15–0.18 vs 0.54), and collateral falls to 0.07.
- **The pattern.** Efficacy above the deletion floor tracks context damage. Centred edits (CINDER, and EMBER re-solved on centred embeddings) level off at the floor and leave the context intact. Under the copy gate the mean edit erases nothing.
- **Conclusion.** The uncentred gain is generic damage, not concept removal.

## Method

- **Runs.** The 8 development collateral runs: Harry Potter 42/43, Ancient Rome 42/43/44, Valentine's Day 42/43/46.
  - CINDER family: the 7 runs where CINDER edits. EMBER family: the 6 where EMBER edits.
  - Unit = run. There are only 3 concepts, so the tests are descriptive.
- **Variants.** All edits are e′ = e − δc, with r = median_t |δc_t|/|e_t| computed from each variant's own lengths, on the grid {0.05 … 100}.
  - CINDER (mapped back, centred) and CINDER-z (raw re-solve).
  - **MEAN-C** = β_t^z μ_B on CINDER's tokens, with β^z = Σ_{i∈F_C} b_i Y_it and b = (YYᵀ+λI)⁻¹Y1 (WP6 Corollary 2).
  - EMBER, and **EMBER-c**: EMBER's Y and F_C with Z re-solved on A − μ_B1ᵀ.
  - **MEAN-E** = β_t^E μ_B on EMBER's tokens.
  - Noise-C and Noise-E (existing baselines).
  - CPU check: CINDER-z − CINDER = MEAN-C and EMBER − EMBER-c = MEAN-E, to relative error 1e-4 in every run.
- **Reuse.** The existing grids, test passes and E2 files of CINDER, CINDER-z, EMBER and Noise are reused. A pilot re-evaluation of CINDER-z (Harry Potter 42, r = 1) reproduced the frozen grid point exactly.
- **Measures** (frozen question sets, existing code paths):
  - validation efficacy, specificity, collateral and H3v at every r;
  - T1 copy at every grid point with r ≤ 2.5, for all variants including the existing ones;
  - collateral-rule and copy-gated (E3, copy ≥ 0.9) selection, with test passes (collateral renormalised as in amendment 6);
  - the E2 judge-free tests (copy, context recall, context-MC, answer-NLL excess over the nonce, echo, z > 3) at r ∈ {0.25, 1, 10, 100} and at the selected points, on the parent's word list;
  - nonce and deletion floors for CINDER's and EMBER's token sets.
- **Not done:** the delta-matched grids (dropped under amendment 1 and replaced by interpolation), T3 labels, neutral generations, Alpaca.

## Backed findings

### CINDER family (7 runs)

**F1. The mean edit alone erases more than CINDER-z at r ≥ 2.** Validation efficacy, mean over 7 runs:

| r | 0.1 | 0.25 | 0.5 | 1 | 2 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| CINDER | 0.08 | 0.26 | 0.33 | 0.35 | 0.32 | 0.35 | 0.34 | 0.34 | 0.36 |
| CINDER-z | 0.03 | 0.15 | 0.29 | 0.42 | 0.46 | 0.50 | 0.52 | 0.55 | 0.54 |
| **MEAN-C** | 0.00 | 0.07 | 0.23 | 0.47 | 0.60 | 0.70 | 0.74 | 0.76 | 0.76 |
| Noise-C | 0.02 | 0.03 | 0.13 | 0.28 | 0.33 | 0.31 | 0.28 | 0.27 | 0.27 |
| Collateral, CINDER / CINDER-z / MEAN-C | .98/.98/.99 | .93/.95/.97 | .88/.91/.90 | .82/.77/.73 | .76/.69/.61 | .73/.61/.43 | .72/.60/.28 | .73/.53/.09 | .73/.52/.07 |

- **Over r ∈ {1, …, 100}:**
  - Uncentred gain Gz = eff(CINDER-z) − eff(CINDER) = +0.155 [95% CI 0.04, 0.29], positive in 7/7 runs.
  - Mean-only efficacy Em = 0.671, which is 4.3× Gz; Em ≥ 0.5·Gz in 7/7 runs.
- **Not just any big perturbation.** MEAN-C − Noise-C at r ∈ {10, 30, 100} is +0.48 [0.32, 0.62], 7/7 runs, Wilcoxon p = 0.016.
- **Per run, the gain follows the mean edit's reach:**
  - Ancient Rome 42/44: gain 0.47/0.38 (MEAN-C 0.95/0.97).
  - Harry Potter 42/43: gain 0.07/0.10 (MEAN-C 0.75/0.81).
  - Valentine's Day: gain 0.02 in all three runs (MEAN-C 0.76/0.23/0.24). There the median β is 0.31–0.35 (vs 0.53–0.66 elsewhere) and the tokens' own μ coordinate is 0.83–0.95 (vs 1.35–1.38).

**F2. The same holds at the size the mean term has inside CINDER-z** (approximate decomposition, amendment 1).
- At CINDER-z's grid point r, its mean part has size 0.52–0.73·r.
- Interpolating MEAN-C's grid at that size gives 0.51–0.76 at r ≥ 2. CINDER-z itself reaches 0.46–0.55, and its centred part (CINDER at δ_z) 0.33–0.35.
- Averaged over r ≥ 1: the mean part alone reaches 0.62, against an uncentred gain of 0.16 (7/7 runs).
- The two parts are sub-additive: at r ≥ 2, adding the centred part to the mean part lowers efficacy.

**F3. The extra efficacy is bought with context damage.** E2 judge-free tests, 7 runs. MEAN-C and CINDER-z use the same words; other pairs use their common words.

| r | Copy (CINDER / CINDER-z / MEAN-C) | Context-MC (real word 0.54) | Answer-NLL excess over nonce (nats) |
|---|---|---|---|
| 0.25 | 0.95 / 0.94 / 0.97 | 0.51 / 0.53 / 0.52 | +0.11 / +0.09 / +0.11 |
| 1 | 0.52 / 0.42 / 0.36 | 0.52 / 0.49 / 0.42 | +0.04 / +0.25 / +0.60 |
| 10 | 0 / 0 / 0 | 0.53 / 0.34 / 0.18 | +0.12 / +1.17 / +1.28 |
| 100 | 0 / 0 / 0 | 0.53 / 0.31 / 0.15 | +0.25 / +1.27 / +1.19 |

- **At r = 1:**
  - MEAN-C's context-MC is below CINDER-z's in 7/7 runs (−0.07, p = 0.016) and below CINDER's in 7/7 (−0.08).
  - Its NLL excess is larger than both in 7/7 runs.
  - Its copy is below CINDER's in 7/7 runs.
- **At r ≥ 10,** context-MC falls below chance (0.25): after a sentence containing the word, the model no longer answers an unrelated question.
- Specificity (questions with no edited tokens) stays ≥ 0.89.

**F4. Efficacy above the deletion floor tracks context damage.**
- **The floor.** For CINDER's token set (unedited model, the set's tokens deleted from the questions), the deletion floor is 0.30 on validation (nonce 0.37) and 0.19 on test (nonce 0.29).
- **Efficacy minus the deletion floor at r = 1 / 10 / 100:**
  - CINDER: +0.05 / +0.04 / +0.06
  - Noise-C: −0.02 / −0.02 / −0.03
  - CINDER-z: +0.12 / +0.22 / +0.24
  - MEAN-C: +0.17 / +0.44 / +0.46 (above the floor in 7/7 runs at r ≥ 10)
- **Correlation.** Over the 63 points (run × variant × r ∈ {1, 10, 100}), this excess correlates with the context-MC drop (Spearman 0.54) and with the answer-NLL excess (0.49).
- **CINDER is the control.** Its words are destroyed just as completely at r ≥ 10 (copy 0, deletion-likeness 0.62), yet it stays at the floor with the context intact.

**F5. Per unit of word damage, the centred edit erases most.** Efficacy interpolated at a given copy rate along r ≤ 2:

| Copy rate | CINDER | CINDER-z | MEAN-C |
|---|---|---|---|
| 0.9 | 0.30 | 0.20 | 0.13 |
| 0.7 | 0.34 | 0.31 | 0.26 |
| 0.5 | 0.35 | 0.38 | 0.38 |

- At copy 0.9, CINDER − CINDER-z = +0.10 [0.04, 0.16] (6/7, p = 0.03) and CINDER-z − MEAN-C = +0.07 (6/7, p = 0.16). At copy 0.5 the three are equal.
- Below r = 0.5 the mean edit erases least (r = 0.25: 0.07 vs 0.15 for CINDER-z and 0.26 for CINDER).

**F6. Selected points (test split).**

| Rule | Variant | r* median (values) | Efficacy | Collateral (rn) | H3v |
|---|---|---|---|---|---|
| Collateral rule | CINDER | 0.5 (0.25 ×2, 0.5 ×3, 5 ×2) | 0.21 | 0.87 | 0.42 |
| | CINDER-z | 0.5 (0.5 ×6, 1) | 0.23 | 0.91 | 0.41 |
| | **MEAN-C** | 0.5 (0.5 ×6, 1) | 0.24 | 0.90 | 0.43 |
| Copy-gated (E3) | CINDER | 0.1 (0.05–0.25) | 0.12 | 0.98 | 0.24 |
| | CINDER-z | 0.25 | 0.10 | 0.98 | 0.22 |
| | **MEAN-C** | 0.25 | 0.01 | 0.98 | 0.04 |

- **Collateral rule: the concept-free edit ties both methods** (H3v differences ≤ 0.02, 3–4 wins of 7). At Valentine's Day 42 it reaches test efficacy 0.67 at r = 1.
- **That point still does damage:** copy 0.70, context-MC 0.49 vs 0.55, answer-NLL excess +0.27 (CINDER-z +0.11, CINDER +0.08).
- **Copy gate: MEAN-C erases essentially nothing** (0.00–0.06 per run). It trails CINDER by −0.10 (0 wins in 4 non-tied runs) and CINDER-z by −0.09 (0 wins in 5).

### EMBER family (6 runs)

**F7. EMBER replicates the pattern.** EMBER's raw fit is uncentred; EMBER-c is its centred twin. Validation efficacy:

| r | 0.25 | 0.5 | 1 | 2 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|
| EMBER-c | 0.12 | 0.26 | 0.34 | 0.37 | 0.33 | 0.32 | 0.31 | 0.30 |
| EMBER | 0.05 | 0.20 | 0.31 | 0.38 | 0.42 | 0.44 | 0.50 | 0.52 |
| **MEAN-E** | 0.05 | 0.16 | 0.37 | 0.54 | 0.65 | 0.71 | 0.77 | 0.77 |
| Noise-E | 0.02 | 0.12 | 0.22 | 0.28 | 0.30 | 0.29 | 0.31 | 0.31 |

- **Collateral (EMBER-c / EMBER / MEAN-E):** r = 1: 0.84 / 0.85 / 0.77; r = 10: 0.69 / 0.67 / 0.28; r = 100: 0.68 / 0.65 / 0.05.
- **Gain and mean-only efficacy.** Gain G_E = eff(EMBER) − eff(EMBER-c) over r ≥ 1 is +0.097 [0.03, 0.19], 6/6 runs. Mean-only efficacy is 0.63.
- **Not just any big perturbation.** MEAN-E − Noise-E at r ≥ 10 is +0.44 [0.24, 0.63], 6/6.
- **Approximate decomposition.** At r ≥ 5 the mean part alone reaches 0.57–0.77, vs 0.42–0.52 for EMBER.
- **Floor and context.** The deletion floor of EMBER's tokens is 0.31 (validation).
  - Efficacy minus that floor at r = 1 / 10 / 100: EMBER-c +0.02 / +0.01 / −0.01; EMBER −0.01 / +0.13 / +0.20; MEAN-E +0.06 / +0.40 / +0.46.
  - Context-MC at r = 10 / 100: EMBER-c 0.54 / 0.54 (equal to the real word, above the nonce 0.52); EMBER 0.45 / 0.38; MEAN-E 0.17 / 0.15.
  - Answer-NLL excess at r = 10 / 100: EMBER-c +0.12 / +0.22; EMBER +1.08 / +1.22; MEAN-E +1.28 / +1.20.
  - EMBER − EMBER-c on context-MC at r = 10: −0.10, 6/6 runs.
  - Spearman between floor excess and context drop: 0.57 (54 points).
- **Copy-matched efficacy.** At copy 0.9: EMBER-c 0.22 > EMBER 0.15 > MEAN-E 0.11. At copy 0.7: 0.30 / 0.23 / 0.22 (EMBER-c − EMBER +0.07, 6/6, p = 0.03).
- **Selected points (test).**
  - Collateral rule: EMBER-c efficacy 0.25, collateral 0.87, H3v 0.48 (r* median 1.5, because its collateral stays above the gate longer). EMBER: 0.22 / 0.93 / 0.42. MEAN-E: 0.13 / 0.87 / 0.26.
  - EMBER-c − EMBER on H3v: +0.06 (4 of 6 runs, p = 0.56).
  - E3 test efficacy: EMBER-c 0.08, EMBER 0.09, MEAN-E 0.02.
  - So the mean term costs EMBER nothing under either rule when removed. Unlike MEAN-C, MEAN-E does not tie its parent at the collateral-rule point.

**F8. The coordinator's question: does EMBER's mean term explain why CINDER's direction advantage is gone at r = 1?** Validation efficacy. "1b" is the existing cell with EMBER's tokens and CINDER's direction (5 runs).

| r | EMBER | EMBER-c | MEAN-E | 1b | EMBER-c − EMBER (wins/losses) | CINDER − CINDER-z (7 runs) |
|---|---|---|---|---|---|---|
| 0.25 | 0.05 | 0.12 | 0.05 | 0.20 | +0.066 (4/0) | +0.117 (7/0) |
| 0.5 | 0.20 | 0.26 | 0.16 | 0.33 | +0.059 (3/2) | +0.041 (4/1) |
| 1 | 0.31 | 0.34 | 0.37 | 0.31 | +0.031 (4/2) | −0.071 (0/6) |
| 2 | 0.38 | 0.37 | 0.54 | 0.30 | −0.012 (2/2) | −0.139 (0/7) |

- **On the 5 runs that have the 1b cell:**
  - At r = 0.5, centring raises EMBER from 0.21 to 0.24, against 0.33 for 1b: about a quarter of the CINDER-direction gap.
  - At r = 1, it raises EMBER from 0.25 to 0.31, against 0.31 for 1b: nearly the whole gap.
- **Reading.**
  - The centred edits (CINDER, 1b, EMBER-c) plateau near the deletion floor (≈0.30–0.37) from r ≈ 0.5–1 on.
  - EMBER keeps rising past that plateau. The fastest riser is the concept-free MEAN-E (0.16 → 0.37 → 0.54).
  - So the hypothesis fits in direction. The mean component's contribution grows with r, and at r = 1 centring explains the 1b − EMBER difference.
  - But on these dev runs the mean term is not yet a net gain for EMBER at r = 1 (EMBER-c ≥ EMBER). It becomes one only at r ≥ 2–5.
  - The r = 1 loss of Δ_dir is better described as "CINDER's direction has already reached the floor plateau, while EMBER is still climbing, partly through generic damage" (EMBER's NLL excess at r = 1 exceeds EMBER-c's by +0.14, 6/6).
- **Caveat:** dev runs, validation split, n = 5–6, 3 concepts. Test on the expansion concepts before relying on it.

### S2 and S4 (CPU)

**F9. S2: the large norm weights explain half to three quarters of the read-gain gap.** WP6 runs, 27–31 concepts × seeds, concept means. The 8 largest layer-0 norm weights g are 3.98–5.97 (median 1.17) and hold 36% of μ̂_B's energy.

| | actual g | g = 1 | top-8 g set to 0 |
|---|---|---|---|
| μ̂_B read gain (× random) | 2.93 | 2.10 | 1.74 |
| EMBER / CINDER-z | 1.88 / 1.99 | 1.59 / 1.71 | 1.36 / 1.48 |
| CINDER / EMBER-c | 1.18 / 1.10 | 1.28 / 1.21 | 1.20 / 1.15 |
| Gap EMBER − CINDER | +0.68 (27/27) | +0.32 (27/27) | +0.17 (21/27) |
| Gap CINDER-z − CINDER | +0.82 | +0.44 | +0.28 |

With μ̂_B projected out, every variant is read at 1.28–1.35× under all three settings of g.

**F10. S4: the linear proxy predicts layer 0, not the output.** 50 tokens from 5 runs, gemma-2-2b-it in float32. Each value is relative to the token's mean over 4 random directions; s = perturbation size / |e_t|.
- **Layer 0 matches the proxy.** Spearman(proxy, measured layer-0 write gain) is 0.76 / 0.81 / 0.78 at s = 0.01 / 0.25 / 1 (350 token-directions).
- **Uncentred and mean directions spread to the context.** Layer 0 writes more for them (in 92–100% of tokens). They also push much more into the later positions after layer 0:

  | Direction | MEAN | CINDER-z | EMBER | CINDER | EMBER-c |
  |---|---|---|---|---|---|
  | Change at later positions (× random) | 2.5–4.3 | 2.1–2.6 | 1.6–1.9 | 1.1–1.2 | 0.8–1.1 |

- **The token's own prediction runs the other way at small s.** Change in the token's next-token distribution (KL, × random) at s = 0.01:

  | CINDER | EMBER-c | CINDER-z | EMBER | MEAN-C | MEAN-E |
  |---|---|---|---|---|---|
  | 5.4 | 3.9 | 2.6 | 2.3 | 1.6 | 1.3 |

  Uncentred exceeds centred in only 10% of tokens. At s = 1 the variants are close (2.2–3.5; uncentred > centred in 52%).
- **Reading.** The mean direction is "read strongly" in the sense that it spreads into the context, not in the sense that it changes what the model predicts at the token.

## Prediction scorecard (PREREG)

| # | Prediction | Outcome |
|---|---|---|
| P1 | Tier-1 verdict: S1 supported | **Supported.** Em 0.67 ≥ 0.5 × Gz 0.155; copy(MEAN-C, r = 1) 0.36 ≤ 0.42 + 0.1; context-MC 0.17 ≤ 0.32 + 0.05. Gz was as predicted (~0.15); Em far exceeded the predicted ~0.35. |
| P2 | Mean part ≥ 0.5 × decomposition gap | Supported (approximate, amendment 1): 0.62 vs 0.16 |
| P3 | Damage ordering at r = 1 and r ≥ 10 | Supported: copy 0.36 ≤ 0.52; context-MC 0.15–0.18 ≤ 0.40; NLL excess +1.2–1.3 ≥ 0.8 |
| P4 | MEAN-C − Noise-C ≥ +0.10 at r ≥ 10 | Supported: +0.48, 7/7 |
| P5 | EMBER gain ≥ +0.08 in ≥ 5/6 runs; Em_E ≥ 0.5 G_E | Supported: +0.097, 6/6; Em_E 0.63 |
| P6 | EMBER-c shows no derailment at r ≥ 10; EMBER ≥ 0.08 below the nonce | Supported: EMBER-c 0.54 vs nonce 0.52; EMBER 0.45 / 0.38 |
| P7 | EMBER-c ≥ EMBER + 0.03 at r ∈ {0.25, 0.5} | Supported: +0.066 / +0.059 |
| P8 | At copy 0.5: CINDER ≥ CINDER-z ≥ MEAN-C | Refuted at 0.5 (0.35 / 0.38 / 0.38); holds at copy 0.9 and 0.7 |
| P9 | E3: MEAN-C ≤ 0.05; collateral rule: H3v(MEAN-C) < H3v(CINDER-z) | First part supported (0.01). Second refuted (0.43 vs 0.41, a tie) |
| P10 | Collateral rule: EMBER-c within ±0.05 of EMBER | Marginally refuted: +0.057, in EMBER-c's favour (4/6 runs, p = 0.56) |
| S2 | μ̂_B ≤ 1.5× and gap shrinks ≥ 60% under both g settings | Mostly refuted: μ̂_B 2.10 / 1.74; gap −53% (g = 1), −75% (top-8 = 0) |
| S4 | Uncentred > centred at layer 0 in ≥ 70% of tokens, ρ ≥ 0.5; at s = 1, KL MEAN ≥ uncentred > centred | First part supported (92–100%, ρ 0.76–0.81). Second refuted: at s = 1 the KLs tie, and at s = 0.01 the order is reversed |

## Speculations (each with a test)

- **S5. The mean direction derails the prompt through attention.** F10 shows it spreads into later positions after layer 0 but barely changes the token's own prediction, and F3 shows unrelated questions fail.
  - Speculation: a token far out along −μ̂_B becomes a sink-like attention outlier for later positions.
  - *Test:* measure attention mass from later positions onto the edited position in layers 0–3, MEAN-C vs CINDER at r ∈ {1, 10}, on the E2 carriers.
- **S6. Concepts whose tokens carry less mean gain less from uncentring.** In Valentine's Day, β and the tokens' μ coordinate are about half those of Harry Potter and Ancient Rome, and the gain is 0.02 in all three runs.
  - *Test:* across the 12 expansion concepts, regress the CINDER-z − CINDER efficacy gain (existing grids) on the median β^z. CPU only.
- **S7. Context-blind metrics reward this damage.** On CINDER's tokens, MEAN-C ties CINDER at the collateral-rule point.
  - *Test:* add a word-free context check to the selection rule (e.g. context-MC within 0.03 of the real word), using the saved grids plus context-MC at grid points r ≤ 2.
- **S8. EMBER-c is a better EMBER under the collateral-aware evaluation.** It erases more at r ≤ 1 and shows no context damage.
  - *Test:* run EMBER-c on the 12 expansion concepts at the collateral rule and at fixed r = 0.5. No new fits are needed.

## Paper paragraph

Re-solving CINDER's dictionary on uncentered embeddings (CINDER-z) adds exactly β_t μ_B to every edit, a term that carries no concept information. We therefore evaluated it on its own. On CINDER's tokens, with CINDER-z's coefficients, the edit e_t − δβ_t μ_B erases more than CINDER-z at every strength r ≥ 2 (validation efficacy 0.60–0.76 vs 0.46–0.55; 7 runs), and more than a random direction of the same size (+0.48 at r ≥ 10, 7/7). It does so by breaking the prompt around the word. An unrelated multiple-choice question placed after a sentence that contains the edited word drops from 0.54 to 0.15–0.18 accuracy, below chance, and collateral accuracy falls to 0.07. Across variants and strengths, efficacy above the deletion floor tracks this context damage. Centred edits destroy the words as completely at large r, yet stay at the floor and leave the context intact. The same holds for EMBER, whose raw fit is uncentered: re-solving its dictionary on centred embeddings removes its large-r gain (+0.10 over r ≥ 1) and the context damage, and raises efficacy in the subtraction regime (+0.06 at r ≤ 0.5). When the edited words must remain copyable, the mean edit erases almost nothing (test efficacy 0.01 vs 0.12 for CINDER). At equal word damage the centred edit erases most (0.30 vs 0.20 for CINDER-z and 0.13 for the mean edit at 90% copy). The extra strength of uncentered edits is therefore generic damage, not concept removal.

## Appendix text

**Edit sets.**
- Edited tokens are the concept-only columns in the support of Y_{F_C}, where Y (k × n) holds the sparse coefficients and F_C the concept features.
- **CINDER:** c_t = Σ_{i∈F_C} Y_it z̃_i T⁻¹ (mapped back).
- **CINDER-z:** Z_o = AYᵀ(YYᵀ+λI)⁻¹, with A the raw TL-scaled E_{V*}ᵀ and λ = 1e-4.
- **MEAN-C:** c_t = β_t μ_B, with β_t = Σ_{i∈F_C} b_i Y_it and b = (YYᵀ+λI)⁻¹Y1_n. By WP6 Corollary 2, CINDER-z = (centred re-solve edit) + MEAN-C; here the centred re-solve equals the mapped-back dictionary to 1e-4.
- **EMBER-c:** Z_c = (A − μ_B1ᵀ)Yᵀ(YYᵀ+λI)⁻¹ on EMBER's plain-fit Y. **MEAN-E** uses b from EMBER's Y.
- **Magnitudes:**
  - |μ_B| = 34.5 (TL scale) and |e| ≈ 80–105.
  - The tokens' own mean coordinate e·μ_B/|μ_B|² is 0.83–1.84 (per-run medians).
  - Median β is 0.31–0.66 for CINDER and 0.31–0.87 for EMBER.
  - At an uncentred grid point r, the mean part alone has size 0.52–0.73·r (EMBER 0.45–0.58·r).

**Grid and selection.**
- r ∈ {0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 100}, with δ = r / median_t(|c_t|/|e_t|).
- **Validation sets:** qa_train 50, simdom_train 50, MMLU train 50, and the collateral validation items on the variant's own edited multi-meaning tokens.
- **Collateral rule:** MMLU-frac ≥ 0.9 and collateral ≥ 0.8; maximise H3v = HM(efficacy, specificity, collateral); ties go to the smaller r.
- **E3:** the same rule over grid points with r ≤ 2.5 and T1 copy ≥ 0.9 (2 carriers, ≤ 60 words, seeded sample).
- **Test pass:** qa_test, simdom_test, MMLU test 1000, and collateral test, renormalised by the unedited model's per-token test accuracy.

**Integrity.**
- E2 `measure` / `static`, with each run's existing orig / nonce / del references.
- Word lists come from the parent's existing r = 1 file (CINDER-z for MEAN-C; EMBER for EMBER-c and MEAN-E), so new and old configs share words.
- Exception: Ancient Rome has > 60 words, so contrasts there use the pairwise common words.

**Floors.** `last_mile.unreadable` on qa_train and qa_test, unedited model, with the token set's tokens replaced by nonces or deleted (question and options). Floor efficacy = 1 − mc_norm(acc, unedited acc).

**Decomposition (amendment 1).**
- At CINDER-z's grid point r (δ_z = r / median μ_z), CINDER(δ_z) sits at r_c = r · median μ_c / median μ_z on its own grid.
- MEAN-C(δ_z) sits at r_m = r · median(|βμ_B|/|e|) / median μ_z on its own grid.
- Efficacies are interpolated linearly in log r. The ratios are of medians, so per-token lengths match only approximately. The same construction is used for EMBER.

**S2.** Read gain = |[W_q; W_k; W_v](g ⊙ û)| divided by the mean over 500 random unit directions (seed 0) under the same g; median over edited tokens; concept means over the WP6 runs.

**S4.**
- **Carrier:** BOS + "Yesterday my cousin kept talking about" + the token + " while we cooked dinner."
- **Perturbation:** −s|e_t|u at the token's input (inputs_embeds, with the model's √d scaling applied).
- **Measures:**
  - l0_write = |Δ(h1 − h0)| / |Δh0|.
  - l1_after = mean over later positions of |Δh1| / |Δh0|.
  - KL(p_orig ‖ p_edit) of the next-token distribution, at the token and averaged over later positions.
  - Each is divided by the token's mean over 4 random unit directions.

**Per-run tables:** `analysis.json` (keys curves, c1, c2, copy, integrity, selected, floors, floor_excess, excess_vs_damage, ember_small_r, s2s4, checks) and `fig_*.csv`.

## Figures

All in `/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/paper_analysis/mean_edit/`, as PDF and PNG, with the plotted data in a CSV of the same name.
- `fig_mean_curves.pdf` — validation efficacy (top) and collateral (bottom) vs edit size r, mean over runs.
  - Left: CINDER family (CINDER, CINDER-z, MEAN-C, Noise-C; 7 runs). Right: EMBER family (EMBER-c, EMBER, MEAN-E, Noise-E; 6 runs).
  - Dashed line: deletion floor of the token set. Grey line: the collateral gate at 0.8.
- `fig_eff_vs_damage.pdf`
  - Left: efficacy vs the copy rate of the edited words along r ≤ 2 (each marker is a grid point, mean over runs).
  - Right: efficacy vs context-MC (an unrelated MMLU question after a sentence containing the edited word) at r ∈ {0.25, 1, 10, 100}. Vertical lines mark the real-word reference and chance.
- `fig_s2_s4.pdf`
  - Left: layer-0 read gain of μ̂_B and the edits under the actual norm weights, g = 1, and the top-8 g set to 0.
  - Middle: S4 measured layer-0 write gain vs the linear proxy (s = 0.01, 50 tokens × 3 directions, relative to random).
  - Right: next-token KL at the edited token relative to random, at s = 0.01 (solid) and s = 1 (light).

## Sources

- **Code** (`experiments/paper_analysis/mean_edit/`): `sets.py` (edit sets and the patch of the pipeline's edit sets), `run.py` (GPU stages), `queue_worker.py` (claim-based task queue), `analyze.py`, `figures.py`, `s2s4.py`, `local.sh`, `local_worker.sh`.
- **Outputs** (`results/paper_analysis/mean_edit/`):
  - `PREREG.md`, `analysis.json`, `pilot.json`, `s2_read_gain.json`, `s4_forward.json`
  - `runs/<concept>/s<seed>/<variant>/{grid, copygrid, result_sel, result_e3}.json`
  - `integrity/<concept>/s<seed>/*.json`, `floors/*.json`
  - `logs/`, `slurm/`, `queue/`
- **Inputs (read only):** `results/varwidth/layers/collateral/` (edit sets, grids, results, E2 files and references), the frozen `data/collateral_questions.json`, preps and fits, `results/paper_analysis/edit_direction/` (definitions), gemma-2-2b-it.
- **Compute:** 2 SLURM GPU workers (job 1068899, about 5.4 h each) and local c-001 GPU 0 (about 5.7 h); CPU jobs 1068896 (S2) and 1068897 (S4). Nothing is still running, and nothing was committed.