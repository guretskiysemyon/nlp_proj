<!-- Saved by the lead from the WP agent's hand-back (a hook blocked the agent's own write); text unchanged. -->

# WP1 `quiet_concept`: is the concept "quiet", and what does CS change?

## Question

From the motivation document:

- "Motivation: Harry Potter concept direction has low variance in the original training data – they were trained less, so they got less rich representation."
- "We can see that after removing the background biases, while embracing the concept biases we can emphasize the Harry Potter feature direction ... Todo: understand the exact mathematics behind it."
- "Whitened against background: no Harry Potter direction found. Whitened against background and sharpened against concept page: Harry Potter direction found."
- "For contrastive, the embedding matrix factorizes much more 'sparsely' ..." (HP: SVD rank matching the sparse-MF error 67 with CS vs 5 raw).

Lead's add-on (from WP6): add a centring-only control and the raw matrix with μ̂_B projected out, and split the raw → CS gain into centring, whitening and sharpening.

## Method (short)

- **Matrices** (Gemma, 38 concepts, V\* from the prep cache): raw `E_V*ᵀ` (EMBER); raw⊥μ̂ `(I − μ̂μ̂ᵀ)E_V*ᵀ`, where μ̂ = μ_B/‖μ_B‖; centred `(E_V* − μ_B)ᵀ`; whiten-only (γ = 0) `W(e − μ_B)`; CS (the prep "whitened" matrix, η = 0.05, γ = 1).
- **Fits.** EMBER's RS-1% fit (`fixed`, k = 100, refit off), 38 concepts × seeds 42–46 for every matrix. Raw, CS and CS+GS fits were already saved. I ran 3 × 190 new fits: whiten-only, centred and raw⊥μ̂, all on CPU or one idle local GPU. No γ = 0 fits existed.
- **Judge-free concept feature.** For each fit, the feature with the most lexicon hits (its first 20 distinct source words matched against the 20 words in `concept_words_v1.json`; ties go to the better ρ rank). A run counts as "found" when that feature has ≥ 5 hits (EJ@5). ρ is EMBER's `ratio_abs`. No Gemini calls.
- **Selection-free concept directions** in each matrix: md = mean of the concept-only columns minus the mean of the neutral-only columns; ge = top generalized eigenvector of (M_C, M_N + 0.1·tr(M_N)/d·I); lex = top left singular vector of the lexicon tokens (concept-only tokens whose main source word is in the lexicon). For each direction I report its energy share, its PCA rank (where its energy would sit in the eigenvalue spectrum of AAᵀ) and the purity of its top-m tokens (m = ⌈0.01n⌉ = 48 or 49).
- **Seeded single feature.** A rank-1 hard-thresholded ALS started on the lexicon tokens (u ← top singular vector of A_S; S ← the top-m tokens by |uᵀa_t|), iterated to its fixed point. It reports purity, and value σ₁(A_S)² as a percentile of the feature energies ‖z_i‖² in that matrix's saved fits.
- **SVD-rank matching.** The smallest r with √(1 − Σ_{j≤r}λ_j/ΣAᵀA) ≤ the saved fit's rel_err, per seed.
- **Frequency.** 6.18 M tokens (5,632 articles in seeded random order) of the cached `wikimedia/wikipedia` 20231101.en shard 40, tokenized with Gemma's tokenizer.
- **Statistics.** The concept is the unit (median over its 5 seeds). Tests are paired Wilcoxon signed-rank with a bootstrap 95% CI of the median difference. n = 38, with the 18 EMBER concepts checked separately.

## Backed findings

**B1. The exact mathematics (proofs in A.1, numerical checks in `math_checks.json`).**
- (i) EMBER's ALS depends on A only through the token Gram matrix AᵀA. A fit on A and on QA (Q random orthogonal) gives identical supports, with max |ΔY| = 5·10⁻⁷ after 60 iterations.
- (ii) Under CS the Gram matrix becomes a page kernel, (1 − η)·mean_{j∈page} k(s,j)k(t,j) + η·k(s,t), where k is the background-Mahalanobis inner product. Relative error ≤ 1.4·10⁻⁸ on HP, Ancient Rome and Valentine's Day.
- (iii) The sharpening gains are g_j = (1 − η)λ_j + η, where λ_j are the generalized eigenvalues of (M_page, Σ_B + εI). Error ≤ 10⁻¹⁴. The energy along each p_j is multiplied by exactly g_j (error ≤ 1.4·10⁻⁸).
- Across concepts, g₁ has median 15.7 (range 10.9–29.3), and 870 of 2,304 directions have g > 1.

**B2. In EMBER's matrix the concept is not quiet. It is loud, because of μ_B (this resolves the norm puzzle).** n = 38.
- Concept-only tokens carry the same energy per token as neutral-only tokens: ratio 1.00 raw, 0.99 whiten-only.
- The concept feature's ‖z‖² sits at the 86th percentile of its fit (HP 95th). Its direction has PCA rank 2 and 5.1% of the energy (90th percentile among the fit's features; found runs).
- 42% of its ‖z‖² lies along μ̂_B, against 12% for the median feature (37/38 concepts, p = 1.5·10⁻¹¹).
- Its support tokens are the rarest in V\*: median 2.5 vs 23 per million for the other features' supports, the 3rd percentile, 38/38 concepts. Their projection on μ̂_B is 2.8× larger (38 vs 13.6, 98th percentile, 38/38).
- Removing μ_B removes the effect. The ‖z‖² percentile falls to 41 with centring (38/38 concepts lower, p = 8·10⁻⁸) and to 66 with raw⊥μ̂ (35/38, p = 6·10⁻⁷).

**B3. Without μ_B, the concept is quiet relative to other features (the refined form of the TODO).**
- In the centred fits, the concept feature's direction has the 14.5th-percentile energy among its fit's 100 features (35/38 below the median, p = 4·10⁻⁷).
- Its atom lies along low background variance relative to the other atoms: 13th percentile (32/38, p = 5·10⁻⁶). Whitening lifts its energy percentile to 72.5.
- The selection-free ge direction has background variance 1.05× that of a random direction (rank 625 of 2,304).

**B4. Frequency: concept tokens are rarer, and rarity is the μ̂_B axis, not low variance.**
- Concept-only tokens are rarer than neutral-only tokens: median 9.4 vs 12.9 per million, 35/38 concepts, p = 1.7·10⁻⁷.
- Their norms are not smaller: raw ‖e‖² is 0.9% larger (31/38) and whitened energy is 0.9% larger.
- Across the vocabulary (98,940 seen ids), the projection on μ̂_B falls monotonically with frequency, from about +40 below 1 per million to about −48 at 10⁴ per million. Spearman is −0.38 over the vocabulary, −0.58 over V\* ids and −0.69 within a concept.
- Rare tokens put a smaller share of their centred energy on the top-10 background PCs: 0.07 vs 0.65 at the frequent end (within-concept Spearman +0.53). For concept-only vs neutral-only tokens this share is 0.111 vs 0.123 (lower in 38/38 concepts). About half of that gap remains within frequency quintiles.
- So the data support "seen less": concept tokens are rarer, and the tokens a concept feature collects are the rarest. They contradict "less variance". Rare Gemma embeddings share a large common component along μ̂_B (which per WP6 lies mostly on the top background PC), so they look alike, not small.

**B5. Decomposition, raw → centring → whitening → sharpening** (Table A.3).
- **Norm puzzle:** centring or μ̂_B removal (86 → 41 / 66).
- **SVD-rank matching:** mostly whitening. Raw 4, centred 4, raw⊥μ̂ 22, whiten-only 66, CS 49, CS+GS 52 (each step 38/38). HP is 5 → 49, as in `CINDER_improvements.md`. The brief's "67" matches HP whiten-only (66), not CS. The mechanism is the spectrum: raw has a top-PC share of 10.6% and participation ratio 75; whiten-only is flat (PR 1,283); CS re-concentrates on page directions (PR 249). The sparse MF explains 16% / 11% / 26% of the energy (raw / whiten-only / CS).
- **Purity of the concept's own direction:** removing μ̂_B is the step. The top-m concept-only share along the lexicon direction is raw 0.60, raw⊥μ̂ 0.91, centred 0.12, whiten-only 0.90, CS 0.92. Subtracting μ_B does not help, because rare tokens keep a large μ̂_B component after centring. CS adds nothing over whitening here (p = 0.15).
- **Seeded single feature:** raw drifts away (purity 0.24, keeps 0% of the lexicon tokens); centred 0.04; raw⊥μ̂ 0.27. Whiten-only makes the concept a stable, pure cluster (purity 0.86, keeps 41%), but its value is ordinary (70th percentile). Sharpening raises the value to the 91st percentile (37/38, p = 2·10⁻⁷) at the same purity.
- **Energy of the lexicon direction:** 3.1× an average direction whiten-only (PCA rank 49.5) vs 11.4× with CS (rank 16.5), 38/38.
- **Fits, share of runs with a lexicon concept feature:** raw 56%, raw⊥μ̂ 61%, centred 59%, whiten-only 49%, CS 69%. Per concept, raw → CS is 15 up / 3 down (p = 0.011); whiten-only → CS is 17 up / 0 down (p = 3·10⁻⁴); raw⊥μ̂ → CS is 12 / 4 (p = 0.039); raw → centred and raw → raw⊥μ̂ are n.s. On the 18 EMBER concepts, raw → CS is 6 up / 1 down (p = 0.15, n.s.) and whiten-only → CS is 9 up / 0 down (p = 0.007).
- **Fits, neutral-only tokens in the concept feature:** 4.5 / 4.5 / 4.5 / 4.0 / 3.0 (raw → CS 30 lower / 1 higher, p = 7·10⁻⁶; whiten-only → CS 23 / 5, p = 6·10⁻⁴; raw⊥μ̂ → CS 28 / 1, p = 1.4·10⁻⁵).
- **In short:** centring or μ̂_B removal explains the norm puzzle and part of the spectrum change. Whitening explains most of the SVD-rank change and gives a pure but unprofitable concept cluster; on its own it finds fewer concept features than raw (lexicon hits lower in 18 concepts, higher in 11; p = 0.017). Sharpening is the step that finds more concept features and cleaner ones.

**B6. The HP "11th by ρ" does not generalize.**
- Over 190 raw runs, when a lexicon concept feature exists its ρ rank is 1 (median). The ρ rank of the max-lexicon feature is 2 raw vs 1 CS (13 better / 12 worse, p = 0.92).
- HP raw is seed-dependent, with ranks 12, 1, 28, 48, 1. CS ranks HP 1st in all 5 seeds (45–46 concept-only tokens, 0–1 neutral).
- Whiten-only on HP does find a feature: in 4/5 seeds a merged magic + HP feature at ρ rank 1 (ρ 48–83; top words magic, wizard, witchcraft, Hogwarts, Voldemort). "Whitened: no HP direction found" is therefore not reproduced. Sharpening separates HP from magic.

**B7. The energy rank of selection-free concept directions does not generally rise with CS.** n = 38, raw → CS.
- md: rank 3 → 5 (share 2.0% → 1.25%, p = 1·10⁻⁴).
- ge: rank 120 → 219.
- lex: rank 2 → 16.5. Raw md and lex are high-energy only because they align with μ̂_B and the top PCs: their top-m tokens are 9% and 60% concept-only. With μ̂_B projected out, lex sits at rank 9 (raw⊥μ̂) vs 16.5 (CS), with a similar share (0.44% vs 0.50%, p = 0.24).
- What CS changes is the energy relative to whitening (lex 3.1× → 11.4×) and the value of the concept's m-token cluster. It does not move a single concept direction to the top of the spectrum.

**Aside, for the lead.** GS makes the concept feature 1st by ρ less often than CS (18% vs 38% of runs; 18 concepts lower vs 2 higher, p = 0.002). Its support holds more concept tokens (54 vs 41.5). Judge-selected F_C from the saved summaries is non-empty in 54% (raw), 58% (CS) and 56% (CS+GS) of runs.

## Speculations (with tests)

- **S1. The rarity axis μ̂_B is the tied-embedding "representation degeneration" of rare tokens** (Gao et al. 2019). Test: plot projection vs frequency for Llama's untied input embeddings; WP6 notes that μ̂_B is not on Llama's top PC.
- **S2. Part of sharpening's gain is circular.** CS raises concept-only token energy to 1.91× neutral by construction of the page target. Test: rerun the wo → CS step of B5 (found rate, seeded value) with a leak-free target (the page minus V\*, or a held-out half of the page).
- **S3. The concepts never "found" are a lexicon/vocabulary mismatch, not a geometry failure.** Seven concepts are found in no matrix (e.g. World War II, The Matrix, Sigmund Freud), even though their seeded clusters are pure in whiten-only/CS. Test: hand-label (blind) the max-hit features of those concepts.
- **S4. The lexicon-seeded probe overstates raw drift,** because it ignores competition from the other 99 features. Test: seed the concept feature inside a full fit (initialize one row of Y on the lexicon tokens) and track its purity.

## Paper paragraph

Why does contrastive sharpening help sparse factorization find a concept? Squared-error sparse MF with a fixed width spends a feature where m tokens are jointly energetic and aligned, and its alternating fit depends on the embeddings only through the token Gram matrix. CS replaces that Gram matrix by a page kernel: two tokens are similar when they resemble the same tokens of the concept page in the background metric, and the energy along each contrastive direction is multiplied by a gain that equals the page-to-background Rayleigh ratio. Across 38 concepts and 5 seeds, concept tokens in Gemma's raw embeddings are not low-energy. They are rarer than neutral tokens (median 9.4 vs 12.9 per million), and rare tokens share a large component along the background mean, which makes the raw concept feature one of the highest-norm features (86th percentile) and mixes it with unrelated rare tokens. Whitening removes this shared direction and makes the concept's tokens a pure, stable cluster, but on its own it lowers their value and finds fewer concept features (49% of runs vs 56% raw). Sharpening raises that value (from the 70th to the 91st percentile of feature energies) and yields a lexicon-matching concept feature in 69% of runs, with fewer neutral tokens (3 vs 4.5). It does not change the ρ rank of concept features that EMBER already finds.

## Appendix text

### A.1 Mathematics

**Notation.**
- A = [a_1, …, a_n] ∈ ℝ^{d×n} is the factorized matrix (EMBER: a_t = e_t; CS: a_t = x_t = S_η^{γ/2}W(e_t − μ_B)).
- A ≈ ZY with Z ∈ ℝ^{d×k} and Y ∈ ℝ^{k×n}. Each row y_i of Y has m = ⌈0.01n⌉ nonzeros and unit norm.
- K = AᵀA is the token Gram matrix. E_A(u) = Σ_t(uᵀa_t)² is the energy along a unit u, and V_m(u) is the sum of the m largest (uᵀa_t)².
- Σ = Σ_B + εI and W = Σ^{-1/2}, so ũ_t = W(e_t − μ_B).
- P is the set of page token types (|P| = m_P). M_P = (1/m_P)Σ_{j∈P}(e_j − μ_B)(e_j − μ_B)ᵀ.
- S_η = (1 − η)WM_PW + ηI = Σ_j g_j p_j p_jᵀ, and T = W S_η^{γ/2}.

**Lemma 1 (Gram dependence).** Let G = YYᵀ + λI. Then Z = AYᵀG⁻¹, ZᵀZ = G⁻¹YKYᵀG⁻¹ and ZᵀA = G⁻¹YK. So the dense solve Y_dense = (ZᵀZ + λI)⁻¹ZᵀA, the per-row top-m mask, the row normalization and ‖A − ZY‖² = tr K − 2⟨ZᵀA, Y⟩ + ⟨ZᵀZ, YYᵀ⟩ all depend on A only through K. Two matrices with equal Gram matrices give identical Y from the same initialization. An orthogonal map changes nothing, and any preprocessing acts on the decomposition only through XᵀX.

**Lemma 2 (value of one feature).** Fix the other features and let R = A − Σ_{j≠i}z_j y_j. The best feature supported on S (|S| = m) lowers ‖R‖² by

  σ₁(R_S)² = (Σ_{t∈S}‖r_t‖²)·κ(S),  where κ(S) = λ_max(R_SᵀR_S)/tr(R_SᵀR_S) ∈ [1/m, 1].

Over all supports the maximum is max_u V_m(u) (Eckart–Young on R_S, then exchange the two maxima). A fixed-width feature therefore buys energy × coherence. Its fixed point is S = top-m(|uᵀr_t|), with u the top singular vector of R_S.

A concept fails in one of two ways:
- **Unprofitable:** V_m along the concept is below what other directions offer.
- **Impure:** along the concept tokens' best direction, other tokens project more strongly, so the fixed point absorbs them and ρ falls.

**Proposition 1 (page kernel).** Let k(s,t) = (e_s − μ_B)ᵀΣ⁻¹(e_t − μ_B). The Gram matrix the fit sees is:

| Matrix | K(s,t) |
|---|---|
| Raw | e_sᵀe_t |
| Whitening only | k(s,t) |
| CS, γ = 1 | (1 − η)(1/m_P)Σ_{j∈P}k(s,j)k(t,j) + η·k(s,t) |

**Proposition 2 (gains).**
- (a) g_j = (1 − η)λ_j + η, where λ_j are the generalized eigenvalues of (M_P, Σ). In particular λ₁ = max_v vᵀM_Pv / vᵀΣv, attained at v = Wp₁: the Fisher / contrastive-PCA objective. With M_P = Σ_P + δδᵀ and δ = μ_P − μ_B, the numerator counts both spread and mean shift.
- (b) E_X(p_j) = g_j^γ·E_Ũ(p_j), because S_η^{γ/2}p_j = g_j^{γ/2}p_j. The same factor multiplies V_m(p_j).

**Corollary (whitening vs sharpening).** After whitening, the background has identity second moment. On V\* both neutral-only and concept-only tokens carry 1.00 energy per direction (medians 1.008 and 1.001), so no direction is favoured.
- The shared directions that make raw concept features impure (μ̂_B and the top PCs) are scaled down. This is the purity gain (B5).
- No direction is raised either. V_m is then set by small groups of tokens that stick out, and the concept cluster is ordinary (70th percentile).
- Sharpening multiplies each direction's energy by g ∈ [η, g₁], a range of g₁/η ≈ 220–590. By (a), the large-g directions are those where the page, and hence the concept, has much more second moment than the background. The concept cluster's value rises (91st percentile).
- Along a fixed p_j, sharpening scales concept and neutral projections alike, so it cannot change purity there. Purity changes only through the kernel of Proposition 1.

### A.2 Definitions and settings

- **Fits.** `VWConfig(method="fixed", refit=False)`, k = 100, 1% per row, ridge 10⁻⁴, tol 10⁻⁴, patience 200. Seeds 42–46. eps_rel = 10⁻⁴.
- **Found.** The concept feature has ≥ 5 lexicon hits among its first 20 distinct normalized source words (j3 normalization).
- **ρ rank.** 1-based rank of ratio_abs (mean |Y| over concept-only tokens / (mean over neutral-only + 10⁻⁸)).
- **PCA rank of u.** 1 + #{λ_j(AAᵀ) > uᵀAAᵀu}. "×avg" is uᵀAAᵀu / (‖A‖²/d).
- **Seeded probe.** Started on the lexicon tokens (median 31 per concept), at most 100 iterations, stopped when the support repeats.
- **Frequency proxy.** log₁₀((count + 0.5)/6,175,963 × 10⁶).

### A.3 Decomposition (medians over 38 concepts; source: `tables/decomposition.md`)

| metric | raw (EMBER) | raw ⊥μ̂_B | centred | whiten only | CS | CS+GS |
|---|---|---|---|---|---|---|
| runs with a lexicon concept feature (%) | 56 | 61 | 59 | 49 | 69 | 63 |
| runs where it ranks 1st by ρ (%) | 32 | 31 | 29 | 35 | 38 | 18 |
| ρ rank of the max-lexicon feature | 2.0 | 2.0 | 2.0 | 1.0 | 1.0 | 3.5 |
| neutral-only tokens in it | 4.5 | 4.5 | 4.5 | 4.0 | 3.0 | 3.0 |
| its ‖z‖² percentile | 86 | 66 | 41 | 54 | 58 | 56 |
| lexicon direction: PCA rank | 2.0 | 9.0 | 2.0 | 49.5 | 16.5 | 16.5 |
| lexicon direction: energy / average direction | 170.6 | 10.0 | 132.7 | 3.1 | 11.4 | 11.4 |
| lexicon direction: concept-only share of top-m | 0.60 | 0.91 | 0.12 | 0.90 | 0.92 | 0.92 |
| seeded feature: purity | 0.24 | 0.27 | 0.04 | 0.86 | 0.86 | 0.86 |
| seeded feature: value percentile | 100 | 97 | 100 | 70 | 91 | 84 |
| spectrum participation ratio | 75 | 313 | 37 | 1283 | 249 | 249 |
| top PC share of energy | 0.106 | 0.044 | 0.161 | 0.005 | 0.038 | 0.038 |
| SVD rank matching sparse-MF error | 4 | 22 | 4 | 66 | 49 | 52 |
| concept/neutral energy per token | 1.00 | 1.01 | 0.98 | 0.99 | 1.91 | 1.91 |

All paired tests, with CIs and both scopes (38 and 18 concepts), are in `results/paper_analysis/quiet_concept/stats.json` (keys `feature_metrics`, `spectra`, `rho_rank`).

### A.4 Per-concept table

The full table is in `results/paper_analysis/quiet_concept/tables/appendix_table.md`, one row per concept († marks the 20 non-EMBER concepts). Its columns are:
- found counts (of 5 seeds) for raw / ⊥μ̂ / centred / whiten-only / CS / GS;
- the uncensored ρ rank of the max-lexicon feature;
- the ‖z‖² percentile;
- neutral-only tokens in the concept feature;
- the SVD-matched rank;
- the lexicon-direction PCA rank;
- seeded purity and value;
- g₁;
- the median log-frequency of concept-only vs neutral-only tokens.

## Figures (`results/paper_analysis/quiet_concept/figures/`, PDF + PNG + CSV)

- **(a) `fig_direction_rank`** (6.8 in). PCA rank of the energy of three selection-free concept directions (concept − neutral mean; generalized eigenvector; the lexicon tokens' top direction) in raw, raw⊥μ̂_B, centred, whiten-only and CS matrices. Grey lines are concepts and black is HP; large markers are medians. Raw directions rank high only through μ̂_B. Whitening drops the concept direction to rank ~50 and sharpening lifts it to ~17.
- **(b) `fig_rho_rank`** (3.3 in). Share of the 190 runs per matrix in which the judge-free concept feature ranks 1, 2–3, 4–20 or > 20 by ρ, or is not found (no feature with ≥ 5 lexicon hits).
- **`fig_purity_value`** (3.3 in). Fixed point of a single feature seeded on the concept's lexicon tokens: purity (concept-only share) vs value (percentile among the fit's feature energies), per concept. Large markers are medians and the line is HP. Raw and centred are valuable but impure, whiten-only is pure but ordinary, and CS is pure and valuable.
- **`fig_feature_energy`** (6.8 in). Norm puzzle across concepts: the ‖z‖² percentile of the concept feature (left) and the neutral-only tokens in its support (right), per matrix.
- **`fig_svd_rank`** (3.3 in). Truncated-SVD rank matching the sparse-MF error per concept.
- **(c) `fig_frequency`** (6.8 in). Over the Gemma vocabulary (Wikipedia frequency bins; median and IQR): projection on μ̂_B and top-10 background-PC share of centred energy vs log₁₀ frequency (left, middle). Right: per concept, median log-frequency of concept-only vs neutral-only tokens.

## Sources

- Code (`experiments/paper_analysis/quiet_concept/`):
  - `common.py`: loaders and matrices.
  - `fit_whiten_only.py`: whiten-only / centred / raw⊥μ̂ fits, stored in `results/paper_analysis/quiet_concept/fits/<Slug>/fixed_refit-False_{g0,ctr,rawp}/seed*/fit.pt`.
  - `features.py`: per-fit concept feature.
  - `spectra.py`: spectra, directions, seeded probe, SVD rank.
  - `frequency.py`, `norm_puzzle.py`, `math_checks.py`, `aggregate.py`, `figures.py`, `report_tables.py`.
- Inputs:
  - `results/varwidth/prep/*_S300.pt`
  - `results/varwidth/<Slug>/{fixed_refit-False_plain, fixed_refit-False, rn_refit-False, rn_refit-False_plain}/seed42–46`
  - `experiments/whitening/data/gemma/{bg_wiki_stats.pt, defaults.json, targets/}`
  - `experiments/varwidth/j3_data/concept_words_v1.json`
  - HF cache `wikimedia/wikipedia` 20231101.en shard 40
- Outputs (`results/paper_analysis/quiet_concept/`): `features/`, `spectra/`, `frequency/`, `tables/`, `stats.json`, `norm_puzzle.json`, `math_checks.json`, `logs/`.
- Compute: CPU on c-001, a SLURM CPU array `cl_pa_qc_ctr` (cancelled after 15/38 tasks while queued; the rest ran locally), and GPU 0 on c-001 while it was idle (now released). Nothing of mine is still running or queued. Gemini: 0 calls.
- Nothing committed. Files exist only in the two WP folders. REPORT.md itself was not written (hook).
