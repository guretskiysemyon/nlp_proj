# WP6 edit_direction: why CINDER's mapped-back edit is read weakly, and what CINDER-z changes

2026-10-22 · CPU only, no Gemini (0 NIS). Code in `experiments/paper_analysis/edit_direction/`, outputs here.
(Saved by the lead from the agent's hand-back; content unchanged except formatting.)

## Question

The brief's mechanism for CINDER-z: "whitening inflates low-variance directions, so the mapped-back edit lies where
the model's first layer barely reads." Explain the read-gain gap (CINDER 1.0–1.2× random vs EMBER 1.7–2.0×,
CINDER-z 1.8–2.3×) across concepts.

**Short answer: the brief's mechanism does not hold.**

- The mapped-back and re-solved dictionaries differ by one rank-one term along the background mean μ_B.
- The cause is centering, not the metric. The CS transform subtracts μ_B before the fit, so the mean goes into
  the intercept and never into the features. The raw re-solve (CINDER-z) has no intercept, so every feature takes
  on a share of μ_B.
- Layer 0 reads μ̂_B at 2.9× a random direction (Gemma) and 20× (Llama); this one direction accounts for the whole
  read-gain gap. With μ̂_B projected out, EMBER's, CINDER's and CINDER-z's edits are read the same and sit in the
  same part of the spectrum.
- Low-variance directions are read more strongly than average, not less.

## Method

- Gemma-2-2B, 38 concepts × seeds 42–46. EMBER = fit L0 + TJ; CINDER = fit L12 + HJ (F_C from saved
  `layer_cache.json`). 103 runs (30 concepts) with an EMBER edit, 137 (31) with a CINDER edit, 92 (27) both.
- Edit sets match the cached collateral edit sets (204/204 sets, cos ≥ 1 − 5e-15); read gains reproduce D16's
  `a3z_offline.json` within 2.4e-6.
- Variants: EMBER; EMBER-c (EMBER's Y, F_C, Z re-solved on the centered A − μ_B1ᵀ); CINDER (Z̃ mapped back with
  T⁻¹); CINDER-z (Z_o = AYᵀ(YYᵀ+λI)⁻¹ on raw A, λ = 1e-4).
- Read gain (D16 definition): ‖[W_q;W_k;W_v](g ⊙ ĉ)‖ at layer 0, g = 1 + input-norm weight, relative to the mean
  over 500 random unit directions (seed 0); median over edited tokens. Also MLP and combined reads.
- Spectrum (Σ_B, μ_B from `bg_wiki_stats.pt`): energy per eigen-decile (1 = largest), Rayleigh ratio
  vᵀΣ_Bv/(‖v‖²λ̄), energy along μ̂_B; all also for the μ̂_B-projected edit.
- Alignment: cos(c_t, e_t); 1 − ‖e_t − c_t‖/‖e_t‖ at δ = 1; share of c·e through μ̂_B.
- Random clouds: 2,304 eigenvectors; 3,000 spectrally tilted Gaussian directions (α ~ U[−3,3]); 1,000 mixtures
  √(1−s) r⊥ + √s μ̂_B; 2,000 isotropic; 5,000 vocabulary rows.
- Llama-3.1-8B: saved ML-EMBER (A2) and ML-CINDER (A3) fits (read only), that study's read definition
  ([q;k;v;gate;up]); 21 concepts with A3 edits, 10 also with A2.
- Statistics: concept = unit; mean difference, bootstrap 95% CI (10,000), Wilcoxon signed-rank.

## Backed findings

**F1. CINDER-z = CINDER + β_t μ_B, exactly.** In all 137 runs, c^z − c has ≥ 0.9995 (median 1.0000) of its energy
along μ̂_B. The mapped-back dictionary equals the centered least-squares dictionary for the same Y
(‖Z_map − Z_c‖/‖Z_c‖ median 1.9e-4, max 8e-3; edits agree to cos ≥ 0.99997). cos(CINDER-z, CINDER) median 0.79
(0.48–1.00); ‖c^z − c‖/‖c‖ median 0.88.

**F2. The read-gain gap is the μ_B component** (concept means; ⊥μ = μ̂_B projected out):

| Comparison (n) | Read a / b | Δ [95% CI], wins, p | μ̂_B share a / b | Read ⊥μ a / b | Δ⊥μ, p |
|---|---|---|---|---|---|
| EMBER vs CINDER (27) | 1.86 / 1.18 | +0.68 [0.60, 0.77], 27/27, 1.5e-8 | 0.44 / 0.05 | 1.31 / 1.32 | −0.01 [−0.10, 0.08], 0.44 |
| CINDER-z vs CINDER (31) | 1.99 / 1.18 | +0.82 [0.76, 0.87], 31/31, 9e-10 | 0.48 / 0.05 | 1.32 / 1.32 | 0 |
| EMBER vs EMBER-c (30) | 1.88 / 1.10 | +0.78 [0.71, 0.86], 30/30, 1.9e-9 | 0.47 / 0.08 | 1.33 / 1.33 | 0 |
| EMBER-c vs CINDER (27) | 1.09 / 1.18 | −0.08 [−0.13, −0.04], 5/27, 1.1e-4 | 0.07 / 0.05 | 1.31 / 1.32 | −0.01, 0.44 |
| CINDER-z vs EMBER (27) | 2.02 / 1.86 | +0.16 [0.08, 0.24], 21/27, 7e-4 | 0.48 / 0.44 | 1.32 / 1.31 | +0.01, 0.44 |

Over 480 run × variant edits, read gain tracks the μ̂_B share (Spearman 0.75) more than the Rayleigh ratio (0.63).
MLP and combined reads move the same way (combined: EMBER 1.26 vs CINDER 0.93; CINDER-z 1.34 vs 0.94).

**F3. Whitening does not move the edit toward low-variance directions.** With μ̂_B included, EMBER looks
higher-variance (Rayleigh 45 vs 8.7; decile-1 energy 0.62 vs 0.43), all because of μ̂_B (90% of its energy in
decile 1). With μ̂_B projected out: Rayleigh 4.3 vs 4.4 (p = 0.99), read 1.31 vs 1.32 (p = 0.44), decile profiles
overlap. EMBER-c vs CINDER: Rayleigh 9.6 vs 8.7 (p = 0.50). CINDER has less energy in the lowest decile than EMBER
(0.009 vs 0.027, 26/27).

**F4. Layer 0 reads low-variance directions more strongly than average.** Eigenvector read gain is U-shaped:
decile 1: 1.08; deciles 2–8: 0.81–0.88; decile 10: 1.59; lowest 1%: 3.4. Low-variance tilts (α < −2): 5.3×;
high-variance tilts: 0.95–1.07×; across the tilted cloud read falls with Rayleigh ratio (Spearman −0.74). Llama:
decile 1 1.12, 2–8 0.68–0.83, decile 10 1.63. Strongly read directions concentrate on a few coordinates with
large layer-0 norm weight g (up to 6.0; median 1.17): the 8 largest-g coordinates hold 36% of μ̂_B's energy and
24% of the lowest-1% eigenvectors' (random: 0.35%). On Gemma μ̂_B lies mostly on the top PC (energy 0.67;
λ₁ = 129 λ̄); on Llama it does not (0.06), yet is read at 20×.

**F5. Alignment with the token is mostly the μ component.**

| | EMBER | CINDER | CINDER-z | EMBER-c |
|---|---|---|---|---|
| cos(c_t, e_t) | 0.50 | 0.30 | 0.50 | 0.35 |
| Share of c·e through μ̂_B | 0.56 | 0.12 | 0.58 | 0.18 |
| cos(c_t, e_t), μ̂_B removed from both | 0.31 | 0.29 | 0.29 | 0.31 |
| Fraction of ‖e_t‖ removed at δ = 1 | 0.117 | 0.043 | 0.101 | 0.063 |
| ‖c‖/‖e‖ | 0.36 | 0.23 | 0.33 | 0.27 |
| β = c·μ_B/‖μ_B‖² | 0.66 | 0.05–0.07 | 0.56 | 0.14 |

With μ̂_B removed the cosine does not differ (p = 0.15). The edited tokens' own μ coordinate is ≈ 1.06 (median):
uncentered edits remove more than half of each token's mean component at δ = 1; CINDER about 5%.

**F6. The proposition "mapped-back is optimal in the transform's metric, not the Euclidean one" is false.** For
fixed Y the least-squares dictionary is the same in every metric (Prop. 1). The two dictionaries are optimal for
different models: CINDER for the affine model e_t ≈ μ_B + Z y_t, CINDER-z for the linear e_t ≈ Z y_t. The affine
model reconstructs raw embeddings better: Gemma all V\* 0.917 vs 0.932 (137/137), edited concept tokens 0.821 vs
0.845 (135/137); Llama 0.970 vs 0.976 (21/21). Without μ_B the mapped-back dictionary gives 0.971 (Gemma).

**F7. Llama shows the same decomposition** (combined read, seed 42):

| Edit | Read range | Median | n |
|---|---|---|---|
| ML-EMBER (A2) | 2.6–7.6 | 5.7 | 10 |
| A2, centered Z | 1.1–2.6 | 1.7 | 10 |
| ML-CINDER (A3) | 1.0–1.8 | 1.24 | 21 |
| A3, centered Z | 1.0–2.1 | 1.37 | 21 |
| A3z | 2.7–10.4 | 5.4 | 21 |

A3z − A3 = +4.48 (21/21): +4.34 is the μ term (energy share on μ̂_B 1.000 in 21/21), +0.14 the sample-to-full
extrapolation. cos(A3z, A3) 0.84–0.94 (median 0.90). μ̂_B is read at 20.4× (25.7× qkv only), so a 7% share
suffices. After centering, ML-EMBER is still read somewhat more than ML-CINDER (1.76 vs 1.33, 9/10, p = 0.014).
Values match DIAGNOSIS.md within ~0.01.

## Speculations (each with a test)

- **S1. Much of the "stronger" uncentered edit is generic, not about the concept.** Subtracting β μ_B makes a
  token atypical. Consistent with D16 (CINDER-z and EMBER destroy words at the paper-rule points: 0.97, 0.99 vs
  0.76 for CINDER) and with CINDER-z's significant context damage. *Test (GPU erasure, not launched):* on the 8
  collateral runs, evaluate a pure-mean edit e_t − β_t μ_B on CINDER's tokens (β from CINDER-z), against CINDER and
  CINDER-z; if it reproduces most of CINDER-z's efficacy gain at r ≥ 1 and its destruction labels, the gain is
  not concept-specific.
- **S2. Read gain reflects the layer-0 norm weights on a few outlier coordinates.** *Test:* recompute with g = 1
  or the top-8 g coordinates zeroed.
- **S3. The "norm puzzle" may be this μ term:** in the raw fit a feature whose tokens share the mean component with
  the same sign carries ≈ √w ‖μ_B‖ of norm from μ alone. *Test:* split ‖z_i‖² into its μ̂_B part and the rest in L0.
- **S4. Read gain is a crude proxy** (linear layer-0 sensitivity, ignores renormalization and later layers).
  *Test:* forward passes measuring the layer-1 residual change (or the token's own logit) per unit ‖δc‖.

## Paper paragraph

Re-solving CINDER's dictionary in the original space with its coefficients fixed changes the edit by a single
rank-one term along the background mean μ_B. For fixed coefficients the least-squares dictionary does not depend
on the metric, so the map back already yields the Euclidean optimum for the centered embeddings; the CS transform
centers at μ_B, so the mean is absorbed by an intercept and never enters the features. The uncentered re-solve
distributes the mean over the features and adds β_t μ_B to every edit (median β = 0.56 at δ = 1; 48% of the
edit's energy on Gemma), and the first layer reads this direction 2.9× (Gemma) and 20× (Llama) more strongly than
a random direction. This component accounts for the entire read-strength gap: with μ_B projected out, EMBER's and
CINDER's edits are read alike (1.31× vs 1.32×, 27 concepts) and occupy the same part of the background spectrum,
and EMBER's own edit re-solved on centered embeddings is read like CINDER's (1.09×). Whitening therefore
determines which tokens and coefficients the edit uses, not where the edit lies in embedding space.

## Appendix text

**Setup.** A ∈ ℝ^{d×n}: TL-scaled embeddings of V\*; A_c = A − μ_B1ᵀ. CS fits X̃ = TᵀA_c (row form X = (E − μ_B)T).
Each `core.py` iteration: Z̃ ← X̃Yᵀ(YYᵀ+λI)⁻¹; Y ← mask ⊙ (Z̃ᵀZ̃+λI)⁻¹Z̃ᵀX̃; unit-norm rows. Map back
Z_map = T^{−ᵀ}Z̃. Edit c_t = Σ_{i∈F_C} Y_it z_i.

**Proposition 1 (the dictionary step does not depend on the metric).** For invertible T, B ∈ ℝ^{d×n},
Y ∈ ℝ^{k×n}, λ ≥ 0, YYᵀ + λI ≻ 0: argmin_Z ‖Tᵀ(B − ZY)‖²_F + λ‖TᵀZ‖²_F = BYᵀ(YYᵀ+λI)⁻¹, independent of T.
*Proof.* Z ↦ Z̃ = TᵀZ is a bijection; in Z̃ the objective is ‖TᵀB − Z̃Y‖² + λ‖Z̃‖², strictly convex, stationary at
Z̃\* = TᵀBYᵀ(YYᵀ+λI)⁻¹; hence Z\* = T^{−ᵀ}Z̃\* = BYᵀ(YYᵀ+λI)⁻¹. ∎ (For λ = 0 it holds for any M = TTᵀ ≻ 0.)

**Corollary 1.** At a fixed point of the Z-step, Z_map = A_cYᵀ(YYᵀ+λI)⁻¹ =: Z_c: the mapped-back dictionary is
optimal in both metrics for the affine model e_t ≈ μ_B + Z y_t. (Gemma: ‖Z_map − Z_c‖/‖Z_c‖ ≤ 8e-3; Llama holds up
to the sample-to-full extrapolation, cos 0.92–0.96.)

**Corollary 2 (CINDER-z adds the mean).** Z_o = AYᵀ(YYᵀ+λI)⁻¹ = Z_c + μ_B bᵀ with b = (YYᵀ+λI)⁻¹Y1_n, hence
c^z_t − c_t = β_t μ_B, β_t = Σ_{i∈F_C} b_i Y_it. The μ_B-orthogonal part of the edit is unchanged.

**Corollary 3.** CINDER's edit lies in span{e_s − μ_B : s ∈ supp Y_{F_C}}; CINDER-z's in that span plus μ_B. The
transform's metric enters only through the coefficient step (an M-orthogonal projection, M = TTᵀ), the supports,
the features and the judge's F_C: whitening chooses which tokens are grouped and how they are weighted, not where
the edit can lie.

**Remark (size of the mean term).** With near-disjoint unit-norm rows, YYᵀ ≈ I and b_i ≈ Σ_t Y_it; for a feature of
width w with same-sign entries ≈ 1/√w, b_i ≈ √w and β_t ≈ 1 whatever the width, so at δ = 1 the uncentered edit
subtracts about one full μ_B and the feature gains ≈ √w‖μ_B‖ of norm from the mean alone. Observed median β:
EMBER 0.66, CINDER-z 0.56, CINDER 0.05; the tokens' own μ coordinate 1.06.

**Per-concept tables:** see `gemma_runs.csv` (per run), `summary.json`, `llama_runs.json`.

## Figures

- `fig_spectral_deciles.pdf` — energy share per Σ_B eigen-decile (1 = highest variance), concept mean with
  bootstrap band; left: edits (EMBER and CINDER-z concentrate in decile 1, which holds 90% of μ̂_B); right: μ̂_B
  projected out (EMBER, EMBER-c and CINDER = CINDER-z coincide).
- `fig_read_vs_rayleigh.pdf` — layer-0 read gain of unit directions vs Rayleigh ratio (left) and μ̂_B share
  (right); clouds of tilted, eigen, mixture directions; method markers; μ̂_B star.
- `fig_read_decomposition.pdf` — per-concept read gain of each variant, Gemma (left) and Llama (right).

## Sources

`experiments/paper_analysis/edit_direction/{common,gemma,llama,summarize}.py`; outputs `gemma_runs.{json,csv}`,
`gemma_cloud.json`, `check_collateral.json`, `token_mean_coordinate.json`, `llama_runs.json`, `summary.json`,
`log_*.txt`. Inputs (read only): Gemma fits/caches, preps, `bg_wiki_stats.pt`, model weights (HF cache), collateral
`edits_s*.pt`, `diag/a3z_offline.json`, `../nlp_proj_ml/results/ml/{runs, data/bg_ml_llama.pt, diagnose_a3}`.
Not done: erasure runs (S1) and forward-pass tests (S4).
