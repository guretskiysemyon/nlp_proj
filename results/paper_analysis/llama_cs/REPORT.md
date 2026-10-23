<!-- Saved by the lead from the WP agent's hand-back (a hook blocked the agent's own write); text unchanged. -->

# WP8 `llama_cs`: single-language contrastive sharpening (CS) on Llama-3.1-8B-Instruct

## Question

From the motivation document:

> "NOTE: for the 'naïve', single language pipeline, in Llama we have an issue with applying CINDER (contrastive PCA cleaning). Llama's embedding dimension is larger than Gemma (4096 vs 2304). So a concept page may not have enough tokens from the concept page to obtain a full rank matrix required for sharpening. Claude proposed solution with the hyper-parameter (increase eta, or decrease gamma = sharpening strength) – TODO: look into it."

**Short answer.** The rank deficiency is real: 15 of 18 English pages have fewer token types than d = 4096. But it is not what breaks CS on Llama.

- **The rank deficiency is harmless.** The fit ignores the null-space (floor) directions. Pages with m > d (Ancient Rome, Nazism, Pornography) fail in the same way, and Gemma stays fine when its page is cut below d.
- **The real problem is noise.** Llama's whitened page second moment is almost all estimation noise. Only about 12 eigenvalues rise above a same-size random-token null, against about 74 on Gemma.
- **Each page token echoes itself.** Inside the noisy bulk, a page token's own outer product (its self-echo) makes up about half of its sharpened energy. Sharpening therefore boosts page membership rather than concept directions.
- **The symptoms.** About 92% of features pass ρ ≥ 2. They are page-token grab-bags whose cross-model semantic coherence is 4× lower than EMBER's. The lexicon-identified concept feature falls from a geometric-mean ρ-rank of about 10 (EMBER) to about 38.
- **The proposed hyperparameter fix does not work.** Raising η up to 0.5 or lowering γ to 0.5 changes nothing, because the echo sits in the non-null bulk, not on the floor.
- **What works.** Sharpen only the directions above the noise edge. This needs no new hyperparameter, gives a median rank of 5.0 vs 10.3 for EMBER over 18 concepts, and leaves CS on Gemma unchanged.

## Method

**Embeddings and fit sizes.**
- Only the input embedding matrix is read, straight from the safetensors shard.
- Llama-3.1-8B-Instruct: d = 4096, unscaled, k = 200.
- Gemma-2-2B-it: d = 2304, scaled ×√d, k = 100.

**Llama background (new).**
- English Wikipedia token types from the same 300,000 sentences as Gemma's existing background.
- I regenerated the sentence stream with `experiments/whitening/background.py`'s `SentenceSource` (shard 40 of `wikimedia/wikipedia` 20231101.en).
- I found the stopping chunk by matching Gemma's stored type count exactly (107,472 = 107,472).
- Result: 60,467 Llama types, each counted once; ε = 1e-4·mean λ, as for Gemma.

**Targets.**
- Page target: the token types of the English page (`experiments/whitening/data/pages/*.json`, the same source as the Gemma targets).
- Check: re-tokenizing with Gemma reproduces all 18 stored Gemma targets exactly.
- Ancient Rome comes from the multilingual study's raw page, which is the same revision.
- Leak-free (lf) control: the page minus every V\* type.
- ml8 control: the Llama types of the concept's 300 sentences in all 8 languages of the multilingual study.

**V\*.**
- Built with `experiments/varwidth/prep.py` (`build_dataset` / `build_vstar`) and the Llama tokenizer: 300 concept + 300 neutral sentences, neutral seed 42.
- |V\*| = 4,522–5,303, always above d. Harry Potter (HP) = 5,025, which matches `eta_sweep.EXPECTED_VSTAR`.

**Fits.**
- All fits use `experiments/varwidth/core.fit_varwidth` with method `fixed` (RS-1%) and refit False, which is EMBER's fit.
- Inputs: raw E (EMBER), whitened U (γ = 0), and X = U S_η^{γ/2} over η ∈ {0.05, 0.1, 0.2, 0.5} × γ ∈ {0.5, 1}.
- Llama controls: lf, ml8, k = 100, and noise-edge truncation (defined in F5).
- Gemma controls: sub-sampled page, k = 200, lf.
- The full grid ran on 4 concepts (HP, Ancient Rome, Valentine's Day, Baseball) × seeds 42–44.
- All 18 EMBER concepts × seeds 42–44 ran with raw E, γ = 0, the default (η = 0.05, γ = 1) and the truncated rule.
- On Gemma I reused the existing L0/L1 fits and added my own truncated and γ = 0 fits.
- Validation: my pipeline reproduces the stored Gemma HP L1 fit exactly (585 iterations, relative error 0.8581).
- Total: 774 Llama fits and 288 Gemma fits on one local GPU.

**Judge-free metrics (no Gemini calls).** Computed per fit in `llama_cs/metrics.py`:
- **ρ_i:** the mass ratio, as in `ember.utils.compute_embedding_stats`.
- **Lexicon hits lo_i:** how many of the concept's 20 words (`concept_words_v1.json`) appear among the feature's first 20 distinct source words. This is the EJ match from `layer_eval.py`.
- **Concept feature:** the feature with the most lexicon hits among all k features; ties go to the higher ρ. I record its ρ-rank, ρ, width and concept-only purity.
- **R3:** the ρ-rank of the highest-ρ feature with at least 3 lexicon hits.
- **Share ρ ≥ 2:** n_τ / k.
- **AUC:** concept-only vs neutral-only tokens, scored by max |Y| over the ρ ≥ 2 features, as in `eta_sweep.py`.
- **Coherence** (`llama_cs/coherence.py`):
  - Mean pairwise cosine of a feature's 20 source words, embedded in the *other* model's background-whitened input embeddings. Llama fits are scored in Gemma space and vice versa.
  - Averaged over the 20 highest-ρ features, which is the judge window.
  - Random 60-token sets score 0.006 in Gemma space and 0.003 in Llama space.

**Spectra** (`llama_cs/spectra.py`, 18 concepts × 2 models).
- Eigenvalues of M = U_Cᵀ U_C / m and of S_η.
- Noise null: M₀ built from m random background types that are not on the page.
- n_signal: the number of eigenvalues of M above the largest eigenvalue of M₀.
- Ledoit–Wolf shrinkage intensity toward I.
- Split-half reproducibility of the top-10 eigenvectors.
- V\* diagnostics: self-share, the on-page / off-page energy boost, and the share of V\* concept tokens that are on the page.

**Statistics.**
- The unit is the concept: per-concept seed mean, or seed median for ranks. Ranks are analysed on a log₂ scale.
- Tests: Wilcoxon signed-rank, and bootstrap 95% CIs of the mean paired difference over concepts.
- n = 18 unless stated otherwise.

## Backed findings

**F1. Llama pages are below d, but both models have such pages and it is not the failure.**

Page sizes:
- Llama: m = 1,236–4,583 (median 3,157); m/d = 0.30–1.12 (median 0.77). 15 of 18 are below d; the exceptions are Ancient Rome (1.05), Nazism (1.11) and Pornography (1.12).
- Gemma: m = 1,165–4,410; m/d = 0.51–1.91 (median 1.29). 3 of 18 are below d: Gambling 0.51, Cannabis 0.86, Golf 0.92.

The floor is a sample-size effect:
- At η = 0.05, the share of S_η directions at the floor (g < 1.1η) is 0.23 on Llama (median; range 0.01–0.70) and 0 on Gemma (range 0–0.49).
- The random-token null with the same m has the *same* floor share.
- The condition numbers of S_η at η = 0.05 are *lower* on Llama (114–278) than on Gemma (142–586).

The fit ignores the floor (`mapback.json`, Llama, 4 concepts):
- The floor is 20–36% of all dimensions.
- At the default it carries at most 0.5% of the squared norm of the mapped-back concept feature, even after the η^{−γ/2} amplification of T⁻¹.
- At η = 0.5, γ = 0.5 it carries at most 1.9% (mean over the top-20 features).
- So the null space neither helps nor creates degenerate edit directions; it is simply dropped.

**F2. Default CS fails on Llama English, and the η/γ grid does not rescue it.**

On 4 concepts × 3 seeds (Table A1, `fig_grid_rank`):

| Setting | Share ρ ≥ 2 | Concept-feature ρ-rank (median) | Coherence (top-20) |
|---|---|---|---|
| CS, all 8 grid points | 0.87–0.93 | 12–41 | 0.013–0.023 |
| EMBER | 0.30 | 6.5 | 0.070 |
| γ = 0 (whitening only) | 0.24 | 4.0 | 0.059 |

Purity (0.77–0.86) and AUC (0.81–0.88) go *up* under CS. They measure page membership, not concept discovery (see F4).

On all 18 concepts, default CS vs EMBER:

| Metric | Default CS | EMBER | Test |
|---|---|---|---|
| Share ρ ≥ 2 | 0.92 | 0.32 | p = 2e-4 |
| Concept-feature ρ-rank (geometric mean) | 38 | 10.3 | Δlog₂ = +1.81, 95% CI [1.19, 2.56], p = 1e-4 |
| R3 (mean) | 39.8 | 18.4 | p = 0.012 |
| Coherence (median) | 0.016 | 0.072 | p = 8e-6 |

The three Llama concepts with m > d fail in the same way:

| Concept | Share ρ ≥ 2 | ρ-rank |
|---|---|---|
| Ancient Rome | 0.90 | 79 |
| Nazism | 0.90 | 36 |
| Pornography | 0.89 | 17 |

Across the 18 concepts, the share ρ ≥ 2 stays inside 0.86–0.95. Its Spearman correlation with m/d is −0.60 within that narrow band; for the rank it is −0.20.

**F3. Rank and k are not the cause (two causal controls).**

- **Gemma, page cut to 50% of its types** (m/d 0.53–0.91; 4 concepts × 3 seeds): share ρ ≥ 2 is 0.65 vs 0.63 at the default, and concept-feature ranks are 1–2 vs 1–3 at the default.
- **Gemma's own m < d pages** (Gambling, Cannabis, Golf): rank 1 at the default.
- **Fit size k:** Llama at k = 100 still inflates (0.88); Gemma at k = 200 does not (0.66).

**F4. The mechanism: a noise-dominated page moment plus each page token's self-echo (checked numerically).**

How weak the signal is:

| Quantity | Llama | Gemma |
|---|---|---|
| n_signal (eigenvalues above the null maximum) | 7–16, median 12 | 47–98, median 74 |
| λ₁ (median) | 7.8 | 15.4 |
| Null edge (median) | 4.4 | 3.7 |
| Ledoit–Wolf shrinkage η\* | 0.87–0.93 | 0.60–0.67 |
| Split-half top-10 principal-angle sine (median; null = 0.998) | 0.91 | 0.79 |

So Llama's M is about 90% noise in the Frobenius sense, and its top enriched directions barely replicate across halves of the page.

The self-echo, in formulas and numbers:
- For a page token t, u_tᵀ M u_t = ‖u_t‖⁴/m + (1/m) Σ_{j≠t} (u_jᵀ u_t)². The first term is the token's own contribution.
- If the tokens were isotropic, the own term's share of the γ = 1 energy would be ≈ d/(d + m).
- The energy boost of page tokens over off-page tokens would be ≈ (1 − η)(1 + d/m) + η.
- Across the 36 (model, concept) pairs, these predictions match the measured values with r = 0.99 for the self-share and r = 0.94 for the boost (0.98 and 0.998 within each model).
- Measured self-share: Llama median 0.54 (range 0.45–0.73); Gemma median 0.36.
- Measured boost: Llama median 2.37; Gemma median 2.13.

Why this boosts exactly the concept-only tokens:
- 98–99% of V\* concept-only tokens are on the page; only 2–13% of neutral-only tokens are. (Gambling is the outlier: 0.47 and 0.01.)
- So S_η multiplies the energy of almost exactly the concept-only tokens by about 2.4×.
- On Gemma, this rides on about 70 strong, coherent spikes. On Llama, it is most of what S_η does.

What it looks like in the features (Valentine's Day, seed 42):
- The six highest-ρ default-CS features are mixtures like *vatican, parchment, valencia, 情人節, reply, imprisonment, edmund*, with concept-only share 0.90–0.96 and ρ ≈ 3·10⁵.
- EMBER's top features are topical: religion; love and affection; festivity.
- On Gemma, CS does not lower coherence: 0.038 vs 0.039 for EMBER (n = 18, measured in Llama space).

**F5. Dose-response: sharpening only above the noise edge removes the failure (`fig_truncation`).**

The truncated transform keeps g_j = (1 − η)λ_j + η for the top r eigen-directions and sets g = 1 (the background level) everywhere else.

Llama, 4 concepts:

| r (sharpened directions) | Share ρ ≥ 2 | Concept-feature ρ-rank (median) | Coherence |
|---|---|---|---|
| Noise edge (r = n_signal = 11–14) | 0.35 | 8 (R3 1.5) | 0.065 |
| 50 | 0.73 | 3.5 | 0.032 |
| 100 | 0.88 | 22 | 0.021 |
| All d (default CS) | 0.93 | 40 | 0.012 |

The failure grows steadily as noise directions are added back.

On all 18 Llama concepts, the truncated rule (η = 0.05, γ = 1):
- **vs EMBER:** share ρ ≥ 2 0.37 vs 0.32; rank 5.0 vs 10.3 (geometric mean, p = 0.015); R3 6.0 vs 18.4 (p = 0.004); purity 0.69 vs 0.63 (p = 0.006); coherence 0.070 vs 0.072 (median; Δ −0.003, p = 0.02).
- **vs default CS:** rank 5.0 vs 38 (p < 1e-4); coherence 0.070 vs 0.016 (p = 8e-6).
- **vs whitening only (γ = 0):** rank 2.34 vs 2.46 on the log₂ scale (p = 0.65), so no rank gain; purity +0.07 (p = 4e-4).

On Gemma (18 concepts), the truncated rule keeps 47–98 directions and cannot be told apart from the default:
- Share ρ ≥ 2: 0.605 vs 0.617 (p = 0.45).
- Rank: p = 0.31.
- AUC: 0.706 vs 0.706.

**F6. Pooling languages helps only together with truncation.**

ml8 (8-language pooled target):
- Size: m ≈ 11,000–12,500 (m/d 2.7–3.1); n_signal 83–88; self-share 0.23–0.25.
- On its own: share ρ ≥ 2 0.82 at η = 0.05 (rank 15.5) and 0.76 at η = 0.5 (rank 9.5).
- With truncation at r = 85: share 0.38, rank 4.5, coherence 0.054.

Leak-free targets:
- They remove the inflation (share 0.18) but also the signal.
- Most page types are V\* types, so only m = 109–1,439 types remain and n_signal is 0–2.
- Result: rank 25, coherence 0.033.

**F7. A noisier background does not explain Llama's weak page signal.** I re-whitened Gemma with a random 34k-type background, which has Llama's d/n_B = 0.068. n_signal went *up*, from 56–75 to 115–123 (`spectra/bg_noise_check.json`).

**F8. Gemma side note.**
- At seeds 42–44, CS does not change Gemma's lexicon-identified concept-feature rank overall: geometric mean 1.7 vs 2.1 (p = 0.37). EMBER already ranks the concept feature first for most concepts.
- The large gains are concentrated in HP (12 → 1) and Ancient Rome (16 → 2).
- Whitening only (γ = 0) also gives rank 1 for HP on Gemma.

## Speculations (with tests)

**S1. Why the page contrast is weak on Llama.**
- Observation: Llama's page tokens sit *closer* to the background mean than the average background type (tr M / d = 0.79–0.86, vs 1.02–1.12 on Gemma).
- Speculation: rare tokens dominate Llama's background variance, and topical variance is spread thinly over 4096 dimensions.
- Test: measure n_signal on other d = 4096 models (Mistral-7B, Llama-2-7B) and with a frequency-weighted background.

**S2. Gemma's CS gains may partly be the same echo.**
- Evidence for: purity and AUC also rise on Gemma (+0.05 and +0.03), and Gemma's leak-free target loses lexicon hits (2.2 vs 6.0).
- Evidence against: the truncated rule removes the bulk echo on Gemma without changing any metric, which suggests Gemma's gains come from its spikes.
- Test: a cross-fitted target, where each fold of V\* is sharpened with an S built without that fold's tokens, on Gemma's 18 concepts.

**S3. Concept identification on Llama is noisier at k = 200.**
- Observation: the best feature's lexicon hits for Ancient Rome, Valentine's Day and Baseball are 2.7–3.7, against 5 or more on Gemma.
- Likely cause: k = 200 splits each concept across several features. EMBER at k = 100 gives 4.2 hits and rank 3.
- Consequence: the ranks for those three concepts are noisier than HP's.
- Test: blind hand labels of the top-20 features for one seed per configuration.

**S4. Erasure.**
- The truncated map keeps T⁻¹ close to W⁻¹ outside about 12 directions, so CINDER edits on Llama should behave like whitened-EMBER edits.
- Untested: the edit stage is outside WP8's scope.

## Recommendation

**Don't:**
- Do not choose η from m/d.
- Do not require m ≥ d.
- No η ∈ [0.05, 0.5] and no γ ∈ {0.5, 1} works on Llama, and the floor itself is harmless.

**Do: sharpen only the resolvable directions.**
- S = I + Σ_{j : λ_j > λ̂_edge} ((1 − η)λ_j + η − 1) p_j p_jᵀ.
- λ̂_edge is the largest eigenvalue of the same moment computed from m random background types.
- The Marchenko–Pastur edge (1 + √(d/m))² is a close closed-form stand-in: 4.5 vs the empirical 4.3 on Llama HP.
- This adds no hyperparameter, leaves Gemma's CS unchanged (F5), and removes the Llama failure.
- Report n_signal alongside each fit.

**What to expect when n_signal is small (about 20 or less, which covers every English page on Llama):**
- CS will add little over background whitening; the truncated CS ranks the same as γ = 0 (F5).
- For a real contrast, pool more text into the target. The 8-language pool reaches n_signal ≈ 85; apply truncation there too (F6).

## Paper paragraph

On Llama-3.1-8B, single-language contrastive sharpening fails, but not for the reason one might expect. Most English concept pages have fewer distinct tokens than the embedding dimension (15 of 18 pages; median m/d = 0.77), yet the fitted features ignore the resulting null space (under 1% of their norm); pages with m > d fail just the same, and Gemma is unaffected when its page is cut below d. The actual cause is estimation noise: only about 12 eigenvalues of Llama's whitened page moment exceed a same-size random-token null (about 74 on Gemma), and in that noise-dominated bulk each page token's own outer product supplies about half of its sharpened energy. Sharpening therefore amplifies page membership: 92% of features pass the ρ filter, their semantic coherence falls close to chance, and the concept feature drops from a ρ-rank of about 10 to about 38. Changing η or γ does not help. Restricting sharpening to the eigenvalues above the noise edge restores a ρ-rank of about 5 on Llama and leaves Gemma's results unchanged.

## Appendix text

**Definitions.**
- Whitened rows: u = (e − μ_B)W, with W = (Σ_B + εI)^{−1/2} and ε = 10⁻⁴·mean λ(Σ_B).
- Page moment: M = (1/m) Σ_{j∈page} u_j u_jᵀ, with eigenpairs (λ_j, p_j) in descending order.
- Sharpening matrix: S_η = (1 − η)M + ηI, with eigenvalues g_j = (1 − η)λ_j + η.
- CS matrix: X = U S_η^{γ/2}. Directions map back with T⁻¹ = S_η^{−γ/2} W⁻¹.
- Floor share: the fraction of j with g_j < 1.1η.
- Noise null: the same statistic computed for m random English Wikipedia background types not on the page (seed 1).
- n_signal: #{j : λ_j > λ₁(null)}.
- Ledoit–Wolf intensity toward I: η\* = min(1, β²/‖M − I‖²_F), with β² = (Σ_j ‖u_j‖⁴/m − ‖M‖²_F)/m.
- Self-share at γ = 1: (1 − η)‖u_t‖⁴/m ÷ [(1 − η)u_tᵀ M u_t + η‖u_t‖²].
- Boost: the mean of ‖x_t‖²/‖u_t‖² over V\* tokens on the page, divided by the same mean for V\* tokens off the page.
- Isotropic predictions: self-share ≈ d/(d + m); boost ≈ (1 − η)(1 + d/m) + η.
- Truncated rule (config name `cs_e<η>_g<γ>_mp`): keep g_j for j ≤ n_signal and set g_j = 1 otherwise. `topR` keeps the top R directions.

**Settings.**
- Sparse MF exactly as EMBER/L0: `VWConfig(method="fixed", sparsity=0.01, refit=False, ridge=1e-4, max_iter=20000, patience=200)`, randn initialisation, seeds 42–44.
- k = 200 for Llama and 100 for Gemma, unless a `_k` suffix says otherwise.
- Lexicon: concept_words_v1. A feature's words are its first 20 distinct normalized source words in |y| order (`overlap_judge.feature_words`, with counts over all 600 sentences).
- Llama background: 60,467 types from 300,000 sentences. 300,000 was the cap of the original stopping rule; Gemma's type count matches exactly.
- The Llama background does not meet the original stability criterion at that cap: the split-half top-100 subspace sine is 0.43, against a 0.1 threshold. F7 shows background noise is not the bottleneck.

**Table A1.** Llama, 4 concepts × seeds 42–44; each value is the median over concepts of the per-concept seed summaries (`tables/llama_4concepts_by_config.md`):

| config | share ρ≥2 | cf ρ-rank | R3 | cf lexicon hits | cf purity | AUC | coherence top-20 | rel. err |
|---|---|---|---|---|---|---|---|---|
| `raw` | 0.30 | 6.5 | 7.0 | 3.2 | 0.66 | 0.67 | 0.070 | 0.922 |
| `white` | 0.24 | 4.0 | 4.2 | 3.8 | 0.60 | 0.62 | 0.059 | 0.954 |
| `cs_e0.05_g1` | 0.93 | 39.5 | 52.2 | 3.2 | 0.86 | 0.88 | 0.012 | 0.909 |
| `cs_e0.05_g0.5` | 0.90 | 32.0 | 39.8 | 2.7 | 0.80 | 0.85 | 0.017 | 0.931 |
| `cs_e0.1_g1` | 0.93 | 34.0 | 51.5 | 3.0 | 0.85 | 0.88 | 0.013 | 0.911 |
| `cs_e0.1_g0.5` | 0.90 | 21.0 | 25.0 | 3.0 | 0.80 | 0.85 | 0.016 | 0.933 |
| `cs_e0.2_g1` | 0.92 | 41.0 | 56.0 | 3.2 | 0.86 | 0.87 | 0.013 | 0.914 |
| `cs_e0.2_g0.5` | 0.90 | 12.0 | 14.8 | 2.7 | 0.83 | 0.84 | 0.018 | 0.935 |
| `cs_e0.5_g1` | 0.91 | 25.5 | 11.8 | 3.3 | 0.85 | 0.86 | 0.016 | 0.927 |
| `cs_e0.5_g0.5` | 0.87 | 20.5 | 23.2 | 2.8 | 0.77 | 0.81 | 0.023 | 0.942 |
| `raw_k100` | 0.32 | 3.0 | 3.0 | 4.2 | 0.70 | 0.63 | 0.075 | 0.950 |
| `cs_e0.05_g1_k100` | 0.88 | 19.0 | 11.5 | 3.8 | 0.86 | 0.80 | 0.025 | 0.944 |
| `cs_e0.05_g1_lf` | 0.18 | 25.0 | 28.2 | 2.3 | 0.55 | 0.59 | 0.033 | 0.916 |
| `cs_e0.5_g1_lf` | 0.20 | 41.5 | 41.0 | 3.3 | 0.52 | 0.61 | – | 0.939 |
| `cs_e0.05_g1_ml8` | 0.82 | 15.5 | 14.2 | 2.8 | 0.72 | 0.80 | 0.026 | 0.940 |
| `cs_e0.5_g1_ml8` | 0.76 | 9.5 | 9.5 | 3.5 | 0.70 | 0.75 | 0.037 | 0.947 |
| `cs_e0.05_g1_mp` | 0.35 | 8.0 | 1.5 | 4.7 | 0.65 | 0.67 | 0.065 | 0.936 |
| `cs_e0.05_g0.5_mp` | 0.30 | 3.5 | 3.0 | 4.2 | 0.67 | 0.65 | 0.064 | 0.948 |
| `cs_e0.05_g1_top50` | 0.73 | 3.5 | 3.5 | 4.2 | 0.78 | 0.76 | 0.032 | 0.923 |
| `cs_e0.05_g1_top100` | 0.88 | 22.0 | 25.8 | 3.3 | 0.88 | 0.83 | 0.021 | 0.916 |
| `cs_e0.05_g1_ml8_top85` | 0.38 | 4.5 | 4.5 | 3.7 | 0.62 | 0.66 | 0.054 | 0.949 |

**Table A2.** Spectra, 18 concepts × 2 models (`tables/spectra_18.csv`). Each cell is min / median / max:

| Quantity | Llama | Gemma |
|---|---|---|
| m/d | 0.30 / 0.77 / 1.12 | 0.51 / 1.29 / 1.91 |
| Floor share (η = 0.05) | 0.01 / 0.23 / 0.70 | 0 / 0 / 0.49 |
| cond(S_η), η = 0.05 | 114 / 149 / 278 | 142 / 258 / 586 |
| n_signal | 7 / 12 / 16 | 47 / 74 / 98 |
| λ₁ | 6.1 / 7.8 / 14.6 | 11.4 / 15.4 / 30.8 |
| Null λ₁ | 3.6 / 4.4 / 7.9 | 3.0 / 3.7 / 6.1 |
| Ledoit–Wolf η\* | 0.87 / 0.90 / 0.93 | 0.60 / 0.65 / 0.67 |
| Self-share | 0.45 / 0.54 / 0.73 | 0.28 / 0.36 / 0.55 |
| Boost, γ = 1 | 1.96 / 2.37 / 4.22 | 1.82 / 2.13 / 3.54 |
| Boost, γ = 0.5 | 1.52 / 1.76 / 2.89 | 1.41 / 1.57 / 2.39 |
| Split-half top-10 sine | 0.87 / 0.91 / 0.96 | 0.73 / 0.79 / 0.81 |
| tr M / d | 0.79 / 0.83 / 0.86 | 1.02 / 1.05 / 1.12 |

Per-concept Llama values:

| Concept | m/d | Floor share (η = 0.05) | n_signal | Self-share |
|---|---|---|---|---|
| Gambling | 0.30 | 0.70 | 7 | 0.73 |
| Cannabis | 0.52 | 0.48 | 10 | 0.63 |
| Golf | 0.55 | 0.45 | 12 | 0.61 |
| Uranium | 0.63 | 0.37 | 11 | 0.59 |
| Baseball | 0.64 | 0.36 | 12 | 0.58 |
| Halloween | 0.64 | 0.36 | 12 | 0.59 |
| Valentine's Day | 0.71 | 0.29 | 13 | 0.57 |
| Gun | 0.74 | 0.26 | 12 | 0.53 |
| Heroin | 0.74 | 0.26 | 12 | 0.56 |
| HP | 0.80 | 0.20 | 11 | 0.54 |
| Culture of Greece | 0.80 | 0.20 | 11 | 0.54 |
| World War II | 0.81 | 0.19 | 16 | 0.53 |
| COVID-19 pandemic | 0.90 | 0.11 | 13 | 0.50 |
| Republic of Ireland | 0.95 | 0.08 | 12 | 0.49 |
| Artificial intelligence | 0.96 | 0.08 | 12 | 0.48 |
| Ancient Rome | 1.05 | 0.03 | 14 | 0.47 |
| Nazism | 1.11 | 0.01 | 16 | 0.45 |
| Pornography | 1.12 | 0.01 | 13 | 0.45 |

**Table A3.** Per concept and configuration, both models: `tables/per_concept_18.csv`. Paired tests: `tables/stats18.json`.

Llama, 18 concepts, concept-feature ρ-rank (seed median):

| Concept | Raw | Default CS | Truncated | γ = 0 |
|---|---|---|---|---|
| Gambling | 2 | 47 | 1 | 2 |
| Cannabis | 7 | 34 | 21 | 20 |
| Golf | 81 | 199 | 13 | 27 |
| Uranium | 15 | 90 | 6 | 42 |
| Baseball | 7 | 23 | 16 | 6 |
| Halloween | 8 | 25 | 10 | 14 |
| Valentine's Day | 16 | 55 | 13 | 5 |
| Gun | 1 | 14 | 1 | 2 |
| Heroin | 61 | 25 | 7 | 1 |
| Culture of Greece | 7 | 70 | 7 | 3 |
| HP | 5 | 24 | 3 | 3 |
| World War II | 22 | 13 | 4 | 2 |
| COVID-19 pandemic | 4 | 28 | 3 | 4 |
| Republic of Ireland | 43 | 125 | 19 | 52 |
| Artificial intelligence | 7 | 25 | 4 | 2 |
| Ancient Rome | 6 | 79 | 1 | 3 |
| Nazism | 14 | 36 | 5 | 3 |
| Pornography | 22 | 17 | 2 | 8 |

Llama share ρ ≥ 2: default CS 0.86–0.95 on every concept; truncated 0.26–0.50; raw 0.24–0.40.

## Figures

**`figures/fig_spectrum_hp.pdf`** (single column).
- Shows the eigenvalues g_j of S_η (η = 0.05, page target) for HP, Gemma-2-2B (green, m/d = 1.36) vs Llama-3.1-8B (purple, m/d = 0.80).
- Solid lines: the page. Dashed lines: m random background types (the noise null).
- Dots mark the last eigenvalue above the null maximum: 56 for Gemma, 11 for Llama.
- Llama's spectrum drops to the floor η at j = m, but so does its null, so the floor is a sample-size artifact. What separates the two models is the gap between page and null in the top directions.
- Data: `fig_spectrum_hp.json`.

**`figures/fig_grid_rank.pdf`** (two columns).
- Llama, 4 concepts × seeds 42–44.
- Top row: ρ-rank of the lexicon-identified concept feature. Bottom row: share of the 200 features with ρ ≥ 2.
- Blue: CS over η ∈ {0.05, 0.1, 0.2, 0.5}, γ = 1 (solid) and γ = 0.5 (dashed); median over seeds, band = min–max.
- Gray: EMBER on raw E. Green dashed: CS truncated at the noise edge. Black star: Gemma default CS (existing fits).
- Dotted line: the judge window (top 20).
- Data: `fig_grid_rank.csv`.

**`figures/fig_truncation.pdf`** (two columns).
- Llama CS (η = 0.05, γ = 1) sharpening only the top r eigen-directions, with the rest at the background level.
- r = noise edge (11–14), 50, 100, or all 4096.
- Panels: share of features with ρ ≥ 2; concept-feature ρ-rank; coherence of the top-20 features in Gemma's embedding space.
- Thin lines: individual concepts. Thick line: median. Gray: EMBER, per concept.
- Data: `fig_truncation.csv`.

## Sources

Code, all in `experiments/paper_analysis/llama_cs/`:
- `common.py`: embeddings, targets, V\*, background, CS helpers.
- `background.py`: the Llama English background. Writes `data/bg_wiki_llama_stats.pt`, `bg_wiki_llama_meta.json`, `bg_wiki_{llama,gemma}_types.json` and `wiki_sentences.jsonl`.
- `spectra.py`: writes `spectra/spectra.json` and `spectra_<model>_<slug>.pt`.
- `spectra_targets.py`: writes `spectra/targets.json` (page / lf / ml8).
- `bg_noise_check.py`: F7.
- `fits.py`: all fits, in `fits/<model>/<slug>/<config>/seed<N>/{fit.pt, metrics.json}`, plus `transform_diag.json`.
- `metrics.py`: the judge-free metrics.
- `gemma_ref.py`: metrics on the existing Gemma fits, written to `gemma_ref.json`.
- `coherence.py`: writes `coherence.json`; the random baseline is in `coherence_random_baseline.json`.
- `mapback.py`: writes `mapback.json`.
- `report.py`: tables and figures.

Inputs:
- `results/varwidth/<Slug>/{fixed_refit-False_plain,fixed_refit-False}/seed4[2-6]/fit.pt`
- `results/varwidth/prep/*_plain_S300.pt`
- `experiments/whitening/data/gemma/bg_wiki_stats.pt`
- `experiments/whitening/data/pages/*.json`
- `../nlp_proj_ml/results/ml/data/{raw/Ancient_Rome/en.json, full/*.json}`
- `experiments/varwidth/j3_data/concept_words_v1.json`

Gemini spend: 0. Compute: one local c-001 GPU (TITAN Xp), now released.
