# WP6b mean_edit: pre-registration

Written 2026-10-22, before any MEAN-C / EMBER-c / MEAN-E / delta-matched grid point, copy test, test pass or
integrity test of this work package was run, and before S2 / S4 were computed. Only the existing development results
(D9, D14, D16 of `CINDER_collateral_evaluation.md`) and the CPU check of the new edit sets below were seen.

## Question

WP6 (`results/paper_analysis/edit_direction/REPORT.md`) showed CINDER-z = CINDER + beta_t mu_B exactly, and EMBER =
EMBER-c + beta_t^E mu_B (EMBER's raw fit is uncentred; 44% of its edit energy lies on mu_B-hat). Layer 0 reads
mu_B-hat at 2.9x a random direction. Speculation S1: *much of the "stronger" uncentred edit is generic, not about the
concept: subtracting beta mu_B makes a token atypical.* Does the mean term help erase the concept, or does it only
damage the edited word (token destruction) and its context?

## Runs

The 8 development collateral runs (D8; the runs of the E2 integrity tests and the D16 offline check): Harry Potter
42, 43; Ancient Rome 42, 43, 44; Valentine's Day 42, 43, 46. A variant exists where its parent edits:
CINDER family on 7 runs (no CINDER edit at Ancient Rome 43), EMBER family on 6 (no EMBER edit at Ancient Rome 42 and
Valentine's Day 43). Both families: Harry Potter 42, 43, Ancient Rome 44, Valentine's Day 42, 46.

## Edit variants (rows c_t subtracted as e' = e - delta c; r = median_t |delta c_t| / |e_t|, each variant's own lengths)

| Name | Tokens | Row c_t | Status |
|---|---|---|---|
| CINDER | CINDER's | mapped-back CS+GS dictionary (centred) | existing grids/tests (reused) |
| CINDER-z | CINDER's | CINDER's Y, F_C; Z re-solved on raw A (uncentred) | existing (reused) |
| MEAN-C | CINDER's | beta_t^z mu_B, beta^z = sum_{i in F_C} b_i Y_it, b = (YY^T + lam I)^-1 Y 1 (WP6 Cor. 2) | **new** |
| EMBER | EMBER's | plain fit, own Z (uncentred) | existing (reused) |
| EMBER-c | EMBER's | EMBER's Y, F_C; Z re-solved on A - mu_B 1^T (centred) | **new** |
| MEAN-E | EMBER's | beta_t^E mu_B (b from EMBER's Y) | **new** |
| MEAN-Cd, MEAN-Ed | as parent | same rows as MEAN-C / MEAN-E, but grid point r uses the delta of CINDER-z / EMBER at r, so the row is exactly the mean part of the uncentred edit at that grid point | **new**, secondary |
| Noise-C, Noise-E | CINDER's / EMBER's | the existing Noise baseline e + delta |c_t| u_t | existing grids; copy test new |

CPU check (`experiments/paper_analysis/mean_edit/sets.py`, before any GPU run): in all 7 CINDER runs
||(c^z - c) - beta^z mu_B|| / ||c^z - c|| = 1e-4 and the difference has energy share 1.000 on mu_B-hat; in all 6 EMBER
runs ||c^E - c^E_centred - beta^E mu_B|| / ||c^E|| = 1e-4. Median beta^z 0.31-0.66, beta^E 0.31-0.87. At a grid
point r of CINDER-z, its mean part alone has r_m = 0.52-0.73 r; for EMBER 0.45-0.58 r. Tokens' own mu coordinate
e.mu/|mu|^2 is 0.83-1.84 (|mu_B| = 34.5, |e| ~ 80-105), so MEAN-C at r ~ 0.5 roughly removes a token's whole mean
component and r ~ 1 flips it.

Grids: r in {0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 100} for MEAN-C, EMBER-c, MEAN-E; {0.25, 0.5, 1, 2, 5, 10, 30,
100} for the delta-matched MEAN-Cd / MEAN-Ed.

## Measures (frozen question sets and existing code paths; no new items; no Gemini)

- **Validation grid** (`erase_v2.grid`, monkey-patched edit sets and output directory): efficacy (qa_train 50),
  specificity HM(MMLU-frac, simdom-frac), collateral on the variant's own edited multi-meaning tokens, H3v.
- **Copy at every grid point with r <= 2.5** (`last_mile.copygrid` protocol: T1, 2 carriers, <= 60 words, seeded
  sample) for CINDER, CINDER-z, MEAN-C, Noise-C, EMBER, EMBER-c, MEAN-E, Noise-E.
- **Selection**: collateral rule (MMLU-frac >= 0.9, collateral >= 0.8, max H3v, ties to smaller r) and the
  integrity-gated rule E3 (the same restricted to copy >= 0.9). Test pass (qa_test, simdom_test, MMLU test 1000,
  collateral test renormalised as amendment 6) at each selected point; an existing test pass at the same point is
  reused (deterministic evaluation).
- **Judge-free integrity** (E2 `integrity.measure` and `static`, refs orig / nonce / del from the existing E2 files):
  copy, ctx_pass, del_likeness, context-MC, answer-NLL excess over the nonce, echo, z > 3, nn-cos; at the collateral-
  rule point, the E3 point and r in {0.25, 1, 10, 100} for MEAN-C, EMBER-c, MEAN-E; at the E3 points of CINDER,
  CINDER-z, EMBER (their collateral-rule and sweep points exist). Word lists: the parent's existing E2 list at r = 1
  (`sub_cinder_z@1.0` for MEAN-C, `sub_ember@1.0` for EMBER-c and MEAN-E), so new and old configs share words;
  contrasts with existing configs that sampled different words (> 60 words) use the common words.
- **Floors**: nonce / deletion floors of efficacy for CINDER's and EMBER's token sets (`last_mile.unreadable`, the
  unedited model on qa with the set's tokens replaced or deleted), on qa_test (as amendment 9) and qa_train (the
  grid's split). New computation (floors exist only for the expansion concepts), same code.
- Not done: T3 free-text labels (no labels), neutral generation (the pipeline scores it only with judges), Alpaca.

## Contrasts

C1 (as specified). At fixed r (validation): G_z(r) = eff(CINDER-z) - eff(CINDER) vs M(r) = eff(MEAN-C) - 0
(no edit has efficacy 0 by construction). Same for the EMBER family: G_E(r) = eff(EMBER) - eff(EMBER-c) vs
eff(MEAN-E). Also at the selected points (test).
C2 (exact decomposition, secondary). At CINDER-z's grid point r (delta_z), CINDER-z = CINDER(delta_z) +
MEAN-Cd(r). Compare eff(CINDER-z, r) with eff(MEAN-Cd, r) and eff(CINDER at delta_z), the last interpolated
log-linearly in r from CINDER's own grid at r * median mu_c / median mu_z. Same for EMBER.
C3 (generic baseline). MEAN-C vs Noise-C, MEAN-E vs Noise-E at fixed r.
C4 (efficacy per unit of damage). Efficacy vs copy along r <= 2 (efficacy interpolated at copy = 0.5 and 0.9);
efficacy vs context damage (context-MC, answer-NLL excess) at r in {0.25, 1, 10, 100}; efficacy relative to the
deletion floor.

Unit: run (concept-seed); n = 7 (CINDER family) or 6 (EMBER family), 3 concepts. Paired means, wins/losses,
Wilcoxon signed-rank over runs and bootstrap 95% CI over runs (10,000). Runs of one concept are correlated; with 3
concepts the tests are descriptive. Per-run values are reported.

## Decision rule for S1 (primary; validation; CINDER family; R1 = {1, 2, 5, 10, 30, 100})

Gz = mean over runs and R1 of G_z(r); Em = mean over runs and R1 of eff(MEAN-C, r).

- **Supported** if Em >= 0.5 Gz AND MEAN-C damages at least as much as CINDER-z: copy(MEAN-C, r = 1) <=
  copy(CINDER-z, r = 1) + 0.10 AND mean context-MC over r in {10, 100} of MEAN-C <= that of CINDER-z + 0.05.
- **Refuted** if Em < 0.5 Gz AND copy(MEAN-C, r = 1) <= copy(CINDER, r = 1) (the mean edit damages words at least as
  much as CINDER at equal size without reproducing the gain).
- Otherwise **mixed**; we report which part holds.

Caveat stated in advance: the efficacy half of this rule is weak, because any edit that destroys the tokens reaches
the deletion floor (D18: 0.23-0.28 on the expansion), and the floor alone may exceed 0.5 Gz. The verdict is therefore
read together with C2-C4 below; if Tier 1 says "supported" only because MEAN-C reaches the floor while C2 shows
CINDER-z > CINDER(delta) + (MEAN-Cd - 0) (synergy), we will say so.

## Predictions (scored in the report; my expectations, not tests of significance)

| # | Prediction | Expected |
|---|---|---|
| P1 | Tier 1 verdict: S1 supported (Em >= 0.5 Gz, damage condition met) | yes (p ~ 0.65); Gz ~ +0.15, Em ~ 0.35 |
| P2 | C2: mean over runs and R1 of eff(MEAN-Cd) >= 0.5 x [eff(CINDER-z) - eff(CINDER at delta_z)] | yes (~0.65) |
| P3 | Damage: copy(MEAN-C, r = 1) <= copy(CINDER, r = 1); context-MC(MEAN-C) <= 0.40 and answer-NLL excess >= +0.8 nats at r in {10, 100} (CINDER: 0.53, +0.1-0.2) | yes (~0.6) |
| P4 | C3: eff(MEAN-C) - eff(Noise-C) >= +0.10 averaged over r in {10, 30, 100} (the mean direction derails more than a random one) | yes (~0.55) |
| P5 | EMBER family: G_E averaged over R1 >= +0.08 with >= 5/6 runs positive; Em_E >= 0.5 G_E | yes (~0.65) |
| P6 | EMBER-c at r in {10, 100}: context-MC within 0.05 of the nonce reference (no derailment), EMBER: >= 0.08 below | yes (~0.6) |
| P7 | Subtraction regime: eff(EMBER-c) - eff(EMBER) >= +0.03 averaged over r in {0.25, 0.5} (centring helps at small r, as CINDER > CINDER-z there) | yes (~0.55) |
| P8 | At copy = 0.5 (interpolated, r <= 2): eff(CINDER) >= eff(CINDER-z) >= eff(MEAN-C) (mean over runs) | yes (~0.5) |
| P9 | E3 rule (copy >= 0.9): MEAN-C test efficacy <= 0.05; at the collateral-rule point test H3v(MEAN-C) < H3v(CINDER-z) | yes (~0.7) |
| P10 | Collateral rule: H3v(EMBER-c) within +-0.05 of H3v(EMBER) (test, 6 runs) | yes (~0.5) |

## CPU follow-ups

**S2 (read gain and the layer-0 norm weights).** Recompute the WP6 read gain (|[W_q;W_k;W_v](g * u)| / random) of
mu_B-hat and of the EMBER / EMBER-c / CINDER / CINDER-z / MEAN edit rows on all WP6 runs, with (a) the actual g,
(b) g = 1, (c) g with its 8 largest coordinates set to 0 (random baseline recomputed under each g).
Prediction: under (b) and (c) mu_B-hat's read gain falls to <= 1.5x and the EMBER - CINDER read-gain gap shrinks
by >= 60% (p ~ 0.6). Refuted if mu_B-hat stays >= 2x under both.

**S4 (forward-pass sensitivity).** gemma-2-2b-it in float32. 50 tokens: 5 runs with both families x 5 tokens
sampled from CINDER's set and 5 from EMBER's set (seed 0). Each token placed in the first E2 copy carrier at the token-id
level. For each token and each direction of its family (CINDER, CINDER-z, MEAN-C, random | EMBER, EMBER-c, MEAN-E,
random; 4 random unit directions per token, seed 0), perturb the input row by s * |e_t| * u at s in {0.01, 0.25,
1}: (i) layer-0 write gain = |delta(h1 - h0)| / |delta h0| at the token position, relative to the random-direction mean
of that token; (ii) |delta h| at the final layer at the token position and at the last position; (iii) KL of the
next-token distribution at the token position and mean KL over the following positions. Compare with the linear proxy
(WP6 read gain of the same unit direction): Spearman over (token, direction).
Prediction: at s = 0.01 the uncentred directions (EMBER, CINDER-z, MEAN) have larger layer-0 write gain than the
centred ones (CINDER, EMBER-c) in >= 70% of tokens, and Spearman(proxy, layer-0 write gain) >= 0.5 (p ~ 0.6); at
s = 1 the final-layer KL ranks MEAN >= uncentred > centred (p ~ 0.5).

## Compute plan

GPU (job names `cl_pa_mean*`, <= 2 SLURM GPU jobs, `afterany` only; plus local c-001 GPU 0 if idle). One job per run,
in priority order: (1) own-r grids of MEAN-C / EMBER-c / MEAN-E, (2) copy at r <= 2.5, (3) test passes, (4) integrity,
(5) delta-matched grids. A pilot first times one grid point and re-evaluates one existing grid point (CINDER-z, r = 1,
Harry Potter 42) to check that the reused grids reproduce. Budget <= 25 GPU-hours; if the estimate exceeds it, the
CINDER family (CINDER, CINDER-z, MEAN-C) runs first and the rest is reported as pending.

## Amendments

1. (22:15, after the CINDER-family grids, copy tests and integrity tests were complete, before any analysis of them
   was run or read.) On the coordinator's instruction to save usage, the delta-matched grids MEAN-Cd / MEAN-Ed (C2,
   P2) are dropped. C2 is instead approximated from the own-r grids: the mean part of CINDER-z at grid point r has
   r_m = (median |beta mu| / |e|) / (median |c^z| / |e|) x r (0.52-0.73 r, table above), so eff(MEAN-C at r_m) is
   interpolated log-linearly from MEAN-C's own grid, as eff(CINDER at delta_z) from CINDER's. P2 is scored on that
   interpolation and flagged as approximate. The queue order became: CINDER-family test passes and floors before the
   remaining EMBER-family work.
2. The pilot (Harry Potter 42) reproduced the frozen CINDER-z grid point at r = 1 exactly (efficacy 0.491,
   collateral 0.792, all accuracies equal), so the existing grids are reused as planned; one grid point takes 93 s on a
   TITAN Xp.
