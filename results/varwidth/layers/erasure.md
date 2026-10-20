# Erasure anchor (EMBER, gemma-2-2b-it, MC protocol, seed 42)

Grid on the validation split gated by MMLU-frac ≥ 0.9, δ = argmax HM(efficacy, specificity); test split at δ: Con = concept MC accuracy, Sim = similar-domain MC accuracy, MM = MMLU (1000), coherence from 50 AlpacaEval prompts (Gemini judge). efficacy = 1 − qa_frac, specificity = HM(mmlu_frac, simdom_frac), H = HM(efficacy, specificity, coherence). Alpaca is not in the grid (Gemini budget): a deviation from the paper applied identically to every rung.

Unedited baseline over the same concepts: Con 73.4, Sim 89.3, MM 55.3.

| rung | concepts | Con | Sim | MM | efficacy | specificity | coherence | H | H2 | delta | n_edit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMBER | 14 | 54.29 | 86.43 | 55.16 | 0.36 | 0.97 | 0.98 | 0.52 | 0.47 | 104.29 | 30.50 |
| +CS | 14 | 62.43 | 88.57 | 55.27 | 0.21 | 0.99 | 0.99 | 0.38 | 0.31 | 23.79 | 39.21 |
| +CS+GS | 14 | 62.14 | 88.00 | 55.29 | 0.22 | 0.99 | 0.99 | 0.39 | 0.33 | 16.50 | 40.93 |
| +CS+GS+TWJ (CINDER) | 14 | 59.71 | 87.57 | 55.26 | 0.28 | 0.98 | 0.98 | 0.49 | 0.41 | 34.36 | 50.79 |
| EMBER+TWJ (expl.) | 14 | 50.14 | 86.14 | 55.11 | 0.47 | 0.97 | 0.98 | 0.65 | 0.59 | 132.86 | 37.00 |
| EMBER+TJ∨WJ+share (expl.) | 14 | 50.00 | 85.29 | 55.04 | 0.47 | 0.96 | 0.97 | 0.65 | 0.58 | 112.14 | 44.64 |
| +CS+GS+TJ∨WJ+share (expl.) | 14 | 61.43 | 87.86 | 55.29 | 0.23 | 0.99 | 0.99 | 0.40 | 0.34 | 16.50 | 43.71 |

Paired H differences over concepts:

- EMBER → +CS: Δ H = -0.146, 95% CI [-0.323, +0.043], 3 up / 9 down of 14
- +CS → +CS+GS: Δ H = +0.018, 95% CI [-0.092, +0.139], 4 up / 7 down of 14
- +CS+GS → +CS+GS+TWJ (CINDER): Δ H = +0.097, 95% CI [+0.000, +0.248], 2 up / 0 down of 14
- EMBER → +CS+GS+TWJ (CINDER): Δ H = -0.030, 95% CI [-0.236, +0.192], 5 up / 8 down of 14

## Proxy validity (Spearman over 98 concept × rung points)

| proxy | efficacy | specificity | H |
|---|---|---|---|
| clean | 0.480 | -0.285 | 0.490 |
| coverage | 0.622 | -0.504 | 0.619 |
| collateral_eval | 0.447 | -0.375 | 0.426 |
| collateral_wiki | 0.410 | -0.288 | 0.397 |

## Paper-grid check (Harry Potter, L0): Alpaca in the grid vs our MMLU-only gate

Paper grid: δ = 200, Con 58.0, Sim 88.0, MM 55.2, H 0.727. Our gate: δ = 200, Con 58.0, Sim 88.0, MM 55.2, H 0.726.

## Fidelity of L0 (our EMBER reimplementation vs the paper)

Paper Table 6 (EMBER, Gemma, 5 concepts × 5 seeds): Con 52.1, Sim 84.6, MM 53.6. Ours (L0, seed 42, 5 of the same 5 concepts): Con 48.4, Sim 83.2, MM 55.1.

| concept | Con | Sim | MM | delta | n_edit |
|---|---|---|---|---|---|
| Baseball | 48.0 | 66.0 | 55.2 | 200.0 | 40 |
| COVID-19 pandemic | 48.0 | 84.0 | 55.2 | 200.0 | 44 |
| Culture of Greece | 42.0 | 82.0 | 54.8 | 200.0 | 52 |
| Gambling | 46.0 | 96.0 | 55.1 | 200.0 | 41 |
| Harry Potter | 58.0 | 88.0 | 55.2 | 200.0 | 31 |
