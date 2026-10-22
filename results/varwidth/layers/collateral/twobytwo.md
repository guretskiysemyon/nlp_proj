# 2×2 at fixed strength: direction vs token set (PREREG amendment 14)

ee = EMBER tokens × EMBER direction; 1b = EMBER tokens × CINDER direction; 1a = CINDER top-N tokens × CINDER direction; cross = CINDER top-N tokens × EMBER direction. Test split, fixed r, no selection. Δ_dir = mean of (1b − ee) and (1a − cross); Δ_tok = mean of (cross − ee) and (1a − 1b). Unit = concept (mean over seeds), Wilcoxon, 95% cluster bootstrap; Holm over Δ_dir and Δ_tok × efficacy and H3v at r = 0.5.

## r = 0.5 (primary)

| Cell | runs | efficacy | collateral | specificity | H3v | achieved r |
|---|---|---|---|---|---|---|
| ee | 41 | 0.201 | 0.934 | 0.987 | 0.389 | 0.50 |
| 1b | 38 | 0.256 | 0.909 | 0.985 | 0.456 | 0.50 |
| 1a | 38 | 0.263 | 0.918 | 0.987 | 0.475 | 0.50 |
| cross | 38 | 0.204 | 0.940 | 0.987 | 0.389 | 0.50 |

| Effect | metric | concepts (runs) | mean | 95% CI | +/− concepts | p | Holm p |
|---|---|---|---|---|---|---|---|
| Δ_dir | efficacy | 11 (38) | +0.073 | [+0.012, +0.150] | 10/1 | 0.032 | 0.097 |
| Δ_tok | efficacy | 11 (38) | +0.016 | [-0.001, +0.039] | 8/2 | 0.160 | 0.203 |
| Δ_dir | H3v_rn | 11 (38) | +0.097 | [+0.029, +0.174] | 10/1 | 0.014 | 0.055 |
| Δ_tok | H3v_rn | 11 (38) | +0.024 | [+0.002, +0.049] | 8/3 | 0.102 | 0.203 |
| Δ_dir | collateral_rn | 11 (38) | -0.019 | [-0.035, +0.001] | 1/10 | 0.054 | – |
| Δ_tok | collateral_rn | 11 (38) | +0.004 | [-0.007, +0.018] | 6/5 | 0.831 | – |

| Contrast | metric | concepts (runs) | mean | 95% CI | p |
|---|---|---|---|---|---|
| 1b−ee | efficacy | 11 (38) | +0.082 | [+0.031, +0.154] | 0.004 |
| 1a−cross | efficacy | 11 (38) | +0.065 | [-0.014, +0.149] | 0.042 |
| cross−ee | efficacy | 11 (38) | +0.025 | [-0.002, +0.068] | 0.156 |
| 1a−1b | efficacy | 11 (38) | +0.007 | [-0.007, +0.022] | 0.426 |
| 1b−ee | H3v_rn | 11 (38) | +0.102 | [+0.042, +0.174] | 0.010 |
| 1a−cross | H3v_rn | 11 (38) | +0.093 | [+0.008, +0.177] | 0.042 |
| cross−ee | H3v_rn | 11 (38) | +0.028 | [-0.004, +0.071] | 0.232 |
| 1a−1b | H3v_rn | 11 (38) | +0.020 | [-0.005, +0.047] | 0.413 |
| 1b−ee | collateral_rn | 11 (38) | -0.020 | [-0.036, +0.002] | 0.054 |
| 1a−cross | collateral_rn | 11 (38) | -0.019 | [-0.037, +0.002] | 0.123 |
| cross−ee | collateral_rn | 11 (38) | +0.003 | [-0.006, +0.013] | 0.557 |
| 1a−1b | collateral_rn | 11 (38) | +0.004 | [-0.011, +0.025] | 0.577 |

