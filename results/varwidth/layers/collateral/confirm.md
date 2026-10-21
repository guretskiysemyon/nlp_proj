# Confirmatory expansion (PREREG amendment 8)

12 concepts × seeds 42–46. Test split. Collateral renormalized by the unedited model's per-token test accuracy. Paired contrasts on runs where both methods edit; unit = concept (mean over seeds); two-sided Wilcoxon across concepts; Holm over the 4 primary contrasts; 95% cluster-bootstrap CI.

## Primary: collateral rule, H3v_rn

| Variant − EMBER | concepts (pairs) | mean Δ | 95% CI | wins/losses | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|
| CINDER | 11 (38) | +0.042 | [-0.009, +0.096] | 6/5 | 0.206 | 0.412 |
| CINDER-RS (CS + RS-1% + TWJ) | 11 (37) | +0.034 | [-0.043, +0.112] | 8/3 | 0.413 | 0.413 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 11 (38) | +0.132 | [+0.048, +0.212] | 9/2 | 0.019 | 0.074 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 11 (37) | +0.084 | [+0.020, +0.153] | 9/2 | 0.024 | 0.074 |

| Variant | mean H3v_rn (runs where it edits) | edit set exists | edit rate | failure rate (H3v < 0.25, no edit = fail) | failure rate when it edits |
|---|---|---|---|---|---|
| EMBER | 0.50 | 0.68 | 0.68 | 0.37 | 0.07 |
| CINDER | 0.53 | 0.87 | 0.87 | 0.17 | 0.04 |
| CINDER-RS (CS + RS-1% + TWJ) | 0.51 | 0.92 | 0.92 | 0.17 | 0.09 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 0.58 | 0.87 | 0.87 | 0.18 | 0.06 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 0.56 | 0.92 | 0.92 | 0.15 | 0.07 |

## Secondary: fixed r = 0.5, H3v_rn

| Variant − EMBER | concepts (pairs) | mean Δ | 95% CI | wins/losses | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|
| CINDER | 11 (38) | +0.151 | [+0.080, +0.226] | 10/1 | 0.003 | 0.012 |
| CINDER-RS (CS + RS-1% + TWJ) | 11 (37) | +0.141 | [+0.053, +0.238] | 10/1 | 0.019 | 0.037 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 11 (38) | +0.086 | [+0.008, +0.148] | 10/1 | 0.054 | 0.054 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 11 (37) | +0.115 | [+0.065, +0.165] | 10/1 | 0.003 | 0.012 |

| Variant | mean H3v_rn (runs where it edits) | edit set exists | edit rate | failure rate (H3v < 0.25, no edit = fail) | failure rate when it edits |
|---|---|---|---|---|---|
| EMBER | 0.39 | 0.68 | 0.68 | 0.43 | 0.17 |
| CINDER | 0.52 | 0.87 | 0.87 | 0.17 | 0.04 |
| CINDER-RS (CS + RS-1% + TWJ) | 0.49 | 0.92 | 0.92 | 0.18 | 0.11 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 0.47 | 0.87 | 0.87 | 0.28 | 0.17 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 0.46 | 0.92 | 0.92 | 0.23 | 0.16 |

## Secondary: paper rule, H2

| Variant − EMBER | concepts (pairs) | mean Δ | 95% CI | wins/losses | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|
| CINDER | 11 (38) | -0.134 | [-0.202, -0.056] | 1/10 | 0.019 | 0.056 |
| CINDER-RS (CS + RS-1% + TWJ) | 11 (37) | -0.153 | [-0.249, -0.041] | 2/9 | 0.042 | 0.084 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 11 (38) | +0.161 | [+0.087, +0.242] | 11/0 | 0.001 | 0.004 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 11 (37) | +0.088 | [+0.026, +0.166] | 9/2 | 0.042 | 0.084 |

| Variant | mean H2 (runs where it edits) | edit set exists | edit rate | failure rate (H3v < 0.25, no edit = fail) | failure rate when it edits |
|---|---|---|---|---|---|
| EMBER | 0.59 | 0.68 | 0.68 | 0.37 | 0.07 |
| CINDER | 0.46 | 0.87 | 0.87 | 0.17 | 0.04 |
| CINDER-RS (CS + RS-1% + TWJ) | 0.44 | 0.92 | 0.92 | 0.17 | 0.09 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 0.77 | 0.87 | 0.87 | 0.13 | 0.00 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 0.67 | 0.92 | 0.92 | 0.10 | 0.02 |

## Secondary: paper rule, H3v_rn

| Variant − EMBER | concepts (pairs) | mean Δ | 95% CI | wins/losses | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|
| CINDER | 11 (38) | -0.061 | [-0.112, +0.002] | 2/9 | 0.067 | 0.202 |
| CINDER-RS (CS + RS-1% + TWJ) | 11 (37) | -0.089 | [-0.178, +0.013] | 3/8 | 0.102 | 0.203 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 11 (38) | +0.115 | [+0.039, +0.199] | 10/1 | 0.014 | 0.055 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 11 (37) | +0.049 | [-0.015, +0.123] | 7/4 | 0.278 | 0.278 |

| Variant | mean H3v_rn (runs where it edits) | edit set exists | edit rate | failure rate (H3v < 0.25, no edit = fail) | failure rate when it edits |
|---|---|---|---|---|---|
| EMBER | 0.59 | 0.68 | 0.68 | 0.37 | 0.07 |
| CINDER | 0.53 | 0.87 | 0.87 | 0.17 | 0.04 |
| CINDER-RS (CS + RS-1% + TWJ) | 0.50 | 0.92 | 0.92 | 0.17 | 0.09 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 0.71 | 0.87 | 0.87 | 0.13 | 0.00 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 0.65 | 0.92 | 0.92 | 0.10 | 0.02 |

## Secondary: E2 integrity at the selected point (judge-free)

| Variant | copy | echo | runs |
|---|---|---|---|
| EMBER | 0.40 | 0.40 | 41 |
| CINDER | 0.42 | 0.38 | 52 |
| CINDER-RS (CS + RS-1% + TWJ) | 0.41 | 0.37 | 55 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 0.44 | 0.41 | 52 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 0.41 | 0.39 | 55 |

| Variant − EMBER, copy | concepts (pairs) | mean Δ | 95% CI | Wilcoxon p |
|---|---|---|---|---|
| CINDER | 11 (38) | +0.036 | [-0.225, +0.276] | 0.966 |
| CINDER-RS (CS + RS-1% + TWJ) | 11 (37) | -0.038 | [-0.236, +0.169] | 0.765 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 11 (38) | +0.046 | [-0.167, +0.214] | 0.147 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 11 (37) | +0.021 | [-0.119, +0.160] | 0.700 |

## Secondary: T3 labels at the selected point (seed 42, blind agents)

Cross-agent agreement on 202 expansion duplicates: exact label 0.97, destruction yes/no 0.99, leak yes/no 0.99.

| Variant | n | graceful | leak | confab | destruction | destruction, exclusive (n) | destruction, multi (n) |
|---|---|---|---|---|---|---|---|
| EMBER | 342 | 0.27 | 0.15 | 0.08 | 0.49 | 0.65 (66) | 0.45 (276) |
| CINDER | 544 | 0.29 | 0.09 | 0.10 | 0.52 | 0.70 (64) | 0.49 (480) |
| CINDER-RS (CS + RS-1% + TWJ) | 507 | 0.23 | 0.08 | 0.08 | 0.60 | 0.77 (70) | 0.58 (437) |
| CINDER-z (A3z: CINDER selection, original-space Z) | 544 | 0.22 | 0.08 | 0.09 | 0.61 | 0.79 (61) | 0.58 (483) |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 507 | 0.25 | 0.06 | 0.07 | 0.61 | 0.81 (64) | 0.58 (443) |

| Variant − EMBER, destruction | concepts | mean Δ | 95% CI | Wilcoxon p |
|---|---|---|---|---|
| CINDER | 7 | -0.118 | [-0.467, +0.185] | 0.688 |
| CINDER-RS (CS + RS-1% + TWJ) | 6 | +0.194 | [-0.207, +0.531] | 0.312 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 7 | +0.051 | [-0.114, +0.216] | 0.609 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 6 | +0.238 | [+0.002, +0.481] | 0.188 |
