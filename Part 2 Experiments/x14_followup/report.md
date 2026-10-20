# X14: follow-up analyses of the fair run (held-out tests, 18 concepts)

Arms: ember, cinder, ember_hj, ember_x, cinder_x, ml_sep. Reliable cells only (pristine QA_test ≥ 0.4). foreign_ret = mean retention over the reliable de/es/fr/ru cells (lower = more erased); leak = foreign_ret − ret_en. Definitions in the script docstring.

## Means over the 18 concepts (no edit counts as retention 1)

| arm | no_edit | ret_en | foreign_ret | leak |
|---|---|---|---|---|
| ember | 4 | 0.754 | 0.859 | 0.105 |
| cinder | 3 | 0.741 | 0.887 | 0.146 |
| ember_hj | 0 | 0.674 | 0.827 | 0.153 |
| ember_x | 4 | 0.743 | 0.789 | 0.046 |
| cinder_x | 3 | 0.73 | 0.837 | 0.107 |
| ml_sep | 1 | 0.588 | 0.686 | 0.097 |

## 1–2. Paired differences: leak and foreign retention, all 18 concepts and only where both arms edit

Two-level bootstrap 95% CI, Wilcoxon over concepts, Holm within each block of rows (same analysis label).

| analysis | pair | concepts | mean | ci_low | ci_high | lower | higher | equal | wilcoxon_p | holm_p |
|---|---|---|---|---|---|---|---|---|---|---|
| leak, all 18 | ember_x − ember | 18 | -0.059 | -0.102 | -0.023 | 12 | 2 | 4 | 0.0012 | 0.0096 |
| leak, all 18 | cinder_x − cinder | 18 | -0.039 | -0.088 | -0.004 | 10 | 4 | 4 | 0.0203 | 0.1421 |
| leak, all 18 | cinder − ember | 18 | 0.041 | -0.06 | 0.137 | 9 | 9 | 0 | 0.3927 | 1.0 |
| leak, all 18 | cinder_x − ember_x | 18 | 0.061 | -0.035 | 0.15 | 6 | 12 | 0 | 0.1084 | 0.542 |
| leak, all 18 | ember_hj − ember | 18 | 0.048 | -0.006 | 0.119 | 0 | 4 | 14 |  | 1.0 |
| leak, all 18 | cinder − ember_hj | 18 | -0.007 | -0.094 | 0.081 | 11 | 7 | 0 | 0.7337 | 1.0 |
| leak, all 18 | ml_sep − ember | 18 | -0.008 | -0.087 | 0.074 | 8 | 6 | 4 | 0.7609 | 1.0 |
| leak, all 18 | ml_sep − ember_x | 18 | 0.051 | -0.027 | 0.136 | 4 | 13 | 1 | 0.0714 | 0.4284 |
| foreign_ret, all 18 | ember_x − ember | 18 | -0.07 | -0.109 | -0.04 | 14 | 0 | 4 | 0.0001 | 0.0008 |
| foreign_ret, all 18 | cinder_x − cinder | 18 | -0.05 | -0.095 | -0.022 | 12 | 2 | 4 | 0.0023 | 0.0138 |
| foreign_ret, all 18 | cinder − ember | 18 | 0.028 | -0.037 | 0.089 | 9 | 9 | 0 | 0.3927 | 1.0 |
| foreign_ret, all 18 | cinder_x − ember_x | 18 | 0.048 | -0.048 | 0.126 | 8 | 10 | 0 | 0.3465 | 1.0 |
| foreign_ret, all 18 | ember_hj − ember | 18 | -0.032 | -0.081 | -0.004 | 4 | 0 | 14 |  | 1.0 |
| foreign_ret, all 18 | cinder − ember_hj | 18 | 0.06 | 0.014 | 0.113 | 6 | 11 | 1 | 0.015 | 0.075 |
| foreign_ret, all 18 | ml_sep − ember | 18 | -0.174 | -0.266 | -0.088 | 13 | 0 | 5 | 0.0002 | 0.0014 |
| foreign_ret, all 18 | ml_sep − ember_x | 18 | -0.103 | -0.198 | -0.009 | 10 | 7 | 1 | 0.1089 | 0.4356 |
| leak, both edit | ember_x − ember | 14 | -0.076 | -0.126 | -0.033 | 12 | 2 | 0 | 0.0012 | 0.0096 |
| leak, both edit | cinder_x − cinder | 15 | -0.047 | -0.105 | -0.004 | 10 | 4 | 1 | 0.0203 | 0.1421 |
| leak, both edit | cinder − ember | 11 | 0.013 | -0.096 | 0.126 | 6 | 5 | 0 | 0.9658 | 1.0 |
| leak, both edit | cinder_x − ember_x | 11 | 0.033 | -0.081 | 0.142 | 4 | 7 | 0 | 0.6377 | 1.0 |
| leak, both edit | ember_hj − ember | 14 | 0.0 | 0.0 | 0.0 | 0 | 0 | 14 |  | 1.0 |
| leak, both edit | cinder − ember_hj | 15 | 0.018 | -0.074 | 0.117 | 8 | 7 | 0 | 0.7197 | 1.0 |
| leak, both edit | ml_sep − ember | 14 | -0.047 | -0.121 | 0.024 | 8 | 3 | 3 | 0.083 | 0.498 |
| leak, both edit | ml_sep − ember_x | 14 | 0.029 | -0.049 | 0.111 | 4 | 10 | 0 | 0.2676 | 1.0 |
| foreign_ret, both edit | ember_x − ember | 14 | -0.09 | -0.134 | -0.057 | 14 | 0 | 0 | 0.0001 | 0.0008 |
| foreign_ret, both edit | cinder_x − cinder | 15 | -0.06 | -0.111 | -0.028 | 12 | 2 | 1 | 0.0023 | 0.014 |
| foreign_ret, both edit | cinder − ember | 11 | 0.053 | -0.011 | 0.123 | 5 | 6 | 0 | 0.2061 | 0.6183 |
| foreign_ret, both edit | cinder_x − ember_x | 11 | 0.077 | -0.017 | 0.16 | 4 | 7 | 0 | 0.1475 | 0.59 |
| foreign_ret, both edit | ember_hj − ember | 14 | 0.0 | 0.0 | 0.0 | 0 | 0 | 14 |  | 1.0 |
| foreign_ret, both edit | cinder − ember_hj | 15 | 0.044 | -0.002 | 0.1 | 6 | 8 | 1 | 0.104 | 0.52 |
| foreign_ret, both edit | ml_sep − ember | 14 | -0.156 | -0.254 | -0.064 | 10 | 0 | 4 | 0.002 | 0.014 |
| foreign_ret, both edit | ml_sep − ember_x | 14 | -0.066 | -0.164 | 0.03 | 7 | 7 | 0 | 0.5416 | 1.0 |

## 3. Collateral on SimdomQA_test: share of pristine-correct questions lost, by whether the question contains an edited token

| arm | lang_kind | touched | questions | lost |
|---|---|---|---|---|
| cinder | en | False | 549 | 0.0 |
| cinder | en | True | 115 | 0.217 |
| cinder | foreign | False | 2103 | 0.0 |
| cinder | foreign | True | 67 | 0.284 |
| cinder_x | en | False | 534 | 0.0 |
| cinder_x | en | True | 130 | 0.223 |
| cinder_x | foreign | False | 1915 | 0.001 |
| cinder_x | foreign | True | 255 | 0.161 |
| ember | en | False | 582 | 0.0 |
| ember | en | True | 46 | 0.522 |
| ember | foreign | False | 1994 | 0.001 |
| ember | foreign | True | 63 | 0.429 |
| ember_hj | en | False | 744 | 0.0 |
| ember_hj | en | True | 61 | 0.443 |
| ember_hj | foreign | False | 2564 | 0.001 |
| ember_hj | foreign | True | 73 | 0.37 |
| ember_x | en | False | 561 | 0.0 |
| ember_x | en | True | 67 | 0.478 |
| ember_x | foreign | False | 1896 | 0.003 |
| ember_x | foreign | True | 161 | 0.404 |
| ml_sep | en | False | 641 | 0.002 |
| ml_sep | en | True | 114 | 0.316 |
| ml_sep | foreign | False | 2194 | 0.004 |
| ml_sep | foreign | True | 262 | 0.431 |

## 4. Exposure: share of other concepts' sentences that contain an edited token

| arm | de | en | es | fr | ru | foreign_mean | concepts |
|---|---|---|---|---|---|---|---|
| cinder | 0.0083 | 0.0191 | 0.0056 | 0.0072 | 0.0005 | 0.0054 | 15 |
| cinder_x | 0.0269 | 0.0257 | 0.0259 | 0.0296 | 0.0183 | 0.0252 | 15 |
| ember | 0.0073 | 0.0084 | 0.0079 | 0.004 | 0.0002 | 0.0048 | 14 |
| ember_hj | 0.0083 | 0.0095 | 0.0074 | 0.0043 | 0.0003 | 0.0051 | 18 |
| ember_x | 0.0273 | 0.0141 | 0.0229 | 0.0244 | 0.01 | 0.0212 | 14 |
| ml_sep | 0.0219 | 0.0469 | 0.0178 | 0.0172 | 0.0102 | 0.0168 | 17 |

