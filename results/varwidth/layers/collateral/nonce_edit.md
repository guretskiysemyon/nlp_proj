# Nonce-matching edit (PREREG amendment 13)

Seed 42. Selection as E3 (validation: MMLU ≥ 0.9, collateral ≥ 0.8, copy ≥ 0.9, max H3v); test split. Nonce point: test efficacy ≥ per-run nonce floor − 0.05, copy ≥ 0.9, test collateral ≥ 0.8.

| Variant | runs with a set | edits (passes the gates) | nonce point reached | test efficacy | nonce floor | collateral (rn) | copy | copy, exclusive | specificity | H3v | median r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nm:cinder | 10 | 10 | 1 | 0.04 | 0.36 | 0.99 | 0.99 | 0.97 | 1.00 | 0.08 | 0.18 |
| nm:free | 10 | 9 | 2 | 0.18 | 0.36 | 0.86 | 0.99 | 1.00 | 0.98 | 0.35 | 3 |
| nm:ember | 8 | 8 | 2 | 0.02 | 0.27 | 0.96 | 0.97 | 0.92 | 1.00 | 0.06 | 0.32 |
| nm:efree | 8 | 7 | 3 | 0.21 | 0.26 | 0.86 | 0.99 | 0.98 | 0.98 | 0.41 | 3.2 |
| nm:qa | 10 | 10 | 3 | 0.22 | 0.36 | 0.89 | 1.00 | 0.99 | 0.99 | 0.40 | 1.6 |

E3 baselines (copy-gated collateral rule) on the same runs:

| Baseline | edits | test efficacy | collateral (rn) | H3v | median r |
|---|---|---|---|---|---|
| sub:ember (E3) | 8 | 0.07 | 0.98 | 0.16 | 0.25 |
| sub:cinder (E3) | 10 | 0.14 | 0.97 | 0.30 | 0.25 |
| sub:cinder_z (E3) | 10 | 0.06 | 0.99 | 0.14 | 0.05 |

Paired test H3v, NM variant − E3 baseline (unit = concept; runs where both edit):

| Contrast | concepts | mean Δ | wins/losses | Wilcoxon p |
|---|---|---|---|---|
| nm:cinder − sub:ember (E3) | 7 | -0.113 | 0/4 | 0.125 |
| nm:cinder − sub:cinder_z (E3) | 10 | -0.052 | 3/4 | 0.578 |
| nm:free − sub:ember (E3) | 6 | +0.154 | 4/1 | 0.125 |
| nm:free − sub:cinder_z (E3) | 9 | +0.203 | 7/1 | 0.055 |
| nm:ember − sub:ember (E3) | 8 | -0.092 | 1/4 | 0.188 |
| nm:ember − sub:cinder_z (E3) | 7 | -0.099 | 1/3 | 0.625 |
| nm:efree − sub:ember (E3) | 7 | +0.276 | 6/1 | 0.047 |
| nm:efree − sub:cinder_z (E3) | 6 | +0.217 | 4/2 | 0.219 |
| nm:qa − sub:ember (E3) | 7 | +0.169 | 5/1 | 0.156 |
| nm:qa − sub:cinder_z (E3) | 10 | +0.260 | 7/1 | 0.039 |

Judge-free E2 at the selected point (means over runs; references over the same runs' words):

| Config | runs | copy | echo | context-MC | answer-NLL excess vs nonce |
|---|---|---|---|---|---|
| nm:cinder@sel | 10 | 0.99 | 0.94 | 0.53 | +0.16 |
| nm:free@sel | 9 | 0.99 | 0.47 | 0.52 | +0.17 |
| nm:ember@sel | 8 | 0.97 | 0.95 | 0.56 | +0.17 |
| nm:efree@sel | 7 | 0.99 | 0.34 | 0.52 | +0.19 |
| nm:qa@sel | 10 | 1.00 | 0.49 | 0.52 | +0.14 |
| sub:ember@sel | 8 | 0.44 | 0.47 | 0.53 | +0.21 |
| sub:cinder_z@sel | 10 | 0.38 | 0.35 | 0.49 | +0.39 |
| ref_orig / ref_nonce | – | 1.00 / 1.00 | 0.96 / 0.75 | 0.53 / 0.53 | 0 |

Renamed rate: the answer to the T3 prompt does not contain the word and names a capitalised non-dictionary string instead (string check):

| Config | answers | renamed |
|---|---|---|
| nm:cinder@sel | 544 | 0.01 |
| nm:efree@sel | 301 | 0.47 |
| nm:ember@sel | 342 | 0.01 |
| nm:free@sel | 488 | 0.36 |
| nm:qa@sel | 544 | 0.32 |
| ref_nonce (words of nm:cinder@sel) | 544 | 0.17 |
| ref_nonce (words of nm:efree@sel) | 301 | 0.17 |
| ref_nonce (words of nm:ember@sel) | 342 | 0.17 |
| ref_nonce (words of nm:free@sel) | 488 | 0.17 |
| ref_nonce (words of nm:qa@sel) | 544 | 0.17 |
| ref_nonce (words of sub:cinder_z@sel) | 544 | 0.17 |
| ref_nonce (words of sub:ember@sel) | 342 | 0.17 |
| ref_orig (words of nm:cinder@sel) | 544 | 0.00 |
| ref_orig (words of nm:efree@sel) | 301 | 0.00 |
| ref_orig (words of nm:ember@sel) | 342 | 0.00 |
| ref_orig (words of nm:free@sel) | 488 | 0.00 |
| ref_orig (words of nm:qa@sel) | 544 | 0.00 |
| ref_orig (words of sub:cinder_z@sel) | 544 | 0.01 |
| ref_orig (words of sub:ember@sel) | 342 | 0.00 |
| sub:cinder_z@sel | 544 | 0.08 |
| sub:ember@sel | 342 | 0.09 |

T3 labels at the selected point (seed 42, blind agents):

| Config | n | graceful | leak | confab | destruction |
|---|---|---|---|---|---|
| nm:cinder@sel | 544 | 0.55 | 0.39 | 0.04 | 0.01 |
| nm:efree@sel | 301 | 0.16 | 0.14 | 0.11 | 0.59 |
| nm:ember@sel | 342 | 0.37 | 0.53 | 0.06 | 0.04 |
| nm:free@sel | 488 | 0.24 | 0.14 | 0.12 | 0.50 |
| nm:qa@sel | 544 | 0.25 | 0.16 | 0.12 | 0.47 |
| sub:cinder_z@sel | 544 | 0.22 | 0.08 | 0.09 | 0.61 |
| sub:ember@sel | 342 | 0.27 | 0.15 | 0.08 | 0.49 |

Cross-agent agreement on 152 NM duplicates: exact 0.92, destruction 0.97.

Generation damage on the neutral prompts (blind OK/BROKEN vs the unedited answer):

| Config | prompts | broken |
|---|---|---|
| nm:cinder@nm | 200 | 0.02 |
| nm:efree@nm | 140 | 0.15 |
| nm:ember@nm | 160 | 0.04 |
| nm:free@nm | 180 | 0.16 |
| nm:qa@nm | 200 | 0.15 |

Agreement on 88 cross-batch duplicates: 1.00.

For reference (amendment 9, collateral-rule points): sub:cinder@sel 0.20, sub:cinder_rs@sel 0.27, sub:cinder_rs_z@sel 0.21, sub:cinder_z@paper 0.35, sub:cinder_z@sel 0.15, sub:ember@paper 0.31, sub:ember@sel 0.14.
