# Last-mile checks (PREREG amendment 9)

## Unreadability floor of efficacy

Efficacy (1 − normalized accuracy) of the UNEDITED model when the edit set's tokens are replaced by a nonce or deleted in the efficacy test questions, against the edited model's efficacy at the collateral-rule point (sel) and the paper-rule point. Means over runs; 'share ≤ del' = share of runs whose efficacy is at or below the deletion floor.

| Variant | runs | questions touched | floor: nonce | floor: deletion | efficacy @sel | share @sel ≤ del | efficacy @paper | share @paper ≤ del |
|---|---|---|---|---|---|---|---|---|
| EMBER | 41 | 0.71 | 0.29 | 0.23 | 0.29 | 0.37 | 0.47 | 0.17 |
| CINDER | 52 | 0.78 | 0.36 | 0.28 | 0.32 | 0.40 | 0.32 | 0.40 |
| CINDER-RS (CS + RS-1% + TWJ) | 55 | 0.77 | 0.35 | 0.26 | 0.31 | 0.53 | 0.30 | 0.55 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 52 | 0.78 | 0.36 | 0.28 | 0.39 | 0.17 | 0.68 | 0.02 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 55 | 0.77 | 0.35 | 0.26 | 0.38 | 0.22 | 0.56 | 0.07 |

## Generation damage (neutral prompts, blind OK/BROKEN labels)

Labelled 1440/1440; agreement on 144 cross-batch duplicates 0.99. 'token present' = prompts whose target token id is in the tokenized prompt.

| Config | prompts | broken | broken (token present) |
|---|---|---|---|
| sub:cinder@sel | 200 | 0.20 | 0.21 |
| sub:cinder_rs@sel | 180 | 0.27 | 0.28 |
| sub:cinder_rs_z@sel | 180 | 0.21 | 0.22 |
| sub:cinder_z@paper | 200 | 0.35 | 0.37 |
| sub:cinder_z@sel | 200 | 0.15 | 0.16 |
| sub:ember@paper | 240 | 0.31 | 0.33 |
| sub:ember@sel | 240 | 0.14 | 0.15 |

Matched concepts (every config present: Baseball, COVID-19 pandemic, Cannabis, Gambling, Gun, Halloween, Heroin, Pornography, Uranium):

| Config | prompts | broken |
|---|---|---|
| sub:cinder@sel | 180 | 0.22 |
| sub:cinder_rs@sel | 180 | 0.27 |
| sub:cinder_rs_z@sel | 180 | 0.21 |
| sub:cinder_z@paper | 180 | 0.34 |
| sub:cinder_z@sel | 180 | 0.16 |
| sub:ember@paper | 180 | 0.24 |
| sub:ember@sel | 180 | 0.15 |

## Alpaca coherence and H4, seed 42 (amendment 10; descriptive)

| Variant | point | runs | instr frac | fluency frac | coherence | share coherence < 0.9 | H (paper) | H4 |
|---|---|---|---|---|---|---|---|---|
| EMBER | sel | 8 | 1.00 | 0.95 | 0.97 | 0.00 | 0.53 | 0.58 |
| EMBER | paper | 8 | 0.99 | 0.96 | 0.97 | 0.00 | 0.76 | 0.74 |
| EMBER | e3 | 4 | 1.00 | 0.96 | 0.98 | 0.00 | 0.21 | 0.24 |
| CINDER | sel | 10 | 1.00 | 0.97 | 0.98 | 0.00 | 0.53 | 0.57 |
| CINDER | e3 | 4 | 0.99 | 0.95 | 0.97 | 0.00 | 0.28 | 0.33 |
| CINDER-RS (CS + RS-1% + TWJ) | sel | 10 | 1.00 | 0.96 | 0.98 | 0.00 | 0.53 | 0.57 |
| CINDER-RS (CS + RS-1% + TWJ) | e3 | 4 | 1.00 | 0.95 | 0.97 | 0.00 | 0.00 | 0.00 |
| CINDER-z (A3z: CINDER selection, original-space Z) | sel | 10 | 1.00 | 0.96 | 0.98 | 0.00 | 0.70 | 0.72 |
| CINDER-z (A3z: CINDER selection, original-space Z) | paper | 10 | 0.99 | 0.95 | 0.97 | 0.00 | 0.81 | 0.76 |
| CINDER-z (A3z: CINDER selection, original-space Z) | e3 | 4 | 1.00 | 0.96 | 0.98 | 0.00 | 0.00 | 0.00 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | sel | 10 | 1.00 | 0.95 | 0.97 | 0.00 | 0.61 | 0.65 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | e3 | 4 | 1.00 | 0.98 | 0.99 | 0.00 | 0.20 | 0.23 |

Paired H4 vs EMBER at the collateral-rule point, seed 42 (concepts where both edit):

| Variant − EMBER | concepts | mean ΔH4 | wins/losses |
|---|---|---|---|
| CINDER | 7 | -0.015 | 3/4 |
| CINDER-RS (CS + RS-1% + TWJ) | 6 | -0.032 | 3/3 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 7 | +0.140 | 7/0 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 6 | +0.064 | 4/2 |

Paired H4 vs EMBER at the E3 point (amendment 12), seed 42 (concepts where both edit):

| Variant − EMBER | concepts | mean ΔH4 | wins/losses |
|---|---|---|---|
| CINDER | 4 | +0.093 | 3/1 |
| CINDER-RS (CS + RS-1% + TWJ) | 4 | -0.239 | 0/2 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 4 | -0.239 | 0/2 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 4 | -0.011 | 1/2 |

## Word-free efficacy (amendment 11)

Efficacy on questions that contain none of the edited tokens: new = written to avoid every token edited for the concept (kept if the unedited model answers them); old = existing test questions without the run's edited tokens; word-free = both pooled. Means over runs.

| Variant | point | runs | items new / old (mean) | efficacy (all questions) | word-free: new | word-free: old | word-free: pooled | pooled / all |
|---|---|---|---|---|---|---|---|---|
| EMBER | sel | 1 | 51 / 6 | 0.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| EMBER | paper | 1 | 51 / 6 | 0.78 | 0.00 | 0.00 | 0.00 | 0.00 |
| CINDER | sel | 2 | 47 / 4 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 |
| CINDER | paper | 2 | 47 / 4 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 |
| CINDER-RS (CS + RS-1% + TWJ) | sel | 3 | 46 / 5 | 0.28 | 0.00 | 0.00 | 0.00 | 0.00 |
| CINDER-RS (CS + RS-1% + TWJ) | paper | 3 | 46 / 5 | 0.27 | 0.00 | 0.00 | 0.00 | 0.00 |
| CINDER-z (A3z: CINDER selection, original-space Z) | sel | 2 | 47 / 4 | 0.48 | 0.00 | 0.00 | 0.00 | 0.00 |
| CINDER-z (A3z: CINDER selection, original-space Z) | paper | 2 | 47 / 4 | 0.77 | 0.00 | 0.00 | 0.00 | 0.00 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | sel | 3 | 46 / 5 | 0.37 | 0.00 | 0.00 | 0.00 | 0.00 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | paper | 3 | 46 / 5 | 0.64 | 0.00 | 0.00 | 0.00 | 0.00 |

Paired vs EMBER at the collateral-rule point (unit = concept, Wilcoxon; Holm over 4):

| Variant − EMBER | concepts (pairs) | Δ word-free efficacy [95% CI] (p, Holm) | Δ H3v with word-free efficacy [95% CI] (p, Holm) |
|---|---|---|---|
| CINDER | 1 (1) | +0.000 [+0.000, +0.000] (1.000, 1.000) | +0.000 [+0.000, +0.000] (1.000, 1.000) |
| CINDER-RS (CS + RS-1% + TWJ) | 1 (1) | +0.000 [+0.000, +0.000] (1.000, 1.000) | +0.000 [+0.000, +0.000] (1.000, 1.000) |
| CINDER-z (A3z: CINDER selection, original-space Z) | 1 (1) | +0.000 [+0.000, +0.000] (1.000, 1.000) | +0.000 [+0.000, +0.000] (1.000, 1.000) |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 1 (1) | +0.000 [+0.000, +0.000] (1.000, 1.000) | +0.000 [+0.000, +0.000] (1.000, 1.000) |

## E3: integrity-gated selection (copy ≥ 0.9; amendment 11)

| Variant − EMBER | concepts (pairs) | mean ΔH3v | 95% CI | wins/losses | p | Holm p |
|---|---|---|---|---|---|---|
| CINDER | 11 (38) | +0.096 | [-0.006, +0.214] | 7/3 | 0.131 | 0.422 |
| CINDER-RS (CS + RS-1% + TWJ) | 11 (37) | +0.092 | [-0.012, +0.192] | 7/3 | 0.105 | 0.422 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 11 (38) | +0.029 | [-0.107, +0.154] | 7/3 | 0.375 | 0.750 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 11 (37) | +0.026 | [-0.088, +0.139] | 5/4 | 0.570 | 0.750 |

| Variant | edit rate under E3 | mean efficacy | mean collateral | mean H3v | median r | copy, exclusive words | copy, multi-meaning words |
|---|---|---|---|---|---|---|---|
| EMBER | 0.68 | 0.06 | 0.98 | 0.12 | 0.1 | 0.98 | 0.99 |
| CINDER | 0.87 | 0.12 | 0.97 | 0.25 | 0.25 | 0.96 | 0.97 |
| CINDER-RS (CS + RS-1% + TWJ) | 0.92 | 0.14 | 0.98 | 0.27 | 0.25 | 0.94 | 0.97 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 0.87 | 0.07 | 0.99 | 0.16 | 0.25 | 0.97 | 0.99 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 0.92 | 0.07 | 0.99 | 0.16 | 0.1 | 0.98 | 0.98 |

## Context damage at the collateral-rule point (existing E2 outputs)

Context-MC: an unrelated MMLU question asked right after a sentence that contains the edited word (same items for the edited model and the references). Answer-NLL excess: mean answer NLL after the word's sentence minus the same with the word's nonce (nats). Means over runs; the references are the same words in the unedited model.

| Variant | runs | context-MC | ref: real word | ref: nonce | Δ vs real word | answer-NLL excess over nonce |
|---|---|---|---|---|---|---|
| EMBER | 41 | 0.525 | 0.540 | 0.533 | -0.015 | +0.285 |
| CINDER | 52 | 0.530 | 0.535 | 0.529 | -0.005 | +0.128 |
| CINDER-RS (CS + RS-1% + TWJ) | 55 | 0.543 | 0.549 | 0.541 | -0.006 | +0.123 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 52 | 0.492 | 0.533 | 0.526 | -0.041 | +0.388 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 55 | 0.503 | 0.548 | 0.538 | -0.044 | +0.365 |

| Variant − EMBER | concepts (pairs) | Δ context-MC [95% CI] | p | Δ answer-NLL excess [95% CI] | p |
|---|---|---|---|---|---|
| CINDER | 11 (38) | +0.014 [-0.012, +0.047] | 0.638 | -0.178 [-0.424, +0.002] | 0.102 |
| CINDER-RS (CS + RS-1% + TWJ) | 11 (37) | +0.021 [-0.009, +0.055] | 0.322 | -0.177 [-0.430, -0.007] | 0.175 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 11 (38) | -0.027 [-0.057, +0.006] | 0.102 | +0.043 [-0.224, +0.241] | 0.102 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 11 (37) | -0.029 [-0.058, +0.001] | 0.102 | +0.081 [-0.111, +0.217] | 0.054 |
