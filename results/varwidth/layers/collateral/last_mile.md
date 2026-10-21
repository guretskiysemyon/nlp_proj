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
| CINDER | sel | 10 | 1.00 | 0.97 | 0.98 | 0.00 | 0.53 | 0.57 |
| CINDER-RS (CS + RS-1% + TWJ) | sel | 10 | 1.00 | 0.96 | 0.98 | 0.00 | 0.53 | 0.57 |
| CINDER-z (A3z: CINDER selection, original-space Z) | sel | 10 | 1.00 | 0.96 | 0.98 | 0.00 | 0.70 | 0.72 |
| CINDER-z (A3z: CINDER selection, original-space Z) | paper | 10 | 0.99 | 0.95 | 0.97 | 0.00 | 0.81 | 0.76 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | sel | 10 | 1.00 | 0.95 | 0.97 | 0.00 | 0.61 | 0.65 |

Paired H4 vs EMBER at the collateral-rule point, seed 42 (concepts where both edit):

| Variant − EMBER | concepts | mean ΔH4 | wins/losses |
|---|---|---|---|
| CINDER | 7 | -0.015 | 3/4 |
| CINDER-RS (CS + RS-1% + TWJ) | 6 | -0.032 | 3/3 |
| CINDER-z (A3z: CINDER selection, original-space Z) | 7 | +0.140 | 7/0 |
| CINDER-RS-z (A3z: CINDER-RS selection, original-space Z) | 6 | +0.064 | 4/2 |
