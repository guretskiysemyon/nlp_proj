# Multilingual erasure: A1 / A2 / A3 / A3z (Llama-3.1-8B-Instruct, NF4, 8 languages)

Test score = mean over eligible languages of H_L = HM(efficacy, specificity, fluency); 0 when a gate fails
(* = gated on test, ungated value in parentheses). [v2] = it/pt/hi/th on the cleaned question sets.
— = no selected feature, no edit (score 0).

| Concept | A1 per-language EMBER | A2 multilingual EMBER | A3 multilingual CINDER | A3z CINDER, original-space edit |
|---|---|---|---|---|
| Culture of Greece | 0.31 [0.30] δ10 | 0.28 [0.27] δ200 | 0.22 [0.23] δ50 | 0.53 [0.55] δ10 |
| Golf | 0.33 [0.33] δ5 | — | — | — |
| Republic of Ireland | 0.33 [0.24] δ10 | — | — | — |
| Ancient Rome | 0.41 [0.42] δ10 | — | — | — |
| Baseball | 0.51 [0.54] δ10 | 0.47 [0.52] δ5 | 0.34 [0.35] δ200 | 0.67 [0.69] δ10 |
| Uranium | 0.59 [0.58] δ10 | — | 0.15 [0.12] δ200 | 0.18 [0.15] δ10 |
| Harry Potter | 0.80 [0.79] δ10 | 0.67 [0.65] δ10 | 0.24 [0.26] δ5 | 0.34 [0.34] δ200 |
| Cannabis | 0.42 [0.41] δ2 | — | 0.59 [0.57] δ200 | 0.87 [0.85] δ10 |
| Gambling | 0.53 [0.54] δ5 | 0.49 [0.50] δ10 | 0.20 [0.22] δ100 | 0.40 [0.38] δ10 |
| Gun | 0.21 [0.22] δ2 | 0.40 [0.39] δ10 | 0.21 [0.18] δ200 | 0.54 [0.53] δ10 |
| Pornography | 0.67 [0.64] δ5 | 0.74 [0.74] δ10 | 0.31 [0.28] δ10 | 0.00* (0.76) [0.00] δ50 |
| COVID-19 pandemic | 0.74 [0.70] δ5 | 0.48 [0.42] δ200 | 0.38 [0.37] δ200 | 0.75 [0.73] δ5 |
| Valentine's Day | — | — | — | — |
| World War II | 0.27 [0.27] δ50 | — | — | — |
| Nazism | — | — | 0.10 [0.09] δ200 | 0.34 [0.33] δ50 |
| Heroin | — | — | — | — |
| Halloween | — | — | — | — |
| Artificial intelligence | 0.16 [0.22] δ2 | 0.00* (0.48) [0.00] δ10 | 0.32 [0.33] δ200 | 0.00* (0.52) [0.00] δ10 |

## Summary

| | A1 | A2 | A3 | A3z |
|---|---|---|---|---|
| concepts finished | 18 | 18 | 18 | 18 |
| concepts with an edit | 14 | 8 | 11 | 11 |
| mean test score, all 18 (no edit = 0) | 0.350 | 0.196 | 0.170 | 0.257 |
| mean v2 score, all 18 | 0.344 | 0.194 | 0.166 | 0.253 |

Shared concepts for A1/A2/A3 (n=8: Culture of Greece, Baseball, Harry Potter, Gambling, Gun, Pornography, COVID-19 pandemic, Artificial intelligence): A1 0.492, A2 0.442, A3 0.278; wins A1 5, A2 2, A3 1

Shared concepts for A1/A2/A3/A3z (n=8: Culture of Greece, Baseball, Harry Potter, Gambling, Gun, Pornography, COVID-19 pandemic, Artificial intelligence): A1 0.492, A2 0.442, A3 0.278, A3z 0.404; wins A1 2, A2 1, A3 1, A3z 4

Shared concepts for A2/A3z (n=8: Culture of Greece, Baseball, Harry Potter, Gambling, Gun, Pornography, COVID-19 pandemic, Artificial intelligence): A2 0.442, A3z 0.404; wins A2 4, A3z 4

Shared concepts for A3/A3z (n=11: Culture of Greece, Baseball, Uranium, Harry Potter, Cannabis, Gambling, Gun, Pornography, COVID-19 pandemic, Nazism, Artificial intelligence): A3 0.279, A3z 0.420; wins A3 2, A3z 9

## Efficacy per language at the chosen δ (mean over concepts with an edit, test)

| | en | de | fr | it | pt | hi | es | th |
|---|---|---|---|---|---|---|---|---|
| A1 | 0.48 | 0.32 | 0.31 | 0.40 | 0.34 | 0.06 | 0.30 | 0.09 |
| A2 | 0.72 | 0.28 | 0.29 | 0.43 | 0.36 | 0.08 | 0.40 | 0.19 |
| A3 | 0.21 | 0.16 | 0.13 | 0.15 | 0.13 | 0.02 | 0.16 | 0.09 |
| A3z | 0.65 | 0.41 | 0.44 | 0.53 | 0.47 | 0.08 | 0.44 | 0.15 |

## Selection cost per concept (mean)

| | factorizations | rows factorized | ALS iterations | judged candidates |
|---|---|---|---|---|
| A1 | 8 | 30,793 | 3,136 | 80 |
| A2 | 1 | 6,540 | 357 | 10 |
| A3 | 1 | 6,540 | 355 | 10 |
| A3z | A3's fit + one k×k ridge solve | | | as A3 |
