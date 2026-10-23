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
| French Revolution | — | — | — | — |
| Tuberculosis | — | — | 0.12 δ5 | 0.11 δ1 |
| Cricket | 0.56 δ100 | — | — | — |
| Buddhism | 0.02 δ1 | — | 0.10 δ100 | 0.67 δ100 |
| Plutonium | — | — | 0.41 δ10 | 0.31 δ5 |
| Judaism | 0.07 δ2 | — | 0.29 δ50 | 0.00* (0.64) δ5 |
| Karate | 0.00 δ0 | — | — | — |
| Existentialism | 0.47 δ5 | — | — | — |
| Prime number | 0.00 δ0 | — | 0.16 δ200 | 0.52 δ10 |
| The Matrix | — | — | — | — |
| Louvre | — | — | — | — |
| American Revolutionary War | 0.56 δ100 | — | — | — |
| Robot | 0.57 δ5 | 0.80 δ50 | 0.35 δ100 | 0.70 δ5 |
| Sigmund Freud | — | — | — | — |
| Giraffe | — | — | — | — |
| Surfing | 0.50 δ5 | — | 0.31 δ200 | 0.76 δ5 |
| Flush toilet | — | — | — | — |
| Cloud | 0.00* (0.17) δ50 | — | 0.04 δ2 | 0.08 δ1 |
| Doctor of Philosophy | 0.18 δ5 | — | 0.38 δ50 | 0.75 δ10 |
| Tourism | 0.29 δ2 | 0.00* (0.63) δ50 | 0.12 δ200 | 0.32 δ2 |

## Summary

| | A1 | A2 | A3 | A3z |
|---|---|---|---|---|
| concepts finished | 38 | 38 | 38 | 38 |
| concepts with an edit | 26 | 10 | 21 | 21 |
| mean test score, all 18 (no edit = 0) | 0.251 | 0.114 | 0.141 | 0.232 |
| mean v2 score, all 18 | partial | partial | partial | partial |

Shared concepts for A1/A2/A3 (n=10: Culture of Greece, Baseball, Harry Potter, Gambling, Gun, Pornography, COVID-19 pandemic, Artificial intelligence, Robot, Tourism): A1 0.479, A2 0.434, A3 0.269; wins A1 6, A2 3, A3 1

Shared concepts for A1/A2/A3/A3z (n=10: Culture of Greece, Baseball, Harry Potter, Gambling, Gun, Pornography, COVID-19 pandemic, Artificial intelligence, Robot, Tourism): A1 0.479, A2 0.434, A3 0.269, A3z 0.425; wins A1 2, A2 2, A3 1, A3z 5

Shared concepts for A2/A3z (n=10: Culture of Greece, Baseball, Harry Potter, Gambling, Gun, Pornography, COVID-19 pandemic, Artificial intelligence, Robot, Tourism): A2 0.434, A3z 0.425; wins A2 5, A3z 5

Shared concepts for A3/A3z (n=21: Culture of Greece, Baseball, Uranium, Harry Potter, Cannabis, Gambling, Gun, Pornography, COVID-19 pandemic, Nazism, Artificial intelligence, Tuberculosis, Buddhism, Plutonium, Judaism, Prime number, Robot, Surfing, Cloud, Doctor of Philosophy, Tourism): A3 0.255, A3z 0.421; wins A3 5, A3z 16

## Efficacy per language at the chosen δ (mean over concepts with an edit, test)

| | en | de | fr | it | pt | hi | es | th |
|---|---|---|---|---|---|---|---|---|
| A1 | 0.40 | 0.30 | 0.27 | 0.32 | 0.28 | 0.03 | 0.26 | 0.06 |
| A2 | 0.77 | 0.37 | 0.43 | 0.46 | 0.40 | 0.07 | 0.43 | 0.29 |
| A3 | 0.21 | 0.14 | 0.14 | 0.14 | 0.12 | 0.01 | 0.17 | 0.06 |
| A3z | 0.67 | 0.39 | 0.47 | 0.48 | 0.43 | 0.05 | 0.48 | 0.16 |

## Selection cost per concept (mean)

| | factorizations | rows factorized | ALS iterations | judged candidates |
|---|---|---|---|---|
| A1 | 8 | 30,598 | 3,119 | 80 |
| A2 | 1 | 6,474 | 342 | 10 |
| A3 | 1 | 6,474 | 365 | 10 |
| A3z | A3's fit + one k×k ridge solve | | | as A3 |
