# CINDER vs EMBER (English fit, 6 concepts)

## Features and chosen strength

| concept | EMBER features / tokens | EMBER δ | EMBER h_alpaca | CINDER features (TJ, LO@5) / tokens | CINDER δ | CINDER h_alpaca |
|---|---|---|---|---|---|---|
| Cannabis | 1 / 45 | 200 | 0.568 | 1 (TJ [42], LO@5 [42]) / 34 | 5 | 0.434 |
| Gun | 2 / 61 | 5 | 0.663 | 2 (TJ [37, 40], LO@5 [37]) / 67 | 100 | 0.641 |
| Nazism | 1 / 41 | 100 | 0.772 | 0: no edit | - | - |
| Republic of Ireland | 1 / 31 | 100 | 0.589 | 0: no edit | - | - |
| Culture of Greece | 1 / 52 | 200 | 0.938 | 1 (TJ [10], LO@5 -) / 106 | 100 | 0.822 |
| Golf | 2 / 70 | 10 | 0.672 | 1 (TJ [33], LO@5 [33]) / 44 | 10 | 0.526 |

## Retention of concept knowledge on QA_test (lower = more erased; leak = r_L - r_en)

| concept | method | en | de | es | fr | ru | mean leak de/es/fr/ru |
|---|---|---|---|---|---|---|---|
| Cannabis | EMBER | 0.39 | 0.40 | 0.45 | 0.38 | 0.90 | +0.14 |
| Cannabis | CINDER | 0.59 | 0.65 | 0.66 | 0.67 | 0.85 | +0.12 |
| Gun | EMBER | 0.54 | 1.00 | 0.95 | 0.89 | 1.00 | +0.42 |
| Gun | CINDER | 0.68 | 0.95 | 0.95 | 0.89 | 1.00 | +0.26 |
| Nazism | EMBER | 0.63 | 0.60 | 0.79 | 0.71 | 1.00 | +0.14 |
| Nazism | CINDER (no edit) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 |
| Republic of Ireland | EMBER | 0.67 | 0.85 | 0.86 | 0.88 | 1.00 | +0.23 |
| Republic of Ireland | CINDER (no edit) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 |
| Culture of Greece | EMBER | 0.73 | 0.63 | 0.95 | 0.86 | 1.00 | +0.13 |
| Culture of Greece | CINDER | 0.46 | 0.63 | 0.89 | 0.79 | 1.00 | +0.37 |
| Golf | EMBER | 0.75 | 0.84 | 0.74 | 0.62 | 1.00 | +0.06 |
| Golf | CINDER | 0.89 | 0.88 | 0.87 | 0.70 | 1.00 | -0.03 |
| **mean, concepts CINDER edits** | EMBER | 0.60 | 0.72 | 0.77 | 0.69 | 0.98 | +0.19 |
| **mean, concepts CINDER edits** | CINDER | 0.66 | 0.78 | 0.84 | 0.76 | 0.96 | +0.18 |
| **mean, all 6** | EMBER | 0.62 | 0.72 | 0.79 | 0.72 | 0.98 | +0.18 |
| **mean, all 6** | CINDER | 0.77 | 0.85 | 0.90 | 0.84 | 0.98 | +0.12 |
