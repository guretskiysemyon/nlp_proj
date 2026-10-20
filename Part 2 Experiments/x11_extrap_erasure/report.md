# X11: which extrapolation works best? (EMBER, 6 concepts, real erasure, train questions)

ret_L = chance-corrected retention of QA_train in L vs the pristine model (lower = more erased); leak = mean(ret de/es/fr/ru) - ret_en; sim_L = SimdomQA_train accuracy change vs pristine (closer to 0 = less collateral); mmlu_frac = MMLU (50 train items) / pristine. δ = the fair run's EMBER pick and its neighbour on the grid. Variants: see the script docstring.

## Mean over concepts, δ = pick

| variant | concepts | ret_en | ret_de | ret_es | ret_fr | ret_ru | ret_foreign | leak | sim_en | sim_foreign | mmlu_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMBER | 6 | 0.634 | 0.76 | 0.794 | 0.719 | 0.976 | 0.813 | 0.178 | -0.027 | -0.007 | 1.0 |
| X | 6 | 0.582 | 0.601 | 0.719 | 0.494 | 0.911 | 0.681 | 0.099 | -0.04 | -0.022 | 1.0 |
| X_foreign | 6 | 0.588 | 0.587 | 0.751 | 0.484 | 0.911 | 0.683 | 0.096 | -0.033 | -0.02 | 1.0 |
| X_clean | 6 | 0.582 | 0.601 | 0.727 | 0.504 | 0.95 | 0.696 | 0.114 | -0.04 | -0.018 | 1.0 |

- X_foreign − X (pick): ret_foreign +0.002 (lower in 3 of 6), sim_foreign +0.002
- X_clean − X (pick): ret_foreign +0.014 (lower in 0 of 6), sim_foreign +0.004
- EMBER − X (pick): ret_foreign +0.131 (lower in 0 of 6), sim_foreign +0.015

## Mean over concepts, δ = neighbour

| variant | concepts | ret_en | ret_de | ret_es | ret_fr | ret_ru | ret_foreign | leak | sim_en | sim_foreign | mmlu_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMBER | 6 | 0.67 | 0.738 | 0.809 | 0.734 | 0.976 | 0.814 | 0.144 | -0.033 | -0.008 | 1.0 |
| X | 6 | 0.625 | 0.6 | 0.693 | 0.476 | 0.86 | 0.657 | 0.033 | -0.047 | -0.027 | 0.994 |
| X_foreign | 6 | 0.625 | 0.599 | 0.726 | 0.486 | 0.86 | 0.668 | 0.043 | -0.04 | -0.024 | 0.994 |
| X_clean | 6 | 0.625 | 0.593 | 0.702 | 0.486 | 0.884 | 0.666 | 0.042 | -0.047 | -0.022 | 0.994 |

- X_foreign − X (neighbour): ret_foreign +0.010 (lower in 1 of 6), sim_foreign +0.002
- X_clean − X (neighbour): ret_foreign +0.009 (lower in 0 of 6), sim_foreign +0.005
- EMBER − X (neighbour): ret_foreign +0.157 (lower in 0 of 6), sim_foreign +0.018

## Per concept

| concept | variant | delta | which | n_tokens | ret_en | ret_de | ret_es | ret_fr | ret_ru | ret_foreign | leak | sim_en | sim_foreign | mmlu_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cannabis | EMBER | 200.0 | pick | 45 | 0.738 | 0.511 | 0.673 | 0.424 | 0.92 | 0.632 | -0.106 | 0.0 | 0.0 | 1.0 |
| Cannabis | EMBER | 100.0 | neighbour | 45 | 0.738 | 0.511 | 0.673 | 0.424 | 0.92 | 0.632 | -0.106 | 0.0 | 0.0 | 1.0 |
| Cannabis | X | 200.0 | pick | 120 | 0.639 | 0.467 | 0.673 | 0.39 | 0.76 | 0.573 | -0.066 | -0.02 | 0.0 | 1.0 |
| Cannabis | X | 100.0 | neighbour | 120 | 0.639 | 0.467 | 0.592 | 0.424 | 0.6 | 0.521 | -0.118 | 0.0 | 0.0 | 1.0 |
| Cannabis | X_foreign | 200.0 | pick | 110 | 0.672 | 0.422 | 0.673 | 0.39 | 0.76 | 0.561 | -0.111 | 0.0 | 0.0 | 1.0 |
| Cannabis | X_foreign | 100.0 | neighbour | 110 | 0.672 | 0.422 | 0.592 | 0.424 | 0.6 | 0.509 | -0.163 | 0.0 | 0.0 | 1.0 |
| Cannabis | X_clean | 200.0 | pick | 113 | 0.639 | 0.467 | 0.673 | 0.39 | 0.76 | 0.573 | -0.066 | -0.02 | 0.0 | 1.0 |
| Cannabis | X_clean | 100.0 | neighbour | 113 | 0.639 | 0.467 | 0.592 | 0.424 | 0.6 | 0.521 | -0.118 | 0.0 | 0.0 | 1.0 |
| Gun | EMBER | 5.0 | pick | 61 | 0.843 | 1.0 | 1.0 | 0.939 | 1.0 | 0.985 | 0.142 | -0.04 | 0.005 | 1.0 |
| Gun | EMBER | 10.0 | neighbour | 61 | 0.804 | 0.92 | 1.0 | 0.879 | 1.0 | 0.95 | 0.146 | -0.06 | 0.0 | 1.0 |
| Gun | X | 5.0 | pick | 191 | 0.765 | 1.0 | 0.946 | 0.939 | 0.943 | 0.957 | 0.192 | -0.04 | 0.01 | 1.0 |
| Gun | X | 10.0 | neighbour | 191 | 0.686 | 1.0 | 0.838 | 0.758 | 0.771 | 0.842 | 0.156 | -0.08 | -0.01 | 1.0 |
| Gun | X_foreign | 5.0 | pick | 173 | 0.765 | 1.0 | 0.946 | 0.879 | 0.943 | 0.942 | 0.177 | -0.04 | 0.01 | 1.0 |
| Gun | X_foreign | 10.0 | neighbour | 173 | 0.686 | 1.0 | 0.838 | 0.818 | 0.771 | 0.857 | 0.171 | -0.06 | -0.005 | 1.0 |
| Gun | X_clean | 5.0 | pick | 177 | 0.765 | 1.0 | 0.946 | 1.0 | 0.943 | 0.972 | 0.207 | -0.04 | 0.015 | 1.0 |
| Gun | X_clean | 10.0 | neighbour | 177 | 0.686 | 1.0 | 0.838 | 0.818 | 0.771 | 0.857 | 0.171 | -0.08 | -0.005 | 1.0 |
| Nazism | EMBER | 100.0 | pick | 41 | 0.556 | 0.59 | 0.641 | 0.535 | 1.0 | 0.692 | 0.136 | -0.06 | -0.035 | 1.0 |
| Nazism | EMBER | 200.0 | neighbour | 41 | 0.556 | 0.538 | 0.641 | 0.535 | 1.0 | 0.678 | 0.122 | -0.06 | -0.035 | 1.0 |
| Nazism | X | 100.0 | pick | 122 | 0.467 | 0.333 | 0.436 | 0.349 | 1.0 | 0.529 | 0.062 | -0.06 | -0.085 | 1.0 |
| Nazism | X | 200.0 | neighbour | 122 | 0.511 | 0.282 | 0.487 | 0.349 | 0.935 | 0.513 | 0.002 | -0.06 | -0.09 | 1.0 |
| Nazism | X_foreign | 100.0 | pick | 115 | 0.467 | 0.333 | 0.436 | 0.349 | 1.0 | 0.529 | 0.062 | -0.06 | -0.08 | 1.0 |
| Nazism | X_foreign | 200.0 | neighbour | 115 | 0.511 | 0.282 | 0.487 | 0.349 | 0.935 | 0.513 | 0.002 | -0.06 | -0.08 | 1.0 |
| Nazism | X_clean | 100.0 | pick | 112 | 0.467 | 0.333 | 0.487 | 0.349 | 1.0 | 0.542 | 0.075 | -0.06 | -0.07 | 1.0 |
| Nazism | X_clean | 200.0 | neighbour | 112 | 0.511 | 0.282 | 0.538 | 0.349 | 0.935 | 0.526 | 0.015 | -0.06 | -0.07 | 1.0 |
| Republic of Ireland | EMBER | 100.0 | pick | 31 | 0.721 | 0.818 | 1.0 | 0.818 | 1.0 | 0.909 | 0.188 | -0.02 | 0.0 | 1.0 |
| Republic of Ireland | EMBER | 200.0 | neighbour | 31 | 0.767 | 0.818 | 1.0 | 0.879 | 1.0 | 0.924 | 0.157 | -0.02 | 0.0 | 1.0 |
| Republic of Ireland | X | 100.0 | pick | 88 | 0.674 | 0.697 | 0.806 | 0.091 | 1.0 | 0.649 | -0.025 | -0.02 | -0.01 | 1.0 |
| Republic of Ireland | X | 200.0 | neighbour | 88 | 0.721 | 0.697 | 0.806 | 0.091 | 1.0 | 0.649 | -0.072 | -0.02 | -0.01 | 1.0 |
| Republic of Ireland | X_foreign | 100.0 | pick | 83 | 0.674 | 0.697 | 1.0 | 0.091 | 1.0 | 0.697 | 0.023 | -0.02 | -0.01 | 1.0 |
| Republic of Ireland | X_foreign | 200.0 | neighbour | 83 | 0.721 | 0.697 | 1.0 | 0.091 | 1.0 | 0.697 | -0.024 | -0.02 | -0.01 | 1.0 |
| Republic of Ireland | X_clean | 100.0 | pick | 83 | 0.674 | 0.697 | 0.806 | 0.091 | 1.0 | 0.649 | -0.025 | -0.02 | -0.01 | 1.0 |
| Republic of Ireland | X_clean | 200.0 | neighbour | 83 | 0.721 | 0.697 | 0.806 | 0.091 | 1.0 | 0.649 | -0.072 | -0.02 | -0.01 | 1.0 |
| Culture of Greece | EMBER | 200.0 | pick | 52 | 0.388 | 0.806 | 0.852 | 1.0 | 1.0 | 0.914 | 0.526 | -0.04 | -0.01 | 1.0 |
| Culture of Greece | EMBER | 100.0 | neighbour | 52 | 0.429 | 0.806 | 0.852 | 1.0 | 1.0 | 0.914 | 0.485 | -0.04 | -0.01 | 1.0 |
| Culture of Greece | X | 200.0 | pick | 151 | 0.388 | 0.355 | 0.852 | 0.548 | 0.826 | 0.645 | 0.257 | -0.08 | -0.025 | 1.0 |
| Culture of Greece | X | 100.0 | neighbour | 151 | 0.429 | 0.355 | 0.926 | 0.548 | 0.913 | 0.685 | 0.256 | -0.08 | -0.025 | 1.0 |
| Culture of Greece | X_foreign | 200.0 | pick | 142 | 0.388 | 0.355 | 0.852 | 0.548 | 0.826 | 0.645 | 0.257 | -0.06 | -0.025 | 1.0 |
| Culture of Greece | X_foreign | 100.0 | neighbour | 142 | 0.429 | 0.355 | 0.926 | 0.548 | 0.913 | 0.685 | 0.256 | -0.06 | -0.025 | 1.0 |
| Culture of Greece | X_clean | 200.0 | pick | 143 | 0.388 | 0.355 | 0.852 | 0.548 | 1.0 | 0.689 | 0.301 | -0.08 | -0.02 | 1.0 |
| Culture of Greece | X_clean | 100.0 | neighbour | 143 | 0.429 | 0.355 | 0.926 | 0.548 | 1.0 | 0.707 | 0.278 | -0.08 | -0.02 | 1.0 |
| Golf | EMBER | 10.0 | pick | 70 | 0.559 | 0.837 | 0.6 | 0.6 | 0.939 | 0.744 | 0.185 | 0.0 | 0.0 | 1.0 |
| Golf | EMBER | 50.0 | neighbour | 70 | 0.729 | 0.837 | 0.689 | 0.689 | 0.939 | 0.788 | 0.059 | -0.02 | -0.005 | 1.0 |
| Golf | X | 10.0 | pick | 206 | 0.559 | 0.755 | 0.6 | 0.644 | 0.939 | 0.735 | 0.176 | -0.02 | -0.02 | 1.0 |
| Golf | X | 50.0 | neighbour | 206 | 0.763 | 0.796 | 0.511 | 0.689 | 0.939 | 0.734 | -0.029 | -0.04 | -0.025 | 0.962 |
| Golf | X_foreign | 10.0 | pick | 195 | 0.559 | 0.714 | 0.6 | 0.644 | 0.939 | 0.724 | 0.165 | -0.02 | -0.015 | 1.0 |
| Golf | X_foreign | 50.0 | neighbour | 195 | 0.729 | 0.837 | 0.511 | 0.689 | 0.939 | 0.744 | 0.015 | -0.04 | -0.025 | 0.962 |
| Golf | X_clean | 10.0 | pick | 196 | 0.559 | 0.755 | 0.6 | 0.644 | 1.0 | 0.75 | 0.191 | -0.02 | -0.02 | 1.0 |
| Golf | X_clean | 50.0 | neighbour | 196 | 0.763 | 0.755 | 0.511 | 0.689 | 1.0 | 0.739 | -0.024 | -0.04 | -0.025 | 0.962 |

