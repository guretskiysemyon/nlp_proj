# Fair comparison: EMBER vs CINDER (option B)

Same model, data, seed 42, judge model (gemini-3.5-flash-lite), grid, pick rule (fast harmonic, MMLU gate, no Alpaca) and tests; only the method differs. ember = EMBER (paper's factorization and judge), ember_hj = EMBER's factorization + Ori's judge (top-20 by rho, TJ or LO@5), cinder = Ori's L123. ember_hj rows with reused_from = ember have exactly ember's edit, so ember's results. ember_x / cinder_x = ember / cinder + extrapolation to native de/es/fr/ru tokens (same features, the parent's edit kept, at the parent's δ; foreign native text chooses the added tokens). ret_L = retention of QA_test in L (1 = nothing erased); leak = mean(ret de/es/fr/ru) - ret_en; simdom_L = SimdomQA_test accuracy change vs the pristine model. No feature kept → no edit → the pristine model's numbers.

## Per concept

| concept | method | n_features | n_tokens_edited | delta | train_harmonic | ret_en | ret_de | ret_es | ret_fr | ret_ru | leak | simdom_en | mmlu_test | edit_check | reused_from |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ancient Rome | cinder | 1 | 92 | 5.0 | 0.706 | 0.818 | 0.939 | 0.814 | 0.744 | 1.0 | 0.056 | -0.04 | 0.549 | True |  |
| Ancient Rome | cinder_x | 1 | 200 | 5.0 | 0.786 | 0.782 | 0.879 | 0.674 | 0.692 | 0.826 | -0.014 | -0.04 | 0.549 | True |  |
| Ancient Rome | ember | 2 | 66 | 10.0 | 0.569 | 0.782 | 0.879 | 0.953 | 0.949 | 1.0 | 0.163 | -0.06 | 0.551 | True |  |
| Ancient Rome | ember_hj | 2 | 66 | 10.0 | 0.569 | 0.782 | 0.879 | 0.953 | 0.949 | 1.0 | 0.163 | -0.06 | 0.551 | True | ember |
| Ancient Rome | ember_x | 2 | 191 | 10.0 | 0.533 | 0.818 | 0.939 | 0.814 | 0.795 | 1.0 | 0.069 | -0.06 | 0.551 | True |  |
| Artificial intelligence | cinder | 1 | 31 | 2.0 | 0.131 | 0.891 | 0.962 | 0.882 | 0.962 | 1.0 | 0.06 | 0.0 | 0.551 | True |  |
| Artificial intelligence | cinder_x | 1 | 109 | 2.0 | 0.131 | 0.891 | 0.925 | 0.843 | 0.887 | 1.0 | 0.023 | 0.0 | 0.551 | True |  |
| Artificial intelligence | ember | 1 | 39 | 50.0 | 0.19 | 0.782 | 1.0 | 0.922 | 1.0 | 1.0 | 0.199 | 0.0 | 0.547 | True |  |
| Artificial intelligence | ember_hj | 1 | 39 | 50.0 | 0.19 | 0.782 | 1.0 | 0.922 | 1.0 | 1.0 | 0.199 | 0.0 | 0.547 | True | ember |
| Artificial intelligence | ember_x | 1 | 149 | 50.0 | 0.245 | 0.709 | 0.849 | 0.765 | 0.736 | 0.86 | 0.093 | 0.0 | 0.546 | True |  |
| Baseball | cinder | 3 | 61 | 50.0 | 0.38 | 0.61 | 0.784 | 1.0 | 0.677 | 1.0 | 0.255 | -0.06 | 0.552 | True |  |
| Baseball | cinder_x | 3 | 158 | 50.0 | 0.422 | 0.561 | 0.568 | 0.657 | 0.613 | 1.0 | 0.148 | -0.08 | 0.552 | True |  |
| Baseball | ember | 1 | 39 | 200.0 | 0.335 | 0.707 | 0.838 | 1.0 | 0.677 | 1.0 | 0.172 | -0.02 | 0.55 | True |  |
| Baseball | ember_hj | 1 | 39 | 200.0 | 0.335 | 0.707 | 0.838 | 1.0 | 0.677 | 1.0 | 0.172 | -0.02 | 0.55 | True | ember |
| Baseball | ember_x | 1 | 93 | 200.0 | 0.333 | 0.659 | 0.568 | 0.829 | 0.742 | 1.0 | 0.126 | -0.06 | 0.55 | True |  |
| COVID-19 pandemic | cinder | 1 | 63 | 2.0 | 0.596 | 0.897 | 0.931 | 0.862 | 1.0 | 0.684 | -0.028 | -0.04 | 0.549 | True |  |
| COVID-19 pandemic | cinder_x | 1 | 158 | 2.0 | 0.593 | 0.897 | 1.0 | 0.862 | 1.0 | 0.684 | -0.01 | -0.04 | 0.549 | True |  |
| COVID-19 pandemic | ember | 1 | 42 | 50.0 | 0.531 | 0.692 | 0.379 | 0.724 | 0.778 | 0.474 | -0.103 | -0.1 | 0.546 | True |  |
| COVID-19 pandemic | ember_hj | 1 | 42 | 50.0 | 0.531 | 0.692 | 0.379 | 0.724 | 0.778 | 0.474 | -0.103 | -0.1 | 0.546 | True | ember |
| COVID-19 pandemic | ember_x | 1 | 138 | 50.0 | 0.523 | 0.692 | 0.31 | 0.793 | 0.704 | 0.474 | -0.122 | -0.1 | 0.544 | True |  |
| Cannabis | cinder | 1 | 34 | 5.0 | 0.338 | 0.594 | 0.651 | 0.662 | 0.673 | 0.854 | 0.116 | 0.0 | 0.55 | True |  |
| Cannabis | cinder_x | 1 | 70 | 5.0 | 0.384 | 0.565 | 0.619 | 0.692 | 0.673 | 0.805 | 0.132 | -0.02 | 0.55 | True |  |
| Cannabis | ember | 1 | 45 | 200.0 | 0.468 | 0.391 | 0.397 | 0.446 | 0.382 | 0.902 | 0.141 | -0.02 | 0.55 | True |  |
| Cannabis | ember_hj | 1 | 45 | 200.0 | 0.468 | 0.391 | 0.397 | 0.446 | 0.382 | 0.902 | 0.141 | -0.02 | 0.55 | True | ember |
| Cannabis | ember_x | 1 | 120 | 200.0 | 0.578 | 0.333 | 0.302 | 0.446 | 0.418 | 0.463 | 0.074 | -0.08 | 0.55 | True |  |
| Cannabis | ml_sep | 4 | 109 | 100.0 | 0.64 | 0.304 | 0.302 | 0.354 | 0.309 | 0.902 | 0.163 | -0.08 | 0.549 | True |  |
| Culture of Greece | cinder | 1 | 106 | 100.0 | 0.755 | 0.459 | 0.63 | 0.892 | 0.793 | 1.0 | 0.37 | -0.06 | 0.549 | True |  |
| Culture of Greece | cinder_x | 1 | 231 | 100.0 | 0.78 | 0.459 | 0.407 | 0.676 | 0.448 | 0.852 | 0.137 | -0.06 | 0.547 | True |  |
| Culture of Greece | ember | 1 | 52 | 200.0 | 0.91 | 0.73 | 0.63 | 0.946 | 0.862 | 1.0 | 0.13 | -0.02 | 0.545 | True |  |
| Culture of Greece | ember_hj | 1 | 52 | 200.0 | 0.91 | 0.73 | 0.63 | 0.946 | 0.862 | 1.0 | 0.13 | -0.02 | 0.545 | True | ember |
| Culture of Greece | ember_x | 1 | 151 | 200.0 | 0.903 | 0.73 | 0.556 | 0.459 | 0.724 | 0.926 | -0.064 | -0.02 | 0.545 | True |  |
| Gambling | cinder | 2 | 56 | 2.0 | 0.64 | 0.492 | 1.0 | 0.843 | 0.822 | 1.0 | 0.424 | 0.0 | 0.552 | True |  |
| Gambling | cinder_x | 2 | 139 | 2.0 | 0.675 | 0.424 | 1.0 | 0.804 | 0.822 | 1.0 | 0.482 | 0.0 | 0.552 | True |  |
| Gambling | ember | 1 | 41 | 10.0 | 0.64 | 0.661 | 0.964 | 0.804 | 0.778 | 1.0 | 0.226 | 0.0 | 0.549 | True |  |
| Gambling | ember_hj | 1 | 41 | 10.0 | 0.64 | 0.661 | 0.964 | 0.804 | 0.778 | 1.0 | 0.226 | 0.0 | 0.549 | True | ember |
| Gambling | ember_x | 1 | 115 | 10.0 | 0.771 | 0.593 | 0.964 | 0.608 | 0.733 | 1.0 | 0.233 | 0.0 | 0.55 | True |  |
| Golf | cinder | 1 | 44 | 10.0 | 0.427 | 0.891 | 0.882 | 0.872 | 0.702 | 1.0 | -0.027 | -0.06 | 0.552 | True |  |
| Golf | cinder_x | 1 | 108 | 10.0 | 0.427 | 0.818 | 0.804 | 0.872 | 0.702 | 1.0 | 0.027 | -0.06 | 0.551 | True |  |
| Golf | ember | 2 | 70 | 10.0 | 0.578 | 0.745 | 0.843 | 0.745 | 0.617 | 1.0 | 0.056 | 0.0 | 0.549 | True |  |
| Golf | ember_hj | 2 | 70 | 10.0 | 0.578 | 0.745 | 0.843 | 0.745 | 0.617 | 1.0 | 0.056 | 0.0 | 0.549 | True | ember |
| Golf | ember_x | 2 | 206 | 10.0 | 0.608 | 0.636 | 0.647 | 0.745 | 0.532 | 1.0 | 0.095 | 0.0 | 0.549 | True |  |
| Gun | cinder | 2 | 67 | 100.0 | 0.543 | 0.684 | 0.951 | 0.949 | 0.892 | 1.0 | 0.264 | -0.06 | 0.552 | True |  |
| Gun | cinder_x | 2 | 166 | 100.0 | 0.639 | 0.649 | 0.902 | 0.897 | 0.73 | 0.805 | 0.184 | -0.08 | 0.552 | True |  |
| Gun | ember | 2 | 61 | 5.0 | 0.567 | 0.544 | 1.0 | 0.949 | 0.892 | 1.0 | 0.416 | -0.1 | 0.553 | True |  |
| Gun | ember_hj | 2 | 61 | 5.0 | 0.567 | 0.544 | 1.0 | 0.949 | 0.892 | 1.0 | 0.416 | -0.1 | 0.553 | True | ember |
| Gun | ember_x | 2 | 191 | 5.0 | 0.659 | 0.544 | 0.902 | 0.846 | 0.892 | 0.805 | 0.317 | -0.1 | 0.553 | True |  |
| Gun | ml_sep | 5 | 157 | 5.0 | 0.659 | 0.579 | 0.902 | 1.0 | 0.784 | 1.0 | 0.343 | -0.1 | 0.553 | True |  |
| Halloween | cinder | 1 | 76 | 10.0 | 0.625 | 0.543 | 0.652 | 0.826 | 0.739 | 1.0 | 0.261 | -0.04 | 0.551 | True |  |
| Halloween | cinder_x | 1 | 178 | 10.0 | 0.615 | 0.543 | 0.652 | 0.652 | 0.913 | 1.0 | 0.261 | -0.06 | 0.551 | True |  |
| Halloween | ember | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Halloween | ember_hj | 1 | 41 | 5.0 | 0.511 | 0.543 | 0.652 | 0.652 | 1.0 | 1.0 | 0.283 | 0.0 | 0.549 | True |  |
| Halloween | ember_x | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Harry Potter | cinder | 1 | 51 | 2.0 | 0.719 | 0.782 | 0.886 | 0.636 | 0.68 | 1.0 | 0.018 | 0.0 | 0.551 | True |  |
| Harry Potter | cinder_x | 1 | 105 | 2.0 | 0.719 | 0.782 | 0.886 | 0.636 | 0.6 | 1.0 | -0.002 | 0.0 | 0.551 | True |  |
| Harry Potter | ember | 1 | 39 | 10.0 | 0.69 | 0.673 | 0.771 | 0.515 | 0.6 | 1.0 | 0.048 | -0.04 | 0.551 | True |  |
| Harry Potter | ember_hj | 1 | 39 | 10.0 | 0.69 | 0.673 | 0.771 | 0.515 | 0.6 | 1.0 | 0.048 | -0.04 | 0.551 | True | ember |
| Harry Potter | ember_x | 1 | 100 | 10.0 | 0.69 | 0.673 | 0.714 | 0.455 | 0.44 | 1.0 | -0.021 | -0.06 | 0.551 | True |  |
| Heroin | cinder | 1 | 86 | 50.0 | 0.528 | 0.628 | 1.0 | 1.0 | 0.714 | 1.0 | 0.3 | 0.0 | 0.55 | True |  |
| Heroin | cinder_x | 1 | 187 | 50.0 | 0.582 | 0.628 | 0.926 | 1.0 | 0.619 | 0.84 | 0.218 | 0.0 | 0.549 | True |  |
| Heroin | ember | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Heroin | ember_hj | 1 | 46 | 100.0 | 0.721 | 0.581 | 1.0 | 1.0 | 0.714 | 1.0 | 0.348 | -0.02 | 0.549 | True |  |
| Heroin | ember_x | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Nazism | cinder | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Nazism | cinder_x | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Nazism | ember | 1 | 41 | 100.0 | 0.693 | 0.633 | 0.6 | 0.787 | 0.707 | 1.0 | 0.14 | -0.04 | 0.549 | True |  |
| Nazism | ember_hj | 1 | 41 | 100.0 | 0.693 | 0.633 | 0.6 | 0.787 | 0.707 | 1.0 | 0.14 | -0.04 | 0.549 | True | ember |
| Nazism | ember_x | 1 | 122 | 100.0 | 0.693 | 0.714 | 0.556 | 0.617 | 0.756 | 1.0 | 0.018 | -0.06 | 0.548 | True |  |
| Pornography | cinder | 1 | 69 | 5.0 | 0.62 | 0.719 | 0.869 | 0.882 | 0.922 | 0.956 | 0.188 | 0.0 | 0.552 | True |  |
| Pornography | cinder_x | 1 | 167 | 5.0 | 0.62 | 0.754 | 0.836 | 0.804 | 0.843 | 0.867 | 0.084 | 0.0 | 0.552 | True |  |
| Pornography | ember | 1 | 43 | 5.0 | 0.492 | 0.684 | 0.77 | 0.765 | 0.804 | 0.956 | 0.14 | 0.0 | 0.551 | True |  |
| Pornography | ember_hj | 1 | 43 | 5.0 | 0.492 | 0.684 | 0.77 | 0.765 | 0.804 | 0.956 | 0.14 | 0.0 | 0.551 | True | ember |
| Pornography | ember_x | 1 | 143 | 5.0 | 0.492 | 0.719 | 0.77 | 0.686 | 0.725 | 0.956 | 0.065 | 0.0 | 0.55 | True |  |
| Republic of Ireland | cinder | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Republic of Ireland | cinder_x | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Republic of Ireland | ember | 1 | 31 | 100.0 | 0.489 | 0.673 | 0.854 | 0.862 | 0.879 | 1.0 | 0.226 | -0.04 | 0.551 | True |  |
| Republic of Ireland | ember_hj | 1 | 31 | 100.0 | 0.489 | 0.673 | 0.854 | 0.862 | 0.879 | 1.0 | 0.226 | -0.04 | 0.551 | True | ember |
| Republic of Ireland | ember_x | 1 | 88 | 100.0 | 0.487 | 0.673 | 0.659 | 0.724 | 0.576 | 1.0 | 0.067 | -0.06 | 0.55 | True |  |
| Uranium | cinder | 1 | 52 | 200.0 | 0.528 | 0.622 | 0.677 | 0.76 | 0.926 | 1.0 | 0.219 | 0.02 | 0.548 | True |  |
| Uranium | cinder_x | 1 | 120 | 200.0 | 0.582 | 0.676 | 0.806 | 0.76 | 0.852 | 1.0 | 0.178 | 0.02 | 0.548 | True |  |
| Uranium | ember | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Uranium | ember_hj | 1 | 38 | 10.0 | 0.582 | 0.622 | 0.355 | 0.76 | 1.0 | 1.0 | 0.157 | -0.02 | 0.55 | True |  |
| Uranium | ember_x | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Valentine's Day | cinder | 1 | 52 | 50.0 | 0.391 | 0.714 | 1.0 | 0.789 | 1.0 | 1.0 | 0.233 | -0.02 | 0.55 | True |  |
| Valentine's Day | cinder_x | 1 | 146 | 50.0 | 0.391 | 0.714 | 1.0 | 0.579 | 1.0 | 0.0 | -0.069 | -0.02 | 0.549 | True |  |
| Valentine's Day | ember | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| Valentine's Day | ember_hj | 1 | 38 | 2.0 | 0.453 | 0.81 | 1.0 | 0.789 | 0.882 | 1.0 | 0.108 | 0.0 | 0.551 | True |  |
| Valentine's Day | ember_x | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| World War II | cinder | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| World War II | cinder_x | 0 | 0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.553 | None |  |
| World War II | ember | 1 | 36 | 5.0 | 0.564 | 0.878 | 0.846 | 0.892 | 0.892 | 1.0 | 0.029 | 0.0 | 0.55 | True |  |
| World War II | ember_hj | 1 | 36 | 5.0 | 0.564 | 0.878 | 0.846 | 0.892 | 0.892 | 1.0 | 0.029 | 0.0 | 0.55 | True | ember |
| World War II | ember_x | 1 | 101 | 5.0 | 0.564 | 0.878 | 0.692 | 0.784 | 1.0 | 1.0 | -0.009 | 0.0 | 0.551 | True |  |
| World War II | ml_sep | 1 | 36 | 5.0 | 0.564 | 0.878 | 0.846 | 0.892 | 0.892 | 1.0 | 0.029 | 0.0 | 0.55 | True |  |

## Means, all 18 concepts

| method | concepts | no_edit | ret_en | ret_de | ret_es | ret_fr | ret_ru | leak | simdom_en | mmlu_test | train_harmonic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cinder | 18 | 3 | 0.741 | 0.879 | 0.87 | 0.847 | 0.972 | 0.15 | -0.02 | 0.551 | 0.44 |
| cinder_x | 18 | 3 | 0.73 | 0.845 | 0.8 | 0.8 | 0.871 | 0.099 | -0.024 | 0.551 | 0.464 |
| ember | 18 | 4 | 0.754 | 0.821 | 0.851 | 0.823 | 0.963 | 0.11 | -0.024 | 0.55 | 0.429 |
| ember_hj | 18 | 0 | 0.674 | 0.765 | 0.806 | 0.801 | 0.963 | 0.16 | -0.027 | 0.55 | 0.555 |
| ember_x | 18 | 4 | 0.743 | 0.746 | 0.754 | 0.765 | 0.916 | 0.052 | -0.033 | 0.55 | 0.449 |
| ml_sep | 3 | 0 | 0.587 | 0.683 | 0.749 | 0.662 | 0.967 | 0.178 | -0.06 | 0.551 | 0.621 |

Paired over concepts (all 18 concepts), cinder − ember:

- ret_en: mean -0.013, cinder lower in 7, higher in 11, equal in 0 of 18 concepts (Wilcoxon p = 0.913)
- leak: mean +0.040, cinder lower in 9, higher in 9, equal in 0 of 18 concepts (Wilcoxon p = 0.369)
- train_harmonic: mean +0.012, cinder lower in 8, higher in 9, equal in 1 of 18 concepts (Wilcoxon p = 0.943)

Paired over concepts (all 18 concepts), ember_hj − ember:

- ret_en: mean -0.080, ember_hj lower in 4, higher in 0, equal in 14 of 18 concepts
- leak: mean +0.050, ember_hj lower in 0, higher in 4, equal in 14 of 18 concepts
- train_harmonic: mean +0.126, ember_hj lower in 0, higher in 4, equal in 14 of 18 concepts

Paired over concepts (all 18 concepts), cinder − ember_hj:

- ret_en: mean +0.067, cinder lower in 4, higher in 12, equal in 2 of 18 concepts (Wilcoxon p = 0.079)
- leak: mean -0.009, cinder lower in 11, higher in 7, equal in 0 of 18 concepts (Wilcoxon p = 0.695)
- train_harmonic: mean -0.114, cinder lower in 11, higher in 6, equal in 1 of 18 concepts (Wilcoxon p = 0.089)

Paired over concepts (all 18 concepts), ember_x − ember:

- ret_en: mean -0.011, ember_x lower in 5, higher in 3, equal in 10 of 18 concepts (Wilcoxon p = 0.312)
- leak: mean -0.058, ember_x lower in 12, higher in 2, equal in 4 of 18 concepts (Wilcoxon p = 0.001)
- train_harmonic: mean +0.020, ember_x lower in 5, higher in 5, equal in 8 of 18 concepts (Wilcoxon p = 0.264)

Paired over concepts (all 18 concepts), cinder_x − cinder:

- ret_en: mean -0.011, cinder_x lower in 6, higher in 2, equal in 10 of 18 concepts (Wilcoxon p = 0.211)
- leak: mean -0.052, cinder_x lower in 10, higher in 4, equal in 4 of 18 concepts (Wilcoxon p = 0.020)
- train_harmonic: mean +0.023, cinder_x lower in 2, higher in 8, equal in 8 of 18 concepts (Wilcoxon p = 0.010)

Paired over concepts (all 18 concepts), cinder_x − ember_x:

- ret_en: mean -0.013, cinder_x lower in 8, higher in 10, equal in 0 of 18 concepts (Wilcoxon p = 0.931)
- leak: mean +0.047, cinder_x lower in 7, higher in 11, equal in 0 of 18 concepts (Wilcoxon p = 0.267)
- train_harmonic: mean +0.015, cinder_x lower in 9, higher in 9, equal in 0 of 18 concepts (Wilcoxon p = 0.913)

Paired over concepts (all 18 concepts), ml_sep − ember:

- ret_en: mean -0.017, ml_sep lower in 1, higher in 1, equal in 1 of 3 concepts
- leak: mean -0.017, ml_sep lower in 1, higher in 1, equal in 1 of 3 concepts
- train_harmonic: mean +0.088, ml_sep lower in 0, higher in 2, equal in 1 of 3 concepts

Paired over concepts (all 18 concepts), ml_sep − ember_x:

- ret_en: mean +0.002, ml_sep lower in 1, higher in 1, equal in 1 of 3 concepts
- leak: mean +0.051, ml_sep lower in 0, higher in 3, equal in 0 of 3 concepts
- train_harmonic: mean +0.021, ml_sep lower in 0, higher in 1, equal in 2 of 3 concepts

## Means, clean 12 (no CINDER tuning / lexicon-prompt concepts)

| method | concepts | no_edit | ret_en | ret_de | ret_es | ret_fr | ret_ru | leak | simdom_en | mmlu_test | train_harmonic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cinder | 12 | 2 | 0.723 | 0.866 | 0.888 | 0.839 | 0.958 | 0.164 | -0.02 | 0.551 | 0.435 |
| cinder_x | 12 | 2 | 0.71 | 0.843 | 0.833 | 0.814 | 0.917 | 0.142 | -0.027 | 0.551 | 0.462 |
| ember | 12 | 3 | 0.728 | 0.804 | 0.84 | 0.793 | 0.944 | 0.118 | -0.027 | 0.551 | 0.399 |
| ember_hj | 12 | 0 | 0.623 | 0.721 | 0.791 | 0.769 | 0.944 | 0.184 | -0.03 | 0.55 | 0.551 |
| ember_x | 12 | 3 | 0.714 | 0.723 | 0.775 | 0.756 | 0.892 | 0.073 | -0.038 | 0.55 | 0.429 |
| ml_sep | 2 | 0 | 0.442 | 0.602 | 0.677 | 0.546 | 0.951 | 0.253 | -0.09 | 0.551 | 0.65 |

Paired over concepts (clean 12 (no CINDER tuning / lexicon-prompt concepts)), cinder − ember:

- ret_en: mean -0.004, cinder lower in 5, higher in 7, equal in 0 of 12 concepts (Wilcoxon p = 0.970)
- leak: mean +0.046, cinder lower in 5, higher in 7, equal in 0 of 12 concepts (Wilcoxon p = 0.424)
- train_harmonic: mean +0.036, cinder lower in 5, higher in 6, equal in 1 of 12 concepts (Wilcoxon p = 0.812)

Paired over concepts (clean 12 (no CINDER tuning / lexicon-prompt concepts)), ember_hj − ember:

- ret_en: mean -0.104, ember_hj lower in 3, higher in 0, equal in 9 of 12 concepts
- leak: mean +0.066, ember_hj lower in 0, higher in 3, equal in 9 of 12 concepts
- train_harmonic: mean +0.151, ember_hj lower in 0, higher in 3, equal in 9 of 12 concepts

Paired over concepts (clean 12 (no CINDER tuning / lexicon-prompt concepts)), cinder − ember_hj:

- ret_en: mean +0.100, cinder lower in 2, higher in 8, equal in 2 of 12 concepts (Wilcoxon p = 0.064)
- leak: mean -0.019, cinder lower in 7, higher in 5, equal in 0 of 12 concepts (Wilcoxon p = 0.691)
- train_harmonic: mean -0.115, cinder lower in 7, higher in 4, equal in 1 of 12 concepts (Wilcoxon p = 0.175)

Paired over concepts (clean 12 (no CINDER tuning / lexicon-prompt concepts)), ember_x − ember:

- ret_en: mean -0.014, ember_x lower in 4, higher in 2, equal in 6 of 12 concepts (Wilcoxon p = 0.438)
- leak: mean -0.045, ember_x lower in 7, higher in 2, equal in 3 of 12 concepts (Wilcoxon p = 0.027)
- train_harmonic: mean +0.029, ember_x lower in 3, higher in 4, equal in 5 of 12 concepts (Wilcoxon p = 0.203)

Paired over concepts (clean 12 (no CINDER tuning / lexicon-prompt concepts)), cinder_x − cinder:

- ret_en: mean -0.014, cinder_x lower in 5, higher in 2, equal in 5 of 12 concepts (Wilcoxon p = 0.328)
- leak: mean -0.022, cinder_x lower in 5, higher in 4, equal in 3 of 12 concepts (Wilcoxon p = 0.250)
- train_harmonic: mean +0.026, cinder_x lower in 2, higher in 6, equal in 4 of 12 concepts (Wilcoxon p = 0.039)

Paired over concepts (clean 12 (no CINDER tuning / lexicon-prompt concepts)), cinder_x − ember_x:

- ret_en: mean -0.004, cinder_x lower in 5, higher in 7, equal in 0 of 12 concepts (Wilcoxon p = 0.970)
- leak: mean +0.069, cinder_x lower in 4, higher in 8, equal in 0 of 12 concepts (Wilcoxon p = 0.151)
- train_harmonic: mean +0.033, cinder_x lower in 6, higher in 6, equal in 0 of 12 concepts (Wilcoxon p = 0.952)

Paired over concepts (clean 12 (no CINDER tuning / lexicon-prompt concepts)), ml_sep − ember:

- ret_en: mean -0.026, ml_sep lower in 1, higher in 1, equal in 0 of 2 concepts
- leak: mean -0.025, ml_sep lower in 1, higher in 1, equal in 0 of 2 concepts
- train_harmonic: mean +0.132, ml_sep lower in 0, higher in 2, equal in 0 of 2 concepts

Paired over concepts (clean 12 (no CINDER tuning / lexicon-prompt concepts)), ml_sep − ember_x:

- ret_en: mean +0.003, ml_sep lower in 1, higher in 1, equal in 0 of 2 concepts
- leak: mean +0.058, ml_sep lower in 0, higher in 2, equal in 0 of 2 concepts
- train_harmonic: mean +0.031, ml_sep lower in 0, higher in 1, equal in 1 of 2 concepts

Lower ret_en = more English erasure; lower leak = less leak; higher train_harmonic = better search score.

