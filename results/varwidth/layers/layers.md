# Per-layer diagnostics

Secondary / exploratory (PREREG): effect sizes and CIs, no significance claims.

## L1 (CS): concept recoverability, independent of the judge

Oracle hit@k = a P-labeled feature among the top-k by ρ (share of runs, mean over concepts). n_tau = features with ρ ≥ 2 (the paper's candidate count; its scale changes under CW). AUC = C-vs-N separability of the V* columns along the mean-difference direction (prep matrix only).

| scope | config | hit@1 | hit@3 | hit@5 | hit@20 | n_tau | top-ρ share |
|---|---|---|---|---|---|---|---|
| non-prompt (primary) | EMBER matrix, RS-1% | 0.353 | 0.512 | 0.559 | 0.718 | 43.371 | 0.873 |
| non-prompt (primary) | CS, RS-1% | 0.376 | 0.535 | 0.618 | 0.741 | 64.541 | 0.887 |
| original | EMBER matrix, RS-1% | 0.452 | 0.600 | 0.652 | 0.739 | 42.722 | 0.889 |
| original | CS, RS-1% | 0.522 | 0.678 | 0.722 | 0.774 | 62.574 | 0.908 |
| new | EMBER matrix, RS-1% | 0.173 | 0.347 | 0.400 | 0.667 | 44.160 | 0.846 |
| new | CS, RS-1% | 0.120 | 0.307 | 0.440 | 0.680 | 67.000 | 0.864 |
| wide | EMBER matrix, RS-1% | 0.200 | 0.333 | 0.333 | 0.667 | 50.467 | 0.806 |
| wide | CS, RS-1% | 0.200 | 0.400 | 0.533 | 0.733 | 72.200 | 0.839 |
| all | EMBER matrix, RS-1% | 0.342 | 0.500 | 0.553 | 0.711 | 43.289 | 0.872 |
| all | CS, RS-1% | 0.363 | 0.532 | 0.611 | 0.737 | 64.321 | 0.891 |

C-vs-N AUC over 38 concepts: plain 0.690, CW 0.797; CW higher for 38 / 38.

## L2 (GS): fragmentation, coverage, breadth adaptivity, stability

Fragmentation = distinct P groups among the top-20 per run. Best-P = the P candidate with the largest held-out coverage. Width adaptivity = Spearman over concepts between the best-P feature's width (mean over seeds) and breadth = number of concept-only tokens in V*. Stability = mean pairwise Jaccard across seeds of the edited set E of the rung's F_C.

| config | fragmentation | best-P width | best-P coverage | Spearman(width, breadth) | concepts with a P feature |
|---|---|---|---|---|---|
| CS, RS-1% | 0.895 | 47.000 | 0.043 | 0.989 | 32 |
| CS, GS-norm | 0.953 | 57.291 | 0.047 | 0.054 | 31 |

| rung | seed Jaccard of E | concepts |
|---|---|---|
| +CS | 0.672 | 26 |
| +CS+GS | 0.727 | 25 |
| +CS+GS+TWJ (CINDER) | 0.707 | 29 |

### Wide topics (Cloud, Doctor of Philosophy, Tourism): best-P width per config

- L1: Cloud 43 tokens (5 seeds with P), Doctor of Philosophy 45 tokens (1 seeds with P), Tourism 47 tokens (5 seeds with P)
- L12: Cloud 22 tokens (5 seeds with P), Doctor of Philosophy 41 tokens (2 seeds with P), Tourism 52 tokens (5 seeds with P)

## L2 width-rule table at cap 3% (judge-free; 38 concepts)

Concept feature = the top-20 candidate with the most EJ matches (ties: higher ρ), so no Gemini or labels are needed. EJ ≥ 5 = that feature passes EJ@5. Width and neutral share of that feature; seed Jaccard of its edited tokens; coverage on held-out concept text; collateral per 10k general tokens.

| config | concepts | EJ ≥ 5 | width | concept share | seed Jaccard | coverage | collateral (wiki) | Spearman(width, breadth) |
|---|---|---|---|---|---|---|---|---|
| CS, RS-1% | 38 | 0.674 | 46.921 | 0.876 | 0.652 | 0.040 | 3.286 | 0.990 |
| CS, RS-2% | 38 | 0.674 | 93.395 | 0.838 | 0.654 | 0.053 | 6.543 | 0.997 |
| CS, GS-gain | 38 | 0.600 | 77.495 | 0.855 | 0.644 | 0.048 | 5.081 | 0.139 |
| CS, GS-entropy | 38 | 0.621 | 72.384 | 0.862 | 0.629 | 0.047 | 4.879 | 0.115 |
| CS, GS-norm | 38 | 0.626 | 64.042 | 0.851 | 0.675 | 0.045 | 4.043 | 0.052 |
| no CS, RS-1% (EMBER) | 38 | 0.553 | 46.921 | 0.839 | 0.533 | 0.037 | 2.640 | 0.990 |
| no CS, GS-norm | 38 | 0.526 | 43.211 | 0.856 | 0.567 | 0.035 | 2.501 | 0.085 |

## Cross-ablations

EJ@5 / TWJ applied to the fixed-width rungs (L3 without L2); columns TJ, LO = EJ@5, HJ = TWJ, strict clean-hit rate on the non-prompt concepts (mean over concepts):

| config | TJ | LO | HJ |
|---|---|---|---|
| no CS, RS-1% | 0.459 | 0.400 | 0.471 |
| CS, RS-1% | 0.435 | 0.400 | 0.418 |
| CS, GS-norm | 0.412 | 0.400 | 0.400 |

## L3 judges on the CS + GS-norm runs (top-20 candidates)

Micro precision / recall over candidates (strict: P positive; lenient: P or B), runs hit = runs with an accepted P feature, macro recall = mean over concepts. Recall is over the labeled pool (TJ, CJ, LO ≥ 2 or top 3 by ρ), so it is an upper bound.

TJ noise (re-judged 230 candidates): accept-decision flip rate 0.004, is_member flip rate 0.004.

### non-prompt (primary) (34 concepts)

Runs: 170; runs with a P candidate: 125.

| rule | accepts | P strict | R strict | runs hit | macro R | P len | R len |
|---|---|---|---|---|---|---|---|
| TJ (EMBER) | 136 | 0.779 | 0.635 | 92 | 0.647 | 0.993 | 0.265 |
| WJ | 199 | 0.704 | 0.838 | 107 | 0.839 | 0.950 | 0.371 |
| WJ+share | 184 | 0.755 | 0.832 | 107 | 0.836 | 0.967 | 0.349 |
| EJ@3 | 240 | 0.492 | 0.707 | 102 | 0.732 | 0.946 | 0.445 |
| EJ@4 | 180 | 0.567 | 0.611 | 98 | 0.662 | 0.983 | 0.347 |
| EJ@5 | 126 | 0.651 | 0.491 | 78 | 0.505 | 0.992 | 0.245 |
| TWJ = TJ ∨ EJ@5 | 183 | 0.634 | 0.695 | 101 | 0.714 | 0.989 | 0.355 |
| TJ ∨ EJ@4 | 218 | 0.560 | 0.731 | 107 | 0.760 | 0.982 | 0.420 |
| TJ ∨ WJ+share | 200 | 0.715 | 0.856 | 111 | 0.864 | 0.965 | 0.378 |
| TWJ-R = WJ+share ∨ EJ@4 | 263 | 0.578 | 0.910 | 119 | 0.926 | 0.966 | 0.498 |

### original (19 concepts)

Runs: 95; runs with a P candidate: 71.

| rule | accepts | P strict | R strict | runs hit | macro R | P len | R len |
|---|---|---|---|---|---|---|---|
| TJ (EMBER) | 81 | 0.852 | 0.711 | 59 | 0.697 | 1.000 | 0.318 |
| WJ | 120 | 0.658 | 0.814 | 59 | 0.814 | 0.925 | 0.435 |
| WJ+share | 107 | 0.729 | 0.804 | 59 | 0.809 | 0.953 | 0.400 |
| EJ@3 | 127 | 0.638 | 0.835 | 68 | 0.843 | 0.906 | 0.451 |
| EJ@4 | 97 | 0.701 | 0.701 | 67 | 0.759 | 0.969 | 0.369 |
| EJ@5 | 87 | 0.736 | 0.660 | 63 | 0.709 | 0.989 | 0.337 |
| TWJ = TJ ∨ EJ@5 | 109 | 0.706 | 0.794 | 66 | 0.801 | 0.991 | 0.424 |
| TJ ∨ EJ@4 | 116 | 0.681 | 0.814 | 68 | 0.826 | 0.974 | 0.443 |
| TJ ∨ WJ+share | 113 | 0.717 | 0.835 | 62 | 0.847 | 0.956 | 0.424 |
| TWJ-R = WJ+share ∨ EJ@4 | 143 | 0.629 | 0.928 | 70 | 0.956 | 0.944 | 0.529 |

### new (15 concepts)

Runs: 75; runs with a P candidate: 54.

| rule | accepts | P strict | R strict | runs hit | macro R | P len | R len |
|---|---|---|---|---|---|---|---|
| TJ (EMBER) | 55 | 0.673 | 0.529 | 33 | 0.581 | 0.982 | 0.212 |
| WJ | 79 | 0.772 | 0.871 | 48 | 0.871 | 0.987 | 0.306 |
| WJ+share | 77 | 0.792 | 0.871 | 48 | 0.871 | 0.987 | 0.298 |
| EJ@3 | 113 | 0.327 | 0.529 | 34 | 0.583 | 0.991 | 0.439 |
| EJ@4 | 83 | 0.410 | 0.486 | 31 | 0.533 | 1.000 | 0.325 |
| EJ@5 | 39 | 0.462 | 0.257 | 15 | 0.233 | 1.000 | 0.153 |
| TWJ = TJ ∨ EJ@5 | 74 | 0.527 | 0.557 | 35 | 0.597 | 0.986 | 0.286 |
| TJ ∨ EJ@4 | 102 | 0.422 | 0.614 | 39 | 0.672 | 0.990 | 0.396 |
| TJ ∨ WJ+share | 87 | 0.713 | 0.886 | 49 | 0.887 | 0.977 | 0.333 |
| TWJ-R = WJ+share ∨ EJ@4 | 120 | 0.517 | 0.886 | 49 | 0.887 | 0.992 | 0.467 |
