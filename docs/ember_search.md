# EMBER Pass 2 delta search — status by concept

Tracks `judge_rerun/pass2_delta_search.py` progress (judge all 8 candidate
deltas' saved Alpaca completions, pick the best delta) for each of the 18
concepts. Updated 2026-10-05.

| Concept | Status | Person |
|---|---| --- |
| Ancient Rome | 979023 | Mary |
| Artificial intelligence | 979024 | Mary |
| Baseball | 979043 | Mary |
| COVID-19 pandemic | x | |
| Cannabis | x | |
| Culture of Greece | x | |
| Gambling | x | |
| Golf | x | |
| Gun | x | |
| Halloween | x | |
| Harry Potter | Finished | Semyon |
| Heroin | 981330 | Semyon |
| Nazism | 981331 | Semyon |
| Pornography | 981347 | Semyon |
| Republic of Ireland | x | |
| Uranium | x | |
| Valentine's Day | x | |
| World War II | x | |

## How to submit

Submit one concept's full 8-delta grid (checkpoints per delta, so a timeout
or preemption just needs the identical command resubmitted to continue):

```
sbatch judge_rerun/run_pass2_delta_search.sbatch --concept "<Concept Name>"
```

e.g.:

```
sbatch judge_rerun/run_pass2_delta_search.sbatch --concept "World War II"
```
