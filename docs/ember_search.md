# EMBER Pass 2 delta search — status by concept

Tracks `judge_rerun/pass2_delta_search.py` progress (judge all 8 candidate
deltas' saved Alpaca completions, pick the best delta) for each of the 18
concepts. Updated 2026-10-05.

| Concept | Status | Person |
|---|---| --- |
| Ancient Rome | Finished | Mary |
| Artificial intelligence | 979024 | Mary |
| Baseball | Finished | Mary |
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
sbatch judge_rerun/run_pass2_delta_search_anygpu.sbatch --concept "<Concept Name>"
```

e.g.:

```
sbatch judge_rerun/run_pass2_delta_search_anygpu.sbatch --concept "World War II"
```

Use `run_pass2_delta_search_anygpu.sbatch`, not `run_pass2_delta_search.sbatch`:
the latter pins `--gres=gpu:geforce_rtx_2080:3`, which only exists on 2 of the
5 `studentkillable` nodes (`s-004`, `s-005`) — if both are busy, the job queues
even when other nodes (titan, on `s-002`/`s-003`/`s-006`) are idle. The `anygpu`
variant requests `--gres=gpu:3` with no model pinned, so it schedules onto
whichever node has 3 free GPUs. Confirmed working end-to-end on titan GPUs
(jobs 981330/981331/981347).
