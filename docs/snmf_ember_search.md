# SNMF+EMBER HP search — status

The paper's combined method (`snmf_ember`, results folder `snmf_ef`), redone
with the Qwen judge (Gemma-2-2b-it, rank 100, seed 42). Started 2026-10-10.

**Status (2026-10-11 ~12:00):** Pass 1 finished for all 18 concepts (every
concept has `top3_hps.json` + 3 cells of 50 answers; the 10 fresh grids are
complete, 144 rows each in `results/snmf_ef/.../train_mc/<Concept>/hps.csv`).
**Pass 2 has not started**: its 4 jobs (1007174–1007177) are queued behind
other jobs on the RTX 2080 nodes (Slurm reason `Priority`), not waiting on
dependencies. Nothing judged yet.

## What it is

The same two-pass search as SNMF ([`snmf_search.md`](snmf_search.md)), with
one difference: each concept's **best EMBER δ** (the Qwen-judged pick in
`judge_rerun/Ember/pass2_outputs/.../best_deltas_summary.csv`) is applied under
every SNMF cell, and the 144-cell SNMF grid is **searched again** on top of it.
This is what `ember/erasure/pipeline.py` does when `ember_step.enabled` is true
(`configs/snmf_ember_gemma.yaml`). The EMBER δ is reused; the SNMF settings are
not (the best SNMF cell can change once the embeddings are edited).

Same rules as SNMF: top 3 cells by the judge-free `harmonic`, 50 Alpaca answers
each, Qwen2.5-32B judge, pick by `harmonic_alpaca`, no Alpaca gate.

## Code

The SNMF scripts take `--with-ember` (2026-10-10):

- `judge_rerun/pass1_snmf_top3.py --with-ember`: loads `configs/snmf_ember_gemma.yaml`,
  runs/reuses grids under `results/snmf_ef/` (archived: `archive_gemini/results/snmf_ef/`),
  writes `judge_rerun/SNMF_EMBER/pass1_outputs/`. Refuses a grid whose rows do not
  all carry exactly the concept's EMBER δ. `top3_hps.json` records `ember_delta_embed`.
- `judge_rerun/pass1_snmf_screen.py --with-ember`: the loop screen, same switch.
- `judge_rerun/pass2_snmf_judge.py --with-ember`: reads/writes `judge_rerun/SNMF_EMBER/`.
  `best_hp` now always includes `delta_embed` (0.0 for plain SNMF).

Tested without a GPU: plain SNMF resolves the same grids and top 3 as before;
`--with-ember` finds the 8 archived grids with matching δ; the δ and config
guards reject mismatches; Pass 2 skips a concept whose Pass 1 is missing, and
resuming Harry Potter's SNMF checkpoint reproduces its pick exactly.

## Grids

8 concepts reuse the Gemini-era combined grid (144 cells, run with exactly the
Qwen-era EMBER δ): Harry Potter (10), World War II (5), Halloween (10),
Uranium (100), Golf (10), Baseball (100), Gambling (100), Culture of Greece (200).
The other 10 run a fresh grid: Ancient Rome (200), Artificial intelligence (100),
COVID-19 pandemic (50), Cannabis (200), Gun (100), Heroin (100), Nazism (100),
Pornography (5), Republic of Ireland (100), Valentine's Day (200).

## Jobs (submitted 2026-10-10 ~23:30)

| Pass | Job | Concepts | Expected | Outcome (2026-10-11) |
|---|---|---|---|---|
| 1 | 1007155 | the 8 archived-grid concepts (answers only, ~10 min each) | ~1.5 h | done, 1h11m |
| 1 | 1007156–1007165 | one fresh-grid concept each (order as listed above) | ~2–2.5 h each; Ireland and Valentine's waited for GPUs | all done, 1.5–2.3 h each (Ireland and Valentine's ran 06:53–07:56, ~1 h) |
| 2 | 1007174 | Harry Potter, World War II, Halloween, Uranium | starts after 1007155 | pending (`Priority`) |
| 2 | 1007175 | Golf, Baseball, Culture of Greece, Gambling | starts after 1007155 | pending (`Priority`) |
| 2 | 1007176 | Ancient Rome, AI, COVID-19, Cannabis, Gun | starts after 1007156–1007160 | pending (`Priority`) |
| 2 | 1007177 | Heroin, Nazism, Pornography, Republic of Ireland, Valentine's Day | starts after 1007161–1007165 | pending (`Priority`) |

Pass 2 jobs use `--dependency=afterany:<Pass 1 jobs>`: Slurm starts them by
itself when those Pass 1 jobs end. A concept whose Pass 1 did not finish is
skipped (no `top3_hps.json`), never judged half-done. ~2.6 h per concept on
3× RTX 2080 Ti, so ~11–14 h per job; at most 4 run at once (12-GPU user cap).
Logs: `pass1_snmf_top3_<job>.out`, `pass2_snmf_<job>.out` in the repo root
(Pass 2 with `--open-mode=append`, so a requeue does not wipe them).

Pass 2 stays pinned to `geforce_rtx_2080` even while the titan nodes sit idle:
the 32B judge runs ~40× slower on titans (`run_pass2_snmf.sbatch` header).

## To check when they finish

1. Every concept has `judge_rerun/SNMF_EMBER/pass2_outputs/.../<Concept>/best_hp__qwen2.5-32b-instruct.json`
   with `complete: true`. A concept skipped by Pass 2 (Pass 1 failed or was
   preempted) needs its Pass 1 resubmitted, then Pass 2 for it.
2. **Loops**: count looping Alpaca answers per cell (rule in `snmf_search.md`).
   The combined edit is stronger than SNMF alone (several archived grids score
   0.98–1.00 on MC), so broken text is more likely. If a concept's pick is
   broken, redo it with `pass1_snmf_screen.py --with-ember` (move its outputs aside first).
3. Build `best_hps_summary.csv` and a `judge_rerun/SNMF_EMBER/README.md` as for SNMF.
4. Add `snmf_ember` to `crosslingual/run_edited.py` and run the cross-lingual test.
