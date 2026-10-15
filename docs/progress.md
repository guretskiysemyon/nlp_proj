# Progress log

**Last updated:** 2026-10-15
**Status:**
- **MC scoring repair (2026-10-15): EMBER and SNMF done; SNMF+EMBER and the
  cross-lingual reruns still to do.** The multiple-choice scoring (the paper's
  `ember/evals/mc.py` / `mmlu.py`) generated only 4 tokens (MMLU 6). Under the
  EMBER embedding edit Gemma answers "The answer is **C**", the cut ends before
  the letter, and the answer counted as **wrong**. That inflated erasure for
  EMBER (10% invalid at the picks) and most of all for SNMF+EMBER (15%). Also
  found: lowercase letters rejected, and a response-extraction bug when an
  option ends with "model". Fix: `judge_rerun/mc_fixed.py` (32 tokens, fixed
  parser, new tokens only; `ember/` untouched). Full write-up:
  [`docs/Part 1/ember_search.md`](Part%201/ember_search.md), "MC scoring repair".
  **Still provisional:** the SNMF+EMBER picks and the cross-lingual `ember` /
  `snmf_ember` numbers.
- **EMBER search: final after the repair (2026-10-15).** Best δ for all 18
  concepts, judged with Qwen2.5-32B-Instruct (picked by the agreement study,
  §3.1), top-3 × 50-question policy, fixed MC scoring:
  `judge_rerun/Ember/pass2_rescored/google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv`.
  4 picks changed (Baseball 100 → 200, Gambling 100 → 10, Gun 100 → 5,
  Uranium 100 → 200); mean efficacy 0.520 → 0.430, mean `harmonic_alpaca`
  0.735 → 0.667. The pre-repair picks stay in `judge_rerun/Ember/pass2_outputs/`
  (layout: `judge_rerun/Ember/README.md`).
- **SNMF search: finished (2026-10-10), checked after the repair
  (2026-10-15): all 18 picks unchanged.** Best HP cell for all 18 concepts,
  same judge, grid checked complete (144 cells) for every concept. Ancient
  Rome's first pick broke generation and was redone with a loop screen.
  Results: `judge_rerun/SNMF/pass2_outputs/`; after the repair
  `judge_rerun/SNMF/pass2_rescored/` (same cells; Pornography's score 0.880 →
  0.864). Layout: `judge_rerun/SNMF/README.md`; repair check:
  [`snmf_search.md`](Part%201/snmf_search.md).
- **Cross-lingual leakage, round 2: all runs done (2026-10-14), analysis
  open.** "Erase in English, ask in de/es/fr/ru", with the Qwen-era picks,
  judge-free MC (`crosslingual/`). **`pristine`, `ember`, `snmf` and
  `snmf_ember` are done** for all 18 concepts × 5 languages (jobs 1006226,
  1007121, 1007145, 1007146, 1040077; every edit check reproduced the HP
  search exactly). First look: `ember` leaks (mean non-English Δ +0.24,
  Russian +0.41), `snmf` does not (+0.01, 95% CI [−0.07, +0.09]) but loses
  more similar-domain accuracy and MMLU. That damage is not from the new SNMF
  selection: the Gemini-era picks had it too. `snmf_ember` erases English the
  most (r_en 0.27 vs 0.52 / 0.45) but leaks again (+0.17 [+0.08, +0.27],
  Russian +0.37), with specificity between the two and the most invalid
  answers (English 0.16). Numbers:
  [`docs/Part 1/crosslingual_next.md`](Part%201/crosslingual_next.md) §2.3–§2.5.
- **SNMF+EMBER search: finished (2026-10-13).** Best cell for all 18
  concepts (EMBER δ fixed, SNMF grid searched again on top), same judge and
  policy as SNMF. All grids complete, no pick loops. Mean `harmonic_alpaca`
  **0.909**, against 0.821 for SNMF and 0.735 for EMBER; highest of the three
  for 17 of 18 concepts (Harry Potter is the exception). These are selection
  scores on `QA_train`, so they are optimistic.
  Results: `judge_rerun/SNMF_EMBER/pass2_outputs/` (layout:
  `judge_rerun/SNMF_EMBER/README.md`), numbers and notes:
  [`docs/Part 1/snmf_ember_search.md`](Part%201/snmf_ember_search.md).
- **Running (submitted 2026-10-15 22:00): SNMF+EMBER repair**, 24 chained
  jobs (1042781–1042804): fresh grids at the new δ for Baseball, Gambling, Gun
  and Uranium; re-score of the cells with invalid answers for the other 14
  (Pornography in full); then Alpaca answers + Qwen judging only for cells new
  to a top 3, old answers and scores reused for the rest. Expected done
  2026-10-16 morning. Details:
  [`snmf_ember_search.md`](Part%201/snmf_ember_search.md), "Repair".
- **Next:** cross-lingual `ember` (4 changed δ fresh; for the rest regenerate
  only the invalid questions) and `snmf_ember` (after its repair); `snmf`
  stays. First point `crosslingual/run_edited.py` at the repaired picks
  (`pass2_rescored/`). The cross-lingual analysis proper
  ([`crosslingual_next.md`](Part%201/crosslingual_next.md) §6) waits for it.
- **Part 2, multilingual feature search (step 1, World War II): done, result too
  weak.** Features fit on en+ru+de+es+fr together, then filtered with the same
  Gemini judge as the English features: 0 embedding + 2 MLP features kept
  (English: 1 + 15). Causes found: rank 100 too small for 5 languages, foreign
  neutral sentences are definitions (unlike English), some broken by the
  sentence splitter. Write-up:
  [`docs/Part 2/multilingual_features.md`](Part%202/multilingual_features.md).
- **Part 2, foreign sentence data (2026-10-14): in progress.** Raw Wikipedia
  articles downloaded for all 18 concepts × en/ru/de/es/fr (all exist), plus
  related articles where one article is too short. Hand-curating related
  articles did not converge (each refill drifts into neighbour topics: other
  holidays, drugs, sports). Options under discussion: sentence-level filter on
  native Wikipedia, or translating the English sets. Same write-up, "Trial:
  adding related articles" and "Options".
- **Part 2, translated sentence sets (2026-10-15): ready (steps 1–4b done; next: step 7, refit
  World War II on them).** The English
  concept and neutral sets are translated into ru/de/es/fr with
  gemini-3.5-flash-lite (chosen after a World War II pilot against Flash; $3.79 in total),
  output in `experiments/multilingual_fit/data_translated/`. Quality scores
  (COMET-Kiwi median 0.86–0.88 per language): the ~3% that scored low, mostly sports and
  weapons jargon, were re-translated with Flash; 483 of 684 replaced where the new version
  scored higher ($0.19). Term check against Wikipedia's interlanguage titles plus a
  gemini-3.8-flash judge ($0.82): terms 97.7–99.2% and names 99.2–99.7% correct; 117 errors
  in 111 sentences. Also found: ~890 translations drop parenthesised text. Step 4b fixed 90 of
  the 111 and restored the parentheses (missing now in 2–5% instead of 19–35%), $0.50. No human
  reviewer, so the human spot-check is replaced by the automatic judges. Total Gemini
  cost for the translated data: ~$5.90, including the Flash pilot and the judge comparison tests.
  Not error-free: 21 known term errors, ~120 missing parentheticals, and errors no check can
  see; ~0.1% of the data, too little to affect the fit. An optional clean-up is described in the
  write-up, "Known remaining errors".
  Plan and current step: same write-up, "Translated sentence sets (option 6)".

Per-concept status: [`docs/Part 1/snmf_search.md`](Part%201/snmf_search.md) (SNMF),
[`docs/Part 1/ember_search.md`](Part%201/ember_search.md) (EMBER) and
[`docs/Part 1/snmf_ember_search.md`](Part%201/snmf_ember_search.md) (SNMF+EMBER).

This file tracks *where things stand right now* — what's done, what's next, why.
It is not a results dump: numbers and detailed write-ups live in the files linked
below, so this page stays short enough to read in one sitting.

---

## 1. History — the Gemini-era run

The project originally ran its cross-lingual erasure experiment (8 of 18
concepts, embedding + MLP erasure, evaluated in 5 languages) using **Gemini**
as the judge for open-ended QA correctness and AlpacaEval coherence scoring —
the same judge the paper itself used.

That entire run (results CSVs, baselines, the per-question records, Gemini's
usage log, and the write-up of what it found) has been moved to
[`archive_gemini/`](../archive_gemini/) at the repo root, untouched. The full
execution record for that era — what was run, what broke, the actual numbers —
is at [`archive_gemini/docs/progress_gemini.md`](../archive_gemini/docs/progress_gemini.md),
and the results summary at
[`archive_gemini/docs/crosslingual_results_summary.md`](../archive_gemini/docs/crosslingual_results_summary.md).
Nothing from that era is deleted — it's kept as a reference point to compare
the new judge's numbers against, and the agreement study below re-judges its
exact saved completions for that reason.

## 2. What we're doing now, and why

Gemini costs real money at the volume this project needs (tens of thousands of
judge calls for a full grid search across 18 concepts), so we're switching to a
**free, local** judge. One constraint: the judge must not be the same model
family as whatever we're erasing (currently Gemma-2-2b-it, later Llama), so it
stays an independent measuring instrument.

Full reasoning and the detailed design are in
[`docs/judge/judge_swap_plan.md`](judge/judge_swap_plan.md). In short:

- **Why decoupled generation + judging ("Pass 1" / "Pass 2"):** the model
  being erased and the judge can't both sit in GPU memory at once on this
  hardware (11 GB RTX 2080 Ti / 12 GB Titan Xp per card). So the plan splits
  every evaluation into (1) generate the model's answers and save them to
  disk, no judge involved, then (2) separately load a judge and score what's
  saved. This also means Pass 2 gets the *entire* GPU budget for the judge,
  since the model under erasure isn't loaded at all at that point.
- **Scope for now:** Gemma only (Llama needs a bigger GPU budget — parked for
  later). One erasure method first — EMBER's own δ (strength) search, since
  that's the one place the judge's score directly decides the result. All 18
  concepts. EMBER is done (2026-10-09). SNMF is done (2026-10-10).
  SNMF+EMBER is done (2026-10-13).
  Cross-lingual round 2 runs are done (`pristine`, `ember`, `snmf`
  2026-10-11; `snmf_ember` 2026-10-14). On 2026-10-15 the MC scoring was
  found to cut answers off (see Status); the repair comes before the analysis.

## 3. Where things stand

**Done:**
- Archived the Gemini-era results/baselines (see §1).
- Downloaded fresh, complete features for all 18 concepts (Gemma, both
  erasure tracks) from the public features dataset.
- **Pass 1** (`judge_rerun/pass1_ember_delta.py`): for all 18 concepts, ran
  EMBER's 8 candidate deltas, scored MMLU/concept-QA/SimDom-QA (letter-graded,
  no judge), and saved the 150 AlpacaEval completions per delta per concept,
  unscored. Output under `judge_rerun/Ember/pass1_outputs/<model>/rank<rank>/seed<seed>/<Concept>/`.
- **Pristine baseline pass** (`judge_rerun/pass1_baseline.py`): the same
  machinery with no edit applied. Computes MMLU (global, 50 items) and the
  150 Alpaca completions once, then loops over all 18 concepts for QA/SimDom
  accuracy (concept-specific item sets). Output under `.../_baseline/`.
  Pristine `mmlu_acc=0.52` matches the archived Gemini-era baseline exactly —
  a good reproducibility check before trusting anything built on top of it.
- **Judge model chosen — see §3.1 for the full writeup.** Qwen2.5-32B-Instruct,
  4-bit (bitsandbytes NF4), confirmed via a rigorous same-completions agreement
  study against Gemini's archived scores.
- The pristine baseline's Alpaca completions are judged (both 14B and 32B, for
  comparison) — `baseline_alpaca_scores__<judge_id>.json` under `.../_baseline/`.
- **Pass 2 proper** (`judge_rerun/pass2_delta_search.py`): judges all 8 deltas'
  saved completions for one concept, computes `harmonic_alpaca`, and picks the
  best δ (`io.pick_best_embed_delta`, same formula the original pipeline uses
  — see §3.2). Per-delta *and* per-item checkpointing, so a timeout or
  preemption only loses the one delta in flight — resubmitting the identical
  `sbatch` command continues from there (see §3.3). Submitted per concept via
  `judge_rerun/run_pass2_delta_search_anygpu.sbatch --concept "<Concept>"`,
  each concept kicked off by hand. **Finished for all 18 concepts
  (2026-10-09)** under the top-3 × 50-question policy (6 judged directly, 12
  re-derived from the cached full-grid scores; see
  `judge_rerun/Ember/README.md`). Per-concept status:
  [`docs/Part 1/ember_search.md`](Part%201/ember_search.md).

- **SNMF Pass 1** (`judge_rerun/pass1_snmf_top3.py`), all 18 concepts: for
  each concept, narrows SNMF's 144-cell hyperparameter grid to the top 3 cells
  by the judge-free `harmonic` (reusing the archived Gemini-era grid when one
  exists, otherwise running the grid fresh, ~2h10m per concept) and generates —
  without judging — the first 50 Alpaca completions for each of those 3 cells.
  Output under `judge_rerun/SNMF/pass1_outputs/`.
- **SNMF Pass 2** (`judge_rerun/pass2_snmf_judge.py`), **all 18 concepts
  (2026-10-10)** (jobs 1004452–1004455, then 1006221 for Pornography, Republic
  of Ireland, Valentine's Day; all clean): judges the 3 cells × 50 completions
  with the Qwen2.5-32B judge and picks the best cell by `harmonic_alpaca`. It
  uses the same Alpaca baseline as EMBER and, like SNMF's original validate
  stage, has **no Alpaca gate**. Output under `judge_rerun/SNMF/pass2_outputs/`,
  one-row-per-concept `best_hps_summary.csv`. Every grid was checked complete
  (144 distinct, fully scored cells). Per-concept picks: [`docs/Part 1/snmf_search.md`](Part%201/snmf_search.md).
- **Ancient Rome SNMF, redone with a loop screen** (jobs 1006121 + 1006227):
  all three original top cells degenerated ("Roman Roman …" in 45–49 of 50
  completions). `judge_rerun/pass1_snmf_screen.py` walked the grid ranking,
  skipped duplicate and looping cells, and kept the first 3 healthy ones (16
  rows walked; grid ranks 9, 11, 16). New pick: efficacy 0.62, coherence 0.95,
  `harmonic_alpaca` 0.80. Old results in `judge_rerun/SNMF/archive/`. A
  paragraph for the article is in `snmf_search.md`.
- 2026-10-10 reorganization: the SNMF outputs moved from
  `judge_rerun/pass{1,2}_outputs_snmf/` to `judge_rerun/SNMF/pass{1,2}_outputs/`
  (same layout as `judge_rerun/Ember/`), and the three SNMF scripts' default
  paths were updated. SNMF Slurm logs moved to `judge_rerun/SNMF/archive/slurm_logs/`.
- **Cross-lingual pristine baselines** (`crosslingual/run_baseline.py`, job
  1006226, 42 min on 1 GPU): the unedited model on `QA_test` + `SimdomQA_test`
  (50 MC questions each) for 18 concepts × 5 languages, judge-free. Output:
  `results/crosslingual/google_gemma-2-2b-it/baseline/<Concept>/<lang>.json`
  + `summary.csv`. Mean `QA_test` accuracy en 0.73, de 0.64, es 0.63, fr 0.60,
  ru 0.52; 2 of 90 cells below the 0.40 cutoff (Harry Potter/ru,
  Valentine's Day/ru). Details: `crosslingual_next.md` §2.
- **Cross-lingual edited runs, `ember` + `snmf`** (`crosslingual/run_edited.py`,
  pilot 1007121 on Harry Potter, then 1007145 / 1007146, ~2 h each on 1 GPU,
  finished 2026-10-11): each concept's pick applied, the same MC evaluation as
  the baseline, plus per-(concept, method) English MMLU on the 1000-question
  test split (pristine 0.553). All 36 edits reproduced the search's English
  `qa_acc` / `simdom_acc` / `mmlu_acc` exactly. Output:
  `results/crosslingual/google_gemma-2-2b-it/{ember,snmf}/`. First-look
  results and caveats (`ember`'s invalid-answer rate on a few concepts):
  `crosslingual_next.md` §2.3.
- **SNMF+EMBER Pass 1** (`pass1_snmf_top3.py --with-ember`, jobs
  1007155–1007165, finished 2026-10-11): top 3 cells × 50 Alpaca answers for
  all 18 concepts in `judge_rerun/SNMF_EMBER/pass1_outputs/`; 8 concepts
  reused the archived combined grid, 10 ran a fresh one (144 cells each,
  `results/snmf_ef/`).
- **SNMF+EMBER Pass 2** (`pass2_snmf_judge.py --with-ember`, jobs
  1007174–1007177, 2026-10-11 → 2026-10-13): all 18 concepts judged and
  picked, `complete: true`, 3 of 3 cells each. The 4 jobs were preempted 28
  times in total and resumed from their checkpoints every time (no
  tracebacks; 43 h of judging over 54 cells). Loop check: at most 3 looping
  answers of 50 in any cell, at most 1 in a picked cell, so no concept needed
  the loop screen (Ancient Rome, which broke under plain SNMF, has 0). Four
  picks have an Alpaca fraction below 0.9 (Pornography 0.85, Valentine's Day
  0.87, Baseball 0.88, Uranium 0.89; no gate, by design). Summary:
  `judge_rerun/SNMF_EMBER/pass2_outputs/.../best_hps_summary.csv` (with
  `delta_embed`). Slurm logs moved to `judge_rerun/SNMF_EMBER/archive/slurm_logs/`.

  | Method (mean of 18) | efficacy | specificity | coherence | harmonic_alpaca |
  |---|---|---|---|---|
  | EMBER | 0.520 | 0.968 | 0.992 | 0.735 |
  | SNMF | 0.724 | 0.848 | 0.959 | 0.821 |
  | SNMF+EMBER | 0.877 | 0.909 | 0.960 | 0.909 |

  Per concept and caveats: [`snmf_ember_search.md`](Part%201/snmf_ember_search.md).
- **`snmf_ember` cross-lingual run** (`crosslingual/run_edited.py --methods
  snmf_ember`, job 1040077, 2026-10-14, 2h22m on 1 GPU, no preemption): all
  18 concepts × 5 languages, all 18 edit checks exact (Δ 0.0, features
  match). Output: `results/crosslingual/google_gemma-2-2b-it/snmf_ember/`.
  Results: English `QA_test` 0.38 (r_en 0.27, strongest of the three), mean
  non-English Δ +0.17 [+0.08, +0.27] (Russian +0.37), `SimdomQA_test`
  retention 0.69–0.81, English MMLU 0.511, `QA_test` invalid rate 0.16 in
  English (up to 0.56 in a cell). `crosslingual_next.md` §2.5.
- **MC scoring repair, EMBER + SNMF** (2026-10-15). EMBER: all 144 δ cells
  re-scored with `mc_fixed.py` (jobs 1041943, 1042045, 1042275), new top-3 δ
  judged (1042338, 1042339), final picks in `judge_rerun/Ember/pass2_rescored/`
  (4 changed). SNMF: the 328 cells with invalid answers re-scored
  (1042306–1042309), Pornography and COVID-19 on all 144 cells (1042373,
  1042489), COVID-19's one new top-3 cell generated + judged (1042615,
  1042618); all 18 picks unchanged, final file in
  `judge_rerun/SNMF/pass2_rescored/`. Determinism: one Pornography QA question
  answers differently than in the October runs (systematic, so its grids were
  re-scored in one run); elsewhere only single-answer flips in broken cells.
  Details: [`ember_search.md`](Part%201/ember_search.md) "MC scoring repair",
  [`snmf_search.md`](Part%201/snmf_search.md) "MC scoring repair check".

  | Method (mean of 18) | efficacy | specificity | coherence | harmonic_alpaca |
  |---|---|---|---|---|
  | EMBER, after repair | 0.430 | 0.970 | 0.992 | 0.667 |
  | SNMF, after repair | 0.721 | 0.848 | 0.959 | 0.820 |
  | SNMF+EMBER | not yet repaired | | | |

**In progress / open:**
- `judge_rerun/pass2_outputs_ori/` (Ori's EMBER side folder, owned by
  `oripetel`) is a verified byte-identical copy of
  `judge_rerun/Ember/archive/pass2_outputs_ori/` and nothing reads it. Ori can
  delete it (`rm -rf`), then commit the deletion.

**Not done yet:**
- Cross-lingual analysis proper (paired method differences, the effect of
  invalid answers, targeted leak; plain means are in §2.3 / §2.5), the
  `english`-wrapper check, and open-QA cross-lingual judging:
  `crosslingual_next.md` §6.
- The final-test stage and relearning are later steps.
- Cleanup done: `judge_rerun/smoke_qwen_judge*.py` and their `.sbatch` files
  were one-off hardware-validation scripts (do 4-bit Qwen + this GPU even
  work together), superseded by `qwen_judge.py` as the real reusable module —
  deleted 2026-10-06. Also deleted `judge_rerun/run_pass2_delta_search.sbatch`
  (pinned `--gres=gpu:geforce_rtx_2080:3`, which only exists on 2 of the 5
  `studentkillable` nodes and queued behind busy 2080 jobs even when titan
  nodes sat idle); `run_pass2_delta_search_anygpu.sbatch` (`--gres=gpu:3`,
  no model pinned) replaces it and is confirmed working on titan GPUs.
- Cleanup 2026-10-09: `judge_rerun/` now holds only the Pass 1 / Pass 2
  pipeline for EMBER and SNMF. Removed the one-off check scripts and their
  sbatch wrappers: the §3.1 agreement study (`agreement_study.py`,
  `run_agreement_study*.sbatch`), the Gemma-4 smoke/sensitivity checks
  (`smoke_gemma_judge.py`, `judge_sensitivity.py`, `run_smoke_gemma.sbatch`,
  `run_gemma4_judge_eval.sbatch`), the Titan GPU diagnostics and benchmark
  (`titan_*.py`, `run_titan_*.sbatch`, `run_rtx_diag4.sbatch`,
  `titan_bench_results.jsonl`) and `run_judge_baseline.sbatch` (it defaulted to
  the rejected 14B judge). All are in git history at commit `7f86c2e`, e.g.
  `git checkout 7f86c2e -- judge_rerun/agreement_study.py`. The agreement-study
  result files were kept, in `docs/judge/judge_selection/`.

### 3.1 Judge model: which one, and why

**Constraint (judge_swap_plan.md §2.1):** must not be the same family as
either erasure target (Gemma, Llama), and should be genuinely multilingual
for the later cross-lingual re-judging. Qwen2.5-Instruct fits both.

**The real question was size: 14B vs 32B**, both run locally in 4-bit NF4
via bitsandbytes (free, no API key, no rate limit — the local-judge path the
plan had originally deprioritized for lack of GPU headroom, which turned out
to be less of a problem than expected once Pass 2 gets the whole card with no
model-under-erasure loaded). Evaluated with the plan's own §6 agreement
study: re-judge Gemini's *exact* already-scored 150 Alpaca-train completions
(`archive_gemini/data/baseline_samples/samples_alpaca_train_*.json`) with
Qwen, same prompts, compare item-by-item. This is more rigorous than
comparing mean scores across *different* completions (no confound from
generation nondeterminism) — `agreement_study.py` (removed from `judge_rerun/`
2026-10-09; in git history at `7f86c2e`).

| | Qwen2.5-14B-Instruct | **Qwen2.5-32B-Instruct** | gate (plan §6) |
|---|---|---|---|
| instruct mean_delta | -0.100 | **-0.073** | within ±0.15 |
| instruct exact-match | 91.3% | **92.7%** | "healthy" |
| instruct MAE | 0.100 | **0.073** | — |
| fluency mean_delta | **+0.200 (fails)** | **+0.040** | within ±0.15 |
| fluency exact-match | 61.3% | 64.0% | "healthy" |
| fluency MAE | 0.413 | 0.373 | — |
| fluency prompt used | Qwen-strengthened (see below) | original, verbatim Gemini | — |
| parse failures | 0/150, 0/150 | 0/150, 0/150 | — |

**32B passes the gate on both axes, with the original unmodified prompts.
14B fails on fluency even after a prompt fix**, so 32B is the pick for Pass 2.
Full per-item confusion matrices: `docs/judge/judge_selection/agreement_study_train.json`
(14B) and `docs/judge/judge_selection/agreement_study_train__qwen2.5-32b-instruct.json` (32B);
the rejected Gemma-4-12B candidate is `docs/judge/judge_selection/agreement_study_train__gemma-4-12b-it.json`.
Gemma-4-12B-it was rejected on two counts: it is the same family as the erasure
target, and on the same 150 items its fluency mean_delta is **+0.367** (fails the
±0.15 gate; instruct is fine, +0.013). It gave the ceiling score almost
regardless of quality. It also needed its own venv (`transformers==4.53.3` can't
load its tokenizer). The smoke-test log is in git history (`docs/last_step.md`,
deleted 2026-10-14).

**Caveat worth remembering:** exact-match on fluency is ~62-64% and MAE is
~0.37-0.41 for *both* models — a real per-item scatter that happens to cancel
out in the mean. The gate is about the mean (which is what `harmonic_alpaca`
actually consumes — a mean score across 150 items, not an item-level
pass/fail), so this doesn't block adoption, but it means don't over-trust any
*single* item's score from either judge.

**What the 14B run surfaced along the way (kept here since it explains why
32B's prompt is the plain original, not a tuned one):** the verbatim
`ALPACA_FLUENCY_PROMPT` tells the judge once, in passing, to ignore a
cut-off final sentence (every Pass 1 completion is generated at
`max_new_tokens=200`, so this is common, not an edge case). Qwen2.5-14B
didn't reliably follow that — its own reasoning repeatedly penalized
completions for "ending abruptly" despite the instruction, dragging mean
fluency down (`mean_fluency_score=1.15` vs Gemini's archived `1.613`, from
`baseline_alpaca_scores.json`, the very first baseline-only check before the
rigorous agreement study existed). A more forceful, Qwen-specific restatement
of the same rule (`ALPACA_FLUENCY_PROMPT_QWEN` in `qwen_judge.py`) brought
14B's mean fluency to 1.85 — overshot the other way, still outside the gate.
**32B needed no such tuning**: it follows the plain original instruction
correctly, which is the more trustworthy signal (a prompt that needs
model-specific surgery to behave is a smell).

### 3.2 The delta-selection formula Pass 2 has to replicate

From `ember/erasure/methods/ember.py::grid_eval_kwargs` and
`ember/evals/orchestrator.py` (traced, not reinvented):

```
mmlu_frac = clip((mmlu_acc − 0.25) / (mmlu_base − 0.25), 0, 1)
qa_frac   = clip((qa_acc   − 0.25) / (qa_base   − 0.25), 0, 1)
sim_frac  = clip((simdom_acc − 0.25) / (simdom_base − 0.25), 0, 1)
efficacy    = 1 − qa_frac
specificity = harmonic_mean(mmlu_frac, sim_frac)
harmonic    = harmonic_mean(efficacy, specificity)

alp_instr_frac = min(1, mean_instruct_score / alp_instr_base)   ← needs Qwen-judged baseline
alp_flu_frac   = min(1, mean_fluency_score  / alp_flu_base)
coherence       = harmonic_mean(alp_instr_frac, alp_flu_frac)
harmonic_alpaca = harmonic_mean(efficacy, specificity, coherence)

gate: if mmlu_frac < 0.90 → zero everything for that row
gate: if alp_instr_frac < 0.9 or alp_flu_frac < 0.9 → harmonic_alpaca = 0

best δ = row with max harmonic_alpaca (ties → smaller δ);
         if every row is gated to 0 → fall back to δ=0.0 (no edit beats editing)
```

`mmlu_base`/`qa_base`/`sim_base` (per-concept, letter-parsed, judge-independent)
are already cached in `archive_gemini/data/baselines/google_gemma-2-2b-it/` —
safe to reuse as-is. `alp_instr_base`/`alp_flu_base` could **not** reuse the
archived Gemini numbers (mixing judge instruments silently corrupts the
fraction — `judge_swap_plan.md` §2.3), which is why the pristine baseline's
Alpaca completions were judged fresh with Qwen (§3 above).

### 3.3 Operational findings worth not re-discovering

- **Look at raw outputs, not only rates (2026-10-15).** A high "invalid" rate
  in MC scoring was read for days as "the edit breaks generation". It was the
  4-token budget cutting off "The answer is **C**". One look at the saved
  `model_raw` strings would have shown it. Whenever an invalid rate is above a
  few percent, print the most common raw outputs before interpreting it. And
  save per-question records in every scoring run: the HP grids kept only
  totals, which is why they have to be re-scored instead of re-parsed.
- **Greedy decoding is a prefix property.** With `do_sample=False`, the first
  N tokens of a longer generation equal an N-token generation on the same GPU
  type / dtype / batch size. That lets a rescore with a larger budget check
  itself against the old numbers (`mc_fixed.py` stores `old_raw` /
  `old_letter`).
- **…but runs are not bit-identical (2026-10-15).** Re-running the same edit
  on the same GPU type a few weeks later changed single answers: rarely in
  normal cells, more often in broken near-random ones, and systematically for
  one Pornography `QA_train` question (+1 correct in every EMBER δ and every
  SNMF cell). Two pristine baselines (Gemini-era vs Qwen-era) also differ by
  one question for 8 concepts. So never rank cells scored in different runs
  against each other when they are within about one question: re-score the
  whole comparison set in one run (that is why EMBER was re-scored on all 144
  cells, and Pornography / COVID-19 SNMF on all 144).
- **`max_qa_acc` is not applied in MC mode.** `configs/snmf*_gemma.yaml` set
  `max_qa_acc: 0.6`, but `ember/evals/orchestrator.py` checks it only in the
  open-QA branch. In the `train_mc` grids only `min_mmlu` gates. Reproducing a
  grid's `harmonic` with that gate added gives wrong numbers for ~45% of cells.
- **EMBER barely changes Alpaca answers.** The edit touches only the concept
  tokens' embeddings, which the general Alpaca questions rarely contain: 48–50
  of 50 answers are identical across δ, and so are the Qwen scores. Judging a
  new δ can copy per-item scores for identical answers instead of 48 min of
  Qwen.

- **bitsandbytes/Triton/python3-dev gotcha:** `bitsandbytes.nn` eagerly
  imports Triton-based SwitchBack modules (unused for plain NF4 4-bit) that
  JIT-compile a C extension on first import. Needs `python3-dev` headers,
  present on the login node but **not** on this cluster's compute nodes →
  crashes before the judge even loads. Fixed by stubbing that submodule out
  before import (`qwen_judge.py::_stub_out_bnb_triton`).
- **GPU headroom:** a single 11 GB RTX 2080 Ti leaves only ~0.6 GB free after
  loading Qwen2.5-14B in 4-bit — any batched call risks OOM. Fix: shard across
  2 GPUs (`device_map="auto"`, no manual `max_memory` tuning needed) for 14B,
  3 GPUs for 32B (~16-18 GB of weights alone). `qwen_judge.py` also has an
  adaptive batch-halving retry on `CUDA OutOfMemoryError` as a safety net.
- **Page cache, not "re-downloading":** model load time varies wildly by
  node — 18s warm vs. ~68 min cold for the 32B model — because Linux's page
  cache for the weight files is per-node, and each compute node
  (`s-002..s-006`) is independent. This is a **one-time, per-node** tax, not
  a repeated one: it does not multiply with the number of concepts judged in
  a job that keeps the model loaded across concepts. It also doesn't explain
  the bigger problem below — even a perfectly warm cache barely moves a
  multi-hour judging job.
- **The real bottleneck is GPU compute time, not I/O.** Measured throughput:
  Qwen2.5-14B ≈ 12-13s/item, Qwen2.5-32B ≈ 28-32s/item (batched). One
  concept's full 8-delta Alpaca grid is 2,400 judge calls → **~8.3h (14B) or
  ~19.3h (32B) of judging per concept**, independent of caching. The
  `studentkillable` partition caps a single job at 24h wall time
  (`MaxTime=1-00:00:00`), and jobs on it can be preempted
  (`PreemptMode=REQUEUE`) — so running several concepts sequentially in one
  job (to amortize one model load across them) needs checkpointing *across*
  concepts/deltas, not just within a judging pass, or it needs to be split
  into more, smaller per-concept jobs. Full grid volume, corrected (the
  pristine baseline is global, not per-concept, so it's not ×18):
  `18 concepts × 8 deltas × 150 items × 2 axes = 43,200 calls` (+300 for the
  baseline, already done).
- **Running Pass 2 across two cluster accounts to split the GPU budget:**
  `judge_rerun/run_pass2_delta_search_anygpu.sbatch` hardcodes `UV_CACHE_DIR` and
  `HF_HOME` to paths under the original user's home dir, which are
  read-only for a second user — a collaborator's `sbatch` fails at `uv`
  cache init (`Permission denied`) before the job even starts. Fix is either
  (a) the second user exports their own `UV_CACHE_DIR`/`HF_HOME` before
  calling `sbatch` (Slurm exports the submitting shell's environment by
  default, so the script's `${VAR:-default}` picks that up) — simplest, but
  costs that user a fresh ~62GB download of the Qwen2.5-32B-Instruct judge
  weights; or (b) `chmod -R o+rwX` the original user's `.cache/uv` and
  `.cache/huggingface` to share the already-downloaded weights — this
  filesystem has no ACL support (`setfacl` → "Operation not supported"), so
  there's no tighter-scoped way to share than opening to everyone with
  filesystem access, same breadth as the repo's own `chmod 777`.
- **Per-user GPU cap on `studentkillable`:** the partition's QoS
  (`gpustudentkill`) limits each user to `MaxTRESPerUser=gres/gpu=12`
  running at once, with no separate cap on job count. Since each Pass 2 job
  requests 3 GPUs, that's **up to 4 concurrent Pass 2 jobs per user** —
  submitting more just queues them `PENDING` until GPUs free up, rather than
  failing. Two users each running near that cap can have ~8 concepts'
  grids in flight simultaneously. Each concept should go to only one user's
  job at a time — two jobs judging the same concept would race on the same
  checkpoint files.

## 4. Code map (new code only)

| Path | What |
|---|---|
| `judge_rerun/pass1_ember_delta.py` | Pass 1: generate + letter-graded scoring, 8 deltas, one concept |
| `judge_rerun/pass1_baseline.py` | pristine-model counterpart of the above: no edit, MMLU+Alpaca once, QA/SimDom per concept |
| `judge_rerun/run_pass1.sbatch`, `run_pass1_baseline.sbatch` | SLURM wrappers for the two above |
| `judge_rerun/submit_pass1_all.sh` | submits one Pass 1 job per concept, all 18 |
| `judge_rerun/qwen_judge.py` | reusable `QwenJudge`: batched generation, OOM-halving retry, resume-from-checkpoint, the Qwen-tuned fluency prompt |
| `judge_rerun/pass2_judge_baseline.py` | judges the pristine baseline's Alpaca completions with Qwen (any size via `--judge-model`) |
| `judge_rerun/pass2_delta_search.py` | Pass 2 proper: judges one concept's 8-delta grid, picks the best δ (see §3.2) |
| `judge_rerun/run_pass2_delta_search_anygpu.sbatch` | SLURM wrapper for the above, one concept per job, 3 GPUs (any model), 24h |
| `judge_rerun/run_judge_baseline_32b.sbatch` | SLURM wrapper for `pass2_judge_baseline.py` with the adopted Qwen2.5-32B judge (produces the baseline denominators Pass 2 needs) |
| `judge_rerun/pass2_recompute_topk.py` | legacy tool: re-derives a concept's top-3/n=50 pick from its cached full-grid judge scores, no GPU (used once for the 12 old-policy concepts; see `judge_rerun/Ember/README.md`) |
| `judge_rerun/pass1_snmf_top3.py`, `run_pass1_snmf_top3.sbatch` | SNMF Pass 1: narrows the 144-cell HP grid to the top 3 cells by the judge-free harmonic and generates (does not judge) their first 50 Alpaca completions; output `judge_rerun/SNMF/pass1_outputs/` |
| `judge_rerun/Ember/pass1_outputs/` | Pass 1 + baseline output: `grid_raw.csv` / `baseline_raw.csv`, saved completions, judged scores, per concept |
| `judge_rerun/Ember/pass2_outputs/` | Pass 2 output with the original 4-token MC scoring, **superseded by `judge_rerun/Ember/pass2_rescored/`** (top-3/n=50, all 18 concepts): per-concept `delta_metrics__<judge_id>.csv`, `best_delta__<judge_id>.json`, `alpaca_judged__<judge_id>.json`, plus `best_deltas_summary.csv`. Layout, policy and the `archive/` of older runs: `judge_rerun/Ember/README.md` |
| `docs/judge/judge_swap_plan.md` | the full plan and reasoning behind all of the above |
| `judge_rerun/SNMF/pass1_outputs/` | SNMF Pass 1 output: per concept `top3_hps.json` and `alpaca_completions/cell_0..2.json` (3 cells × 50 completions); Ancient Rome also `screen/` |
| `judge_rerun/pass1_snmf_screen.py`, `run_pass1_snmf_screen.sbatch` | SNMF Pass 1 variant with a loop screen (used for Ancient Rome only): walks the full grid ranking generating answers, skips duplicates (identical answers) and cells with >5/50 looping answers, stops at the first 3 distinct non-looping cells, and writes them in Pass 1's format plus `screen/screen_report.csv` |
| `judge_rerun/pass2_snmf_judge.py`, `run_pass2_snmf.sbatch` | SNMF Pass 2: judges the 3 cells × 50 completions with Qwen2.5-32B (3× RTX 2080 Ti), picks the best cell; several concepts per job, resumable |
| `judge_rerun/SNMF/pass2_outputs/` | SNMF Pass 2 output with the original MC scoring (all 18 concepts; the repair kept every pick, final copy in `judge_rerun/SNMF/pass2_rescored/`): per concept `best_hp__qwen2.5-32b-instruct.json` (the pick, with the full HP dict), `cell_metrics__*.csv`, `alpaca_judged__*.json`, `_rows__*.json`, plus `best_hps_summary.csv`. Layout and `archive/`: `judge_rerun/SNMF/README.md` |
| `docs/Part 1/ember_search.md` | per-concept EMBER Pass 2 status and picks |
| `docs/Part 1/snmf_search.md` | per-concept SNMF status (Pass 1 + Pass 2, the picks, completeness check, the Ancient Rome problem and fix) |
| `docs/Part 1/crosslingual_next.md` | plan and status of cross-lingual round 2 |
| `judge_rerun/mc_fixed.py` | **fixed MC / MMLU scoring** (2026-10-15): 32-token budget, `parse_letter_fixed`, decodes only the new tokens; each record also keeps the old-equivalent (`old_raw`, `old_letter`) as a determinism control. Same prompts, shuffle and batching as `ember/evals` |
| `judge_rerun/rescore_ember_grid.py`, `run_rescore_ember.sbatch` | re-scores the EMBER δ grid with `mc_fixed.py` (only δ with an invalid answer, or `--all-deltas`; `--reparse` re-applies the parser to saved records on the CPU), saves per-question records, checks the control, and compares old vs new top 3 / pick on the CPU. Output `judge_rerun/Ember/rescore_mc/`; never writes into `pass1_outputs/` / `pass2_outputs/` |
| `judge_rerun/pass2_ember_rescored.py`, `run_pass2_ember_rescored.sbatch` | EMBER Pass 2 on the re-scored grid: same policy as `pass2_delta_search.py`, reuses old Qwen means for top-3 δ already judged, judges the rest (first 50 saved completions), recomputes `harmonic_alpaca`. Output and **final EMBER picks**: `judge_rerun/Ember/pass2_rescored/` |
| `judge_rerun/rescore_snmf_grid.py`, `run_rescore_snmf.sbatch` | the same for the SNMF grid (`--with-ember`: the SNMF+EMBER grid, EMBER δ from the repaired picks; a concept whose δ changed gets a fresh 144-cell grid): re-scores cells with an invalid answer (`--cells all` for the whole grid) with Pass 1's edit code, skips QA / SimDom for cells still MMLU-gated, checks the control, re-ranks as Pass 1 (only `min_mmlu` gates). Output `judge_rerun/SNMF/rescore_mc/` |
| `judge_rerun/pass2_snmf_rescored.py`, `run_pass2_snmf_rescored_gen.sbatch`, `run_pass2_snmf_rescored_judge.sbatch` | SNMF (and with `--with-ember` SNMF+EMBER, output `judge_rerun/SNMF_EMBER/pass2_rescored/`) Pass 1 + 2 on the re-scored grid. `--stage generate`: first 50 Alpaca answers + loop check (from `pass1_snmf_screen.py`) for cells new to the top 3, next cell moves up if one loops; Ancient Rome keeps its screen. `--stage judge`: Qwen for new cells only, old Qwen means for the rest, pick by `harmonic_alpaca`. Output and **final SNMF picks**: `judge_rerun/SNMF/pass2_rescored/` |
| `crosslingual/` | **new** cross-lingual package (MC, judge-free): `data.py` (translated items, language-aligned option shuffle), `prompts.py` (native-language scaffold), `mc_eval.py` (one concept × language cell), `run_baseline.py` + `run_baseline.sbatch` (pristine model), `run_edited.py` + `run_edited.sbatch` (apply each concept's `ember` / `snmf` / `snmf_ember` pick, edit check, per-concept MMLU) |
| `results/crosslingual/google_gemma-2-2b-it/baseline/` | pristine cross-lingual baseline: `<Concept>/<lang>.json` (per-question records) + `summary.csv`, pristine MMLU in `_mmlu.json` |
| `results/crosslingual/google_gemma-2-2b-it/{ember,snmf,snmf_ember}/` | edited cross-lingual runs: `<Concept>/<lang>.json` + `edit.json` (HP, edit check, MMLU), `summary.csv` with retention and leak per (concept, language) |
| `judge_rerun/SNMF_EMBER/` | **Final** SNMF+EMBER search (`--with-ember` on the three SNMF scripts), all 18 concepts: `pass1_outputs/`, `pass2_outputs/` (same files as SNMF, plus `best_hps_summary.csv` with `delta_embed`), `archive/slurm_logs/`. Layout: `judge_rerun/SNMF_EMBER/README.md` |
| `docs/Part 1/snmf_ember_search.md` | SNMF+EMBER status, per-concept picks, completeness and loop check, comparison with EMBER / SNMF alone |
| `archive_gemini/crosslingual/` | the Gemini-era cross-lingual runner, kept for reference (not used) |

Everything under `ember/` (the paper's own reference implementation) is
untouched — `judge_rerun/` and `crosslingual/` only import from it, never edit it.
