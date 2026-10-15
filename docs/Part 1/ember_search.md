# EMBER Pass 2 delta search — status by concept

Tracks `judge_rerun/pass2_delta_search.py` progress (judge the candidate
deltas' saved Alpaca completions, pick the best delta) for each of the 18
concepts. Updated 2026-10-15.

> **⚠ The picks below are provisional (2026-10-15).** The multiple-choice
> scoring used by the grid cut answers off after 4 tokens. Under the EMBER edit
> that counted many real answers as wrong and inflated efficacy at large δ.
> Repair in progress: see "MC scoring repair" below.

## MC scoring repair (2026-10-15, in progress)

### The problem

The paper's MC evaluators (`ember/evals/mc.py`, `ember/evals/mmlu.py`) generate
only **4 new tokens** (MMLU: 6) and take the first capital A–D in the output.
The pristine model answers with a bare letter, so that works. Under the EMBER
embedding edit, Gemma often answers in a sentence instead: "The answer is
**C**", "La respuesta correcta es …", "Die richtige Antwort ist …". The 4-token
cut ends at `"The answer is **"`, no letter is found, and the answer is counted
as **wrong** ("invalid"). An invalid answer looks like successful erasure.

- **Size in the EMBER grid:** invalid `QA_train` answers are 0 at δ ≤ 5, 1.6%
  at δ = 10, 8.7% at δ = 50, 11.3% at δ = 100, 12.0% at δ = 200 (mean over 18
  concepts). At the chosen δ the mean is 10%, up to 28% (Culture of Greece,
  Uranium). SimDom and MMLU are almost unaffected (≤ 1%), so the format change
  is specific to questions about the erased concept.
- **Effect:** efficacy at large δ is inflated, so the fast screen and the final
  pick may have favoured large δ partly for this reason.
- **Not garbage:** in the saved cross-lingual records, about 90% of invalid
  answers are a cut-off answer phrase (`"The answer is **"` alone is 404 of
  ~650). Real nonsense (`"##  verw"`, `"Bitte,"`) is about 15 answers.
- **Two smaller bugs found along the way:**
  - the parser rejects lowercase letters (`"c"`); this happens even for the
    pristine model, about 12 answers per method;
  - `WrappedHFModel._extract_response` cuts the decoded text at the first
    `"model\n"`. When an option ends with the word "model", the "answer" is
    the rest of the prompt (seen 4 times: `'B. Bellman equation\nC. …model\nD'`
    parsed as B, the real answer is D).
- **Why it was not caught:** the high invalid rates were reported in the
  cross-lingual docs (2026-10-11) but read as "broken generation". Nobody
  opened the raw outputs, so the truncation was not seen.

### What is affected

| Result | Affected? | Evidence |
|---|---|---|
| EMBER δ search (this file) | **yes** | 60 of 144 grid cells have an invalid answer; 10% at the picks |
| SNMF+EMBER search | **yes, most** | 1,955 of 2,592 cells; 15% at the picks (Uranium 48%, Gambling 40%) |
| SNMF search | **possibly** | picks have 0 invalid, but 328 of 2,592 cells have some (mostly MMLU, 280), which lowered their specificity, so they could enter the top 3 |
| Cross-lingual `ember`, `snmf_ember` | **yes** | `QA_test` invalid 6% / 10%; also depends on the picks above |
| Cross-lingual `snmf`, pristine baseline | no (0.3%, lowercase letters, re-scorable on CPU) | |
| Qwen Alpaca judging, judge study | no | open-ended text, not MC |

**The grids saved only totals per cell**, not the per-question answers
(`pass1_ember_delta.py` and `ember/erasure/pipeline.py` drop the records as
`_`). So a cell cannot be fixed question by question: every cell with at least
one invalid answer is re-scored in full. A cell with no invalid answer cannot
change (see determinism below) and is skipped.

### The fix (`judge_rerun/mc_fixed.py`, `ember/` untouched)

- **Budget:** 32 new tokens for MC and MMLU (`MC_MAX_NEW_TOKENS`,
  `MMLU_MAX_NEW_TOKENS`). Generation stops at end-of-turn, so bare-letter
  answers cost nothing extra.
- **Decode only the new tokens:** no `"model\n"` search, so the extraction bug
  is gone.
- **Parser** `parse_letter_fixed`, three rules in order:
  1. the whole output is one letter, any case, any wrapping: `c`, `C.`,
     `**B**`, `(D)`;
  2. it starts with a capital letter plus a delimiter: `C. Hadrian`, `D)`,
     `**B** …` (the delimiter is required, so "A fire lance is …" is not read
     as A);
  3. an answer phrase (answer / Antwort / respuesta / réponse / ответ / option)
     followed by a capital letter on the same line.
  No "first capital anywhere" fallback.
- **Checked on the CPU:** 21 unit cases pass. On all 35,232 saved cross-lingual
  answers the old parser read as valid, the new parser gives the same letter;
  the only newly valid answers are 58 lowercase letters.
- **Unchanged:** prompts, option shuffle, one-question-at-a-time MC, batch-8
  MMLU, bf16, RTX 2080 Ti. These keep the generated tokens identical.

**Determinism control.** Greedy decoding means the first 4 (MMLU: 6) tokens of
a 32-token run are exactly the old run's output. Every record therefore also
stores `old_raw` / `old_letter` (the first 4/6 tokens, parsed with the old
parser). Their totals must reproduce `grid_raw.csv` exactly. If they do not,
the δ is flagged `CONTROL FAILED` and the rescore is not comparable.

### The rescore (`judge_rerun/rescore_ember_grid.py`)

For each δ: re-apply the edit exactly as Pass 1 did (feature counts checked
against the grid), re-score MMLU / `QA_train` / `SimdomQA_train` with the fixed
evaluator, and save every per-question record. Then, on the CPU, Pass 2's
policy runs on the old and the new grid (min_mmlu gate, top 3 by fast
harmonic, `harmonic_alpaca` from the cached Qwen scores) and the two are
compared. Run on the old grid, this reproduces the current Uranium pick
exactly (δ = 100, 0.9200).

- **Output:** `judge_rerun/Ember/rescore_mc/<model>/rank100/seed42/<Concept>/`
  with `records/delta_<δ>.json`, `grid_rescored.csv` (new metrics under
  `grid_raw.csv`'s column names, plus `grid_*` = old grid and `old_*` =
  old-equivalent, `control_ok`, `features_match`) and `compare.json`.
  `pass1_outputs/` and `pass2_outputs/` are never written to.
- **Run:** `sbatch judge_rerun/run_rescore_ember.sbatch --concept <C> [--all-deltas]`
  (1× RTX 2080 Ti). The default re-scores only δ with an invalid answer;
  `--all-deltas` re-scores all 8. `--compare-only` redoes the CPU comparison.
- **If a new δ enters the top 3:** Pass 1 already saved 150 Alpaca completions
  for all 8 δ of all 18 concepts, so nothing is generated. Only the first 50
  are judged with Qwen (`pass2_delta_search.py --deltas <δ>`). If the top 3
  stays the same, the pick is recomputed on the CPU.

### Plan and status

1. ~~Fix the scoring and check it on saved outputs~~ (done 2026-10-15).
2. **Pilot: Uranium, all 8 δ** (most invalid answers in the EMBER grid; δ = 50 /
   100 / 200 had 24 / 28 / 30% invalid). Job **1041943**, submitted 2026-10-15,
   `s-005`. Result: pending.
3. The other 17 concepts (only δ with invalid answers: 60 cells in total).
   Judge new top-3 δ, if any.
4. SNMF+EMBER: concepts whose δ changes need a fresh grid at the new δ plus
   Pass 1 + Pass 2. The others need their invalid cells re-scored
   ([`snmf_ember_search.md`](snmf_ember_search.md)).
5. SNMF: re-score the 328 cells with invalid answers, with the `min_mmlu` gate
   applied again to the corrected MMLU. Needs its own script: Pass 1 generated
   Alpaca answers only for the old top 3, and stores them by rank (`cell_0..2`),
   so new cells are generated separately (first 50 only).
6. Cross-lingual `ember` / `snmf_ember` (and `snmf` if its picks change) with
   the fixed scoring, then the docs.

## Delta-selection policy (updated 2026-10-08)

Judging all 8 deltas on the full 150 saved Alpaca completions is the cost
driver (~2.5h/delta on the rtx_2080 nodes). The policy is now two-staged to
cut that ~8x, without changing any of the underlying formulas:

1. **Fast screen (judge-free).** `grid_raw.csv`'s mmlu/qa/simdom accuracies
   already give every delta's `efficacy`/`specificity`/`harmonic` with zero
   judge calls. Rank the deltas that pass `min_mmlu` by that fast `harmonic`
   and keep only the top 3.
2. **Judge, cheaply.** Qwen-judge only those 3 deltas, and only their first
   50 saved Alpaca completions (`idx < 50` — the saved completion order is
   fixed across every delta/concept via `load_alpaca_indices("train")`, so
   "first 50" is the same 50 questions everywhere).
3. **Pick.** Same `harmonic_alpaca` formula and `io.pick_best_embed_delta`
   tie-break as before, just computed from 3×50 instead of 8×150.

This is implemented as `--top-k`/`--alpaca-n` flags on
`pass2_delta_search.py` (defaults 3/50; pass `--top-k 8 --alpaca-n 150` to
recover the exact original behavior), so it's what new `sbatch` submissions
use by default now. For a concept Pass 2 had **already fully judged** under
the old (8×150) policy, `judge_rerun/pass2_recompute_topk.py` re-derives the
new-policy pick from the cached per-item judge scores already on disk — no
GPU, no judge calls, and it never overwrites the original
`_rows__`/`alpaca_judged__`/`best_delta__` files (it writes
`best_delta_topk__{judge}.json` / `delta_metrics_topk__{judge}.csv`
alongside them instead). It has been run for all 12 such concepts (2026-10-09)
and the results promoted into `judge_rerun/Ember/pass2_outputs/`; 10 of the 12
picks equal the old full-grid pick (e.g. Harry Potter, delta=10.0), see the
table below for the two that differ.

**SNMF**: the same top-3 + 50-question policy applies there. Pass 1
(`judge_rerun/pass1_snmf_top3.py`) is done for all 18 concepts, and Pass 2
(`judge_rerun/pass2_snmf_judge.py`) for 15 (2026-10-10). Per-concept status is
in `docs/Part 1/snmf_search.md`.

Final results for all 18 concepts: `judge_rerun/Ember/pass2_outputs/` — start with
`google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv` (layout and policy: `judge_rerun/Ember/README.md`).
"Best δ" below is the winner under the top-3 / n=50 policy.

| Concept | Status | Best δ | Person |
|---|---|---|---|
| Ancient Rome | Finished (re-derived from full grid) | 200 | Mary |
| Artificial intelligence | Finished (re-derived from full grid) | 100 | Mary |
| Baseball | Finished (re-derived from full grid) | 100 (old full-grid pick: 200) | Mary |
| COVID-19 pandemic | Finished (re-derived from full grid) | 50 | Mary |
| Cannabis | Finished (re-derived from full grid) | 200 | Mary |
| Culture of Greece | Finished | 200 | Mary |
| Gambling | Finished (re-derived from full grid) | 100 | Ori |
| Golf | Finished | 10 | Ori |
| Gun | Finished (re-derived from full grid) | 100 | Ori |
| Halloween | Finished (re-derived from full grid) | 10 (old full-grid pick: 5) | Ori |
| Harry Potter | Finished (re-derived from full grid) | 10 | Semyon |
| Heroin | Finished | 100 | Ori |
| Nazism | Finished | 100 | Ori |
| Pornography | Finished | 5 | Ori |
| Republic of Ireland | Finished (re-derived from full grid) | 100 | Ori |
| Uranium | Finished (re-derived from full grid) | 100 | Ori |
| Valentine's Day | Finished (re-derived from full grid) | 200 | Ori |
| World War II | Finished | 5 | Ori |

**Provenance (2026-10-09).** Six concepts were judged directly under top-3/n=50: Culture of Greece (Mary, job 999320),
World War II (job 1001318), and Golf / Heroin / Nazism / Pornography (Ori; jobs 1001317 / 1000902 / 1000903 / 1000904).
The other twelve had already been judged on the old 8×150 grid, so their top-3/n=50 result was re-derived from the cached
per-item scores with `pass2_recompute_topk.py` (Halloween's was the last one missing). Ten of those twelve picks equal the old
full-grid pick; **Baseball (200 → 100)** and **Halloween (5 → 10)** differ, and both are near-ties (identical judge-free
score, Alpaca term within 0.001).

**Permissions (resolved 2026-10-09).** The standard output dirs for Golf / Heroin / Nazism / Pornography were owned by
`semyong` with mode 755, so nobody else could write there: Golf failed on this twice (Mary's job 993617, then her multi-concept job 999320, which
had finished Culture of Greece and died at Golf with a `PermissionError` before reaching the rest), and Ori's four
concepts had to be run with `--out-dir …/pass2_outputs_ori`. Everything is now consolidated under
`judge_rerun/Ember/` (a world-writable tree), so that workaround is no longer needed; the old side folders
(`pass2_outputs_ori`, `pass2_outputs_topk`) and the full-grid originals are kept in `judge_rerun/Ember/archive/`.
(The original `judge_rerun/pass2_outputs_ori/` is owned by `oripetel`, so only Ori can delete it.)

## How to submit

Submit one concept (fast screen to top-3, then judges only those 3 on their
first 50 Alpaca questions by default; checkpoints per delta, so a timeout or
preemption just needs the identical command resubmitted to continue):

```
sbatch judge_rerun/run_pass2_delta_search_anygpu.sbatch --concept "<Concept Name>"
```

e.g.:

```
sbatch judge_rerun/run_pass2_delta_search_anygpu.sbatch --concept "World War II"
```

To run the original full 8-delta/150-question grid instead, add
`--top-k 8 --alpaca-n 150`.

Inputs and outputs default to `judge_rerun/Ember/pass1_outputs/` (read) and
`judge_rerun/Ember/pass2_outputs/` (written); that tree is world-writable, so
no `--out-dir` is needed. A concept whose `best_delta__*.json` already exists
there is skipped — add `--overwrite` (or use another `--out-dir`) to redo it.

**Batching several concepts into one job.** `--concept` takes multiple
values and runs them back-to-back in the same process, loading the Qwen
judge only once. The judge load itself is cheap (~20-30s on the rtx_2080
nodes, confirmed from the Halloween run's log) -- the real cost this avoids
is SLURM's queue wait for a free `s-004`/`s-005` slot, which `sacct` has
shown running 15-18h when both pinned nodes are busy. A concept already
finished (its `best_delta__*.json` exists) is skipped immediately with no
GPU touched, so a timed-out multi-concept job can be resubmitted with the
identical command and will only resume the unfinished ones:

```
sbatch judge_rerun/run_pass2_delta_search_anygpu.sbatch \
    --concept "Culture of Greece" "Golf" "Heroin" "Nazism" "Pornography" "World War II"
```

At ~3h/concept under the default top-3/n=50 policy, about 6 concepts fit
comfortably inside the sbatch script's existing `--time=24:00:00` limit in a
single job (the example above is the batch Mary submitted on 2026-10-08).

Despite its name, `run_pass2_delta_search_anygpu.sbatch` is pinned to
`--gres=gpu:geforce_rtx_2080:3`, i.e. only `s-004`/`s-005`. Jobs may queue
while the titan nodes (`s-002`/`s-003`/`s-006`) are idle — that's intended.
Unpinned runs on titan (jobs 981330/981331/981347) did not work: on TITAN Xp
the Qwen-32B judge ran ~1190 s/batch vs ~30 s on 2080 Ti (likely the bf16
4-bit compute dtype, which Pascal lacks), and on `s-002` CUDA failed to
initialize so the jobs silently ran on CPU. All three hit the 24h limit without
finishing a single delta. An fp16 variant for titan was to be tested separately;
the Titan diagnostic/benchmark scripts were removed from `judge_rerun/` on 2026-10-09
(still in git history, commit `7f86c2e`).


## multi language add on
run eg.
sbatch run_train_mf_features.sbatch --concepts "World War II" --languages en+ru+de+es+fr


batch job 1010008