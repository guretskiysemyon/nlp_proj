# EMBER Pass 2 delta search — status by concept

Tracks `judge_rerun/pass2_delta_search.py` progress (judge the candidate
deltas' saved Alpaca completions, pick the best delta) for each of the 18
concepts. Updated 2026-10-15.

> **Final EMBER picks: `judge_rerun/Ember/pass2_rescored/google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv`
> (2026-10-15).** The multiple-choice scoring used by the original grid cut
> answers off after 4 tokens, which counted many real answers as wrong and
> inflated efficacy at large δ. After the repair 4 picks changed (Baseball
> 100 → 200, Gambling 100 → 10, Gun 100 → 5, Uranium 100 → 200) and mean
> `harmonic_alpaca` fell from 0.735 to 0.667: see "MC scoring repair" below.
> The status table further down ("Delta-selection policy") shows the
> **pre-repair** picks and is kept for the record.

## MC scoring repair (2026-10-15, done)

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
| Result | Affected? | Evidence | Outcome (2026-10-15) |
|---|---|---|---|
| EMBER δ search (this file) | **yes** | 60 of 144 grid cells have an invalid answer; 10% at the picks | 4 picks changed (Results below) |
| SNMF+EMBER search | **yes, most** | 1,955 of 2,592 cells; 15% at the picks (Uranium 48%, Gambling 40%) | to redo ([`snmf_ember_search.md`](snmf_ember_search.md)) |
| SNMF search | **possibly** | picks have 0 invalid, but 328 of 2,592 cells have some (mostly MMLU, 280), which lowered their specificity, so they could enter the top 3 | checked: all 18 picks unchanged ([`snmf_search.md`](snmf_search.md)) |
| Cross-lingual `ember`, `snmf_ember` | **yes** | `QA_test` invalid 6% / 10%; also depends on the picks above | to redo |
| Cross-lingual `snmf`, pristine baseline | no (0.3%, lowercase letters, re-scorable on CPU) | | valid as is |
| Qwen Alpaca judging, judge study | no | open-ended text, not MC | valid as is |

**The grids saved only totals per cell**, not the per-question answers
(`pass1_ember_delta.py` and `ember/erasure/pipeline.py` drop the records as
`_`). So a cell cannot be fixed question by question: it is re-scored in full.
For EMBER all 144 cells were re-scored (cheap, and it keeps each concept in one
run, see "Determinism"); for SNMF only the cells with an invalid answer, plus
full grids where runs would otherwise mix ([`snmf_search.md`](snmf_search.md)).

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
  are judged with Qwen, by `judge_rerun/pass2_ember_rescored.py` (below). If
  the top 3 stays the same, the pick is recomputed on the CPU.
- **All 8 δ, not only the invalid ones.** The first plan skipped δ with no
  invalid answer, on the argument that greedy decoding makes them identical.
  That argument holds only if generation is bit-identical across runs, and it
  is not quite (see "Determinism" below). So every concept was re-scored on
  all 8 δ, and each concept's ranking comes from one run.

### Final pass 2 (`judge_rerun/pass2_ember_rescored.py`)

Pass 2's selection on the re-scored grid, exactly as `pass2_delta_search.py`
(min_mmlu 0.90, top 3 by fast harmonic, Qwen2.5-32B on the first 50 saved
Alpaca completions, `harmonic_alpaca`, `io.pick_best_embed_delta`), with one
difference: Qwen means already computed by the old Pass 2 for the same δ and
the same 50 completions are reused. `harmonic_alpaca` is always recomputed
from the NEW efficacy / specificity. A concept with no new top-3 δ needs no GPU.

- **Output:** `judge_rerun/Ember/pass2_rescored/<model>/rank100/seed42/<Concept>/`
  (`delta_metrics__qwen2.5-32b-instruct.csv`, `best_delta__….json` with the
  old pick alongside, `alpaca_judged__….json` for newly judged δ) and
  **`best_deltas_summary.csv`, the final EMBER picks.**
- **Run:** `sbatch judge_rerun/run_pass2_ember_rescored.sbatch --concept <C> …`
  (3× RTX 2080 Ti); resumable per δ.

### Results (2026-10-15, final)

**Jobs.** Rescore: 1041943 (Uranium pilot, all 8 δ, 20 min), 1042045 (the δ
with invalid answers of the other 17 concepts, 56 min), 1042275 (their 80
clean δ, 1 h 02). Judging: 1042338 (Uranium, Cannabis, Gambling, 4 h 03) and
1042339 (Gun, Harry Potter, Heroin, Pornography, 4 h 14). No errors.

**What the fix changes in the grid.** At δ = 50 / 100 / 200 the recovered
answers were mostly correct, so `QA_train` accuracy rises and efficacy falls,
e.g. Uranium δ = 100: QA 0.32 → 0.52, invalid 0.28 → 0, efficacy 0.82 → 0.31
(pristine 0.64; 10 of the 14 cut-off answers were right). δ ≤ 5 did not change
at all. A few answers stay invalid after the fix, and they are real
non-answers: empty output (Harry Potter, Gun, Valentine's Day), "I need to know
the answer to this question." (COVID-19), broken MMLU output (Valentine's Day).

**Top 3 and picks.** 10 concepts keep their top 3; 8 got a new δ in it (judged
now: Cannabis 5, Gambling 10, Gun 5, Harry Potter 5, Heroin 10, Pornography 1,
Uranium 2 and 5). **4 picks change:**

| Concept | δ old → new | efficacy old → new | harmonic_alpaca old → new |
|---|---|---|---|
| Ancient Rome | 200 | 0.764 → 0.618 | 0.871 → 0.816 |
| Artificial intelligence | 100 | 0.281 → 0.211 | 0.536 → 0.442 |
| **Baseball** | **100 → 200** | 0.339 → 0.203 | 0.596 → 0.428 |
| COVID-19 pandemic | 50 | 0.364 | 0.630 |
| Cannabis | 200 | 0.441 → 0.305 | 0.701 → 0.568 |
| Culture of Greece | 200 | 0.980 → 0.857 | 0.983 → 0.938 |
| **Gambling** | **100 → 10** | 0.588 → 0.471 | 0.800 → 0.720 |
| Golf | 10 | 0.407 | 0.672 |
| **Gun** | **100 → 5** | 0.576 → 0.407 | 0.793 → 0.663 |
| Halloween | 10 | 0.400 | 0.659 |
| Harry Potter | 10 | 0.526 | 0.767 |
| Heroin | 100 | 0.615 → 0.564 | 0.827 → 0.795 |
| Nazism | 100 | 0.627 → 0.549 | 0.820 → 0.772 |
| Pornography | 5 | 0.367 → 0.327 | 0.635 → 0.592 |
| Republic of Ireland | 100 | 0.419 → 0.326 | 0.681 → 0.589 |
| **Uranium** | **100 → 200** | 0.821 → 0.359 | 0.920 → 0.621 |
| Valentine's Day | 200 | 0.439 | 0.684 |
| World War II | 5 | 0.400 | 0.659 |
| **Mean** | | **0.520 → 0.430** | **0.735 → 0.667** |

Specificity (0.968 → 0.970) and coherence (0.992) barely move. **EMBER erases
less than reported before**: mean efficacy 0.43, not 0.52. Where a value is
unchanged, the chosen δ had no cut-off answers.

**Determinism.** The old-scoring control reproduced the October grid exactly
for 136 of 144 cells. All 8 failures are Pornography: every δ gets exactly one
more `QA_train` question right than on 2026-10-02 (SimDom and MMLU match).
`data/mc_questions.json` has not changed since September, so this is one
near-tie question answered differently on today's node. Comparing a re-scored
δ with an old one would therefore mix two runs; with all 8 δ from one run it
does not matter. The same Pornography question also shows up in the SNMF
rescore ([`snmf_search.md`](snmf_search.md)).

**EMBER's Alpaca answers barely depend on δ.** For Uranium, 49 of the first 50
answers are identical to the unedited model's at every δ; for every newly judged
δ, 48–50 of 50 answers were identical to an already-judged δ of the same
concept, and Qwen gave them the same scores (e.g. Uranium δ = 2 / 5 / 50 / 100 /
200 all 1.94 / 1.68). The EMBER edit only changes the concept tokens'
embeddings, and the general Alpaca questions rarely contain them. So for EMBER
coherence is almost constant across δ, and the pick is decided by efficacy and
specificity. Practical consequence for later judging: per-item Qwen scores can
be copied for identical answers instead of re-judged (not done here; the jobs
were already running).

### Plan and status

1. ~~Fix the scoring and check it on saved outputs~~ (done 2026-10-15).
2. ~~Pilot: Uranium, all 8 δ~~ (job 1041943; control exact, top 3 changed).
3. ~~The other 17 concepts, all 8 δ; judge new top-3 δ~~ (done; final picks in
   `judge_rerun/Ember/pass2_rescored/…/best_deltas_summary.csv`).
4. ~~SNMF: re-score the cells with invalid answers~~ (done 2026-10-15: all 18
   picks unchanged, see [`snmf_search.md`](snmf_search.md)).
5. **SNMF+EMBER** (running since 2026-10-15 22:00, jobs 1042781–1042804):
   fresh grids at the new δ for Baseball, Gambling, Gun and Uranium, re-score of
   the cells with invalid answers for the other 14 (Pornography in full), then
   Alpaca + Qwen for cells new to a top 3
   ([`snmf_ember_search.md`](snmf_ember_search.md), "Repair").
6. **Cross-lingual** `ember` (all 18; the 4 changed δ fresh, the rest by
   regenerating only the invalid questions, whose records are saved) and
   `snmf_ember` after its search. Point `crosslingual/run_edited.py` at the new
   picks first. `snmf` stays as it is.

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

Pre-repair results for all 18 concepts: `judge_rerun/Ember/pass2_outputs/` —
`google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv` (layout and policy: `judge_rerun/Ember/README.md`).
"Best δ" below is the winner under the top-3 / n=50 policy **with the old 4-token
scoring**; the final picks are in "MC scoring repair → Results" above.

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