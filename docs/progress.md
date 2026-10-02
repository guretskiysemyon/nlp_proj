# Progress log

**Last updated:** 2026-10-02
**Status:** Switching the evaluation judge away from Gemini. Pass 1 (generation
only, no judge) is built, verified on one concept, and running across all 18.

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
the new judge's numbers against.

## 2. What we're doing now, and why

Gemini costs real money at the volume this project needs (tens of thousands of
judge calls for a full grid search across 18 concepts), so we're switching to a
**free** judge. One constraint: the judge must not be the same model family as
whatever we're erasing (currently Gemma-2-2b-it), so it stays an independent
measuring instrument.

Full reasoning and the detailed design are in
[`docs/judge_swap_plan.md`](judge_swap_plan.md). In short:

- **Which judge:** a free, open-weight, non-Gemma/non-Llama model (a
  Qwen-class model is the leading candidate — either via a free hosted API,
  or run locally). Not finalized yet.
- **Why decoupled generation + judging ("Pass 1" / "Pass 2"):** the model
  being erased and the judge can't both sit in GPU memory at once, and a
  free API judge is rate-limited. So the plan splits every evaluation into
  (1) generate the model's answers and save them to disk, no judge involved,
  then (2) separately load a judge and score what's saved. This also means
  the GPU never sits idle waiting on a slow/rate-limited judge call.
- **Scope for now:** Gemma only (Llama needs a bigger GPU than this project
  has used so far — parked for later). One erasure method first — EMBER's
  own δ (strength) search, since that's the one place the judge's score
  directly decides the result. All 18 concepts.

## 3. Where things stand

**Done:**
- Archived the Gemini-era results/baselines (see §1).
- Downloaded fresh, complete features for all 18 concepts (Gemma, both
  erasure tracks) from the public features dataset.
- Built `judge_rerun/pass1_ember_delta.py` — for one concept, runs EMBER's
  8 candidate deltas, scores MMLU/concept-QA/SimDom-QA (no judge needed,
  letter-graded), and saves the AlpacaEval completions as plain text
  *without* scoring them yet.
- Verified it end-to-end on Harry Potter — sensible, varying numbers, real
  saved completions. Output lives under `judge_rerun/pass1_outputs/`.
- Submitted the same job for the remaining 17 concepts
  (`judge_rerun/submit_pass1_all.sh`), running now.

**Not done yet:**
- **Pass 2** doesn't exist yet: load a judge, score every saved completion,
  compute the final score (`harmonic_alpaca`), and pick the best δ per
  concept. Blocked only on picking the actual judge model/provider.
- Nothing beyond EMBER's δ search — SNMF, SNMF+EMBER, validate/final-test
  stages, relearning, and the cross-lingual re-evaluation are all later
  steps, once this first piece is proven end-to-end on one concept.

## 4. Code map (new code only)

| Path | What |
|---|---|
| `judge_rerun/pass1_ember_delta.py` | Pass 1: generate + letter-graded scoring, one concept |
| `judge_rerun/run_pass1.sbatch` | SLURM job wrapper for the above |
| `judge_rerun/submit_pass1_all.sh` | submits one job per concept, all 18 |
| `judge_rerun/pass1_outputs/` | output: one `grid_raw.csv` + saved completions per concept |
| `docs/judge_swap_plan.md` | the full plan and reasoning behind all of the above |

Everything under `ember/` (the paper's own reference implementation) and
`crosslingual/` (the original language-leakage experiment code) is untouched —
`judge_rerun/` only imports from them, never edits them.
