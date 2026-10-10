# Cross-lingual leakage, round 2 — plan and status

**Status (2026-10-10):** the HP searches it depends on are finished (EMBER and
SNMF, all 18 concepts). A new, judge-free MC package is in `crosslingual/`,
and **the pristine baselines are done** for 18 concepts × 5 languages
(job 1006226). **Next: the edited-model runner** (§4). Nothing is running.

[`crosslingual_plan.md`](../archive_gemini/docs/crosslingual_plan.md) is still the
design reference (metric, screening rule, F1–F6 principles), and
[`archive_gemini/docs/crosslingual_results_summary.md`](../archive_gemini/docs/crosslingual_results_summary.md)
holds the Gemini-era results that this round redoes.

## 1. The experiment

Erase a concept with the edit that was tuned **in English**, then ask the
concept's held-out multiple-choice questions (`QA_test`, 50 per concept) in
**en, de, es, fr, ru**. If the concept is still answerable in language L but
not in English, the erasure leaked.

Per (concept, method, language):

```
retention r_L = clip((acc_L − 0.25) / (pristine_acc_L − 0.25), 0, 1)   # per-language denominator
leak      Δ_L = r_L − r_en          # > 0: the concept survives better in L than in English
```

A (concept, language) cell is screened out when the pristine model scores
below 0.40 there (it never knew the concept in L). The edit is applied once and
never re-tuned per language. `SimdomQA_test` (50 similar-domain questions per
concept, also in all 5 languages) measures in-language specificity. Details:
`crosslingual_plan.md` §3–§4.

## 2. What is ready

| Input | Where | State |
|---|---|---|
| EMBER best δ, 18 concepts (Qwen judge) | `judge_rerun/Ember/pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv` | done |
| SNMF best HP, 18 concepts (Qwen judge) | `judge_rerun/SNMF/pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_hps_summary.csv` (full dict per concept in `<Concept>/best_hp__qwen2.5-32b-instruct.json`) | done ([`snmf_search.md`](snmf_search.md)) |
| Translated QA, 18 concepts × 5 languages, MC + open | `data/qa_lang/` | repaired and validated (`scripts/validate_qa_lang.py` → PASS) |
| MC evaluation code | `crosslingual/` (`data.py`, `prompts.py`, `mc_eval.py`, `run_baseline.py`) | **new, works**; used for the baselines |
| Pristine baselines, 18 × 5, `QA_test` + `SimdomQA_test` | `results/crosslingual/google_gemma-2-2b-it/baseline/<Concept>/<lang>.json` + `summary.csv` | **done** (job 1006226, 42 min, 1 GPU) |
| Gemini-era results, 8 concepts × 3 methods, MC | `archive_gemini/results/cross_lang_test/` | reference only (see §2.2) |

The old Gemini-era runner (`run.py`, `best_hp.py`, `baseline.py`,
`evaluate.py`) was moved to `archive_gemini/crosslingual/` and is not used.

### 2.1 How the new package asks the questions

- **Prompt fully in the target language** (the `native` wrapper): instruction
  lines, question and options are all in L; only the answer letters stay Latin
  A–D, because the parser reads them back. The English prompt is byte-identical
  to the HP search's `ember.evals.mc.build_mc_prompt`. (`crosslingual/prompts.py`)
- **Language-aligned option shuffle**: the order is seeded by (concept, set,
  question index, seed) only, so question *i* has the same correct letter in
  every language (checked for all 90 cells). The model's letter-position bias
  is then identical across languages and cannot show up as a leak.
  (`crosslingual/data.py`)
- **Greedy decoding, `max_new_tokens=4`, first A–D letter parsed**, one question
  at a time, as in `ember.evals.mc.evaluate_mc_generation`. No judge anywhere.
- Every cell JSON keeps the per-question records (prompt options, raw output,
  parsed letter), so Δ can be paired by question.

### 2.2 Baseline results

Pristine `QA_test` accuracy (50 questions per cell, `*` = below the 0.40 cutoff):

| Concept | en | de | es | fr | ru |
|---|---|---|---|---|---|
| Ancient Rome | 0.80 | 0.58 | 0.68 | 0.64 | 0.48 |
| Artificial intelligence | 0.78 | 0.78 | 0.76 | 0.78 | 0.68 |
| Baseball | 0.66 | 0.62 | 0.60 | 0.56 | 0.48 |
| COVID-19 pandemic | 0.64 | 0.48 | 0.54 | 0.52 | 0.44 |
| Cannabis | 0.94 | 0.88 | 0.90 | 0.80 | 0.66 |
| Culture of Greece | 0.62 | 0.52 | 0.62 | 0.54 | 0.52 |
| Gambling | 0.84 | 0.80 | 0.76 | 0.70 | 0.58 |
| Golf | 0.80 | 0.76 | 0.72 | 0.72 | 0.50 |
| Gun | 0.82 | 0.66 | 0.64 | 0.62 | 0.66 |
| Halloween | 0.60 | 0.48 | 0.48 | 0.48 | 0.48 |
| Harry Potter | 0.80 | 0.60 | 0.58 | 0.50 | 0.36* |
| Heroin | 0.68 | 0.52 | 0.46 | 0.46 | 0.50 |
| Nazism | 0.74 | 0.68 | 0.72 | 0.66 | 0.50 |
| Pornography | 0.82 | 0.84 | 0.76 | 0.76 | 0.70 |
| Republic of Ireland | 0.74 | 0.66 | 0.54 | 0.58 | 0.48 |
| Uranium | 0.62 | 0.56 | 0.50 | 0.52 | 0.46 |
| Valentine's Day | 0.46 | 0.44 | 0.44 | 0.42 | 0.32* |
| World War II | 0.74 | 0.62 | 0.62 | 0.62 | 0.48 |
| **Mean** | **0.73** | **0.64** | **0.63** | **0.60** | **0.52** |

- 88 of 90 cells pass the 0.40 screen. Invalid (unparseable) answers are low;
  the worst is COVID-19 in one language at 0.14.
- **Valentine's Day** is weak everywhere, including English (0.46). With
  pristine − 0.25 ≈ 0.2 as the retention denominator, its Δ will be very noisy.
- **These do not reproduce the Gemini-era baselines, by design.** For the 8
  shared concepts, 36 of 40 cells differ (by up to 0.14). The questions are
  identical, but the old code used a different option shuffle: only 86 of the
  2,000 questions got the same option order, and the model gave the same answer
  on all 86. The other questions were asked with the options in a different
  order. The model has a letter preference (it picks C/D more than A/B), so on
  uncertain questions the order decides: 176 went right → wrong and 160
  wrong → right, which mostly cancels. So the differences come from the option
  order, not from a bug. Neither shuffle is "right"; the new one is better for
  Δ (§2.1). Consequence: the old plan's check "`ember` must reproduce the
  archived `qa_acc` exactly" no longer applies.

## 3. Scope (decided)

| Choice | Decision | Why |
|---|---|---|
| Format | **MC only** | judge-free, cheap, and the Gemini-era headline was MC too. Open QA is a later step (§6) |
| Concepts | **all 18** | the Gemini run had n=8 and its method differences landed at t≈2.0, just short of significance (\|t\|≈2.4 needed). One cell's Δ has noise ±0.30, so only averages over concepts mean anything. n=18 cuts the across-concept SE by ~1.5× |
| Methods | **`pristine` (done), `ember`, `snmf`** | `snmf_ember` has no Qwen-era pick; see §5 |
| Languages | en, de, es, fr, ru | 2 cells screened out (Harry Potter/ru, Valentine's Day/ru) |
| Wrapper | **`native` only for now** | the Gemini run found the wrapper effect negligible (mean −0.014). The `english` wrapper (English instructions, translated question) is not implemented in the new package; add it to `prompts.py` if wanted |
| Test split | `QA_test` + `SimdomQA_test` | `QA_train` picked the HPs, so it would flatter English |

**Cost.** The baseline took ~28 s per (concept, language) cell on one
RTX 2080 Ti (both sets). An edited model is the same 5 cells plus applying
the edit and one English MMLU pass: a few minutes per (concept, method), so
**~2–3 GPU-hours for all 36 edited models**. One 1-GPU job (or a few) is enough;
no Qwen, no 3-GPU pinning.

## 4. Next: the edited-model runner

New CLI next to `run_baseline.py` (e.g. `crosslingual/run_edited.py` +
`.sbatch`) that reuses `mc_eval.evaluate_cell` unchanged, so edited and
pristine runs see the same items, shuffle, prompts and decoding. Per
(concept, method): load the model, apply the edit, evaluate all 5 languages,
restore or reload, next.

### 4.1 Reading the picks

- `ember`: `{"delta_embed": best_delta_embed}` from `best_deltas_summary.csv`.
  `ember.hp_columns()` (`io.EMBED_COLUMNS`) also lists `k_features_embed` and
  `n_tokens_edited`, but `pass1_ember_delta.py` records those from the
  edit's returned info rather than passing them in. Confirm `ember.apply`
  needs only `delta_embed`.
- `snmf`: `best_hp` from `<Concept>/best_hp__qwen2.5-32b-instruct.json`, plus
  `delta_embed = 0.0`, exactly as `judge_rerun/pass1_snmf_top3.py::_build_hp`
  applies it. `neurons_thresh` is in `io.SNMF_HP_COLUMNS` but not in the JSON
  (it is empty in the grid CSV too), so pass it the same way `_build_hp` does.
- Load the YAML from `configs/ember_gemma.yaml` / `snmf_gemma.yaml`, since
  `method.apply` reads other config fields from it.
- **Check the edit before trusting it:** the English `QA_test` accuracy is not
  comparable to the search's (different split), but the edited model's
  `QA_train` English accuracy (or its MMLU) should match the grid row's
  `qa_acc` / `mmlu_acc` for the same HP. That replaces the old
  "reproduce the archive" check.

Write the resolved HP into every output cell, so each result says which edit
produced it.

### 4.2 MMLU: one per edited model, keyed by concept

The Gemini-era `baseline.py` cached an edited model's MMLU under
`_mmlu_by_config/<config>/` with **no concept in the path**, so the first
concept's MMLU was reused for every later concept (`mmlu_acc_en` is identical
for all 8 concepts within each method in that run). Its specificity /
harmonic / "MMLU retained" columns are wrong; its retention and Δ (the headline)
do not use MMLU and are unaffected. The new runner must compute MMLU per
(concept, method), using the same MMLU set as the HP search (pristine
`mmlu_acc = 0.52`).

### 4.3 Output

`results/crosslingual/google_gemma-2-2b-it/<method>/<Concept>/<lang>.json`, the
same cell format as `baseline/` (plus the HP and MMLU), and a `summary.csv`
per method. Skip existing cells, so a preempted job resumes.

## 5. `snmf_ember` — options (still open)

The paper's headline config. The Gemini run found it leaks about as much as
`ember` (Δ +0.22 vs +0.25) while erasing English harder.

1. **Skip it in round 2** (recommended). `ember` vs `snmf` is already the core
   contrast the question asks about (embedding edit vs MLP edit).
2. **Proper re-run**: the SNMF grid with each concept's EMBER δ applied
   (`configs/snmf_ember_gemma.yaml`, results under `snmf_ef/`), through the
   same Pass 1 / Pass 2 split. ~2h10m per concept for the fresh 144-cell grid
   (1 GPU), plus ~2.5 h of Qwen judging (3 GPUs). That is ~80 GPU-hours for
   18 concepts, a separate project step.
3. **Composed shortcut**: apply the best SNMF HP + the best EMBER δ without
   tuning them together. Cheap, but it is not the paper's method, so it
   must be labelled as such.

## 6. After the edited runs

- **Analysis**: Δ per method and language, averaged over concepts with a
  bootstrap over concepts (the Gemini summary §5 has the code). Report how many
  cells were screened out, and `invalid_rate` per language next to retention.
- **Open QA** with the decoupled design: generate answers (Gemma, 1 GPU), then
  judge them (Qwen2.5-32B, 3 GPUs). Add the multilingual line to the judge
  prompt (`crosslingual_plan.md` §5.7). Hand-check ~30 judged items per
  language (Semyon can check Russian), and watch for refusals on sensitive
  concepts (Nazism, Heroin, Pornography, Gun, Cannabis).
- **Token overlap → leak** (`crosslingual_plan.md` §8): the "money figure",
  CPU only. With 18 concepts the Gemini-era correlations (r −0.68 / −0.91 at n=8)
  can actually be tested.
- **Cross-lingual relearning** (`crosslingual_plan.md` §10).
- Part 2 ideas (zero-shot foreign edit directions, layer sweep):
  [`Ideas and Discoveries for Part 2.md`](Ideas%20and%20Discoveries%20for%20Part%202.md).

## 7. Risks to watch

- **A broken model looks like successful erasure.** If generation collapses,
  MC answers stop parsing and count as wrong. Always report `invalid_rate` per
  language next to retention. SNMF picks have no Alpaca gate (Pornography 0.85,
  Golf 0.87); Ancient Rome's broken pick was replaced (`snmf_search.md`).
- **Near-chance baselines**: the 0.40 screen handles Russian; Valentine's Day is
  near-chance-ish in every language (§2.2).
- **Noise**: n=50 per cell, and reshuffling the options alone moves a cell by
  up to ±0.14 (§2.2). Report averages over concepts with SEs or a bootstrap,
  never single cells.
- **Specificity is mixed** (English MMLU + in-language SimDom). Say so.

## 8. Decisions

Resolved: all 18 concepts; Ancient Rome SNMF fixed (loop screen); SNMF Pass 2
finished for all concepts before the cross-lingual runs.

Still open:

1. `snmf_ember`: skip / proper re-run / composed shortcut (§5)?
2. Add the `english` wrapper check, or report `native` only?

## 9. Implementation order

1. ~~Pristine baselines~~ — done (job 1006226).
2. `crosslingual/run_edited.py` + sbatch (§4): read the picks, apply, evaluate,
   per-concept MMLU, resumable.
3. Pilot: one concept (e.g. Harry Potter) × `ember` + `snmf`. Check the edit
   reproduces the grid row (§4.1) and the English `QA_test` accuracy drops.
4. Submit the remaining 17 concepts.
5. Analysis (§6).
