# Cross-lingual leakage, round 2 — analysis before implementing

**Status (2026-10-10):** analysis only, **nothing implemented or submitted yet.**
This is the next experiment now that the Qwen-era HP searches are done. It
replaces nothing: [`crosslingual_plan.md`](crosslingual_plan.md) is still the
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
never re-tuned per language. MMLU is measured once per edited model, in
English. Details: `crosslingual_plan.md` §3–§4.

## 2. What is ready

| Input | Where | State |
|---|---|---|
| EMBER best δ, 18 concepts (Qwen judge) | `judge_rerun/Ember/pass2_outputs/google_gemma-2-2b-it/rank100/seed42/best_deltas_summary.csv` | done |
| SNMF best HP (Qwen judge) | `judge_rerun/pass2_outputs_snmf/.../<Concept>/best_hp__qwen2.5-32b-instruct.json` | 14 usable; Ancient Rome broken; 3 not judged yet ([`snmf_search.md`](snmf_search.md)) |
| Translated QA, 18 concepts × 5 languages, MC + open | `data/qa_lang/` | repaired and validated (`scripts/validate_qa_lang.py` → PASS) |
| Cross-lingual runner | `crosslingual/` (`run.py`, `best_hp.py`, `baseline.py`, `evaluate.py`, ...) | works; it produced the Gemini-era run. Needs the changes in §4 |
| Gemini-era results, 8 concepts × 3 methods, MC | `archive_gemini/results/cross_lang_test/` | reference and reproducibility check |

**The MC path needs no judge at all**: answers are graded by parsing the
letter. So the first experiment can run now, with no API key and no Qwen.
Only open-ended QA needs a judge.

**Same EMBER edits as the Gemini era for the 8 old concepts.** The Qwen-era δ
equals the Gemini-era δ for all 8 (Harry Potter 10, World War II 5,
Halloween 10, Uranium 100, Golf 10, Baseball 100, Culture of Greece 200,
Gambling 100). MC grading is greedy and judge-free, so re-running
`pristine` + `ember` on those 8 should reproduce the archived `qa_acc` exactly.
That is a free correctness check for any code change (like the
`mmlu_acc = 0.52` check in `progress.md` §3).

SNMF picks match the Gemini era in δ and layers for 4 of the 8 (Harry
Potter, World War II, Uranium, Baseball). Halloween and Culture of Greece
differ only in the `up_proj` layer range. Golf and Gambling differ outright.

## 3. Recommended scope for the first run

| Choice | Recommendation | Why |
|---|---|---|
| Format | **MC only** | judge-free, cheap, and the Gemini-era headline was MC too. Open QA is a later step (§6) |
| Concepts | **all 18** (SNMF: the usable ones) | the Gemini run had n=8 and its method differences landed at t≈2.0, just short of significance (|t|≈2.4 needed). One cell's Δ has noise ±0.30, so only averages over concepts mean anything. n=18 cuts the across-concept SE by ~1.5× |
| Methods | **`pristine`, `ember`, `snmf`** | `snmf_ember` has no Qwen-era pick: its grid (SNMF with the EMBER δ applied) was never re-run. See §5 |
| Languages | en, de, es, fr, ru | data ready. Expect Russian to be screened out for some concepts (Harry Potter/ru was, at 0.30) |
| Wrapper | both `native` and `english` | Phase C is cheap. The Gemini run found the wrapper effect negligible (mean −0.014), so `native` is the primary condition and `english` the check |
| Test split | `QA_test` only | `QA_train` picked the HPs, so it would flatter English |

**Cost.** The Gemini-era cross-lingual eval took ~0.5 h per concept per
job on 1 GPU (MC, 5 languages, both wrappers). Fixing the MMLU bug (§4.3) adds one
1000-question MMLU pass per edited model, a few minutes each. Estimate:
**~1 GPU-hour per concept for pristine + ember + snmf, ~18 GPU-hours total**,
as 18 small 1-GPU jobs (no Qwen, no 3-GPU pinning).

## 4. Code changes needed (all small, in `crosslingual/`)

### 4.1 Read the Qwen-era picks (`crosslingual/best_hp.py`)

`resolve_hp()` currently finds the HP in `results/<method>/.../hps.csv` and
`top_hps_valid.csv` through `pipeline._pick_best_hp_for_test`. Those are the
Gemini-era files, now under `archive_gemini/results/`. The new picks live
elsewhere and in a different format. Add an `--hp-source {qwen,legacy}` switch
(default `qwen`):

- `ember`: `{"delta_embed": best_delta_embed}` from `best_deltas_summary.csv`.
  `ember.hp_columns()` (`io.EMBED_COLUMNS`) also lists `k_features_embed` and
  `n_tokens_edited`, but `pass1_ember_delta.py` records those from the
  edit's returned info rather than passing them in. Confirm `ember.apply`
  needs only `delta_embed`.
- `snmf`: `best_hp` from `best_hp__qwen2.5-32b-instruct.json`, plus
  `delta_embed = 0.0`. These are the columns `pass1_snmf_top3.py::_build_hp`
  applied. `neurons_thresh` is in `io.SNMF_HP_COLUMNS` but not in the JSON
  (it is empty in the grid CSV too), so pass it the same way `_build_hp` does.
- Keep loading the YAML from `configs/ember_gemma.yaml` / `snmf_gemma.yaml`,
  since `method.apply` reads other config fields from it.
- `legacy` keeps today's behaviour, with `--results-root archive_gemini/results`,
  for reproducing the Gemini run.

Print the resolved HP and save it into each row of the summary CSV, so every
result says which edit produced it.

### 4.2 Restore the English pristine baselines

`crosslingual/baseline.py` reads the pristine English MMLU from
`data/baselines/google_gemma-2-2b-it/baseline_mmlu_test.json`. `data/baselines/`
is **empty** now, because everything moved to
`archive_gemini/data/baselines/google_gemma-2-2b-it/`. The MMLU and MC files there
are letter-graded and judge-free, so copying them back is safe. **Do not**
copy the `*_open.json` or `*alpaca*` files: those were Gemini-judged.

The per-language pristine baselines (`data/crosslingual_baselines/`) are also
empty (archived copies cover only the 8 old concepts). Recompute them for all
18 concepts. It is cheap, and it gives one provenance. Then diff the 8 old
concepts' MC numbers against `archive_gemini/data/crosslingual_baselines/` as
a check. Pass `--qa-kinds mc`: `compute_language_baseline` creates a
`GeminiEvaluator` as soon as `open` is requested.

### 4.3 Bug: the edited-MMLU cache is keyed by method only

`crosslingual/baseline.py` caches an edited model's MMLU under
`data/crosslingual_baselines/_mmlu_by_config/<config>/`, with **no concept in
the path**. The first concept's MMLU is therefore reused for every later
concept. It happened in the Gemini-era run: `mmlu_acc_en` is identical for
all 8 concepts within each method (ember 0.553, snmf 0.541,
snmf_ember 0.539). That run's **specificity / harmonic and "MMLU retained"
columns are wrong**. Its retention and Δ (the headline) do not use MMLU and
are unaffected. Fix: key the cache by concept and HP (e.g.
`_mmlu_by_config/<config>/<concept_safe>/`), and never reuse the archived
`_mmlu_by_config/`.

### 4.4 Output location

Write to a new `--out-dir`, e.g. `results/cross_lang_qwen/`, so nothing mixes
with the Gemini-era layout. Keep the per-question records (`run.py` already
saves them): Δ is paired by question (record *i* in de is record *i* in en),
which is what the qualitative failure analysis needs.

## 5. `snmf_ember` — options

The paper's headline config. The Gemini run found it leaks about as much as
`ember` (Δ +0.22 vs +0.25) while erasing English harder.

1. **Skip it in round 2** (recommended for the first run). `ember` vs `snmf`
   is already the core contrast the question asks about (embedding edit vs MLP edit).
2. **Proper re-run**: the SNMF grid with each concept's EMBER δ applied
   (`configs/snmf_ember_gemma.yaml`, results under `snmf_ef/`), through the
   same Pass 1 / Pass 2 split. ~2h10m per concept for the fresh 144-cell grid
   (1 GPU), plus ~2.5 h of Qwen judging (3 GPUs). That is ~80 GPU-hours for
   18 concepts, a separate project step.
3. **Composed shortcut**: apply the best SNMF HP + the best EMBER δ without
   tuning them together. Cheap, but it is not the paper's method, so it
   must be labelled as such.

## 6. After the first run

- **Open QA** with the decoupled design: generate answers (Gemma, 1 GPU), then
  judge them (Qwen2.5-32B, 3 GPUs). `evaluate.py::evaluate_open` calls Gemini
  inline today, so it needs splitting. Add the multilingual line to the judge
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

- **A broken model looks like successful erasure.** If generation collapses
  (Ancient Rome's SNMF pick: "Roman Roman Roman …"), MC answers stop parsing
  and count as wrong. Always report `qa_invalid` per language next to retention.
  SNMF picks have no Alpaca gate (Golf's pick is at 0.87).
- **Near-chance Russian baselines**: the 0.40 screen handles it. Report how
  many cells were screened out.
- **Noise**: n=50 per cell, so report averages over concepts with SEs or a
  bootstrap over concepts (the Gemini summary §5 has the code), never single cells.
- **Specificity is mixed** (English MMLU + in-language SimDom). Say so.

## 8. Decisions needed before implementing

1. Concepts: all 18 (recommended), or only the 8 from the Gemini run?
2. `snmf_ember`: skip / proper re-run / composed shortcut (§5)?
3. Ancient Rome SNMF: fix the pick first, or drop it from SNMF ([`snmf_search.md`](snmf_search.md))?
4. Finish SNMF Pass 2 for Pornography, Republic of Ireland, Valentine's Day
   first, or start cross-lingual with `ember` while it runs?

## 9. Implementation order (for whoever picks this up)

1. §4.2: copy the judge-free English baselines back into `data/baselines/`.
2. §4.3: fix the MMLU cache key.
3. §4.1: `--hp-source qwen` in `best_hp.py`.
4. `python crosslingual/run.py --concept "Harry Potter" --configs pristine ember --qa-kinds mc --out-dir results/cross_lang_qwen --dry-run`,
   then a real 1-concept pilot. Check the `ember` `qa_acc` per language against
   `archive_gemini/results/cross_lang_test/summary.csv`: it should match exactly.
5. Submit the rest (`scripts/run_crosslingual_all.sh` is the Gemini-era
   driver; check its flags and Slurm settings before reusing it).
6. Analysis: adapt the code in the Gemini summary §5.
