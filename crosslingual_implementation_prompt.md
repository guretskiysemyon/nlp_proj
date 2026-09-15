# Cross-lingual evaluation — implementation spec for Claude Code

You are adding a new, self-contained entrypoint to the `nlp_proj` repo that measures
whether EMBER's (and SNMF's) concept erasure survives translation into other
languages. This file is the full spec. Read it fully before writing any code, and
read the referenced existing files yourself (paths given below) rather than
guessing their signatures — several of them have non-obvious behavior that this
spec calls out explicitly.

**Do not modify any existing file under `ember/`.** Every existing English
pipeline run must keep working byte-for-byte identically after this change. All
new code goes in a new top-level folder, `crosslingual/`, that imports from
`ember/` but is never imported by it.

## 0. Known blocker — resolve before writing the model-loading code

`results/`, `mf_outputs/`, and `data/baselines/` do not currently exist in this
checkout. That means there is nowhere yet to read a cached best-HP row
(`hps.csv` / `top_hps_valid.csv` / `best_embed.csv`) from for any method. Before
implementing the "apply best HP" step, confirm with the user where those
directories actually live (most likely: only on the cluster where the English
grid search ran, not synced to this machine) and either (a) sync them down, or
(b) point `--results-root` at wherever they are. Build the best-HP lookup as a
parameter (`--results-root`, default `results/`) rather than a hardcoded path,
so this is a one-flag fix once the location is known. Do not attempt to
re-run any grid search to work around a missing cache — that defeats the whole
point of this script.

## 1. What varies across the matrix, and what doesn't

For one concept, the full cell space is:

- **config** ∈ {pristine, ember, snmf, snmf_ember} — which edit (if any) is
  applied to the model. This is EDIT-time and is entirely language-independent:
  the same edited weights get evaluated in every language.
- **language** ∈ {en, de, es, fr, ru} — which language the QA content and
  chat/completion output are in. This is EVAL-time only.
- **wrapper** ∈ {english, native} — for non-English languages only, whether the
  handful of instruction/label strings around the question are in English or
  translated. For `language == en` this axis is meaningless (english wrapper
  IS the native wrapper) — do not evaluate it twice.
- **qa_kind** ∈ {mc, open} — both question formats, evaluated for every cell.
- Each (qa_kind) cell always evaluates **QA** (concept retention) *and*
  **SimDom** (specificity / collateral damage) — that's the existing subset
  split (`QA_*` vs `SimdomQA_*`), not a separate axis to choose.

So the real per-concept workload is: 4 configs × 5 languages × (1 or 2
wrappers) × 2 qa_kinds × 2 subsets. **The only thing that changes between
cells within one config is which prompt text gets generated from and how it's
graded — never the model weights.** This is the core design constraint: apply
each config's edit exactly once, then loop over language × wrapper × qa_kind
against the resident in-memory model, before moving to the next config.

MMLU and Alpaca (used only for `specificity`/`coherence`, not for the concept
signal) are **not** re-run per language — they're general-domain, not
concept- or language-specific. Reuse the existing English MMLU/Alpaca numbers
for a given config via `ember.evals.baselines.compute_or_read_baseline` /
`ember.erasure.eval.get_baselines`, computed once per config, and pass them
into every language's specificity calculation. Do not add a `--eval-alpaca`
sweep across languages; it would be pure waste.

## 2. Folder layout to create

```
crosslingual/
  __init__.py
  data.py          # language-aware QA loaders
  prompts.py       # wrapper-language templates + prompt builders
  evaluate.py       # language/wrapper-aware MC + open-QA eval loops
  baseline.py       # per-language, per-concept pristine QA/SimDom baseline (+ cache)
  best_hp.py        # locate + apply a config's already-chosen best HP
  results_io.py     # output path/schema + resumability
  run.py            # the CLI entrypoint described in §5
scripts/
  run_crosslingual_all.sh   # thin loop over the agreed concept set (see §7)
```

Everything here is new code. Nothing in `ember/` is edited.

## 3. Data contract (verified against the current repo — use this, don't re-derive it)

Two data sources exist for the translated QA; **use the aggregate ones, not the
per-concept ones under `data/qa_lang/translated/<lang>/...`.** Both were
repaired and are structurally byte-identical in shape to the canonical English
files, which makes the loader trivial:

- English (canonical, unchanged): `data/mc_questions.json`, `data/open_questions.json`
- Per-language aggregates: `data/qa_lang/mc_questions_{en,de,es,fr,ru}.json`,
  `data/qa_lang/open_questions_{en,de,es,fr,ru}.json`

Shape (identical across all 5 languages, verified by hand):

```jsonc
// mc_questions_<lang>.json
{
  "Harry Potter": {
    "QA_train": [ {"q": "...", "correct_answer": "...", "options": ["...", "...", "...", "..."]}, ... ],
    "QA_test":  [ ... ],
    "SimdomQA_train": [ ... ],
    "SimdomQA_test":  [ ... ]
  },
  "Golf": { ... },
  ... // 18 concepts, exact same key spelling as data/mc_questions.json
}
// open_questions_<lang>.json: same nesting, items are {"q": "...", "a": "..."}
```

Because the shape is identical to `ember.local_datasets`'s existing loaders,
`crosslingual/data.py` should be a **parameterized copy** of
`load_mc_qa_items` / `load_open_qa_examples` (read them at
`ember/local_datasets.py` to copy exactly), adding a `lang: str = "en"`
argument that only changes which file gets opened:

```python
def _mc_path(lang: str) -> Path:
    return DATA_DIR / "mc_questions.json" if lang == "en" \
        else DATA_DIR / "qa_lang" / f"mc_questions_{lang}.json"
# same pattern for open_questions
```

Reuse `ember.local_datasets.MCQAItem` / `OpenQAItem` dataclasses directly
(import them, don't redefine). Do not touch the per-concept
`translated/<lang>/{mc,open}_questions/<Concept>.json` files at all — they're
a legacy intermediate format from the translation step and their filenames
don't even follow `ember.utils._safe_concept` consistently (e.g. "COVID-19
pandemic" → `_safe_concept` gives `COVID-19_pandemic`, but the actual file is
`COVID_19_pandemic.json` — dash vs. underscore). The aggregate files sidestep
this entirely since you index by the exact concept string, not a filename.

The 18 concept name strings (must match exactly, they're dict keys):
`"Ancient Rome"`, `"Artificial intelligence"`, `"Baseball"`, `"COVID-19
pandemic"`, `"Cannabis"`, `"Culture of Greece"`, `"Gambling"`, `"Golf"`,
`"Gun"`, `"Halloween"`, `"Harry Potter"`, `"Heroin"`, `"Nazism"`,
`"Pornography"`, `"Republic of Ireland"`, `"Uranium"`, `"Valentine's Day"`,
`"World War II"`.

## 4. Prompt wrapper (the native/English ablation)

Read `ember/evals/mc.py::build_mc_prompt` and `ember/evals/open_qa.py`'s
`f"Question: {ex.question}\nAnswer:"` line — both hardcode English scaffold
text and are called from inside `evaluate_mc_generation` /
`evaluate_open_qa`, which can't be parameterized without touching those
files. So `crosslingual/prompts.py` re-implements just the prompt string
assembly (not the generation/grading loop) with a `wrapper_lang` argument:

```python
WRAPPER_STRINGS = {
    "en": {"choose": "Choose the single best answer.", "choices": "Choices:",
           "answer_fmt": "Answer only with one letter (A, B, C, or D) and no formatting.",
           "question_label": "Question:", "answer_label": "Answer:"},
    "de": { ... translated ... },
    "es": { ... },
    "fr": { ... },
    "ru": { ... },
}

def build_mc_prompt(question, options_by_letter, wrapper_lang="en"): ...
def build_open_prompt(question, wrapper_lang="en"): ...
```

For a cell where `language == wrapper_lang`, this reproduces the native-wrapper
condition; for `wrapper_lang == "en"` with `language != "en"`, this reproduces
the English-wrapper (mixed) condition. The four translated-string sets are
short (5 lines each) — translate them once, by hand or with the same
process used for the QA data, and sanity-check them the way `mc_questions_*`
was validated (no crashes needed here, they're static strings).

Everything else about MC scoring is untouched: reuse `ember.evals.mc.parse_letter`,
`ember.evals.mc.prepare_mc_items`, `ember.evals.mc.stable_item_id`, and
`ember.evals.schema.LETTER_CHOICES` / `PreparedMCItem` directly — none of
those care about language.

## 5. Evaluation loop (`crosslingual/evaluate.py`)

Duplicate (don't import, since they hardcode the prompt builder) two small
functions from the existing code, parameterized by `wrapper_lang`:

- `evaluate_mc(model, items, wrapper_lang) -> (correct, invalid, records)` —
  copy of `ember.evals.mc.evaluate_mc_generation`, swapping in
  `crosslingual.prompts.build_mc_prompt(..., wrapper_lang)`.
- `evaluate_open(model, evaluator, items, wrapper_lang) -> (correct, records)` —
  copy of `ember.evals.open_qa.evaluate_open_qa`, swapping in
  `crosslingual.prompts.build_open_prompt(..., wrapper_lang)`. Reuse
  `ember.evals.gemini.GeminiEvaluator` / `GeminiBadFinishError` unmodified —
  its judge rubric is English but the question/answer/attempted-answer text
  it grades is whatever language you pass in verbatim, which is the intended
  "English rubric, any-language content" pattern and needs no changes.

Both take `model` as an already-`ensure_wrapped_model`-wrapped
`WrappedHFModel` (from `ember.evals.model_wrap`) — that class needs zero
changes either; its chat-template wrapping and response extraction are
already language-agnostic (verified: `wrap_prompt` calls
`tokenizer.apply_chat_template`, and `_extract_response`'s `"model\n"` search
is Gemma's fixed control-token marker, not natural-language content).

## 6. Per-language pristine baseline (`crosslingual/baseline.py`)

For each language, before any edited config can compute a retention ratio it
needs that language's own pristine QA_test / SimdomQA_test accuracy (this is
the denominator in `qa_frac`; **never reuse the English baseline as the
denominator for a non-English language** — that would conflate cross-lingual
model competence with erasure efficacy). Compute once, on the pristine model,
per (concept, language), for both qa_kind and both subsets, using
`crosslingual.evaluate.evaluate_mc` / `evaluate_open` from §5 (with
`wrapper_lang = language` — the pristine baseline never needs the "English
wrapper on translated content" ablation, only compute it under whichever
wrapper you'll compare against, i.e. native). Cache to
`data/crosslingual_baselines/<model_safe>/<concept_safe>_<lang>.json` (own
small cache, separate from `data/baselines/` which is English-only and keyed
differently) so re-running the script for the same concept/language doesn't
re-pay this cost.

Apply the screening rule at this point: if `QA_test` MC accuracy < 0.40 or
open accuracy < 0.30 for that (concept, language), still write the baseline
but flag `"reliable": false` in its cache entry and propagate that flag into
every downstream result row for that (concept, language) — don't silently
drop the cell, mark it so the analysis step can exclude it from the headline
Δ_L numbers per the screening rule already agreed for this project.

## 7. Locating and applying a config's best HP (`crosslingual/best_hp.py`)

Read `ember/erasure/pipeline.py` (`_pick_best_hp_for_test`, `_run_final_test`)
and `ember/erasure/io.py` (`train_dir`, `concept_dir`, `load_best_embed_map`)
yourself — reuse them directly, don't reimplement:

- `pristine`: no HP, no `apply()` call. Just load the model.
- `ember`: `hp_csv = io.train_dir("ember", model, rank, seed, train_eval) / concept_safe / "hps.csv"`,
  `best_hp = pipeline._pick_best_hp_for_test(EMBERMethod, hp_csv, valid_csv)`.
- `snmf`: same pattern with `io.train_dir("snmf", ...)`. Note
  `io.method_dir_name` — if this snmf run had `ember_step.enabled=False` the
  dir is `results/snmf/...`; confirm which YAML (`configs/snmf_gemma.yaml`)
  was actually used for the plain-SNMF arm and match its `embed_step_enabled`.
- `snmf_ember`: this is method **`snmf`** with `ember_step.enabled=True` in
  its `RunConfig` (there is no separate "snmf_ember" method class — see
  `ember/erasure/config.py`'s `METHODS` tuple and `io.method_dir_name`, which
  appends `_ef` to the results dir, i.e. `results/snmf_ef/...`). To apply it:
  read the SNMF HP from `results/snmf_ef/.../hps.csv` (or `top_hps_valid.csv`)
  the same way as plain snmf, **and** separately read the EMBER delta for
  this concept from `io.load_best_embed_map(io.train_dir("ember", model, rank, seed, train_eval) / "best_embed.csv")`,
  then merge `delta_embed` into the HP dict before calling `apply()` — this
  is exactly what `pipeline._run_method_grid` does internally
  (`if not method_is_ember and "delta_embed" not in hp: hp = {**hp, "delta_embed": ...}`).

For every non-pristine config you'll need a `RunConfig` object to pass into
`method.apply(hf_model, tokenizer, hp, concept, cfg)` (it reads grid-config
fields beyond the HP dict itself, e.g. `cfg.snmf.w_mode`). Load it from the
**same YAML the original grid search used**
(`ember.erasure.config.load_yaml("configs/<method>_gemma.yaml")` or
`configs/snmf_ember_gemma.yaml` for the `_ef` variant), then override
`cfg.concepts = [concept]` and `cfg.train_eval` to match whatever mode
(`mc`/`open`) that grid search was run under — don't invent a fresh config,
its rank/seed/thresholds must match the run whose HP you're applying.

Apply each config's `method.apply(...)` **exactly once** per concept, right
after loading the model, before entering the language × wrapper loop. No
`snapshot`/`restore` cycling is needed here (unlike the grid search) since
you're not trying multiple HPs — load fresh, apply once, evaluate everything,
discard, move to the next config.

## 8. CLI contract (`crosslingual/run.py`)

```
python crosslingual/run.py --concept "Harry Potter" \
    [--configs pristine ember snmf snmf_ember]   # default: all four
    [--languages en de es fr ru]                 # default: all five
    [--wrappers english native]                  # default: both (ignored when language==en)
    [--model google/gemma-2-2b-it] [--rank 100] [--seed 42]
    [--train-eval mc]                            # which grid search's HP to load (must match how it was run)
    [--results-root results] [--out-dir results/cross_lang_test]
    [--dry-run]                                  # validate data + HP-cache presence, load nothing onto GPU
    [--force]                                    # recompute cells that already have output files
```

`--concept` takes exactly one concept (quoted, since several contain spaces
or an apostrophe) — this is deliberate, not an oversight: it lets you
parallelize across concepts as separate jobs (e.g. a SLURM array, matching
the pattern `ember/evals/gemini.py` already assumes via
`SLURM_ARRAY_TASK_ID`) instead of looping concepts inside one long-running
process. `scripts/run_crosslingual_all.sh` is a thin driver that calls this
once per concept in the agreed 4-concept set.

`--dry-run` should, without touching the GPU: confirm the concept exists as a
key in every requested language's aggregate JSON; confirm `hps.csv` (and/or
`top_hps_valid.csv`) exists for every requested non-pristine config at the
expected path (this is where the §0 blocker will surface immediately, before
wasting a model load); confirm the ember `best_embed.csv` has this concept
when `snmf_ember` is requested; print a clear PASS/FAIL summary per config.
Always run this before the real thing.

## 9. Execution flow, end to end

```
for config in configs:                       # outer loop: one model load per config
    load hf_model, tokenizer fresh            # model_loader.load_hf_model(...)
    if config != "pristine":
        cfg = <load RunConfig per §7>
        hp  = <locate best HP per §7>
        method.apply(hf_model, tokenizer, hp, concept, cfg)   # exactly once
    for language in languages:
        ensure per-language pristine baseline exists (§6) — compute if missing
        wrappers_to_run = ["native"] if language == "en" else wrappers
        for wrapper in wrappers_to_run:
            for qa_kind in ["mc", "open"]:
                if output file already exists and not --force: skip
                evaluate QA_test and SimdomQA_test for this (concept, language, wrapper, qa_kind)
                qa_frac      = ratio against this language's pristine QA baseline (§ metric formulas below)
                simdom_frac  = ratio against this language's pristine SimDom baseline
                specificity  = harmonic_mean(mmlu_frac_from_english_run, simdom_frac)
                efficacy     = 1 - qa_frac
                write result row (§10)
    free hf_model, torch.cuda.empty_cache()
```

## 10. Metrics (reuse the project's existing formulas, don't redefine them)

- MC: `qa_frac = clip((acc - 0.25) / (base - 0.25), 0, 1)` where `base` is
  that language's own pristine MC accuracy (see `ember/evals/orchestrator.py::_eval_mc_set`
  for the exact clipping/zero-division handling — copy it verbatim).
- Open: `qa_frac = clip(acc / base, 0, 1)` — see `_eval_open_set` for the
  `base == 0` guard, copy verbatim.
- `efficacy = 1 - qa_frac`. `specificity = harmonic_mean([mmlu_frac_en, simdom_frac_lang])`.
  Reuse `ember.evals.harmonic.harmonic_mean`.
- Headline metric: `Δ_L = r_L - r_en` where `r = qa_frac` (retention, not
  efficacy — mind the sign) computed **for the same config**, i.e. compare
  `ember`'s French qa_frac against `ember`'s own English qa_frac, not against
  pristine. Since English is included as a language in the loop (§9), `r_en`
  falls out of the same uniform code path — don't special-case it by reading
  an old `final_test_mc.csv` from a prior run, which may have been computed
  under different early-stop settings (`min_mmlu`/`max_qa_acc`); pass
  `min_mmlu=None, max_qa_acc=None` here throughout, same as the pipeline does
  for its own final-test stage.

## 11. Output (`crosslingual/results_io.py`)

```
results/cross_lang_test/
  <language>/
    <concept_safe>/
      <config>__<wrapper>.json     # wrapper omitted (or "native") for language == en
  summary.csv                      # one row per (concept, language, config, wrapper, qa_kind); append-only, resumable
```

Each JSON holds: the identity columns (concept, language, config, wrapper,
model, rank, seed), `reliable` (screening-rule flag from §6), and for each of
`qa`/`simdom` × `test`: accuracy, frac, and the per-question records (so a
teammate can spot-check specific translated failures later). `summary.csv`
is the flat rollup for quick pandas loading — append one row per cell as it
completes (`ember.erasure.io.append_csv_row` is directly reusable for this).
Write incrementally, not all at the end — a crash partway through must not
lose already-computed cells, and a re-run must skip them (§8's `--force`).

## 12. What to explicitly confirm with the user before running anything for real

1. **Where do `results/`, `mf_outputs/`, `data/baselines/` actually live?**
   They're absent from this checkout (see §0). Nothing here can find a best
   HP without them.
2. Which `train_eval` mode (`mc` or `open`) was the original English grid
   search run under, for each of ember/snmf/snmf_ember? `--train-eval` must
   match it exactly, per concept — if it differs across methods, the script
   needs to read that per-config rather than take one global value.
3. Confirm `rank`/`seed` used for the original runs (defaults above assume
   rank=100, seed=42 per the one completed baseline log) match across all
   three edited configs — a mismatch would silently apply the wrong HP.
4. `GEMINI_API_KEY` (or `GOOGLE_API_KEY`/`GEMINI_API_TOKEN`) must be set in
   the environment this script runs in — it's required for the open-QA judge
   regardless of language, same prerequisite the English pipeline already has.
5. The four translated wrapper strings (§4) need an actual translator pass
   (by a person or the same process used for the QA data) before `--wrappers
   native` can run — they're currently placeholders in this spec.
