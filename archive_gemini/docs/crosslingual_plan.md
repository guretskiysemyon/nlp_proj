# Part 1 — Cross-lingual leakage: analysis and run plan

> **2026-10-10:** this plan was written for, and executed in, the Gemini era
> (8 concepts; results in `archive_gemini/`). The design below (metric,
> screening, F1–F6) still holds. For re-running it with the Qwen-era HP picks,
> including the code changes needed and a cache bug found in the old run, see
> [`crosslingual_next.md`](../../docs/Part%201/crosslingual_next.md).

Scope: proposal question (1), *"does concept information re-emerge when prompts are
translated into other languages, and does this differ between embedding-based and
MLP-based erasure?"*

Decided scope: **Gemma-2-2b-it only**, methods **`ember`, `snmf`, `snmf+ember`**,
**4 concepts**, languages **en / de / es / fr / ru**, prompt scaffolding tested
**both** English and native.

---

## 0. The experiment at a glance

The design is a **before/after over one shared item set**: the same held-out questions,
in the same five languages, scored once on the pristine model and once after a single
English-tuned edit. The metric is the ratio between the two.

```
STAGE 1 — English only, already built, unchanged
--------------------------------------------------------------------------------
   English Wikipedia          mf_outputs/                 grid search
   S_C + S_N sentences  -->   V' . F . G . labels   -->   train_mc (en)
                                                                |
                                                         best d |
STAGE 2 — one model load, one edit                              |
--------------------------------------------------------------------------------
                        gemma-2-2b-it  (loaded once)            |
                                 |                              |
                 +---------------+---------------+              |
          no edit |                              | <------------+
                  v                      apply best d . ONCE
      +----------------------+                   v
      | PHASE A . PRISTINE   |       +----------------------+
      | en de es fr ru       |       | PHASE C . EDITED     |
      | QA_test + Simdom_test|       | en de es fr ru       |
      +----------+-----------+       | the same items       |
                 v                   +----------+-----------+
              base_L                            v
                 |                           post_L
                 |                              |
STAGE 3 — compare
--------------------------------------------------------------------------------
                 +--------------+---------------+
   token overlap                v
   (CPU, no model) --->  r_L = (post_L - .25) / (base_L - .25)
   -> coverage_L         D_L = r_L - r_en                <-- the answer
                         then correlate D_L against coverage_L
```

**Read as a sequence:**

1. **Nothing in stage 1 changes.** The factorization and the grid search are English by
   construction; the grid search runs on `train_mc` and yields one `best δ` (or HP row)
   per concept per method. `ember` is already done; `snmf` and `snmf_ember` still need it.
2. **Load the model once.** Every language shares one model state — this is what keeps
   stages 2 and 3 cheap.
3. **Phase A runs first, on the untouched model**, because its output is the denominator.
   Five languages × two formats × the concepts in scope. Then apply the §4 screen and
   drop any cell where the pristine model was already at chance.
4. **Apply the edit exactly once.** Not per language, not re-tuned per language (F5).
5. **Phase C repeats Phase A verbatim** on the edited model — same items, same order, same
   scaffold. The only thing that differs between the two passes is the edit.
6. **Everything grey in the diagram is measurement, everything on the edit path is the
   intervention.** MMLU and Alpaca sit *inside* the English pass only, and are fed forward
   to the other languages via `precomputed_metrics` (F4).
7. **Stage 3 joins them.** `Δ_L` is the answer to Q1; `coverage_L` from the free CPU-side
   token-overlap job is the explanation for it.

Two variants hang off the same skeleton at no structural cost: the **scaffold ablation**
re-runs Phase C with a native-language prompt wrapper, and the optional **cross-lingual
relearning** (§10) inserts an English relearning step between Phase B and Phase C.

---

## 1. The one-paragraph version

The erasure itself needs **no changes at all**. EMBER's edit is defined entirely by
English data (`concept_sentences.json` → V′ → factorization → token labels), and that
is precisely what we are testing. Everything language-dependent lives on the
*evaluation* side and is confined to two loader functions plus the baseline cache key.
So the work is: (a) add a `lang` axis to the eval path, (b) compute **per-language
pristine baselines** (non-negotiable — the metric is a ratio against them),
(c) tune in English exactly as the paper does, then evaluate that one edited model in
every language. The compute trick is that **MMLU and Alpaca never need translating or
re-running per language**, which removes ~3× of the naive cost.

---

## 2. How the pipeline works today

`run_erasure.py` → `pipeline.run(cfg)`, four stages:

| Stage | What happens | Where |
|---|---|---|
| 0 | Load HF model; compute/read baselines for `train_<mode>`, `test_mc`, `test_open` | `evals/baselines.py` |
| 1 | Resolve EMBER δ per concept (skipped when `method: ember`) | `pipeline._resolve_ember_step` |
| 2 | Method HP grid; one row per cell → `hps.csv` | `pipeline._run_method_grid` |
| 3 | Re-evaluate top-K with Alpaca → `top_hps_valid.csv` | `pipeline._run_validate_topk` |
| 4 | Apply best HP, eval `test_mc` **and** `test_open`, then relearning | `pipeline._run_final_test` |

Data enters at exactly two functions:

```python
# ember/local_datasets.py
load_mc_qa_items(set_name, concept)      # -> DATA_DIR/"mc_questions.json"
load_open_qa_examples(set_name, concept) # -> DATA_DIR/"open_questions.json"
```

Metrics (`evals/orchestrator.py`) are all **fractions of the pristine model's score**:

```
mmlu_frac  = (mmlu_acc - .25) / (mmlu_base - .25)
qa_frac    = (qa_acc  - .25) / (qa_base  - .25)      # MC,  clipped to [0,1]
qa_frac    =  qa_acc / qa_base                        # open
efficacy   = 1 - qa_frac
specificity= HM(mmlu_frac, simdom_frac)
harmonic   = HM(efficacy, specificity)
```

`qa_base` comes from `data/baselines/<model>/baseline_<set>.json` →
`meta.per_concept[<concept>]`.

---

## 3. What is and isn't language-dependent

**F1 — The erasure is English by construction, and needs no edit.**
`V′` = unique token ids occurring in `concept_sentences.json` ∪ `neutral_sentences.json`
(English Wikipedia). Token labels come from English co-occurrence. `embed_edit.py`
edits only tokens labelled `== concept`. Nothing there is language-aware. This *is*
the paper's stated limitation (§Limitations, "Language coverage"); Q1 measures it.

**F2 — The language surface is tiny.** Only the two loaders above, plus the baseline
cache key. That is why this stays architecturally simple.

**F3 — Per-language baselines are mandatory.** `qa_frac` divides by the pristine
model's accuracy *in that language*. Gemma-2-2b knows far less in Russian than in
English. Reuse the English baseline and a Russian result of 0.35 is unreadable: it
could mean "erased" or "never knew it". Only the per-language ratio separates those.

**F4 — MMLU and Alpaca must NOT be translated, and must NOT be re-run per language.**
They measure a property of the *edited model* (general capability, coherence), not of
the eval language. Measure once, in English, per configuration; reuse for every
language via `precomputed_metrics`. MMLU test is **1000 questions** and Alpaca is
**150 generations + 300 Gemini calls** — these dominate every eval call, so this is
the single biggest lever.
*Caveat for the writeup:* specificity in language L is then "English MMLU +
in-language SimDom". Defensible, because SimDom carries the in-language signal — but
say so explicitly.

**F5 — Do not re-tune per language.** Tune in English (MC, as in the paper), evaluate
that model everywhere. Re-tuning per language would confound *"does it leak"* with
*"can you tune the leak away"*, and costs 5×.

**F6 — Load the model once, apply the edit once, loop languages inside.** All
languages share the same model state, so the loop order falls out:
pristine baselines for all langs → apply edit → evaluate all langs.

---

## 4. The measurement

Per (concept `C`, language `L`, format `F ∈ {mc, open}`):

- `base` = pristine accuracy (per-language baseline)
- `post` = post-erasure accuracy
- **retention** `r = qa_frac` (the formulas in §2)
- **efficacy** `= 1 − r`
- **leakage gap** `Δ_L = r_L − r_en` ← *this is the headline number.*
  `Δ_L > 0` ⇒ the concept survives erasure better in L than in English.

**Screening rule (apply before anything else).** If `base` is near chance the
denominator is noise. With n = 50 the SE on an accuracy is ≈ 0.07, so a MC base of
0.35 gives a denominator of 0.10 ± 0.07 — meaningless.

> Drop any (concept, language) cell with **MC `base` < 0.40** or **open `base` < 0.30**,
> and report it as *"model did not know C in L"* rather than as erasure success.

This is why Step 2 below comes before every other GPU job: the screen may tell you to
drop Russian, or to swap a concept, *before* you spend anything else.

Always report raw `qa_acc` next to `qa_frac`, and report `qa_invalid` per language
(the MC letter-parse failure rate rises in non-English and can masquerade as erasure).

---

## 5. Code changes

One new parameter, `lang`, threaded through four files, plus one new entrypoint.
Everything defaults to `"en"`, so **existing English results and caches stay valid**.

### 5.1 `ember/local_datasets.py`

```python
QA_LANG_DIR = DATA_DIR / "qa_lang"

def _qa_path(kind: str, lang: str) -> Path:        # kind: "mc_questions"|"open_questions"
    if lang in (None, "en"):
        return DATA_DIR / f"{kind}.json"           # canonical English, untouched
    return QA_LANG_DIR / f"{kind}_{lang}.json"     # mc_questions_ru.json, ...

def load_mc_qa_items(set_name, concept=None, lang="en"): ...
def load_open_qa_examples(set_name, concept=None, lang="en"): ...
```

On-disk layout (as it now stands):

```
data/mc_questions.json            data/open_questions.json          <- English, canonical
data/qa_lang/mc_questions_en.json data/qa_lang/open_questions_en.json  <- identical copy, unused
data/qa_lang/mc_questions_de.json data/qa_lang/open_questions_de.json
data/qa_lang/mc_questions_es.json ...  es / fr / ru
data/qa_lang/translated/<lang>/{mc,open}_questions/<Concept>.json     <- per-concept sources
```

Two more things inside `load_mc_qa_items`:

1. **`.strip()` options and `correct_answer`** — fixes the two broken items in §6.
   No-op for English (verified), so English ids/shuffles are untouched.
2. Optionally add an `item_index: int` field to `MCQAItem`/`OpenQAItem`.
   *Safe*: `stable_item_id` keys on `("concept","subset","split","question","answer",
   "correct_answer")` + options, so a new field changes no id and no option shuffle.
   Do **not** add a field named `question_id` — that one *is* consumed by
   `stable_item_id` and would invalidate the English `correct_ids_*.json` caches.

### 5.2 `ember/evals/baselines.py`

```python
def compute_or_read_baseline(model, set_name, *, lang="en", concepts=None, ...):
    # cache path: baseline_<set>.json           when lang == "en"   (back-compatible)
    #             baseline_<set>__<lang>.json    otherwise
```

- pass `lang` into the loaders inside `_compute_qa_open` / `_compute_qa_mc`;
- **restrict to `concepts`**: today `_compute_qa_*` loads all 18 concepts
  (900 questions/set). With 4 concepts that is 200. **4.5× saving on every baseline.**
- **merge, don't replace**: the current "missing concepts → recompute" branch
  overwrites the whole file. With concept restriction you must merge new
  `per_concept` entries into the existing JSON, or concept #2's run wipes concept #1's.
- `get_baselines_for_mode(..., lang="en")`: pass `lang` to the qa/simdom baselines,
  and always call mmlu/alpaca with `lang="en"` so they hit the existing cache.

### 5.3 `ember/evals/orchestrator.py`

`evaluate_model_for_mode(..., lang="en")` → forward to `_eval_mc_set` / `_eval_open_set`
→ the loaders. Nothing else changes; `precomputed_metrics` + `eval_mmlu=False` is an
existing, already-exercised path (`_run_validate_topk` uses it).

### 5.4 `ember/erasure/eval.py`

Thread `lang` through `get_baselines()` and `evaluate_model()`. Pure pass-through.

### 5.5 New `ember/run_crosslingual.py`

Shaped like `run_eval.py`. Reuse `pipeline._pick_best_hp_for_test` rather than
reimplementing best-HP selection (it handles `ember` reading `hps.csv` vs `snmf`
reading `top_hps_valid.csv`).

```python
cfg   = parse_args()                       # + --langs de es fr ru  + --scaffold en|native
langs = ["en"] + cfg.langs
model, tok = model_loader.load_hf_model(...)          # ONCE

# --- Phase A: pristine baselines (model untouched) ---
B = {}
for lang in langs:
    for mode in ("test_mc", "test_open"):
        B[lang, mode] = eval_mod.get_baselines(model, mode, lang=lang,
                                               required_concepts=cfg.concepts)

# --- Phase B: apply the English-tuned edit, ONCE ---
best_hp = pipeline._pick_best_hp_for_test(method, hps_csv, top_hps_valid_csv)
method.apply(model, tok, best_hp, concept, cfg)

# --- Phase C: evaluate every language on that one edited model ---
en_metrics = None
for lang in langs:
    for mode in ("test_mc", "test_open"):
        m, recs = eval_mod.evaluate_model(
            model, baselines=B[lang, mode], concept_name=concept, mode=mode,
            lang=lang,
            eval_mmlu   = (lang == "en"),          # F4: measure once
            eval_alpaca = (lang == "en"),
            precomputed_metrics = None if lang == "en" else en_metrics,
            min_mmlu=None, max_qa_acc=None,        # never early-stop in this script
            tokenizer=tok)
        append_row(crosslingual_csv, {**ids, "lang": lang, "mode": mode,
                                      "scaffold": cfg.scaffold, **m})
        dump_json(records_dir / f"{lang}_{mode}.json", recs)   # for failure analysis
```

**Set `min_mmlu=None, max_qa_acc=None`.** The orchestrator's early-stop would
otherwise zero out a language's metrics and silently skip its SimDom pass.

**Dump the per-question records.** The pipeline currently throws them away
(`metrics, _ = ...`). They are what the proposal's *"analyse qualitatively failure
cases where concepts reappear"* needs. Item order is preserved by both evaluators, and
all 72 (concept, set) cells have identical counts across languages (verified), so
**record `i` in de pairs with record `i` in en** — per-question paired analysis works
with no further changes.

### 5.6 Scaffolding ablation

`evals/mc.py::build_mc_prompt` — add `lang="en"` selecting from a 5-language dict of
the two instruction strings ("Choose the single best answer." / "Answer only with one
letter…"). Same for the `f"Question: {q}\nAnswer:"` prefix in `open_qa.py`.
Run **`scaffold="en"` as the primary condition** (isolates the variable: the concept
is *named* in L, without confounding it with a 2B model's instruction-following in L),
and `scaffold="native"` as the second pass. Phase C is the cheap phase, so doubling it
is fine.

### 5.7 Gemini judge

Add one line to `OPEN_QA_PROMPT`:

> *"The question and answers may be in a language other than English. Judge semantic
> correctness only; do not penalise an answer for being in a different language than
> the reference."*

Without it you risk the judge marking a correct German answer wrong for language
mismatch. Then **validate the judge**: hand-check ~30 judged items per language and
report the agreement rate — that is a real methodological point for the report, and
Semyon can check the Russian directly.

---

## 6. Data fixes — done

All repairs are applied, in both the flat aggregates and the per-concept source files
under `data/qa_lang/translated/`, so the defects do not return if the aggregates are
regenerated. English (`data/*.json`) was **not** touched — it had no defects, and the
cached English baselines depend on its exact strings.

- `scripts/fix_qa_lang.py` — the one-off repair, with every substitution listed inline.
- `scripts/validate_qa_lang.py` — the pre-flight check. Run it before any GPU job; it
  mirrors the loader's filters and calls the **real** `prepare_mc_items`, so a latent
  `AssertionError` surfaces locally instead of half an hour into a Slurm job. Exits
  non-zero on failure. Currently: **PASS**, all five languages, 3600 items each,
  72 (concept, set) cells matching English exactly.

### What was wrong

**1 — Wrong option count (would crash, non-deterministically).** `prepare_mc_items`
shuffles all options and then keeps `options[:4]`, so an item with five options crashes
only when the correct answer happens to shuffle out of the top four.

| lang | concept / set / idx | was | now |
|---|---|---|---|
| fr | Harry Potter / QA_train / 42 | 5 options, correct = `'匈牙利和角龙'` (a hallucinated Chinese string) | 4 options, correct = `'Hongrois à pointes'` |

**2 — Duplicate options (19 items: de 5, es 3, fr 7, ru 4).** Translation collapsed two
distinct English options into one string. Where the *correct* answer was the duplicated
one (13 of the 19), the grading was actively wrong: `prepare_mc_items` assigns
`correct_letter` on the **last** match, so a model picking the first copy scored as
incorrect. In each case the colliding **distractor** was given a distinct rendering;
the correct answer was never changed.

Examples: de `Racket`/`Bat` both → `Schläger` (bat is now `Baseballschläger`);
fr `Shamrock`/`Clover` both → `Trèfle` (clover is now `Trèfle à quatre feuilles`);
ru `Crookshanks`/`Scabbers` both → `Живоглот` (Scabbers is now `Короста`).
Full list in `scripts/fix_qa_lang.py`.

> One judgement call worth reviewing: fr Gun/SimdomQA_train/16, English
> `Port`/`Starboard`/`Larboard`/`Aft`. *Port* and *Larboard* are the same word in French
> (`Bâbord`), so the distractor became `Proue` (bow) — a valid wrong nautical answer, but
> not a translation of *Larboard*. Every other substitution is a faithful rendering.

**3 — Untrimmed whitespace (30 items).** Two of these (` Kürbiskuchen`, ` Tombuctú`)
put `correct_answer ∉ options` and would have crashed outright. All stripped.

**4 — One mojibake.** `Pekannubkuchen` → `Pekannusskuchen`. Checked and isolated: the
German files contain 770 correct `ß` characters, so this is a single glitch, not a
systematic encoding failure.

### Known and harmless

One question is duplicated within `Culture of Greece / QA_train` in English and in every
translation (*"What is Katharevousa…"*). It is graded twice; n = 49 unique there.

## 7. Run plan, in cost order

| # | Step | Where | Cost |
|---|---|---|---|
| 0 | Data fixes + re-run the validation script | laptop | minutes |
| 1 | **Token-overlap analysis** (§8) | CPU / login node | minutes |
| 2 | **Pristine per-language baselines**, 4 concepts, `test_mc`+`test_open`, 5 langs | GPU | ~1 job |
| 3 | English grids for `snmf` and `snmf_ember` (`ember` already done) | GPU | the expensive part |
| 4 | Cross-lingual eval: 3 configs × 5 langs × 2 formats | GPU | cheap |
| 5 | Scaffold ablation (`native`) on the same edited models | GPU | = step 4 |
| 6 | Analysis + figures | laptop | — |
| 7 | *Optional, high value:* cross-lingual **relearning** (§10) | GPU | 1 relearn + 5 cheap evals per config |

**Step 1 gates step 2, and step 2 gates everything else.** Step 1 costs nothing and
predicts the result; step 2's screen tells you whether the ratios will be readable.

### Cost accounting (why steps 4–5 are cheap)

Fixed per eval call, language-independent, measured **once per configuration**:
MMLU test = **1000** questions (batched 8), Alpaca = **150** generations × 200 tokens
+ **300** Gemini calls.

Variable per concept per language: 50 QA + 50 SimDom = **100** MC generations
(4 tokens, unbatched) or **100** open generations (200 tokens, batched 64) + **100**
Gemini calls.

- Step 2, per new language: 400 MC + 400 open generations + 400 Gemini. No MMLU, no
  Alpaca. × 4 languages.
- Step 4, per config: one full English pass, then per non-English language
  4 concepts × 200 = 800 generations + 400 Gemini. × 4 languages = 3200 gens + 1600
  judge calls.
- **Without** the F4 skip and the concept restriction, each config would additionally
  drag 8 × (1000 MMLU + 150 Alpaca + 300 judge) ≈ 9,200 generations and 2,400 judge
  calls. The two optimisations together cut roughly **3×** off each configuration.

---

## 8. The free result: token-overlap analysis

Do this **first**. CPU only, no model load — just the tokenizer and the concept's
`mf_outputs/` pickle (already downloaded on the cluster).

`embed_edit.concept_edit_tokens()` already returns exactly `(token_string, ‖c_i‖)` for
every token EMBER edits. For token *ids* (better for matching), write ~20 lines reusing
`features.load_embedding_payload` + `embed_edit._build_concept_token_set` +
`_eligible_tids_rows`.

For each concept `C` and language `L`, tokenize all `QA_test` questions (+ MC options)
and report:

- **type coverage** = |unique prompt tokens ∩ E_C| / |E_C|
- **token coverage** = fraction of prompt token *occurrences* falling in E_C
- **edit mass** = Σ ‖c_i‖ over those occurrences, normalised

Prediction: `coverage_en ≫ de/es/fr ≫ ru ≈ 0`, and the leakage gap `Δ_L` should track
`−coverage_L`.

> **A scatter of `Δ_L` against `coverage_L` over all (concept, language) cells is the
> money figure.** It turns "erasure leaks" into "erasure leaks *exactly to the extent
> that the edited tokens are absent from the prompt*" — which is the mechanistic claim
> the proposal promises about tokenization and language-specific representations.

---

## 9. Concept selection (4, chosen for tokenization contrast)

| Concept | Regime | Why |
|---|---|---|
| **Harry Potter** | mixed | de/es keep *Harry Potter*, *Hogwarts*; **fr renames** (Poudlard, Rogue, Serpentard); **ru transliterates** to Cyrillic (Гарри Поттер, Хогвартс). Four regimes in one concept — the best single concept, and you already have its English run. |
| **World War II** | fully translated | *Zweiter Weltkrieg / Segunda Guerra Mundial / Seconde Guerre mondiale / Вторая мировая война.* Almost no shared surface form ⇒ predicts **maximum** leakage. |
| **Halloween** | loanword | Identical string in de/es/fr, Cyrillic in ru ⇒ predicts **minimum** leakage in Latin script. A clean control. |
| **Uranium** | cognate | *Uran / uranio / uranium* — partial subword overlap. The intermediate regime, and mildly safety-flavoured. |

Optional 5th: **Nazism** (cognates *Nazismus / nazismo / nazisme / нацизм*, and the
strongest safety framing). **But** flag the risk: the Gemini judge retries a refusal
3× then defaults the item to *incorrect*, so judge refusals on a sensitive concept
would show up as successful erasure. Uranium avoids that; add Nazism only if you'll
audit its refusal rate.

---

## 10. Optional step 7 — cross-lingual relearning (highest value per GPU-hour)

EMBER's *headline* claim is robustness to relearning, not raw efficacy. So the sharper
version of Q1 is: **is the robustness gain also English-only?** `_run_final_test`
already runs relearning on English paragraphs; all that's missing is a per-language
eval afterwards. Relearn once (English `relearn_paragraphs.json`), then run the same
cheap 5-language × 2-format eval on the relearned model — 5 cheap evals on top of a
relearning run you're doing anyway.

If the concept was never suppressed in ru/fr to begin with, or if English relearning
restores it everywhere, then the paper's robustness result is language-bounded — a
much stronger finding than an efficacy gap alone.

---

## 11. Risks and open questions

1. **Near-chance Russian baselines.** A 2B model may simply not know these concepts in
   ru, breaking the ratio. Mitigation: the §4 screen; report raw accuracies alongside
   fracs; consider dropping ru or noting it needs Llama-8B.
2. **Judge reliability across languages**, plus the refusal→incorrect bias on sensitive
   concepts (§9). Mitigate with the prompt line in §5.7 and the 30-item hand-check.
3. **Rising MC invalid rate** in non-English (model answers in prose, not a letter).
   Already tracked as `qa_invalid` — report it; it is the main argument for the
   English-scaffold primary condition.
4. **Translation quality.** MT artifacts are present (§6). Spot-check ~20 items per
   language with a speaker.
5. **Specificity is a mixed measure** (English MMLU + in-language SimDom) — state it.
6. **`evaluate_mc_generation` is unbatched** (one question at a time). At 100 questions
   per concept per language it is not the bottleneck; leave it alone, since batching
   with left-padding can perturb greedy decoding and would break comparability with the
   cached English baselines.

---

## 12. Where Part 2 (multilingual extension) would plug in

Not part of this plan, but the design above doesn't block it. Q2 needs a *different*
V′ and a *different* factorization: draw `S_C` from Wikipedia in multiple languages →
new `concept_sentences.json` → re-run `train_mf_features.py` (embedding track only,
`--skip-mlp`) → `interpret_features.py` → a new `mf_outputs/` tree. The erasure and
eval code are untouched; only the features change. Keep it under a separate
`--rank`/`--seed`/output root so the English features stay intact for comparison.
