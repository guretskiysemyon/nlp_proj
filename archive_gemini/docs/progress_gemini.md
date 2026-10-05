# Cross-lingual erasure — progress log

**Last updated:** 2026-09-15
**Status:** ✅ **8 concepts complete end-to-end** (8 × 4 configs × 5 languages × MC =
288/288 cells). Batch 1 (Harry Potter, World War II, Halloween, Uranium) and batch 2
(Golf, Baseball, Culture of Greece, Gambling) are both finished. Batch 2 was chosen
specifically to break a confound in batch 1 — **it worked** (§4.5). Remaining work is
**analysis**, not compute — see §8.

This file is the **execution record**: what was actually run, what came out, what broke
and why, and what the next person should do. It is deliberately separate from:

| File | Purpose |
|---|---|
| [`docs/crosslingual_plan.md`](crosslingual_plan.md) | The **design**: why this experiment, what it should measure, screening rules, concept selection. Read this first. |
| [`crosslingual_implementation_prompt.md`](../crosslingual_implementation_prompt.md) | The **implementation spec** the `crosslingual/` package was built from. |
| **this file** | The **execution record**: what ran, results, gotchas. |

---

## 1. The question being answered

From the proposal, via `docs/crosslingual_plan.md`:

> Does concept information re-emerge when prompts are translated into other languages,
> and does this differ between embedding-based and MLP-based erasure?

The paper ([`docs/Main Article.pdf`](Main%20Article.pdf)) predicts the answer in its own
Limitations section: EMBER edits *English* token embeddings, so "erasure applied to
English tokens would not suppress the same concept when accessed through tokens in other
languages." This experiment measures how much.

---

## 2. Design in one paragraph

For one concept, apply each erasure config **once** (the edit is language-independent),
then evaluate the resident model against QA questions in five languages. The edit's
hyper-parameters are **not re-tuned** — they are read back from the English grid search
that the original pipeline already ran. The headline number is

```
Δ_L = qa_frac(language L) − qa_frac(English)      for the SAME config
```

where `qa_frac` is retention (1.0 = concept fully intact, 0 = erased to chance).
A **positive Δ_L means the erasure leaked** — the concept survives better in that
language than in English.

Matrix per concept: 4 configs × 5 languages × (1 wrapper for `en`, 2 for the rest)
× qa_kind = 36 cells at `--qa-kinds mc`.

---

## 3. What has been run

All on `google/gemma-2-2b-it`, `rank=100`, `seed=42`, `--train-eval mc`.

### 3.1 Phase 1 — hyper-parameter search (the paper's Stages 1–3)

Ran the project's **existing, unmodified** pipeline (`ember.run_erasure`), stopped after
the validate stage. Nothing under `ember/` was edited.

All **24** (concept × arm) grids are complete and verified at full size:

| Arm | Results dir | Grid cells | Validate | Concepts done |
|---|---|---|---|---|
| `ember` | `results/ember/.../train_mc/<C>/` | 8 | n/a¹ | all 8 |
| `snmf` | `results/snmf/.../train_mc/<C>/` | 144 | 15 | all 8 |
| `snmf_ember` | `results/snmf_ef/.../train_mc/<C>/` | 144 | 15 | all 8 |

¹ EMBER overrides `writes_topk_csv()` to `False`, so it has no validate stage; its best
δ is picked directly from `hps.csv` and mirrored into `best_embed.csv`.

**Selected hyper-parameters** (`layers` shown as `lo-hi`; `snmf` always has
`delta_embed=0.0` because plain SNMF applies no embedding edit, while `snmf_ember`
inherits ember's δ via `best_embed.csv`):

| Concept | ember δ (tokens edited) | snmf in / out | snmf_ember in / out |
|---|---|---|---|
| Harry Potter | 10 (39) | 7@0-12 / 1@0-8 | 10@0-8 / 1@0-8 |
| World War II | 5 (36) | 10@0-25 / 1@13-25 | 10@0-25 / 1@0-8 |
| Halloween | 10 (80) | 10@0-25 / 7@13-25 | 7@0-12 / 10@13-25 |
| Uranium | 100 (70) | 10@0-25 / 1@0-8 | 4@0-12 / 10@13-25 |
| Golf | 10 (70) | 10@0-12 / 10@9-17 | 1@0-8 / 1@9-17 |
| Baseball | 100 (79) | 1@0-12 / 10@0-8 | 4@0-25 / 7@13-25 |
| Culture of Greece | **200** (52) | 1@0-12 / 10@0-8 | 1@0-8 / 1@0-8 |
| Gambling | 100 (69) | 7@0-12 / 4@0-8 | 1@0-25 / 7@0-8 |

`snmf_ember` inherits ember's δ via `best_embed.csv`; plain `snmf` always has
`delta_embed=0.0`.

⚠️ **The grid's chosen δ varies by 40×** (5 → 200). Culture of Greece at δ=200 and
Baseball/Gambling/Uranium at δ=100 are far above Harry Potter's and World War II's 5–10.
This is the grid's own selection, not a bug, but δ is *not* a controlled variable across
concepts — keep that in mind before attributing a cross-concept difference to the
concept itself. `n_tokens_edited` is a direct input to the `coverage_L` analysis (§8.2).

### 3.2 Phase 2 — cross-lingual evaluation

| Output | Contents |
|---|---|
| `results/cross_lang_test/summary.csv` | **288 rows** = 8 concepts × 4 configs × 9 cells. One per (concept, language, config, wrapper, qa_kind). **One shared file**; new concepts/configs append rows. Verified: no duplicates, no NaNs. |
| `results/cross_lang_test/<lang>/<Concept>/<config>__<wrapper>.json` | Per-question records (prompt, model output, correctness) for spot-checking translated failures. **288 files.** |
| `data/crosslingual_baselines/google_gemma-2-2b-it/<Concept>_<lang>.json` | Per-language pristine baselines — the **denominators**. **40 files** (8 concepts × 5 langs). |

⚠️ **Harry Potter's baselines contain both MC and open; the other seven are MC-only.**
Harry Potter was baselined before the switch to `--qa-kinds mc` (§6.4). If you ever run
`--qa-kinds mc open`, the other seven concepts will need their open baselines computed
(a GPU + Gemini job), and Harry Potter's `reliable` flag will flip to `false` for
de/es/fr/ru because its open baselines are 0.04–0.24 (§6.3).

### 3.3 SLURM job history

**Harry Potter** (Sep 10–12) — includes both failures and their fixes:

| Job | What | Elapsed | Outcome |
|---|---|---|---|
| 871643 | English baselines, all 18 concepts | 0:22 | ✅ (pre-existing) |
| 876141 | Phase 2, pristine+ember | 0:32 | ❌ CUDA OOM — see §6.1 |
| 876536 | Phase 2, pristine+ember (after fix) | 0:33 | ✅ |
| 877584 | Phase 1, snmf | 2:18 | ❌ CUDA OOM in validate — see §6.2. Grid survived. |
| 880463 | Phase 1, snmf (after fix) | 2:16 | ✅ |
| 881287 | Phase 1, snmf_ember | 4:32 | ✅ |
| 882471 | Phase 2, snmf + snmf_ember | 0:32 | ✅ |

**World War II / Halloween / Uranium** (Sep 12) — all ✅, **zero failures**, run largely in
parallel via `scripts/run_erasure_grid.sh`:

| Concept | ember | snmf | snmf_ember | Phase 2 eval |
|---|---|---|---|---|
| World War II | 884339 / 882967 | 884340 (3:24) | 884341 (3:30) | 884342 (0:25) |
| Halloween | 884343 (1:13) | 884344 (3:30) | 884345 (4:18) | 884346 (0:30) |
| Uranium | 884347 (1:20) | 884348 (3:24) | 884349 (4:31) | 884350 (0:30) |

**Batch 2** (Sep 15) — Golf, Baseball, Culture of Greece, Gambling. Submitted in one
command via `scripts/submit_all.sh`. Two infrastructure failures, both mine, both fixed
(§6.8, §6.9); **no pipeline or science bug**, and no GPU time lost to either. Grids:
895126–895141 + 895319–895321. Evals: 896156/57/58/60.

**Per-concept GPU cost** (from each arm's `timing.json`; Harry Potter's grid predates
the timing instrumentation):

| Concept | ember | snmf | snmf_ember | total |
|---|---|---|---|---|
| World War II | 1.18h | 3.28h | 3.41h | ~7.9h |
| Halloween | 1.21h | 3.40h | 4.19h | ~8.8h |
| Uranium | 1.31h | 3.30h | 4.43h | ~9.0h |
| Golf | 1.29h | 3.83h | 4.34h | ~9.5h |
| Baseball | 1.22h | 3.87h | 3.91h | ~9.0h |
| Culture of Greece | 1.21h | 4.14h | 3.47h | ~8.8h |
| Gambling | 1.18h | 3.27h | 3.37h | ~7.8h |

≈ **9 GPU-hours per concept**, ~85% of it the two 144-cell SNMF grids. Phase 2 (the
actual cross-lingual measurement) is only ~0.5h per concept. Total for 8 concepts:
roughly **70 GPU-hours**.

---

## 4. Results — all 8 concepts, MC

### 4.1 Pristine per-language accuracy — the denominators

The **unedited** model. The single most important table for interpreting everything else:
every `qa_frac` is a ratio against the cell's own language column here.

| Concept | en | de | es | fr | ru |
|---|---|---|---|---|---|
| Harry Potter | **0.88** | 0.62 | 0.58 | 0.56 | **0.30** ✗ |
| World War II | 0.74 | 0.68 | 0.72 | 0.64 | 0.52 |
| Halloween | 0.62 | 0.54 | 0.52 | 0.50 | 0.46 |
| Uranium | 0.60 | 0.52 | 0.48 | **0.62** | 0.58 |
| Golf | 0.80 | 0.66 | 0.68 | 0.66 | 0.46 |
| Baseball | 0.64 | 0.56 | 0.60 | 0.50 | 0.52 |
| Culture of Greece | 0.68 | 0.46 | 0.68 | 0.58 | 0.66 |
| Gambling | 0.82 | 0.78 | 0.64 | 0.72 | 0.56 |

(MC `QA_test` accuracy. ✗ = below the 0.40 screening floor.)

Two things to notice:

- **Harry Potter's Russian is the ONLY screened-out cell in all 40.** Its 0.30 sits just
  above the 0.25 chance floor. Every other (concept, language) passes, including
  Baseball's Russian at 0.52 — which had been flagged as the main screening risk of
  batch 2 (US/Japan-centric sport, estimated ~0.46). It held.
- **Harry Potter has by far the steepest English→other-language drop** (0.88 → 0.56–0.62,
  and 0.30 in ru). Everything else is much flatter. Harry Potter is the *atypical*
  concept, not the representative one — a conclusion drawn from the original
  single-concept run would have been wrong.

### 4.2 English retention after erasure (`qa_frac_en`)

This is the denominator of every Δ_L, and it varies enormously:

| Concept | ember | snmf | snmf_ember |
|---|---|---|---|
| Harry Potter | 0.587 | 0.619 | **0.238** |
| World War II | **0.878** | **0.143** | 0.469 |
| Halloween | 0.514 | 0.514 | 0.405 |
| Uranium | 0.257 | 0.486 | **0.000** |
| Golf | 0.527 | 0.164 | 0.455 |
| Baseball | **0.795** | 0.385 | 0.333 |
| Culture of Greece | 0.349 | 0.349 | 0.302 |
| Gambling | 0.368 | 0.333 | 0.158 |

> ⚠️ **Read Δ_L with this table open.** Δ_L is a *difference of ratios*. When English
> retention sits near a floor or ceiling the delta compresses or inflates: World War II
> `ember` was barely erased at all (0.878), so its near-zero Δ_L means "nothing was
> erased anywhere", not "the erasure transferred". Conversely Uranium `snmf_ember`
> erased English completely (0.000), so any non-zero non-English retention becomes a
> large positive Δ_L by construction.

### 4.3 Δ_L — the headline (native wrapper, reliable cells only)

Positive = **erasure leaked** (concept survives better in that language than in English).

| Concept | config | de | es | fr | ru |
|---|---|---|---|---|---|
| **Halloween** | ember | +0.280 | +0.264 | +0.246 | **+0.486** |
| | snmf | **+0.418** | +0.042 | +0.166 | **+0.486** |
| | snmf_ember | +0.112 | +0.002 | +0.035 | **+0.595** |
| **Harry Potter** | ember | −0.128 | −0.133 | +0.155 | ✗ |
| | snmf | −0.106 | +0.139 | −0.135 | ✗ |
| | snmf_ember | +0.113 | −0.087 | +0.117 | ✗ |
| **Uranium** | ember | −0.072 | **+0.482** | **+0.473** | **+0.743** |
| | snmf | +0.144 | +0.080 | +0.082 | −0.213 |
| | snmf_ember | 0.000 | +0.217 | **+0.405** | **+0.515** |
| **World War II** | ember | −0.110 | −0.005 | +0.020 | +0.122 |
| | snmf | −0.027 | +0.219 | +0.139 | **+0.339** |
| | snmf_ember | +0.019 | +0.063 | +0.223 | **+0.457** |

Batch 2, same view (`ember` shown; full table in `summary.csv`):

| Concept | config | de | es | fr | ru |
|---|---|---|---|---|---|
| **Golf** | ember | +0.375 | +0.380 | −0.064 | **+0.473** |
| **Baseball** | ember | +0.205 | +0.148 | −0.035 | +0.205 |
| **Culture of Greece** | ember | +0.270 | **+0.605** | **+0.469** | **+0.651** |
| **Gambling** | ember | +0.367 | +0.170 | +0.291 | **+0.632** |

Mean over reliable non-English native-wrapper cells, **all 8 concepts**:

| Concept | ember | snmf | snmf_ember |
|---|---|---|---|
| Harry Potter | −0.035 | −0.034 | +0.048 |
| World War II | +0.007 | +0.168 | +0.190 |
| Baseball | +0.131 | +0.103 | +0.189 |
| Golf | +0.291 | +0.178 | +0.148 |
| Halloween | +0.319 | +0.278 | +0.186 |
| Gambling | +0.365 | +0.001 | +0.278 |
| Uranium | +0.406 | +0.023 | +0.284 |
| Culture of Greece | **+0.499** | +0.104 | +0.416 |

**`ember` leaks positively in 7 of 8 concepts.** Only Harry Potter — the atypical one —
holds. The embedding-only edit does not survive translation; that is the headline answer
to the proposal's question.

### 4.4 What this shows

1. **The plan's *qualitative labels* were wrong, but its theory survived.** Halloween was
   designed as the low-leakage control ("identical string in de/es/fr") and instead leaks
   +0.19 to +0.32; World War II, predicted to leak maximally, leaks least for `ember`
   (+0.007). But when surface similarity is **measured** rather than assumed (§4.5),
   Halloween scores 0.135 — *below* Harry Potter's 0.158. The intuition about concept
   names did not match the actual question-level vocabulary overlap. See §4.5: with
   n=8, measured overlap is the **strongest single predictor of leakage**.
2. **Russian leaks hardest almost everywhere** (up to +0.65). Consistent with the
   mechanism: Cyrillic tokens are disjoint from the edited Latin ones.
3. **Harry Potter is the outlier, not the representative case.** It is the only concept
   of 8 where erasure roughly *holds* across languages (mean Δ_L ≈ −0.03). Any conclusion
   drawn from the single-concept run would have been wrong.
4. **`snmf_ember` still erases English hardest** (Gambling → 0.158, Culture of Greece →
   0.302, Uranium → 0.000), reproducing the paper's central claim; but it does **not**
   consistently leak least.
5. **The wrapper effect is small and inconsistent** — mean `qa_frac` native minus English
   wrapper: Halloween +0.081, World War II −0.013, Harry Potter −0.044, Uranium −0.071.
   ⚠️ This **corrects** an earlier read of the Harry-Potter-only data, which showed a
   0.16 German swing and was described as a large effect. It does not generalise; treat
   the wrapper axis as a minor factor unless a specific cell motivates otherwise.

### 4.5 Separating the two explanations — what batch 2 was for

Batch 1 left a **confound**. Two rival explanations of leakage fit it equally well:

| | claim |
|---|---|
| **H1 surface-form** (the plan's) | leak ∝ the edited English tokens being *absent* from the language-L prompt |
| **H2 edit-breadth** | leak ∝ *how many* tokens EMBER edited (r = 0.93 over batch 1's four points) |

In batch 1 the two moved together, so neither could be ruled out. Batch 2's concepts were
picked so the two hypotheses **predict opposite outcomes** (see the comment block in
`scripts/run_experiment.sh`). It worked — across all 8, the two axes are now
**independent**: `r(overlap, n_tokens) = +0.08`.

`overlap` = mean word overlap between the English question and its de/es/fr translation
(the QA files are index-aligned, so questions pair up 1:1); `n_tok` = `n_tokens_edited`
from that concept's ember row in `best_embed.csv`.

| Concept | overlap | n_tok | mean Δ_L (ember) |
|---|---|---|---|
| Harry Potter | 0.158 | 39 | −0.035 |
| World War II | 0.144 | 36 | +0.007 |
| Halloween | 0.135 | 80 | +0.319 |
| Uranium | 0.117 | 70 | +0.406 |
| Golf | **0.177** | 70 | +0.291 |
| Baseball | 0.156 | 79 | +0.131 |
| Culture of Greece | **0.056** | 52 | **+0.499** |
| Gambling | 0.108 | 69 | +0.365 |

|  | r with Δ_L | partial r (controlling for the other) |
|---|---|---|
| **surface overlap** | **−0.688** | **−0.855** |
| tokens edited | +0.513 | +0.789 |

**Both effects are real and neither explains the other away.** Controlling for token
count, overlap still predicts leakage at −0.86; controlling for overlap, token count
still predicts at +0.79.

Each hypothesis won its own discriminating case:

- **Culture of Greece → H1.** Lowest overlap (0.056) but only 1 feature / 52 tokens.
  H2 predicted a *low* leak; it leaked **most of all 8** (+0.499).
- **Golf → H2.** Highest overlap of all 18 concepts (0.177) should have meant minimal
  leakage under H1, but it leaked +0.291 — matching its 70 edited tokens.

> ⚠️ **These are 8-point correlations.** Treat the *direction* as informative and the
> magnitudes as soft. `overlap` is also only a **proxy** — it measures question-vocabulary
> similarity, not whether the *actually edited tokens* appear in the prompt. The
> `coverage_L` analysis (§8.2) measures that directly and is what should settle this.

---

## 5. Metrics — exactly what is computed

Reused **verbatim** from the paper / repo (see `crosslingual/evaluate.py`, copied from
`ember.evals.orchestrator`):

| Metric | Formula | Paper ref |
|---|---|---|
| `qa_frac` (MC) | `clip((acc − 0.25) / (base − 0.25), 0, 1)` | Eq. 22 |
| `qa_frac` (open) | `clip(acc / base, 0, 1)`, NaN if `base == 0` | Eq. 21 |
| `efficacy` | `1 − qa_frac` | Eq. 23 ϕ_efficacy |
| `specificity` | `harmonic_mean(mmlu_frac_en, simdom_frac)` | Eq. 23 ϕ_specificity |

**New for this experiment (NOT in the paper):**

| Metric | Definition |
|---|---|
| `Δ_L` | `qa_frac(L) − qa_frac(en)`, same config. The paper never evaluates non-English. |
| `reliable` | Screening flag from plan §4: false if pristine `QA_test` MC < **0.40** or open < **0.30**. Only the `qa_kinds` actually being run are checked (see §6.3). |

**Two deliberate deviations from the paper — read before comparing to Table 1:**

1. **`harmonic` here is a 2-term mean** (`efficacy`, `specificity`). The paper's H_score
   is 3-term and includes **coherence** (AlpacaEval). Coherence is not computed in
   Phase 2, because no metric the spec asks for consumes it. **Do not compare this
   `harmonic` column to the paper's.**
2. **`mmlu_frac` is computed once per config, in English**, and reused for all five
   languages (MMLU is general-domain, not concept- or language-specific). Cached under
   `data/crosslingual_baselines/_mmlu_by_config/<config>/`.

**Never** compare a non-English accuracy to the English baseline. `qa_frac` always uses
that language's *own* pristine baseline as the denominator — otherwise cross-lingual
model competence gets confounded with erasure efficacy.

---

## 6. Gotchas — read this before debugging anything

### 6.1 Phase 2 OOM: model not freed between configs (FIXED)

Phase 2 loads a **fresh model per config** by design. The first version kept a stale
reference (`_free_model(hf_model)` did `del` on its own *parameter*, and the
`WrappedHFModel` wrapper held a second reference), so model #1 was still resident when
model #2 loaded → `device_map="auto"` offloaded to CPU → OOM on the Gemma-2 lm_head
untie.

Fixed in `crosslingual/run.py`: each config's model is scoped inside `_run_config()`, so
returning drops every reference, then `_cuda_cleanup()` reclaims. **If you add a config
that holds a model reference outside that function, this will regress.**

### 6.2 SNMF validate OOM: `alpaca_batch_size` (FIXED, but the original configs still have it)

`configs/ember_gemma.yaml` sets `eval.alpaca_batch_size: 8`. **`configs/snmf_gemma.yaml`
and `configs/snmf_ember_gemma.yaml` do not** — so it falls back to the pipeline's default
of **50** (`pipeline.py::_ALPACA_BATCH_SIZE_DEFAULT`), which OOMs generating 200-token
AlpacaEval completions on a ~10.5 GB GPU.

This is a **latent bug in the project's own configs**, not something introduced here. It
only surfaced because SNMF had never been run in this checkout. The `*_gridonly.yaml`
copies now set `alpaca_batch_size: 8`.

> ⚠️ **The original `configs/snmf_gemma.yaml` / `configs/snmf_ember_gemma.yaml` are still
> unfixed.** They were left untouched on purpose (they are the paper's configs). If you
> ever run a full non-gridonly pipeline with them, apply the same fix first.

### 6.3 The `reliable` flag is scope-dependent

`_is_reliable(entry, qa_kinds)` only applies a floor for the `qa_kinds` you are actually
running. An MC-only run is not dragged to `reliable=false` by a weak open-ended baseline
it never reports. The flag is **recomputed on every cache hit** and the baseline JSON
records `screened_on`. Consequence: the same cached baseline can legitimately report
`reliable=true` under `--qa-kinds mc` and `false` under `mc open`.

### 6.4 Open-ended QA was dropped for this run

`--qa-kinds mc` only. Pristine non-English open-ended accuracy is 0.04–0.24, far too
close to zero to be a usable denominator — every non-English open cell would be screened
out anyway. It also removes the scaling Gemini cost (~900 judge calls per config vs a
flat ~300 for coherence). The code still fully supports `--qa-kinds mc open`; set
`QA_KINDS="mc open"` when running the driver.

### 6.5 Resume semantics

Both phases are resumable; **just resubmit the identical command**.
- Phase 1 skips grid cells already in `hps.csv` (keyed on HP columns only — so if you
  change `selection.ratio_thresh` you **must** pass `--overwrite`, or you will silently
  reuse rows computed under the old threshold).
- Phase 2 skips cells already on disk; a config whose cells are all present skips its
  model load entirely.

### 6.6 The driver's closing log lines are unconditional

`[experiment] done` and `summary -> ...` print at the end regardless of which phase ran.
Seeing them after a `--phase grids` job does **not** mean Phase 2 ran. Check
`summary.csv` directly.

### 6.7 `results/` is gitignored

`results/`, `mf_outputs/`, `data/baselines/` are all in `.gitignore` — they exist only on
this filesystem. `results/` is only **2.7 MB** but represents ~11 GPU-hours. Back it up:

```bash
tar czf results-backup-$(date +%F).tar.gz results/ data/baselines/ data/crosslingual_baselines/
```

### 6.8 Concurrent jobs race on the HuggingFace feature download (FIXED)

Every grid job calls `ember.data.ensure_features()` on startup. `submit_all.sh` launches
a concept's `ember` and `snmf` jobs simultaneously (they have no data dependency), so
both `snapshot_download` the **same** files into the **same** staging dir
`mf_outputs/.cache/huggingface/download/`. They then race on `shutil.move` of the same
`*.incomplete` temp file and one dies:

```
FileNotFoundError: .../Golf/embedding/QkpvvwK3RJMvPuQaiLr12AAbd_w=.2485...incomplete
```

This killed Golf's first attempt (jobs 895126/895127, both arms). The other three
concepts survived on timing luck alone.

**Fix:** `submit_all.sh` now runs a `prefetch_features` pre-flight on the login node —
GPU-free, ~20s per concept — before submitting anything, so every job starts with a warm
cache and the race window never opens. It refuses to submit if the prefetch fails.
Full `ember ‖ snmf` parallelism is preserved.

### 6.9 `--arms` takes ONE space-separated string, not separate words (FIXED)

`run_experiment.sh` parses `--arms` as a single argument:

```bash
--arms) shift; IFS=' ' read -r -a ARMS <<< "$1"; shift ;;
```

So the correct form is `--arms "ember snmf snmf_ember"`. Passing
`--arms "ember" "snmf" "snmf_ember"` (or `"${ARMS[@]}"` from a script) expands to
separate words, the parser takes `ember` as the value and then dies on the next one:

```
unknown argument: snmf     (exit code 2, job fails in <1s)
```

This killed all three batch-2 eval jobs (895133/895137/895141). No GPU time was lost —
they failed instantly — and the grids they depended on were untouched. `submit_all.sh`
now passes `"${ARMS[*]}"` (star, not `@`).


---

## 7. Code map

**Nothing under `ember/` was modified.** Verified with `git diff --stat -- ember/` (empty).
New code only:

```
crosslingual/            1,916 lines, imports ember/ but is never imported by it
├── data.py              language-aware QA loaders (parameterized copy of local_datasets)
├── prompts.py           wrapper-language scaffold strings + prompt builders
├── evaluate.py          MC/open eval loops with wrapper_lang + the metric formulas
├── baseline.py          per-language pristine baselines, screening, English MMLU
├── best_hp.py           locate + apply an already-chosen HP (never searches)
├── results_io.py        output paths, summary schema, resumability, Δ_L
└── run.py               CLI entrypoint

configs/*_gridonly.yaml  copies of the paper's configs with final-test + relearning off
scripts/submit_all.sh      SUBMIT THIS -- whole batch, dependencies wired, prefetch
scripts/run_experiment.sh  one concept: Phase 1 and/or Phase 2
scripts/run_erasure_grid.sh  one concept x one arm: Phase 1 only
```

**Design rule that must be preserved:** `crosslingual/` reads an already-selected HP out
of `results/` and applies it. It must **never** run its own grid search — the whole point
is that the edit being evaluated is the one the English grid chose.

### Running it

**Normal path — one command for a whole batch.** `scripts/submit_all.sh` prefetches
features (§6.8) then submits 4 jobs per concept with the dependency graph wired:

```
ember  ─┬────────────► snmf_ember ─┐
        │                           ├──► eval
snmf ───┴───────────────────────────┘
```

```bash
./scripts/submit_all.sh --dry-run                  # print, submit nothing
./scripts/submit_all.sh                            # the CONCEPTS list in the script
./scripts/submit_all.sh --concepts "Golf|Baseball" # subset (pipe-separated)
```

`snmf_ember` waits on `ember` because its embedding step reads the `best_embed.csv`
ember writes; `eval` waits on all three. All dependencies are `afterok`, so a failed
grid **cancels** its dependents rather than feeding them a half-written `hps.csv`
(a dependent stuck at `DependencyNeverSatisfied` means its grid failed — check that
grid's log, fix, resubmit).

**Single jobs**, if you need them — note `--arms` takes **one quoted string** (§6.9):

```bash
./scripts/run_experiment.sh --concept "Golf" --arms "ember" --dry-run   # no GPU, ~30s
sbatch scripts/run_experiment.sh --concept "Golf" --arms "snmf" --phase grids
sbatch scripts/run_experiment.sh --concept "Golf" --arms "ember snmf snmf_ember" --phase eval
```

Submit **one arm per grid job** — the 12h SLURM limit will not fit two 144-cell grids.

---

## 8. What is NOT done

### 8.1 Concepts — DONE (8 of 18)

Both batches complete. Batch 1 is the plan's §9 tokenization-contrast set; batch 2 was
chosen to break the confound batch 1 left behind (§4.5).

| Batch | Concept | overlap | n_tok | mean Δ_L (ember) |
|---|---|---|---|---|
| 1 | Harry Potter | 0.158 | 39 | −0.035 |
| 1 | World War II | 0.144 | 36 | +0.007 |
| 1 | Halloween | 0.135 | 80 | +0.319 |
| 1 | Uranium | 0.117 | 70 | +0.406 |
| 2 | Golf | 0.177 | 70 | +0.291 |
| 2 | Baseball | 0.156 | 79 | +0.131 |
| 2 | Culture of Greece | 0.056 | 52 | +0.499 |
| 2 | Gambling | 0.108 | 69 | +0.365 |

Ten concepts remain unrun (Ancient Rome, Artificial intelligence, COVID-19 pandemic,
Cannabis, Gun, Heroin, Nazism, Pornography, Republic of Ireland, Valentine's Day). A
third batch is only worth ~9 GPU-hours/concept **if** it is designed to test something
§4.5 leaves open — not just to raise n. Note `Valentine's Day` has the weakest English
baseline of the 18 (0.52) and is the likeliest to fail screening.

> **The Nazism exclusion no longer applies under `--qa-kinds mc`.** It was excluded
> because the Gemini judge scores a refusal as *incorrect*, which would look like
> successful erasure — but MC never sends concept content to the judge (it parses a
> letter). Nazism has the 2nd-highest overlap of all 18 (0.173) with only 1 embedding
> feature, so it is a *confirmation* case for both hypotheses rather than a
> discriminating one.

Cost was ~9 GPU-hours per concept, ~85% of it the two 144-cell SNMF grids.

### 8.2 The "money figure" from the plan (§8) — not implemented, now the top priority

The plan calls for a scatter of **Δ_L against `coverage_L`** — the fraction of
EMBER-edited tokens that actually appear in the language-L prompt. That turns "erasure
leaks" into "erasure leaks *exactly to the extent the edited tokens are absent*", which
is the mechanistic claim the proposal promises. Nothing computes `coverage_L` yet.

**This is now the single highest-value next step.** §4.5 shows that measured
question-overlap and edited-token-count *both* predict leakage and neither explains the
other away (partial r = −0.86 and +0.79 over n=8). But `overlap` is only a **proxy** — it
measures question-vocabulary similarity, not whether the *actually edited tokens* appear
in the prompt. `coverage_L` measures that directly, and is the quantity the proposal's
mechanistic claim is actually about.

Inputs all exist and it needs **no GPU**:
- edited token list → `ember.erasure.embed_edit.apply_concept_embed_edit_factored`;
  counts per concept are in the §8.1 table (36–80 tokens)
- prompts per language → the 288 per-cell JSONs
- Δ_L → `crosslingual.results_io.compute_delta_l(summary)`

### 8.3 Cross-lingual relearning (plan §10) — not implemented

EMBER's headline claim is robustness to *relearning*, not raw efficacy. The sharper
question — "is the robustness gain also English-only?" — is untested. The plan calls this
the highest value per GPU-hour.

### 8.4 Other gaps

- **δ is not controlled across concepts** (§3.1) — the grid picked anywhere from 5 to
  200. Before attributing any cross-concept difference to the concept, check whether δ
  explains it instead. This is a live confound in the §4.5 correlations.
- **Open-ended QA** across languages (see §6.4) — a deliberate skip. Note the other seven
  concepts have **no** open baselines at all, so enabling it is a GPU + Gemini job, not a
  flag flip.
- **Coherence / AlpacaEval** in Phase 2 (see §5) — would cost a flat ~1,200 Gemini calls
  total; wire it as a config-level number like MMLU if the 3-term H_score is wanted.
- **Llama-3.1-8B** — everything so far is Gemma-2-2B only. The paper covers both.
- **Translated wrapper strings** in `crosslingual/prompts.py` are machine-translated and
  have **not** had a native-speaker review. Lower priority now that §4.4 finding 5 shows
  the wrapper axis is a small, inconsistent effect.

---

## 9. Environment notes

- `GEMINI_API_KEY` lives in `.env`; `ember.run_erasure`, `ember.run_eval` and
  `crosslingual/run.py` all call `load_dotenv()` themselves. Total Gemini spend to date
  across the whole project: **~3.4 NIS** over 15,760 calls (`gemini_usage_log.jsonl`).
- Python is `.venv/bin/python` (3.10). `python` is not on PATH in this shell.
- SNMF features download on demand from the `ClSu/ember-features` HF dataset
  (`features_source: hf`); only Harry Potter is cached locally in `mf_outputs/`.
- Model load can take 7–11 minutes on a cold/slow filesystem — that is not a hang.
