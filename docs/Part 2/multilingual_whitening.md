# Multilingual whitening + sharpening (Llama), 300-per-language extended sets, word-aware judge input

Branch `orip-ml-whitening` (worktree `.claude/worktrees/ml-whitening`), 2026-10-15.

## What was asked

- **Main.** Bring background whitening + target sharpening (`Instructions.txt`) to the multilingual Llama flow
  (8 languages; $V^*$ = up to 40 concept + 40 translated neutral sentences per language).
- Assess the rank problem (Llama $d$ = 4096, Gemma 2304): does the target need more sentences per language?
- Generalize the background and the target to 8 languages.
- 3.2.h extended set (fixed $Z$, sparse $Y$): up to 300 sentences per language from the concept page, neutral matched per language,
  instead of the whole page; pages that are too short filled with Gemini translations. In all multilingual notebooks.
- Judge input switch at the top of each notebook: bare tokens (old) or token → the words it came from, in alphabetical order
  (multilingual: grouped by language).

## 1. Rank assessment

Distinct Llama token types $m$ of each candidate target; the second moment $S$ of $m$ whitened rows has rank $\le m$, and every
direction outside its span sits exactly at the shrinkage floor $g=\eta$ (it is "depleted" only because it was not sampled).

| Target | $m$ | vs $d$ = 4096 |
|---|---|---|
| English concept page (single-language notebooks): HP / AR / VD | 3,282 / 4,303 / 2,903 | HP and VD rank-deficient (≥ 814 directions at $\eta$) |
| Multilingual $T_C$ (concept-only types of the 40-per-language $V^*$) | 2,389–2,991 | rank-deficient for all 6 concepts |
| Multilingual concept pages, 8 languages pooled (native) | 7,371–12,738 | 1.8–3.1 $d$: full rank for all 6 |
| Pooled pages minus $V^*$ (leak-free) | HP 6,808, Halloween 5,113, Pornography 6,804, Gambling 2,762, Gun 3,212, VD 2,922 | full rank for 3 of 6 natively |

**Decision.** The target needs more than the 40 $V^*$ sentences per language. It is the same 300-per-language extended concept
corpus 3.2.h uses (pooled over languages, each type once). The target enters only through a $d\times d$ matrix, so its size costs
nothing in the factorization. The translated top-up adds types and balances the languages (Hindi and Thai get a real share of $S$).

**Leak caveat.** The English `page` target contains 1,846 of the 1,883 $T_C$ tokens (Llama HP). Removing them (`page_minus_vstar`)
takes the AUC from 0.872 to 0.61, below the raw fit's 0.667, and with the leaky target 181–187 of 200 features pass $\rho\ge2$
($\rho_{\max}\approx3\cdot10^5$). So every multilingual `ext` run is paired with `ext_minus_vstar` in the queue; only the latter says
whether sharpening finds a concept subspace or stretches the target's own tokens.

## 2. Background (`BACKGROUND = "wiki_ml"`)

Token types of random Wikipedia sentences, **the same number of sentences per language** (at most 4 random prose sentences per article,
the native-neutral filters and splitter), pooled and counted once — the way $V^*$ itself is sampled. An English-only background gives
Devanagari / Thai / accented-Latin directions tiny variance, so $W\propto\Lambda^{-1/2}$ blows them up and the target looks "enriched" in
language identity; sharpening would then amplify language rather than concept. `wiki_en` and `full` stay available; the notebook's
cPCA cell shows the top enrichment directions under `wiki_ml` and `wiki_en` side by side.

Built by `experiments/whitening/background_ml.py` (one parquet shard per language of `wikimedia/wikipedia` 20231101.<lang>; English
reuses the 300k-sentence sample):

| | value |
|---|---|
| sentences | 40,000 per language, 320,000 total |
| types (pooled) | 62,025 |
| types per language | en 42,504 · es 25,472 · fr 27,314 · de 25,077 · it 26,862 · pt 27,399 · hi 6,302 · th 13,583 |
| types in one language only | en 11,725 · es 1,323 · fr 1,565 · de 1,850 · it 1,253 · pt 1,356 · hi 1,177 · th 2,379 |
| split-half top-100 mean principal-angle sine | 0.504 at 5k/lang → 0.435 at 40k/lang (plateau; English background 0.433 at 300k) |
| $\epsilon$ (`eps_select.py --backgrounds wiki_ml`) | `eps_rel` = 1e-4 (CV error 0.645, flat up to 1e-3) → `defaults_ml.json` ($\eta$ = 0.2, $\gamma$ = 1 from the English sweep) |

The stability criterion (sine ≤ 0.1) is not reachable for either background at these sizes; the curve has flattened, so the cap stopped it.

## 3. Extended sets: 300 sentences per language (`EXT_PER_LANG`, `EXT_TOPUP`)

- **Concept** (`extended_sets.build_concept_n`): per language the 40 $V^*$ concept sentences (always kept), a seeded sample of the rest of
  the page up to 300, then translated top-up to 300 (`source = "translated"`).
- **Neutral** (`build_neutral_n`): per language the $V^*$ neutral sentences plus the replicate's translated pool, as many as the concept
  side, duplicates skipped. So $V^*_{ext}\supseteq V^*$ (previously 546 $V^*$ tokens fell out of $V^*_{ext}$).
- The neutral pools were grown with `neutral_replicates.py --min-pool 260` (Hindi only, 48 sentences per replicate). Batch caches are now
  append-only (`extended_translated._batch`), so growing a pool never re-translates cached sentences and existing files stay identical.
- Files: `<concept extended dir>/concept_ext_n300[_tr].json`, `neutral_ext_translated_rep{r}_n300[_tr].json`; the whole-page files are
  untouched (`EXT_PER_LANG = None` restores them).

### Translated top-up (`topup_translated.py`)

Only 11 of 48 concept × language pages reach 300 sentences; 38 pairs need 6,522 translated sentences.

- Sources: the other languages' page sentences, English first, then by page size (seeded); 15% surplus for QA drops.
- Prompt: the repo's Wikipedia-editor translation prompt plus a glossary — the concept's title in the target language and, for English
  sources, the target-language titles of the English article's linked articles whose anchor text occurs in the batch (interlanguage links).
- QA: script share ≥ 0.8, no echo, length ratio 0.4–2.5 (failures re-translated once with gemini-3.6-flash, then dropped); LaBSE cosine
  per pair (`sentence-transformers/LaBSE` via plain transformers). COMET-Kiwi was not run: its environment is not reachable from here.

Each cell is the number of translated sentences kept, then the median LaBSE score. "—" means the native page already had ≥ 300 sentences.

| Concept | kept | en | es | fr | de | it | pt | hi | th | QA drops |
|---|---|---|---|---|---|---|---|---|---|---|
| Harry Potter | 393 | — | 16 / 0.95 | — | 12 / 0.91 | — | 103 / 0.92 | 262 / 0.89 | — | 0 |
| Gambling | 1599 | 189 / 0.88 | 212 / 0.91 | 121 / 0.90 | 109 / 0.88 | 136 / 0.92 | 268 / 0.91 | 274 / 0.88 | 290 / 0.84 | 6 |
| Gun | 1393 | — | 103 / 0.91 | 228 / 0.90 | 278 / 0.88 | 174 / 0.91 | 215 / 0.90 | 295 / 0.88 | 100 / 0.84 | 2 |
| Halloween | 668 | 16 / 0.90 | 7 / 0.89 | 63 / 0.90 | 203 / 0.88 | — | — | 88 / 0.87 | 291 / 0.84 | 7 |
| Pornography | 1086 | — | 40 / 0.93 | — | 12 / 0.90 | 228 / 0.93 | 263 / 0.90 | 246 / 0.87 | 297 / 0.84 | 12 |
| Valentine's Day | 1383 | — | 258 / 0.93 | 212 / 0.92 | 18 / 0.89 | 258 / 0.93 | 239 / 0.91 | 130 / 0.88 | 268 / 0.86 | 22 |

The 15% surplus covered every QA drop, so every concept × language reaches exactly 300 sentences. Most of the drops are Thai script or
echo failures. Thai has the lowest LaBSE (median 0.84–0.86; 5th percentile 0.62–0.73), and part of that gap is LaBSE scoring Thai
lower in general, not only translation quality. Latin-script medians are 0.88–0.95.

Verified for all 6 concepts × 3 replicates (`check_ext.py`):
- 300 concept and 300 neutral sentences per language;
- every $V^*$ concept and neutral sentence is in the extended set;
- concept and neutral sets don't overlap.

Duplicates inside the native pages leave 4 duplicate concept sentences for Gambling and 7 for Valentine's Day. There is ≤ 1 duplicate
neutral sentence, and it is inside $V^*$ itself. The `--topup none` (native-only) variants were built too.

## 4. Judge input (`JUDGE_FORMAT`)

`ember/judge_words.py`. Words come from $V^*$'s own sentences via tokenizer offsets (pythainlp words for Thai; punctuation tokens shown
as themselves).

| Format | Single language | Multilingual |
|---|---|---|
| `tokens` (paper) | `▁Pot` | `▁Pot` |
| `words` | `▁Pot → Potter, pottery` | `▁Pot → en: Potter, pottery; de: Potter` |
| `top_word` (old multilingual) | — | `▁Pot → Potter (en)` |

At most 8 words (the most frequent, then sorted case-insensitively; "+N more" when cut). Each format has its own Stage-1 system prompt
(`desc_sys_prompt`, passed explicitly to `get_description(..., sys_prompt=...)`) and its own judge cache file; the original format of
each notebook keeps `judge.json`, so cached verdicts stay valid. Defaults: `top_word` in the standard multilingual notebooks, `tokens`
in the whitened notebooks (unchanged behaviour), `words` in the new notebook.

## 5. Notebooks

- **New:** `ember_three_steps_multilingual_whitened_llama.ipynb` and `low_ram_demands/multilingual_whitened_llama.ipynb`, generated by
  `scripts/build_multilingual_whitened_nb.py` from the Gambling multilingual notebook (pinned at 8d5603e). Any of the 6 concepts
  (`CONCEPT`), `BACKGROUND`, `TARGET`, `GAMMA`, `EXT_PER_LANG`, `EXT_TOPUP`, `JUDGE_FORMAT`. 3.1.c′ prints the rank / leak table for all
  target options and the cPCA comparison; 3.1.d′ reuses the standard notebook's fit as the raw reference; 3.2.e′ judges the raw fit;
  3.2.w matches the fits and writes `summary.json`.
- **Patched** by `scripts/patch_notebooks_ml.py` (idempotent): the 12 standard multilingual notebooks (judge switch + capped extended
  sets) and the 12 whitened notebooks (judge switch).
- **Queue:** `experiments/whitening/queue_ml.json`, 19 runs up to Step 3.3: per concept `ext` γ=1, `ext_minus_vstar` γ=1, `ext` γ=0,
  plus HP with `wiki_en`.

    python scripts/run_overnight.py --queue experiments/whitening/queue_ml.json

## 6. Smoke runs (SMOKE=True, stopped before Step 3.3)

**New notebook, HP (`wiki_ml`, `ext`, γ=1, `words`).** Runs end to end through 3.2.w.

| Target | m (types) | Rank | Directions at η | $T_C$ leak |
|---|---|---|---|---|
| `ext` | 12,564 | full | 0 | 2,902 / 2,902 |
| `ext_minus_vstar` | 6,506 | full | 0 | 0 |
| `tc` | 2,902 | 2,902 | 1,194 | — |

cPCA, comparing backgrounds:
- Under `wiki_ml` the top enrichment is $g$ = 2.82. Its top directions are Italian/German/Portuguese function words, the same frequency
  confound seen in English.
- Under `wiki_en` the top enrichment is $g$ = 6.82, and its top directions are es/pt/de/it language clusters.

The language-balanced background removes most of the language-identity enrichment, as predicted.

**Patched `low_ram_demands/whitened_llama.ipynb` (`JUDGE_FORMAT="words"`).**
- Runs.
- The judge sees e.g. `iseum → coliseum`, `-fiction → non-fiction`.
- Separate caches: `judge_words.json` / `judge0_words.json`.

**Not run.** No smoke run of a patched *standard* multilingual notebook with the capped sets. Those notebooks share the
`ext_paths` / `judge_words` code that the two runs above exercise, and all 24 patched notebooks pass a syntax check. Real results
come from the queue.

## Open

- Multilingual $\eta$ sweep ($\eta$, $\gamma$ are the English defaults).
- COMET-Kiwi and a term check for the top-up translations (hi/th never measured in this repo).
- Concepts whose $V^*$ has fewer than 40 sentences in a language stay short there (unchanged on purpose).
