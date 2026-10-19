# Part 2 — plan

## Current plan (decided 2026-10-18): CINDER vs EMBER, English only

**Question.** Does CINDER (Ori: contrastive sharpening, global sparsity, two-way judge)
erase a concept better than EMBER in English, and does the better English edit leak less
when the same questions are asked in de / es / fr / ru?

**Design.** Both methods fitted on English sentences only (`data/`), the same edit
(EMBER's embedding edit; CINDER's features are mapped back into embedding space), the same
judge (gemini-3.5-flash-lite), the same strength search and test as Part 1's EMBER.

| | English erasure | leak in de / es / fr / ru |
|---|---|---|
| EMBER | Part 1 (done) | Part 1 (done) |
| CINDER | strength search (`cinder.grid`, `cinder.pick`) | `cinder.final_test` (same cells as Part 1's repaired run) |

**Concepts (6):**

| EMBER English retention | Concept | Kind | Russian leak in Part 1 |
|---|---|---|---|
| strong | Cannabis (0.39), Gun (0.54) | substance, weapon | +0.51, +0.46 |
| medium | Nazism (0.63), Republic of Ireland (0.67) | ideology, country | +0.37, +0.33 |
| weak | Culture of Greece (0.73), Golf (0.75) | culture, sport | +0.27, +0.25 |

Chosen from the 10 concepts whose EMBER features are identical under the paper's judge and
3.5 (`mf_outputs/en_rejudge35/embedding_judge_agreement.md`), so no EMBER rerun is needed;
spread over erasure strength and kinds of knowledge; all with reliable tests in 5 languages;
none is a CINDER tuning concept (Harry Potter, Ancient Rome, Valentine's Day).

**Steps, owners, status:** [`../progress.md`](../progress.md), "Part 2 status". Code review,
decisions and the implementation: [`cinder_code_review.md`](cinder_code_review.md). Code: `cinder/`.

**Decided 2026-10-18.** CINDER is Ori's whole method, not compared layer by layer: his ladder rung
**L123** (`experiments/varwidth/layer_eval.py`): contrastive sharpening (page target, η 0.05, γ 1,
ε 10⁻⁴), GS-norm (`rn`, `refit=False`, floor 0.2%, cap 3%), and his two-way judge HJ = TJ ∨ LO@5 on
the top 20 features by ρ (LO = matches with his 20 concept words, `j3_data/concept_words_v1.json`).
His newer code, judge data and seed-42 runs for the 6 concepts were copied from his folder
(2026-10-18); refitting here reproduces his fits (same supports, |ΔY| ≤ 5·10⁻⁶). We run the strength
search and the final test ourselves with Part 1's protocol: his `erase.py` picks δ without the Alpaca
judge and scores MC with the old 4-token cut-off, so its numbers are not comparable to Part 1's EMBER.

**CINDER features (L123, seed 42, `judge_rerun/Cinder/features/`):**

| Concept | EMBER features / edited tokens | CINDER features (accepted by) / edited tokens |
|---|---|---|
| Cannabis | 1 / 45 | f42 (TJ and LO 9) / 34 |
| Gun | 2 / 61 | f37 (TJ and LO 6), f40 (TJ) / 67 |
| Nazism | 1 / 41 | **none**: the closest candidate is a broad "political ideologies" feature (LO 2, TJ no); no edit |
| Republic of Ireland | 1 / 31 | **none**: the Irish tokens sit in a 111-token "countries and nationalities" feature outside the top 20 by ρ; no edit |
| Culture of Greece | 1 / 52 | f10 (TJ; LO 4) / 106 |
| Golf | 2 / 70 | f33 (TJ and LO 10) / 44 |

So CINDER edits 4 of the 6 concepts. Caveat: Culture of Greece is one of the 4 example concepts in
Ori's concept-word prompt (`j3_data/prompt_v1.txt`), so its LO count is not clean (it is accepted by TJ
anyway). Ori's own results agree: no edit for Nazism and Republic of Ireland at seed 42
(`results/varwidth/layers/erasure/` in his folder).

**Strength grid (`python -m cinder.edit_size`, `judge_rerun/Cinder/delta_grid.json`).** At the
same δ, CINDER's edit is about half of EMBER's: the median relative edit ‖c_t‖ / ‖e_t‖ is 0.24 for
CINDER and 0.46 for EMBER (paired over the 4 concepts both edit; per concept the ratio is
1.4–2.7). The ratio 1.93 is inside the agreed band (0.5–2), so **CINDER uses EMBER's grid,
0.5, 1, 2, 5, 10, 50, 100, 200** (also the grid of Ori's `erase.py`). On CINDER's scale it spans
relative edits 0.12–48; EMBER's QA
accuracy stops falling at δ ≈ 5–10 (relative edit 2–5), i.e. δ ≈ 10–20 for CINDER, well inside.
Edge rule: if the fast screen's best δ is the largest (smallest) value, one more δ (×2, or ÷2)
is scored.

**Report:** English erasure per concept grouped by EMBER's strength (CINDER's claim is
strongest where EMBER is weak); leak only where both methods erase English; coverage vs leak
for both methods; paired by question, bootstrap over concepts.

**Later:** equal-strength comparison; Ori's full ladder (EMBER → +CS → +CS+GS → +TWJ);
training on other languages (the two-arm design below).

---

# Earlier design: training on other languages (parked, 2026-10-17)

*Decided 2026-10-17 as the Part 2 baseline, parked on 2026-10-18 in favour of the plan above.
The results below (feature search, rank sweeps, tokenization, coverage) stay valid.
Data and feature search details: [`multilingual_features.md`](multilingual_features.md).*

## Question

Part 1 showed that an edit built from **English** features leaks: the concept is still
answerable in other languages (EMBER mean leak +0.19, Russian +0.33;
[`../Part 1/crosslingual_next.md`](../Part%201/crosslingual_next.md)). Part 2 asks whether
features found with **other languages in the data** erase the concept in those languages
too, and which way of adding the languages works better.

## Design: two arms

| Arm | Factorization | Features used for erasure |
|---|---|---|
| **A. Joint** | one fit on en + ru + de + es + fr sentences together | the features of that one fit |
| **B. Per-language** | one fit per language: en, ru, de, es, fr, each on its own sentences | the union of the five languages' features |

Both arms then go through the same steps: feature filter → erasure → cross-lingual
evaluation, and are compared with each other and with Part 1 (English features only).

## Fixed settings

| | |
|---|---|
| Model | Gemma-2-2b-it |
| Factorization | `ember/train_mf_features.py`, rank 100, seed 42, both tracks (embedding for EMBER, MLP for SNMF) |
| Data | **native Wikipedia only**: `experiments/multilingual_fit/data_v2/` (300 concept + 300 neutral sentences per language; English from `data/`, as in the paper). The translated sets (`data_translated/`) are not used for now. |
| Feature filter | `ember/interpret_features.py`, ratio ≥ 2.0, `is_member` with confidence ≥ 0.85 |
| Judge | **gemini-3.5-flash-lite**, for every Part 2 feature list (both arms, all languages) |
| Evaluation | the Part 1 cross-lingual MC test (`crosslingual/`): `QA_test` in en/de/es/fr/ru, retention r_L, leak Δ_L = r_L − r_en, `SimdomQA_test` for specificity, MMLU |

**Why this judge.** The paper's judge (Gemini-2.5-Flash-Lite) is no longer available to us.
Re-judging the paper's English World War II candidates with 3.5 (`mf_outputs/en_rejudge35/`):
embedding identical (1 feature kept, 45/45 decisions agree); MLP 282/300 decisions agree,
15 → 12 features kept, 5 in common; the differences are borderline features at the 0.85
cut-off. So one judge is used for all of Part 2, and the English per-language list is the
paper's English factorization judged by 3.5 (1 + 12), not the paper's list (1 + 15).
Part 1 is not rerun; it keeps the paper's features (state this in the report).

## Steps

| # | Step | Status |
|---|---|---|
| 1 | Native sentence sets, cleaned (bibliography / back-matter removed, every removal logged and read) | **done** 2026-10-17 |
| 2 | Arm A fit, World War II | **done** (job 1046208, 2h04m) → `mf_outputs/native/multi_language/en+ru+de+es+fr/` |
| 3 | Arm B fits, World War II: ru, de, es, fr each alone; en = paper's factorization + 3.5 judge (`mf_outputs/en_rejudge35/`) | **done** (jobs 1046235–8, 25–37 min each) → `mf_outputs/native/multi_language/<lang>/` |
| 3b | Embedding rank sweep per language (en, ru, de, es, fr × World War II, Cannabis, Gun; ranks 100–800; English = check of our code against the paper) | **done** (jobs 1047659–73, 4–12 min each) → `mf_outputs/rank_search/`; results below |
| 4 | Look at the features of both arms: counts per track, which languages the tokens come from, shared vs language-specific features | after 2–3 |
| 5 | Erasure code reads Part 2 feature lists (today it reads only `mf_outputs/`) and takes the union for arm B | to do |
| 6 | Erasure + cross-lingual evaluation for both arms, World War II | to do |
| 7 | Extend to all 18 concepts | to do; cost per concept ≈ 2.5 h (arm A) + 4 × ~40 min (arm B) on 4 GPUs |

## First results: World War II feature search (2026-10-17)

Rank 100, seed 42, judge gemini-3.5-flash-lite, native `data_v2`:

| Run | Embedding: candidates → kept | MLP: candidates → kept |
|---|---|---|
| en (paper's fit, re-judged) | 45 → **1** | 150 → **12** |
| **Arm A** joint en+ru+de+es+fr | 24 → **0** | 53 → **3** |
| Arm B ru | 20 → 0 | 55 → **0** |
| Arm B de | 15 → 0 | 47 → 4 |
| Arm B es | 18 → 0 | 62 → 2 |
| Arm B fr | 22 → 0 | 51 → 3 |
| (old joint fit, 2026-10-14, old data) | 33 → 0 | 51 → 2 |

- **No embedding feature outside English.** The strongest embedding candidates are general
  war words shared across languages (attack / attaque / ataque, occupation, military,
  resistance), country names, or grammar (ru case endings and participle suffixes, fr
  plural verb forms, de conjunctions). The judge rejects them, and the rejections read as
  correct: none is specific to World War II. The English feature ("Nazi regime": Hitler,
  Führer, Reichstag …) has no counterpart; such names split into pieces in Russian.
- **Kept MLP features are on topic:** Axis offensives (with ru "наступ"), "WWII / Weltkrieg
  / 大戦" (projection), Axis nations in several languages (joint); "Weltkrieg / Kriegs",
  KZ and end-of-war dates (de); Allied / Soviet / Nazi factions, concentration camps,
  genocide (fr); Axis / Allied nationalities, occupation (es).
- **Russian keeps nothing:** 0 of 110 judge calls even say "related". Its candidates are
  grammar endings, punctuation, garbage tokens, or generic military words.
- So for World War II at rank 100, arm A gives EMBER **nothing** to edit, and arm B gives
  only the English feature; SNMF would get 3 (A) or up to 21 (B) MLP features.
- Grammar features pass the ratio ≥ 2 pre-filter with high ratios (ru 8.3, fr 8.6),
  although the neutral sets are now random sentences: concept and neutral text still
  differ in style. The judge removes them.
- **Joint vs per-language** (feature vectors compared by cosine of `F_` columns, same
  layer, closest of the 100 features in the other fit; baseline for unrelated features in
  a layer: median 0.4–0.6, 90th percentile 0.6–0.9, so only higher matches count):
  - arm B gives far more MLP features (21: en 12, de 4, fr 3, es 2, ru 0) than arm A (3),
    but 12 of the 21 come from English; neither arm gives EMBER more than the English
    embedding feature;
  - arm A features mix languages in their top tokens (L2 f14: 11 Russian, 9 Latin-script;
    L22 f38: German, French, Spanish nationality words); arm B features show one language;
  - one direction is clearly shared: the L2 "offensive / military" feature (arm A f14,
    en f40, ru f60, es; cosine 0.85–0.88, Russian baseline 90th percentile 0.68). Kept in
    arm A, where its tokens are "наступ" plus "Axis" / "Allied"; rejected in the Russian
    fit, where they are only "наступ" ("military advance in Slavic languages"). So
    whether the judge keeps a direction depends on which language's tokens it shows,
    and Russian's 0 partly reflects that;
  - the L4 "world war" projection feature (arm A f2 ≈ de f72, 0.86) is near the L4
    baseline (0.93), so not clearly shared; in the Russian fit its closest match
    (cosine 0.92) has ratio 1.72 and never reached the judge.
- Caveat: one concept, and World War II was also the concept that failed in the earlier
  word test (`ideas_and_discoveries.md`, finding 1); one seed; rank 100.

## Rank sweep, joint fit, embedding track (Mary, 2026-10-17)

`run_rank_search.sbatch` (jobs 1047067, 1047128): arm A joint fit, embedding track only
(`--skip-mlp`, seconds per fit), ranks 100–800, seed 42, same judge; output and
`analyze_rank_search.py` summaries in `mf_outputs/rank_search/…/rank_search/<Concept>/`.

| Rank | 100 | 200 | 300 | 400 | 600 | 800 |
|---|---|---|---|---|---|---|
| World War II: candidates → kept | 24 → 0 | 43 → 0 | 46 → 0 | 51 → 0 | 53 → 0 | 44 → 0 |
| Cannabis: candidates → kept | 31 → 0 | 52 → 0 | 66 → 0 | 75 → **1** | 81 → **1** | 73 → **1** |
| reconstruction error (WWII) | 0.94 | 0.91 | 0.89 | 0.87 | 0.84 | 0.82 |

(English alone, paper's fit at rank 100: 1 embedding feature for each concept.)

- Rank 100 reproduces job 1046208 (World War II: 24 candidates, 0 kept).
- **Cannabis:** from rank 400 on, one multilingual cannabis feature (cannabis, THC, CBD,
  hemp, marijuana; concept-token mass en 0.37, de 0.22–0.26, es 0.16–0.19, fr 0.18–0.20,
  **ru 0.04**), the same at 400, 600 and 800; it edits 121–148 tokens, 8–9 of them Russian.
- **World War II:** the English "Nazi regime" feature has no counterpart at any rank; at
  ranks 200–800 those names sit in a "totalitarian ideologies" feature together with
  communism, Stalin, Mussolini, which the judge rejects (not member, confidence 0.85).
  The other top candidates stay general war words (attack, destruction, combat, conflict).

### Why more rank does not help the embedding track (2026-10-17)

The embedding track does not fit the text: it factorizes the **embedding-table rows of the
distinct tokens** that occur in the sentences (V′; World War II: 4,603 tokens for en alone,
22,669 joint). The sentences only choose V′ and label each token: concept-only, neutral-only,
or **both** (in concept and neutral sentences; left out of the ratio score). Frequency and
context play no part. (The MLP track, in contrast, fits activations of the text.)

| World War II | V′ | concept-only | neutral-only | both |
|---|---|---|---|---|
| en | 4,603 | 1,693 | 2,303 | 607 (13%) |
| ru | 4,704 | 1,374 | 1,910 | 1,420 (30%) |
| joint | 22,669 | 7,734 | 9,816 | 5,119 (23%) |

English concept words are whole tokens and concept-only (Hitler, Nazi, Führer, Reichstag,
Holocaust; cannabis, THC, marijuana). Russian ones are split into short pieces that also
occur in unrelated text, so they are labelled "both" or neutral and carry no concept
meaning of their own: Гитлер = Г + ит + лер (all "both"), нацист = на + ци + ст,
марихуана = ма + ри + ху + ана, конопля = ко + но + пля. No rank or amount of text can make
an embedding feature for "Hitler" out of Г / ит / лер. In the joint fit the English concept
tokens are still there, but as a small share of 5× more tokens, and they group with nearby
meanings (Nazi with communism / Stalin). The same split explains why the Cannabis joint
feature has only 4% Russian mass.

### Embedding rank sweep per language (2026-10-17)

Embedding track only, seed 42, judge gemini-3.5-flash-lite; cell = embedding features kept
(candidates in `…/rank_search/<Concept>/summary.csv`).

| | Rank | 100 | 200 | 300 | 400 | 600 | 800 |
|---|---|---|---|---|---|---|---|
| **World War II** | en | 1 | 1 | 0 | 0 | 1 | 1 |
| | ru | 0 | 0 | 0 | 0 | 0 | 0 |
| | de | 0 | 1 | 1 | 0 | 0 | 1 |
| | es | 0 | 1 | 0 | 1 | 0 | 0 |
| | fr | 0 | 0 | 1 | 1 | 1 | 1 |
| | joint | 0 | 0 | 0 | 0 | 0 | 0 |
| **Cannabis** | en | 1 | 1 | 1 | 1 | 1 | 1 |
| | ru | 0 | 0 | 0 | 0 | 0 | 0 |
| | de | 1 | 1 | 1 | 1 | 0 | 1 |
| | es | 1 | 0 | 1 | 1 | 1 | 2 |
| | fr | 1 | 1 | 1 | 1 | 1 | 2 |
| | joint | 0 | 0 | 0 | 1 | 1 | 1 |
| **Gun** | en | 2 | 1 | 3 | 4 | 6 | 3 |
| | ru | 0 | 0 | 1 | 2 | 1 | 1 |
| | de | 1 | 1 | 1 | 1 | 1 | 1 |
| | es | 2 | 1 | 2 | 3 | 3 | 2 |
| | fr | 0 | 1 | 2 | 1 | 1 | 0 |

- **Our code reproduces the paper in English:** rank 100 keeps 1 / 1 / 2 features for
  World War II / Cannabis / Gun, the paper's counts, with the same content (Hitler, Nazi,
  Holocaust; cannabis, hemp, CBD; gun, pistol, firearm + shooting).
- **Counts are not monotone in rank** (en World War II: 1, 1, 0, 0, 1, 1): each rank is a
  new fit, and the kept features sit near the judge's 0.85 cut-off. Read a language across
  ranks, not at one rank.
- **de / es / fr get features mostly through tokens shared with English:** World War II =
  Hitler, Mussolini, Stalin, nazi(s), Luftwaffe, Auschwitz; Cannabis = cannabis, THC, CBD,
  marijuana; Gun adds native whole-word tokens (de Waffen, Pist; es armas, pistola, rifle).
- **Russian: features only where the word is a whole token.** World War II and Cannabis:
  0 at every rank (Гитлер, марихуана, конопля are split into pieces). Gun: 1–2 from rank
  300 on, built on whole tokens оружие / оружия / оружи (plus "gun", "pistolet", писто).
  This is the tokenization explanation, now seen inside one language.

### Edit coverage per language (2026-10-18)

`experiments/multilingual_fit/edit_coverage.py` (CPU): reproduces each edit's token set as
`ember/erasure/embed_edit.py` selects it (concept-only tokens of V′ with non-zero G on a kept
feature; equals Part 1's `n_tokens_edited` for all 18 concepts) and measures, per language,
**question coverage** (share of the 50 `QA_test` questions whose question or correct answer
contains an edited token) and vocabulary coverage. Output `mf_outputs/rank_search/coverage/`.

**Coverage predicts the Part 1 leak.** Part 1 EMBER edit, mean over 18 concepts:

| | en | ru | de | es | fr |
|---|---|---|---|---|---|
| question coverage | 0.74 | **0.04** | 0.38 | 0.34 | 0.39 |
| leak | 0 | **+0.33** | +0.10 | +0.19 | +0.15 |

Over the 70 reliable (concept, non-English language) cells, leak vs question coverage:
Spearman −0.56 (p ≈ 5·10⁻⁷); vs the English-minus-L coverage gap +0.61. Example, Cannabis:
de / es / fr coverage 0.86–0.90 (shared tokens " Cannabis", THC, CBD) with leak ≈ 0;
ru coverage 0.08 with leak +0.51.

**New fits, question coverage** (ranges over ranks 100–800):

| | en-only edit | arm A joint | arm B union |
|---|---|---|---|
| World War II ru / de / es / fr | 0 / .12–.16 / .08–.12 / .08–.12 | nothing at any rank | 0 / .10–.36 / .08–.48 / .08–.56 |
| Cannabis ru / de / es / fr | .06–.08 / .86 / .90 / .90 | 0 below rank 400; from 400: **.84–.86** / .90 / .90 / .90 | .08 / .88–.92 / .90 / .90 |
| Gun ru / de / es / fr | 0 / .04 / .08–.20 / .02–.14 | (not fit) | 0 below 300; from 300: **.54–.62** / .26–.48 / .54–.82 / .20–.66 |

- Arm B raises de / es / fr coverage for World War II and Gun; for Cannabis the English
  edit already covers them (shared tokens).
- Russian is reached only where the fit finds Russian tokens: Cannabis joint (rank ≥ 400)
  through **one piece " кан"** (41 of 50 questions; from каннабис), Gun arm B (rank ≥ 300)
  through " оружие" plus pieces (" стре", " огне", " ство").
- Coverage counts any edited token, including word pieces with other uses: " кан" occurs
  in 6 of 300 random Russian articles (кантон, канделя), " стре" in 14 (стремя, стрелять).
  Whether such edits erase without collateral damage needs the erasure test.

## Open decisions (before step 6)

- **Erasure method(s):** EMBER (embedding features) first, as in the proposal; SNMF and
  SNMF+EMBER with the MLP features too?
- **Edit strength:** reuse the Part 1 δ / HP per concept, or a new HP search for the new
  feature lists (expensive: Part 1's searches took days)?
- **Union in arm B:** plain union of the five lists, or drop duplicates that hit the same
  tokens?
- **Rank:** rank 100 may be too small for 5 languages in arm A (the first joint fit found
  0 + 2 features; problem 1 in `multilingual_features.md`). A higher rank is a later
  variant, not part of the baseline.

## Later (after the baseline)

Translated sentence sets as a second data source; higher rank for arm A; the other ideas
in [`ideas_and_discoveries.md`](ideas_and_discoveries.md).
