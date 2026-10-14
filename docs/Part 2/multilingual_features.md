# Multilingual feature search — step 1 (World War II)

*Status 2026-10-14. Gemma-2-2b-it, rank 100, seed 42, one concept (World War II),
languages en+ru+de+es+fr. Feature search only; erasure comes later.*

## Goal

The English features (the ones EMBER and SNMF erase) are found from English sentences
only, and the edits leak in other languages (`docs/Part 1/crosslingual_next.md` §2.3).
Here we find the features from **English + Russian + German + Spanish + French sentences
together**, hoping to get concept features that are shared across languages. If that
works, erasing them should also erase the concept in the other languages.

This follows idea 1 in [`ideas_and_discoveries.md`](ideas_and_discoveries.md). The
2026-10-04 test there (embedding level only, a lighter script) was already negative.
This is the full pipeline: both tracks, all 26 MLP layers, then the same feature filter
the English features went through.

## How features are found (two steps)

1. **Matrix factorization** (`ember/train_mf_features.py`). Run the concept and neutral
   sentences through the model and factorize the activations into **rank** features
   (rank = how many features the factorization creates):
   - embedding track (used by EMBER): 100 features in total;
   - MLP track (used by SNMF): 100 features per layer × 26 layers = 2,600.

   For every feature it saves its top tokens and `ratio_abs` (how much more it fires on
   concept sentences than on neutral ones).
2. **Feature filter** (`ember/interpret_features.py`, LLM judge = Gemini Flash-Lite, the
   same judge that filtered the English features):
   - pre-filter: keep features with `ratio_abs ≥ 2.0`;
   - for each, the LLM describes the feature from its top 20 tokens, then decides whether
     the description is *distinctively* the concept (`is_member`, `confidence`);
   - keep `is_member = true` and `confidence ≥ 0.85` → `potential_features.csv`. This
     is the list erasure reads.

## The data

Built by `experiments/multilingual_fit/fetch_wiki_sentences.py` from live Wikipedia
(2026-10-04). Per language: 300 concept + 300 neutral sentences.

| | English (`data/`, from the paper) | ru / de / es / fr (`experiments/multilingual_fit/data/`) |
|---|---|---|
| Concept sentences | the concept's Wikipedia article | the same topic's article in that language (via interlanguage links): Вторая мировая война, Zweiter Weltkrieg, Segunda Guerra Mundial, Seconde Guerre mondiale. Sentences of 6–60 words, 300 sampled (seed 42) |
| Neutral sentences | random sentences from random articles | **the first sentence** of 300 random articles |
| Concepts covered | all 18 | only 5: Gun, Heroin, Gambling, Ancient Rome, World War II |

All 5 languages together: 3,000 sentences, 109k tokens (English alone: 600 sentences,
19k tokens). Tokens are balanced across languages (17–22% each).

## What was run

| Step | Run | Result |
|---|---|---|
| 1. Factorization | Mary, 8 Slurm jobs 2026-10-11 → 10-13 (`run_train_mf_features.sbatch`); 7 failed, **1021419 completed** | output complete: embedding + 26 MLP layers |
| 2. Filter | 2026-10-14, Gemini, 270 calls, 33 s | `interpretations/` written |

Outputs (same layout as the English `mf_outputs/google_gemma-2-2b-it/`):
`mf_outputs/multi_language/en+ru+de+es+fr/google_gemma-2-2b-it/{csvs,pickles,interpretations}/rank100/seed42/World_War_II/`.

**Factorization cost** (job 1021419, 4× RTX 2080 Ti, 2h21m in total): model load ~10 min,
activations 6.5 min, embedding track 21 s, MLP track ~2 h (26 per-layer fits, ~1.5–18
min each, slowest in layers 0–5).

**How the failed jobs were fixed** (Mary's changes):

| Job | Failure | Fix |
|---|---|---|
| 1010008, 1010788, 1011014, 1011072 | unknown (logs gone) | — |
| 1011433 (2 GPUs) | out of GPU memory on the output logits | 4 GPUs |
| 1011475 (4 GPUs) | TransformerLens puts 26 // 4 = 6 layers per GPU, so layers 24–25 go to a 5th GPU | even layer split, `external/snmf/llm_utils/activation_generator.py` |
| 1014199 | out of GPU memory in the factorization loss (full 3.6 GiB reconstruction) | loss computed in chunks, `external/snmf/factorization/seminmf.py` |

⚠ These two fixes are **not committed**: they are local changes inside the
`external/snmf` submodule. Without them the run cannot be reproduced.

## Results

| | Candidates (ratio ≥ 2): embedding / MLP | **Kept: embedding (EMBER)** | **Kept: MLP (SNMF)** |
|---|---|---|---|
| English (downloaded features) | 45 / 150 | 1 ("Nazi regime": Hitler, Führer, Reichstag, Mussolini …) | 15 (layers 2–18) |
| **Multilingual** | 33 / 51 | **0** | **2**, both exactly at the 0.85 cutoff |

The 2 kept MLP features: layer 1 "military offensives and occupations, Axis forces"
(' Axis', ' offensive', ' немец', ' наступ'); layer 5 "wartime alliances, treaties and
surrender".

**Too few to be useful.** EMBER would have nothing to edit, and SNMF 2 weak features
instead of 15.

The multilingual features *are* cross-lingual, but they are about war in general or
about grammar, not about World War II. Top candidates, all rejected by the judge with
confidence ≥ 0.95:

| Feature | ratio | Top tokens |
|---|---|---|
| armed forces | 8.0 | military, army, militares, militar, militaire, soldiers |
| attack | 7.7 | attack, attaque, ataque, attacked, atacar |
| loss / destruction | 6.8 | loss, lost, pérdida, Verlust, destruction |
| past-tense verbs | 6.6 | were, podían, obtuvo, llegaron, murieron |
| "however" | 6.3 | however, obstante, Однако, toutefois, although |
| "later" | 5.7 | later, afterwards, subsequently, später, затем |

No embedding feature for Hitler / Nazi / Führer / Stalin / Holocaust / 1939 / 1945 exists
at all. The closest is "German / allemand / Alemania / Deutschland", at ratio 1.47, too
weak to reach the judge.

## Problems found

1. **Rank 100 is too small for 5 languages.** Same 100 features for 5.7× more tokens
   and 5 languages. The factorization spends its features on the largest structure shared
   by all the data (general war vocabulary, grammar), and specific directions like the
   English "Nazi regime" feature get merged into "war".
2. **The foreign neutral sentences are a different kind of text.** English neutral
   sentences are random sentences; foreign ones are article opening lines, i.e.
   definitions ("X is a commune in France"):

   | | en | de | es | fr | ru |
   |---|---|---|---|---|---|
   | definitional neutral sentences | 24% | 60% | 79% | 79% | 90% |

   So in 4 languages "concept vs neutral" really compares narrative history text with
   dictionary definitions. Style features (past tense, "however", "later") then look
   concept-specific and pass `ratio ≥ 2`. Those are exactly the top candidates above.
3. **Broken neutral sentences.** The sentence splitter breaks on German and Russian dates
   ("1. Juni …"), leaving fragments: "Juni 1954 in Waren an der Müritz), ist eine …",
   "April 1922 in San Diego, Kalifornien; † 29.", "28 июля 1951, Кристофер, Иллинойс) — …".
4. **The judge sees Russian MLP tokens as bare fragments** ('бар', 'диви', 'мий'), with no
   sentence around them, and calls such features "random subword tokens".
5. **Foreign sentence sets exist for 5 of 18 concepts only.** The raw articles for all
   18 are now downloaded, but many give fewer than 300 sentences (see "Data check").

Not a problem: Wikipedia reference lines in the concept sentences. There are a few
(8 of 1,200 foreign World War II sentences, 17 of 5,400 English sentences across all
concepts), too few to matter.

## Data check: do articles exist for all 18 concepts? (2026-10-14)

`experiments/multilingual_fit/fetch_wiki_articles.py` downloaded the **raw article text**
(no sentence splitting) for all 18 concepts in en/ru/de/es/fr into
`experiments/multilingual_fit/wiki_raw/<lang>/<Concept>.json` (with page and revision
ids; `report.json` = titles and checks). `data/` is not touched.

**Every concept has an article in all 5 languages** (90 articles). Gun needs a detour:
en:"Gun" has no links to other languages, so ru/de/es/fr come via en:"Firearm".
Some foreign articles are a slightly different topic: de "Römisches Reich" (Roman Empire)
for Ancient Rome; ru/de/es/fr "Ireland" (the state article) for Republic of Ireland;
ru/de/es/fr recreational-cannabis articles for Cannabis.

**How the English sentences were made.** The paper: "we sample n = 300 sentences from
concept-related Wikipedia **pages**" (plural), and neutral sentences are random sentences
from English Wikipedia (2024 dump). Share of our 300 English sentences found in today's
main English article: ≥ 0.83 for 14 concepts, but Artificial intelligence 0.69, Cannabis
0.56, Gun 0.32 and Gambling 0.23. The missing Gun sentences are about specific models
(Walther PPK, derringer, Browning) and the Gambling ones about casinos, so for these
concepts the English set also used related articles.

**One article per language is often not enough.** Usable sentences (6–60 words, current
splitter) per article; `*` = fewer than the 300 the English set has:

| Concept | en | ru | de | es | fr |
|---|---|---|---|---|---|
| Ancient Rome | 707 | 253* | 531 | 346 | 1152 |
| Artificial intelligence | 567 | 480 | 486 | 317 | 654 |
| Baseball | 425 | 193* | 480 | 272* | 396 |
| COVID-19 pandemic | 620 | 207* | 1002 | 76* | 815 |
| Cannabis | 365 | 467 | 462 | 226* | 188* |
| Culture of Greece | 345 | 61* | 7* | 210* | 220* |
| Gambling | 138* | 136* | 260* | 53* | 191* |
| Golf | 345 | 123* | 225* | 256* | 381 |
| Gun | 150* | 277* | 63* | 169* | 71* |
| Halloween | 307 | 172* | 136* | 79* | 269* |
| Harry Potter | 390 | 178* | 363 | 191* | 539 |
| Heroin | 341 | 425 | 310 | 139* | 173* |
| Nazism | 640 | 324 | 347 | 276* | 241* |
| Pornography | 652 | 117* | 371 | 172* | 369 |
| Republic of Ireland | 528 | 435 | 418 | 378 | 362 |
| Uranium | 339 | 268* | 302 | 55* | 251* |
| Valentine's Day | 318 | 129* | 312 | 74* | 97* |
| World War II | 543 | 830 | 1850 | 2307 | 944 |

Only 3 concepts (Artificial intelligence, Republic of Ireland, World War II) reach 300 in
every language. Worst: Culture of Greece de (7) and ru (61), Gambling es (53), Uranium es
(55), Gun de (63). To match English, the foreign sets need related articles too, or the
design must accept fewer sentences for some (concept, language) pairs.

## Open

- **Pipeline check:** the English features were downloaded (fit by the paper's authors),
  not fit with our code. A rerun with `--languages en` (to a separate `--outdir`) would
  show whether our pipeline reproduces 1 + 15 kept features. If it does not, part of the
  problem is the pipeline, not the languages.
- **Neutral sentences:** fix problems 2–3 (random sentence from random articles, as in
  English; a splitter that handles dates), then refit World War II. To be decided.
- **Rank:** refit at a higher rank (e.g. 300) to give the fit room for specific features.

## Reproduce

```
# 1. factorization (4× RTX 2080 Ti, ~2.5 h)
sbatch run_train_mf_features.sbatch --concepts "World War II" --languages en+ru+de+es+fr

# 2. filter (no GPU, ~30 s, Gemini key from .env)
umask 002
.venv/bin/python -m ember.interpret_features --concepts "World War II" \
    --outdir mf_outputs/multi_language/en+ru+de+es+fr
```
