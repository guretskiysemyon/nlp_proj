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

## What the paper says about the data

From `docs/references/Main Article.pdf`. It is brief:

- **Concept sentences (S_C):** "we sample n = 300 sentences from concept-related Wikipedia
  pages"; appendix: "we scrape the corresponding Wikipedia pages". It does not say how
  related pages were chosen, and describes no filtering of off-topic sentences.
- **Neutral sentences (S_N):** "another n sentences from English Wikipedia", "a general
  distribution unrelated to C". Random sentences, nothing more.
- **Rank and size:** chosen by inspecting features on 11 concepts, k ∈ {100, 200, 300}
  and |S| ∈ {100, 200, 300}: k = 100 with 300 sentences gave "semantically coherent,
  concept-specific features without over-fragmenting the vocabulary". Tuned for English
  (600 sentences per fit), not for a 5-language fit.
- **Multilingual:** only in Limitations: English-token edits "would not suppress the same
  concept when accessed through tokens in other languages"; "extending EMBER to
  multilingual settings, for example by drawing S_C from Wikipedia pages in multiple
  languages, is a natural direction for future work." So native Wikipedia, not translation.
- **Check:** the paper's Table 3 (features kept after the LLM filter) gives World War II on
  Gemma 1 embedding + 15 MLP-activation + 0 MLP-projection features, the same as the
  English features we compare against.

## Trial: adding related articles (2026-10-14)

Goal: bring every (concept, language) pair to ≥ 300 usable sentences with native
Wikipedia text, as the English sets did with "concept-related pages".

**Method** (`experiments/multilingual_fit/fetch_related_articles.py`). For each pair whose
main article has < 450 usable sentences (450 = room to sample 300): take Wikipedia's own
similarity search (`morelike:<main article>`, top 50) in that language, keep the
articles the main article also links to, and download them in similarity order until
main + related reach 450 (max 15 related). Hand exclusions are read from
`experiments/multilingual_fit/related_exclude.json` (title → reason). Output:
`wiki_raw/<lang>/related/<Concept>/`, details in `wiki_raw/related_report.json`.

**Round 1 (no exclusions).** 57 of 72 pairs needed related articles; 190 downloaded;
every pair reached ≥ 454 sentences. Most are on topic (Gambling: casino, betting,
gambling addiction; Cannabis: THC, health effects; Uranium es: enrichment, U-235/238).
About 35 drifted, mostly into **neighbour topics the SimDom questions test** (other
holidays for Halloween / Valentine's Day, other sports for Golf / Baseball, other
elements for Uranium, other countries' cultures for Culture of Greece, other weapons for
Gun) or off topic (film plots, biographies, a town, "love" in general).

To decide the borderline ones we measured the **concept-mention rate**: the share of an
article's sentences that name the concept in that language (main articles: median 0.43,
min 0.09):

| Article (concept, lang) | Mention | Decision |
|---|---|---|
| Sóftbol (Baseball, es) | 0.04 | drop (English Baseball: softball in 4/300) |
| Halloween films ×2 (ru), Valentinstag (Film) (de) | 0.03–0.24 | drop (film plots) |
| Morfina, Codeína (es), Morphine (fr) | 0.01–0.04 | drop (other drugs, 81–357 sentences each) |
| Opioide (es), Opioïde, Opiacé, Buprénorphine (fr) | 0.00–0.12 | keep (English Heroin mentions opioids in 108/300) |
| Amor, Amigo invisible, Día del Amigo (Valentine's, es) | 0.00–0.02 | drop |
| Roquemaure (Gard), Othon III de Grandson (Valentine's, fr) | 0.02–0.03 | drop (town, poet) |
| Terry Gander (Gun, de) | 0.14 | drop (historian biography) |
| COVID-19 country pages (ru) | 0.14–0.35 | keep |
| 5 Greek-language articles (Culture of Greece, ru) | high | drop (language, not culture) |

**Rounds 2 and 3 (refill after exclusions).** Each refill takes the next candidates, and
brought in new drift: Easter, Palm Sunday, Lent, Christmas, Epiphany, Candlemas, All
Souls', Day of the Dead, International Kissing Day (holiday concepts); fentanyl,
oxycodone, codeine, "drugs" (Heroin); hurling, field hockey (Golf); Montenegrin culture
and the whole "Greece" country article (Culture of Greece); Michael Myers, a
disambiguation page, the given name "Valentin". After round 2 the exclusion list has 53
entries. The round-3 drift is **downloaded but not yet excluded**. Round 3 widened the
candidate list to the top 150 for Gun de and Valentine's Day; Gun de (267) and
Valentine's Day es (248) still fell short of 300.

**Conclusion: hand curation does not converge.** For holidays and drugs the next most
similar articles *are* other holidays and drugs; short main articles run out of good
candidates.

**Article-level threshold does not separate them either.** Drifted articles score
0.00–0.03, but so do legitimate ones that rarely name the concept: Samhain and Lupercalia
(origins of Halloween / Valentine's Day, 0.00–0.09), baseball terminology (0.04), golf
clubs (0.05).

**Sentence-level filter (estimate).** Keep all main-article sentences, and from related
articles only sentences that mention the concept. Drift then contributes ~0 sentences.
With the articles downloaded so far, 16 of 72 pairs stay under 300 (Valentine's Day es 102,
Halloween es 132, Gun de 150, Halloween de 156, Valentine's Day ru 165 / fr 174, Heroin es
196, Culture of Greece ru 200, 8 more at 227–293). Bias: related sentences all name the
concept. The English sets name it in 31–77% of sentences (Halloween 0.31 … Gambling
0.77); foreign sets built this way would be ~0.4–0.75, inside that range.

## Options for the foreign sentence sets

Why the concept set must be *about* the concept: the factorization and the filter keep
features that fire more on concept than on neutral sentences. A Christmas sentence in the
Valentine's set teaches "holidays in general", and those features then get erased.
Sentences from the concept's own article are fine even when they do not name it; sentences
from related articles have no such guarantee.

**Why not drop sentences that match the SimDom questions?** (1) It contaminates the
evaluation: SimDom questions measure specificity, and using them to pick training data
removes exactly what is tested (true for `SimdomQA_train` too, since train and test cover
the same neighbour topics). (2) It only covers what the questions happen to cover; much
drift is not in SimDom ("love", hurling, biographies, the Greece country article).
(3) Matching sentences to questions across 4 languages needs its own judge. It is useful
as an after-the-fact check of finished sets, not as the filter.

| # | Approach | Pros | Cons |
|---|---|---|---|
| 1 | Main article only, accept fewer sentences | simple, no drift | thin pairs get very little (Culture of Greece de: 7) |
| 2 | Hand-curate related articles (tried) | transparent | did not converge; subjective; 72 pairs |
| 3 | Article-level mention threshold | automatic | misjudges Samhain, Lupercalia, terminology pages |
| 4 | Sentence-level keyword filter | automatic, drift → ~0 | names-the-concept bias; 16 pairs < 300 |
| 5 | Sentence-level LLM / embedding judge ("is this sentence about X?") | keeps relevant sentences that do not name the concept | a judge in the data step; a few dollars with Gemini |
| 6 | Translate the English sets (concept **and** neutral) into ru/de/es/fr | same content in every language, only language differs; no drift; fixes the neutral-style problem; matches the cross-lingual test (translated questions) | translated, not native text (the paper suggests native); needs spot checks. ~22,800 sentences, ~1,100 Gemini calls, a few dollars |

Option 6 is the cleanest test of "does a joint fit find language-independent concept
features". Native Wikipedia (options 4–5 on the downloaded articles) follows the paper's
suggestion and can serve as the main result or as a robustness check. Not decided yet.

## Open

- **Data for the foreign sets:** native (option 4 or 5), translation (option 6), or both.
  Then the neutral sets: random sentences from random articles (as in English and the
  paper), not article opening lines, with a splitter that handles dates.
- **Pipeline check:** the English features we compare against match the paper's Table 3,
  but they were downloaded, not fit with our code. A rerun with `--languages en` (to a
  separate `--outdir`) would show whether our pipeline reproduces 1 + 15 kept features.
- **Rank:** refit at a higher rank (e.g. 300) once the data is fixed.

## Reproduce

```
# data (no GPU, Wikipedia API, a few minutes each; never touch data/)
.venv/bin/python -I experiments/multilingual_fit/fetch_wiki_articles.py      # main articles, 18 x 5 languages
.venv/bin/python -I experiments/multilingual_fit/fetch_related_articles.py   # related articles (+ related_exclude.json)
```

```
# 1. factorization (4× RTX 2080 Ti, ~2.5 h)
sbatch run_train_mf_features.sbatch --concepts "World War II" --languages en+ru+de+es+fr

# 2. filter (no GPU, ~30 s, Gemini key from .env)
umask 002
.venv/bin/python -m ember.interpret_features --concepts "World War II" \
    --outdir mf_outputs/multi_language/en+ru+de+es+fr
```
