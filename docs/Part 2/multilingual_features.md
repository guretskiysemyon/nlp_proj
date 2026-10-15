# Multilingual feature search — step 1 (World War II)

*Status 2026-10-15. Gemma-2-2b-it, rank 100, seed 42, one concept (World War II),
languages en+ru+de+es+fr. Feature search only; erasure comes later.*

*Where things stand:* the first fit (2026-10-14) gave too few features (0 + 2), mostly
because of the data. The foreign sentence sets are now being rebuilt along two tracks:
**native Wikipedia** (`data_v2/` usable, 54 of 72 pairs complete; extra pool for all 18
short pairs collected, LLM judge next; see "Native sentence sets") and **translation**
of the English sets (complete for all 18 concepts, quality checks next; see "Translated
sentence sets"). No new fit yet.

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
   English "Nazi regime" feature get merged into "war". → **open** (refit at a higher
   rank once the data is fixed).
2. **The foreign neutral sentences are a different kind of text.** English neutral
   sentences are random sentences; foreign ones are article opening lines, i.e.
   definitions ("X is a commune in France"):

   | | en | de | es | fr | ru |
   |---|---|---|---|---|---|
   | definitional neutral sentences | 24% | 60% | 79% | 79% | 90% |

   So in 4 languages "concept vs neutral" really compares narrative history text with
   dictionary definitions. Style features (past tense, "however", "later") then look
   concept-specific and pass `ratio ≥ 2`. Those are exactly the top candidates above.
   → **fixed** in `data_v2/` (one random sentence from each random article, as in English)
   and in the translated sets (translations of the English neutral set).
3. **Broken neutral sentences.** The sentence splitter breaks on German and Russian dates
   ("1. Juni …"), leaving fragments: "Juni 1954 in Waren an der Müritz), ist eine …",
   "April 1922 in San Diego, Kalifornien; † 29.", "28 июля 1951, Кристофер, Иллинойс) — …".
   → **fixed**: new splitter `sentences.py`, used by everything in `data_v2/`.
4. **The judge sees Russian MLP tokens as bare fragments** ('бар', 'диви', 'мий'), with no
   sentence around them, and calls such features "random subword tokens". → **open**.
5. **Foreign sentence sets exist for 5 of 18 concepts only.** The raw articles for all
   18 are now downloaded, but many give fewer than 300 sentences (see "Data check").
   → **partly fixed**: native `data_v2/` covers all 18 concepts, 54 of 72 pairs with 300
   sentences, 18 short; the translated sets are complete for all 18.
6. **Wikipedia reference lines** (found 2026-10-15). Not a problem in the World War II
   sentences of the first sets (8 of 1,200 foreign sentences, 17 of 5,400 English sentences
   across all concepts). But the plain-text extracts contain the bibliography and reference
   sections, and ~2% of usable sentences in all main and related articles are such lines.
   The first `data_v2/` had 15–185 of them per language in the concept sets (de worst,
   3.7%) and 4–19 of 300 in the neutral sets. → **fixed**: `sentences.py` drops them, and
   `data_v2/` was rebuilt (see "Native sentence sets").
7. **Zero-width spaces in the Spanish text** (found 2026-10-15). The es plain-text
   extracts keep a U+200B ZERO WIDTH SPACE where each footnote marker was
   ("… los santos.<U+200B><U+200B> Algunas teorías …"); the en/ru/de/fr extracts have none.
   The splitter needs a normal space after the period, so the sentences around a footnote
   are glued into one piece of > 60 words, or a sentence ends in the invisible character
   and fails the "ends in . ! ?" check. Halloween es: the extract is 50,703 characters,
   but 61% of the text ends up in > 60-word pieces and 12% fails the end check, leaving
   79 usable sentences (299 without U+200B). All 18 es main articles: 5,506 → 7,180 usable
   sentences (+30%). Kept sentences can still contain it (footnote mid-sentence): 292 of
   4,759 es concept sentences and 25 of 300 es neutral sentences in the old `data_v2/`; also in
   the first sets (`data/`). Affects every es set: concept, neutral (a sentence before a
   footnote could not be picked), related and extra articles; not the translated sets.
   → **fixed** in `data_v2/`: `sentences.py` removes U+200B before splitting (a space
   where it stood between two words), the es concept sets were rebuilt, and the es neutral
   set was re-drawn from new random articles: the old draw had skipped 281 of 581 articles
   as too short, more than any other language, so the bug had also biased which articles
   got in (new draw: 180 of 480). es short pairs 6 → 2 (see "Native sentence sets").
   Still in the first sets (`data/`), which are not touched.

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

The es column is too low because of zero-width spaces (problem 7, found and fixed later);
without them, e.g., Halloween es has 299. The short ru/de/fr articles really are short: 64–98% of
their text is usable, and the rest is headings, list items and lead-ins ending in ":".

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

**Sentence-level filter (estimate; since built as `data_v2/`, see "Native sentence
sets").** Keep all main-article sentences, and from related
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
| 4 | Sentence-level keyword filter (**built**: `data_v2/`) | automatic, drift → ~0 | names-the-concept bias; 18 pairs < 300 |
| 5 | Sentence-level LLM / embedding judge ("is this sentence about X?") (**next**: pool for the short pairs collected) | keeps relevant sentences that do not name the concept | a judge in the data step; a few dollars with Gemini |
| 6 | Translate the English sets (concept **and** neutral) into ru/de/es/fr (**built**, checks in progress) | same content in every language, only language differs; no drift; fixes the neutral-style problem; matches the cross-lingual test (translated questions) | translated, not native text (the paper suggests native); needs spot checks. ~22,800 sentences, ~1,100 Gemini calls, a few dollars |

Option 6 is the cleanest test of "does a joint fit find language-independent concept
features". Native Wikipedia (options 4–5 on the downloaded articles) follows the paper's
suggestion and can serve as the main result or as a robustness check. Decided 2026-10-15:
both, in parallel (next two sections).

## Native sentence sets (options 4–5): status (2026-10-15)

Two parts: `data_v2/` (option 4, usable now, 18 pairs short) and an extra article pool
for the short pairs (collected; to be filtered by an LLM judge, option 5).

### `data_v2/`: keyword filter, new splitter, new neutral sets

Output `experiments/multilingual_fit/data_v2/`, in the loader's format (use with
`--multilingual-data-dir experiments/multilingual_fit/data_v2`). `data/` is never touched.

| Part | Script | What changed from the first sets (`data/`) |
|---|---|---|
| Splitter | `sentences.py` | A period does not end a sentence after a German ordinal ("1. Juni"), an initial, a German Roman numeral or a known abbreviation of the language (problem 3). Drops reference lines (problem 6). Removes zero-width spaces first (problem 7). Keeps 6–60 words, as before. |
| Concept sets | `build_concept_sets.py` | Option 4: every usable main-article sentence; from related articles only sentences that name the concept (per-language stems, `STEMS`). Main article first, related sentences only to fill up to 300. Fixed seed per pair; the source article of every sentence is saved. |
| Neutral sets | `fetch_neutral_sentences.py` | As in English and the paper: one random usable sentence from each random article of ≥ 3 KB, not the first sentence (problem 2). 376–530 random articles drawn per language (es re-drawn after the zero-width fix). |

Result: 20,516 concept sentences + 4 × 300 neutral sentences. **54 of 72 pairs have 300,
18 are short** (empty cell = 300; † = short since the keyword fix below):

| Concept | ru | de | es | fr |
|---|---|---|---|---|
| Baseball | 272 | | | |
| Culture of Greece | | 294 | | 269 |
| Gambling | 291 | 299 | | 287 † |
| Gun | | **108** | | 282 |
| Halloween | 292 | **127** | | 296 |
| Harry Potter | 232 † | | | |
| Heroin | | | 272 | 278 |
| Nazism | | | | 296 |
| Valentine's Day | **154** | | **121** | **146** |

**Zero-width fix (2026-10-15, problem 7).** es only: Halloween 102 → 300, COVID-19
pandemic 194 → 300, Baseball 299 → 300, Harry Potter 291 → 300, Heroin 192 → 272,
Valentine's Day 81 → 121. The other es pairs stay at 300, with more main-article text
(e.g. Gambling es 51 → 95 main-article sentences, Uranium es 54 → 71). Outside es, one de
sentence changed (an IPA transcription with word joiners).

**Keyword fix (2026-10-15).** We listed the words that matched the concept regex
(`STEMS`) in all 4,113 related-article sentences in use. Almost all were real mentions.
Changed:
- Gambling: `azar` and `pari` matched inside other words (es "rechazar", "aparición"),
  and `pari` matched fr "paris" (plural "bets", also the city). Now `azar` must be a whole
  word, and `pari` matches only "pari" and its verb forms, not "paris".
- Harry Potter: Роулинг / Rowling removed. 105 of the 123 ru related sentences in use
  named Rowling, i.e. the author, not the books.

Effect: Harry Potter ru 300 → 232, Harry Potter es 300 → 291 (300 again after the
zero-width fix), Gambling fr 300 → 287. Gambling es stayed at 300, with a different
sample of 90 related sentences. All other pairs were unchanged.

**Reference-line filter.** `sentences.py` drops a sentence with: ISBN / ISSN / doi / URL;
page or volume numbers (p. 4, S. 10–15, Bd. 26, — С. 17, — 269 с.); a Russian imprint
(— М.:); an access date (Consultado el, Abgerufen am, consulté le, Retrieved, Проверено …);
a citation language tag ("(англ.)", a leading "(en)"); "(Hrsg.)"; a leading "In:"; or a
publisher word followed by a year. Checked by hand on samples. Bare "Verlag" and "(лат.)"
were taken out of the pattern because they also occur in normal prose ("…der Verlag die
Veröffentlichung zusagte", "Hippodamia oregonensis (лат.) — вид …"). After the filter:
- concept sets rebuilt; counts dropped in some pairs (Gun de 127 → 108, Halloween de 138 →
  127), and 2 pairs fell just short (Gambling de, Nazism fr);
- neutral sets: the 45 rejected sentences (ru 10, de 19, es 4, fr 12) were replaced by
  another random sentence of the same article (`fetch_neutral_sentences.py --repair`,
  logged under `repaired` in `neutral_report.json`); the other 255–296 per language are
  unchanged. The es set was later re-drawn from scratch (problem 7), so its repair log is
  gone;
- a few reference lines still pass (e.g. "… en Rotten Tomatoes (en inglés).").

### Extra article pool for the short pairs

The related articles (`morelike`) are used up for the short pairs, and mostly off topic.
Halloween de: 22 related articles, 2,199 sentences, 19 name Halloween (Kelten, Silvester,
Irland, Martinstag …). Halloween es: 24 of 1,334 (Navidad, Cuaresma, Purgatorio …).
Valentine's Day es: 10 of 165. So the extra sentences need new sources.
`fetch_extra_articles.py` collects two:
1. **category**: pages in the concept's own category, found through the English
   category's interlanguage link plus the main article's categories that match the stem;
2. **search**: the top 100 full-text search results of each query.

Whole articles are kept, with how each was found, in `wiki_raw/<lang>/extra/<Concept>/`
(`wiki_raw/extra_report.json`). Nothing is filtered at this step; an LLM judge filters
the sentences next.

**First attempt: all 18 concepts, broad (stopped).** Categories with one level of
subcategories, search for the unquoted main title. Stopped after 6 pairs (output kept in
`wiki_raw/_extra_broad/`) because the quality was low:
- Valentine's Day es: namesakes (Rodolfo Valentino, people named Valentín / Valentina,
  the St. Valentine's Day Massacre gangsters, the band Bullet for My Valentine). None of
  6 sampled sentences that name the concept were about the holiday.
- Halloween de: Kategorie:Halloween is mostly the film franchise (Michael Myers, the
  actors, other horror films).
- Ancient Rome ru: on topic but skewed. The subcategories are mostly short consul
  biographies, plus bibliography lines (which led to problem 6).
- Size: ~3,000–20,000 sentences per pair, an estimated 0.5–0.8 M for all 72 pairs.

Decision: no extra pool for pairs whose main article already gives 300. Main articles
are the cleanest source, and the English sets are also mostly main-article text (≥ 0.83
for 14 concepts, "Data check"). So "main article first, extras only to fill the gap" is
the same rule for every concept.

**Second run: short pairs only, targeted (done 2026-10-15).** The 19 pairs that were
short at the time (before the keyword fix and before problem 7 was found).
Categories without subcategories (max 150 pages). Per-pair queries (`QUERIES`):
- exact phrases and alternative names ("Día de los enamorados", "День всех влюблённых",
  "fête des amoureux", "Noche de Brujas", "Schusswaffe");
- `-intitle:` film / Film / película / фильм for the holidays;
- a drug term next to heroína / héroïne, which also mean "heroine".

2,832 articles, ~1 article/s.

**Third run (2026-10-15), marked †:** the 2 pairs short since the keyword fix, with their
own queries: Gambling fr ("jeu d'argent", "jeux d'argent", "jeux de hasard") and Harry
Potter ru ("Гарри Поттер", "Гарри Поттера", "Хогвартс"; the ru main article is the novel
series, "Серия романов о Гарри Поттере"). 442 articles.

Have / Need are the current `data_v2/` counts, and the es rows are recounted after the
zero-width fix (problem 7). Baseball es, COVID-19 pandemic es and Halloween es now reach
300 without the pool.

| Pair | Have | Need | Articles | Usable sentences | Name the concept (from category / search) | On topic, of 6 sampled naming sentences |
|---|---|---|---|---|---|---|
| Baseball es | 300 | 0 | 100 | 5,048 | 690 (34 / 656) | good |
| Gambling de | 299 | 1 | 199 | 7,166 | 1,313 (823 / 490) | not read |
| Nazism fr | 296 | 4 | 200 | 13,937 | 2,408 (1,132 / 1,276) | not read |
| Halloween fr | 296 | 4 | 103 | 4,767 | 594 (7 / 587) | mostly films |
| Culture of Greece de | 294 | 6 | 141 | 10,567 | 1,725 (182 / 1,543) | mixed (ancient Greece, history) |
| Halloween ru | 292 | 8 | 101 | 1,970 | 167 (11 / 156) | mostly films, The Simpsons |
| Gambling ru | 291 | 9 | 184 | 10,211 | 978 (489 / 489) | 4/6 |
| Gun fr | 282 | 18 | 176 | 4,731 | 1,352 (1,065 / 287) | 6/6 |
| Heroin fr | 278 | 22 | 82 | 4,631 | 233 (0 / 233) | 3/6 (also "heroine": Bovary, a Marvel super-héroïne) |
| Baseball ru | 272 | 28 | 100 | 1,587 | 311 (92 / 219) | 5/6 |
| Culture of Greece fr | 269 | 31 | 155 | 14,696 | 2,849 (207 / 2,642) | mixed |
| COVID-19 pandemic es | 300 | 0 | 186 | 13,185 | 2,254 (402 / 1,852) | 5/6 |
| Heroin es | 272 | 28 | 78 | 3,787 | 236 (0 / 236) | 5/6 |
| Gun de | 108 | 192 | 187 | 8,962 | 3,235 (377 / 2,858) | 5/6 |
| Halloween de | 127 | 173 | 138 | 6,645 | 776 (643 / 133) | 3–4/6 (trick-or-treating, costumes, poisoned-candy myths) |
| **Valentine's Day fr** | 146 | 154 | 118 | 6,943 | 203 (13 / 190) | ~3/6 |
| **Valentine's Day ru** | 154 | 146 | 130 | 6,149 | 160 (0 / 160) | ~4/6 |
| **Valentine's Day es** | 121 | 179 | 216 | 12,717 | 243 (0 / 243) | ~2/6 |
| Halloween es | 300 | 0 | 238 | 12,927 | 766 (49 / 717) | 0/6 (films, Michael Myers, a wrestler called "Halloween") |
| Gambling fr † | 287 | 13 | 287 | 12,610 | 1,349 (240 / 1,109) | 5/6 (one Hong Kong film, *Casino Raiders*) |
| Harry Potter ru † | 232 | 68 | 155 | 10,224 | 1,749 (353 / 1,396) | 6/6 (books, films, actors) |

**Expected outcome:** 18 pairs still need the pool, and all 18 have one. Most can reach
300 after the judge.
The bold 3 (Valentine's Day fr / ru / es) probably stay short (~150–250): native
Wikipedia in these languages has little text about the holiday beyond film and
pop-culture pages. For them: accept fewer sentences, or top up with translated sentences
(not decided). Halloween es was in this group before the zero-width fix; its main
article alone now gives 299 sentences.

**Problems seen in the pool** (for the judge design):
- Wrong senses that still name the concept: the *Halloween* films and their actors, a
  wrestler called Halloween; Valentino and people named Valentín; the St. Valentine's Day
  Massacre; Saint-Valentin communes; héroïne / heroína as "heroine". The stems used for
  "names the concept" match all of these.
- In some pairs almost all naming sentences come from search, not the category
  (Halloween es 714 of 763; Valentine's Day es and ru all). These depend most on the judge.
- Names-the-concept bias, as in option 4.
- A few reference lines still pass.

**Wikipedia throttling (fixed).** With the old generic User-Agent, Wikimedia answered
HTTP 429 every few requests (~5.7 s per article). A User-Agent with a contact URL (the
repo's GitHub URL, `fetch_wiki_articles.UA`) stopped the 429s (0.4–1 s per article). All
fetch scripts use it. Long fetches are started detached (`setsid nohup`) on the login
node, so they survive a disconnect.

### Link sentences for the short pairs (2026-10-15)

A third source, `fetch_link_sentences.py`: Wikipedia articles that **link to the main
article**, and in each the sentence that holds the link. The link fixes the sense, which
the stems cannot: a link to "Héroïne" is the drug, not a heroine; a link to
"Saint-Valentin" is the holiday, not Valentino or a commune. Every short pair has
hundreds of linking articles (Valentine's Day fr 501, es 409, ru 458; Gun de 383;
Halloween de 642), except Culture of Greece de (60) and fr (147), which need only 6 and
31 sentences.

Method: backlinks in the main namespace, direct and through redirects; a fixed-seed
random sample of 4 × (sentences missing) articles, at least 100 and at most 500; the
rendered page (`action=parse`, so date templates come out as "le 14 février 1613"); only
body paragraphs, not lists, tables, infoboxes or navboxes (4 of 15 sampled Valentine's
Day fr articles gave no sentence); split and filtered with `sentences.py`. Output:
`wiki_raw/<lang>/links/<Concept>.json` (every sentence with its article, page and revision
ids), `wiki_raw/links_report.json`.

Test sample (15 articles each): Heroin fr, 13 sentences from 11 articles, all the drug
("overdose à l'héroïne", "contrebande d'héroïne", "dépendance à l'héroïne"); Valentine's
Day fr, 11 sentences, all the holiday, but mostly passing mentions ("diffusé le 11 février
2005 durant le jour de la Saint-Valentin", "le 14 février 1613, jour de la
Saint-Valentin"), a few about the holiday itself ("Fête des amoureux, elle est parfois
comparée à la Saint-Valentin"). So the judge's question for these is not the sense but
"about the concept, or only a date / name?".

Result: the run is in progress.

## Translated sentence sets (option 6): plan and status

Decided 2026-10-15: build translated sets now (option 6). Native Wikipedia sets
(options 4–5) continue in parallel. All 300 concept sentences for each of the 18 concepts,
plus the 300 neutral sentences, go from English into ru/de/es/fr: 5,700 sentences ×
4 = 22,800 translations. Sentence *i* in every language is the translation of English
sentence *i* (aligned).

**Tool:** `experiments/multilingual_fit/translate_sentences.py`. It calls the Gemini REST
API directly (the installed SDK cannot turn thinking off). It sends 50 numbered
sentences per call with JSON output, temperature 0 and thinking `minimal`. A
chunk is accepted only if every sentence comes back non-empty, in the right
script, with no word mixing Latin and Cyrillic letters, not copied from the English,
and of plausible length; otherwise it is retried, then translated one sentence at a time.
The cache records the model of every chunk, so reruns resume. Output:
`experiments/multilingual_fit/data_translated/{concept,neutral}_sentences_<lang>.json`,
in the loader's format. Use it with
`--multilingual-data-dir experiments/multilingual_fit/data_translated`. `data/` is never touched.

**Model choice: pilot on World War II (2026-10-15).** Both models translated the same
300 sentences into 4 languages:

| | gemini-3.5-flash-lite (**chosen**) | gemini-3.6-flash |
|---|---|---|
| Cost per concept | $0.21 | $0.30 |
| Words also found in the native World War II article (ru/de/es/fr) | 0.67 / 0.81 / 0.90 / 0.80 | 0.68 / 0.82 / 0.90 / 0.80 |
| Established terms (странная война, Sitzkrieg, drôle de guerre, Völkerbund, Société des Nations, Сталинград …) | correct, except es "Guerra de Bromiza" (not a word) | correct |
| Defects found | de "hardlinerischen" (awkward), es "Cuerpos de Ejército" for Army Groups | 2 mixed-script words (ru "Hинденбурга", "КПK"); now caught by a check |

The two models are equally good, so we use the cheaper one. Both dropped the Wikipedia junk
in English sentence 77 ("Cite error: A <ref> tag is missing…"). That is the one known
gap in alignment. The Flash pilot is kept in `data_translated_flash_pilot/`.

**Plan:**

| # | Step | Status |
|---|---|---|
| 1 | Full Flash-Lite run: 18 concepts + neutral, 4 languages | **done** (2026-10-15): $3.79 in total (including the World War II pilot), 612 calls, 0 thinking tokens, ~8 min |
| 2 | Quality score for every (English, translation) pair: COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`, gated on HF, 1 GPU), or LaBSE cosine similarity as the fallback | **done** (2026-10-15): job 1042276, both scores, 16 min on 1 RTX 2080 |
| 3 | Re-translate the low-scoring sentences (Kiwi < 0.7, ~3%) with gemini-3.6-flash and a prompt that asks for standard sports and weapons terms and translated quotations; score again and keep whichever version scores higher | **done** (2026-10-15): 483 of 684 replaced, $0.19 + job 1042302 (2 min) |
| 4 | Check terms against native Wikipedia for all 18 concepts (`wiki_raw/`): key terms in the native forms | **evaluation done** (2026-10-15): 117 term errors in 111 sentences; plus ~890 translations that drop parenthesised text |
| 4b | Fix: re-translate the 111 sentences with the correct term as a hint, and the ~890 with dropped parentheses with a prompt that keeps them; score again; keep if better | **done** (2026-10-15): 827 of 969 replaced, $0.50 + job 1042417 |
| 5 | ~~Human spot-check~~: no human reviewer is available. Replaced by the automatic judges: Kiwi (step 2), the Wikipedia term check plus the Gemini judge (step 4), and judge agreement (3.8 Flash = 3.1 Pro on 40/40) | replaced |
| 6 | *Optional:* Gemma perplexity on translated vs native sentences (translationese check) | optional |
| 7 | Refit World War II on en+ru+de+es+fr with the translated sets, then the same filter; compare with the native-data fit (0 + 2 features) and English (1 + 15) | **next** (the data is ready) |

**Step 1 result.** All 4 languages are complete (18 × 300 concept sentences + 300 neutral sentences, none empty).
The checks flagged 18 of 22,800 translations (`data_translated/flags.json`); all were
read by hand and are acceptable:
- 13 shorter than expected: the translation drops junk in the English source (coordinates
  `53°N 8°W﻿ / …`, LaTeX `{\displaystyle …}`, "Cite error", `[1]`, "This article … is a
  stub") or a redundant gloss ("Zabastovka (English: "strike action")", the German
  original name of a Nazi office repeated in English);
- 5 Russian sentences with < 50% Cyrillic letters: Latin species and product names
  (Tetragonothrips, Neocurtimorda perpusilla, EA Sports …), correct as written.

One real omission was caught and fixed during the run. A list of 14 countries in neutral
sentence 242 was dropped from the Russian translation; the retry restored it. The first
run crashed on that retry: the one-sentence fallback numbered the sentence by its position in the
chunk, so the answer's id never matched. The script now numbers 1…n in every call.

Known issue for step 2: some English neutral sentences are Wikipedia junk (coordinates,
LaTeX, stub notices), and the translation cleans them up, so those pairs are not exactly
aligned. This affects about 5 of the 300 neutral sentences.

**Step 2 result.** `experiments/multilingual_fit/score_translations.py`, run by
`run_score_translations.sbatch`. It runs in a separate environment,
`/home/morg/NLP_2526b/semyong/.venv-comet`, because `unbabel-comet` pins protobuf < 5,
which would break the Gemini SDK in the project `.venv`. Output:
`data_translated/quality/{scores,summary,worst}.csv`.

| | de | es | fr | ru |
|---|---|---|---|---|
| COMET-Kiwi median (5th percentile) | 0.858 (0.737) | 0.881 (0.753) | 0.880 (0.762) | 0.872 (0.728) |
| LaBSE median (5th percentile) | 0.891 (0.808) | 0.927 (0.854) | 0.919 (0.841) | 0.882 (0.794) |
| Kiwi < 0.7 | 171 | 157 | 146 | 211 |

- **The model choice is confirmed:** on World War II, Flash-Lite and Flash score the same
  (Kiwi 0.860–0.886 vs 0.862–0.887).
- Neutral sentences score only slightly below concept sentences (0.83–0.86 vs 0.84–0.86).
- **The low tail is domain jargon.** Of the 685 pairs with Kiwi < 0.7, 647 have a normal
  English source (only 30 are junk, 8 are fragments). They cluster in Baseball (139), Gun
  (126), Golf (119), Halloween (67) and Valentine's Day (35). Reading the worst Russian
  ones shows a mix:
  - real errors: "Жертвенный бантик … привести бегущего в дом" (sacrifice bunt, literal
    and wrong), "Старты «шаган»" (shotgun start), "премуществом" (typo);
  - false alarms: correct but specialist terms (бабки for astragali, боги / берди / игл
    in golf scoring).
- **Untranslated English:** 39 translations still contain long English passages,
  mostly in Valentine's Day (Chaucer, Donne, Shakespeare quotations kept in the original). Some
  are legitimate titles (*The Decline and Fall of the Roman Empire*, *The Wizarding World of
  Harry Potter*). Kiwi rates them low (mean 0.68), so step 3 retranslates them.
- One Spanish proverb was localized instead of translated ("Saint Valentine brings the keys of
  roots" → "por San Valentín, la primavera pone fin").

**Step 3 result.** `experiments/multilingual_fit/retranslate_low.py` re-translated the 685
pairs with Kiwi < 0.7 using gemini-3.6-flash, adding prompt rules for field terminology,
translated quotations, and titles in their native Wikipedia form. One sentence, a LaTeX
fragment of Wikipedia junk, got no candidate. Then `score_translations.py --candidates`
scored the 684 candidates, and `retranslate_low.py --apply` kept the new version where its
Kiwi was higher and it passed the step-1 checks.

- **483 of 684 replaced** (de 126, es 108, fr 105, ru 144); median gain +0.07 Kiwi. Mean
  Kiwi of the 685 went from 0.62 to 0.70. Pairs below 0.7 went from 685 to 401
  (de 85, es 97, fr 91, ru 129).
- **Every error found in step 2 is fixed:** "жертвенный бант … ранер … доставить его домой"
  (was "бантик … привести бегущего в дом"), "одновременный старт («шотган»)" (was
  "«шаган»"), the "премуществом" typo, the Wiccan and Shakespeare quotations (now
  translated), and the Spanish proverb (now literal).
- **The remaining English** (28 translations with ≥ 3 English function words) is titles and
  organisation names (The R&A, *The Glory of Their Times*, *The Wizarding World of Harry
  Potter*), kept as native Wikipedia keeps them.
- **The 401 still below 0.7** are mostly in Baseball (104), Gun (84) and Golf (71). The ones
  read by hand are correct translations with specialist terms that Kiwi scores low (боги /
  дабл-боги / берди / игл). Step 5 checks a sample of them.
- Files: `data_translated/retranslate/replacements.csv` (every candidate, kept or not, with
  old and new text and both scores, so a replacement can be undone), `candidates*.json`,
  `usage.jsonl`. The data files and `quality/scores.csv` now hold the final versions.

**Step 4 result (evaluation).** There is no human reviewer, so the reference is Wikipedia's
interlanguage links (written by people, independent of the translator) and the judge is a
different, stronger model than the translator.

*Term match* (`experiments/multilingual_fit/term_check.py`, output `data_translated/terms/`).
Every `[[link]]` in each concept's English article, plus the article title, gives an English
term and its ru/de/es/fr article titles (cached in `terms/<Concept>.json` and `.wikitext`).
Each English concept sentence containing a term is checked for the native title in its
translation, by stem, to tolerate inflection. Four problems in the first version made most
mismatches artifacts, and were fixed:
- links whose text names a broader or narrower article ("par" → *Glossary of golf*,
  "Italy invaded France" → *Battle of the Alps*): a form now counts only if it is the
  article's own title (plural allowed) or, for names, its last part ("Hitler" → *Adolf Hitler*);
- case-insensitive matching ("who" matched *WHO*, the word "time" matched *Time* magazine);
- names that must match every word ("Гитлер" vs "Адольф Гитлер"): one word is now
  enough for names, and a name kept as written in English counts ("WHO", "Time");
- a stem too strict for 4-letter words ("игра" vs "игры").

Result: 19,316 checks over ~3,800 distinct terms. Names and places (18,241 checks) matched
92–96%; domain terms (1,075 checks, 125 terms such as squeeze play, flintlock, fairway)
matched 51–58%.

*Judge* (`term_judge.py`, gemini-3.8-flash, thinking `low`, $0.82). For each of the 1,565
mismatches the judge sees the English sentence, the translation, the term and its native
Wikipedia title, and answers OK (inflected form, synonym, abbreviation, or a correct
contextual wording) or ERROR, with the correct term for an ERROR.
- Choice of judge: on 40 random mismatches (20 terms, 20 names), gemini-3.8-flash and
  gemini-3.1-pro-preview agreed on all 40 (both found the same 4 errors). The full run would
  have cost ~$0.80 on Flash vs ~$4.30 on Pro. Qwen2.5-32B (local, a different family) was tried but was
  too slow on 3× RTX 2080 Ti (a 32-prompt smoke test had not finished after 20 min) and
  was cancelled; `--backend qwen` and `run_term_judge.sbatch` remain.
- **1,448 OK, 117 ERROR**, in 111 distinct sentences. As a share of all checks, terms are
  correct in 97.7–99.2% and names in 99.2–99.7%:

  | | de | es | fr | ru |
  |---|---|---|---|---|
  | names: errors / accuracy | 32 / 99.3% | 21 / 99.5% | 13 / 99.7% | 35 / 99.2% |
  | domain terms: errors / accuracy | 7 / 97.7% | 2 / 99.2% | 3 / 99.0% | 4 / 98.1% |

- Error types (from the judge's reasons): **omitted** 44 (27 of them inside parentheses in
  the English), **calque** 29 (ru "Восточная империя" for the Eastern Roman Empire,
  "Длительный COVID" for Long COVID, fr "Âges sombres grecs" for *Siècles obscurs*),
  **misspelling or wrong transliteration** 23 (es "Vespaciano", ru "инффлюация", "Йогал" for
  Youghal, "Дол" for Dáil), **other wrong term** 21 (fr "la Toussaint" for All Hallows'
  *Eve*, de "Weißmark" for White Carniola). By concept: Halloween and Valentine's Day 17
  each, Culture of Greece 15, Nazism 12, Ancient Rome 10, the rest ≤ 9.
- The OK verdicts read by hand are right (ru патрон, базы, фервей; fr gérant for a baseball
  manager; a bogus check where the "hit" link meant a base hit is passed with the
  right reason).

*Dropped parentheses* (found while reading the omissions). Of the 802 English sentences that
contain "(…)", the translation has no parentheses in 35% (ru), 32% (de), 26% (fr) and 19% (es):
~890 translations. Usually the dropped part is redundant (unit conversions "(5.9 inches)",
English glosses of a foreign name), but sometimes it is content: acronyms ("(AVLIS)"),
original-language names ("Alla hjärtans dag" replaced by the Russian name), dates ("(116)").
The probable cause is the step-1 prompt rule "no … transliterations in brackets", which the
model reads as "drop brackets".

**Step 4b result.** `experiments/multilingual_fit/fix_terms_parens.py` (gemini-3.6-flash,
$0.50, 361 calls) re-translated 969 sentences: 111 with term errors (the judge's correct term
given as a "Terms:" hint) and 891 with dropped parentheses (33 are both). Its prompt
drops the bracket rule and says to keep parenthesised content. It then scored them
(`score_translations.py --candidates`, job 1042417) and applied them (`--apply`). A
candidate replaces the current translation only if it passes the step-1 checks, keeps "(" where
that was the problem, contains every hinted term (stem match; alternatives allowed), and
loses at most 0.02 Kiwi. The apply logic is shared with step 3 (`retranslate_low.apply_candidates`).
Log with old and new text: `data_translated/fix4b/replacements.csv`.

- **827 of 969 replaced.** Kiwi of the replaced sentences went up on average
  (0.839 → 0.854), so restoring content did not cost quality.
- **Parentheses:** translations without "(" where the English has them dropped from
  35 / 32 / 26 / 19% to **5 / 5 / 2 / 3%** (ru / de / fr / es). Examples: "(AVLIS)",
  "Alla hjärtans dag («День всех сердец»)".
- **Terms: 90 of 111 sentences fixed**, e.g. инсуффляция, Vespasiano, Йол. The 21 not replaced
  had a Kiwi drop of more than 0.02 (2 also lacked the term). They are mixed: some are correct and
  Kiwi does not reward them (ru "Дол" → "Дойл" ×4, de "Feldmannschaft"); others are worse,
  because the judge's hint was wrong (es "libro de caballerías", a prose genre, for a verse
  romance), was copied in literally ("карагёз / карагиозис"), or the new version adds an
  error (es "erogación", ru "Тишек"). Loosening the rule would let in about as many new errors as
  it fixes, so these ~21 (~0.1% of the data) stay as they are.
- Term match after the fix (raw, before judging): names 92.5–96.0%, terms 51.7–58.0%
  (from 92.0–95.7% / 50.7–58.0%). The remaining mismatches are the ones the judge passed as correct.
- Integrity check: all 4 languages have 18 × 300 concept + 300 neutral sentences, none empty.

**Where the translated data stands.** The automatic checks (Kiwi, term check plus judge) estimate
term accuracy at ~98–99.7%. ~21 known term errors and ~120 sentences without their
parenthetical remain. Not checked by a human (none available). Data:
`experiments/multilingual_fit/data_translated/`, ready for step 7 with
`--multilingual-data-dir experiments/multilingual_fit/data_translated`.

## Open

- **Translated sets:** ready; next is step 7 (refit World War II on the translated sets).
- **Native sets, judge for the extra pool** (all 18 short pairs have a pool):
  - prompt per concept that names the wrong senses (films, namesakes, "heroine");
  - which sentences go to it (proposed: all sentences of category articles, only
    naming sentences of search-only articles, all link sentences);
  - for link sentences the sense is fixed; the question is "about the concept, or only a
    date / name?";
  - hand-check ~50 decisions per language;
  - then rebuild the short pairs: main article → related sentences that name the concept
    → judged extras (category / search pool and link sentences).
- **Native sets, pairs still short after the judge** (likely Valentine's Day es/ru/fr):
  accept fewer sentences, or top up with translated sentences.
- **Neutral sets:** done in `data_v2/` (random sentence from random articles, new
  splitter, reference lines and zero-width spaces removed).
- **Pipeline check:** the English features we compare against match the paper's Table 3,
  but they were downloaded, not fit with our code. A rerun with `--languages en` (to a
  separate `--outdir`) would show whether our pipeline reproduces 1 + 15 kept features.
- **Rank:** refit at a higher rank (e.g. 300) once the data is fixed.

## Reproduce

```
# data (no GPU, Wikipedia API, a few minutes each; never touch data/)
.venv/bin/python -I experiments/multilingual_fit/fetch_wiki_articles.py      # main articles, 18 x 5 languages
.venv/bin/python -I experiments/multilingual_fit/fetch_related_articles.py   # related articles (+ related_exclude.json)
.venv/bin/python -I experiments/multilingual_fit/build_concept_sets.py       # data_v2 concept sets (no network)
.venv/bin/python -I experiments/multilingual_fit/fetch_neutral_sentences.py  # data_v2 neutral sets (~1-2 h; --repair re-picks rejected sentences)
# extra pool for the pairs short in data_v2/concept_report.json (~1 article/s, ~1 h; detached)
setsid nohup .venv/bin/python -I experiments/multilingual_fit/fetch_extra_articles.py \
    > experiments/multilingual_fit/wiki_raw/extra_fetch.log 2>&1 < /dev/null &
# link sentences for the same short pairs (~3,600 articles, shortest pairs first; resumable)
setsid nohup .venv/bin/python -I experiments/multilingual_fit/fetch_link_sentences.py \
    > experiments/multilingual_fit/wiki_raw/links_fetch.log 2>&1 < /dev/null &
```

```
# 1. factorization (4× RTX 2080 Ti, ~2.5 h)
sbatch run_train_mf_features.sbatch --concepts "World War II" --languages en+ru+de+es+fr

# 2. filter (no GPU, ~30 s, Gemini key from .env)
umask 002
.venv/bin/python -m ember.interpret_features --concepts "World War II" \
    --outdir mf_outputs/multi_language/en+ru+de+es+fr
```
