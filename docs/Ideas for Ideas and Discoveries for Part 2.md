# Ideas for Ideas and Discoveries for Part 2

*Status: preliminary proof of concept, 2026-10-04. Gemma-2-2b-it, input-embedding level only, 5 concepts, 1 seed.*

## The question

EMBER finds a concept's features from **English** text only. Is there anything in those
features that is shared with the **same concept's words in other languages**? If the
embedding space has a language-independent concept direction, Part 2 could use it
(fit across languages, or extend the English edit to other languages) instead of
starting from scratch per language.

## What we did

No generation, no judge, seconds of compute. Script:
`experiments/crosslingual_concept_axis.py`; word lists: `experiments/crosslingual_words.json`.

- Concepts: Gun, Heroin, Gambling, Ancient Rome, World War II. Languages: de, es, fr, ru.
- ~25 English concept words per concept, translated word-by-word into each language
  (translations drafted by Claude, **not yet checked by a native speaker**).
- Every word is a point in the 2304-d embedding space. Two views:
  - **A. English EMBER features -> foreign words.** How much of a foreign concept word's
    embedding lies in the 1-3 EMBER directions, compared with ordinary words of that language?
  - **B. Foreign concept direction <-> English concept direction.** Build a direction from
    the foreign concept words (their average minus the average ordinary word of that language)
    and compare it with the English one: angle between directions, and how well each direction
    picks out the other language's concept words.
- Scores are AUC (0.5 = chance, 1.0 = perfect). The key control is comparing a concept's words
  against **other concepts' words in the same language**, plus random-word-set nulls.

## Most interesting findings

**1. A language-independent concept direction exists, even with zero shared tokens.**
A direction built from just ~25 *Russian* words (different script, so no word shares a
surface form with English) ranks the *English* concept words far above English words of
the other four concepts. A random-word direction gives about 0.50.

| Concept | Russian direction -> English words (AUC) | Cosine of the two directions |
|---|---|---|
| Gun | **0.99** | 0.27 |
| Heroin | **0.95** | 0.16 |
| Gambling | **0.95** | 0.19 |
| Ancient Rome | **0.86** | 0.10 |
| World War II | 0.41 (fails) | -0.20 |

Matching concepts have more similar directions than mismatched ones (mean cosine +0.10
vs -0.08; the right concept is picked 4 times out of 5, chance is 1 in 5).

**2. EMBER's own English features capture only a small part of it.**
The 1-3 EMBER directions separate English concept words well (AUC 0.78-0.97, in-sample)
but transfer weakly to Russian words (AUC 0.43-0.65). Only 1-7% of the Russian concept
direction lies inside the EMBER subspace (still 10-80x chance, but small). The shared
structure is there; the English-fit subspace is too narrow to express it.
-> Supports a multilingual re-fit in Part 2.  

**3. Tokenization is a second, independent leakage mechanism.**
Gemma splits most Russian and German words into pieces (`Waffe` = `Wa`+`ffe`;
`пистолет` = `писто`+`лет`). Only 4/50 Russian and 8/50 German Gun/Heroin words are single
tokens. EMBER edits whole-token embedding rows, so it **cannot reach most Russian/German
words by construction**, whether or not the subspace is shared. Worth separating these two
explanations when we analyse leakage.

## Other results

- **Asymmetry:** Russian direction -> English words works much better than English direction
  -> Russian words (0.65 / 0.58 / 0.85 / 0.56 for Gun / Heroin / Gambling / Rome). The English
  side is clean single tokens; the Russian side is averaged from noisy pieces (even a
  Russian direction fit on half the Russian words only reaches ~0.74 on the other half).
- **Failure case, World War II in Russian:** the words are transliterated names
  (`Гитлер` splits into meaningless pieces), and nothing transfers.
- **de / es / fr look huge but are mostly shared surface forms** (names, cognates, loanwords:
  cosine 0.5-0.97 with all words). After dropping same/similar-to-English words, only
  Gun (es, fr; cosine ~0.44-0.48) and Heroin (es) clearly survive. German is weakest
  (Gun/de cosine -0.03).

## Ideas for Part 2

1. **Multilingual fit.** Re-run the factorization with concept sentences from several
   languages (see `crosslingual_plan.md` §12). Sanity check: the joint features should align
   with the foreign-built directions found here.
2. **Zero-shot extension of the edit.** Add the foreign-built direction as extra edit
   directions and measure leakage on the translated QA sets. Cheap, needs no refit.
3. **Layer sweep.** Repeat this test on residual-stream activations (e.g. layers 0-12).
   Cross-lingual alignment is normally much stronger after a few layers, and it would handle
   multi-token words properly. Likely the most informative next experiment.
4. **Explain leakage with these scores.** Correlate per (concept, language) alignment and
   single-token coverage with the leakage gap from the cross-lingual eval. A second
   "money figure" next to the token-overlap scatter.
5. **Multi-token words.** Does editing the pieces of split words help, and what does it cost
   in specificity?
6. **Better word lists without translation.** Extract each language's concept words
   from Wikipedia with the same concept-vs-neutral frequency method used for English, or use a
   bilingual dictionary, so results do not depend on hand-drafted translations.

## Caveats

- Translations are unchecked (please review de/ru). 5 concepts, 1 seed, 1 model.
- "Concept-specific" here means telling one topic apart from four others: a coarse test.
- Mean-pooling word pieces is a crude proxy for how the model builds a word's meaning in its
  early layers, so this level probably underestimates the real overlap.
- Pitfall found on the way: filtering to single-token words keeps mostly loanwords and
  names and inflates results. Always report a version with same/similar words removed.

Outputs: `experiments/crosslingual_results/`. Files tagged `_mean` (all words, pieces averaged)
and `_mean_distinct` (same/similar-to-English words removed) are the results quoted above.
The untagged files are the first single-token-only run (older script version, no random-axis
nulls); keep them only as the example of the loanword-bias pitfall.
