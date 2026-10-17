# Part 2 — main plan (baseline)

*Decided 2026-10-17. This is the baseline design for Part 2; other ideas (see
[`ideas_and_discoveries.md`](ideas_and_discoveries.md)) come after it. Data and feature
search details: [`multilingual_features.md`](multilingual_features.md).*

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
| 2 | Arm A fit, World War II | **running**: job 1046208 → `mf_outputs/native/` (factorization + filter in one job) |
| 3 | Arm B fits, World War II: ru, de, es, fr each alone (`--languages ru` …); en = paper's factorization + 3.5 judge (done: `mf_outputs/en_rejudge35/`) | **next**; 4 jobs, ~35–45 min each on 4 RTX 2080 (estimate) |
| 4 | Look at the features of both arms: counts per track, which languages the tokens come from, shared vs language-specific features | after 2–3 |
| 5 | Erasure code reads Part 2 feature lists (today it reads only `mf_outputs/`) and takes the union for arm B | to do |
| 6 | Erasure + cross-lingual evaluation for both arms, World War II | to do |
| 7 | Extend to all 18 concepts | to do; cost per concept ≈ 2.5 h (arm A) + 4 × ~40 min (arm B) on 4 GPUs |

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
