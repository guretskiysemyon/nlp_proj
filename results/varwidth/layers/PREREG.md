# Pre-registration: layer-ladder evaluation (frozen 2026-10-18, before any held-out-concept metric is computed)

## Ladder (rungs)
| Rung | Matrix | Width rule | Judge | Run tag |
|---|---|---|---|---|
| L0 Original (EMBER) | plain | Fixed-1% | TJ | fixed_refit-False_plain |
| L1 | CW (contrastive whitening) | Fixed-1% | TJ | fixed_refit-False |
| L1+2 | CW | RNP (rn), cap 3% | TJ | rn_refit-False |
| L1+2+3 | CW | RNP, cap 3% | HJ = TJ ∨ LO@5 | rn_refit-False |

Names: CW = Contrastive Whitening; CP = Contribution Pooling (m1); EW = Entropy Widths (m3);
RNP = Row-Normalized Pooling (rn); TJ = Token Judge (paper, J_tok); CJ = Context Judge (J_words);
CJ+S = CJ ∧ concept share ≥ 0.5 (W); LO@t = Lexicon Overlap (J3, 20 × 20 exact, prompt v1); HJ = TJ ∨ LO@5;
HJ-R = CJ+S ∨ LO@4.

## Frozen settings
- **Concepts:** 38 = 23 existing (18 paper + Buddhism, Cricket, French Revolution, Plutonium, Tuberculosis) + 15
  held-out (Judaism, Karate, Existentialism, Prime number, The Matrix, Louvre, American Revolutionary War, Robot,
  Sigmund Freud, Giraffe, Surfing, Flush toilet, Cloud, Doctor of Philosophy, Tourism).
- **Prompt-example concepts** (named in LO prompt v1): Culture of Greece, Ancient Rome, Artificial intelligence,
  World War II. These are excluded from every analysis involving LO. **The primary scope is the other 34.**
- **Factorization:** k = 100, s = 1% (Fixed-1%), 2% (Fixed-2%), no refit; RNP / CP / EW: floor 0.2%, cap 3%;
  seeds 42–46; S = 300 concept sentences, sub-seed 0.
- **CW:** wiki background, page target, γ = 1, η = 0.05, eps_rel = 1e-4 (experiments/whitening/data/gemma/defaults.json).
- **Judge candidates:** the **top 20 features by ρ** (ratio_abs) at every rung, with no τ. The ρ scale changes under CW,
  so τ = 2 gives non-comparable candidate sets. The count with ρ ≥ 2 is reported. Existing runs are re-selected under
  this rule (`run.py --reselect`).
- **TJ / CJ:** gemini-3.5-flash-lite, 2 stages, accept when is_member and confidence ≥ 0.85, top-20 tokens.
- **LO:** concept_words_v1.json (prompt_v1.txt), the feature's first 20 normalized source words, exact match, each
  concept word counted once; LO@5 is in HJ.
- **CJ+S:** concept share ≥ 0.5.
- **Labels** (j3_data/labels.json): P = specific to the concept, B = borderline (broader category / adjacent),
  N = not the concept. Strict: B is negative; lenient: B is positive. The group key is concept | sorted first 6
  feature words.

## Primary endpoint
- **Strict clean-hit rate:** per concept, the share of its 5 seeds where F_C ≠ ∅ and every feature in F_C is
  labeled P (an unlabeled feature in F_C counts as not P).
- **Primary comparisons (3):** L0 → L1, L1 → L1+2, L1+2 → L1+2+3, each on the 34 non-prompt concepts.
  - Test: paired Wilcoxon signed-rank over concepts, **Holm-corrected** across the 3.
  - Reported with the mean paired difference and a 95% bootstrap CI (10,000 resamples over concepts, seed 0).

## Secondary (exploratory: effect sizes and CIs only)
- **Outcome classes:** clean hit / contaminated (an N feature in F_C) / miss; lenient variants.
- **Edited set:** E = support(F_C) ∩ concept-only tokens (EMBER edits only those).
  - **Held-out coverage:** for paper concepts, MC question and answer text; otherwise the unsampled page sentences
    (flagged when fewer than 50).
  - **Collateral exposure:** the occurrence rate of E's tokens in MMLU + Alpaca prompts and in a held-out
    Wikipedia sample.
- **Layer diagnostics:**
  - **L1:** oracle hit@k (k = 1, 3, 5), C-vs-N AUC, ρ ≥ 2 count.
  - **L2:** fragmentation, coverage of the best P feature, Spearman(width of the best P feature, number of
    distinct C tokens in V*) over 38 concepts, seed Jaccard.
  - **L3:** precision, recall and runs hit, macro per concept.
- **Scopes:** original (23), new (15), wide (Cloud, Doctor of Philosophy, Tourism), all (38), non-prompt (34).
- **Cross-ablations:** plain RNP; LO / HJ on the L0 and L1 runs.
- **L2 method table at cap 3%:** Fixed-1%, Fixed-2%, CP, EW, RNP.
- **Judge noise:** a re-judged random 10% of L1+2 original candidates (seed 0) gives the TJ flip rate.
- **Label validity:** 60 groups labeled by the user (stratified, blind); Claude-vs-human κ.

## Erasure anchor
- **Concepts** (10 of the 14 non-prompt paper concepts, random.Random(0).sample): Baseball, COVID-19 pandemic,
  Gambling, Golf, Gun, Halloween, Harry Potter, Heroin, Pornography, Uranium.
- **Setup:** seed 42, all 4 rungs, gemma-2-2b-it, EMBER embedding edit, `--train-eval mc`.
- **Delta grid** {0.5, 1, 2, 5, 10, 50, 100, 200}, gated by MMLU-frac ≥ 0.9 only. Alpaca fluency and instruction
  following are scored on 50 items at the chosen delta. The protocol is identical for all rungs.
- **Metrics:** EMBER's efficacy, specificity and harmonic. Proxy validity = Spearman between the proxies and these
  metrics over the 40 points.
- **Fidelity:** our L0 on Harry Potter and Gun is compared with the paper's reported numbers.

## Budget
- **Gemini hard stop:** 25 NIS (ember/gemini_budget.py, conservative pricing).
- **Cut order:**
  1. L0 down to 3 seeds;
  2. no CJ on the new concepts;
  3. erasure without Alpaca;
  4. thinking off.

## Amendment 1 (2026-10-18, after the judging finished and before any erasure result existed)
- Gemini spend was 6.7 of 25 NIS, so the erasure anchor is extended from 10 to all **14 non-prompt paper concepts**
  (added: Cannabis, Nazism, Republic of Ireland, Valentine's Day). The analysis is unchanged.
- When a rung's F_C equals the F_C of another rung of the same configuration (L123 vs L12), the erasure result is
  copied (identical edit), marked `copied_from`.
- Fidelity: one extra Harry Potter L0 run with the paper's own grid (Alpaca in the grid, 150 train items, Alpaca gate
  0.9), if spend allows (< 18 NIS before it).
- Disjoint-target control for CS, if spend allows: the sharpening target built from page sentences outside the 300
  sampled concept sentences, at L1 and L1+2, judged and evaluated like the main rungs. This is exploratory.
- Reports use the paper names (CS, RS-1%, GS-norm, TJ, WJ, EJ@t, TWJ); the rung keys are unchanged.
