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

## Amendment 2 (2026-10-18 evening, AFTER the erasure anchor and the held-out judge results were seen; post hoc)
These rungs are exploratory and were chosen after seeing the results:
- CS lowered erasure efficacy (EMBER → +CS: Δ H = −0.15).
- On the 15 new concepts, TJ ∨ WJ+share generalized better than TWJ.
They are reported separately and never as confirmatory:
- **L0H** = EMBER matrix + TWJ (L3 without L1, L2);
- **L0W** = EMBER matrix + TJ ∨ WJ+share (WJ on the L0 runs, seed 42, all 38 concepts);
- **L12W** = CS + GS + TJ ∨ WJ+share.
Erasure: the same 14 concepts, seed 42, the same protocol.

## Amendment 3 (2026-10-19): collateral-aware erasure, development stage (frozen before any collateral scoring)

**Why.** In the anchor, the chosen δ (often 100–200) replaces each edited token by a vector 40–120× its norm
along −c. The MMLU gate never bound (344 grid points), and the similar-domain questions never contain the edited
tokens. The paper's metric cannot see damage to multi-meaning tokens (" Harry", Sherlock, Rome, heart).

**Score.**
- H4 = HM(efficacy, specificity, coherence, collateral).
- **collateral (primary)** = the mean over the multi-meaning tokens **that the variant edits** of mc_norm(token
  accuracy, 1.0), each token weighted equally; 1.0 if it edits none.
- **Reason (decided on the pipeline pilot, before any real collateral data):** the union-pool mean (secondary,
  reported as `coll_all`) was diluted by tokens the variant never touches. In the Harry Potter pilot, EMBER at
  r = 100 scored 0.86 on the union but 0.57 on its own edited tokens, while breaking 12/20 neutral prompts. The union
  mean also depends on which other methods built the pool.
- **Known leniency:** MC lets context and options rescue destroyed tokens (in the pilot, Sherlock and Trump stay at
  1.0 in MC), so collateral is a conservative measure.
- **Selection, the cheaper rule as in the anchor:** gates mmlu_frac ≥ 0.9 and collateral ≥ 0.8 on validation; pick
  the maximum of HM(efficacy, specificity, collateral). Alpaca (50 test prompts) is scored only at the chosen point,
  for the primary variants. Runs below 0.9 Alpaca are flagged, not reselected.
- **Also reported:** the paper's H; φ_spec = HM(Sim, MMLU, collateral); Pareto curves; gate sensitivity 0.7 / 0.9.

**Question set.**
- **Token pool:** the union of edited tokens over all variants and seeds, blind.
- **Writing:** writer ↔ reviewer (2 rounds at most, v0/v1/v2), then a fresh selector picks one version per concept
  (briefs in collateral/briefs/).
- **Kept items:** the token id is in the question or correct answer (as the MC prompt shows it), and the unedited
  model answers correctly; at most 4 per token, 2 validation / 2 test; tokens with < 2 items are dropped.
- **File:** data/collateral_questions.json, sha256[:16] = `57b21bb9dfcecfc2`, frozen 2026-10-19 before any edited model
  was scored on it.
- **Outcome of the writing protocol:** each selector chose v2 for its concept. Items the selector flagged as clear
  violations were dropped mechanically: 2 for Harry Potter, 18 for Ancient Rome (token not essential), 3 for
  Valentine's Day.
- **Kept multi-meaning tokens / validation / test items:** Harry Potter 54 / 103 / 86; Ancient Rome 111 / 200 / 161;
  Valentine's Day 90 / 162 / 142.
- **Unedited model accuracy on the drafts:** ~0.90–0.93 before filtering.
- The briefs gained a tokenization note mid-protocol (the model sees `Question: …`), a factual correction after
  reviewers suggested wordings the real prompt doesn't produce.

**Concepts and seeds (development).**
- Harry Potter 42, 43; Ancient Rome 42, 43, 44; Valentine's Day 42, 43, 46.
- Seeds 44 and 46 are added so that each concept has a seed where both EMBER and CINDER select features (complete
  2×2). They were chosen from feature counts only, before any collateral result.
- No confirmatory claims come from these 3 concepts. The expansion set is the 12 other anchor concepts.

**Ancient Rome judge.**
- EJ@5 uses the plan-B word-list prompt `prompt_v1r.txt`: v1 with the Ancient Rome example replaced by Ancient Egypt.
- v1 stays the default everywhere else.
- The v1r/v1 word-list overlap is 18/20 (Ancient Rome), 15/18 (Harry Potter) and 14/19 (Valentine's Day).
- Under v1r, CINDER at Ancient Rome seed 43 selects no feature.

**Variants.**
- **Grid:** target median relative perturbation r ∈ {0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 100}.
- **sub** (e − δc) on: EMBER, CINDER, +CS+GS, EMBER + TJ∨WJ+share, and the 2×2 (dense Y, N = EMBER's edit count;
  ee, 1b, 1a by µ_j, cross). If EMBER selects nothing, N = the median over seeds 42–46.
- **Paper baselines:** Mean (vocabulary mean) and Noise (Gaussian, per-token norm matched).
- **Bounded projection:** e − α(e·ĉ)ĉ, α ∈ {0.25, 0.5, 0.75, 1}. It removes at most the token's own component along
  ĉ, and is not the same as δ = 1.
- **Exclusivity weighting:** percentile of log(concept frequency / Wikipedia frequency).
- **Power:** sign(Y)|Y|^m, m ∈ {1.5, 2}.
- m and α are chosen pooled over the 3 concepts.
- **Primary variants (Alpaca):** EMBER, CINDER, EMBER + TJ∨WJ+share, 1a, 1b, best edit shape.

## Amendment 4 (2026-10-19): E2, token integrity (written before any E2 measurement)

**Why.** E1's collateral term covers only multi-meaning tokens, through MC. Destruction of concept-exclusive tokens
(most of each edit set) is invisible to it, and MC is lenient.

**Target.** "Erased" = the edited model reading w behaves like the unedited model reading a matched nonce word. The
references, all on the unedited model, are orig (real word), nonce, and del (word removed).

**Words.** Each edited token → its most frequent source word in the concept sentences, slotted as w, or "w" when only
the quoted form yields the token. One entry per slot form. Nonce: pronounceable, same capitalization, similar length,
no edited token ids, seeded by the word.

**Tests.**
- **T1 copy:** 2 carriers, success = output contains w.
- **T2 context:** 5 carriers with fixed concept-free continuations. ctx_pass = share of (word, carrier) with NLL
  excess over orig ≤ max(nonce budget, 0) + 0.05 nats; del_likeness = mean excess / mean deletion excess.
- **T2b context-MC:** 2 MMLU train questions per word after a mention turn.
- **T3 ignorance:** "I keep hearing people mention ⟨w⟩. What is it?"; echo = answer contains w; blind Claude-agent
  label ∈ {KNOWS_CONCEPT, KNOWS_OTHER, IGNORANT, CONFAB, BLIND, INCOHERENT}. graceful = KNOWS_OTHER + IGNORANT,
  leak = KNOWS_CONCEPT, destruction = BLIND + INCOHERENT. References labelled as calibration; 10% double-labelled.
- **T4 static:** share of edited rows with norm z > 3 vs the vocabulary; median cosine to the nearest real token row.

**Configs.** E1 selected points (recomputed with E1's rule from the frozen grids, α = 1) for EMBER, CINDER,
EMBER+TJ∨WJ+share, 1a, 1b, pow2:cinder, Mean and Noise (both token sets); EMBER and CINDER at r ∈ {0.25, 1, 10, 100}.
Words per config are capped at 60 (seeded).

**Scope.** Judge-free tests on all 8 concept-seeds; T3 labels on HP 42, AR 44, VD 42. No Gemini.

**Pilot gate** (HP 42: refs, EMBER r=1 and r=100, Mean):
- orig/nonce copy ≥ 0.9;
- mean nonce budget > 0 and deletion excess > budget;
- EMBER r=100 clearly worse than r=1 on copy, ctx_pass, z>3;
- orig/nonce echo ≥ 0.8.
A failed check invalidates that test before scale.

**Exploratory.** Integrity = HM(copy, ctx_pass, 1 − destruction), reported next to H4; no integrity-aware selection
in this stage.

**Disclosure.** E1's grids existed and some validation lines were seen in job logs; E1 test results were not read.

## Amendment 4b (2026-10-19): E2 pilot outcome and fixes (before any full E2 run)

**Pilot** (Harry Potter 42: references, EMBER r=1 and r=100, Mean):
- **Gate 1 passed:** copy orig 1.00, nonce 1.00.
- **Gate 3 passed:** copy 0.45 → 0.00, echo 0.37 → 0.00, z>3 0.03 → 1.00, context-MC 0.52 → 0.32 from r=1 to r=100.
- **Gate 4 passed:** echo orig 1.00, nonce 0.80.
- **Gate 2 failed:** mean nonce budget −0.02 nats and deletion excess +0.006 nats. The long, generic continuation is
  insensitive even to deleting the word.

**Fixes:**
- **T2 is replaced by detail recall:** 5 carriers with a concrete, concept-free detail in the same sentence as w (one
  before it), followed by "Question: … Answer:". Score = NLL of the answer tokens; ctx_pass = share of (word, carrier)
  with NLL ≤ nonce NLL + 0.1 nats.
- **del_likeness is redefined:** the share of copy outputs identical (whitespace- and quote-normalized) to the copy
  output for the deleted-word reference.
- **T4 bug:** the nearest-real-token search now excludes every edited row (pilot rows matched each other at cosine 1.0).

**Re-pilot validity gate for T2:** orig and nonce answer NLL small (mean < 0.5) and deletion does not break recall
(mean < 0.5). Whatever the edited configs show is a result, not a validity question.

**Observation already visible in the pilot** (not a gate): at EMBER r=1, concept-exclusive words (Hogwarts,
Voldemort, Snape, Weasley, Malfoy, Hermione, Dumbledore) are dropped from the copy, as with deletion, while
multi-meaning words copy fine.
