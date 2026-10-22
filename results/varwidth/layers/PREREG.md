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

## Amendment 4c (2026-10-19): second T2 fix (before any full E2 run)

**Re-pilot:**
- **T2 gate failed:** unedited answer NLL mean 2.42, against a gate of < 0.5. Only 1 of 5 carriers is easy: raw
  "Answer:" scoring without the chat template is poorly calibrated for the instruction model.
- **The signal is there:** EMBER r=100 raised the answer NLL by +1.34 nats over the nonce; r=1 and Mean did not.

**New primary T2 = recall:**
- chat generation of "<sentence> Question: <q> Answer with one word.", 8 tokens, case-insensitive match of the detail
  ("nine" also matches "9");
- ctx_pass = recall rate of the edited config;
- **validity gate, checked per run on the references only:** orig, nonce and del recall ≥ 0.9; otherwise T2 is not
  reported for that run.

**Secondary:** answer-NLL excess over the nonce (diagnostic only).

**No further pilot.** The full runs start directly, and the gate is evaluated on each run's references. Every other
test is unchanged; their pilot gates passed.

## Amendment 5 (2026-10-19): CINDER-RS, layers 1 + 3 without 2 (requested by the user after E1/E2 results were seen)

**Candidate:** CINDER-RS = Contrastive Sharpening (L1) + the Two-Way Judge (L3) with EMBER's row sparsity (RS-1%), no
Global Sparsity. In code: rung `L13` = fit `fixed_refit-False` (CS, 1% per feature) + judge HJ (TJ ∨ EJ@5; Ancient
Rome uses the v1r lexicon as in amendment 3). Edit = e − δc with sparse Y, concept-only tokens of support(F_C).

**Motivation (post hoc, disclosed):**
- In E1, CINDER's direction was more efficient at small r.
- CINDER's GS-widened token sets carried more multi-meaning tokens.
- CINDER-RS separates the direction (L1) from the widths (L2).

**Evaluation, identical to E1 and E2:**
- **E1:** the same r grid, frozen question set, gates and cheaper-rule selection; test pass; Alpaca at the chosen
  point (50 prompts).
- **E2:** the selected point plus a sweep r ∈ {0.25, 1, 10, 100}, all judge-free tests, and T3 labels on HP 42,
  AR 44, VD 42 through an incremental blind export, with already-labelled ids unchanged and the same brief including
  its clarifications.

**Coverage:** 89–100% of CINDER-RS's edited tokens are in the frozen pool (VD 46: 68/77). Collateral is over the
edited multi-meaning tokens of the frozen set, as for every variant; uncovered tokens are reported.

**Status:** an exploratory comparison. It is the candidate the user proposed and was not chosen by us from results;
claims need the expansion concepts.

## Amendment 6 (2026-10-19): erratum, test-collateral baseline

**The issue.** The base filter checked every collateral item under the option order of a "val" item, because the MC
evaluator seeds the option shuffle with an id that includes the split. Test items are shuffled differently, and the
unedited model misses some of them. Its test collateral is 0.981 for Harry Potter, 0.964 for Ancient Rome and 0.948
for Valentine's Day (union), identical across seeds, so the evaluation is deterministic.

**What is affected and what is not.**
- **Not affected:** validation, where the baseline of 1 holds by construction, and therefore selection.
- **Affected:** reported test collateral, which assumed a baseline of 1.

**Fix, in reporting only.** collateral_rn = the mean over the variant's edited multi-meaning tokens of
mc_norm(acc_edit, acc_unedited), using the unedited per-token test accuracy from the no-edit rows. This is the
paper's normalization. Tokens whose unedited test accuracy is ≤ 0.25 are left out, and no edit gives 1.0.

**Effect.** Test collateral rises by 0.02–0.04. The H3v and H4 rankings are unchanged. The raw values stay in
poc.md next to the corrected ones.

## Note (2026-10-19): corrections to amendments 5 and 6 found while documenting
- **Amendment 5:** CINDER-RS coverage is 88–100% (VD 46: 68/77 = 88.3%), not 89–100%.
- **Amendment 6:** after renormalizing, variant means rise by 0.02–0.05 (1b and projection on EMBER's tokens +0.05),
  not 0.02–0.04. Rankings are unchanged.

## Amendment 7 (2026-10-20): A3z, CINDER's selection with the edit dictionary re-solved in the original space

**Origin.** Proposed by the user, from the multilingual agents' Llama finding: CINDER's mapped-back edits lie in
low-variance directions that layer 0 barely reads.

**Variants.**
- **CINDER-z:** CINDER's whitened-fit sparse Y (with its GS masks), its HJ-selected F_C and its edited tokens, with
  Z replaced by Z_o = A Yᵀ (Y Yᵀ + λI)⁻¹ (λ = 1e-4, A = raw TL-scaled V* embeddings, as EMBER's A); edit
  c_t = Σ_{i∈F_C} z_o,i y_i,t. No map back.
- **CINDER-RS-z:** the same for CINDER-RS (L13).

**Offline check, before any evaluation** (diag/a3z_offline.json):
- layer-0 attention read strength (relative to random directions): EMBER 1.7–2.0×; CINDER / CINDER-RS mapped back
  1.0–1.2×; A3z 1.8–2.3×;
- median cos(c, e): 0.10–0.44 → 0.34–0.61;
- cos to the original CINDER edit: 0.70–0.87.

**Evaluation.**
- **New evaluation:** E1 (same grid, frozen question set, gates, cheaper rule; Alpaca at the chosen point for the two
  A3z variants) and E2 (selected point plus sweep r ∈ {0.25, 1, 10, 100}; judge-free on 8 runs; T3 labels on the 3
  judge runs through the incremental export).
- **Old evaluation:** the paper-style rule (max HM(efficacy, specificity), MMLU ≥ 0.9, no collateral) on the same
  grids, with a test pass at its point for EMBER, CINDER, CINDER-RS, CINDER-z, CINDER-RS-z and 1b (result_paper.json;
  no Alpaca, to stay within budget), plus E2 judge-free tests at that point.

**Status:** exploratory. Gemini ≈ 2 NIS, total ≈ 23.6 of 25 NIS.

## Amendment 8 (2026-10-20): confirmatory expansion (written before any expansion data exists)

**Question.** Do the CINDER variants differ from EMBER under the new evaluation?

**Concepts and seeds.** The 12 anchor concepts never used for development: Baseball, Cannabis, COVID-19 pandemic,
Gambling, Golf, Gun, Halloween, Heroin, Nazism, Pornography, Republic of Ireland, Uranium. Seeds 42–46 (60 runs).

**Variants.** EMBER (L0), CINDER (L123), CINDER-RS (L13), CINDER-z, CINDER-RS-z. Definitions and settings are
unchanged (frozen on the development concepts): r grid {0.05, …, 100}; gates MMLU ≥ 0.9
and collateral ≥ 0.8.

**Question set.**
- **Blind union pool** of the tokens edited by EMBER, CINDER and CINDER-RS (the A3z sets share CINDER's / CINDER-RS's
  tokens) over seeds 42–46.
- **Protocol:** writer v0 → reviewer F0 → writer v1 → reviewer F1. F1 flags residual violations, and its "drop"
  items are removed mechanically.
- **Shortened from the development protocol:** no v2 and no selector, for time. Every development selector chose
  the latest version, and the main gains came in round 1.
- **Checks and freeze:** token-id check; base filter (unedited model correct), applied with each split's own option
  shuffle, which fixes the cause of the amendment 6 erratum; 2 validation / 2 test items per token; tokens with
  < 2 items dropped; frozen as data/collateral_questions_expansion.json with its hash recorded in the commit before
  any edited model is scored.

**Arms (test split, MC).**
1. **Collateral rule (primary):** cheaper-rule selection on validation by HM(efficacy, specificity, collateral) under
   the gates.
2. **Fixed strength:** r = 0.5 for every variant, so there is no selection noise.
3. **Paper rule (secondary):** max HM(efficacy, specificity), MMLU gate only.

No Alpaca, because about 1.5 NIS of Gemini is left; H3v replaces H4. Test collateral is normalized by the unedited
model's per-token test accuracy, computed in each run.

**Primary endpoints, with Holm correction over the 4 comparisons.** Paired H3v (collateral rule) for CINDER − EMBER,
CINDER-RS − EMBER, CINDER-z − EMBER and CINDER-RS-z − EMBER, on runs where both edit. Unit = concept: mean over seeds
within a concept, then a two-sided Wilcoxon signed-rank test across concepts, plus a 95% cluster-bootstrap CI.

**Secondary endpoints.**
- The same contrasts at fixed r = 0.5 and under the paper rule.
- Failure rate (H3v < 0.25).
- E2 integrity at the selected point (judge-free copy and echo on all runs; T3 destruction and leak on seed 42 via
  blind agent labels with the same brief).
- Exclusive vs multi-meaning destruction.

**Development numbers it is meant to confirm or refute:** CINDER-RS − EMBER +0.09 overall but +0.01 without the
failure run; integrity destruction EMBER 0.30 vs CINDER 0.21 and CINDER-RS 0.19.

**Note (2026-10-20): freeze.** `data/collateral_questions_expansion.json` frozen in commit 8423d03, sha256[:16]
5ba1d287f3b405b1 (12 concepts, 1,026 multi-meaning tokens, 3,501 items), before any edited model was scored on it.

**Note (2026-10-21): deviations.** (1) E2 integrity is measured at the selected point only, the endpoint named above;
the extra configs at r = 0.5 were dropped for time (two early runs have them; unused). (2) Variants with an empty edit
set are skipped (no result file), as "runs where both edit" implies. (3) The finals reuse another rule's test result
when that rule chose the same point (deterministic evaluation). (4) The T3 brief left two cases open (same-referent
proper nouns without a concept link; hedged look-alike corrections), and the agents resolved them differently; batches
were random, and cross-agent agreement on 202 duplicates is 0.97 exact and 0.99 on destruction.

**Results (2026-10-21; confirm.md, CINDER_collateral_evaluation.md D17 and F24–F29).**
- Primary (collateral rule, paired H3v vs EMBER, 11 concepts): CINDER +0.042 (CI −0.009, +0.096; Holm p 0.41),
  CINDER-RS +0.034 (−0.043, +0.112; 0.41), CINDER-z +0.132 (+0.048, +0.212; 0.074), CINDER-RS-z +0.084
  (+0.020, +0.153; 0.074). No contrast is significant after Holm.
- Fixed r = 0.5: +0.151, +0.141, +0.086, +0.115 (Holm p 0.012, 0.037, 0.054, 0.012), 10 of 11 concepts each.
- Paper rule, H2: −0.134, −0.153, +0.161, +0.088 (Holm p 0.056, 0.084, 0.004, 0.084).
- Edit rate: EMBER 41/60, CINDER and CINDER-z 52/60, CINDER-RS and CINDER-RS-z 55/60. Failure rate (no edit
  counts as failure): 0.37, 0.17, 0.17, 0.18, 0.15.
- Integrity at the selected point: copy 0.40–0.44, T3 destruction 0.49–0.61 (EMBER 0.49, CINDER 0.52, CINDER-RS 0.60),
  exclusive words 0.65–0.81; no significant paired difference.
- Against the development numbers: CINDER-RS − EMBER is +0.03 (development +0.09, +0.01 without the failure run);
  the integrity advantage of CINDER and CINDER-RS (0.21 and 0.19 vs 0.30) is refuted (0.52 and 0.60 vs 0.49).

## Amendment 9 (2026-10-21): last-mile checks (post hoc; defined after the amendment 8 results were seen)

Written before any result of items 2–4 exists. Item 1 was computed while writing this amendment.
1. **Intention-to-treat primary.** All 60 runs; a variant with no edit set, or no point passing the gates, scores as
   the unedited model (efficacy 0, so H3v 0). Same unit, test, Holm and bootstrap as amendment 8.
   **Leave one concept out** for the primary contrast.
2. **Gate sensitivity.** The collateral rule re-run on the saved grids at gate 0.7 and 0.9 (MMLU gate unchanged),
   test pass at the new point (`result_g0.7.json`, `result_g0.9.json`); paired as in amendment 8.
3. **Unreadability floor of efficacy.** The unedited model on the efficacy test questions with each (run, variant)
   edit set's tokens made unreadable in the text: replaced by a nonce, or removed. Compared with the efficacy at the
   collateral-rule and paper-rule points. Prediction (the token-destruction thesis): at the paper-rule point, EMBER's
   and CINDER-z's efficacy is close to the deletion floor; at the collateral-rule point, efficacy is mostly below it.
4. **Generation damage.** 20 neutral prompts per concept (blind Claude writer, edited multi-meaning tokens in outside
   meanings), at the concept's first seed where EMBER edits; greedy, 80 tokens; unedited, the 5 variants at their
   collateral-rule point, EMBER and CINDER-z at the paper-rule point. Blind agents label each answer OK or BROKEN
   against the unedited answer (D3's rule). Endpoint: broken rate per config. Prediction: paper-rule points break far
   more prompts than collateral-rule points (development: 3–12 of 20 at δ = 200, 0–1 at δ = 1).
No Gemini.

## Amendment 10 (2026-10-21): Alpaca coherence on seed 42 (user-approved Gemini extension, about 8 NIS)

Alpaca coherence (50 prompts, the development scorer) at seed 42's collateral-rule point of each variant that edits
(48 points), and at the paper-rule points of EMBER and CINDER-z (about 19), for H4 = HM(efficacy, specificity,
coherence, collateral) on one full concept sweep, and for coherence at the destruction regime. Hard stop
GEMINI_BUDGET_NIS = 32 for these jobs (23.49 spent before). Reported per variant and as paired H4 vs EMBER on seed 42
(descriptive, 12 concepts × 1 seed). Prediction: coherence ≥ 0.9 at the collateral-rule points; lower at r = 30–100.

**Results of amendments 9 and 10 (2026-10-21; last_mile.md, confirm.md "Sensitivity", doc F30–F35 and D18).**
- ITT (all 60 runs): CINDER +0.123, CINDER-RS +0.133, CINDER-z +0.167, CINDER-RS-z +0.178 (Holm p 0.103, 0.103,
  0.103, 0.049). LOO: A3z contrasts p ≤ 0.049 with any concept left out; CINDER and CINDER-RS p 0.08–0.70.
- Gate 0.9: +0.174, +0.147, +0.195, +0.102 (p 0.004, 0.004, 0.002, 0.014; all below 0.05 after Holm). Gate 0.7:
  −0.019, −0.041, +0.121, +0.070 (p 0.70, 0.37, 0.07, 0.24).
- Floor: deletion floor 0.23–0.28. Collateral-rule efficacy 0.29–0.39, at or below the floor in 17–53% of runs.
  Paper-rule efficacy EMBER 0.47, CINDER-z 0.68, above the floor in 83% and 98% of runs. Prediction for the paper
  rule (close to the floor) refuted; prediction for the collateral rule (mostly below) not met either (about at it).
- Generation (9 matched concepts): broken 0.15–0.27 at the collateral-rule points, 0.24 (EMBER) and 0.34 (CINDER-z)
  at the paper-rule points. Prediction (paper far worse) supported in direction; the collateral-rule points are not
  near the development δ = 1 rate (0–1 of 20).
- Alpaca seed 42: coherence 0.97–0.98 at every point (none below 0.9), paper-rule points included. H4 paired vs
  EMBER: CINDER-z +0.140 (7/0), CINDER-RS-z +0.064, CINDER −0.015, CINDER-RS −0.032. Gemini 8.33 NIS.

## Amendment 11 (2026-10-21, evening): word-free efficacy and integrity-gated selection (post hoc; one day left)

Written before either result exists. The powered CINDER-z test on new concepts does not fit in the remaining day
(only 3 unused concepts have efficacy questions), so both checks reuse the 60 expansion runs.
1. **Word-free efficacy.** New efficacy questions per concept (blind Claude writers, 60 drafts each) that avoid every
   token edited by any variant at any seed of that concept, kept only if (a) no question or option contains such a
   token id (tokenizer check) and (b) the unedited model answers correctly (option order seed 42). Plus the
   existing qa_test items that contain none of the (run, variant)'s edited token ids. Efficacy_wf = 1 −
   normalized accuracy on these items, at the collateral-rule and paper-rule points. Endpoints: efficacy_wf per
   variant; paired CINDER-z − EMBER on efficacy_wf and on H3v with efficacy_wf (unit = concept, Wilcoxon).
   Question: does each method's efficacy survive when the edited words are absent?
2. **E3, integrity-gated selection.** T1 copy (2 carriers, ≤ 60 words) at every grid point with r ≤ 2.5; points
   above have copy ≈ 0 (F9) and are excluded. Rule: the collateral rule (MMLU ≥ 0.9, collateral ≥ 0.8, max H3v on
   validation) restricted to points with copy ≥ 0.9; test pass at that point (`result_e3.json`). Endpoints: paired
   H3v vs EMBER as amendment 8 (Holm over 4), edit rate under the gate, exclusive-word copy at the selected point.
   Prediction (S2): CINDER variants beat EMBER under E3.
No Gemini.

## Amendment 13 (2026-10-21, night): the nonce-matching edit (post hoc; defined after amendment 11)

Written before any evaluation result of this method exists. Motivation: embedding edits only block access through the
edited tokens (the output head is untied, so prompts without them are unchanged). The achievable target is "the
concept tokens become unknown but intact words": the nonce reference (copy 1.00, T3 destruction 0.06, mostly
CONFAB / IGNORANT) at efficacy near the nonce floor. All methods at r ≈ 1 reach the floor's efficacy only by
destroying the words (copy 0.40–0.44, T3 destruction 0.49–0.61).

**Method.** Only the edited rows change: e'_t = e_t + Δ_t (TL-scaled space), Δ_t learned with the model frozen.
Loss = w_f · forget + copy + retain + 1e-3 · Σ‖Δ_t‖² / ‖e_t‖².
- forget: KL(unedited model with the token replaced by its nonce ‖ edited model with the token), at every position
  from the token on (top-64 + rest approximation of the target), on (i) concept sentences containing the token (the
  SNMF training data of the concept, not evaluation data) and (ii) 3 new definitional chat prompts (not the T3 prompt),
  teacher-forcing the unedited model's greedy answer to the nonce prompt.
- copy: KL(unedited ‖ edited) on the response to 4 new copy carriers with a new instruction (not the evaluation's two
  carriers or instruction), teacher-forcing the unedited model's greedy answer.
- retain: multi-meaning tokens only; KL(unedited ‖ edited) from the token on, on 3 sentences per token that the unedited
  model writes with the word in a sense unrelated to the concept.
- Never trained on: the collateral question sets, qa / simdom / MMLU items, the COPY carriers, the ASK prompt, the
  neutral generation prompts.

**Variants** (seed 42, 12 expansion concepts; a variant with an empty set does not edit):
NM-CINDER (CINDER-z token set; Δ_t in the span of that set's concept components, rank ≤ number of concept features),
NM-free (CINDER-z token set; Δ_t unconstrained), NM-EMBER (EMBER's set and subspace).
**Knob:** w_f ∈ {0.5, 2, 8}; fixed steps and learning rate, tuned on one pilot run (Baseball s42, training loss and
copy only; no evaluation metric used).
**Selection** (validation, as E3): MMLU-frac ≥ 0.9, validation collateral ≥ 0.8, copy (COPY carriers) ≥ 0.9, max
validation H3v. Test pass at the selected point (`result_nm.json`).

**Endpoints.** Primary: nonce-point attainment per run = test efficacy ≥ (per-run nonce floor of the set − 0.05) and
copy ≥ 0.9 and test collateral ≥ 0.8; attainment rate per variant. Secondary: paired test H3v vs the E3-selected
EMBER and CINDER-z (unit = concept, Wilcoxon), exclusive-word copy, echo and context-MC (judge-free), T3 destruction
vs the nonce reference and generation broken rate at the selected points (seed 42, blind labels).
**Predictions.** (a) NM-CINDER reaches the nonce point in most runs where it edits. (b) NM-free ≥ NM-CINDER.
(c) NM-EMBER edits fewer runs (empty sets) and is no better where it edits. No Gemini.

**Note (2026-10-21, 21:30): amendment 11 item 1 stopped early (by construction).** The model loader unties Gemma's
lm_head (`ember/erasure/model_loader.py`), so every edit changes input rows only. A prompt with no edited token id
runs exactly as in the unedited model, so word-free efficacy is 0 unless the MC frame contains an edited token. A CPU
check finds none of the frame's 34 token ids in any of the 255 run-variant edit sets. The word-free jobs were stopped
after 9 run-variants (Golf s44, Gun s43) confirmed it: accuracy on 43–51 new items and 4–6 old items equals the
unedited model at both the collateral-rule and the paper-rule point (efficacy 0.000 in all 18 configs).

## Amendment 12 (2026-10-21, 21:30): Alpaca coherence at the E3 points (user-approved, about 6 NIS)

Written before any E3 result exists. Alpaca coherence (50 prompts, the development scorer) at seed 42's E3 point
(amendment 11 item 2) of each of the 5 variants that edit under E3. A point equal to the collateral-rule or
paper-rule point reuses that score. Reported: coherence per variant, H4 = HM(efficacy, specificity, coherence,
collateral) under E3, and paired H4 vs EMBER on seed 42 (descriptive). Gemini hard stop GEMINI_BUDGET_NIS = 38 for
these jobs (31.82 spent before). Prediction: coherence ≥ 0.95 at every E3 point (smaller r than the collateral rule).

**Amendment 13 addendum (2026-10-21, 22:15; after the pilot, before any efficacy or collateral result).** The pilot
(Baseball s42, training losses and evaluation-carrier copy only) showed the rank-≤3 CINDER subspace barely lowers the
forget loss, while the free edit does (KL 2.70 → 0.28, copy 0.98) at a large move (r = 3.5 at w_f = 0.5). Changes:
(1) variant NM-free-EMBER (EMBER's token set, unconstrained Δ) added, so NM-free (CINDER-z set) vs NM-free-EMBER
isolates the token set; (2) the knob path is w_f = 0.05 → 0.15 → 0.5 → 2 (120, 60, 60, 60 steps), adding weaker forget
weights that keep r near or below 1, so the selection can find points with collateral ≥ 0.8. Grid rows report r,
validation collateral and copy per knob. Prediction for the token set: NM-free ≥ NM-free-EMBER, and NM-free edits more
runs.

**Amendment 13 addendum 2 (2026-10-21, 23:00; before any result of the variant below).** Follow-up variant
**NM-QA**: NM-free on the CINDER-z set plus forget data from concept questions written by the unedited model itself
(3 per token: "Write 3 short quiz questions about {concept} that use the word '{w}'"), target = the unedited model's
greedy answer to the nonce version of each question (KL per answer position, as the definitional prompts). Questions
that match any qa / simdom / collateral item exactly or nearly (difflib ratio ≥ 0.8 after normalisation, or word
Jaccard ≥ 0.6, or word containment ≥ 0.8, added before any data was generated) are dropped. Same knob path and selection, learning rate 0.003 (instead of 0.01) so r can stay near 1.
**Launch rule:** NM-QA runs only if, in at least 2 of the first 3 concepts completed by NM-free, the selected point's
test efficacy is below its nonce floor by more than 0.1. **Scoring caveat:** NM-QA trains on the efficacy test's format
(held-out items), so it is judged on the other formats as well (copy, T3, generation, echo, context-MC); an efficacy
gain without them is not the nonce point.

**Addendum 2 launch (2026-10-22, 00:05).** First 3 NM-free concepts (test efficacy vs nonce floor): Baseball 0.05 vs
0.51, Cannabis 0.34 vs 0.52, COVID-19 0.00 vs 0.00. Rule met (2 of 3 below by > 0.1); NM-QA submitted to start at
01:30 (GPU share).

**Results of amendment 11 (2026-10-22; last_mile.md, doc F36–F38 and D19).**
- Word-free efficacy: 0 by construction (untied output head; frame tokens never edited), confirmed on 9 run-variants
  at both points (efficacy 0.000 in 18 configs); jobs stopped early (note above).
- E3 (copy ≥ 0.9): paired H3v vs EMBER CINDER +0.096, CINDER-RS +0.092, CINDER-z +0.029, CINDER-RS-z +0.026 (Holm p
  0.42, 0.42, 0.75, 0.75): not significant. Intention-to-treat (post hoc, amendment 9's definition): +0.134, +0.166,
  +0.053, +0.063 (Holm 0.015, 0.006, 0.33, 0.33). Efficacy at the E3 point 0.06 (EMBER), 0.12, 0.14, 0.07, 0.07;
  exclusive-word copy 0.94–0.98. Prediction (CINDER variants beat EMBER under E3): supported in direction for CINDER
  and CINDER-RS, significant only under intention-to-treat; not for A3z.
- Context damage at the collateral-rule point (existing E2 outputs, post hoc): context-MC below the real-word
  reference for CINDER-z (9 of 11 concepts, p 0.024) and CINDER-RS-z (10 of 12, p 0.003), not for EMBER, CINDER or
  CINDER-RS.

**Results of amendment 12 (2026-10-22).** All 48 seed-42 E3 points (45 scored, 3 reused), 5.68 NIS (total 37.50 of the
38 stop). Coherence 0.97–0.98 for every variant, none below 0.9 (prediction ≥ 0.95: met). H4 at the E3 point: EMBER
0.18, CINDER 0.35, CINDER-RS 0.20, CINDER-z 0.16, CINDER-RS-z 0.25; paired vs EMBER CINDER +0.169 (6/1), CINDER-RS
−0.086, CINDER-z −0.026, CINDER-RS-z +0.038 (descriptive).
