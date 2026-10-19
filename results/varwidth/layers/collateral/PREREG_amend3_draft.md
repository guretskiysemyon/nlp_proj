
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
- **File:** data/collateral_questions.json, sha256[:16] = `<HASH>`.

**Concepts and seeds (development).**
- Harry Potter 42, 43; Ancient Rome 42, 43, 44; Valentine's Day 42, 43, 46.
- Seeds 44 and 46 are added so that each concept has a seed where both EMBER and CINDER select features (complete
  2×2). They were chosen from feature counts only, before any collateral result.
- No confirmatory claims come from these 3 concepts.

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
