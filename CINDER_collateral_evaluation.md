# Token destruction: collateral-aware evaluation of embedding erasure

Oct 19, 2026 · @Ori

**How to read this.** Part 1 is the overview: the problem, what we observed, the idea, the hypotheses and the planned experiments. Every statement there is marked as an **observation** (backed by a run) or a **hypothesis** (to be tested). Part 2 gives the low-level details: exact definitions, formulas, the protocol, the pilot, result placeholders and caveats. Part 3 holds the naming map, open items, sources and references. Paper section numbers refer to EMBER (Suslik, Shafran and Geva, arXiv:2606.03695v2).

---

## Part 1. Overview

### The problem

EMBER edits a concept-only token by subtracting its concept part: e′ = e − δc, where c is the token's reconstruction from the concept features (§3.3). The paper notes that δ > 1 "over-subtracts which we find to often improve erasure in practice", and tunes δ on {0.5, …, 200} (§C.3). At the top of that grid the edited row is no longer the token minus its concept part. It is a new vector, tens of times longer than the token, pointing along −c. The token is **replaced, not cleaned.**

Many edited tokens mean other things outside the concept: " Harry" (Prince Harry, Harry Styles, Harry Truman), " Sherlock", " Trump", " Beatles", " Lego". Replacing them deletes those meanings too. EMBER's evaluation does not see this. The similar-domain questions do not contain the edited tokens, the MMLU gate never binds, and Alpaca prompts rarely contain them. The efficacy questions do contain them, so a destroyed token counts as erased knowledge. We call this **token destruction**: the protocol rewards it.

### What we observed (backed)

Gemma-2-2b-it, EMBER's MC protocol, the Oct 18 erasure anchor (seed 42) unless noted.

- **Observation 1: the δ search goes to the top of the grid.** For Harry Potter, EMBER chose δ = 200, both with our cheaper rule and with the paper's own Alpaca-in-grid tuning (H 0.726 vs 0.727). EMBER chose δ = 200 on 6 of the 10 anchor concepts it edits. Over all rungs, exploratory ones included, 11 of 41 non-copied edited runs chose δ = 200.
- **Observation 2: at that δ the token is replaced.** The median relative edit magnitude µ = ‖c‖/‖e‖ (§F.1) is 0.605 for EMBER on Harry Potter. At δ = 200 the edited row is about 120× the token's norm and points along −c. Validation efficacy plateaus well before that: δ = 200 beats δ = 5 by one validation question.
- **Observation 3: nothing in the protocol penalizes it.** The MMLU gate never bound in 344 grid points. Alpaca coherence was ≥ 0.87 at every chosen point.
- **Observation 4: the edit set holds many multi-meaning tokens.** EMBER's 31 Harry Potter tokens include " Beatles", " Sherlock", " Trump", " Kipling", " Nestlé", " Lego", " HBO" and " Bloomsbury". CINDER's 51 add wand, wizard and Hog, but also Sherlock, Holmes and Merlin.
- **Observation 5: the damage shows in generation.** On 20 neutral-context prompts at δ = 200, EMBER breaks 12/20 for Harry Potter, and CINDER breaks 9/20. Asked about Prince Harry, the erased model answers about the singer Prince. At δ = 1 both methods break 0–1 prompts per concept. This contradicts the paper's §5 Prince Harry example at the δ that our reproduction of the paper's tuning selects.
- **Observation 6: a collateral term sees the damage.** In the pipeline pilot (Harry Potter, one seed), EMBER's collateral on its own edited multi-meaning tokens falls from 1.00 at small perturbation to 0.57 at r = 100. Over the same range, efficacy rises only to 0.49.

### Why it matters

- **The paper's specificity claim needs this axis.** §5 argues that coherence damage is localized on concept-exclusive tokens. It measures this after tuning, on one prompt per token, and the result never enters the δ search. In the search, damage to multi-meaning tokens is free.
- **Destruction blurs the method comparison.** Once a token is replaced, the subtracted direction barely matters. In the pilot at r = 10–100, Noise of the same magnitude reached efficacy 0.39, the same as an edit along CINDER's direction. The paper's Table 1 argument, that direction matters (Gemma MC Con: EMBER 49.4 vs Mean 62.8 and Noise 62.2, unedited 73.3), was made at the δ the paper's tuning selects, which our Observation 1 suggests is in this regime.
- **The anchor compared different regimes.** On Harry Potter, EMBER was selected at δ = 200 (r ≈ 120), and CINDER at δ = 2 (r ≈ 0.4). A ranking of feature quality, which is what CINDER changes, should compare methods where the direction matters.

### The idea

Add a fourth term, **collateral**: does the edited model still know ordinary facts that use the edited multi-meaning tokens outside the concept? Make it essential to the score, then re-run the search for every method on one grid of relative perturbation.

| Piece | In one sentence |
| --- | --- |
| Collateral term | MC accuracy on questions where an edited multi-meaning token appears in a context unrelated to the concept, normalized like the paper's metrics. |
| H4 | The paper's harmonic mean with collateral as a fourth axis, as heavy as efficacy. |
| Collateral gate | The δ search only accepts points with collateral ≥ 0.8 on validation. |
| Common grid | All variants sweep the same target relative perturbation r, so methods with different µ get comparable sweeps. |
| New edit variants | A 2×2 that separates token selection from direction, and edit shapes that remove less from multi-meaning tokens. |

### Hypotheses (to be tested)

- **Hyp. 1: regime shift.** Under collateral, the selected point moves from the destruction regime (r ≫ 1) to the subtraction regime (r ≲ 1).
- **Hyp. 2: the comparison changes.** In the subtraction regime the direction matters, so the EMBER vs CINDER comparison differs from the anchor. The pilot hints at this: at r = 1, an edit along CINDER's direction on EMBER's tokens reached efficacy 0.42, against 0.28 for EMBER.
- **Hyp. 3: the 2×2 separates the two effects.** Changing only the token set and changing only the direction have separate, measurable effects on efficacy and collateral.
- **Hyp. 4: shaped edits trade better.** Bounded projection and exclusivity-weighted edits reach a better efficacy–collateral trade-off than e − δc.
- **Expected limit: context blindness.** An embedding edit sees the token, not its context. " Harry" in Prince Harry and in Harry Potter is the same row, so no input-embedding edit can erase one and keep the other. The Pareto curves, not a single score, are the honest output.

### Planned experiments

1. **Collateral question set** for the 3 development concepts (Harry Potter, Ancient Rome, Valentine's Day), written and reviewed blind to the methods, checked by script, and frozen with a hash before any edited model is scored on it.
2. **Grids** for all variants on r ∈ {0.05, …, 100}, with validation scores at every point (efficacy, specificity, collateral).
3. **Selection** by the collateral-aware cheaper rule, then the test split and Alpaca at the chosen point for the primary variants.
4. **Report** H, H4 and the paper-structured variant, the 2×2 effects, the edit-shape comparison, Pareto curves and gate sensitivity.
5. **After this stage, with the team:** expand to the other erasure concepts, and test relearning or LoRA healing.

### Where we stand

- **Done:** the premise check (Observation 5), the pipeline pilot (Observation 6), the token pools, and the first question drafts for all 3 concepts.
- **Running:** the review rounds, the selection of one version per concept, and the freeze.
- **Not started:** the real grids. No result in this doc yet uses the frozen question set.

### What to keep in mind

- **Efficacy will likely fall for every method.** The efficacy questions contain the same tokens, so part of today's efficacy is destruction.
- **MC collateral is lenient.** Options and context can rescue a destroyed token: in the pilot, Sherlock and Trump stay at 1.0 in MC while generation fails. So collateral is a conservative measure.
- **The questions are written by Claude agents.** Their blind spots are correlated. The objective checks and a human spot check are the counterweights.
- **3 concepts × 2–3 seeds give direction, not significance.** Claims come after the freeze, on new concepts.

---

## Part 2. Details

### D1. EMBER's edit and the two regimes

**The edit (§3.3).** For each concept-only token t that some concept feature uses:

```latex
e'_t = e_t - \delta\, c_t, \qquad c_t = \sum_{i \in F_C} Y_{i,t}\, z_i, \qquad t \in \{\, t : \mathrm{codes}_t = 1,\ \exists\, i \in F_C:\ Y_{i,t} \neq 0 \,\}
```

Our code edits exactly this set ([erase.py](experiments/varwidth/erase.py), `edit_vectors`: support of the accepted features ∩ codes = 1), for EMBER and for CINDER.

**Size of the edit.** With µ\_t = ‖c\_t‖/‖e\_t‖ (§F.1):

```latex
\frac{\|e'_t\|^2}{\|e_t\|^2} = 1 - 2\,\delta\,\mu_t \cos(c_t, e_t) + \delta^2 \mu_t^2,
\qquad
\frac{e'_t}{\|e'_t\|} \;\to\; -\hat c_t \quad (\delta \mu_t \gg 1)
```

Taking the Harry Potter EMBER medians as one token (µ = 0.605, cos = 0.62): at δ = 1 the row keeps 0.78 of its norm and most of its direction. At δ = 200 it is about 120× longer and points along −ĉ. Harry Potter's EMBER edit uses a single feature (F\_C = {87}), so all 31 edited rows then point along the same z\_87 up to sign, and differ mostly in length.

We call r ≲ 1 the **subtraction regime** (the token minus a part of itself) and r ≫ 1 the **destruction regime** (the token replaced by a multiple of the concept direction).

**Measured µ and alignment** (seed 42, medians over edited tokens):

| Concept | µ, EMBER | µ, CS+GS | cos(c, e), EMBER | cos(c, e), CS+GS |
| --- | --- | --- | --- | --- |
| Harry Potter | 0.605 | 0.189 | 0.62 | 0.36 |
| Baseball | 0.275 | 0.283 | 0.50 | 0.34 |
| Gun | 0.347 | 0.245 | 0.50 | 0.31 |

At δ = 200, the edited embeddings are 40–120× the original norm on these concepts.

**The δ search plateaus early.** Harry Potter, EMBER, validation split (50 questions), H2 = HM(efficacy, specificity):

| δ | 0.5 | 1 | 2 | 5 | 10 | 50 | 100 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qa\_frac ↓ | 0.93 | 0.75 | 0.65 | 0.51 | 0.61 | 0.51 | 0.51 | 0.47 |
| H2 | 0.131 | 0.394 | 0.519 | 0.659 | 0.557 | 0.659 | 0.659 | 0.690 |

δ = 200 wins over δ = 5 by 0.04 in qa\_frac, which is one question out of 50. This matches the paper's Fig. 11 (§C.4): efficacy rises fast up to δ ≈ 50 and then slowly, while specificity and coherence decline slowly.

**Where the anchor selected δ** (non-copied runs that edit at least one token):

| Rung | δ ≤ 10 | δ = 50–100 | δ = 200 |
| --- | --- | --- | --- |
| EMBER | 1 | 3 | 6 |
| All rungs | 17 | 13 | 11 |

The CS rungs mostly chose δ ≤ 10, but their µ is smaller, so δ is not comparable across rungs. On Harry Potter, CS+GS (and CINDER, whose edit is identical there) chose δ = 2: about r = 0.4, against r ≈ 120 for EMBER. This is why the new grid is in r, not δ (D6).

### D2. Why the protocol does not see it

| Metric (§4.1) | Contains the edited tokens? | What we found |
| --- | --- | --- |
| Efficacy (concept MC) | Yes, by construction | A destroyed token makes the question unreadable, which counts as erasure. |
| Similar domain | No | Harry Potter: 100 questions on other franchises (Tolkien, Marvel, …). None contains "Harry". |
| MMLU (gate ≥ 0.9) | Rarely | The gate never bound in 344 grid points. |
| AlpacaEval | Rarely | Coherence ≥ 0.87 at every chosen point. |
| §5 coherence analysis | Yes | Run after tuning, one prompt per token, summarized by TF-IDF. Never part of the selection. |

The paper's own Table 2 shows the signature. Asked "Who is the main character in Harry Potter?", the erased Gemma (SNMF+EMBER) answers "Please provide me with the context! …". The paper labels this a semantic shift and reads it as erasure. Our neutral prompts fail the same way: "Please provide me with the name of the company…" for Nestlé (D3). Our reading is that in both cases the model can no longer read the token, in any context.

The limitation "Noise in the feature selection process" names the source: the concept-labeled set "may include semantically unrelated tokens", and the paper suggests TF-IDF or model-based filtering. Observation 4 shows such tokens in the edit set. The exclusivity-weighted variant (D6) tests that suggestion.

### D3. Premise check (step 0)

**Protocol.** 20 prompts per concept. Most use an edited token in an outside meaning (Prince Harry, Caesar salad, the fashion house Valentino). A few are concept-adjacent topics with no edited token (owls, dragons, the Salem witch trials). Generation is greedy, on the unedited model, EMBER and CINDER, at δ = 1 and δ = 200. We read each answer against the unedited one and count a prompt as broken when the answer is not consistent with it (§5's semantic shift or incoherent). δ = 200 is EMBER's chosen value on Harry Potter; for CINDER it is a stress test (CINDER chose δ = 2 there).

| Concept (seed) | Tokens edited, EMBER / CINDER | Broken at δ = 200, EMBER | Broken at δ = 200, CINDER | Broken at δ = 1 |
| --- | --- | --- | --- | --- |
| Harry Potter (42) | 31 / 51 | 12/20 | 9/20 | 0–1 per method |
| Ancient Rome (44) | 38 / 103 | 3/20 | 12/20 | 0–1 per method |
| Valentine's Day (42) | 45 / 52 | 7/20 | 9/20 | 0–1 per method |

What breaks:

- **Harry Potter, EMBER:** Prince Harry, Harry Styles, Truman, Sherlock, Trump, Beatles, Kipling, Lego, HBO, Nestlé, Remus, Bloomsbury. **CINDER:** Prince Harry, Styles, Truman, Sherlock, Merlin, the Wizard of Oz, Sirius, Remus, Minerva.
- **Ancient Rome, CINDER:** Caesar salad, July/Julius, Alexander, Alexandria, Latin, Napoleonic, Rule Britannia, the Pantheon, Imperial College, the Empire State Building.
- **Valentine's Day, EMBER:** chocolate, Valentino, marshmallows, cookie, amethyst, violets, rose. **CINDER:** chocolate, Easter, Halloween, heart, Valencia, Valentino, dates, marriage, violets.

Examples:

| Prompt | Unedited | Edited, δ = 200 |
| --- | --- | --- |
| "Who is Prince Harry, and who is his wife?" (HP, EMBER) | "Prince Harry is the younger son of King Charles III …" | "Prince was an American singer, songwriter, and multi-instrumentalist. …" |
| "What kind of company is Nestlé, and in which country is it headquartered?" (HP, EMBER) | "Nestlé is a multinational food and beverage conglomerate. …" | "Please provide me with the name of the company you're interested in! …" |
| "Which month of the year is named after Julius Caesar, and what number month is it?" (AR, CINDER) | "The month of July is named after Julius Caesar. It is the 7th month of the year." | "The month named after a Roman a goddess is October, and it is the 10th month. …" |

**Relation to §5.** The paper reports that "the erased Gemma model answers correctly when asked about Prince Harry". Its example is SNMF+EMBER. Under §C.2, Stage 1 sweeps δ for the embedding edit alone and selects it by H, which on our reproduction gives δ = 200 for Harry Potter. At that δ, our EMBER answers about the singer Prince. Our features may differ from the paper's (our EMBER matches Table 6 on average, not token by token), so this contradicts the generality of the claim, not the specific run behind it.

**Premise verdict.** The stop rule asked whether context rescues nearly all collateral prompts. It does not, so the premise holds.

### D4. The collateral term and the score

**Per-token accuracy.** Each kept item is answered correctly by the unedited model, so the baseline accuracy is 1. EMBER's MC normalization (§C.1):

```latex
\mathrm{mc\_norm}(a, b) = \mathrm{clip}\!\left(\frac{a - 0.25}{b - 0.25},\; 0,\; 1\right)
```

**Collateral (primary).** M\_v is the set of multi-meaning tokens of the frozen set that variant v edits:

```latex
\mathrm{coll}(v) = \frac{1}{|M_v|} \sum_{t \in M_v} \mathrm{mc\_norm}(\mathrm{acc}_t,\, 1), \qquad \mathrm{coll}(v) = 1 \ \text{ if } M_v = \emptyset
```

Every token counts equally. With 2 items per token and split, acc\_t ∈ {0, 0.5, 1}, so a token scores 0, 1/3 or 1. **Secondary:** `coll_all`, the same mean over the whole union pool.

**Scores.** The paper's H and our two extensions:

```latex
H = \mathrm{HM}(\phi_{\mathrm{eff}},\, \phi_{\mathrm{spec}},\, \phi_{\mathrm{coh}}), \quad
\phi_{\mathrm{eff}} = 1 - \text{qa\_frac}, \quad
\phi_{\mathrm{spec}} = \mathrm{HM}(\mathrm{Sim}, \mathrm{MMLU}), \quad
\phi_{\mathrm{coh}} = \mathrm{HM}(\mathrm{instr}, \mathrm{flu})
```

```latex
H_4 = \mathrm{HM}(\phi_{\mathrm{eff}},\, \phi_{\mathrm{spec}},\, \phi_{\mathrm{coh}},\, \mathrm{coll}),
\qquad
H' = \mathrm{HM}(\phi_{\mathrm{eff}},\, \phi'_{\mathrm{spec}},\, \phi_{\mathrm{coh}}), \quad \phi'_{\mathrm{spec}} = \mathrm{HM}(\mathrm{Sim}, \mathrm{MMLU}, \mathrm{coll})
```

H4 is primary (pre-registered). It gives collateral the weight of efficacy. H′ keeps the paper's three-axis structure and treats collateral damage as a specificity failure, which is how §5 frames it.

**Selection (cheaper rule, as in the anchor).** On the validation split, over the grid R:

```latex
r^{*} = \arg\max_{r \in R}\; \mathrm{HM}(\phi_{\mathrm{eff}},\, \phi_{\mathrm{spec}},\, \mathrm{coll})
\quad \text{s.t.} \quad \text{MMLU-frac} \ge 0.9, \;\; \mathrm{coll} \ge 0.8
```

Ties go to the smaller r. If no point passes, there is no edit. Alpaca is not in the grid. At r\* we score the test split (qa, similar domain, MMLU 1000, collateral) and, for the primary variants, Alpaca on 50 prompts. A run with Alpaca < 0.9 is flagged, not reselected. The gate makes collateral non-tradeable: a point that is perfect elsewhere but below 0.8 is never chosen. Sensitivity at gates 0.7 and 0.9 comes from the saved grids at no cost.

### D5. The collateral question set

**Token pool.** Per concept, the union of the tokens edited by all variants over the development seeds, decoded with the Gemma tokenizer (leading-space marker and word pieces kept). Writers see the tokens in alphabetical order, with no method or rank. Pool sizes: Harry Potter 73, Ancient Rome 141, Valentine's Day 104.

**Writing loop** (one chain per concept, run in parallel; briefs in [collateral/briefs](results/varwidth/layers/collateral/briefs/)):

| Step | Role | Produces |
| --- | --- | --- |
| 1, 3, 5 | Writer | v0, v1, v2. Each token is classed exclusive (no items) or multi (4 MC items in contexts outside the concept, the exact token in the question or the correct answer). |
| 2, 4 | Reviewer | F0, F1: item-level keep / fix / drop with reasons. No rewriting. |
| 6 | Selector (fresh agent) | Picks one version per concept against a fixed rubric (neutrality, the token being essential, factual certainty, coverage, answerability) and lists residual problems, which are dropped mechanically. |

Two review rounds at most, so the loop does not echo. No agent sees our hypotheses, the results or the token-to-method mapping.

**Programmatic checks** (on every version; final cleanup on the chosen one):

- **Token-id check:** the token id appears in the MC prompt as the model sees it ("Question: …" / "A. …").
- **Base filter:** the unedited model answers correctly.
- **Exclusivity cross-check:** flag tokens marked exclusive that are common in Wikipedia.
- **Split:** per token, 2 items to validation and 2 to test. Tokens with < 2 usable items are dropped and listed.
- **Freeze:** the file's sha256 goes into PREREG amendment 3 before any edited model is scored on it.

**Status (drafts, before selection):**

| Concept | Pool | v0: items (multi / exclusive) | v1 | v2 |
| --- | --- | --- | --- | --- |
| Harry Potter | 73 | 244 (61 / 12); unedited model 222/244 correct | 228 (61 / 12) | 214 (58 / 15) |
| Ancient Rome | 141 | 467 (130 / 11) | — | — |
| Valentine's Day | 104 | 393 (99 / 5) | 348 (97 / 7) | — |

The pilot used Harry Potter v0 as a provisional file (60 tokens, 221 items, hash b42f96805f04f55a), for plumbing only.

**Frozen set.** \[TBD: filled from `data/collateral_questions.json` and PREREG amendment 3 after the selectors and the cleanup.\]

| Concept | Version chosen | Multi tokens kept | Exclusive | Tokens dropped | Items, val / test | Unedited correct |
| --- | --- | --- | --- | --- | --- | --- |
| Harry Potter | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Ancient Rome | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Valentine's Day | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

File hash (sha256[:16]): [TBD]. Human spot check (20 random items per concept): [TBD].

### D6. Variants

**One grid for all.** Each variant sweeps the target median relative perturbation:

```latex
r = \operatorname{median}_t \frac{\|e'_t - e_t\|}{\|e_t\|} \;\in\; \{0.05,\, 0.1,\, 0.25,\, 0.5,\, 1,\, 2,\, 5,\, 10,\, 30,\, 100\},
\qquad e' = e - \delta c \;\Rightarrow\; \delta = \frac{r}{\operatorname{median}_t \mu_t}
```

Every grid row records the achieved r. The grid covers the paper's operating point (EMBER on Harry Potter at δ = 200 is r ≈ 120).

| Group | Variant | Edit | Grid |
| --- | --- | --- | --- |
| Existing rungs | EMBER, CINDER, +CS+GS, EMBER + TJ∨WJ+share | e − δc, sparse Y as published | r |
| Paper baselines (§4.2) | Mean | e ← vocabulary mean | one point |
| | Noise | e + δ‖c\_t‖u\_t, u\_t a fixed random unit vector | r |
| 2×2 (dense Y) | ee, 1b, 1a, cross | e − δc, see below | r |
| Edit shapes | Bounded projection | e − α(e·ĉ)ĉ | α ∈ {0.25, 0.5, 0.75, 1} |
| | Exclusivity-weighted | e − δ w\_t c\_t | r |
| | Power | e − δ c\_t^(m) | r, m ∈ {1.5, 2} |

Mean and Noise run on the EMBER and CINDER token sets; the edit shapes too.

**The 2×2.** N is EMBER's edit count. All four cells use dense coefficients, Y\_dense = (ZᵀZ + λI)⁻¹ZᵀA, so every token has a coefficient on F\_C and the cells are comparable.

| Token set ↓ / direction → | EMBER direction (EMBER's fit and F\_C) | CINDER direction (CS+GS fit, CINDER's F\_C) |
| --- | --- | --- |
| **EMBER's edit set** | ee | 1b |
| **CINDER's top-N concept-only tokens by µ\_j** | cross | 1a |

A difference along a row changes only the direction, at fixed tokens. A difference along a column changes only the tokens, at fixed direction:

```latex
\Delta_{\mathrm{dir}} = \tfrac12\big[(1b - ee) + (1a - cross)\big],\quad
\Delta_{\mathrm{tok}} = \tfrac12\big[(cross - ee) + (1a - 1b)\big],\quad
\Delta_{\times} = (1a - cross) - (1b - ee)
```

Each cell value is a metric (efficacy, collateral, H4) at the cell's selected r, and also along the grid at fixed r. If EMBER selects nothing, N is the median EMBER count over seeds 42–46, flagged. We report the overlap of the µ\_j ranking with a raw-Y ranking.

**Edit shapes.**

```latex
\text{proj}_\alpha:\; e' = e - \alpha\,(e^{\top}\hat c)\,\hat c, \qquad \|e' - e\| = \alpha\,\|e\|\,|\cos(c, e)| \le \alpha\,\|e\|
```

```latex
\text{excl}:\; e'_t = e_t - \delta\, w_t\, c_t, \quad w_t = \frac{\mathrm{rank}(x_t)}{|T|} \in (0, 1], \quad x_t = \log_{10}\frac{f_C(t)}{(n_W(t) + 1)/N_W}
```

```latex
\text{pow}_m:\; e'_t = e_t - \delta\, c^{(m)}_t, \qquad c^{(m)}_t = \sum_{i \in F_C} \operatorname{sign}(Y_{i,t})\,|Y_{i,t}|^{m}\, z_i
```

- **Bounded projection** removes at most the token's own component along ĉ, so it never enters the destruction regime. At α = 1 it is not the same as δ = 1: δ = 1 removes c\_t, whose length µ‖e‖ can be larger or smaller than the projection.
- **Exclusivity weighting** scales the edit by the token's percentile, within the edit set, of its concept-to-Wikipedia log frequency ratio (f\_C from the concept sentences, n\_W from 3,000 Wikipedia articles). It is close to the paper's TF-IDF of §F.2, not identical, and implements the Limitations suggestion.
- **Power** shifts the edit toward tokens with large coefficients. δ is recomputed from the transformed c, so only the shape across tokens changes, not the overall scale.
- **m and α** are chosen once, pooled over the 3 concepts.

**Primary variants** (scored on Alpaca): EMBER, CINDER, EMBER + TJ∨WJ+share, 1a, 1b, and the best edit shape.

### D7. Pipeline pilot

Harry Potter, seed 42, 4 grid points, the provisional v0 items ([log](results/varwidth/layers/slurm/logs/cl_pilot_1055037.out)).

**Checks passed.**
- **Reproduction:** sparse EMBER at δ = 200 gives the anchor's test numbers exactly (qa 0.58, similar domain 0.88, MMLU 0.552).
- **Dense vs sparse Y:** the ee cell and sparse EMBER give identical directions (cosine 1.000).
- **Unedited model:** collateral 1.0 by construction.
- **Mean lowers collateral:** 0.75 on its tokens, so the set can detect damage (a stop rule).

**Collateral on each variant's own edited multi-meaning tokens** (r is the achieved median relative perturbation):

| Variant | Efficacy, r = 0.1 / 1 / 10 / 100 | Collateral, r = 0.1 / 1 / 10 / 100 |
| --- | --- | --- |
| EMBER | 0.00 / 0.28 / 0.42 / 0.49 | 1.00 / 0.93 / 0.67 / 0.57 |
| CINDER | 0.11 / 0.46 / 0.56 / 0.56 | 1.00 / 0.85 / 0.66 / 0.68 |
| 1b (EMBER's tokens, CINDER's direction) | 0.04 / 0.42 / 0.39 / 0.39 | 1.00 / 0.92 / 0.73 / 0.67 |
| Noise (EMBER's tokens) | 0.00 / 0.11 / 0.39 / 0.39 | 0.97 / 0.85 / 0.72 / 0.70 |
| Mean (one point, r = 0.87) | 0.32 | 0.75 |

Bounded projection at α = 1 reached r = 0.62 and efficacy 0.14.

What the pilot shows (one concept, one seed, 50 validation questions; hints only):

- **Efficacy saturates where collateral keeps falling.** EMBER gains 0.07 efficacy from r = 10 to 100, and loses 0.10 collateral.
- **Direction matters at small r, not at large r.** At r = 1, EMBER's efficacy is 0.28, 1b's 0.42 and Noise's 0.11. At r = 10–100, Noise matches 1b (0.39).
- **A 0.8 gate would admit EMBER only up to r = 1** on this grid.

**Design decision made on the pilot, before real data.** The union-pool mean diluted the damage: EMBER at r = 100 scored 0.86 on the union, 0.57 on its own tokens, while breaking 12/20 neutral prompts. It also made a method's score depend on which other methods built the pool. So the primary collateral is over the tokens the variant edits, and the union mean is secondary (PREREG amendment 3).

### D8. Protocol

**Development concepts and seeds.**

| Concept | Seeds | Note |
| --- | --- | --- |
| Harry Potter | 42, 43 | |
| Ancient Rome | 42, 43, 44 | EMBER selects nothing at seed 42. EJ uses the plan-B lexicon prompt v1r. Under v1r, CINDER selects nothing at seed 43. |
| Valentine's Day | 42, 43, 46 | |

Seeds 44 and 46 were added so that each concept has a seed where both EMBER and CINDER select features (a complete 2×2). They were chosen from feature counts only, before any collateral result.

**Ancient Rome lexicon (plan B).** Ancient Rome is one of the examples in the lexicon prompt v1, which is why it was not in the anchor. `prompt_v1r.txt` replaces that example with Ancient Egypt and is used only for Ancient Rome; v1 stays the default. Word-list overlap v1r/v1: Ancient Rome 18/20, Harry Potter 15/18, Valentine's Day 14/19.

**Phases.** Phase 1: grids with qa, similar domain and MMLU on validation. Phase 2: collateral on validation at every grid point, then selection, the test pass and Alpaca for the primary variants. Outputs go to `results/varwidth/layers/collateral/<concept>/s<seed>/<variant>/{grid.json, result.json}`.

**Parallel lanes.** CPU (pools, edit caches, checks; ≤ 4 workers), agents (writer, reviewer and selector chains), and SLURM (`cl_*` jobs, ≤ 6 GPUs). The lanes meet at checkpoints C0 (premise) to C4 (report).

**Stop rules** (pause and report):
- **Premise:** context rescues nearly all collateral prompts. Not triggered (D3).
- **Pilot:** a check fails and cannot be explained. Not triggered (D7).
- **Detection:** Mean does not lower collateral. Not triggered (D7).
- **Budget:** projected Gemini spend passes 22 NIS; hard stop at 25 NIS, shared with the multilingual work.

**Budget.**

| Item | Cost |
| --- | --- |
| Gemini: v1r word lists, Alpaca at the chosen point (6 primaries × 6 runs), pilot Alpaca | ≈ 4.7 NIS for the stage |
| GPU | about 100 grid points per concept and seed, about 1.5 minutes each |
| Claude agents | about 18 calls for the question set |

**After this stage**, with the team: expand to the other erasure concepts of the anchor, and later relearning (§C.6) or LoRA healing of the edited model. A smaller r may be easier to relearn, which §C.6 would show.

### D9. Results

\[TBD: all tables below are filled from `results/varwidth/layers/collateral/<concept>/s<seed>/<variant>/result.json` and `grid.json`, and summarized in `results/varwidth/layers/collateral/poc.md`. Means are over concept × seed.\]

**R1. Main table** (test split, at each variant's selected point):

| Variant | r\* | Efficacy | Specificity | Collateral | Coherence | H | H4 | H′ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CINDER | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| +CS+GS | [TBD] | [TBD] | [TBD] | [TBD] | — | — | — | — |
| EMBER + TJ∨WJ+share | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 1a | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 1b | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Best edit shape | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Mean | — | [TBD] | [TBD] | [TBD] | — | — | — | — |
| Noise | [TBD] | [TBD] | [TBD] | [TBD] | — | — | — | — |

The per-concept version of R1 (one block per concept and seed) goes in poc.md.

**R2. Regime shift (Hyp. 1).** Selected point per concept, old rule vs collateral-aware rule:

| Concept | EMBER r, anchor rule | EMBER r\*, new rule | CINDER r, anchor rule | CINDER r\*, new rule |
| --- | --- | --- | --- | --- |
| Harry Potter | ≈ 120 (δ = 200) | [TBD] | ≈ 0.4 (δ = 2) | [TBD] |
| Ancient Rome | [TBD] | [TBD] | [TBD] | [TBD] |
| Valentine's Day | [TBD] | [TBD] | [TBD] | [TBD] |

**R3. 2×2 effects (Hyp. 3)**, at the selected points and at fixed r = 1:

| Metric | Δ\_dir | Δ\_tok | Δ\_× |
| --- | --- | --- | --- |
| Efficacy | [TBD] | [TBD] | [TBD] |
| Collateral | [TBD] | [TBD] | [TBD] |
| H4 | [TBD] | [TBD] | [TBD] |

µ-ranking vs raw-Y ranking overlap of the top-N sets: [TBD].

**R4. Edit shapes (Hyp. 4)**, on the EMBER and CINDER sets:

| Shape | Parameter | r\* | Efficacy | Collateral | H4 |
| --- | --- | --- | --- | --- | --- |
| e − δc | — | [TBD] | [TBD] | [TBD] | [TBD] |
| Bounded projection | α = [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Exclusivity-weighted | — | [TBD] | [TBD] | [TBD] | [TBD] |
| Power | m = [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**R5. Pareto curves.** \[TBD: efficacy vs collateral on validation, one curve per variant over the r grid, per concept, from the saved grids. The figure path goes here.\]

**R6. Gate sensitivity.**

| Variant | r\* / H4, gate 0.7 | r\* / H4, gate 0.8 | r\* / H4, gate 0.9 |
| --- | --- | --- | --- |
| EMBER | [TBD] | [TBD] | [TBD] |
| CINDER | [TBD] | [TBD] | [TBD] |
| EMBER + TJ∨WJ+share | [TBD] | [TBD] | [TBD] |

**R7. Conclusions.** \[TBD: one line per hypothesis (supported, not supported, unclear), the H vs H4 ranking of methods, and what this means for CINDER_improvements.md.\]

### D10. Known limits

- **Context blindness.** An input-embedding edit cannot separate Prince Harry from Harry Potter. There is a ceiling on efficacy at a given collateral, for every method in this family.
- **Efficacy questions share the tokens.** Lower efficacy under the new rule partly measures less destruction, not worse features.
- **MC leniency.** Recognition survives better than generation, as the paper finds for MC vs OE (§4.1). MC collateral underestimates generation damage (Sherlock and Trump in the pilot).
- **Correlated question writers.** All agents are Claude. The token-id check, the base filter, the exclusivity flag and a human spot check counter this only partly.
- **Exclusive vs multi is a judgment.** A token wrongly called exclusive hides damage; one wrongly called multi forces poor items. The selector's rubric and the exclusivity flag check this.
- **Granularity.** 2 items per token and split give per-token scores in {0, 1/3, 1}, and validation qa has 50 questions (one question ≈ 0.035 qa\_frac on Harry Potter).
- **Small development set.** 3 concepts × 2–3 seeds give direction only. The seeds 44 and 46 were picked for a complete 2×2, from feature counts only.
- **Deviations from the paper.** Top-20 candidate features instead of τ = 2; no Alpaca in the grid; collateral is new.

---

## Part 3. Reference

### Names: experiments → this doc

| In the experiments | In this doc |
| --- | --- |
| L0, `sub:ember` | EMBER |
| L1 | +CS |
| L12, `sub:cs_gs` | +CS+GS |
| L123, `sub:cinder` | CINDER (+CS+GS+TWJ) |
| L0W, `sub:ember_judge` | EMBER + TJ∨WJ+share |
| `sub:ee` / `sub:1b` / `sub:1a` / `sub:cross` | 2×2 cells (dense Y) |
| `mean:<set>`, `noise:<set>` | Mean, Noise (§4.2) |
| `proj:<set>`, `excl:<set>`, `pow1.5:<set>`, `pow2:<set>` | Bounded projection, exclusivity-weighted, power |
| `param`, `r` | δ or α; achieved median relative perturbation |
| `collateral`, `coll_all`, `coll_tok` | Primary collateral, union-pool collateral, per-token accuracy |
| `H2` / `H3v` | HM(efficacy, specificity) / selection score HM(efficacy, specificity, collateral) |
| `H`, `H4` | The paper's H; H4 |
| codes = 1 | Concept-only tokens |
| step 0 / premise | Premise check (D3) |

### Open items

- [ ] Select one version per concept, clean up, freeze `data/collateral_questions.json`, and put the hash in PREREG amendment 3.
- [ ] Human spot check of 20 items per concept.
- [ ] Phase 1 and phase 2 grids for all variants, concepts and seeds; selection; test pass; Alpaca for the primaries.
- [ ] Choose m and α pooled over the 3 concepts.
- [ ] Fill R1–R7 and write `collateral/poc.md`.
- [x] µ and cos(c, e) medians of D1 saved: `python -m experiments.varwidth.collateral_diag mu` → [diag/mu\_cos.json](results/varwidth/layers/collateral/diag/mu_cos.json).
- [x] Expansion list: the 12 anchor concepts not used for development (the 14 anchor concepts minus Harry Potter and Valentine's Day; Ancient Rome is not an anchor concept). The plan's "11" was a miscount.
- [ ] Decide whether to re-run the premise prompts at the selected points.
- [ ] Merge into CINDER_improvements.md once results exist.
- [x] Edit set checked against the code: `edit_vectors` in [erase.py](experiments/varwidth/erase.py) edits support ∩ codes = 1, as in §3.3.
- [x] Variant definitions checked against [erase\_v2.py](experiments/varwidth/erase_v2.py) and [collateral.py](experiments/varwidth/collateral.py).

### Sources of the numbers

- Anchor δ choices, validation curves and gate counts: [results/varwidth/layers/erasure/](results/varwidth/layers/erasure/) (`<concept>/<rung>_s42/{grid.json, result.json}`; the paper-grid run is `Harry_Potter/L0_s42_papergrid`).
- Anchor summary and fidelity: [results/varwidth/layers/SUMMARY.md](results/varwidth/layers/SUMMARY.md), [results/varwidth/layers/erasure.md](results/varwidth/layers/erasure.md).
- Similar-domain questions: [data/mc\_questions.json](data/mc_questions.json).
- Premise check: [premise\_prompts.json](results/varwidth/layers/collateral/premise_prompts.json), [premise\_generations.json](results/varwidth/layers/collateral/premise_generations.json).
- Pilot: [collateral/Harry\_Potter/\_pilot\_s42\_v0metric/](results/varwidth/layers/collateral/Harry_Potter/_pilot_s42_v0metric/) (grids record the union collateral; the own-token collateral of the pilot table is recomputed from the saved per-token accuracies by `python -m experiments.varwidth.collateral_diag pilot` → [diag/pilot\_own\_tokens.json](results/varwidth/layers/collateral/diag/pilot_own_tokens.json)), [cl\_pilot\_1055037.out](results/varwidth/layers/slurm/logs/cl_pilot_1055037.out).
- Token pools and drafts: [collateral/pool/](results/varwidth/layers/collateral/pool/), [collateral/drafts/](results/varwidth/layers/collateral/drafts/).
- Pre-registration: [PREREG\_amend3\_draft.md](results/varwidth/layers/collateral/PREREG_amend3_draft.md), [PREREG.md](results/varwidth/layers/PREREG.md).
- Edit magnitudes µ and cos(c, e): [diag/mu\_cos.json](results/varwidth/layers/collateral/diag/mu_cos.json).
- Code: [erase.py](experiments/varwidth/erase.py), [erase\_v2.py](experiments/varwidth/erase_v2.py), [collateral.py](experiments/varwidth/collateral.py), [collateral\_diag.py](experiments/varwidth/collateral_diag.py).

### References

- Belrose et al. LEACE: Perfect linear concept erasure in closed form. NeurIPS 2023.
- Deeb and Roger. Do unlearning methods remove information from language model weights? arXiv:2410.08827, 2024.
- Eldan and Russinovich. Who's Harry Potter? Approximate unlearning in LLMs. arXiv:2310.02238, 2023.
- Hendrycks et al. Measuring massive multitask language understanding. ICLR 2021.
- Lynch et al. Eight methods to evaluate robust unlearning in LLMs. arXiv:2402.16835, 2024.
- Suslik, Shafran and Geva. Don't forget your embeddings: robust knowledge erasure via precise editing of embeddings (EMBER). arXiv:2606.03695, 2026.
- Taori et al. Stanford Alpaca: an instruction-following LLaMA model. 2023.
