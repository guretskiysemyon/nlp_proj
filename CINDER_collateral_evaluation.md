# Token destruction: collateral-aware evaluation of embedding erasure

Oct 21, 2026 · @Ori

**How to read this.** Part 1 is the overview: the problem, what we observed, the idea, the hypotheses with their verdicts, the planned experiments, the findings and the open speculations. Every statement there is marked as an **observation** or **finding** (backed by a run) or as a **hypothesis** or **speculation** (to be tested). Part 2 gives the low-level details: exact definitions, formulas, the protocol, the pilot, the results and caveats. Part 3 holds the naming map, open items, sources and references. Paper section numbers refer to EMBER (Suslik, Shafran and Geva, arXiv:2606.03695v2).

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
- **Destruction blurs the method comparison.** Once a token is replaced, the subtracted direction barely matters. In the pilot at r = 10–100, Noise of the same magnitude reached efficacy 0.39, the same as an edit along CINDER's direction. Over all runs it holds for CINDER's direction on EMBER's tokens (1b 0.25–0.28 vs Noise 0.29–0.31 at r ≥ 10, validation), but EMBER's own direction stays well above Noise (0.44–0.52; F3, S8). A3z suggests why: at large r, efficacy tracks how strongly layer 0 reads the edit, not the deletion alone (F23, S9). The paper's Table 1 argument, that direction matters (Gemma MC Con: EMBER 49.4 vs Mean 62.8 and Noise 62.2, unedited 73.3), was made at the δ the paper's tuning selects, which our Observation 1 suggests is in this regime.
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
  - **Verdict: supported** (validation; F1, R2). Without collateral, EMBER's selected r is 10–100 in all 6 EMBER runs. With it, r is 0.05–2. The A3z variants shift the same way: the paper-style rule picks r = 5–100 in 5 of 7 (CINDER-z) and 6 of 8 (CINDER-RS-z) runs, the collateral rule r = 0.05–1 (0.5 in 6 of 7 and 6 of 8; F21, F22). **Confirmed on 12 new concepts:** the paper rule selects EMBER and CINDER-z at median r = 100, the collateral rule every variant at median r = 1 (F24, F28).
- **Hyp. 2: the comparison changes.** In the subtraction regime the direction matters, so the EMBER vs CINDER comparison differs from the anchor. The pilot hints at this: at r = 1, an edit along CINDER's direction on EMBER's tokens reached efficacy 0.42, against 0.28 for EMBER.
  - **Verdict: partly** (F2, F3, F13, F18). At matched small r, CINDER's direction erases more (validation efficacy at r = 0.25: 0.26 vs 0.05) and removes more knowledge with little destruction. At each method's selected point the two tie (test H4 0.48 vs 0.48). The pilot's gap at r = 1 (0.42 vs 0.28) shrinks over all runs (1b 0.31 vs EMBER 0.31; 0.31 vs 0.25 on the 5 matched runs); the main gap sits at r = 0.25–0.5. CINDER-RS, CINDER's direction without GS, has the best mean test H4 (0.53), within noise. Under the paper's own rule, A3z changes the old comparison: CINDER's selection with an edit re-solved in the original space (A3z) matches EMBER (test H2 0.55 and 0.60 vs 0.60, against 0.33 for CINDER), so most of CINDER's old-evaluation gap came from its weakly read edit space, not its features (F19, F21). Under the collateral rule A3z adds nothing (F22). **Confirmatory (12 new concepts):** at fixed r = 0.5 every CINDER direction beats EMBER in 10 of 11 concepts (Holm p ≤ 0.054, F26), and under the paper rule EMBER beats CINDER as in the anchor (F28). At the selected points, no variant beats EMBER after Holm correction; the A3z variants come closest (+0.13 and +0.08, Holm p 0.07), and CINDER and CINDER-RS are null (F24). So the comparison changes at matched strength, not at each method's selected point.
- **Hyp. 3: the 2×2 separates the two effects.** Changing only the token set and changing only the direction have separate, measurable effects on efficacy and collateral.
  - **Verdict: partly** (F6, R3). At fixed r the direction raises efficacy (Δ\_dir +0.15 at r = 0.25, +0.09 at r = 1; validation, 5 runs) and the token set changes it little (Δ\_tok −0.01 to −0.03). At the selected points the effects are heterogeneous and inconclusive (test H3v: Δ\_dir +0.06, Δ\_tok −0.08). **Confirmatory 2×2 (amendment 14, 12 new concepts, fixed r = 0.5, test):** Δ\_dir +0.073 efficacy and +0.097 H3v (10 of 11 concepts; p 0.032 and 0.014, Holm 0.097 and 0.055), Δ\_tok +0.016 and +0.024 (n.s.), collateral cost of the direction −0.019 (F44). So at r = 0.5 the separation holds on new concepts: the direction carries the effect, the token set does not. At r = 1 neither effect is present (Δ\_dir −0.016, n.s.).
- **Hyp. 4: shaped edits trade better.** Bounded projection and exclusivity-weighted edits reach a better efficacy–collateral trade-off than e − δc.
  - **Verdict: not supported at this sample size** (F2, F5, R4). Exclusivity weighting on EMBER's tokens has the best mean test H3v (0.50 vs 0.42 for EMBER), within noise. Bounded projection is worse (0.30 and 0.39). The pre-registered best shape (power m = 2 on CINDER's tokens) scored 0.37 on test.
- **Expected limit: context blindness.** An embedding edit sees the token, not its context. " Harry" in Prince Harry and in Harry Potter is the same row, so no input-embedding edit can erase one and keep the other. The Pareto curves, not a single score, are the honest output.

### Planned experiments

1. **Collateral question set** for the 3 development concepts (Harry Potter, Ancient Rome, Valentine's Day), written and reviewed blind to the methods, checked by script, and frozen with a hash before any edited model is scored on it.
2. **Grids** for all variants on r ∈ {0.05, …, 100}, with validation scores at every point (efficacy, specificity, collateral).
3. **Selection** by the collateral-aware cheaper rule, then the test split and Alpaca at the chosen point for the primary variants.
4. **Report** H, H4 and the paper-structured variant, the 2×2 effects, the edit-shape comparison, Pareto curves and gate sensitivity.
5. **E2, token integrity** (D11–D14): copy, context, context-MC, ignorance and static tests on every edited token, exclusive ones included, at E1's selected points and on a regime sweep, against the real-word, nonce and deletion references.
6. **Confirmatory expansion** (amendment 8, done): the 5 main variants on the 12 other erasure concepts × 5 seeds, under the collateral rule, fixed r = 0.5 and the paper rule, with E2 at the selected point (D17, F24–F29).
7. **Later, with the team:** relearning or LoRA healing.

### Beyond multi-meaning tokens: token integrity (Experiment E2)

**The gap.** The collateral term only sees multi-meaning tokens, and only through MC recognition. Some edited tokens are concept-exclusive (" Hogwarts", " Voldemort", "ffindor"), 13–23% of each variant's classified edited words (R11): they have no outside meaning to test, so the question set gives them no items. Yet destruction is not about other meanings. It is about the token itself: after a destructive edit the model cannot read the token in any context. Multi-meaning tokens are only the most visible case, and finding them is not always possible; for some concepts it is artificial.

**What a good erasure should look like.** When an erased token appears in the input, the model should behave like someone who has never heard of the concept: it reads the word as an unknown word, can repeat it, says it does not know what it is, and carries on with the rest of the conversation. A destroyed token instead behaves like a hole or a glitch, with several costs:

1. **The erasure is detectable.** The model reacts to the word in a way no ordinary model does ("Please provide the name…"), and the edited rows stand out in the weights (norms tens of times larger than any real token).
2. **The context breaks.** A destroyed token can derail the surrounding sentence or turn, not just the concept.
3. **It is no better than deleting the token.** An erasure that equals "remove the word from the input" is not a knowledge edit; a tokenizer blocklist would do the same.
4. **The model cannot copy, quote or spell the word**, which breaks ordinary tasks (summaries, translations, transcripts) whenever the word appears.
5. **It disrupts fluency around the word**, even in unrelated text that merely mentions it.
6. **It gives an attacker a fingerprint and an easy undo.** Edited rows are easy to locate and to restore, for example from the unedited output matrix (the paper's own limitation for tied embeddings).

**The idea.** Define the target behaviour by a reference: the **unedited model reading a matched nonce word** (a pronounceable non-word in the same slot). That is what "unaware of the concept" looks like. Two other references bound it: the unedited model with the **real word** (knows the concept) and with the **word deleted** (what destruction resembles). Every edited token, exclusive or not, is tested against them.

| Test | Question it answers | Judge |
| --- | --- | --- |
| T1 Copy | Can the model repeat a sentence containing the word? | none (string match) |
| T2 Context | Does the word derail reading the rest of its sentence, more than an unknown word does? | none (recall of a detail, answer NLL vs the nonce) |
| T2b Context-MC | Does an unrelated question go wrong after a turn that mentions the word? | none (MC) |
| T3 Ignorance | Asked "What is ⟨w⟩?", does it answer like an ignorant model, or leak, go blind or break? | echo check + blind agent label |
| T4 Static | Do the edited rows stand out in the weights? | none (norm z-score, nearest real token) |

**Hypotheses (to be tested).**

- **Hyp. 5: destruction fails integrity on exclusive tokens too.** At r ≫ 1, copy fails, the context excess exceeds the nonce budget and approaches deletion, T3 answers are blind or incoherent, and the rows are norm outliers. At r ≲ 1 the edited model stays within the nonce budget.
  - **Verdict: partly** (F9, F10, F11, F14, F17, F23). At r ≥ 10 the word fails every token test for EMBER, CINDER and CINDER-RS: copy 0, T3 destruction 0.98–1.00, every edited row a norm outlier (CINDER-z and CINDER-RS-z: copy 0, destruction 0.97–0.99). The context mostly survives: same-sentence recall stays ≈ 1.00, and only the strongly read edits, EMBER and the two A3z variants, lower context-MC (F23). The boundary is lower than predicted: at r = 1, copy is already 0.45–0.52 and destruction 0.52–0.57; only r = 0.25 stays near the nonce reference. Exclusive tokens fail first: at r = 1, 0.85–0.96 of the exclusive words are destroyed, against 0.46–0.49 of the multi-meaning ones (R11).
- **Hyp. 6: collateral is a partial proxy.** E1's collateral-aware selection lands at points with high integrity, but not always, so integrity needs its own axis.
  - **Verdict: supported** (F12, F17, R9, R11). Every E1-selected point passed collateral ≥ 0.8 on validation. Yet T3 destruction there is 0.18–0.30 for EMBER, CINDER, CINDER-RS, 1a and 1b, and 0.56 for power m = 2. Example: Harry Potter 42, EMBER at r\* = 2 (validation collateral 0.81): copy 0.03, destruction 0.83. The exclusive words, which collateral cannot see, fare worst: 0.37–0.75 of them are destroyed at the selected points, against 0.09–0.19 of the multi-meaning words (0.52 for power m = 2). The A3z variants also pass the gate at their selected points and destroy 0.74 and 0.78 of their exclusive words (F22).
- **Hyp. 7: graceful ignorance trades against leakage.** At small r the model reads the word and may still know the concept. The interesting question is whether any variant reaches low leakage with high integrity.
  - **Verdict: supported** (F13, R9b). No labelled config reaches leak ≤ 0.1 with destruction ≤ 0.1. For EMBER, leak falls only as destruction rises (r = 0.25: leak 0.58, destruction 0.00; r = 1: 0.07 and 0.57). The best trade measured is CINDER at r = 0.25 (leak 0.25, destruction 0.07), then CINDER-RS at r = 0.25 (0.29 and 0.09) and 1a at its selected points (0.12 and 0.18). A3z at r = 0.25 does not improve on it (CINDER-z 0.28 and 0.06, CINDER-RS-z 0.34 and 0.05; F22).
- **Hyp. 8: Mean and Noise are destructive by construction**, at any strength where they erase.
  - **Verdict: supported for Mean, partly for Noise** (F7, F15, R9). Mean deletes the word at its only point (copy 0, destruction 0.99) and fails E1's collateral gate in every run. Noise at its selected points (median r = 1) destroys 0.25–0.35 of the words while still leaking 0.29, and 0.60–0.64 of its rows are norm outliers.

**How it runs.** E2 reuses E1's frozen grids and selected points; it adds no new search. It runs judge-free on all 8 concept-seeds and, for T3 labels, on one seed per concept, with Claude agent judges (no Gemini). Details in D11–D14.

### Findings (backed)

E1 on 8 concept-seeds (D9), E2 on the same 8 with T3 labels on 3 (D14), A3z and the paper-rule arm through both (D16). Means are over concept-seeds; T3 numbers come from the 3 judge runs. Test collateral, and the H3v and H4 built on it, are renormalized by the unedited model's test accuracy (amendment 6, D4).

- **F1. Regime shift (Hyp. 1).** On validation, the paper-style rule (H2, no collateral) picks EMBER at r = 10–100 in all 6 EMBER runs, and EMBER + TJ∨WJ+share at r = 100 in its one separate run. The collateral rule picks r = 0.05–2. Example, Valentine's Day 42, EMBER: r 100 → 0.5, efficacy 0.44 → 0.24, collateral 0.65 → 0.94. For CINDER the two rules agree in 4 of 7 runs (r = 0.5 or 5). In the other 3 the H2 rule picks r = 10–100. For CINDER-RS the H2 rule picks r = 10 only on Harry Potter (both seeds, → 0.5), and r = 0.5–2 elsewhere (R2).
- **F2. All variants land in a narrow band.** Under the collateral rule, test efficacy is 0.14–0.27 and collateral 0.86–0.96 across variants (Mean and the single EMBER + TJ∨WJ+share run excluded). Mean test H3v: exclusivity-weighted (EMBER's tokens) 0.50; exclusivity-weighted (CINDER's tokens) and power m = 1.5 and 2 (EMBER's tokens) 0.47; CINDER-RS and 1b 0.46; EMBER 0.42; CINDER 0.42. H4 for the primaries with Alpaca: CINDER-RS 0.53, 1b 0.52, EMBER 0.48, CINDER 0.48, power m = 2 (CINDER's tokens) 0.43, 1a 0.39. The differences are within noise: 5–8 runs per variant, 50 test questions per run.
- **F3. Direction vs magnitude (matched r, validation means; R5).** Efficacy at r = 0.25: CINDER 0.26, CINDER-RS 0.21, 1b 0.20, EMBER 0.05. At r = 0.5: 0.33, 0.37, 0.33, 0.20; CINDER-RS's 0.37 is the highest of any variant at r = 0.5. At r = 1: 0.35, 0.34, 0.31, 0.31. At r ≥ 2 EMBER is higher: 0.38 at r = 2, rising to 0.52 at r = 100, against 0.32–0.36 for CINDER, 0.33–0.34 for CINDER-RS and 0.25–0.30 for 1b. On the 5 runs with a full 2×2 the same holds at fixed tokens: ee 0.05, 0.21, 0.25 vs 1b 0.20, 0.33, 0.31 at r = 0.25, 0.5, 1. CINDER's direction is more efficient per unit of perturbation in the subtraction regime. EMBER gains more from magnitude in the destruction regime, and not from magnitude alone: Noise on EMBER's tokens stays at 0.28–0.31 for r ≥ 2 (S8).
- **F4. CINDER's token sets are its liability.** CINDER's GS-widened edit sets are larger in 4 of the 5 runs shared with EMBER and hold a higher share of multi-meaning tokens in all 5: Ancient Rome 44, 82 of 103 edited tokens vs 26 of 38 for EMBER; Valentine's Day 46, 30 of 32 vs 24 of 34. Its validation collateral falls faster with r (0.93 vs 0.98 at r = 0.25, 0.88 vs 0.96 at r = 0.5), so the gate binds earlier. In Valentine's Day 43 and 46, CINDER selected r = 5, just above the gate on validation (collateral 0.81 and 0.80), and fell to 0.68 and 0.72 on test (H3v 0.19 each). CINDER-RS, without GS, has smaller sets on Ancient Rome but a collateral curve no better than CINDER's (D15), so GS is not the whole token-set problem.
- **F5. Winner's curse.** The validation-best edit shape (power m = 2 on CINDER's tokens, pooled validation H3v 0.58) scored 0.37 on test. CINDER's validation lead at the selected point (H3v 0.61 vs 0.45 for EMBER) also vanished on test (0.42 vs 0.42). CINDER-RS, with a lower validation score (0.49), kept most of it on test (0.46).
- **F6. The 2×2 is inconclusive at the selected points and separates at fixed r (R3).** At the selected points, the mean test H3v effects are Δ\_dir +0.06 and Δ\_tok −0.08, with signs that differ across runs (Δ\_dir from −0.06 to +0.34). At fixed r on validation, the direction raises efficacy in 5 of 5 runs at r = 0.25 (mean +0.15) and in 4 of 5 at r = 1 (mean +0.09, the fifth 0.00). The token set changes efficacy by −0.01 to −0.03. The direction's collateral cost is −0.01 to −0.05.
- **F7. Mean fails the gate; Noise is weak.** Mean's single point (r ≈ 0.85–0.95) has validation collateral 0.71 (0.55–0.78), so no Mean edit passes the 0.8 gate in any run. Noise reaches test H3v 0.34 (CINDER's tokens) and 0.40 (EMBER's tokens).
- **F8. E2's references validate the tests (R8).** Copy is 1.00 for orig and nonce and 0.00 for del in every run. Same-sentence recall is 1.00 for nonce and del and 0.99–1.00 for orig in every run, so T2 is valid in all 8. T3 labels: orig leak 0.51 and graceful 0.45 (KNOWS\_OTHER 0.43); nonce mostly CONFAB (0.62) and IGNORANT (0.29), destruction 0.06; del 100% BLIND. Over all 5,883 labels, agreement on 473 duplicates labelled by different agents is 0.97 on the exact label and 0.99 on destruction yes/no (0.98 and 0.99 on the first 284, before amendment 7).
- **F9. Readability collapses between r = 0.25 and r = 1 (R10).** Copy: EMBER 0.98 → 0.45 → 0 (r = 0.25 → 1 → ≥ 10), CINDER 0.95 → 0.52 → 0, CINDER-RS 0.94 → 0.50 → 0. T3 destruction: EMBER 0.00 → 0.57 → 1.00, CINDER 0.07 → 0.52 → 0.98–0.99, CINDER-RS 0.09 → 0.56 → 0.99–1.00.
- **F10. Concept-exclusive tokens go first (pilot, R11).** In the pilot (Harry Potter 42, EMBER r = 1), 14 of 30 words were dropped from the copy. All were exclusive or carried a concept word (Hogwarts, Voldemort, Snape, Weasley…); multi-meaning words copied fine. The reason: r is a median, and exclusive tokens are the high-µ tail. The full split confirms it: at r = 1, copy is 0.13–0.22 for exclusive words and 0.52–0.55 for multi-meaning ones, and T3 destruction 0.85–0.96 vs 0.46–0.49 (EMBER, CINDER, CINDER-RS).
- **F11. Destruction is local (R10).** Same-sentence recall stays ≈ 1.00 at every r. Context-MC drops only for EMBER at r ≥ 10 (0.54 → 0.45 and 0.38). The answer-NLL excess over the nonce is +1.1–1.2 nats for EMBER at r ≥ 10, against +0.12–0.24 for CINDER and +0.11–0.19 for CINDER-RS. "Destroys the whole context" is mostly not supported; "deletes the word" is. The A3z variants behave like EMBER here, not like CINDER (context-MC 0.31–0.37 and +1.2–1.5 nats at r ≥ 10; F23).
- **F12. Collateral is a partial proxy for integrity (Hyp. 6; R9).** At E1's selected points, T3 destruction and leak are: EMBER 0.30 and 0.31, CINDER 0.21 and 0.15, CINDER-RS 0.19 and 0.22, 1a 0.18 and 0.12, 1b 0.21 and 0.18, power m = 2 0.56 and 0.08, Noise 0.25–0.35 and 0.29, Mean 0.99 and 0.00. Copy (all runs): EMBER 0.65, CINDER 0.64, CINDER-RS 0.75, 1b 0.80, 1a 0.65. All these points except Mean's passed the collateral gate. EMBER's 0.30 is driven by Harry Potter 42, where it sits at r = 2 (destruction 0.83); its other two judge runs give 0.00 and 0.20.
- **F13. CINDER's direction removes knowledge without destruction at r = 0.25.** CINDER at r = 0.25: leak 0.25, destruction 0.07, CONFAB 0.17. EMBER at r = 0.25: leak 0.58, destruction 0.00, CONFAB 0.03. CINDER-edited words drift toward unknown-word behaviour (confabulation, as for the nonce). EMBER-edited words at r = 1 go blind instead (destruction 0.57, mostly BLIND). The token sets differ, but on EMBER's own tokens CINDER's direction (1b at its selected points) also confabulates more (0.16) than EMBER at r = 0.25 or 1 (0.03, 0.08). CINDER-RS at r = 0.25 behaves like CINDER (leak 0.29, destruction 0.09, CONFAB 0.15). This is the clearest evidence for CINDER's direction. It is not free: at r = 0.25 CINDER and CINDER-RS already destroy 7 of the 9 exclusive Valentine's Day 42 words, while EMBER destroys none (R11).
- **F14. Static fingerprint (R9, R10).** At r ≥ 10 every edited row is a norm outlier (z > 3 share 1.00). At the selected points the share is EMBER 0.14, CINDER 0.29, CINDER-RS 0.19, 1b 0.11.
- **F15. Mean behaves like deletion without harming the context.** Copy 0, echo 0, T3 destruction 0.99 (BLIND 0.96–0.97), while same-sentence recall stays 0.98–0.99 and context-MC 0.52–0.54.
- **F16. Two E2 context designs failed validity.** A long continuation was insensitive, and raw "Answer:" NLL was poorly calibrated, before chat-generation recall passed (amendments 4b and 4c, D12).
- **F17. The collateral term hides exclusive-token destruction (Hyp. 5, 6; R11).** At E1's selected points EMBER destroys 0.74 of its exclusive words (copy 0.36) and 0.19 of its multi-meaning ones; CINDER-RS 0.65 vs 0.09, 1b 0.43 vs 0.16, 1a 0.46 vs 0.10. At r = 1 each swept method (EMBER, CINDER, CINDER-RS) destroys 0.85–0.96 of the exclusive words and 0.46–0.49 of the multi-meaning ones. The collateral gate sees only multi-meaning tokens, so it admits these points. CINDER's selected points halve the exclusive figure (0.37), mainly because CINDER is selected at a smaller r: on Harry Potter 42 it sits at r = 0.25 and destroys 1 of 13 exclusive words, where EMBER sits at r = 2 and destroys 11 of 11. At the same r = 0.5 on Valentine's Day 42, both destroy most of them (8 of 9, 6 of 8). The judged counts are small (23–29 exclusive words per config).
- **F18. CINDER-RS is the most robust candidate so far, within noise (amendment 5; R1, R5, R9, D15).** It is the only variant with an edit in all 8 runs, Ancient Rome 43 included, where CINDER finds no feature. Test: efficacy 0.24, collateral 0.90, H3v 0.46, H4 0.53 (1b 0.52, EMBER 0.48, CINDER 0.48), mean selected r 1.03. Paired on shared runs (test H3v): +0.09 vs EMBER (2 wins, 2 losses, 2 ties; the mean comes from Ancient Rome 44, +0.51, where EMBER's selection gave no usable edit), +0.05 vs CINDER (3 wins, 2 losses, 2 ties; from Valentine's Day 46, +0.46, where CINDER erased little), +0.08 vs 1b (4 wins, 1 loss). So it is robustness, avoiding each method's failure runs, rather than a consistent per-run gain, and a tendency only (5–7 paired runs). Part of it is a destructive point: on Valentine's Day 46 CINDER-RS sits at r = 5, with copy 0.00, every edited row a norm outlier, test collateral 0.73 and Alpaca fluency 0.88. In E2 at its selected points: copy 0.75, leak 0.22, destruction 0.19, graceful 0.48; multi-meaning words destroyed 0.09 (the lowest), exclusive words 0.65.
- **F19. CINDER's mapped-back edit is weakly read on Gemma too (offline, amendment 7; R13).** Layer-0 read strength relative to random directions: EMBER 1.7–2.0×, CINDER and CINDER-RS 1.0–1.2×, CINDER-z and CINDER-RS-z 1.8–2.3× (1.8–2.1× their parent's in every run). The re-solved edit is also more aligned with the token: median cos(c, e) 0.10–0.44 → 0.34–0.61. Its cosine to the original edit is 0.70–0.87, lower than the 0.87–0.94 reported on Llama, so on Gemma the original-space solve moves the direction more. This confirms the multilingual diagnosis on the CINDER ladder.
- **F20. A3z removes CINDER's efficacy ceiling, at a collateral cost (validation, R14).** Matched-r means: CINDER-z 0.42 at r = 1, 0.46 at r = 2, 0.52 at r = 10, 0.54 at r = 100; CINDER-RS-z 0.42, 0.43, 0.56, 0.63. CINDER stays at 0.32–0.36 from r = 1 on, CINDER-RS at 0.33–0.34, while EMBER rises from 0.31 to 0.52. Collateral at r ≥ 5 falls to 0.52–0.61 (CINDER 0.72–0.73, EMBER 0.65–0.69). Below r = 1, A3z erases less than its parent (r = 0.25: 0.15 and 0.10 vs 0.26 and 0.21; EMBER 0.05), so its curve takes EMBER's shape. The gain depends on the concept: at r = 100 A3z reaches 0.56–1.00 on Harry Potter and Ancient Rome (CINDER 0.44–0.60), but 0.15–0.49 on Valentine's Day. On Valentine's Day 43 and 46, CINDER-z stays at ≤ 0.20 at every r although its read strength there is 2.0× its parent's. So read strength is necessary, not sufficient: these are CINDER's failure runs (F4), and A3z keeps CINDER's selection.
- **F21. Under the old evaluation A3z reaches EMBER, by deleting the words (R15).** Paper-style rule (max HM(efficacy, specificity), MMLU ≥ 0.9, no collateral), test split. Efficacy: EMBER 0.46; CINDER 0.21 → CINDER-z 0.47; CINDER-RS 0.21 → CINDER-RS-z 0.47. H2: EMBER 0.60, CINDER-z 0.55, CINDER-RS-z 0.60 (CINDER 0.33, CINDER-RS 0.34). Paired ΔH2 vs EMBER is +0.06 for both (4 wins and 1 loss, n = 5; 3 and 3, n = 6). Both means rest on Ancient Rome 44 (+0.74 and +0.61), where EMBER's edit erases little, and CINDER-z's loss is its failure run, Valentine's Day 46 (−0.67). Collateral at these points is 0.67–0.72. E2 at the same points: T3 destruction EMBER 0.99 (copy 0.00, z > 3 1.00, context-MC 0.38), CINDER-z 0.97 (copy 0.16), CINDER-RS-z 0.77 (copy 0.15), CINDER 0.76, CINDER-RS 0.51, 1b 0.20. A3z wins the old evaluation the way EMBER does: by deleting the words. 1b is the exception: its efficacy does not grow with r, so the paper rule keeps it at r = 0.5–1.
- **F22. Under the new evaluation A3z does not help, and it loses CINDER's integrity advantage (R16).** Test H3v: CINDER-z 0.41, CINDER-RS-z 0.40 (EMBER 0.42, CINDER 0.42, CINDER-RS 0.46); H4 0.45 and 0.47 (0.48, 0.48, 0.53). Paired vs EMBER: +0.04 (4/1, n = 5) and +0.02 (2/3, n = 6), again from Ancient Rome 44. Paired vs its own parent: −0.01 (3 wins, 3 losses, 1 tie) and −0.06 (0 wins, 5 losses, 3 ties). The selected r stays small (median 0.5). At that point, T3 destruction and leak are 0.30 and 0.16 for CINDER-z and 0.21 and 0.27 for CINDER-RS-z (CINDER 0.21 and 0.15, CINDER-RS 0.19 and 0.22, EMBER 0.30 and 0.31). Exclusive words destroyed: 0.74 and 0.78 (CINDER 0.37, CINDER-RS 0.65, EMBER 0.74). Copy is 0.69 and 0.76. The static fingerprint is lower: z > 3 share 0.08 and 0.06 (CINDER 0.29, CINDER-RS 0.19, EMBER 0.14). Part of CINDER-z's extra destruction comes from where it is selected: in 2 of the 3 judge runs it sits one grid step above CINDER (Harry Potter 42: r = 0.5 vs 0.25; Valentine's Day 42: 1 vs 0.5). At r = 0.25 A3z behaves like CINDER: leak 0.28 and 0.34, destruction 0.06 and 0.05, and fewer exclusive words destroyed (0.18 and 0.17, against 0.28 and 0.35), because it erases less there (F20).
- **F23. At large r, efficacy beyond deletion tracks read strength, and the strongly read edits damage the context (R10, R14, R16).** At r = 100, CINDER deletes the words as completely as EMBER (T3 destruction 0.99 vs 1.00, copy 0 for both). Yet its validation efficacy is 0.36, against 0.52 for EMBER, 0.54 for CINDER-z and 0.63 for CINDER-RS-z. Noise on EMBER's tokens, a random direction, reaches 0.31 there (0.39 in the pilot). Only the strongly read edits harm the rest of the prompt. Context-MC at r = 10 and 100 is 0.34 and 0.31 for CINDER-z, 0.37 and 0.32 for CINDER-RS-z, and 0.45 and 0.38 for EMBER. CINDER stays at 0.53, CINDER-RS at 0.56–0.57, and the references at 0.53–0.56. The answer-NLL excess over the nonce is +1.2–1.5 nats for A3z (EMBER +1.1–1.2), against +0.1–0.2 for CINDER and CINDER-RS. So deletion alone does not explain the high efficacy that the old evaluation rewards (S9).

Findings F24–F29 are confirmatory: 12 new concepts × 5 seeds, settings frozen before any data (amendment 8, D17).

- **F24. Primary endpoint: no variant beats EMBER after correction (D17).** Paired test H3v (collateral rule) minus EMBER, unit = concept, on runs where both edit: CINDER +0.042 (95% CI −0.009 to +0.096; 6 wins, 5 losses; Wilcoxon p 0.21, Holm 0.41; 11 concepts, 38 pairs), CINDER-RS +0.034 (−0.043 to +0.112; 8/3; 0.41, 0.41; 37 pairs), CINDER-z +0.132 (+0.048 to +0.212; 9/2; 0.019, 0.074), CINDER-RS-z +0.084 (+0.020 to +0.153; 9/2; 0.024, 0.074). Mean test H3v over the runs where each edits: EMBER 0.50, CINDER 0.53, CINDER-RS 0.51, CINDER-z 0.58, CINDER-RS-z 0.56. The development ordering (CINDER-RS first, the A3z variants last, F18, F22) does not replicate: on new concepts the A3z variants lead, and CINDER-RS's development lead was its two failure-run rescues (S7).
- **F25. A3z's gain is efficacy at the same strength (D17).** Every variant's median selected r is 1. In the paired runs, CINDER-z and CINDER-RS-z gain +0.11 and +0.09 in efficacy over EMBER, with specificity −0.01 and collateral +0.00 to +0.01. CINDER and CINDER-RS gain +0.02 and +0.03 in efficacy. So on new concepts the stronger read (F19) helps at r ≈ 1, inside the collateral gate, which it did not on the development concepts (F22, where its extra strength appeared only at r ≥ 1, below the gate). The A3z gain varies by concept: CINDER-z is best on Gambling, Golf, Heroin and COVID-19 and worst on Uranium (0.34 vs EMBER 0.85, one EMBER run) and Republic of Ireland (0.39 vs 0.58).
- **F26. At a fixed strength, every CINDER direction beats EMBER (fixed r = 0.5, D17).** Paired H3v: CINDER +0.151 (CI +0.080 to +0.226), CINDER-RS +0.141, CINDER-z +0.086, CINDER-RS-z +0.115, each winning 10 of 11 concepts; Holm p 0.012, 0.037, 0.054 and 0.012. Efficacy at r = 0.5: EMBER 0.20, CINDER 0.30, CINDER-RS 0.28, CINDER-z 0.26, CINDER-RS-z 0.25, all at collateral 0.93. This confirms F3 and F6 on new concepts: CINDER's direction removes more per unit of perturbation. The selected-point comparison (F24) is close because selection lets EMBER go to a larger r, where its efficacy catches up.
- **F27. EMBER often edits nothing (D17).** EMBER's judge selects no feature in 19 of the 60 runs: Heroin, Nazism and Uranium at 4 of 5 seeds, Halloween at 3, Golf at 2, Baseball and COVID-19 at 1. CINDER and CINDER-z have no feature in 8 runs (Nazism at all 5), CINDER-RS and CINDER-RS-z in 5. Whenever a set exists, the collateral rule finds a point that passes the gates. Failure rate (H3v < 0.25, no edit counted as a failure): EMBER 0.37, CINDER 0.17, CINDER-RS 0.17, CINDER-z 0.18, CINDER-RS-z 0.15. When it edits, EMBER fails in 0.07 of runs, the others in 0.04–0.09. The primary endpoint pairs only runs where both edit, so it does not count this.
- **F28. The old evaluation reproduces the anchor and is won by deletion (D17).** Paper rule (max H2, MMLU gate), test split. EMBER and CINDER-z are selected at median r = 100, CINDER-RS-z at 30, CINDER and CINDER-RS at 1. H2 minus EMBER: CINDER −0.134 (1 win of 11, Holm p 0.056), CINDER-RS −0.153 (2/9, 0.084), CINDER-z +0.161 (11/0, 0.004), CINDER-RS-z +0.088 (9/2, 0.084). Efficacy: EMBER 0.47, CINDER 0.32, CINDER-RS 0.30, CINDER-z 0.68, CINDER-RS-z 0.56, with collateral 0.73, 0.86, 0.84, 0.68 and 0.69. This repeats F1 and F21: under the old evaluation CINDER loses to EMBER, and A3z wins by going to the destruction regime.
- **F29. Integrity at the selected point does not separate the methods (D17).** Judge-free copy: EMBER 0.40, CINDER 0.42, CINDER-RS 0.41, CINDER-z 0.44, CINDER-RS-z 0.41; echo 0.37–0.41; no paired difference (Wilcoxon p 0.15–0.97). T3 on seed 42 (2,444 labels; agreement 0.97 exact and 0.99 on destruction over 202 cross-agent duplicates): destruction EMBER 0.49, CINDER 0.52, CINDER-RS 0.60, CINDER-z 0.61, CINDER-RS-z 0.61, leak 0.15, 0.09, 0.08, 0.08, 0.06. Exclusive words destroyed 0.65–0.81, multi-meaning words 0.45–0.58. Paired destruction differences are not significant (6–7 concepts, p ≥ 0.19). At r ≈ 1, about half of every method's edited words become unreadable (as F9 predicts), and the collateral gate admits it (F17). CINDER's lower destruction in development (F12: 0.21 vs EMBER 0.30) does not replicate; it came from CINDER being selected at a smaller r on Harry Potter 42.

Findings F30–F35 come from the last-mile checks (amendments 9 and 10, post hoc, D18).

- **F30. Counting EMBER's empty runs, every CINDER variant beats EMBER (intention-to-treat, D18).** All 60 runs, a run without an edit scored as the unedited model (H3v 0): CINDER +0.123 (CI +0.003 to +0.249; 9/3 concepts), CINDER-RS +0.133 (10/2), CINDER-z +0.167 (10/2), CINDER-RS-z +0.178 (11/1; Holm p 0.049, the others 0.103). Leaving one concept out, the A3z contrasts stay at uncorrected p ≤ 0.049 (means +0.06 to +0.16); the CINDER and CINDER-RS contrasts range from p 0.08 to 0.70.
- **F31. The stricter the collateral gate, the more CINDER wins (D18).** Paired H3v vs EMBER at gate 0.9: CINDER +0.174, CINDER-RS +0.147, CINDER-z +0.195, CINDER-RS-z +0.102 (p 0.004, 0.004, 0.002, 0.014; all survive Holm). At gate 0.8 (primary): +0.04, +0.03, +0.13, +0.08 (F24). At gate 0.7: −0.019, −0.041, +0.121, +0.070 (p 0.70, 0.37, 0.07, 0.24). At gate 0.9 every variant sits at median r = 0.5, and efficacy is EMBER 0.15 vs CINDER 0.26, CINDER-RS 0.24, CINDER-z 0.28, CINDER-RS-z 0.20. At gate 0.7 EMBER moves to median r = 2 (mean 22) and the A3z variants to 2–5, with efficacy 0.36 (EMBER) and 0.47–0.55 (A3z), while CINDER and CINDER-RS stay at r = 1 and 0.31–0.32. This is S2's prediction: CINDER's direction advantage lives in the non-destructive regime, and it shows exactly when the evaluation allows less destruction.
- **F32. At the collateral-rule points, efficacy is close to what plain unreadability would give (D18).** Making an edit set's tokens unreadable in the efficacy questions, on the unedited model, gives efficacy 0.23–0.28 (tokens deleted) or 0.29–0.36 (nonce); 71–78% of the questions contain an edited token. At the collateral-rule points the edited models reach 0.29 (EMBER), 0.32 (CINDER), 0.31 (CINDER-RS), 0.39 (CINDER-z), 0.38 (CINDER-RS-z), at or below the deletion floor in 37%, 40%, 53%, 17% and 22% of runs. At the paper-rule points EMBER reaches 0.47 and CINDER-z 0.68, above the floor in 83% and 98% of runs. So the old rule's efficacy is not explained by unreadability alone (it fits S8/S9: the strongly read edits do more than delete), and most of the new rule's efficacy for CINDER and CINDER-RS is within what deleting the words from the question would do. Only the A3z variants clearly exceed the floor at r ≈ 1.
- **F33. Generation damage: the old rule breaks more neutral prompts, but the new rule's point is not safe either (D18).** 20 neutral prompts per concept, each using an edited multi-meaning token in an outside meaning, 1,440 blind labels (agreement 0.99 on 144 duplicates). Broken on the 9 concepts where every config exists: EMBER 0.15 and CINDER-z 0.16 at the collateral-rule point, CINDER-RS-z 0.21, CINDER 0.22, CINDER-RS 0.27; at the paper-rule point EMBER 0.24 and CINDER-z 0.34. On all concepts, EMBER 0.14 vs 0.31. Typical breaks: the key word lost ("cotton balls" answered as cotton, "hash table" as table), wrong answers (Shanghai for Wuhan, Blur for The Verve), requests for context, loops. In development, δ = 1 broke 0–1 of 20 prompts; at r ≈ 1 on new concepts, 3–5 of 20.
- **F34. The paper's coherence check does not see destruction (D18).** Alpaca coherence on seed 42 is 0.97–0.98 at every point, collateral-rule and paper-rule alike (no point below 0.9). At the paper-rule points (r = 30–100) the same edits break 24–34% of neutral prompts (F33) and destroy every edited word (F21, F28). This confirms Observation 3 on new concepts: the paper's H rises to 0.76 (EMBER) and 0.81 (CINDER-z) at those points, against 0.53 and 0.70 at the collateral-rule points.
- **F35. With coherence added, A3z's lead holds on seed 42 (H4, D18).** H4 at the collateral-rule point: EMBER 0.58, CINDER 0.57, CINDER-RS 0.57, CINDER-z 0.72, CINDER-RS-z 0.65. Paired against EMBER: CINDER-z +0.140 (7 of 7 concepts), CINDER-RS-z +0.064 (4/2), CINDER −0.015 (3/4), CINDER-RS −0.032 (3/3). Descriptive: one seed, 6–7 concepts.

Findings F36–F38 come from amendment 11 and from the existing E2 outputs (post hoc, D19).

- **F36. Embedding edits block access to the concept; they do not remove it from the weights (by construction, D19).** The model loader unties Gemma's output head ([model\_loader.py](ember/erasure/model_loader.py)), so every edit changes input rows only. A prompt that contains no edited token therefore runs exactly as in the unedited model, and word-free efficacy is 0 by construction. Verified on 9 run-variants (Golf s44, Gun s43) at both the collateral-rule and the paper-rule point, r = 100 included: accuracy on 43–51 newly written questions that avoid every edited token, and on the 4–6 existing test questions without them, equals the unedited model (efficacy 0.000 in all 18 configs). The MC frame (instruction, "Question"/"Choices", the letters, the chat template; 34 token ids) shares no token with any of the 255 run-variant edit sets. So every efficacy number in this doc, and in the paper's evaluation, measures whether the concept can still be reached through the edited tokens; through wordings that avoid them it is untouched. This holds for EMBER, CINDER and every variant alike. It sharpens F32: at r ≈ 1, most of what an edit removes is what deleting its tokens from the question removes.
- **F37. At r ≈ 1 the A3z edits already damage the surrounding prompt; CINDER and CINDER-RS do not (D19).** Context-MC at the collateral-rule point (an unrelated MMLU question right after a sentence with the edited word): CINDER-z 0.492 vs 0.533 for the same words in the unedited model (below in 9 of 11 concepts, Wilcoxon p 0.024), CINDER-RS-z 0.503 vs 0.548 (10 of 12, p 0.003); EMBER 0.525 vs 0.540 (p 0.52), CINDER 0.530 vs 0.535 (p 0.56), CINDER-RS 0.543 vs 0.549 (p 0.20). Answer-NLL excess over the nonce: CINDER-z +0.39 and CINDER-RS-z +0.37 nats, EMBER +0.29, CINDER +0.13, CINDER-RS +0.12. Paired against EMBER, context-MC is −0.027 (CI −0.057 to +0.006) for CINDER-z and −0.029 for CINDER-RS-z (p 0.10 each), +0.014 and +0.021 for CINDER and CINDER-RS (n.s.). So part of A3z's extra efficacy over its parent (F25) comes with disruption of the prompt around the edited word: the large-r pattern of F23 is already visible at the operating point. CINDER's and CINDER-RS's edits are the cleanest on this measure.
- **F38. When the words must stay readable, efficacy collapses for every method; CINDER and CINDER-RS lead, A3z's lead disappears (E3, D19).** Selection restricted to grid points where the edited words can still be copied (copy ≥ 0.9; exclusive words reach 0.94–0.98 at the selected points). Efficacy at the E3 point: EMBER 0.06, CINDER 0.12, CINDER-RS 0.14, CINDER-z 0.07, CINDER-RS-z 0.07; H3v 0.12, 0.25, 0.27, 0.16, 0.16; median r 0.1–0.25. The E3 point equals the collateral-rule point in only 16 of 255 cases. Pre-registered paired H3v vs EMBER: CINDER +0.096 (CI −0.006 to +0.214; 7/3; Holm p 0.42), CINDER-RS +0.092 (7/3; 0.42), CINDER-z +0.029 (0.75), CINDER-RS-z +0.026 (0.75), so it is not significant. Counting EMBER's empty runs (intention-to-treat, as in F30; post hoc): CINDER +0.134 (CI +0.057 to +0.221; 10/2; Holm 0.015), CINDER-RS +0.166 (+0.090 to +0.246; 11/1; Holm 0.006), CINDER-z +0.053 and CINDER-RS-z +0.063 (Holm 0.33). The A3z edits break copy at smaller r, so E3 pushes them to r = 0.05 in 24 of 52 and 26 of 55 runs. So S2 holds in direction for CINDER's own edit (about twice EMBER's efficacy at matched integrity) and is significant only when EMBER's failures to edit are counted. In absolute terms, no method removes much while keeping its words. Alpaca coherence at the E3 points (seed 42) is 0.97–0.98 for every variant; H4 there is EMBER 0.18, CINDER 0.35, CINDER-RS 0.20, CINDER-z 0.16, CINDER-RS-z 0.25, and CINDER beats EMBER in 6 of 7 concepts (+0.17; descriptive).

Findings F39–F43 come from the nonce-matching edit (amendment 13 and its addenda, post hoc; seed 42; D20).

- **F39. A learned edit can move a token far (r ≈ 3) and keep it readable, which subtraction cannot (D20).** The nonce-matching edit learns only the edited rows. The objective: behave like the token's nonce on concept sentences and definitional prompts, keep copying, keep the unedited behaviour on outside-meaning sentences. Unconstrained (NM-free, CINDER's set), the selected rows sit at median r = 3 with copy 0.99 (exclusive words 1.00), echo 0.47, context-MC 0.52 (reference 0.53) and collateral 0.86, and break 16% of neutral prompts. Subtraction at r ≈ 1 already has copy 0.38–0.44 (F29), and at copy ≥ 0.9 it must stay at r ≤ 0.25 (F38). Test efficacy is 0.18 for NM-free, 0.21 for NM-free on EMBER's set, and 0.22 with self-written concept questions added (NM-QA). That is 2–3× the copy-gated subtraction on the same runs (E3: EMBER 0.07, CINDER 0.14, CINDER-z 0.06). Paired H3v: NM-free − CINDER-z (E3) +0.20 (7 of 9 concepts, p 0.055); NM-QA − CINDER-z (E3) +0.26 (7/1, p 0.039); NM-free-EMBER − EMBER (E3) +0.28 (6/1, p 0.047), uncorrected.
- **F40. It does not reach the nonce point (D20).** The per-run nonce floor (efficacy when the edited words are replaced by nonces in the questions) averages 0.36 on these runs. The pre-registered attainment (efficacy ≥ floor − 0.05, copy ≥ 0.9, collateral ≥ 0.8) holds in 1 of 10 runs for NM-CINDER, 2 of 9 for NM-free, 2 of 8 for NM-EMBER, 3 of 7 for NM-free-EMBER and 3 of 10 for NM-QA. Almost all of these are runs whose floor is about 0 (COVID-19 0.00, Pornography 0.07–0.11). Only two runs reach a substantial floor: Baseball NM-free-EMBER (0.36 vs 0.31) and Heroin NM-QA (0.44 vs 0.49). Adding concept questions to the forget data (NM-QA) raises efficacy only from 0.18 to 0.22.
- **F41. "Unknown but intact" splits into "copyable" and "nameable", and the free edits keep only the first (D20).** Asked "What is it?" about the word, NM-free and NM-QA usually do not repeat it: echo is 0.47–0.49 (reference 0.96, nonce 0.75). In 32–47% of answers the model instead names a different invented string (string check; nonce 0.17, real word 0.00, subtraction 0.08–0.09). Blind labels count these as BLIND, so T3 destruction is 0.47–0.59, about the subtraction level (0.49–0.61; agreement 0.92 exact and 0.97 on destruction over 152 duplicates). The edited token is read as some unfamiliar word that the model reproduces inside a quote but cannot refer to. The nonce reference behaves differently: it is echoed and mostly given an invented meaning (CONFAB) or an "I don't know".
- **F42. Restricted to the concept subspace, the edit keeps everything and removes nothing (D20).** In the span of the set's concept components (rank ≤ 3), NM-CINDER and NM-EMBER keep the word (copy 0.97–0.99, echo 0.94–0.95, T3 destruction 0.01–0.04, 2–4% broken prompts) at r ≈ 0.2–0.3. But efficacy is 0.04 and 0.02, and T3 leak is 0.39 and 0.53 (reference about 0.5). In the same subspace, subtraction removes about 0.26 at r = 0.5 (F26). So the nonce objective pushes the edit away from the knowledge-removing direction. CINDER's subspace does not carry the nonce behaviour, and the subspace constraint is not what makes CINDER useful here.
- **F43. CINDER's token set adds coverage, not a better edit (D20).** On the same objective, NM-free (CINDER's set) edits 9 runs and NM-free-EMBER 7 (EMBER has no set for Halloween, Heroin, Nazism or Uranium at seed 42). Where both edit, efficacy is similar (0.18 vs 0.21; collateral 0.86 for both). This matches F27: CINDER's practical advantage is that it finds tokens to edit.
- **F44. CINDER's advantage at equal strength is its direction, not its token set (confirmatory 2×2, amendment 14, D21).** Test pass at fixed r = 0.5 on the 4 cells (38 runs, 11 concepts). Changing only the direction (EMBER's → CINDER's, same tokens) raises efficacy by +0.073 (95% CI +0.012 to +0.150; 10 of 11 concepts; p 0.032, Holm 0.097) and H3v by +0.097 (+0.029 to +0.174; 10/1; p 0.014, Holm 0.055). Changing only the tokens (EMBER's → CINDER's top-N, same direction) gives +0.016 efficacy and +0.024 H3v (p 0.16, 0.10). The direction costs a little collateral, −0.019 (p 0.054). Both direction contrasts point the same way: on EMBER's tokens 1b − ee is +0.082 efficacy (p 0.004), on CINDER's tokens 1a − cross is +0.065 (p 0.042). Against F26 (CINDER − EMBER +0.15 H3v at r = 0.5), the direction accounts for most of the gap. This replicates F6 on held-out concepts, with a smaller direction effect than in development (+0.07 vs +0.15 at r = 0.25 to +0.09 at r = 1). At r = 1 (secondary) the direction effect is gone: Δ\_dir −0.016 efficacy and −0.007 H3v (6/5 concepts, p 0.97 and 1.00), Δ\_tok +0.010 and +0.012 (n.s.); all four cells sit at efficacy 0.27–0.29. Development had +0.09 at r = 1 (F6), so that part does not replicate. The direction advantage is confined to small edits: by r = 1, where about half the edited words are destroyed (F9, F29), which direction is subtracted no longer matters.
- **F45. By coverage, CINDER is no worse at the selected point and better in every collateral-aware setting that counts all runs (post hoc; [coverage\_split.md](results/varwidth/layers/collateral/coverage_split.md)).** Each setting is scored two ways: runs where both judges accept a concept feature, and all 60 runs with a missing edit scored as H3v 0. Own selected point: both edit, CINDER +0.042 (CI −0.009 to +0.096) and CINDER-RS +0.034 (−0.043 to +0.112), so both are non-inferior at a margin of 0.05 H3v (about 10% of EMBER's mean; chosen after the results); all runs +0.123 and +0.133 (Holm 0.10). Equal strength (r = 0.5), all runs: CINDER +0.184 (Holm 0.068), CINDER-RS +0.182 (Holm 0.037). Gate 0.9, all runs: all four variants +0.14 to +0.20, Holm 0.004–0.006. Gate 0.7: both edit −0.02 and −0.04 for CINDER and CINDER-RS, all runs +0.09 and +0.10, none significant. H3v at the paper-rule point: both edit −0.06 and −0.09 (n.s.), all runs +0.05 and +0.06 (n.s.). So CINDER is never significantly worse than EMBER under a collateral-aware rule; EMBER leads only under the paper's rule, which rewards destruction.

### Speculations (to be tested)

Each item is a speculation, not a result.

- **S1. For concept-exclusive tokens, complete and graceful erasure may be incompatible for any input-embedding edit.** Their embedding is mostly the concept, so removing the concept removes the token's identity, and copy fails (F9, F10). Every method destroys 0.37–0.75 of its exclusive words at its selected point (F17); this is the open problem. An identity-preserving edit could separate "unknown" from "deleted": keep the component that lets the model copy the token, for example by removing the concept part only orthogonally to the token's own unembedding or identity direction. **Test:** such a variant on the same E1 and E2 settings. **Evidence (amendment 11):** with copy ≥ 0.9 enforced, every method keeps 0.94–0.98 of its exclusive words copyable but removes almost nothing (efficacy 0.06–0.14, F38); and since edits only block access through the edited tokens (F36), "graceful" erasure of an exclusive token would at best make it a nonce-like word. **Tested (amendment 13):** a learned identity-preserving edit keeps the words copyable at r ≈ 3 and removes 2–3× more than copy-gated subtraction (F39), but stays below the nonce floor (F40), and the model can no longer name the word (F41). So "keep it copyable" is achievable; "keep it a nameable but unknown word" is not achieved by this objective.
- **S2. CINDER's direction advantage lives in the subtraction regime, so any evaluation that forbids destruction should favour it** (F3, F6, F13). **Test (proposed E3, not launched):** integrity-gated selection (copy ≥ 0.9 and collateral ≥ 0.8) on the full r grid for EMBER, CINDER, 1b and CINDER-RS. It needs E2's judge-free tests at every grid point. The copy gate should hold on the exclusive words separately, since the pooled copy hides them (F17). **Partial support (amendment 9):** tightening the collateral gate from 0.8 to 0.9 moves every method to r = 0.5, and there every CINDER variant beats EMBER (+0.10 to +0.20, Holm-significant); at 0.7 plain CINDER's edge disappears (F31). **Tested (amendment 11, E3):** supported in direction for CINDER and CINDER-RS (efficacy 0.12–0.14 vs EMBER 0.06 at copy ≥ 0.9), significant only under intention-to-treat; A3z gains nothing (F38).
- **S3. CINDER's token-set liability comes from the concept-only label, amplified by GS widening** (F4). The label admits generic words, and GS adds more of them. CINDER-RS tested the GS part (D15): without GS the Ancient Rome sets shrink and fewer edited tokens are multi-meaning, but validation collateral at matched r is no better than CINDER's. So GS explains the set sizes and some failure runs, not the faster collateral loss. **Test:** an exclusivity filter on the edit set, for the label part.
- **S4. The paper's relearning robustness at δ = 200 may come from destruction.** Rows 40–120× the token's norm cannot be moved back by small fine-tuning steps. **Test:** LoRA relearning (§C.6) at r ≈ 0.5 vs r = 100.
- **S5. For a small model, graceful ignorance means behaving like the nonce, not saying "I don't know".** Gemma-2-2B confabulates on unknown words (nonce: 62% CONFAB, 29% IGNORANT). **Test:** a distribution-matching T3 score, the distance between a config's label distribution and the nonce's, recomputed from the existing labels.
- **S6. MC efficacy in the destruction regime partly measures unreadable questions, so it overstates knowledge removal.** **Test:** OE efficacy, or efficacy controlled by the nonce (the same questions with the word replaced by its nonce, on the unedited model, as the floor). **Tested (amendment 9), partly refuted:** the deletion floor is 0.23–0.28 and the nonce floor 0.29–0.36; at the paper-rule points EMBER and CINDER-z exceed it in 83% and 98% of runs (0.47 and 0.68), so MC efficacy there is more than unreadable questions. At the collateral-rule points CINDER and CINDER-RS sit about at the floor (F32). **Amendment 11:** efficacy can only come from edited tokens in the question (F36): questions without them are answered as before.
- **S7. Method differences below about 0.05–0.08 cannot be resolved with 3 concepts × 2–3 seeds and 50-question tests** (F2, F5, F6, F18). CINDER-RS's lead is one example: its paired means against EMBER and CINDER each come from a single run. **Tested (amendment 8):** confirmed. On 12 new concepts the development ordering did not hold; CINDER-RS's lead shrank to +0.03 and the A3z variants, last in development, came first (F24). With 11 concepts as the unit, a +0.08 to +0.13 paired difference reaches uncorrected significance but not Holm (F24); fixed-strength differences of +0.09 to +0.15 do (F26).
- **S8. EMBER's large-r efficacy may partly be global disruption, not erasure.** At r ≥ 10, EMBER is the only E1 method that lowers context-MC and raises the answer NLL (F11), and it beats Noise at the same r on the same tokens (0.52 vs 0.31 at r = 100, F3). Its rows are huge and share few directions (one feature on Harry Potter, D1). The same disruption could lower MC accuracy on concept questions without removing knowledge. A3z adds support: once CINDER's edit is strongly read, it too lowers context-MC and raises efficacy at large r, while CINDER, equally deleted, does neither (F23). **Test:** per run, relate EMBER's and A3z's large-r efficacy gain to their context-MC drop, and score efficacy on OE or nonce-controlled questions (S6). The nonce/deletion floor (F32) shows the large-r efficacy exceeds what unreadable tokens alone give, consistent with disruption beyond deletion; the context-MC relation is still to be done.
- **S9. A strongly read edit far larger than the token may raise MC efficacy beyond deletion, by steering against the concept or by derailing the prompt** (F23). Two readings fit F23. (a) *Anti-concept steering:* −δc points against the token's concept content; read strongly, it pushes the residual stream away from the concept, which is steering, not knowledge removal. (b) *In-context derailment:* a huge, strongly read row disrupts whatever question it sits in. It lowers the answer to an unrelated question in the same prompt (context-MC 0.31–0.38 at r = 100, references 0.53–0.56), and on the efficacy questions that counts as erasure (S6, S8). The context-MC drop favours (b), at least in part. CINDER's edit at r = 100 is a weakly read deletion (1.0–1.2× random), and it stops at efficacy 0.36, as does Noise at 0.31. **Test:** at r = 10–100 on the EMBER and A3z token sets, replace −δc by a random direction of the same norm and the same layer-0 read strength (for example, a random direction drawn inside layer 0's most-read subspace and rescaled). Equal efficacy and an equal context-MC drop mean derailment; lower efficacy with the same context-MC drop means concept-specific steering. Nonce-controlled efficacy (S6) separates the two as well. It needs new edits and grid points; it is proposed, not launched.

### CINDER-RS: the direction without the widths (amendment 5)

CINDER stacks three layers: Contrastive Sharpening (CS, layer 1), Global Sparsity (GS, layer 2) and the Two-Way Judge (TWJ, layer 3). F3 and F13 credit its direction; F4 blames its GS-widened token sets. **CINDER-RS** keeps layers 1 and 3 and replaces GS with EMBER's row sparsity: CS + RS-1% + TWJ, no GS. It separates CINDER's direction (layer 1) from the GS widths (layer 2). It is evaluated with E1 and E2 settings unchanged: the same grid, question set, gates, selection, test pass and Alpaca, and the same integrity tests, sweep and blind T3 labels. 88–100% of its edited tokens are in the frozen pool (Valentine's Day 46: 68 of 77), so collateral covers it. The user proposed it after E1 and E2 were seen, so the comparison is exploratory.

**Result (F18, D15).** CINDER-RS is the only variant with an edit in all 8 runs, and it has the best mean test H4 of the Alpaca-scored primaries (0.53; 1b 0.52, EMBER 0.48, CINDER 0.48). Paired on shared runs, the lead comes from avoiding each method's failure run, not from a consistent per-run gain, so it is a tendency within noise. At small r it keeps CINDER's direction gain, but its collateral at fixed r is no better than CINDER's, so dropping GS does not fix the collateral curve. At its selected points it destroys 0.19 of the words (EMBER 0.30, CINDER 0.21), but 0.65 of its exclusive words.

### A3z: CINDER's selection, edited in the original space (amendment 7)

The multilingual work found on Llama that ML-CINDER's edits, mapped back from the whitened space, lie in directions that layer 0 barely reads: 1.1–1.7× a random direction, against 2.6–7.6× for ML-EMBER. Erasure stalls near 0.1 even at δ = 200. Their fix, A3z, keeps CINDER's selection and re-solves the dictionary in the raw embedding space. **CINDER-z** and **CINDER-RS-z** apply it on Gemma to CINDER and CINDER-RS. They keep the whitened-fit sparse Y (with GS's masks), the HJ-selected F\_C and the edited tokens, and replace the mapped-back Z by Z\_o = A Yᵀ(YYᵀ + λI)⁻¹, solved on the raw embeddings (D16). They go through the new evaluation (E1 and E2, settings unchanged) and through an **old-evaluation arm**: the paper-style rule on the same grids, with a test pass and the E2 tests at its point, for EMBER, CINDER, CINDER-RS, both A3z variants and 1b. The user proposed A3z after the multilingual result, so the comparison is exploratory.

**Result (F19–F23, D16).**

- **The diagnosis holds on Gemma.** CINDER's mapped-back edit is read at 1.0–1.2× a random direction, EMBER's at 1.7–2.0×, A3z's at 1.8–2.3× (F19).
- **A3z removes CINDER's efficacy ceiling.** Validation efficacy keeps rising with r (0.54 and 0.63 at r = 100, CINDER 0.36), and collateral falls further (F20). Under the paper's evaluation it reaches EMBER's level: test efficacy 0.47 for both variants against 0.46 for EMBER, from 0.21 for CINDER and CINDER-RS (F21).
- **It gets there the way EMBER does.** At the paper-rule points, CINDER-z destroys 0.97 of the words and CINDER-RS-z 0.77; EMBER destroys 0.99 (F21).
- **Under the new evaluation it does not help.** Test H3v is 0.41 and 0.40 (EMBER and CINDER 0.42, CINDER-RS 0.46), and each variant ties or loses against its own parent. At the selected point it destroys exclusive words as much as EMBER does (0.74 and 0.78, against EMBER 0.74 and CINDER 0.37), so it gives up CINDER's integrity advantage (F22).
- **Deletion is not the whole story.** CINDER at r = 100 deletes the words too (destruction 0.99), but its efficacy stays at 0.36. The strongly read edits reach 0.52–0.63 and also damage the surrounding prompt (F23, S9).

"Whiten to select, edit in the original space" holds for erasure strength under the paper's rule. Under an evaluation that charges for destruction it buys nothing: the extra strength appears at r ≥ 1, where mean validation collateral is already below the 0.8 gate (0.77 and 0.75 at r = 1), so the collateral rule does not reach it.

### Confirmatory expansion (amendment 8)

All development results above were within noise. So the 5 main variants (EMBER, CINDER, CINDER-RS, CINDER-z,
CINDER-RS-z) were re-tested, with settings frozen, on the 12 anchor concepts never used for development, seeds 42–46
(60 runs). New blind question sets were written for them (3,501 items on 1,026 multi-meaning tokens, frozen before
any edited model was scored). The primary endpoint is paired test H3v against EMBER under the collateral rule, with the
concept as the unit (11 concepts have a pair), a Wilcoxon test with Holm correction over the 4 contrasts, and a
cluster-bootstrap CI. Details in D17, numbers in [confirm.md](results/varwidth/layers/collateral/confirm.md).

**What it shows.**
- **Primary: no CINDER variant beats EMBER after correction.** CINDER +0.04 (CI −0.01 to +0.10, Holm p 0.41),
  CINDER-RS +0.03 (−0.04 to +0.11, 0.41). The two A3z variants come closest: CINDER-z +0.13 (+0.05 to +0.21) and
  CINDER-RS-z +0.08 (+0.02 to +0.15), each winning 9 of 11 concepts, uncorrected p 0.02, Holm p 0.07. Their gain is
  efficacy at the same strength (median selected r = 1 for every variant), with no collateral cost (F24, F25).
- **At a fixed strength every CINDER direction beats EMBER.** At r = 0.5, +0.09 to +0.15 in 10 of 11 concepts
  (Holm p 0.01–0.05). This confirms F3 on new concepts: CINDER's directions remove more per unit of perturbation, and
  selection closes the gap by letting EMBER pick a larger r (F26).
- **EMBER often edits nothing.** Its judge selects no feature in 19 of 60 runs (CINDER 8, CINDER-RS 5). Counting
  "no edit" as a failure, EMBER fails 37% of runs and the CINDER variants 15–18%. This is the largest practical
  difference in the data, and the primary endpoint does not see it, because it pairs only runs where both edit (F27).
- **The old evaluation reproduces the anchor, and favours A3z.** Under the paper rule, EMBER beats CINDER and
  CINDER-RS (1 and 2 wins of 11 for them), and CINDER-z beats EMBER in all 11 concepts (Holm p 0.004). Both do it at
  r = 100, at collateral 0.68–0.73 (F28).
- **Integrity does not separate the methods.** At the selected points, judge-free copy is 0.40–0.44 for every variant
  and T3 destruction 0.49–0.61; no paired difference is significant. About half of the edited words become
  unreadable at r ≈ 1, for every method, and 0.65–0.81 of the exclusive ones. The development hint that CINDER
  destroys fewer words (F12, F17) does not replicate (F29).

**Last-mile checks** (amendments 9 and 10, post hoc; F30–F35, D18):
- **Counting runs where a method edits nothing** as no erasure, every CINDER variant beats EMBER (+0.12 to +0.18), and
  CINDER-RS-z survives Holm (F30).
- **Tighten the collateral gate to 0.9 and every CINDER variant beats EMBER** (+0.10 to +0.20, all survive Holm);
  loosen it to 0.7 and plain CINDER loses its edge (F31). CINDER's advantage appears exactly when less destruction is
  allowed.
- **Unreadability explains much of the efficacy.** Deleting the edited words from the questions gives the unedited
  model efficacy 0.23–0.28, about what CINDER and CINDER-RS reach at their selected points; only A3z clearly exceeds it.
  The old rule's efficacy (0.47–0.68) exceeds it too, so strongly read edits do more than delete (F32).
- **Free text:** the paper-rule points break 24–34% of neutral prompts, the collateral-rule points 15–27% (F33).
  Alpaca coherence stays at 0.97–0.98 everywhere, so the paper's coherence check does not see this (F34).
- **H4 on seed 42:** CINDER-z +0.14 over EMBER, 7 of 7 concepts (F35).

**Amendment 11** (post hoc; F36–F38, D19):
- **Edits block access; they do not remove knowledge.** Gemma's output head is untied, so a question without an edited
  token is answered exactly as before: word-free efficacy is 0 by construction, verified on 9 run-variants (F36).
- **A3z's extra efficacy comes with damage to the surrounding prompt** at r ≈ 1 (context-MC below the real word in
  9–10 of 11–12 concepts); CINDER and CINDER-RS show none (F37).
- **Forbid destruction (copy ≥ 0.9) and efficacy collapses to 0.06–0.14.** CINDER and CINDER-RS have about twice
  EMBER's; significant under intention-to-treat (Holm 0.006–0.015), not on the paired endpoint. A3z's lead disappears (F38).

**Status:** done. Pre-registered (amendment 8); the deviations are listed in D17.

### Where we stand

- **Done:** the premise check, the pipeline pilot and the frozen question set (D5). **E1:** 1,035 grid points on 8 concept-seeds with collateral at every point, the pooled shape choice (α = 1; best shape power m = 2 on CINDER's tokens), selection, the test split and Alpaca for the primaries, with test collateral renormalized after the erratum of amendment 6 ([poc.md](results/varwidth/layers/collateral/poc.md), D4, D9). **E2:** the pilot and two T2 fixes (amendments 4b and 4c), judge-free tests on 8 concept-seeds, blind T3 labels on 3 of them (3,641 before amendment 7, 5,883 now, with 0.97 agreement on 473 cross-agent duplicates), and the split by token class ([integrity/report.md](results/varwidth/layers/collateral/integrity/report.md), D14, R11). **CINDER-RS** (amendment 5) through E1 and E2 with unchanged settings (D15). **A3z** (amendment 7): the offline read-strength check, CINDER-z and CINDER-RS-z through E1 (with Alpaca) and E2, and the paper-rule arm with a test pass and E2 at its point for 6 variants; 2,242 new T3 labels in all (D16). Gemini spend: 23.5 of 25 NIS, shared with the multilingual work.
- **Confirmatory expansion** (amendment 8): 12 new concepts × 5 seeds, new frozen question sets, the 5 main variants under the collateral rule, fixed r = 0.5 and the paper rule, judge-free E2 on all 60 runs and 2,444 blind T3 labels on seed 42 (F24–F29, D17, [confirm.md](results/varwidth/layers/collateral/confirm.md)). No Gemini.
- **Last-mile checks** (amendments 9 and 10): intention-to-treat and leave-one-out, gate sensitivity (0.7 and 0.9 on all 60 runs), the unreadability floor, 1,440 judged neutral-prompt generations, and Alpaca coherence on seed 42 (66 points, 8.3 NIS; plus 48 E3 points, 5.7 NIS, amendment 12; Gemini total 37.5 NIS) (F30–F35, D18, [last\_mile.md](results/varwidth/layers/collateral/last_mile.md)).
- **Amendment 11:** word-free efficacy (stopped once shown 0 by construction; 9 run-variants), the MC-frame check, context damage from the E2 outputs, and E3 (copy test at 1,530 grid points and copy-gated selection with test passes on all 60 runs) (F36–F38, D19).
- **Amendment 13:** the nonce-matching edit (NM-CINDER, NM-free, NM-EMBER, NM-free-EMBER, NM-QA) on seed 42 of the 12 concepts, with judge-free E2, 2,219 blind T3 labels and 880 generation labels (F39–F43, D20, [nonce\_edit.md](results/varwidth/layers/collateral/nonce_edit.md)). No Gemini.
- **Not started:** the matched-read-strength control (S9), the powered A3z contrast, a nonce-matching objective that also keeps the word nameable (F41), and relearning.

### What to keep in mind

- **Efficacy fell for every method.** EMBER's validation efficacy was 0.29–1.00 under the H2 rule and is 0.00–0.42 under the collateral rule (R2). The efficacy questions contain the same tokens, so part of the old efficacy was destruction.
- **MC collateral is lenient.** Options and context can rescue a destroyed token: in the pilot, Sherlock and Trump stay at 1.0 in MC while generation fails. So collateral is a conservative measure.
- **The questions are written by Claude agents.** Their blind spots are correlated. The objective checks and a human spot check are the counterweights.
- **3 concepts × 2–3 seeds gave direction, not significance.** On the 12 new concepts the development ordering of the variants did not hold (F24). Quote the confirmatory numbers (F24–F29), and treat F1–F23 as development findings.
- **The primary endpoint ignores runs where a method edits nothing.** EMBER edits nothing in 19 of 60 runs (F27). Read the paired contrasts together with the edit and failure rates.
- **The two evaluations rank A3z differently.** Under the paper's rule it ties EMBER; under the collateral rule it is no better than CINDER. Which one to quote depends on whether destruction counts as erasure (F21, F22). The paper-rule arm scores H2 without Alpaca, so it is not the paper's full H.

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

**Per-token accuracy.** On validation each kept item is answered correctly by the unedited model, so the baseline accuracy is 1 (on test it is not; see the erratum below). EMBER's MC normalization (§C.1):

```latex
\mathrm{mc\_norm}(a, b) = \mathrm{clip}\!\left(\frac{a - 0.25}{b - 0.25},\; 0,\; 1\right)
```

**Collateral (primary).** M\_v is the set of multi-meaning tokens of the frozen set that variant v edits:

```latex
\mathrm{coll}(v) = \frac{1}{|M_v|} \sum_{t \in M_v} \mathrm{mc\_norm}(\mathrm{acc}_t,\, 1), \qquad \mathrm{coll}(v) = 1 \ \text{ if } M_v = \emptyset
```

Every token counts equally. With 2 items per token and split, acc\_t ∈ {0, 0.5, 1}, so a token scores 0, 1/3 or 1. **Secondary:** `coll_all`, the same mean over the whole union pool.

**Test baseline (erratum, amendment 6).** The base filter checked every item under the option order of the validation split, because the MC evaluator seeds the option shuffle with an id that includes the split. Test items are shuffled differently, and the unedited model scores 0.981 (Harry Potter), 0.964 (Ancient Rome) and 0.948 (Valentine's Day) on the union pool's test items. Validation, and therefore selection, are unaffected. Every test collateral in this doc, and the test H3v, H4 and H′ built on it, uses mc\_norm(acc\_t, acc\_t of the unedited model on test), the paper's normalization, leaving out tokens the unedited model scores ≤ 0.25 on. This raises each edited variant's mean test collateral by 0.02–0.05 and leaves the H3v and H4 rankings unchanged; poc.md keeps the raw values next to the corrected ones.

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

**Frozen set** (`data/collateral_questions.json`, PREREG amendment 3). Every selector chose v2. Items the selector listed as clear violations were dropped mechanically (Harry Potter 2, Ancient Rome 18, Valentine's Day 3), then tokens with fewer than 2 usable items.

| Concept | Version chosen | Multi tokens kept | Exclusive (v2) | Tokens dropped | Items, val / test | Unedited correct (v2 draft) |
| --- | --- | --- | --- | --- | --- | --- |
| Harry Potter | v2 | 54 | 15 | 4 | 103 / 86 | 0.91 |
| Ancient Rome | v2 | 111 | 13 | 17 | 200 / 161 | 0.93 |
| Valentine's Day | v2 | 90 | 9 | 5 | 162 / 142 | 0.90 |

Kept items are all answered correctly by the unedited model under the validation option order, by construction. Test items are shuffled differently, and the unedited model misses a few of them (amendment 6, D4). File hash (sha256[:16]): `57b21bb9dfcecfc2`. Human spot check (20 random items per concept): [TBD, optional].

The selectors' scores (neutrality / essential / certainty / coverage / answerability, 1–5) were v0 2/3/3–4/3–5/3–4 and v2 5/4/4–5/4/4–5 across concepts. The revisions bought neutrality and token essentiality at the cost of 4–9 tokens of coverage per concept.

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

**Primary variants** (scored on Alpaca): EMBER, CINDER, EMBER + TJ∨WJ+share, 1a, 1b, and the best edit shape; CINDER-RS was added later (amendment 5, D15).

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
| Gemini: v1r word lists, Alpaca at the chosen point (6 primaries × 6 runs), pilot Alpaca | ≈ 4.7 NIS for the stage (actual: 5.2 NIS, CINDER-RS included; total spend then 21.6 of 25 NIS) |
| Gemini: Alpaca for CINDER-z and CINDER-RS-z at the chosen point (amendment 7; none at the paper-rule points) | ≈ 1.9 NIS; total spend 23.5 of 25 NIS ([gemini\_usage.jsonl](results/gemini_usage.jsonl)) |
| GPU | about 100 grid points per concept and seed, about 1.5 minutes each |
| Claude agents | about 18 calls for the question set |

**After this stage**, with the team: expand to the other erasure concepts of the anchor, and later relearning (§C.6) or LoRA healing of the edited model. A smaller r may be easier to relearn, which §C.6 would show.

### D9. Results

**Sources.** Per-run `result.json` (test, at the selected point) and `grid.json` (validation, every grid point) in `results/varwidth/layers/collateral/<concept>/s<seed>/<variant>/`, summarized in [poc.md](results/varwidth/layers/collateral/poc.md): the per-run test table, means, 2×2, regime shift, gate sensitivity, pooled shape choice and Pareto PNGs. Per-concept detail stays in poc.md. Values that poc.md does not print (specificity, coherence, H, H′, median r\*, the fixed-r 2×2 effects, gate-sensitivity r\*, matched-r means, CINDER-RS's regime shift and the paired differences) are computed from the same files with the rules of [collateral\_report.py](experiments/varwidth/collateral_report.py); H′ uses the D4 formula. Test collateral, H3v, H4 and H′ use the renormalized collateral (amendment 6, D4).

**Means and run sets.** Means are over the concept-seeds where a variant edits, so n differs per row and rows are not on identical runs. EMBER selects no feature at Ancient Rome 42 and Valentine's Day 43, CINDER none at Ancient Rome 43, CINDER-RS edits in all 8, and the 2×2 is complete in 5 runs (Harry Potter 42, 43; Ancient Rome 44; Valentine's Day 42, 46).

**R1. Main table** (test split, at each variant's selected point; mean over n runs; collateral renormalized):

| Variant | n runs | r\*, median (range) | Efficacy | Specificity | Collateral | Coherence | H | H4 | H′ | H3v |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | 6 | 0.5 (0.05–2) | 0.22 | 1.00 | 0.93 | 0.99 | 0.43 | 0.48 | 0.43 | 0.42 |
| CINDER | 7 | 0.5 (0.25–5) | 0.21 | 1.00 | 0.87 | 0.97 | 0.42 | 0.48 | 0.42 | 0.42 |
| CINDER-RS (CS + RS-1% + TWJ) | 8 | 0.5 (0.25–5) | 0.24 | 1.00 | 0.90 | 0.97 | 0.47 | 0.53 | 0.47 | 0.46 |
| CINDER-z (A3z, amendment 7) | 7 | 0.5 (0.5–1) | 0.23 | 0.99 | 0.91 | 0.98 | 0.41 | 0.45 | 0.41 | 0.41 |
| CINDER-RS-z (A3z, amendment 7) | 8 | 0.5 (0.05–1) | 0.20 | 1.00 | 0.90 | 0.98 | 0.41 | 0.47 | 0.41 | 0.40 |
| +CS+GS | 0 | — | — | — | — | — | — | — | — | — |
| EMBER + TJ∨WJ+share | 1 | 0.5 | 0.00 | 0.98 | 0.93 | 0.98 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1a | 7 | 0.5 (0.25–5) | 0.17 | 1.00 | 0.88 | 0.97 | 0.34 | 0.39 | 0.34 | 0.34 |
| 1b | 5 | 0.5 (0.5–1) | 0.23 | 0.99 | 0.93 | 0.99 | 0.46 | 0.52 | 0.46 | 0.46 |
| Best edit shape (power m = 2, CINDER's tokens) | 7 | 1 (0.25–2) | 0.19 | 0.99 | 0.86 | 0.97 | 0.38 | 0.43 | 0.38 | 0.37 |
| Mean (EMBER's / CINDER's tokens) | 6 / 7 | no point passes | 0.00 | 1.00 | 1.00 / 1.00 (unedited) | — | — | — | — | 0.00 |
| Noise (EMBER's tokens) | 6 | 1 (0.05–2) | 0.22 | 0.99 | 0.91 | — | — | — | — | 0.40 |
| Noise (CINDER's tokens) | 7 | 1 (0.5–5) | 0.18 | 0.98 | 0.89 | — | — | — | — | 0.34 |

- **+CS+GS has no run of its own.** In the 5 runs where it selects features it has CINDER's F\_C and fit, so its edit is identical to CINDER's and the code runs it once. It selects no feature at Ancient Rome 43 and Valentine's Day 43 and 46. Likewise, EMBER + TJ∨WJ+share equals EMBER wherever EMBER edits, so its only separate run is Ancient Rome 42.
- **CINDER-RS is the only variant with an edit in all 8 runs.** Its Valentine's Day 46 point is r = 5 (test efficacy 0.44, collateral 0.73); Valentine's Day 43 is r = 1 with efficacy 0.07 (H3v 0.19, as for CINDER).
- **Alpaca flags** (fluency < 0.9; flagged, not reselected): 1a at Ancient Rome 44 (0.89), power m = 2 at Valentine's Day 43 (0.89) and 46 (0.88), CINDER-RS at Valentine's Day 46 (0.88).
- **The A3z rows** are detailed in D16. No A3z run is flagged for Alpaca (fluency 0.90–1.00).
- **Test collateral is renormalized** (amendment 6, D4). Mean never passes the gate, so its rows score the unedited model and give 1.00 by construction (raw 0.94 / 0.96).

**R2. Regime shift (Hyp. 1).** Selected r on validation under the paper-style rule (H2 = HM(efficacy, specificity), no collateral) and under the collateral rule, with validation efficacy and collateral at both points:

| Concept, seed | EMBER r, H2 → coll. rule | EMBER efficacy; collateral | CINDER r, H2 → coll. rule | CINDER efficacy; collateral | CINDER-RS r, H2 → coll. rule | CINDER-RS efficacy; collateral |
| --- | --- | --- | --- | --- | --- | --- |
| Harry Potter 42 | 30 → 2 | 0.49 → 0.42; 0.58 → 0.81 | 10 → 0.25 | 0.56 → 0.49; 0.64 → 1.00 | 10 → 0.5 | 0.56 → 0.49; 0.65 → 0.88 |
| Harry Potter 43 | 30 → 1 | 0.56 → 0.42; 0.55 → 0.88 | 30 → 0.25 | 0.60 → 0.53; 0.59 → 1.00 | 10 → 0.5 | 0.56 → 0.53; 0.70 → 0.90 |
| Ancient Rome 42 | no feature; EMBER + TJ∨WJ+share 100 → 0.5 | 0.95 → 0.22; 0.21 → 0.82 | 0.5 → 0.5 | 0.55; 0.84 (same point) | 0.5 → 0.25 | 0.36 → 0.15; 0.76 → 0.92 |
| Ancient Rome 43 | 30 → 0.5 | 1.00 → 0.15; 0.61 → 0.91 | no feature | — | 0.5 → 0.25 | 0.51 → 0.15; 0.78 → 0.89 |
| Ancient Rome 44 | 10 → 0.05 | 0.29 → 0.00; 0.78 → 1.00 | 100 → 0.5 | 0.51 → 0.47; 0.74 → 0.83 | 0.5 → 0.25 | 0.36 → 0.15; 0.75 → 0.92 |
| Valentine's Day 42 | 100 → 0.5 | 0.44 → 0.24; 0.65 → 0.94 | 0.5 → 0.5 | 0.24; 0.96 (same point) | 0.5 → 0.5 | 0.24; 0.95 (same point) |
| Valentine's Day 43 | no feature | — | 5 → 5 | 0.20; 0.81 (same point) | 0.5 → 1 | 0.15 → 0.15; 0.86 → 0.88 |
| Valentine's Day 46 | 100 → 0.5 | 0.34 → 0.29; 0.75 → 0.97 | 5 → 5 | 0.20; 0.80 (same point) | 2 → 5 | 0.39 → 0.39; 0.81 → 0.83 |

CINDER-RS reaches the destruction regime under the H2 rule only on Harry Potter. On Ancient Rome the H2 rule already picks r = 0.5, where validation collateral is 0.75–0.78, and the gate pushes it to 0.25. In Valentine's Day 43 and 46 the collateral rule picks a larger r than H2 does: efficacy ties, and validation collateral is slightly higher at the larger r (0.88 vs 0.86, 0.83 vs 0.81), within the noise of 2 items per token. The r grid tops out at 100. For reference, the anchor (δ grid, H with Alpaca) picked Harry Potter EMBER at δ = 200 (r ≈ 120) and CINDER at δ = 2 (r ≈ 0.4).

**R3. 2×2 effects (Hyp. 3).** Each cell is Δ\_dir / Δ\_tok / Δ\_×, the mean over the 5 runs with a complete 2×2:

| Metric | Selected points (test) | r = 0.25 (validation) | r = 0.5 (validation) | r = 1 (validation) |
| --- | --- | --- | --- | --- |
| Efficacy | +0.02 / −0.05 / +0.08 | +0.15 / −0.01 / +0.00 | +0.12 / −0.02 / +0.00 | +0.09 / −0.03 / +0.04 |
| Collateral | −0.01 / −0.02 / −0.04 | −0.01 / 0.00 / +0.01 | −0.05 / −0.01 / +0.01 | −0.01 / −0.01 / −0.02 |
| H3v | +0.06 / −0.08 / +0.09 | +0.27 / −0.01 / +0.02 | +0.16 / −0.04 / −0.01 | +0.11 / −0.05 / +0.08 |
| H4 | — | — | — | — |

H4 needs Alpaca, which ee and cross do not have. Test collateral and H3v are renormalized (D4). At the selected points the per-run Δ\_dir on H3v ranges from −0.06 to +0.34 (poc.md); Valentine's Day 46 carries most of Δ\_tok (−0.50), because 1a and cross select r = 5 and 2 there and erase nothing on test.

µ-ranking vs raw-Y ranking overlap of the top-N sets: 0.89–1.00 (mean 0.94) over the 7 runs with a 1a cell.

**R4. Edit shapes (Hyp. 4)**, on the EMBER and CINDER sets (test at the selected point, collateral renormalized; validation H3v is the pooled score of the shape choice):

| Shape | Token set | Parameter | n runs | r\*, median | Efficacy | Collateral | H3v, test | H3v, validation | H4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e − δc | EMBER | — | 6 | 0.5 | 0.22 | 0.93 | 0.42 | 0.45 | 0.48 |
| Bounded projection | EMBER | α = 1 | 6 | 0.59 (none at AR 43) | 0.14 | 0.96 | 0.30 | 0.22 | — |
| Exclusivity-weighted | EMBER | — | 6 | 0.75 | 0.27 | 0.91 | 0.50 | 0.55 | — |
| Power | EMBER | m = 1.5 | 6 | 1 | 0.25 | 0.87 | 0.47 | 0.54 | — |
| Power | EMBER | m = 2 | 6 | 0.75 | 0.24 | 0.91 | 0.47 | 0.54 | — |
| e − δc | CINDER | — | 7 | 0.5 | 0.21 | 0.87 | 0.42 | 0.61 | 0.48 |
| Bounded projection | CINDER | α = 1 | 7 | 0.36 | 0.21 | 0.95 | 0.39 | 0.45 | — |
| Exclusivity-weighted | CINDER | — | 7 | 0.5 | 0.25 | 0.90 | 0.47 | 0.57 | — |
| Power | CINDER | m = 1.5 | 7 | 0.5 | 0.22 | 0.87 | 0.42 | 0.58 | — |
| Power | CINDER | m = 2 | 7 | 1 | 0.19 | 0.86 | 0.37 | 0.58 | 0.43 |

- **α = 1 won for both sets**, and validation H3v rose monotonically with α (EMBER's tokens 0.03, 0.08, 0.15, 0.22; CINDER's 0.11, 0.27, 0.40, 0.45). The bound itself costs efficacy. Projection keeps the highest collateral (0.95–0.96); on EMBER's tokens it also has the lowest efficacy (0.14).
- **The validation ranking does not survive the test split** (F5): power m = 2 on CINDER's tokens is the best shape on validation (0.58) and the worst of CINDER's rows on test (0.37). Plain CINDER drops from 0.61 to 0.42.

**R5. Pareto curves.** Efficacy vs collateral on validation, one curve per variant over the r grid, per concept-seed: [collateral/pareto/](results/varwidth/layers/collateral/pareto/) (`<concept>_s<seed>.png`, also embedded in poc.md). Mean over concept-seeds at matched r (efficacy / collateral, validation):

| Variant (n runs) | r = 0.1 | 0.25 | 0.5 | 1 | 2 | 5 | 10 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER (6) | 0.01 / 0.99 | 0.05 / 0.98 | 0.20 / 0.96 | 0.31 / 0.85 | 0.38 / 0.73 | 0.42 / 0.69 | 0.44 / 0.67 | 0.52 / 0.65 |
| CINDER (7) | 0.08 / 0.98 | 0.26 / 0.93 | 0.33 / 0.88 | 0.35 / 0.82 | 0.32 / 0.76 | 0.35 / 0.73 | 0.34 / 0.72 | 0.36 / 0.73 |
| CINDER-RS (8) | 0.05 / 0.98 | 0.21 / 0.94 | 0.37 / 0.85 | 0.34 / 0.79 | 0.33 / 0.75 | 0.33 / 0.72 | 0.33 / 0.69 | 0.34 / 0.72 |
| 1b (5) | 0.04 / 0.99 | 0.20 / 0.97 | 0.33 / 0.92 | 0.31 / 0.86 | 0.30 / 0.78 | 0.27 / 0.72 | 0.25 / 0.71 | 0.27 / 0.69 |
| Noise, EMBER's tokens (6) | 0.01 / 0.96 | 0.02 / 0.95 | 0.12 / 0.93 | 0.22 / 0.87 | 0.28 / 0.76 | 0.30 / 0.70 | 0.29 / 0.69 | 0.31 / 0.68 |

Mean's single point (r ≈ 0.85–0.95) gives efficacy 0.31–0.32 at collateral 0.71. CINDER saturates at about 0.35 from r = 0.5 on; EMBER keeps rising to r = 100 while its collateral falls to 0.65. CINDER-RS peaks at r = 0.5 (0.37, the highest of any variant there) and then stays at 0.33–0.34; its collateral at r = 0.5 is below CINDER's and 1b's (0.85 vs 0.88 and 0.92). On the 7 runs it shares with CINDER, the two curves stay within 0.05 in efficacy and 0.03 in collateral at every r of the grid.

**R6. Gate sensitivity.** Median selected r and validation H3v of the selected point (mean over runs, 0 when no point passes). H4 needs Alpaca, which exists only at the gate-0.8 point:

| Variant | r\* / H3v, gate 0.7 | r\* / H3v, gate 0.8 | r\* / H3v, gate 0.9 |
| --- | --- | --- | --- |
| EMBER | 2 / 0.61 | 0.5 / 0.45 | 0.5 / 0.42 |
| CINDER | 0.5 / 0.61 | 0.5 / 0.61 | 0.25 / 0.41 |
| CINDER-RS | 0.5 / 0.60 | 0.5 / 0.49 | 0.25 / 0.38 |
| EMBER + TJ∨WJ+share (1 run) | 1 / 0.69 | 0.5 / 0.44 | 0.1 / 0.10 |
| 1a | 2 / 0.57 | 0.5 / 0.51 | 0.25 / 0.34 |
| 1b | 0.5 / 0.59 | 0.5 / 0.59 | 0.5 / 0.54 |
| Power m = 2 (CINDER's tokens) | 1 / 0.58 | 1 / 0.58 | 0.1 / 0.41 |
| Mean (EMBER's / CINDER's tokens) | passes in 4 of 6 / 5 of 7 runs; 0.32 / 0.30 | none / 0.00 | none / 0.00 |

At gate 0.7, EMBER returns to the destruction regime in 2 of 6 runs (r\* = 10 and 100); CINDER-RS stays at r\* ≤ 1 in 7 of 8 (the eighth, Valentine's Day 46, is r = 5 at every gate up to 0.8). 1b is the most gate-robust variant. All variants are in poc.md.

**R7. Conclusions (E1).**

- **Hyp. 1, regime shift: supported** on validation (F1, R2).
- **Hyp. 2, the comparison changes: partly.** CINDER's direction wins at matched small r (F3), but the bundles tie at their selected points (F2).
- **Hyp. 3, the 2×2 separates the effects: partly.** Clean at fixed r, inconclusive at the selected points (F6, R3).
- **Hyp. 4, shaped edits trade better: not supported at this sample size** (F2, F5, R4).
- **H vs H4 ranking.** The order is the same under both: CINDER-RS > 1b > EMBER ≈ CINDER > power m = 2 > 1a (H 0.47, 0.46, 0.43, 0.42, 0.38, 0.34; H4 0.53, 0.52, 0.48, 0.48, 0.43, 0.39). After the gate, mean test collateral is 0.86–0.93 for every primary, so the fourth axis lifts all scores together and reorders nothing. Collateral acts through the selection, not through the score.
- **CINDER-RS (amendment 5, exploratory): the best mean H4 and the only variant with an edit in every run, within noise** (F18). Its lead over EMBER and CINDER comes from one run each, where the other method's selection failed.
- **Test-collateral erratum (amendment 6).** Renormalizing raised test collateral by 0.02–0.05 per variant and changed no ranking (D4).
- **For CINDER\_improvements.md.** CINDER's claim should move from "better erasure at the selected point" to "a more efficient direction in the non-destructive regime, held back by its token set". The combined reading with E2 is at the end of R12.

### D10. Known limits

- **Context blindness.** An input-embedding edit cannot separate Prince Harry from Harry Potter. There is a ceiling on efficacy at a given collateral, for every method in this family.
- **Efficacy questions share the tokens.** Lower efficacy under the new rule partly measures less destruction, not worse features.
- **MC leniency.** Recognition survives better than generation, as the paper finds for MC vs OE (§4.1). MC collateral underestimates generation damage (Sherlock and Trump in the pilot).
- **Correlated question writers.** All agents are Claude. The token-id check, the base filter, the exclusivity flag and a human spot check counter this only partly.
- **Exclusive vs multi is a judgment.** A token wrongly called exclusive hides damage; one wrongly called multi forces poor items. The selector's rubric and the exclusivity flag check this.
- **Granularity.** 2 items per token and split give per-token scores in {0, 1/3, 1}, and validation qa has 50 questions (one question ≈ 0.035 qa\_frac on Harry Potter).
- **Small development set.** 3 concepts × 2–3 seeds give direction only. The seeds 44 and 46 were picked for a complete 2×2, from feature counts only.
- **Deviations from the paper.** Top-20 candidate features instead of τ = 2; no Alpaca in the grid; collateral is new.
- **Collateral covers only multi-meaning tokens.** Concept-exclusive tokens have no items, so E1 cannot see their destruction. E2 covers them (R11).
- **Test baseline below 1.** The base filter ran under the validation option order, so the unedited model misses a few test items. Resolved by amendment 6: test collateral is renormalized by the unedited test accuracy (D4).
- **Unequal run sets.** Variant means cover 5–7 runs, not always the same ones (D9). Only the 2×2 effects are computed on matched runs.
- **T3 on 3 runs.** Agent labels come from one seed per concept; the other integrity columns cover 5–7 runs.
- **A3z is post hoc, and its offline check has no script in the repo.** It was defined after the multilingual result and after E1 and E2 were seen (amendment 7). The read-strength numbers come from `diag/a3z_offline.json` alone; the code that wrote it is not committed, so its exact definition is taken from amendment 7 (layer-0 attention read strength relative to random directions).
- **The paper-rule arm is not the paper's full protocol.** It selects by H2 = HM(efficacy, specificity) on the r grid, as the anchor's cheaper rule did, and it has no Alpaca (budget), so it reports H2, not H.
- **BLIND includes fragments and look-alikes.** Under the judge brief's clarifications, an answer about a piece of the word or about a similar real word counts as BLIND. Destruction therefore includes multi-token words that lose a piece.

### D11. E2: why collateral is not enough

**What the collateral term misses.**

| Edited token kind | Share of edited tokens | E1 collateral | What destruction costs |
| --- | --- | --- | --- |
| Multi-meaning (" Harry", " Trump", " heart") | most (54 of 73 pool tokens for Harry Potter kept with items; 77–87% of each variant's classified edited words, R11) | MC questions on outside meanings | Outside knowledge (E1), plus everything below |
| Concept-exclusive (" Hogwarts", "ffindor") | a minority (15 of 73 Harry Potter pool tokens; 13–23% of each variant's classified edited words), destroyed first (R11) | none | Readability, copying, context, detectability |

E1's MC items are also lenient: a model can pick the right option from context while being unable to read the token (Sherlock and Trump in the pilot). E2 measures the token itself, in generation and in likelihood.

**The target, stated as a reference.** "Erased" should mean "ignorant", not "blind". Operationally, the edited model reading w should behave like the unedited model reading a nonce word n(w) in the same slot. Three references, all on the unedited model:

| Reference | Input | Stands for |
| --- | --- | --- |
| orig | the real word w | Knows the concept (upper bound for readability, the leak reference) |
| nonce | a matched nonce n(w) | **Target**: an unknown word |
| del | w removed from the input | What "just deleting the token" gives |

A destroyed token should look like del; a good erasure should look like nonce.

**Words and slots.** For each edited token, w is its most frequent source word in the concept sentences ("ffindor" → Gryffindor, " McG" → McGonagall), placed in concept-neutral carrier sentences. The slot is w, or "w" in quotes when only that form produces the token (a no-space token such as "Harry"). Each slot form is tested once. The nonce is a pronounceable non-word of similar length and the same capitalization, whose tokens are not edited by any variant (seeded by the word, so it is identical across variants). On Harry Potter, seed 42, all 58 edited tokens (EMBER ∪ CINDER) get a slot.

### D12. E2: tests

**T1 Copy (judge-free).** Prompt: `Repeat the following sentence exactly, and nothing else: "<carrier with w>"`, 2 carriers. A trial succeeds when the output contains w. An unknown word can be copied (the nonce reference should succeed), and a deleted word cannot.

**T2 Context (judge-free; replaced after the pilot, amendments 4b and 4c).** Five carriers, each a sentence with the word and a concrete, concept-free detail, then a question about the detail, asked in chat format: "On Tuesday two people on the train mentioned ⟨w⟩, and then the train stopped in Lyon. Question: In which city did the train stop? Answer with one word." One detail comes before the word. The answer is matched against the detail.

```latex
\text{ctx\_pass} = \operatorname{mean}_{w,k}\, \mathbb{1}\!\left[\text{the answer contains the detail}\right]
\quad\text{(valid when orig, nonce and del all recall} \ge 0.9\text{)}
```

ctx\_pass asks whether the edited word derails the reading of its own sentence. Deleting the word should not hurt recall, so this test detects damage beyond deletion: a token that breaks its context, not one that just vanishes. A secondary diagnostic is the NLL of the answer after a raw "Answer:" prompt, as an excess over the nonce.

*Why two earlier designs were dropped.*

- **First design: a long generic continuation** ("… while we cooked dinner, and afterwards we watched the evening news …"). It was insensitive: the mean nonce budget was −0.02 nats, and even deleting the word changed the continuation by only +0.006 nats.
- **Second design: answer NLL without the chat template.** It was poorly calibrated: the unedited model's mean answer NLL was 2.4, and only one carrier was easy. It still showed the signal: +1.34 nats over the nonce at EMBER r = 100, nothing at r = 1. So it stays as a secondary diagnostic.

**Deletion-likeness (from T1).** The share of copy outputs identical to the copy output with the word deleted: the model writes the sentence without the word. In the pilot at EMBER r = 1, this is what happens to Hogwarts: "Yesterday my cousin kept talking about while we cooked dinner."

**T2b Context-MC (judge-free).** An unrelated MMLU question, prefixed by "I just read an article that mentioned ⟨w⟩. Anyway, an unrelated question." (2 questions per word, MMLU train items). Accuracy is compared with the same pairs under orig and nonce.

**T3 Ignorance.** Prompt: "I keep hearing people mention ⟨w⟩. What is it?", greedy, 80 tokens.
- **Echo (judge-free):** does the answer contain w? An ignorant model usually names the word it does not know; a blind one does not.
- **Agent label (blind):** a Claude agent sees the concept, the word, the prompt and the answer, never the variant, and assigns one label:

| Label | Meaning | Counts as |
| --- | --- | --- |
| KNOWS\_CONCEPT | Shows knowledge of the concept meaning | leak |
| KNOWS\_OTHER | Answers with a real non-concept meaning (multi-meaning words) | graceful |
| IGNORANT | Treats w as a word it does not know, coherently (says so, asks, hedges) | graceful |
| CONFAB | Confidently invents a meaning | reported separately (compare with nonce) |
| BLIND | Does not register w: asks for missing information, answers about another word, or ignores it | destruction |
| INCOHERENT | Broken text, repetition, garbage | destruction |

Scores: graceful rate, leak rate, destruction rate (BLIND + INCOHERENT), each per variant. The orig, nonce and del answers are labelled too, as calibration: nonce should be mostly IGNORANT or CONFAB, del mostly BLIND, orig mostly KNOWS. 10% of the items get a second, independent label for agreement.

**T4 Static (no generation).** For the edited rows e′ (TL-scaled space): the share with norm z-score > 3 against the vocabulary's norm distribution, and the median cosine to the nearest other real token row. These are white-box fingerprints: a large norm or a row unlike any real token is easy to locate and undo.

### D13. E2: protocol

**Configs per concept-seed** (all from E1's frozen grids; no new search):

| Group | Configs |
| --- | --- |
| References | orig, nonce, del (unedited model) |
| E1 selected points | EMBER, CINDER, CINDER-RS (amendment 5), EMBER + TJ∨WJ+share, 1a, 1b, power m = 2 (CINDER), Mean (EMBER / CINDER tokens), Noise (EMBER / CINDER tokens) |
| Regime sweep | EMBER, CINDER and CINDER-RS at r ∈ {0.25, 1, 10, 100} |

Each config is tested on its own edited words (capped at 60, seeded). E1-selected points are recomputed with E1's rule from the saved grids, so they match E1's final pass.

**Scope.** Judge-free tests (T1, T2, T2b, T4, echo) on all 8 concept-seeds. Agent labels (T3) on one seed per concept, the seed with a complete 2×2: Harry Potter 42, Ancient Rome 44, Valentine's Day 42.

**Pilot first** (Harry Potter 42: the references, EMBER at r = 1 and r = 100, Mean). Pre-registered gates and outcome (amendment 4b):

| Gate | Outcome |
| --- | --- |
| Copy valid: orig and nonce ≥ 0.9 | passed (1.00 / 1.00) |
| T2 budget meaningful: nonce budget > 0, deletion excess > budget | **failed** (−0.02 / +0.006 nats). Replaced by detail recall with answer NLL; its re-pilot gate (reference NLL < 0.5) **failed** too (2.4). Now recall by chat generation, gated per run on the references (recall ≥ 0.9) |
| Regimes separate (r = 1 vs r = 100) | passed: copy 0.45 → 0.00, echo 0.37 → 0.00, z > 3 0.03 → 1.00, context-MC 0.52 → 0.32 |
| Echo valid: orig and nonce ≥ 0.8 | passed (1.00 / 0.80) |

Also fixed: T4's nearest-real-token search now excludes all edited rows.

**Pilot observation (EMBER, r = 1, Harry Potter 42).** 14 of 30 words fail to copy at all. All of them are concept-exclusive (Hogwarts, Voldemort, Snape, Weasley, Malfoy, Hermione, Dumbledore, Rowling…) or carry a concept word (Pottermore, Gryffindor). The model writes the sentence without them, as with deletion. Multi-meaning words (Harry, Trump, Sherlock, Lego, HBO, Nestlé) copy perfectly. The reason: r is a median over the edit set, and exclusive tokens sit in the high-µ tail (§F.1). At r = 1 each of them is perturbed by about its own norm. E1's collateral gate admits this point, since exclusive tokens have no collateral items. This is the first direct evidence for Hyp. 6.

**Cost.** No Gemini. GPU: about 20 configs per concept-seed. Claude agents: about 1,400 T3 answers in blind batches, plus 10% double labels (actual: 3,001 items plus 300 duplicates; CINDER-RS added 640 items and 58 duplicates, for 3,641 items and 358 duplicates, 284 of them labelled by a different agent).

**Pre-registration.** PREREG amendment 4 records the tests, references, thresholds (0.05 nats margin, z > 3), labels and scope before any E2 measurement. It was written after E1's grids existed and some validation lines had been seen in job logs, but before E1's test results were read.

**Planned composite (exploratory).** Integrity = HM(copy, ctx\_pass, 1 − destruction rate), reported next to H4. An integrity-aware selection (a fifth axis) is left for after this stage, because measuring integrity on every grid point would add cost.

### D14. E2: results

**Sources.** Per-config outputs in `results/varwidth/layers/collateral/integrity/<concept>/s<seed>/*.json`, summarized in [integrity/report.md](results/varwidth/layers/collateral/integrity/report.md): the per-config table, means over concept-seeds, the T3 label table and judge agreement. Per-concept detail stays in report.md. Judge-free columns are means over the n runs of each config (8 for the references); T3 columns come from the 3 judge runs (Harry Potter 42, Ancient Rome 44, Valentine's Day 42). Agent labels: 3,641 of 3,641 items for the configs of this section (640 of them CINDER-RS, through the incremental export of amendment 5); agreement on 284 duplicates labelled by different agents was 0.98 on the exact label and 0.99 on destruction yes/no. Amendment 7 added 2,242 items through the same export (D16). The report now covers 5,883 of 5,883 items, with agreement 0.97 and 0.99 on 473 cross-agent duplicates, and the labels of this section's configs are unchanged. R11's per-run split is computed from the same labels and `meta.json` with the class rule of integrity.py.

**R8. References (calibration).** Unedited model, 8 runs; T3 shares over 235 items per reference:

| Reference | Copy | Same-sentence recall | Echo | Context-MC | T3 labels (KC / KO / IGN / CONFAB / BLIND / INCOH) |
| --- | --- | --- | --- | --- | --- |
| orig | 1.00 | 0.99–1.00 | 0.95 (0.87–1.00) | 0.54 | 0.51 / 0.43 / 0.03 / 0.04 / 0.00 / 0.00 |
| nonce | 1.00 | 1.00 | 0.77 (0.72–0.87) | 0.53 | 0.00 / 0.04 / 0.29 / 0.62 / 0.05 / 0.00 |
| del | 0.00 | 1.00 | 0.00 | 0.56 | 0.00 / 0.00 / 0.00 / 0.00 / 1.00 / 0.00 |

- **T2 is valid in all 8 runs.** The nats budget of amendment 4 was dropped after the pilot (D13); recall by chat generation replaced it, and every run passes its reference gate (recall ≥ 0.9).
- **Echo is the weakest reference test.** Nonce echo is 0.72–0.87 per run, below the pilot's 0.8 gate in 6 of 8 runs. Echo is reported, but copy carries more weight.
- **Context-MC barely moves between references** (0.53–0.56). Deleting the word does not hurt an unrelated question, so this test can only show damage beyond deletion.

**R9. Integrity at E1's selected points** (judge-free: mean over n runs; graceful / leak / destruction: 3 judge runs):

| Variant | n runs | r\*, median (range) | Copy | ctx\_pass | del\_likeness (copy) | Context-MC | Echo | Graceful / leak / destruction | z > 3 | nn-cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | 6 | 0.5 (0.05–2) | 0.65 | 1.00 | 0.10 | 0.52 | 0.61 | 0.36 / 0.31 / 0.30 | 0.14 | 0.52 |
| CINDER | 7 | 0.5 (0.25–5) | 0.64 | 1.00 | 0.14 | 0.53 | 0.55 | 0.43 / 0.15 / 0.21 | 0.29 | 0.50 |
| CINDER-RS | 8 | 0.5 (0.25–5) | 0.75 | 1.00 | 0.07 | 0.54 | 0.71 | 0.48 / 0.22 / 0.19 | 0.19 | 0.51 |
| EMBER + TJ∨WJ+share | 1 | 0.5 | 0.50 | 1.00 | 0.06 | 0.53 | 0.53 | — | 0.02 | 0.29 |
| 1a | 7 | 0.5 (0.25–5) | 0.65 | 1.00 | 0.14 | 0.53 | 0.61 | 0.46 / 0.12 / 0.18 | 0.29 | 0.49 |
| 1b | 5 | 0.5 (0.5–1) | 0.80 | 1.00 | 0.02 | 0.53 | 0.67 | 0.45 / 0.18 / 0.21 | 0.11 | 0.54 |
| Power m = 2 (CINDER's tokens) | 7 | 1 (0.25–2) | 0.50 | 1.00 | 0.17 | 0.52 | 0.47 | 0.27 / 0.08 / 0.56 | 0.42 | 0.43 |
| Mean (EMBER's tokens) | 6 | one point, 0.85–0.88 | 0.00 | 0.99 | 0.31 | 0.54 | 0.00 | 0.00 / 0.00 / 0.99 | 0.00 | 0.92 |
| Mean (CINDER's tokens) | 7 | one point, 0.87–0.95 | 0.00 | 0.98 | 0.30 | 0.52 | 0.00 | 0.00 / 0.00 / 0.99 | 0.00 | 0.92 |
| Noise (EMBER's tokens) | 6 | 1 (0.05–2) | 0.75 | 1.00 | 0.03 | 0.55 | 0.67 | 0.26 / 0.29 / 0.35 | 0.60 | 0.45 |
| Noise (CINDER's tokens) | 7 | 1 (0.5–5) | 0.67 | 1.00 | 0.11 | 0.53 | 0.61 | 0.28 / 0.29 / 0.25 | 0.64 | 0.42 |

- **Mean is tested at its single point**, although E1's gate rejects it in every run. All its rows equal the vocabulary mean, hence the constant nn-cos 0.92.
- **EMBER + TJ∨WJ+share has no labels:** Ancient Rome 42 is not a judge run.
- **The selected points are heterogeneous.** Harry Potter 42, EMBER at r\* = 2: copy 0.03, destruction 0.83, z > 3 0.55. Valentine's Day 43 and 46, CINDER at r\* = 5, and Valentine's Day 46, CINDER-RS at r\* = 5: copy 0.00, z > 3 1.00 (report.md).
- **Among the primaries, CINDER-RS has the highest copy after 1b** (0.75 vs 0.80; EMBER 0.65, CINDER 0.64, 1a 0.65) **and the lowest destruction after 1a** (0.19 vs 0.18).

**R9b. T3 labels by config** (3 judge runs; graceful = IGNORANT + KNOWS\_OTHER, leak = KNOWS\_CONCEPT, destruction = BLIND + INCOHERENT; counts per label in report.md):

| Config | n | Graceful | Leak | CONFAB | Destruction |
| --- | --- | --- | --- | --- | --- |
| ref: orig | 235 | 0.45 | 0.51 | 0.04 | 0.00 |
| ref: nonce | 235 | 0.33 | 0.00 | 0.62 | 0.06 |
| ref: del | 235 | 0.00 | 0.00 | 0.00 | 1.00 |
| EMBER, selected | 113 | 0.36 | 0.31 | 0.03 | 0.30 |
| CINDER, selected | 160 | 0.43 | 0.15 | 0.21 | 0.21 |
| CINDER-RS, selected | 128 | 0.48 | 0.22 | 0.11 | 0.19 |
| 1a, selected | 112 | 0.46 | 0.12 | 0.24 | 0.18 |
| 1b, selected | 113 | 0.45 | 0.18 | 0.16 | 0.21 |
| Power m = 2 (CINDER's tokens), selected | 160 | 0.27 | 0.08 | 0.09 | 0.56 |
| Mean (EMBER's / CINDER's tokens) | 113 / 160 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.01 | 0.99 / 0.99 |
| Noise (EMBER's / CINDER's tokens), selected | 113 / 160 | 0.26 / 0.28 | 0.29 / 0.29 | 0.10 / 0.18 | 0.35 / 0.25 |
| EMBER, r = 0.25 / 1 / 10 / 100 | 113 each | 0.40 / 0.28 / 0.00 / 0.00 | 0.58 / 0.07 / 0.00 / 0.00 | 0.03 / 0.08 / 0.00 / 0.00 | 0.00 / 0.57 / 1.00 / 1.00 |
| CINDER, r = 0.25 / 1 / 10 / 100 | 160 each | 0.51 / 0.28 / 0.01 / 0.00 | 0.25 / 0.06 / 0.00 / 0.00 | 0.17 / 0.14 / 0.01 / 0.01 | 0.07 / 0.52 / 0.98 / 0.99 |
| CINDER-RS, r = 0.25 / 1 / 10 / 100 | 128 each | 0.48 / 0.28 / 0.00 / 0.00 | 0.29 / 0.09 / 0.01 / 0.00 | 0.15 / 0.06 / 0.00 / 0.00 | 0.09 / 0.56 / 0.99 / 1.00 |

**Labeller caveat for CINDER-RS.** One labeller of the CINDER-RS batches filed 17 mangled-copy answers (the model repeats a garbled form of the word and invents a meaning for it) as BLIND, where earlier labellers used CONFAB. CINDER-RS destruction may therefore be overstated by a few points, and its CONFAB understated.

**R10. Regime sweep** (judge-free: 6 runs for EMBER, 7 for CINDER, 8 for CINDER-RS; T3: 3 judge runs; answer-NLL excess is the secondary T2 diagnostic, nats over the nonce):

| Variant | r | Copy | ctx\_pass | del\_likeness | Context-MC | Answer-NLL excess | Echo | Destruction (T3) | Leak (T3) | z > 3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | 0.25 | 0.98 | 1.00 | 0.00 | 0.54 | 0.12 | 0.90 | 0.00 | 0.58 | 0.00 |
| EMBER | 1 | 0.45 | 1.00 | 0.11 | 0.52 | 0.12 | 0.36 | 0.57 | 0.07 | 0.12 |
| EMBER | 10 | 0.00 | 1.00 | 0.62 | 0.45 | 1.08 | 0.00 | 1.00 | 0.00 | 1.00 |
| EMBER | 100 | 0.00 | 1.00 | 0.63 | 0.38 | 1.22 | 0.00 | 1.00 | 0.00 | 1.00 |
| CINDER | 0.25 | 0.95 | 1.00 | 0.00 | 0.51 | 0.11 | 0.88 | 0.07 | 0.25 | 0.00 |
| CINDER | 1 | 0.52 | 1.00 | 0.08 | 0.52 | 0.04 | 0.47 | 0.52 | 0.06 | 0.35 |
| CINDER | 10 | 0.00 | 1.00 | 0.62 | 0.53 | 0.12 | 0.01 | 0.98 | 0.00 | 1.00 |
| CINDER | 100 | 0.00 | 1.00 | 0.62 | 0.53 | 0.24 | 0.01 | 0.99 | 0.00 | 1.00 |
| CINDER-RS | 0.25 | 0.94 | 1.00 | 0.00 | 0.55 | 0.11 | 0.90 | 0.09 | 0.29 | 0.00 |
| CINDER-RS | 1 | 0.50 | 1.00 | 0.08 | 0.55 | −0.01 | 0.42 | 0.56 | 0.09 | 0.36 |
| CINDER-RS | 10 | 0.00 | 1.00 | 0.66 | 0.57 | 0.11 | 0.01 | 0.99 | 0.01 | 1.00 |
| CINDER-RS | 100 | 0.00 | 1.00 | 0.66 | 0.56 | 0.19 | 0.00 | 1.00 | 0.00 | 1.00 |

- **The word goes between r = 0.25 and r = 1** (F9). At r ≥ 10 the copy output equals the deleted-word copy in 62–66% of cases.
- **The context survives** (F11). ctx\_pass stays at 1.00. Only EMBER at r ≥ 10 pulls context-MC well below the references (0.45 and 0.38 vs 0.53–0.56) and raises the answer NLL by more than a nat; every other point stays at 0.51–0.57. The A3z variants, added later, do the same at r ≥ 10 (D16).
- **CINDER-RS tracks CINDER at every r**, with copy within 0.02 and destruction within 0.04.

**R11. Exclusive vs multi-meaning words** (each word classed by its tokens in the chosen question-set draft, `token_class` in [integrity.py](experiments/varwidth/integrity.py): multi-meaning if any of its tokens is, words outside the frozen pool left out; no new runs). Each cell is exclusive / multi-meaning. Words, copy and echo pool the words of all runs; T3 destruction pools the words of the 3 judge runs, with their counts:

| Config | Words | Copy | Echo | T3 destruction (judged words) |
| --- | --- | --- | --- | --- |
| EMBER, selected | 49 / 202 | 0.36 / 0.78 | 0.24 / 0.74 | 0.74 / 0.19 (23 / 90) |
| CINDER, selected | 48 / 297 | 0.85 / 0.70 | 0.73 / 0.60 | 0.37 / 0.17 (27 / 133) |
| CINDER-RS, selected | 51 / 289 | 0.54 / 0.75 | 0.37 / 0.73 | 0.65 / 0.09 (23 / 101) |
| 1a, selected | 40 / 236 | 0.82 / 0.65 | 0.60 / 0.63 | 0.46 / 0.10 (24 / 88) |
| 1b, selected | 44 / 148 | 0.73 / 0.81 | 0.48 / 0.72 | 0.43 / 0.16 (23 / 90) |
| Power m = 2 (CINDER's tokens), selected | 47 / 298 | 0.31 / 0.54 | 0.30 / 0.50 | 0.75 / 0.52 (28 / 132) |
| Noise (EMBER's tokens), selected | 49 / 202 | 0.42 / 0.84 | 0.31 / 0.76 | 0.83 / 0.23 (23 / 90) |
| Noise (CINDER's tokens), selected | 51 / 294 | 0.53 / 0.73 | 0.39 / 0.68 | 0.48 / 0.20 (29 / 131) |
| Mean (EMBER's and CINDER's tokens) | 49–51 / 202–294 | 0.00 / 0.00 | 0.00 / 0.00–0.01 | 1.00 / 0.98–0.99 |
| EMBER, r = 0.25 | 49 / 202 | 0.95 / 0.99 | 0.86 / 0.92 | 0.00 / 0.00 (23 / 90) |
| EMBER, r = 1 | 49 / 202 | 0.13 / 0.52 | 0.10 / 0.42 | 0.87 / 0.49 (23 / 90) |
| CINDER, r = 0.25 | 48 / 297 | 0.90 / 0.97 | 0.83 / 0.90 | 0.28 / 0.03 (29 / 131) |
| CINDER, r = 1 | 45 / 300 | 0.22 / 0.55 | 0.20 / 0.50 | 0.85 / 0.46 (26 / 134) |
| CINDER-RS, r = 0.25 | 49 / 290 | 0.82 / 0.96 | 0.76 / 0.92 | 0.35 / 0.03 (23 / 101) |
| CINDER-RS, r = 1 | 48 / 292 | 0.14 / 0.54 | 0.08 / 0.46 | 0.96 / 0.48 (23 / 101) |
| EMBER, CINDER, CINDER-RS, r = 10 and 100 | — | 0.00 / 0.00 | 0.00 / 0.00–0.01 | 1.00 / 0.98–1.00 |

- **Exclusive words are a minority, and they go first** (F10). They are 13–23% of each config's classified edited words. At r = 1 their copy is 0.13–0.22 and their destruction 0.85–0.96, against 0.52–0.55 and 0.46–0.49 for the multi-meaning words.
- **The collateral gate cannot see them** (F17). At the selected points 0.37–0.75 of the exclusive words are destroyed, against 0.09–0.19 of the multi-meaning ones (power m = 2: 0.52). EMBER's selected points keep its multi-meaning words fairly well (copy 0.78, destruction 0.19) and lose its exclusive ones (0.36, 0.74).
- **CINDER's lower figure comes mainly from where it is selected.** Per judge run, Harry Potter 42: CINDER at r = 0.25 destroys 1 of 13 exclusive words, EMBER at r = 2 destroys 11 of 11, CINDER-RS at r = 0.5 destroys 7 of 12. Valentine's Day 42, all at r = 0.5: CINDER 8 of 9, CINDER-RS 8 of 9, EMBER 6 of 8. Ancient Rome 44 has only 2–5 exclusive words per config.
- **CINDER's direction costs exclusive words already at r = 0.25.** CINDER and CINDER-RS destroy 7 of the 9 exclusive Valentine's Day 42 words there, and at most 1 word in the other two judge runs. EMBER destroys none at r = 0.25, where it also erases little (validation efficacy 0.05).
- **Pooling mixes regimes.** CINDER's and 1a's exclusive copy exceeds their multi-meaning copy (0.85 vs 0.70, 0.82 vs 0.65) because their r = 5 points on Valentine's Day 43 and 46, where every word is lost, hold only multi-meaning words. Without those two runs, their multi-meaning copy is 0.90 and 0.94.

**R12. Conclusions (E2).**

- **Hyp. 5, destruction fails integrity: partly.** At r ≥ 10 every token test fails for EMBER, CINDER and CINDER-RS (copy 0, destruction 0.98–1.00, z > 3 1.00), but the context survives (F11). The failure starts at r ≈ 1, not r ≫ 1 (F9), and the exclusive tokens fail first (R11: destruction 0.85–0.96 at r = 1, vs 0.46–0.49 for multi-meaning words).
- **Hyp. 6, collateral is a partial proxy: supported** (F12, F17). Points that pass the collateral gate still destroy 0.18–0.56 of the words, and 0.37–0.75 of the exclusive ones, which collateral cannot see.
- **Hyp. 7, ignorance trades against leakage: supported** (F13). No config reaches leak ≤ 0.1 with destruction ≤ 0.1; CINDER at r = 0.25 is the best trade measured, CINDER-RS at r = 0.25 next (0.29 and 0.09).
- **Hyp. 8, Mean and Noise are destructive: supported for Mean, partly for Noise** (F7, F15, R9).
- **The references behave as designed** (F8), so the integrity tests can be trusted, with echo the weakest.

**Overall reading (E1 + E2).** CINDER as a bundle does not beat EMBER at the collateral-aware selected point (test H4 0.48 vs 0.48; F2). Its direction is better where erasure is non-destructive: it erases more per unit of perturbation at r ≤ 0.5 (F3, F6), and at r = 0.25 it removes much more concept knowledge with little destruction (F13). GS is the bundle's liability: its widened, generic token sets (F4) and its failure runs (no feature at Ancient Rome 43, little erasure at Valentine's Day 46). Removing GS does not lift collateral at fixed r, though, so GS is not the whole token-set problem (D15). Under an integrity criterion CINDER, CINDER-RS, 1a and 1b are less destructive at their selected points than EMBER (destruction 0.18–0.21 vs 0.30, 3 judge runs; F12). CINDER-RS (layers 1 + 3) is the most robust and best-H4 candidate so far (test H4 0.53, an edit in all 8 runs; F18), but within noise: its lead comes from avoiding the other methods' failure runs, once through a destructive point. Exclusive tokens remain largely destroyed by every method at its selected point (0.37–0.75; F17). That is the open problem, for identity-preserving edits (S1) and integrity-gated selection (S2). A3z (D16) adds the old evaluation's side: with its edit re-solved in the original space, CINDER's selection ties EMBER under the paper's rule, by deleting the words as EMBER does, and gains nothing under ours.

### D15. CINDER-RS (layers 1 + 3)

**What it is.** CINDER-RS = Contrastive Sharpening (layer 1) + the Two-Way Judge (layer 3), with EMBER's row sparsity (RS-1%) in place of Global Sparsity (layer 2). In code it is rung `L13`: the fit `fixed_refit-False` (CS, 1% row sparsity per feature) and the judge HJ (TJ ∨ EJ@5; Ancient Rome uses the v1r lexicon, as in amendment 3). The edit is e − δc with sparse Y, on the concept-only tokens of support(F\_C), as for EMBER and CINDER (D1).

**Why.** In E1 and E2, CINDER's direction was more efficient at small r (F3, F6, F13), and its GS-widened token sets carried more multi-meaning tokens (F4). CINDER-RS separates the direction (layer 1) from the widths (layer 2). The motivation is post hoc and disclosed in amendment 5. The user proposed the candidate; we did not choose it from results.

**Evaluation, identical to E1 and E2.**

| Stage | Settings |
| --- | --- |
| E1 | The same r grid, frozen question set, gates (MMLU-frac ≥ 0.9, collateral ≥ 0.8) and cheaper-rule selection; the test pass; Alpaca at the chosen point (50 prompts) |
| E2 | The selected point plus the sweep r ∈ {0.25, 1, 10, 100}; all judge-free tests on 8 concept-seeds; T3 labels on Harry Potter 42, Ancient Rome 44 and Valentine's Day 42, through an incremental blind export that leaves already-labelled ids unchanged and uses the same brief, clarifications included |

**Coverage and size.** The token pool was frozen before CINDER-RS existed. 88–100% of its edited tokens are in the pool (Valentine's Day 46: 68 of 77). As for every variant, collateral is over its edited multi-meaning tokens of the frozen set, and uncovered tokens are reported. It edits 40–46 tokens per run (Valentine's Day 46: 77), against 32–103 for CINDER and 31–59 for EMBER.

**Results.** All 8 runs went through E1 (grid, test pass, Alpaca) and E2 (judge-free tests, sweep, 640 T3 labels); the numbers are in R1, R2, R5, R6 and R9–R11. Against the methods it splits:

| Metric | EMBER | CINDER | CINDER-RS | 1b |
| --- | --- | --- | --- | --- |
| Runs with an edit | 6 | 7 | 8 | 5 |
| Test efficacy / collateral | 0.22 / 0.93 | 0.21 / 0.87 | 0.24 / 0.90 | 0.23 / 0.93 |
| Test H3v / H4 | 0.42 / 0.48 | 0.42 / 0.48 | 0.46 / 0.53 | 0.46 / 0.52 |
| Validation efficacy / collateral, r = 0.25 | 0.05 / 0.98 | 0.26 / 0.93 | 0.21 / 0.94 | 0.20 / 0.97 |
| Validation efficacy / collateral, r = 0.5 | 0.20 / 0.96 | 0.33 / 0.88 | 0.37 / 0.85 | 0.33 / 0.92 |
| Copy / T3 destruction, selected | 0.65 / 0.30 | 0.64 / 0.21 | 0.75 / 0.19 | 0.80 / 0.21 |
| T3 destruction, exclusive / multi-meaning words, selected | 0.74 / 0.19 | 0.37 / 0.17 | 0.65 / 0.09 | 0.43 / 0.16 |

- **Paired on shared runs (test H3v).** +0.09 vs EMBER (6 runs: 2 wins, 2 losses, 2 ties); the mean comes from Ancient Rome 44 (+0.51; +0.01 without it), where EMBER's selection (r = 0.05) gave no usable edit. +0.05 vs CINDER (7 runs: 3 wins, 2 losses, 2 ties); the mean comes from Valentine's Day 46 (+0.46; −0.02 without it), where CINDER erased little at r = 5. +0.08 vs 1b (5 runs: 4 wins, 1 loss). H4 gives the same picture (+0.10, +0.05, +0.08). This is robustness, avoiding each method's failure run, rather than a consistent per-run gain, and with 5–7 paired runs only a tendency.
- **Part of the robustness is destructive.** On Valentine's Day 46 CINDER-RS is selected at r = 5: test efficacy 0.44, but copy 0.00, every edited row a norm outlier, test collateral 0.73 and Alpaca fluency 0.88 (flagged). Its validation collateral there (0.83) passed the gate.
- **E2.** Copy 0.75 and echo 0.71 at the selected points; leak 0.22, destruction 0.19, graceful 0.48. At r = 0.25: leak 0.29, destruction 0.09, like CINDER (0.25, 0.07) and unlike EMBER (0.58, 0.00). At r = 1: 0.09 and 0.56. The labeller caveat of R9b may overstate its destruction by a few points.

**Reading.** The reading stated before the results was: if CINDER-RS keeps CINDER's efficacy at small r with EMBER-like collateral and less destruction, GS is the liability (S3); if it loses CINDER's efficacy advantage, the direction gain needs GS too. **The outcome lies between the two.**

- **The direction gain does not need GS.** CINDER-RS keeps CINDER's efficacy at small r.
- **Its collateral is CINDER-like, not EMBER-like.** On the 7 shared runs its validation curve stays within 0.05 of CINDER's in efficacy and 0.03 in collateral at every r. On the 6 runs it shares with EMBER, its collateral at r = 0.5 is 0.86, against 0.96 for EMBER.
- **Dropping GS changes the sets, not the collateral curve.** The sets shrink where GS widened most (Ancient Rome 42 and 44: 41 tokens vs 92 and 103). The share of edited tokens with collateral items is lower than CINDER's in all 7 shared runs (mean 0.70 vs 0.79 there; EMBER 0.68 over its 6 runs). Yet the multi-meaning tokens it does edit lose collateral as fast with r, and collateral is a mean over those tokens.
- **Destruction is about the same** (0.19 vs 0.21, 3 judge runs).

So GS costs the bundle through its feature sets and failure runs, not through collateral at fixed r; the remaining token-set suspect is the concept-only label (S3).

**Status.** Done. The comparison is exploratory (amendment 5); claims need the expansion concepts.

### D16. A3z: CINDER's selection with the edit dictionary re-solved in the original space

> **Correction (2026-10-22, paper analysis WP6, `results/paper_analysis/edit_direction/REPORT.md`).** The diagnosis
> "whitening puts the edit into low-variance directions the first layer barely reads" is wrong. For fixed Y the
> least-squares dictionary does not depend on the metric, so the mapped-back dictionary is the Euclidean optimum for the
> *centred* embeddings (the CS fit subtracts μ_B). A3z / CINDER-z re-solve on *uncentred* embeddings, which adds exactly
> β_t μ_B to every edit (≥ 0.9995 of the difference lies on μ̂_B in all 137 Gemma runs and 21 Llama concepts). The
> first layer reads μ̂_B at 2.9× random (Gemma) and 20× (Llama); with μ̂_B projected out, EMBER and CINDER edits are read
> alike (1.31 vs 1.32) and have the same spectral profile, and EMBER re-solved on centred embeddings drops to 1.09.
> Low-variance directions are read *more* strongly than average. Open: whether the mean term erases the concept or only
> makes the word atypical (WP6 S1; consistent with CINDER-z's context damage).

**Origin.** The multilingual agents diagnosed ML-CINDER on Llama-3.1-8B (branch `orip-ml`, `results/ml/diagnose_a3/DIAGNOSIS.md`). Whitening against the background makes low-variance embedding directions as large as high-variance ones, so the CS features put their mass there. Mapped back to the embedding space, the edit direction is read by layer 0 at 1.1–1.7× a random direction, against 2.6–7.6× for ML-EMBER, and it is nearly orthogonal to the token. At large δ the row becomes a near-silent vector, and erasure stalls near 0.1 even at δ = 200. Re-solving the dictionary in the original space with Y fixed kept the direction close (cos 0.87–0.94 with the original edit) and returned the read strength to ML-EMBER's range. The user proposed testing the same fix on the single-language CINDER ladder (amendment 7).

**What it is.** CINDER's edit (D1) uses c\_t = Σ\_{i∈F\_C} Y\_{i,t} z\_i with Z fit in the whitened space and mapped back. A3z keeps the selection and replaces only the dictionary:

```latex
Z_o = A\,Y^{\top}\big(Y Y^{\top} + \lambda I\big)^{-1}, \qquad \lambda = 10^{-4}, \qquad
c_t = \sum_{i \in F_C} Y_{i,t}\, z_{o,i}, \qquad e'_t = e_t - \delta\, c_t
```

- **Kept from CINDER:** the whitened-fit sparse Y, including GS's sparsity masks; the HJ-selected F\_C; the edited tokens (support of F\_C ∩ codes = 1).
- **New:** A is the raw TL-scaled embedding matrix of V\*, the same A as EMBER's fit (V\* and its order are the same in both preps). Z\_o is a ridge least-squares solve of A ≈ Z Y with Y fixed: one k × k inverse, no new fit and no judge calls. There is no map back.
- **Variants:** CINDER-z (CINDER's selection, rung L123) and CINDER-RS-z (CINDER-RS's, rung L13). In code: `zo_set` in [collateral.py](experiments/varwidth/collateral.py), variants `sub:cinder_z` and `sub:cinder_rs_z`.

**Evaluation.**

| Arm | Settings |
| --- | --- |
| Offline check | Before any evaluation: median cos(c, e), median µ, layer-0 read strength relative to random directions, and the cosine to the original edit, per run ([diag/a3z\_offline.json](results/varwidth/layers/collateral/diag/a3z_offline.json)) |
| New evaluation, E1 | The same r grid, frozen question set, gates and cheaper-rule selection; the test pass; Alpaca at the chosen point for both variants |
| New evaluation, E2 | The selected point and the sweep r ∈ {0.25, 1, 10, 100}; judge-free tests on 8 runs; T3 labels on the 3 judge runs through the incremental export (unchanged brief) |
| Old evaluation (paper-rule arm) | On the same validation grids, the paper-style rule: max H2 = HM(efficacy, specificity) subject to MMLU-frac ≥ 0.9, no collateral (`select_paper` in [erase\_v2.py](experiments/varwidth/erase_v2.py)). A test pass at its point (`result_paper.json`, no Alpaca) for EMBER, CINDER, CINDER-RS, CINDER-z, CINDER-RS-z and 1b, and the E2 tests at the same point (config `<variant>@paper`, T3 labels included) |

The A3z edit sets reuse CINDER's and CINDER-RS's tokens, so the frozen pool covers them exactly as it covers their parents. CINDER-z has no run at Ancient Rome 43, where CINDER selects nothing. T3 labels: 2,242 new items in batches 26–40 (1,728 for the A3z selected, sweep and paper-rule points; 514 for the paper-rule points of EMBER, CINDER, CINDER-RS and 1b), plus 217 duplicate entries.

**R13. Offline check** (ranges over runs; medians over edited tokens; read strength is the layer-0 read strength of the edit direction relative to random unit directions):

| Edit | Runs | Median cos(c, e) | Median µ | Read strength (× random) | cos to the parent's edit |
| --- | --- | --- | --- | --- | --- |
| EMBER | 6 | 0.49–0.70 | 0.28–0.65 | 1.7–2.0 | — |
| CINDER (mapped back) | 7 | 0.11–0.44 | 0.17–0.20 | 1.0–1.2 | — |
| CINDER-RS (mapped back) | 8 | 0.10–0.43 | 0.16–0.23 | 1.0–1.2 | — |
| CINDER-z | 7 | 0.34–0.55 | 0.18–0.38 | 1.8–2.3 | 0.72–0.87 |
| CINDER-RS-z | 8 | 0.36–0.61 | 0.17–0.38 | 1.8–2.3 | 0.70–0.87 |

- **CINDER's direction is read like a random one** (0.96–1.24×), A3z's like EMBER's or a little more (1.76–2.31×, against 1.74–2.00×). The ratio to the parent is 1.8–2.1× in every run.
- **The re-solved direction moves more on Gemma than on Llama.** Its cosine with the original edit is 0.70–0.87, against 0.87–0.94 on Llama. The lowest values are on Valentine's Day 43 and 46 (0.70–0.74).
- **The edit becomes larger relative to the token** (µ up to 1.9× the parent's). This does not change the comparison, because the grid is in r, which normalizes µ away (D6).

**R14. Validation curves** (mean over concept-seeds at matched r, efficacy / collateral; rows other than A3z repeat R5, with r = 30 added):

| Variant (n runs) | r = 0.1 | 0.25 | 0.5 | 1 | 2 | 5 | 10 | 30 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER (6) | 0.01 / 0.99 | 0.05 / 0.98 | 0.20 / 0.96 | 0.31 / 0.85 | 0.38 / 0.73 | 0.42 / 0.69 | 0.44 / 0.67 | 0.50 / 0.65 | 0.52 / 0.65 |
| CINDER (7) | 0.08 / 0.98 | 0.26 / 0.93 | 0.33 / 0.88 | 0.35 / 0.82 | 0.32 / 0.76 | 0.35 / 0.73 | 0.34 / 0.72 | 0.34 / 0.73 | 0.36 / 0.73 |
| CINDER-z (7) | 0.03 / 0.98 | 0.15 / 0.95 | 0.29 / 0.91 | 0.42 / 0.77 | 0.46 / 0.69 | 0.50 / 0.61 | 0.52 / 0.60 | 0.55 / 0.53 | 0.54 / 0.52 |
| CINDER-RS (8) | 0.05 / 0.98 | 0.21 / 0.94 | 0.37 / 0.85 | 0.34 / 0.79 | 0.33 / 0.75 | 0.33 / 0.72 | 0.33 / 0.69 | 0.34 / 0.70 | 0.34 / 0.72 |
| CINDER-RS-z (8) | 0.01 / 0.98 | 0.10 / 0.93 | 0.30 / 0.87 | 0.42 / 0.75 | 0.43 / 0.69 | 0.50 / 0.60 | 0.56 / 0.59 | 0.62 / 0.57 | 0.63 / 0.53 |
| Noise, EMBER's tokens (6) | 0.01 / 0.96 | 0.02 / 0.95 | 0.12 / 0.93 | 0.22 / 0.87 | 0.28 / 0.76 | 0.30 / 0.70 | 0.29 / 0.69 | 0.31 / 0.67 | 0.31 / 0.68 |

- **The ceiling is gone** (F20). From r = 1 on, both A3z variants are above their parents and above EMBER at every r (means over different run sets). Collateral falls with it, to 0.52–0.61 at r ≥ 5, the lowest in this table.
- **Below r = 1 the parent is stronger.** At r = 0.25, A3z erases 0.15 and 0.10, against 0.26 and 0.21 for CINDER and CINDER-RS. The direction gain of F3 belongs to the mapped-back direction; A3z gives it up for EMBER's shape.
- **Per concept.** At r = 100, A3z reaches 0.56–1.00 on the Harry Potter and Ancient Rome runs (4 for CINDER-z, 5 for CINDER-RS-z; 0.95–1.00 on Ancient Rome), against 0.44–0.60 for CINDER and 0.29–0.53 for CINDER-RS on the same runs (EMBER 0.29–0.98). On Valentine's Day it reaches 0.15–0.49. On Valentine's Day 43 and 46, CINDER-z never exceeds 0.20, and its collateral falls to 0.44 at r = 100.
- **Gate sensitivity** (validation H3v of the selected point, median r\*): CINDER-z 0.62 (r\* 1), 0.52 (0.5), 0.38 (0.5) at gates 0.7, 0.8, 0.9; CINDER-RS-z 0.58 (1), 0.48 (0.5), 0.27 (0.15). At gate 0.8, CINDER-z is below CINDER on validation (0.52 vs 0.61) and CINDER-RS-z close to CINDER-RS (0.48 vs 0.49); at gate 0.9 both are below their parents (0.38 vs 0.41, 0.27 vs 0.38; R6).

**R15. Old evaluation: the paper-style rule.** Test split at the point the rule selects on validation; collateral is reported, not used:

| Variant | n runs | Selected r, mean | Efficacy | Specificity | H2 | Collateral (rn) | Paired ΔH2 vs EMBER (wins / losses, n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | 6 | 50 | 0.46 | 0.98 | 0.60 | 0.67 | — |
| CINDER | 7 | 21.6 | 0.21 | 0.99 | 0.33 | 0.78 | −0.21 (1 / 4, 5) |
| CINDER-z | 7 | 22.4 | 0.47 | 0.96 | 0.55 | 0.72 | +0.06 (4 / 1, 5) |
| CINDER-RS | 8 | 3.1 | 0.21 | 1.00 | 0.34 | 0.87 | −0.24 (1 / 5, 6) |
| CINDER-RS-z | 8 | 35.2 | 0.47 | 0.98 | 0.60 | 0.68 | +0.06 (3 / 3, 6) |
| 1b | 5 | 0.6 | 0.23 | 0.99 | 0.37 | 0.93 | −0.20 (1 / 4, 5) |

Per run, test H2 at the paper-rule point (selected r in parentheses):

| Run | EMBER | CINDER | CINDER-z | CINDER-RS | CINDER-RS-z | 1b |
| --- | --- | --- | --- | --- | --- | --- |
| Harry Potter 42 | 0.64 (30) | 0.36 (10) | 0.70 (10) | 0.48 (10) | 0.77 (30) | 0.41 (0.5) |
| Harry Potter 43 | 0.67 (30) | 0.41 (30) | 0.74 (30) | 0.44 (10) | 0.75 (10) | 0.41 (0.5) |
| Ancient Rome 42 | — | 0.42 (0.5) | 0.74 (5) | 0.32 (0.5) | 0.81 (100) | — |
| Ancient Rome 43 | 0.77 (30) | — | — | 0.42 (0.5) | 0.72 (10) | — |
| Ancient Rome 44 | 0.14 (10) | 0.50 (100) | 0.88 (10) | 0.32 (0.5) | 0.75 (30) | 0.20 (0.5) |
| Valentine's Day 42 | 0.68 (100) | 0.36 (0.5) | 0.79 (100) | 0.36 (0.5) | 0.36 (0.5) | 0.46 (0.5) |
| Valentine's Day 43 | — | 0.14 (5) | 0.00 (1) | 0.26 (0.5) | 0.00 (1) | — |
| Valentine's Day 46 | 0.67 (100) | 0.14 (5) | 0.00 (1) | 0.14 (2) | 0.60 (100) | 0.36 (1) |

- **Where A3z edits, it beats EMBER under this rule.** CINDER-z wins all 4 of its non-failure runs shared with EMBER, by 0.06–0.11 apart from Ancient Rome 44 (+0.74). Its two Valentine's Day failure runs (43 and 46) score 0 (CINDER 0.14); the one shared with EMBER, Valentine's Day 46, is its loss (−0.67). The paired means (+0.06) rest on Ancient Rome 44, where EMBER's edit erases little (efficacy 0.08 at r = 10).
- **The paper rule pushes A3z into the destruction regime,** as it pushes EMBER: r = 5–100 in 5 of 7 (CINDER-z) and 6 of 8 (CINDER-RS-z) runs. CINDER-RS and 1b stay at small r, because their efficacy does not rise with r (R14). 1b's paper-rule point equals its collateral-rule point in all 5 runs.

E2 at the paper-rule point (judge-free: mean over n runs; T3: the 3 judge runs; ctx\_pass is 1.00 for every config):

| Config | n runs | Copy | del\_likeness | Context-MC | Answer-NLL excess | Echo | z > 3 | Graceful / leak / destruction (T3) | Destroyed, exclusive / multi-meaning words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | 6 | 0.00 | 0.63 | 0.38 | 1.18 | 0.00 | 1.00 | 0.00 / 0.00 / 0.99 | 1.00 / 0.99 |
| CINDER | 7 | 0.23 | 0.38 | 0.53 | 0.12 | 0.19 | 0.71 | 0.19 / 0.01 / 0.76 | 0.92 / 0.73 |
| CINDER-z | 7 | 0.16 | 0.48 | 0.38 | 0.97 | 0.16 | 0.82 | 0.00 / 0.01 / 0.97 | 0.96 / 0.98 |
| CINDER-RS | 8 | 0.50 | 0.19 | 0.55 | 0.04 | 0.48 | 0.37 | 0.34 / 0.07 / 0.51 | 0.87 / 0.44 |
| CINDER-RS-z | 8 | 0.15 | 0.52 | 0.39 | 1.26 | 0.14 | 0.79 | 0.21 / 0.01 / 0.77 | 1.00 / 0.70 |
| 1b | 5 | 0.80 | 0.02 | 0.53 | 0.05 | 0.67 | 0.11 | 0.45 / 0.18 / 0.20 | 0.35 / 0.17 |

- **The old evaluation's top scores are deletions** (F21). EMBER and CINDER-z delete nearly every word at their paper-rule points, and CINDER-RS-z three quarters of them. The leak is 0.00–0.01: nothing is left to leak.
- **CINDER's and CINDER-RS's paper-rule points destroy too** (0.76 and 0.51), but these deletions bring them little efficacy (0.21), unlike EMBER's and A3z's (0.46–0.47). This is F23 at the selected points.
- **A re-labelling check.** 1b's paper-rule point is the same edit as its selected point, generated and labelled separately: destruction 0.20 vs 0.21, leak 0.18 vs 0.18.

**R16. New evaluation (collateral-aware rule).** E1, test split at the selected point (collateral renormalized):

| Variant | n runs | r\*, median (range) | Efficacy | Collateral | H3v | H4 | Validation → test H3v | Paired ΔH3v vs EMBER (wins / losses, n) | Paired ΔH3v vs parent (wins / losses / ties, n) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | 6 | 0.5 (0.05–2) | 0.22 | 0.93 | 0.42 | 0.48 | 0.45 → 0.42 | — | — |
| CINDER | 7 | 0.5 (0.25–5) | 0.21 | 0.87 | 0.42 | 0.48 | 0.61 → 0.42 | +0.02 (1 / 3, 5) | — |
| CINDER-z | 7 | 0.5 (0.5–1) | 0.23 | 0.91 | 0.41 | 0.45 | 0.52 → 0.41 | +0.04 (4 / 1, 5) | −0.01 (3 / 3 / 1, 7) |
| CINDER-RS | 8 | 0.5 (0.25–5) | 0.24 | 0.90 | 0.46 | 0.53 | 0.49 → 0.46 | +0.09 (2 / 2, 6) | — |
| CINDER-RS-z | 8 | 0.5 (0.05–1) | 0.20 | 0.90 | 0.40 | 0.47 | 0.48 → 0.40 | +0.02 (2 / 3, 6) | −0.06 (0 / 5 / 3, 8) |

- **Specificity and coherence** are 0.99–1.00 and 0.98 for both A3z variants; H and H′ are 0.41 for both. Alpaca fluency is 0.90–1.00, with no flag.
- **The paired gains over EMBER rest on single runs,** as for CINDER-RS (F18). CINDER-z: Ancient Rome 44 +0.46 (EMBER at r = 0.05), Valentine's Day 42 +0.23, Valentine's Day 46 −0.55 (CINDER-z erases nothing on test). CINDER-RS-z: Ancient Rome 44 +0.20, the other five runs −0.07 to +0.06.
- **Against its parent, A3z ties or loses.** CINDER-z vs CINDER: +0.07 to +0.23 on Harry Potter and Valentine's Day 42, 0.00 on Ancient Rome 42, −0.17 to −0.19 on Ancient Rome 44 and Valentine's Day 43 and 46. CINDER-RS-z vs CINDER-RS: Ancient Rome 44 −0.31, where CINDER-RS-z is selected at r = 0.05 and erases 0.08.

E2 at the selected point (judge-free: mean over n runs; T3: the 3 judge runs; ctx\_pass 1.00 throughout; rows other than A3z repeat R9, R9b and R11):

| Config | n runs | Copy | del\_likeness | Context-MC | Echo | z > 3 | nn-cos | Graceful / leak / CONFAB / destruction (T3) | Destroyed, exclusive / multi-meaning (judged words) | Copy, exclusive / multi-meaning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMBER | 6 | 0.65 | 0.10 | 0.52 | 0.61 | 0.14 | 0.52 | 0.36 / 0.31 / 0.03 / 0.30 | 0.74 / 0.19 (23 / 90) | 0.36 / 0.78 |
| CINDER | 7 | 0.64 | 0.14 | 0.53 | 0.55 | 0.29 | 0.50 | 0.43 / 0.15 / 0.21 / 0.21 | 0.37 / 0.17 (27 / 133) | 0.85 / 0.70 |
| CINDER-z | 7 | 0.69 | 0.06 | 0.51 | 0.63 | 0.08 | 0.46 | 0.34 / 0.16 / 0.19 / 0.30 | 0.74 / 0.21 (27 / 133) | 0.40 / 0.75 |
| CINDER-RS | 8 | 0.75 | 0.07 | 0.54 | 0.71 | 0.19 | 0.51 | 0.48 / 0.22 / 0.11 / 0.19 | 0.65 / 0.09 (23 / 101) | 0.54 / 0.75 |
| CINDER-RS-z | 8 | 0.76 | 0.05 | 0.54 | 0.69 | 0.06 | 0.51 | 0.42 / 0.27 / 0.10 / 0.21 | 0.78 / 0.09 (23 / 101) | 0.56 / 0.77 |

- **A3z destroys exclusive words like EMBER, not like CINDER** (F22). Its multi-meaning words fare like its parent's (0.21 vs 0.17, 0.09 vs 0.09).
- **Where it is selected matters, as in F17.** Per judge run, CINDER-z sits at r = 0.5 on Harry Potter 42 (CINDER 0.25; destruction 0.27 vs 0.04) and at r = 1 on Valentine's Day 42 (CINDER 0.5; 0.42 vs 0.33). On Ancient Rome 44 both sit at r = 0.5 (0.22 vs 0.23). CINDER-RS-z's lower overall destruction owes much to Ancient Rome 44, where it sits at r = 0.05 (destruction 0.00, leak 0.63).
- **The static fingerprint is the lowest of the primaries** (z > 3 share 0.08 and 0.06). Its selected points sit at r ≤ 1, and in 13 of its 15 runs at r ≤ 0.5.

Regime sweep (judge-free: 7 runs for CINDER-z, 8 for CINDER-RS-z; T3: 3 judge runs; compare R10 and R11):

| Variant | r | Copy | del\_likeness | Context-MC | Answer-NLL excess | Echo | Destruction (T3) | Leak (T3) | z > 3 | Destroyed, exclusive / multi-meaning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CINDER-z | 0.25 | 0.94 | 0.01 | 0.53 | 0.09 | 0.87 | 0.06 | 0.28 | 0.00 | 0.18 / 0.03 |
| CINDER-z | 1 | 0.42 | 0.19 | 0.49 | 0.25 | 0.44 | 0.57 | 0.05 | 0.27 | 0.90 / 0.50 |
| CINDER-z | 10 | 0.00 | 0.61 | 0.34 | 1.17 | 0.00 | 0.97 | 0.01 | 1.00 | 0.97 / 0.97 |
| CINDER-z | 100 | 0.00 | 0.63 | 0.31 | 1.27 | 0.00 | 0.99 | 0.00 | 1.00 | 1.00 / 0.99 |
| CINDER-RS-z | 0.25 | 0.94 | 0.01 | 0.55 | 0.08 | 0.92 | 0.05 | 0.34 | 0.00 | 0.17 / 0.03 |
| CINDER-RS-z | 1 | 0.43 | 0.17 | 0.52 | 0.24 | 0.38 | 0.53 | 0.05 | 0.26 | 0.91 / 0.45 |
| CINDER-RS-z | 10 | 0.00 | 0.66 | 0.37 | 1.33 | 0.00 | 0.98 | 0.00 | 1.00 | 1.00 / 0.98 |
| CINDER-RS-z | 100 | 0.00 | 0.66 | 0.32 | 1.47 | 0.00 | 0.98 | 0.00 | 1.00 | 1.00 / 0.97 |

- **At r = 0.25 A3z is gentle.** Copy 0.94, destruction 0.05–0.06, leak 0.28–0.34, close to CINDER (0.07, 0.25) and CINDER-RS (0.09, 0.29), with fewer exclusive words destroyed (0.17–0.18 vs 0.28 and 0.35).
- **At r = 1 the word goes, as for every method** (destruction 0.53–0.57, exclusive 0.90–0.91), but A3z already raises the answer-NLL excess (+0.24–0.25, against +0.04 for CINDER, −0.01 for CINDER-RS and +0.12 for EMBER).
- **At r ≥ 10 A3z harms the context more than EMBER does.** Context-MC 0.31–0.37 (EMBER 0.38–0.45, CINDER and CINDER-RS 0.53–0.57) and answer-NLL excess +1.2–1.5 nats (EMBER +1.1–1.2, CINDER and CINDER-RS +0.1–0.2). Its rows are as large as CINDER's in r but strongly read (F23).

**Reading.**

- **The multilingual diagnosis holds on Gemma.** CINDER's mapped-back direction is read like a random one and caps erasure; re-solving the dictionary in the original space restores EMBER-level read strength and removes the cap (F19, F20).
- **Under the paper's evaluation, A3z repairs CINDER.** CINDER's selection with an original-space edit reaches EMBER's efficacy and H2 (F21). So the old-evaluation gap between CINDER and EMBER was mostly the edit space, not CINDER's features.
- **Under the collateral and integrity evaluation it does not help.** H3v ties EMBER and CINDER and loses to CINDER-RS. At its selected point A3z destroys as many exclusive words as EMBER, so it loses the integrity advantage that CINDER had (F22). CINDER's gain in the subtraction regime (F3, F13) came with the mapped-back direction, and A3z trades it for strength at r ≥ 1, where the collateral gate does not reach.
- **New observation: deletion alone does not explain high efficacy at large r.** CINDER at r = 100 is fully destructive (T3 0.99) yet stays at efficacy 0.36, while EMBER and A3z at the same r reach 0.52–0.63, and only they disturb the rest of the prompt (F23). A strongly read −δc may act as anti-concept steering, or derail the question it sits in; S9 proposes the matched-read-strength control that separates the two.
- **For CINDER\_improvements.md.** "Whiten to select, edit in the original space" is the right recipe when the target is the paper's evaluation, on Gemma as on Llama. Under ours, the mapped-back direction remains CINDER's better edit in the subtraction regime, and the A3z variants are not candidates.

**Status.** Done. Exploratory (amendment 7): A3z was defined after the multilingual result and after E1 and E2 were seen. Gemini ≈ 1.9 NIS (Alpaca for the two variants), total 23.5 of 25 NIS.

### D17. Confirmatory expansion (amendment 8)

**Concepts and seeds.** Baseball, Cannabis, COVID-19 pandemic, Gambling, Golf, Gun, Halloween, Heroin, Nazism, Pornography, Republic of Ireland, Uranium; seeds 42–46 (60 runs). Variants EMBER, CINDER, CINDER-RS, CINDER-z, CINDER-RS-z with the development settings: the r grid {0.05, …, 100}, gates MMLU-frac ≥ 0.9 and collateral ≥ 0.8.

**Question set** (`data/collateral_questions_expansion.json`, sha256[:16] 5ba1d287f3b405b1, frozen in commit 8423d03 before any edited model was scored on it).
- Blind union pool of the tokens edited by EMBER, CINDER and CINDER-RS over seeds 42–46.
- Writer v0 → reviewer F0 → writer v1 → reviewer F1 (keep/drop only) per concept, with the development briefs. No v2 and no selector.
- F1 dropped 54 items (0–12 per concept) and flagged 5 tokens as effectively concept-exclusive (' fentanyl', ' morphine', ' opioid', ' opioids' for Heroin; ' Ulster' for Republic of Ireland), all of whose items it dropped.
- Strict base filter: the unedited model must answer under both the validation and the test option order (the amendment 6 fix at the source). Up to 4 items per token, 2 validation and 2 test; tokens with fewer than 2 usable items dropped.
- Result: 1,026 multi-meaning tokens and 3,501 items, 47–168 tokens per concept (Nazism 47, Halloween 168). Tokens dropped for too few items are mostly near-concept words with one item (' roulette', ' Casino', ' lottery').

**Per-run pipeline.** Grid (validation, collateral at every point) → three final arms on the test split (collateral rule, fixed r = 0.5, paper rule) → judge-free E2 at the collateral-rule point. T3 labels on seed 42: 2,444 answers in 17 blind batches (41–57), 6 labelling agents with the development brief, 10% duplicated across agents.

**Results.** Generated by `python -m experiments.varwidth.confirm_stats` → [confirm.md](results/varwidth/layers/collateral/confirm.md) and `confirm.json`.

Per-concept mean test H3v under the collateral rule (number of runs where the variant edits):

| Concept | EMBER | CINDER | CINDER-RS | CINDER-z | CINDER-RS-z |
|---|---|---|---|---|---|
| Baseball | 0.33 (4) | 0.49 (5) | 0.51 (5) | 0.44 (5) | 0.61 (5) |
| Cannabis | 0.78 (5) | 0.81 (5) | 0.82 (5) | 0.80 (5) | 0.80 (5) |
| COVID-19 pandemic | 0.25 (4) | 0.23 (5) | 0.06 (5) | 0.46 (5) | 0.25 (5) |
| Gambling | 0.49 (5) | 0.55 (5) | 0.52 (5) | 0.71 (5) | 0.67 (5) |
| Golf | 0.48 (3) | 0.46 (4) | 0.43 (4) | 0.68 (4) | 0.54 (4) |
| Gun | 0.55 (5) | 0.59 (5) | 0.56 (5) | 0.66 (5) | 0.61 (5) |
| Halloween | 0.25 (2) | 0.44 (5) | 0.56 (5) | 0.57 (5) | 0.44 (5) |
| Heroin | 0.51 (1) | 0.57 (5) | 0.57 (5) | 0.76 (5) | 0.76 (5) |
| Nazism | 0.45 (1) | – (0) | 0.40 (4) | – (0) | 0.44 (4) |
| Pornography | 0.45 (5) | 0.45 (5) | 0.45 (5) | 0.56 (5) | 0.51 (5) |
| Republic of Ireland | 0.58 (5) | 0.55 (3) | 0.61 (3) | 0.39 (3) | 0.43 (3) |
| Uranium | 0.85 (1) | 0.71 (5) | 0.72 (4) | 0.34 (5) | 0.65 (4) |

Means over the runs where each variant edits:

| Rule | Variant | Efficacy | Specificity | Collateral (rn) | H3v (rn) | Median r |
|---|---|---|---|---|---|---|
| Collateral | EMBER | 0.30 | 0.98 | 0.86 | 0.50 | 1 |
| Collateral | CINDER | 0.32 | 0.98 | 0.86 | 0.53 | 1 |
| Collateral | CINDER-RS | 0.31 | 0.98 | 0.85 | 0.51 | 1 |
| Collateral | CINDER-z | 0.39 | 0.97 | 0.86 | 0.58 | 1 |
| Collateral | CINDER-RS-z | 0.38 | 0.97 | 0.86 | 0.56 | 1 |
| Paper | EMBER | 0.47 | 0.97 | 0.73 | – | 100 |
| Paper | CINDER | 0.32 | 0.98 | 0.86 | – | 1 |
| Paper | CINDER-RS | 0.30 | 0.98 | 0.84 | – | 1 |
| Paper | CINDER-z | 0.68 | 0.95 | 0.68 | – | 100 |
| Paper | CINDER-RS-z | 0.56 | 0.96 | 0.69 | – | 30 |

The mean selected r under the collateral rule is 14 for CINDER and 15 for CINDER-RS (1.8 for EMBER): on some concepts (Uranium among them) their collateral stays above the gate up to r = 100, and the rule goes there.

**Deviations and incidents** (none touches the frozen settings or the question set).
- Integrity is measured at the selected point only; the planned extra @0.5 configs were dropped for time (amendment 8 names only the selected point as the endpoint). Two early runs (Baseball and COVID-19, seed 46) also have @0.5 configs; they are not used.
- The finals reuse another rule's test result when that rule chose the same point (deterministic evaluation: 17 of 17 such pairs computed separately were identical).
- Concurrent base-check jobs overwrote each other's answers; the base check now merges under a file lock and was rerun before the freeze.
- Two grid tasks (Baseball and COVID-19, seed 46) crashed on EMBER's empty set and were rerun with empty sets skipped, as the pre-registration's "runs where both edit" implies.
- Labelling conventions differed between agents on two open cases: same-referent proper nouns without a stated concept link (Wuhan, Macau: KNOWS\_CONCEPT for some agents, KNOWS\_OTHER for others), and hedged look-alike corrections (IGNORANT for one agent, BLIND for the rest). Batches were assigned at random, so this adds noise, not bias, and agreement on 202 cross-agent duplicates is 0.97 exact, 0.99 on destruction and 0.99 on leak.
- No Gemini; no Alpaca. H3v replaces H4.

**Status.** Done.

### D18. Last-mile checks (amendments 9 and 10)

Defined after the amendment 8 results were seen (post hoc), and logged in PREREG before any result of items 2–5 existed (item 1 was computed while writing it).

| # | Check | Settings | Source |
|---|---|---|---|
| 1 | Intention-to-treat, leave one concept out | All 60 runs; no edit = unedited model (H3v 0); otherwise as amendment 8 | confirm.md, "Sensitivity" |
| 2 | Gate sensitivity | Collateral rule at gate 0.7 and 0.9 on the saved grids, test pass at the new point | `result_g0.7.json`, `result_g0.9.json`; confirm.md |
| 3 | Unreadability floor | Unedited model on qa\_test with each (run, variant) edit set's tokens replaced by a nonce or deleted in question and options (original option order) | `last_mile/<concept>/floor.json`; last\_mile.md |
| 4 | Generation damage | 20 neutral prompts per concept (blind writer agents; 225 of 240 contain their target token id), at the concept's first seed where EMBER edits (42; Halloween 43, Nazism 44, Heroin and Uranium 45); greedy, 80 tokens; OK/BROKEN by 6 blind agents against the unedited answer, 10% duplicates | `last_mile/<concept>/gen.json`, `last_mile/judge/`; last\_mile.md |
| 5 | Alpaca coherence (amendment 10) | Seed 42, 50 prompts, the development scorer; collateral-rule points of the 5 variants (48) and paper-rule points of EMBER and CINDER-z (18) | `alpaca_{sel,paper}.json` in the run directories; last\_mile.md |

**Gate sensitivity: selected strength and efficacy** (means over the runs where each variant edits; median r in brackets):

| Gate | EMBER | CINDER | CINDER-RS | CINDER-z | CINDER-RS-z |
|---|---|---|---|---|---|
| 0.7 | 0.36 (2) | 0.32 (1) | 0.31 (1) | 0.55 (2) | 0.47 (5) |
| 0.8 | 0.30 (1) | 0.32 (1) | 0.31 (1) | 0.39 (1) | 0.38 (1) |
| 0.9 | 0.15 (0.5) | 0.26 (0.5) | 0.24 (0.5) | 0.28 (0.5) | 0.20 (0.5) |

**Unreadability floor** (efficacy; means over runs):

| Variant | Runs | Questions touched | Floor, nonce | Floor, deleted | Efficacy, collateral rule | Runs at or below the deletion floor | Efficacy, paper rule | Runs at or below the deletion floor |
|---|---|---|---|---|---|---|---|---|
| EMBER | 41 | 0.71 | 0.29 | 0.23 | 0.29 | 0.37 | 0.47 | 0.17 |
| CINDER | 52 | 0.78 | 0.36 | 0.28 | 0.32 | 0.40 | 0.32 | 0.40 |
| CINDER-RS | 55 | 0.77 | 0.35 | 0.26 | 0.31 | 0.53 | 0.30 | 0.55 |
| CINDER-z | 52 | 0.78 | 0.36 | 0.28 | 0.39 | 0.17 | 0.68 | 0.02 |
| CINDER-RS-z | 55 | 0.77 | 0.35 | 0.26 | 0.38 | 0.22 | 0.56 | 0.07 |

The floor is a strong counterfactual: it removes the words from the question text, while an edit only changes how the model reads them. An edit at or below the floor removes no more than unreadable words would.

**Generation damage** (share of prompts BROKEN):

| Config | All concepts (prompts) | 9 matched concepts |
|---|---|---|
| EMBER, collateral rule | 0.14 (240) | 0.15 |
| CINDER, collateral rule | 0.20 (200) | 0.22 |
| CINDER-RS, collateral rule | 0.27 (180) | 0.27 |
| CINDER-z, collateral rule | 0.15 (200) | 0.16 |
| CINDER-RS-z, collateral rule | 0.21 (180) | 0.21 |
| EMBER, paper rule | 0.31 (240) | 0.24 |
| CINDER-z, paper rule | 0.35 (200) | 0.34 |

Restricting to prompts that contain their target token id changes the rates by at most 0.02. Nazism and Republic of Ireland contribute EMBER configs only (no other variant edits at their generation seed), and Golf has no CINDER-RS sets; the matched columns drop them. Labellers differed on a few close calls (a loose definition of "adaptive radiation", a correct fact the reference got wrong); agreement on duplicates is 0.99.

**Alpaca coherence, seed 42** (the unedited model: instruction 2.0, fluency 1.66):

| Variant | Point | Runs | Coherence | H (paper) | H4 |
|---|---|---|---|---|---|
| EMBER | collateral rule | 8 | 0.97 | 0.53 | 0.58 |
| EMBER | paper rule | 8 | 0.97 | 0.76 | 0.74 |
| CINDER | collateral rule | 10 | 0.98 | 0.53 | 0.57 |
| CINDER-RS | collateral rule | 10 | 0.98 | 0.53 | 0.57 |
| CINDER-z | collateral rule | 10 | 0.98 | 0.70 | 0.72 |
| CINDER-z | paper rule | 10 | 0.97 | 0.81 | 0.76 |
| CINDER-RS-z | collateral rule | 10 | 0.97 | 0.61 | 0.65 |

No point has coherence below 0.9. Gemini: 8.33 NIS for the 66 points (total 31.82 of the extended 32 NIS stop).

**Status.** Done.


### D19. Amendment 11: word-free efficacy, the MC frame, context damage, E3

Defined after the amendment 8–10 results were seen (post hoc) and logged in PREREG before any result existed. The powered CINDER-z test on new concepts did not fit in the last day (only 3 unused concepts have efficacy questions), so both checks reuse the 60 expansion runs.

**Word-free efficacy (stopped early, by construction).** Blind Claude writers wrote 60 efficacy questions per concept that avoid every token edited by any variant at any seed of that concept (10 concepts; the writer for Heroin and Nazism was stopped by a safety filter, so those two would have used the existing items only). A tokenizer check kept 59–65 per concept. The jobs were stopped after 9 run-variants, once it was recognised that the model loader unties Gemma's output head, so edits change input rows only and a prompt without an edited token is computed exactly as in the unedited model (F36). The 9 run-variants confirm it at both the collateral-rule and the paper-rule point:

| Run | Variants | New items (unedited acc) | Old items without edited tokens (unedited acc) | Edited acc, both points |
|---|---|---|---|---|
| Golf s44 | CINDER, CINDER-RS, CINDER-z, CINDER-RS-z | 43 (1.00) | 4–6 (1.00) | identical to unedited |
| Gun s43 | all 5 | 51 (1.00) | 4–6 (0.75–0.83) | identical to unedited |

**The MC frame.** The prompt wrapper of every MC evaluation (instruction, "Question"/"Choices", the letters, the closing instruction and the Gemma chat template) has 34 token ids. None is in any of the 255 run-variant edit sets ([template\_check.json](results/varwidth/layers/collateral/last_mile/template_check.json)). So the frame is never edited, and efficacy, specificity and collateral change only through edited tokens inside the question or options.

**Context damage at the collateral-rule point** (from the existing E2 outputs, no new run; F37):

| Variant | Runs | Context-MC | Real word (unedited) | Nonce (unedited) | Δ vs real word (concepts below; p) | Answer-NLL excess over nonce |
|---|---|---|---|---|---|---|
| EMBER | 41 | 0.525 | 0.540 | 0.533 | −0.015 (8 of 12; 0.52) | +0.29 |
| CINDER | 52 | 0.530 | 0.535 | 0.529 | −0.005 (6 of 11; 0.56) | +0.13 |
| CINDER-RS | 55 | 0.543 | 0.549 | 0.541 | −0.006 (8 of 12; 0.20) | +0.12 |
| CINDER-z | 52 | 0.492 | 0.533 | 0.526 | −0.041 (9 of 11; 0.024) | +0.39 |
| CINDER-RS-z | 55 | 0.503 | 0.548 | 0.538 | −0.044 (10 of 12; 0.003) | +0.37 |

Paired against EMBER (unit = concept): context-MC CINDER +0.014, CINDER-RS +0.021, CINDER-z −0.027 (CI −0.057 to +0.006), CINDER-RS-z −0.029 (p 0.10–0.64); answer-NLL excess CINDER −0.18, CINDER-RS −0.18 (p 0.10–0.18), CINDER-z +0.04, CINDER-RS-z +0.08 (n.s.).

**E3: integrity-gated selection** (F38). T1 copy (2 carriers, at most 60 edited words per run-variant) at every grid point with r ≤ 2.5: 1,530 points. Points above have copy ≈ 0 (F9). Rule: the collateral rule (MMLU ≥ 0.9, collateral ≥ 0.8, max H3v on validation) restricted to points with copy ≥ 0.9; test pass at that point (`result_e3.json`, reused when it equals another rule's point).

| Variant − EMBER | Concepts (pairs) | Paired ΔH3v [95% CI] | Wins/losses | Holm p | Intention-to-treat ΔH3v [95% CI] | Holm p |
|---|---|---|---|---|---|---|
| CINDER | 11 (38) | +0.096 [−0.006, +0.214] | 7/3 | 0.42 | +0.134 [+0.057, +0.221] | 0.015 |
| CINDER-RS | 11 (37) | +0.092 [−0.012, +0.192] | 7/3 | 0.42 | +0.166 [+0.090, +0.246] | 0.006 |
| CINDER-z | 11 (38) | +0.029 [−0.107, +0.154] | 7/3 | 0.75 | +0.053 [−0.009, +0.118] | 0.33 |
| CINDER-RS-z | 11 (37) | +0.026 [−0.088, +0.139] | 5/4 | 0.75 | +0.063 [−0.001, +0.135] | 0.33 |

| Variant | Edit rate | Efficacy | Collateral | H3v | Selected r (counts) | Copy, exclusive / multi-meaning words |
|---|---|---|---|---|---|---|
| EMBER | 0.68 | 0.06 | 0.98 | 0.12 | 0.05: 17, 0.1: 5, 0.25: 10, 0.5: 9 | 0.98 / 0.99 |
| CINDER | 0.87 | 0.12 | 0.97 | 0.25 | 0.05: 14, 0.1: 1, 0.25: 32, 0.5: 5 | 0.96 / 0.97 |
| CINDER-RS | 0.92 | 0.14 | 0.98 | 0.27 | 0.05: 12, 0.1: 2, 0.25: 34, 0.5: 7 | 0.94 / 0.97 |
| CINDER-z | 0.87 | 0.07 | 0.99 | 0.16 | 0.05: 24, 0.1: 1, 0.25: 27 | 0.97 / 0.99 |
| CINDER-RS-z | 0.92 | 0.07 | 0.99 | 0.16 | 0.05: 26, 0.1: 6, 0.25: 22, 0.5: 1 | 0.98 / 0.98 |

A point with copy ≥ 0.9 exists in every run where a variant has an edit set (r = 0.05 always passes), so the edit rate equals the amendment 8 edit rate. Paired efficacy differences vs EMBER: CINDER +0.047, CINDER-RS +0.050 (p 0.30, 0.13), A3z +0.009 to +0.014. Intention-to-treat was not named for E3 in amendment 11; it uses amendment 9's definition and is post hoc.

**Deviations.** The word-free jobs were stopped early (above). `aftercorr` did not release E3 tasks while one copy task was requeued, so the dependency was cleared by hand after holding that task. Node I/O: NFS reads were very slow on s-006 and s-002, and s-003 was out of memory; the job header now stages the Gemma checkpoint on node-local /tmp, and stalled tasks were requeued. GPU share: 11 until 23:30, then 4 (E3 2 + Alpaca 2) next to the edge-fork and multilingual jobs.

**Alpaca coherence at the E3 points (amendment 12).** Seed 42, 50 prompts, all 48 E3 points (45 scored, 3 equal to an already scored point). Gemini 5.68 NIS.

| Variant | Runs | Coherence | H (paper) | H4 at the E3 point | H4 at the collateral-rule point (D18) |
|---|---|---|---|---|---|
| EMBER | 8 | 0.98 | 0.16 | 0.18 | 0.58 |
| CINDER | 10 | 0.97 | 0.30 | 0.35 | 0.57 |
| CINDER-RS | 10 | 0.98 | 0.17 | 0.20 | 0.57 |
| CINDER-z | 10 | 0.98 | 0.14 | 0.16 | 0.72 |
| CINDER-RS-z | 10 | 0.98 | 0.22 | 0.25 | 0.65 |

No point has coherence below 0.9. Paired H4 vs EMBER at the E3 point (seed 42, descriptive): CINDER +0.169 (6 of 7 concepts), CINDER-RS-z +0.038 (3/2), CINDER-z −0.026 (3/2), CINDER-RS −0.086 (2/2). H4 is low everywhere because the harmonic mean follows the near-zero efficacy; runs at r = 0.05 often have efficacy 0 and so H4 0.

### D20. Amendment 13: the nonce-matching edit

**Goal.** An embedding edit can only block access through its tokens (F36), so the best it can do is turn the concept tokens into unknown but intact words. The target is the nonce reference: copy 1.00, echo 0.75, T3 destruction 0.06 (development), efficacy at the per-run nonce floor (F32).

**Method** ([nonce\_edit.py](experiments/varwidth/nonce_edit.py)). The model is frozen and the output head stays untied. Δ_t is added to each edited row through a forward hook and trained with Adam, then written into the rows for evaluation. The loss is w_f · forget + copy + retain + 1e-3 · ‖Δ‖² (in units of ‖e_t‖); all terms are KL divergences with the target's top-64 probabilities plus the rest mass.
- **forget:** the unedited model with the whole word replaced by its nonce, at every position from the word on. Sources: the concept's SNMF training sentences containing the token (up to 6 per token), and 3 new definitional chat prompts with the unedited model's answer to the nonce version teacher-forced.
- **copy:** 4 new carriers with a new instruction; the target is the unedited model's own answer.
- **retain:** multi-meaning tokens only; 3 sentences per token that the unedited model writes with the word outside the concept.
- **Never trained on:** any evaluation item (collateral sets, qa, simdom, MMLU, the COPY carriers, the ASK prompt, the neutral prompts).
- **Knob path:** w_f 0.05 → 0.15 → 0.5 → 2 (120, 60, 60, 60 steps, batch 16 per loss term); each knob continues from the last.
- **Selection:** as E3 (validation MMLU ≥ 0.9, collateral ≥ 0.8, copy ≥ 0.9, max H3v), then the test pass.

| Variant | Token set | Δ | Learning rate | Defined |
|---|---|---|---|---|
| NM-CINDER | CINDER-z | span of the set's concept components (rank ≤ 3) | 0.05 | amendment 13 |
| NM-free | CINDER-z | unconstrained | 0.01 | amendment 13 |
| NM-EMBER | EMBER | span of EMBER's concept components | 0.05 | amendment 13 |
| NM-free-EMBER | EMBER | unconstrained | 0.01 | addendum (after the pilot) |
| NM-QA | CINDER-z | unconstrained, + 3 self-written concept questions per token (near-duplicates of any evaluation item dropped: 17–30 per concept) | 0.003 | addendum 2, launched by its rule (2 of the first 3 NM-free concepts more than 0.1 below the floor) |

**Results** (seed 42; [nonce\_edit.md](results/varwidth/layers/collateral/nonce_edit.md)):

| Variant | Runs edited / with a set | Nonce point | Test efficacy | Nonce floor | Collateral | Copy (exclusive) | Echo | T3 destruction | T3 leak | Renamed | Broken prompts | Median r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NM-CINDER | 10 / 10 | 1 | 0.04 | 0.36 | 0.99 | 0.99 (0.97) | 0.94 | 0.01 | 0.39 | 0.01 | 0.02 | 0.18 |
| NM-free | 9 / 10 | 2 | 0.18 | 0.36 | 0.86 | 0.99 (1.00) | 0.47 | 0.50 | 0.14 | 0.36 | 0.16 | 3.0 |
| NM-EMBER | 8 / 8 | 2 | 0.02 | 0.27 | 0.96 | 0.97 (0.92) | 0.95 | 0.04 | 0.53 | 0.01 | 0.04 | 0.32 |
| NM-free-EMBER | 7 / 8 | 3 | 0.21 | 0.26 | 0.86 | 0.99 (0.98) | 0.34 | 0.59 | 0.14 | 0.47 | 0.15 | 3.2 |
| NM-QA | 10 / 10 | 3 | 0.22 | 0.36 | 0.89 | 1.00 (0.99) | 0.49 | 0.47 | 0.16 | 0.32 | 0.15 | 1.6 |
| E3 EMBER / CINDER / CINDER-z (F38) | 8 / 10 / 10 | – | 0.07 / 0.14 / 0.06 | – | 0.97–0.99 | ≥ 0.9 by the rule | – | – | – | – | – | 0.05–0.25 |
| Collateral-rule EMBER / CINDER-z (F29, F33) | – | – | 0.30 / 0.39 | – | 0.86 | 0.44 / 0.38 | 0.47 / 0.35 | 0.49 / 0.61 | 0.15 / 0.08 | 0.09 / 0.08 | 0.14 / 0.15 | 1 |

"Renamed" is the share of answers to the T3 prompt that drop the word and name a capitalised non-dictionary string (real word 0.00, nonce 0.17). Gambling's NM-free and NM-free-EMBER have no knob passing validation collateral ≥ 0.8. Attainment on substantial floors: Baseball NM-free-EMBER and Heroin NM-QA; the other attainments are COVID-19 (floor 0.00) and Pornography (0.07–0.11).

Per concept (test efficacy / collateral; * = nonce point):

| Concept | Floor (CINDER / EMBER set) | NM-CINDER | NM-free | NM-EMBER | NM-free-EMBER | NM-QA |
|---|---|---|---|---|---|---|
| Baseball | 0.51 / 0.31 | 0.00 / 1.00 | 0.05 / 0.87 | 0.00 / 1.00 | 0.36 / 0.89* | 0.21 / 0.93 |
| Cannabis | 0.52 / 0.49 | 0.15 / 1.00 | 0.34 / 0.90 | 0.12 / 1.00 | 0.31 / 0.83 | 0.37 / 0.90 |
| COVID-19 | 0.00 / 0.00 | 0.00 / 0.98* | 0.00 / 0.87* | 0.00 / 0.97* | 0.11 / 0.82* | 0.00 / 0.97* |
| Gambling | 0.35 / 0.32 | 0.04 / 0.97 | no edit | 0.00 / 0.87 | no edit | 0.00 / 0.91 |
| Golf | 0.36 / 0.25 | 0.00 / 1.00 | 0.25 / 0.82 | 0.00 / 0.94 | 0.18 / 0.92 | 0.25 / 0.74 |
| Gun | 0.37 / 0.31 | 0.00 / 0.99 | 0.17 / 0.80 | 0.00 / 0.86 | 0.14 / 0.86 | 0.24 / 0.79 |
| Halloween | 0.16 / – | 0.00 / 0.98 | 0.16 / 0.79 | – | – | 0.05 / 0.91 |
| Heroin | 0.49 / – | 0.00 / 1.00 | 0.22 / 0.92 | – | – | 0.44 / 0.92* |
| Nazism | – | – | – | – | – | – |
| Pornography | 0.11 / 0.07 | 0.00 / 1.00 | 0.11 / 0.91* | 0.04 / 1.00* | 0.11 / 0.95* | 0.21 / 0.89* |
| Republic of Ireland | – / 0.39 | – | – | 0.04 / 1.00 | 0.24 / 0.77 | – |
| Uranium | 0.69 / – | 0.17 / 1.00 | 0.29 / 0.89 | – | – | 0.46 / 0.90 |

**Predictions.**
- (a) NM-CINDER reaches the nonce point in most runs: **refuted** (1 of 10, on a floor of 0).
- (b) NM-free ≥ NM-CINDER: **supported** (efficacy 0.18 vs 0.04, H3v 0.35 vs 0.08).
- (c) NM-EMBER edits fewer runs and is no better: **supported** (8 vs 10 runs; efficacy 0.02).
- Addendum, NM-free ≥ NM-free-EMBER and edits more runs: more runs, yes (9 vs 7); better per run, no (0.18 vs 0.21).

**Pilot and incidents.**
- **Pilot** (Baseball s42, losses and copy only): the first pilot evaluated with the training hook still active, which applied Δ twice (copy 0.00); fixed before any evaluation. The pilot showed the rank-3 subspace plateau, which led to addendum 1.
- **Learning rate:** Adam makes w_f a weak brake on r (the free edits reach r ≈ 3 even at w_f 0.05). NM-QA's lower learning rate gives median r 1.6.
- **T3 labelling round 1, batches 63–65:** one labeller first counted answers that name a different made-up string as CONFAB or IGNORANT, against the brief. After revising to the brief, 90 labels changed to BLIND, and destruction in those batches rose from 0.12–0.15 to 0.32–0.33.
- **Agents:** T3 labels from 7 agents in 3 rounds (agent per batch group recorded in `judge/nm_round2_agents.json` and the report); generation labels from 4 agents (agreement 1.00 on 88 duplicates). The generation comparison with the subtraction points is approximate: those used each concept's first EMBER seed, which is not 42 for 4 concepts.

**Status.** Done. Post hoc (amendment 13 and addenda), seed 42 only.

### D21. Amendment 14: the confirmatory 2×2 at fixed strength

**Question.** CINDER beats EMBER at equal edit strength (F26). Is that its direction or its token set? The development 2×2 (D6, F6) said direction, on 5 runs.

**Design.** The 4 cells of the existing edit sets (dense Y, as in development): ee = EMBER tokens × EMBER direction, 1b = EMBER tokens × CINDER direction, 1a = CINDER top-N tokens × CINDER direction, cross = CINDER top-N tokens × EMBER direction. ee exists in the 41 expansion runs where EMBER has a set; the other three need a CINDER set too, so the full 2×2 has 38 runs (Nazism 44 and Republic of Ireland 42/43 have ee only; where EMBER has no set, no cell exists). Test pass at fixed r = 0.5 (primary) and r = 1 (secondary), no grid and no selection (`erase_v2 final --rule fixed05 / fixed1`, files `result_fixed05.json`, `result_fixed1.json` in the cell directories). Δ\_dir = mean of (1b − ee) and (1a − cross); Δ\_tok = mean of (cross − ee) and (1a − 1b); unit = concept (mean over seeds), Wilcoxon, 95% cluster bootstrap, Holm over Δ\_dir and Δ\_tok × efficacy and H3v.

**Results at r = 0.5** (from [twobytwo.md](results/varwidth/layers/collateral/twobytwo.md)):

| Cell | Runs | Efficacy | Collateral | Specificity | H3v |
|---|---|---|---|---|---|
| ee (EMBER tokens, EMBER direction) | 41 | 0.201 | 0.934 | 0.987 | 0.389 |
| 1b (EMBER tokens, CINDER direction) | 38 | 0.256 | 0.909 | 0.985 | 0.456 |
| 1a (CINDER tokens, CINDER direction) | 38 | 0.263 | 0.918 | 0.987 | 0.475 |
| cross (CINDER tokens, EMBER direction) | 38 | 0.204 | 0.940 | 0.987 | 0.389 |

| Effect | Metric | Mean | 95% CI | Concepts +/− | p | Holm p |
|---|---|---|---|---|---|---|
| Δ\_dir | efficacy | +0.073 | [+0.012, +0.150] | 10/1 | 0.032 | 0.097 |
| Δ\_dir | H3v | +0.097 | [+0.029, +0.174] | 10/1 | 0.014 | 0.055 |
| Δ\_dir | collateral | −0.019 | [−0.035, +0.001] | 1/10 | 0.054 | – |
| Δ\_tok | efficacy | +0.016 | [−0.001, +0.039] | 8/2 | 0.160 | 0.203 |
| Δ\_tok | H3v | +0.024 | [+0.002, +0.049] | 8/3 | 0.102 | 0.203 |
| Δ\_tok | collateral | +0.004 | [−0.007, +0.018] | 6/5 | 0.831 | – |

Single contrasts (efficacy): 1b − ee +0.082 (p 0.004), 1a − cross +0.065 (p 0.042), cross − ee +0.025 (p 0.16), 1a − 1b +0.007 (p 0.43).

**Verdicts on the predictions.** Δ\_dir > 0: supported in direction and nominally (10 of 11 concepts, p 0.014–0.032), not after Holm (0.055–0.097), and smaller than predicted (+0.07 vs +0.10 to +0.15). Δ\_tok ≈ 0: supported (+0.016, n.s.; predicted −0.01 to −0.03). Collateral cost of the direction: supported (−0.019; predicted −0.01 to −0.05).

**Results at r = 1 (secondary):**

| Cell | Runs | Efficacy | Collateral | Specificity | H3v |
|---|---|---|---|---|---|
| ee | 41 | 0.274 | 0.858 | 0.984 | 0.465 |
| 1b | 38 | 0.267 | 0.850 | 0.978 | 0.446 |
| 1a | 38 | 0.289 | 0.845 | 0.973 | 0.480 |
| cross | 38 | 0.280 | 0.876 | 0.980 | 0.457 |

Δ\_dir: efficacy −0.016 [−0.072, +0.027] (6/5, p 0.97), H3v −0.007 (p 1.00), collateral −0.013 (p 0.18). Δ\_tok: efficacy +0.010 (p 0.56), H3v +0.012 (p 0.97). Development predicted Δ\_dir +0.09 at r = 1 (F6); not replicated.

**Reading.** At small edits (r = 0.5), CINDER's direction removes more concept per unit of change on either token set, and its token set adds little. At r = 1 the four cells are indistinguishable: once edits are large enough to destroy many words, the direction stops mattering. With F27 (coverage) this gives CINDER two separable advantages: a better direction in the small-edit regime, and finding tokens where EMBER finds none.

**Status.** Done (primary and secondary). Post hoc (amendment 14). No Gemini.

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
| L13, `sub:cinder_rs` | CINDER-RS (CS + RS-1% + TWJ, no GS; amendment 5) |
| A3z, `zo_set` | CINDER's selection with Z re-solved in the original space, Z\_o = A Yᵀ(YYᵀ + λI)⁻¹ (amendment 7, D16); A3 is the multilingual ML-CINDER |
| `sub:cinder_z` (fit tag `…z`) | CINDER-z (A3z on CINDER, rung L123) |
| `sub:cinder_rs_z` | CINDER-RS-z (A3z on CINDER-RS, rung L13) |
| `--rule paper`, `select_paper` | Paper-rule arm, the old evaluation: max H2 subject to MMLU-frac ≥ 0.9, no collateral |
| `result_paper.json` | Test result at the paper-rule point (no Alpaca) |
| `<variant>@paper`, `<variant>@sel`, `<variant>@<r>` | E2 config at the paper-rule point, at the collateral-rule point, at a sweep r |
| read strength | Layer-0 read strength of the edit direction relative to random unit directions (offline check, R13) |
| `sub:ee` / `sub:1b` / `sub:1a` / `sub:cross` | 2×2 cells (dense Y) |
| `mean:<set>`, `noise:<set>` | Mean, Noise (§4.2) |
| `proj:<set>`, `excl:<set>`, `pow1.5:<set>`, `pow2:<set>` | Bounded projection, exclusivity-weighted, power |
| `param`, `r` | δ or α; achieved median relative perturbation |
| `collateral`, `coll_all`, `coll_tok` | Primary collateral, union-pool collateral, per-token accuracy |
| `H2` / `H3v` | HM(efficacy, specificity) / selection score HM(efficacy, specificity, collateral) |
| `H`, `H4` | The paper's H; H4 |
| codes = 1 | Concept-only tokens |
| step 0 / premise | Premise check (D3) |
| E1 | The collateral-aware erasure stage (D4–D9) |
| E2, `integrity.py` | Token integrity (D11–D14) |
| `ref_orig` / `ref_nonce` / `ref_del` | E2 references: real word / nonce word / word deleted |
| `copy`, `ctx_pass`, `del_likeness`, `ctxmc`, `echo`, `static_*` | T1, T2, T2, T2b, T3 echo, T4 |
| `ctx_excess` | Answer-NLL excess over the nonce (secondary T2 diagnostic) |
| KC, KO, IGN, INCOH | T3 labels KNOWS\_CONCEPT, KNOWS\_OTHER, IGNORANT, INCOHERENT |
| F1–F45, S1–S9 | Findings and speculations of Part 1 (F45 coverage split, post hoc; F24–F29 confirmatory, amendment 8; F36–F38 amendment 11; F39–F43 amendment 13; F44 amendment 14; F30–F35 last-mile, amendments 9–10) |
| `result_fixed1.json`, `--rule fixed1`, `confirm_stats 2x2` | 2×2 test pass at fixed r = 1; the amendment 14 analysis (D21) |
| `last_mile.py`, `result_g0.7.json` / `result_g0.9.json`, `alpaca_{sel,paper}.json` | Last-mile checks (D18): floor, generations, gate sensitivity, seed-42 Alpaca |
| `result_fixed05.json`, `confirm_stats.py` | Test result at fixed r = 0.5; the amendment 8 analysis (D17) |
| `collateral_rn`, `H3v (rn)`, `H4 (rn)` | Test collateral renormalized by the unedited test accuracy, and the scores built on it (amendment 6) |

### Open items

- [x] Select one version per concept, clean up, freeze `data/collateral_questions.json`, and put the hash in PREREG amendment 3.
- [ ] Human spot check of 20 items per concept.
- [x] Phase 1 and phase 2 grids for all variants, concepts and seeds.
- [x] Selection, test pass and Alpaca for the primaries (3 runs flagged for fluency < 0.9, R1).
- [x] Choose m and α pooled over the 3 concepts: α = 1; best shape power m = 2 on CINDER's tokens ([shape\_choice.json](results/varwidth/layers/collateral/shape_choice.json)).
- [x] Fill R1–R7 and write `collateral/poc.md`.
- [x] E2: pilot checks (two T2 fixes, amendments 4b and 4c), full judge-free runs on 8 concept-seeds, agent labels on 3 (3,001 items), fill R8–R10 and R12.
- [x] CINDER-RS (amendment 5): E1 grids, test pass and Alpaca; E2 tests and incremental T3 labels (640 items); R1, R2, R5, R6, R9, R9b, R10 and D15 filled.
- [x] R11: split the integrity metrics by token class from the per-word outputs (no new runs).
- [x] Find why the unedited model misses some test collateral items, and report test collateral against that ceiling: resolved by amendment 6 (the base filter used the validation option order; test collateral is renormalized, D4).
- [x] A3z (amendment 7): offline check, E1 grids, test pass and Alpaca for CINDER-z and CINDER-RS-z; the paper-rule arm (`result_paper.json`) for 6 variants; E2 judge-free tests and 2,242 incremental T3 labels (batches 26–40) at the selected, sweep and paper-rule points; D16, F19–F23 and S9 written.
- [x] Commit the script behind `diag/a3z_offline.json`: `python -m experiments.varwidth.collateral_diag a3z`.
- [ ] Proposed (S9): the matched-read-strength random-direction control at r = 10–100. Needs approval: new edits and grid points.
- [ ] Re-check the 17 mangled-copy CINDER-RS answers labelled BLIND against the earlier labellers' CONFAB convention (R9b).
- [ ] Decide whether echo stays a reported test (nonce echo below the 0.8 pilot gate in 6 of 8 runs, R8).
- [ ] Proposed E3 (S2): integrity-gated selection on the full r grid for EMBER, CINDER, 1b and CINDER-RS. Needs approval: it adds E2's judge-free tests at every grid point.
- [ ] Proposed: an identity-preserving edit variant (S1), the nonce-matching T3 score (S5, from existing labels), and OE or nonce-controlled efficacy (S6, S8).
- [x] µ and cos(c, e) medians of D1 saved: `python -m experiments.varwidth.collateral_diag mu` → [diag/mu\_cos.json](results/varwidth/layers/collateral/diag/mu_cos.json).
- [x] Expansion list: the 12 anchor concepts not used for development (the 14 anchor concepts minus Harry Potter and Valentine's Day; Ancient Rome is not an anchor concept). The plan's "11" was a miscount.
- [ ] Decide whether to re-run the premise prompts at the selected points.
- [ ] Merge into CINDER_improvements.md, with the reframed claim of R7, the overall reading of R12 and the A3z reading of D16 (the Gemma side of "whiten to select, edit in the original space").
- [x] The 12 expansion concepts (S7; amendment 8, D17, F24–F29).
- [ ] LoRA relearning at r ≈ 0.5 vs r = 100 (S4).
- [x] Edit set checked against the code: `edit_vectors` in [erase.py](experiments/varwidth/erase.py) edits support ∩ codes = 1, as in §3.3.
- [x] Variant definitions checked against [erase\_v2.py](experiments/varwidth/erase_v2.py) and [collateral.py](experiments/varwidth/collateral.py).

### Sources of the numbers

- Anchor δ choices, validation curves and gate counts: [results/varwidth/layers/erasure/](results/varwidth/layers/erasure/) (`<concept>/<rung>_s42/{grid.json, result.json}`; the paper-grid run is `Harry_Potter/L0_s42_papergrid`).
- Anchor summary and fidelity: [results/varwidth/layers/SUMMARY.md](results/varwidth/layers/SUMMARY.md), [results/varwidth/layers/erasure.md](results/varwidth/layers/erasure.md).
- Similar-domain questions: [data/mc\_questions.json](data/mc_questions.json).
- Premise check: [premise\_prompts.json](results/varwidth/layers/collateral/premise_prompts.json), [premise\_generations.json](results/varwidth/layers/collateral/premise_generations.json).
- Pilot: [collateral/Harry\_Potter/\_pilot\_s42\_v0metric/](results/varwidth/layers/collateral/Harry_Potter/_pilot_s42_v0metric/) (grids record the union collateral; the own-token collateral of the pilot table is recomputed from the saved per-token accuracies by `python -m experiments.varwidth.collateral_diag pilot` → [diag/pilot\_own\_tokens.json](results/varwidth/layers/collateral/diag/pilot_own_tokens.json)), [cl\_pilot\_1055037.out](results/varwidth/layers/slurm/logs/cl_pilot_1055037.out).
- Token pools and drafts: [collateral/pool/](results/varwidth/layers/collateral/pool/), [collateral/drafts/](results/varwidth/layers/collateral/drafts/).
- Pre-registration: [PREREG\_amend3\_draft.md](results/varwidth/layers/collateral/PREREG_amend3_draft.md), [PREREG.md](results/varwidth/layers/PREREG.md) (amendment 3: E1; 4, 4b, 4c: E2 and its pilot fixes; 5: CINDER-RS; 6: the test-collateral erratum; 7: A3z and the paper-rule arm; 8: the confirmatory expansion).
- Edit magnitudes µ and cos(c, e): [diag/mu\_cos.json](results/varwidth/layers/collateral/diag/mu_cos.json).
- E1 results: [collateral/poc.md](results/varwidth/layers/collateral/poc.md) (written by `python -m experiments.varwidth.collateral_report report`); per run `collateral/<concept>/s<seed>/<variant>/{grid.json, result.json}`; Pareto PNGs in [collateral/pareto/](results/varwidth/layers/collateral/pareto/); shape choice in [shape\_choice.json](results/varwidth/layers/collateral/shape_choice.json). Edit sets and the 2×2 top-N overlap (`_info.topN_overlap_rawY`): `collateral/<concept>/edits_s<seed>.pt`.
- E2 results: [integrity/report.md](results/varwidth/layers/collateral/integrity/report.md) (written by `python -m experiments.varwidth.integrity report`); per config `integrity/<concept>/s<seed>/*.json`, with the words, nonces and tokens in `meta.json`; the pilots in `integrity/Harry_Potter/_pilot1_s42/` and `_pilot2_s42/`.
- A3z (D16): the offline check [diag/a3z\_offline.json](results/varwidth/layers/collateral/diag/a3z_offline.json) (per run and edit: median cos(c, e), median µ, read strength, and for A3z the cosine to the parent's edit; script: `python -m experiments.varwidth.collateral_diag a3z`); per run `collateral/<concept>/s<seed>/sub_cinder_z/` and `sub_cinder_rs_z/` (`grid.json`, `result.json`); the paper-rule arm in `result_paper.json` next to `result.json` for `sub_ember`, `sub_cinder`, `sub_cinder_rs`, `sub_cinder_z`, `sub_cinder_rs_z` and `sub_1b`, summarized in poc.md ("Old evaluation: the paper-style rule" and "New evaluation, paired vs EMBER"). R14's matched-r means and gate sensitivity, R15's per-run H2 and R16's paired differences against the parent are computed from the same files with the rules of collateral\_report.py. The Llama diagnosis is on branch `orip-ml`: `results/ml/diagnose_a3/DIAGNOSIS.md`.
- Confirmatory 2×2 (D21): [twobytwo.md](results/varwidth/layers/collateral/twobytwo.md) and `twobytwo.json` (written by `python -m experiments.varwidth.confirm_stats 2x2`); per run `collateral/<concept>/s<seed>/sub_{ee,1b,1a,cross}/result_fixed05.json` (and `result_fixed1.json`).
- Confirmatory expansion (D17): [confirm.md](results/varwidth/layers/collateral/confirm.md) and `confirm.json` (written by `python -m experiments.varwidth.confirm_stats`); question set `data/collateral_questions_expansion.json` with its trail in `collateral/drafts/<concept>/{v0,F0,v1,F1}.json`, `final_chosen_expansion.json` and `base_answers_expansion.json`; per run `collateral/<concept>/s<seed>/<variant>/{grid.json, result.json, result_fixed05.json, result_paper.json}` and `integrity/<concept>/s<seed>/`; T3 batches 41–57.
- Amendment 11 (D19): word-free questions and inputs in `last_mile/wordfree_draft/` and `wordfree_inputs/` (brief `briefs/wordfree_writer.md`), `wordfree.json` in the run directories, [template\_check.json](results/varwidth/layers/collateral/last_mile/template_check.json), `copygrid.json` and `result_e3.json` in the run directories, all summarised in last\_mile.md ("Word-free efficacy", "E3", "Context damage") by `python -m experiments.varwidth.last_mile report`.
- Last-mile checks (D18): [last\_mile.md](results/varwidth/layers/collateral/last_mile.md) and `last_mile.json` (`python -m experiments.varwidth.last_mile report`); the sensitivity rows in confirm.md; per concept `last_mile/<concept>/{floor.json, gen.json}`; prompts in `last_mile/prompts.json` (with `prompts_meta.json` and the writers' inputs in `prompt_inputs/`); blind labels in `last_mile/judge/`; briefs `briefs/premise_writer.md` and `briefs/premise_judge.md`.
- Gemini spend: [results/gemini\_usage.jsonl](results/gemini_usage.jsonl) (37.5 NIS in total: 8.3 for amendment 10 and 5.7 for amendment 12, after user-approved extensions of the 25 NIS stop).
- T3 labels: [integrity/judge/](results/varwidth/layers/collateral/integrity/judge/): blind batches `batch_*.json`, agent labels `labels_*.json` (5,883 items plus 577 duplicate entries under 568 distinct ids, 473 of them labelled by a different agent than the original; batches 21–25 are CINDER-RS, batches 26–40 amendment 7), the id-to-config key `key.json`, and `pre_clarification/` (the first labels of batches 00–02, superseded after the brief's clarifications; 347 of 450 unchanged). Brief: [briefs/judge\_t3.md](results/varwidth/layers/collateral/briefs/judge_t3.md).
- Code: [erase.py](experiments/varwidth/erase.py), [erase\_v2.py](experiments/varwidth/erase_v2.py), [collateral.py](experiments/varwidth/collateral.py), [collateral\_diag.py](experiments/varwidth/collateral_diag.py), [collateral\_report.py](experiments/varwidth/collateral_report.py), [integrity.py](experiments/varwidth/integrity.py), [nonce\_edit.py](experiments/varwidth/nonce_edit.py), [confirm\_stats.py](experiments/varwidth/confirm_stats.py), [last\_mile.py](experiments/varwidth/last_mile.py).

### References

- Belrose et al. LEACE: Perfect linear concept erasure in closed form. NeurIPS 2023.
- Deeb and Roger. Do unlearning methods remove information from language model weights? arXiv:2410.08827, 2024.
- Eldan and Russinovich. Who's Harry Potter? Approximate unlearning in LLMs. arXiv:2310.02238, 2023.
- Hendrycks et al. Measuring massive multitask language understanding. ICLR 2021.
- Lynch et al. Eight methods to evaluate robust unlearning in LLMs. arXiv:2402.16835, 2024.
- Suslik, Shafran and Geva. Don't forget your embeddings: robust knowledge erasure via precise editing of embeddings (EMBER). arXiv:2606.03695, 2026.
- Taori et al. Stanford Alpaca: an instruction-following LLaMA model. 2023.
