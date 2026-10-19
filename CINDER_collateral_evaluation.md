# Token destruction: collateral-aware evaluation of embedding erasure

Oct 19, 2026 · @Ori

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
- **Destruction blurs the method comparison.** Once a token is replaced, the subtracted direction barely matters. In the pilot at r = 10–100, Noise of the same magnitude reached efficacy 0.39, the same as an edit along CINDER's direction. Over all runs it holds for CINDER's direction on EMBER's tokens (1b 0.25–0.28 vs Noise 0.29–0.31 at r ≥ 10, validation), but EMBER's own direction stays well above Noise (0.44–0.52; F3, S8). The paper's Table 1 argument, that direction matters (Gemma MC Con: EMBER 49.4 vs Mean 62.8 and Noise 62.2, unedited 73.3), was made at the δ the paper's tuning selects, which our Observation 1 suggests is in this regime.
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
  - **Verdict: supported** (validation; F1, R2). Without collateral, EMBER's selected r is 10–100 in all 6 EMBER runs. With it, r is 0.05–2.
- **Hyp. 2: the comparison changes.** In the subtraction regime the direction matters, so the EMBER vs CINDER comparison differs from the anchor. The pilot hints at this: at r = 1, an edit along CINDER's direction on EMBER's tokens reached efficacy 0.42, against 0.28 for EMBER.
  - **Verdict: partly** (F2, F3, F13, F18). At matched small r, CINDER's direction erases more (validation efficacy at r = 0.25: 0.26 vs 0.05) and removes more knowledge with little destruction. At each method's selected point the two tie (test H4 0.48 vs 0.48). The pilot's gap at r = 1 (0.42 vs 0.28) shrinks over all runs (1b 0.31 vs EMBER 0.31; 0.31 vs 0.25 on the 5 matched runs); the main gap sits at r = 0.25–0.5. CINDER-RS, CINDER's direction without GS, has the best mean test H4 (0.53), within noise.
- **Hyp. 3: the 2×2 separates the two effects.** Changing only the token set and changing only the direction have separate, measurable effects on efficacy and collateral.
  - **Verdict: partly** (F6, R3). At fixed r the direction raises efficacy (Δ\_dir +0.15 at r = 0.25, +0.09 at r = 1; validation, 5 runs) and the token set changes it little (Δ\_tok −0.01 to −0.03). At the selected points the effects are heterogeneous and inconclusive (test H3v: Δ\_dir +0.06, Δ\_tok −0.08).
- **Hyp. 4: shaped edits trade better.** Bounded projection and exclusivity-weighted edits reach a better efficacy–collateral trade-off than e − δc.
  - **Verdict: not supported at this sample size** (F2, F5, R4). Exclusivity weighting on EMBER's tokens has the best mean test H3v (0.50 vs 0.42 for EMBER), within noise. Bounded projection is worse (0.30 and 0.39). The pre-registered best shape (power m = 2 on CINDER's tokens) scored 0.37 on test.
- **Expected limit: context blindness.** An embedding edit sees the token, not its context. " Harry" in Prince Harry and in Harry Potter is the same row, so no input-embedding edit can erase one and keep the other. The Pareto curves, not a single score, are the honest output.

### Planned experiments

1. **Collateral question set** for the 3 development concepts (Harry Potter, Ancient Rome, Valentine's Day), written and reviewed blind to the methods, checked by script, and frozen with a hash before any edited model is scored on it.
2. **Grids** for all variants on r ∈ {0.05, …, 100}, with validation scores at every point (efficacy, specificity, collateral).
3. **Selection** by the collateral-aware cheaper rule, then the test split and Alpaca at the chosen point for the primary variants.
4. **Report** H, H4 and the paper-structured variant, the 2×2 effects, the edit-shape comparison, Pareto curves and gate sensitivity.
5. **E2, token integrity** (D11–D14): copy, context, context-MC, ignorance and static tests on every edited token, exclusive ones included, at E1's selected points and on a regime sweep, against the real-word, nonce and deletion references.
6. **After this stage, with the team:** expand to the other erasure concepts, and test relearning or LoRA healing.

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
  - **Verdict: partly** (F9, F10, F11, F14, F17). At r ≥ 10 the word fails every token test for EMBER, CINDER and CINDER-RS: copy 0, T3 destruction 0.98–1.00, every edited row a norm outlier. The context mostly survives: same-sentence recall stays ≈ 1.00, and only EMBER lowers context-MC. The boundary is lower than predicted: at r = 1, copy is already 0.45–0.52 and destruction 0.52–0.57; only r = 0.25 stays near the nonce reference. Exclusive tokens fail first: at r = 1, 0.85–0.96 of the exclusive words are destroyed, against 0.46–0.49 of the multi-meaning ones (R11).
- **Hyp. 6: collateral is a partial proxy.** E1's collateral-aware selection lands at points with high integrity, but not always, so integrity needs its own axis.
  - **Verdict: supported** (F12, F17, R9, R11). Every E1-selected point passed collateral ≥ 0.8 on validation. Yet T3 destruction there is 0.18–0.30 for EMBER, CINDER, CINDER-RS, 1a and 1b, and 0.56 for power m = 2. Example: Harry Potter 42, EMBER at r\* = 2 (validation collateral 0.81): copy 0.03, destruction 0.83. The exclusive words, which collateral cannot see, fare worst: 0.37–0.75 of them are destroyed at the selected points, against 0.09–0.19 of the multi-meaning words (0.52 for power m = 2).
- **Hyp. 7: graceful ignorance trades against leakage.** At small r the model reads the word and may still know the concept. The interesting question is whether any variant reaches low leakage with high integrity.
  - **Verdict: supported** (F13, R9b). No labelled config reaches leak ≤ 0.1 with destruction ≤ 0.1. For EMBER, leak falls only as destruction rises (r = 0.25: leak 0.58, destruction 0.00; r = 1: 0.07 and 0.57). The best trade measured is CINDER at r = 0.25 (leak 0.25, destruction 0.07), then CINDER-RS at r = 0.25 (0.29 and 0.09) and 1a at its selected points (0.12 and 0.18).
- **Hyp. 8: Mean and Noise are destructive by construction**, at any strength where they erase.
  - **Verdict: supported for Mean, partly for Noise** (F7, F15, R9). Mean deletes the word at its only point (copy 0, destruction 0.99) and fails E1's collateral gate in every run. Noise at its selected points (median r = 1) destroys 0.25–0.35 of the words while still leaking 0.29, and 0.60–0.64 of its rows are norm outliers.

**How it runs.** E2 reuses E1's frozen grids and selected points; it adds no new search. It runs judge-free on all 8 concept-seeds and, for T3 labels, on one seed per concept, with Claude agent judges (no Gemini). Details in D11–D14.

### Findings (backed)

E1 on 8 concept-seeds (D9), E2 on the same 8 with T3 labels on 3 (D14). Means are over concept-seeds; T3 numbers come from the 3 judge runs. Test collateral, and the H3v and H4 built on it, are renormalized by the unedited model's test accuracy (amendment 6, D4).

- **F1. Regime shift (Hyp. 1).** On validation, the paper-style rule (H2, no collateral) picks EMBER at r = 10–100 in all 6 EMBER runs, and EMBER + TJ∨WJ+share at r = 100 in its one separate run. The collateral rule picks r = 0.05–2. Example, Valentine's Day 42, EMBER: r 100 → 0.5, efficacy 0.44 → 0.24, collateral 0.65 → 0.94. For CINDER the two rules agree in 4 of 7 runs (r = 0.5 or 5). In the other 3 the H2 rule picks r = 10–100. For CINDER-RS the H2 rule picks r = 10 only on Harry Potter (both seeds, → 0.5), and r = 0.5–2 elsewhere (R2).
- **F2. All variants land in a narrow band.** Under the collateral rule, test efficacy is 0.14–0.27 and collateral 0.86–0.96 across variants (Mean and the single EMBER + TJ∨WJ+share run excluded). Mean test H3v: exclusivity-weighted (EMBER's tokens) 0.50; exclusivity-weighted (CINDER's tokens) and power m = 1.5 and 2 (EMBER's tokens) 0.47; CINDER-RS and 1b 0.46; EMBER 0.42; CINDER 0.42. H4 for the primaries with Alpaca: CINDER-RS 0.53, 1b 0.52, EMBER 0.48, CINDER 0.48, power m = 2 (CINDER's tokens) 0.43, 1a 0.39. The differences are within noise: 5–8 runs per variant, 50 test questions per run.
- **F3. Direction vs magnitude (matched r, validation means; R5).** Efficacy at r = 0.25: CINDER 0.26, CINDER-RS 0.21, 1b 0.20, EMBER 0.05. At r = 0.5: 0.33, 0.37, 0.33, 0.20; CINDER-RS's 0.37 is the highest of any variant at r = 0.5. At r = 1: 0.35, 0.34, 0.31, 0.31. At r ≥ 2 EMBER is higher: 0.38 at r = 2, rising to 0.52 at r = 100, against 0.32–0.36 for CINDER, 0.33–0.34 for CINDER-RS and 0.25–0.30 for 1b. On the 5 runs with a full 2×2 the same holds at fixed tokens: ee 0.05, 0.21, 0.25 vs 1b 0.20, 0.33, 0.31 at r = 0.25, 0.5, 1. CINDER's direction is more efficient per unit of perturbation in the subtraction regime. EMBER gains more from magnitude in the destruction regime, and not from magnitude alone: Noise on EMBER's tokens stays at 0.28–0.31 for r ≥ 2 (S8).
- **F4. CINDER's token sets are its liability.** CINDER's GS-widened edit sets are larger in 4 of the 5 runs shared with EMBER and hold a higher share of multi-meaning tokens in all 5: Ancient Rome 44, 82 of 103 edited tokens vs 26 of 38 for EMBER; Valentine's Day 46, 30 of 32 vs 24 of 34. Its validation collateral falls faster with r (0.93 vs 0.98 at r = 0.25, 0.88 vs 0.96 at r = 0.5), so the gate binds earlier. In Valentine's Day 43 and 46, CINDER selected r = 5, just above the gate on validation (collateral 0.81 and 0.80), and fell to 0.68 and 0.72 on test (H3v 0.19 each). CINDER-RS, without GS, has smaller sets on Ancient Rome but a collateral curve no better than CINDER's (D15), so GS is not the whole token-set problem.
- **F5. Winner's curse.** The validation-best edit shape (power m = 2 on CINDER's tokens, pooled validation H3v 0.58) scored 0.37 on test. CINDER's validation lead at the selected point (H3v 0.61 vs 0.45 for EMBER) also vanished on test (0.42 vs 0.42). CINDER-RS, with a lower validation score (0.49), kept most of it on test (0.46).
- **F6. The 2×2 is inconclusive at the selected points and separates at fixed r (R3).** At the selected points, the mean test H3v effects are Δ\_dir +0.06 and Δ\_tok −0.08, with signs that differ across runs (Δ\_dir from −0.06 to +0.34). At fixed r on validation, the direction raises efficacy in 5 of 5 runs at r = 0.25 (mean +0.15) and in 4 of 5 at r = 1 (mean +0.09, the fifth 0.00). The token set changes efficacy by −0.01 to −0.03. The direction's collateral cost is −0.01 to −0.05.
- **F7. Mean fails the gate; Noise is weak.** Mean's single point (r ≈ 0.85–0.95) has validation collateral 0.71 (0.55–0.78), so no Mean edit passes the 0.8 gate in any run. Noise reaches test H3v 0.34 (CINDER's tokens) and 0.40 (EMBER's tokens).
- **F8. E2's references validate the tests (R8).** Copy is 1.00 for orig and nonce and 0.00 for del in every run. Same-sentence recall is 1.00 for nonce and del and 0.99–1.00 for orig in every run, so T2 is valid in all 8. T3 labels: orig leak 0.51 and graceful 0.45 (KNOWS\_OTHER 0.43); nonce mostly CONFAB (0.62) and IGNORANT (0.29), destruction 0.06; del 100% BLIND. On 284 duplicates labelled by different agents, agreement is 0.98 on the exact label and 0.99 on destruction yes/no.
- **F9. Readability collapses between r = 0.25 and r = 1 (R10).** Copy: EMBER 0.98 → 0.45 → 0 (r = 0.25 → 1 → ≥ 10), CINDER 0.95 → 0.52 → 0, CINDER-RS 0.94 → 0.50 → 0. T3 destruction: EMBER 0.00 → 0.57 → 1.00, CINDER 0.07 → 0.52 → 0.98–0.99, CINDER-RS 0.09 → 0.56 → 0.99–1.00.
- **F10. Concept-exclusive tokens go first (pilot, R11).** In the pilot (Harry Potter 42, EMBER r = 1), 14 of 30 words were dropped from the copy. All were exclusive or carried a concept word (Hogwarts, Voldemort, Snape, Weasley…); multi-meaning words copied fine. The reason: r is a median, and exclusive tokens are the high-µ tail. The full split confirms it: at r = 1, copy is 0.13–0.22 for exclusive words and 0.52–0.55 for multi-meaning ones, and T3 destruction 0.85–0.96 vs 0.46–0.49 (EMBER, CINDER, CINDER-RS).
- **F11. Destruction is local (R10).** Same-sentence recall stays ≈ 1.00 at every r. Context-MC drops only for EMBER at r ≥ 10 (0.54 → 0.45 and 0.38). The answer-NLL excess over the nonce is +1.1–1.2 nats for EMBER at r ≥ 10, against +0.12–0.24 for CINDER and +0.11–0.19 for CINDER-RS. "Destroys the whole context" is mostly not supported; "deletes the word" is.
- **F12. Collateral is a partial proxy for integrity (Hyp. 6; R9).** At E1's selected points, T3 destruction and leak are: EMBER 0.30 and 0.31, CINDER 0.21 and 0.15, CINDER-RS 0.19 and 0.22, 1a 0.18 and 0.12, 1b 0.21 and 0.18, power m = 2 0.56 and 0.08, Noise 0.25–0.35 and 0.29, Mean 0.99 and 0.00. Copy (all runs): EMBER 0.65, CINDER 0.64, CINDER-RS 0.75, 1b 0.80, 1a 0.65. All these points except Mean's passed the collateral gate. EMBER's 0.30 is driven by Harry Potter 42, where it sits at r = 2 (destruction 0.83); its other two judge runs give 0.00 and 0.20.
- **F13. CINDER's direction removes knowledge without destruction at r = 0.25.** CINDER at r = 0.25: leak 0.25, destruction 0.07, CONFAB 0.17. EMBER at r = 0.25: leak 0.58, destruction 0.00, CONFAB 0.03. CINDER-edited words drift toward unknown-word behaviour (confabulation, as for the nonce). EMBER-edited words at r = 1 go blind instead (destruction 0.57, mostly BLIND). The token sets differ, but on EMBER's own tokens CINDER's direction (1b at its selected points) also confabulates more (0.16) than EMBER at r = 0.25 or 1 (0.03, 0.08). CINDER-RS at r = 0.25 behaves like CINDER (leak 0.29, destruction 0.09, CONFAB 0.15). This is the clearest evidence for CINDER's direction. It is not free: at r = 0.25 CINDER and CINDER-RS already destroy 7 of the 9 exclusive Valentine's Day 42 words, while EMBER destroys none (R11).
- **F14. Static fingerprint (R9, R10).** At r ≥ 10 every edited row is a norm outlier (z > 3 share 1.00). At the selected points the share is EMBER 0.14, CINDER 0.29, CINDER-RS 0.19, 1b 0.11.
- **F15. Mean behaves like deletion without harming the context.** Copy 0, echo 0, T3 destruction 0.99 (BLIND 0.96–0.97), while same-sentence recall stays 0.98–0.99 and context-MC 0.52–0.54.
- **F16. Two E2 context designs failed validity.** A long continuation was insensitive, and raw "Answer:" NLL was poorly calibrated, before chat-generation recall passed (amendments 4b and 4c, D12).
- **F17. The collateral term hides exclusive-token destruction (Hyp. 5, 6; R11).** At E1's selected points EMBER destroys 0.74 of its exclusive words (copy 0.36) and 0.19 of its multi-meaning ones; CINDER-RS 0.65 vs 0.09, 1b 0.43 vs 0.16, 1a 0.46 vs 0.10. At r = 1 each swept method (EMBER, CINDER, CINDER-RS) destroys 0.85–0.96 of the exclusive words and 0.46–0.49 of the multi-meaning ones. The collateral gate sees only multi-meaning tokens, so it admits these points. CINDER's selected points halve the exclusive figure (0.37), mainly because CINDER is selected at a smaller r: on Harry Potter 42 it sits at r = 0.25 and destroys 1 of 13 exclusive words, where EMBER sits at r = 2 and destroys 11 of 11. At the same r = 0.5 on Valentine's Day 42, both destroy most of them (8 of 9, 6 of 8). The judged counts are small (23–29 exclusive words per config).
- **F18. CINDER-RS is the most robust candidate so far, within noise (amendment 5; R1, R5, R9, D15).** It is the only variant with an edit in all 8 runs, Ancient Rome 43 included, where CINDER finds no feature. Test: efficacy 0.24, collateral 0.90, H3v 0.46, H4 0.53 (1b 0.52, EMBER 0.48, CINDER 0.48), mean selected r 1.03. Paired on shared runs (test H3v): +0.09 vs EMBER (2 wins, 2 losses, 2 ties; the mean comes from Ancient Rome 44, +0.51, where EMBER's selection gave no usable edit), +0.05 vs CINDER (3 wins, 2 losses, 2 ties; from Valentine's Day 46, +0.46, where CINDER erased little), +0.08 vs 1b (4 wins, 1 loss). So it is robustness, avoiding each method's failure runs, rather than a consistent per-run gain, and a tendency only (5–7 paired runs). Part of it is a destructive point: on Valentine's Day 46 CINDER-RS sits at r = 5, with copy 0.00, every edited row a norm outlier, test collateral 0.73 and Alpaca fluency 0.88. In E2 at its selected points: copy 0.75, leak 0.22, destruction 0.19, graceful 0.48; multi-meaning words destroyed 0.09 (the lowest), exclusive words 0.65.

### Speculations (to be tested)

Each item is a speculation, not a result.

- **S1. For concept-exclusive tokens, complete and graceful erasure may be incompatible for any input-embedding edit.** Their embedding is mostly the concept, so removing the concept removes the token's identity, and copy fails (F9, F10). Every method destroys 0.37–0.75 of its exclusive words at its selected point (F17); this is the open problem. An identity-preserving edit could separate "unknown" from "deleted": keep the component that lets the model copy the token, for example by removing the concept part only orthogonally to the token's own unembedding or identity direction. **Test:** such a variant on the same E1 and E2 settings.
- **S2. CINDER's direction advantage lives in the subtraction regime, so any evaluation that forbids destruction should favour it** (F3, F6, F13). **Test (proposed E3, not launched):** integrity-gated selection (copy ≥ 0.9 and collateral ≥ 0.8) on the full r grid for EMBER, CINDER, 1b and CINDER-RS. It needs E2's judge-free tests at every grid point. The copy gate should hold on the exclusive words separately, since the pooled copy hides them (F17).
- **S3. CINDER's token-set liability comes from the concept-only label, amplified by GS widening** (F4). The label admits generic words, and GS adds more of them. CINDER-RS tested the GS part (D15): without GS the Ancient Rome sets shrink and fewer edited tokens are multi-meaning, but validation collateral at matched r is no better than CINDER's. So GS explains the set sizes and some failure runs, not the faster collateral loss. **Test:** an exclusivity filter on the edit set, for the label part.
- **S4. The paper's relearning robustness at δ = 200 may come from destruction.** Rows 40–120× the token's norm cannot be moved back by small fine-tuning steps. **Test:** LoRA relearning (§C.6) at r ≈ 0.5 vs r = 100.
- **S5. For a small model, graceful ignorance means behaving like the nonce, not saying "I don't know".** Gemma-2-2B confabulates on unknown words (nonce: 62% CONFAB, 29% IGNORANT). **Test:** a distribution-matching T3 score, the distance between a config's label distribution and the nonce's, recomputed from the existing labels.
- **S6. MC efficacy in the destruction regime partly measures unreadable questions, so it overstates knowledge removal.** **Test:** OE efficacy, or efficacy controlled by the nonce (the same questions with the word replaced by its nonce, on the unedited model, as the floor).
- **S7. Method differences below about 0.05–0.08 cannot be resolved with 3 concepts × 2–3 seeds and 50-question tests** (F2, F5, F6, F18). CINDER-RS's lead is one example: its paired means against EMBER and CINDER each come from a single run. **Test:** the 12 expansion concepts, before any verdict on CINDER.
- **S8. EMBER's large-r efficacy may partly be global disruption, not erasure.** At r ≥ 10, EMBER is the only method that lowers context-MC and raises the answer NLL (F11), and it beats Noise at the same r on the same tokens (0.52 vs 0.31 at r = 100, F3). Its rows are huge and share few directions (one feature on Harry Potter, D1). The same disruption could lower MC accuracy on concept questions without removing knowledge. **Test:** per run, relate EMBER's large-r efficacy gain to its context-MC drop, and score efficacy on OE or nonce-controlled questions (S6).

### CINDER-RS: the direction without the widths (amendment 5)

CINDER stacks three layers: Contrastive Sharpening (CS, layer 1), Global Sparsity (GS, layer 2) and the Two-Way Judge (TWJ, layer 3). F3 and F13 credit its direction; F4 blames its GS-widened token sets. **CINDER-RS** keeps layers 1 and 3 and replaces GS with EMBER's row sparsity: CS + RS-1% + TWJ, no GS. It separates CINDER's direction (layer 1) from the GS widths (layer 2). It is evaluated with E1 and E2 settings unchanged: the same grid, question set, gates, selection, test pass and Alpaca, and the same integrity tests, sweep and blind T3 labels. 88–100% of its edited tokens are in the frozen pool (Valentine's Day 46: 68 of 77), so collateral covers it. The user proposed it after E1 and E2 were seen, so the comparison is exploratory.

**Result (F18, D15).** CINDER-RS is the only variant with an edit in all 8 runs, and it has the best mean test H4 of the Alpaca-scored primaries (0.53; 1b 0.52, EMBER 0.48, CINDER 0.48). Paired on shared runs, the lead comes from avoiding each method's failure run, not from a consistent per-run gain, so it is a tendency within noise. At small r it keeps CINDER's direction gain, but its collateral at fixed r is no better than CINDER's, so dropping GS does not fix the collateral curve. At its selected points it destroys 0.19 of the words (EMBER 0.30, CINDER 0.21), but 0.65 of its exclusive words.

### Where we stand

- **Done:** the premise check, the pipeline pilot and the frozen question set (D5). **E1:** 1,035 grid points on 8 concept-seeds with collateral at every point, the pooled shape choice (α = 1; best shape power m = 2 on CINDER's tokens), selection, the test split and Alpaca for the primaries, with test collateral renormalized after the erratum of amendment 6 ([poc.md](results/varwidth/layers/collateral/poc.md), D4, D9). **E2:** the pilot and two T2 fixes (amendments 4b and 4c), judge-free tests on 8 concept-seeds, 3,641 blind T3 labels on 3 of them with 0.98 agreement on cross-agent duplicates, and the split by token class ([integrity/report.md](results/varwidth/layers/collateral/integrity/report.md), D14, R11). **CINDER-RS** (amendment 5) through E1 and E2 with unchanged settings (D15). Gemini spend: 21.6 of 25 NIS, shared with the multilingual work.
- **Not started:** the proposed E3 (S2), the identity-preserving edit (S1), the expansion concepts and relearning.

### What to keep in mind

- **Efficacy fell for every method.** EMBER's validation efficacy was 0.29–1.00 under the H2 rule and is 0.00–0.42 under the collateral rule (R2). The efficacy questions contain the same tokens, so part of the old efficacy was destruction.
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
| Gemini: v1r word lists, Alpaca at the chosen point (6 primaries × 6 runs), pilot Alpaca | ≈ 4.7 NIS for the stage (actual: 5.2 NIS, CINDER-RS included; total spend 21.6 of 25 NIS) |
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

**Sources.** Per-config outputs in `results/varwidth/layers/collateral/integrity/<concept>/s<seed>/*.json`, summarized in [integrity/report.md](results/varwidth/layers/collateral/integrity/report.md): the per-config table, means over concept-seeds, the T3 label table and judge agreement. Per-concept detail stays in report.md. Judge-free columns are means over the n runs of each config (8 for the references); T3 columns come from the 3 judge runs (Harry Potter 42, Ancient Rome 44, Valentine's Day 42). Agent labels: 3,641 of 3,641 items (640 of them CINDER-RS, through the incremental export of amendment 5); agreement on 284 duplicates labelled by different agents is 0.98 on the exact label and 0.99 on destruction yes/no. R11's per-run split is computed from the same labels and `meta.json` with the class rule of integrity.py.

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
- **The context survives** (F11). ctx\_pass stays at 1.00. Only EMBER at r ≥ 10 pulls context-MC well below the references (0.45 and 0.38 vs 0.53–0.56) and raises the answer NLL by more than a nat; every other point stays at 0.51–0.57.
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

**Overall reading (E1 + E2).** CINDER as a bundle does not beat EMBER at the collateral-aware selected point (test H4 0.48 vs 0.48; F2). Its direction is better where erasure is non-destructive: it erases more per unit of perturbation at r ≤ 0.5 (F3, F6), and at r = 0.25 it removes much more concept knowledge with little destruction (F13). GS is the bundle's liability: its widened, generic token sets (F4) and its failure runs (no feature at Ancient Rome 43, little erasure at Valentine's Day 46). Removing GS does not lift collateral at fixed r, though, so GS is not the whole token-set problem (D15). Under an integrity criterion CINDER, CINDER-RS, 1a and 1b are less destructive at their selected points than EMBER (destruction 0.18–0.21 vs 0.30, 3 judge runs; F12). CINDER-RS (layers 1 + 3) is the most robust and best-H4 candidate so far (test H4 0.53, an edit in all 8 runs; F18), but within noise: its lead comes from avoiding the other methods' failure runs, once through a destructive point. Exclusive tokens remain largely destroyed by every method at its selected point (0.37–0.75; F17). That is the open problem, for identity-preserving edits (S1) and integrity-gated selection (S2).

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
| F1–F18, S1–S8 | Findings and speculations of Part 1 |
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
- [ ] Re-check the 17 mangled-copy CINDER-RS answers labelled BLIND against the earlier labellers' CONFAB convention (R9b).
- [ ] Decide whether echo stays a reported test (nonce echo below the 0.8 pilot gate in 6 of 8 runs, R8).
- [ ] Proposed E3 (S2): integrity-gated selection on the full r grid for EMBER, CINDER, 1b and CINDER-RS. Needs approval: it adds E2's judge-free tests at every grid point.
- [ ] Proposed: an identity-preserving edit variant (S1), the nonce-matching T3 score (S5, from existing labels), and OE or nonce-controlled efficacy (S6, S8).
- [x] µ and cos(c, e) medians of D1 saved: `python -m experiments.varwidth.collateral_diag mu` → [diag/mu\_cos.json](results/varwidth/layers/collateral/diag/mu_cos.json).
- [x] Expansion list: the 12 anchor concepts not used for development (the 14 anchor concepts minus Harry Potter and Valentine's Day; Ancient Rome is not an anchor concept). The plan's "11" was a miscount.
- [ ] Decide whether to re-run the premise prompts at the selected points.
- [ ] Merge into CINDER_improvements.md, with the reframed claim of R7 and the overall reading of R12.
- [ ] After this stage: the 12 expansion concepts (S7) and LoRA relearning at r ≈ 0.5 vs r = 100 (S4).
- [x] Edit set checked against the code: `edit_vectors` in [erase.py](experiments/varwidth/erase.py) edits support ∩ codes = 1, as in §3.3.
- [x] Variant definitions checked against [erase\_v2.py](experiments/varwidth/erase_v2.py) and [collateral.py](experiments/varwidth/collateral.py).

### Sources of the numbers

- Anchor δ choices, validation curves and gate counts: [results/varwidth/layers/erasure/](results/varwidth/layers/erasure/) (`<concept>/<rung>_s42/{grid.json, result.json}`; the paper-grid run is `Harry_Potter/L0_s42_papergrid`).
- Anchor summary and fidelity: [results/varwidth/layers/SUMMARY.md](results/varwidth/layers/SUMMARY.md), [results/varwidth/layers/erasure.md](results/varwidth/layers/erasure.md).
- Similar-domain questions: [data/mc\_questions.json](data/mc_questions.json).
- Premise check: [premise\_prompts.json](results/varwidth/layers/collateral/premise_prompts.json), [premise\_generations.json](results/varwidth/layers/collateral/premise_generations.json).
- Pilot: [collateral/Harry\_Potter/\_pilot\_s42\_v0metric/](results/varwidth/layers/collateral/Harry_Potter/_pilot_s42_v0metric/) (grids record the union collateral; the own-token collateral of the pilot table is recomputed from the saved per-token accuracies by `python -m experiments.varwidth.collateral_diag pilot` → [diag/pilot\_own\_tokens.json](results/varwidth/layers/collateral/diag/pilot_own_tokens.json)), [cl\_pilot\_1055037.out](results/varwidth/layers/slurm/logs/cl_pilot_1055037.out).
- Token pools and drafts: [collateral/pool/](results/varwidth/layers/collateral/pool/), [collateral/drafts/](results/varwidth/layers/collateral/drafts/).
- Pre-registration: [PREREG\_amend3\_draft.md](results/varwidth/layers/collateral/PREREG_amend3_draft.md), [PREREG.md](results/varwidth/layers/PREREG.md) (amendment 3: E1; 4, 4b, 4c: E2 and its pilot fixes; 5: CINDER-RS; 6: the test-collateral erratum).
- Edit magnitudes µ and cos(c, e): [diag/mu\_cos.json](results/varwidth/layers/collateral/diag/mu_cos.json).
- E1 results: [collateral/poc.md](results/varwidth/layers/collateral/poc.md) (written by `python -m experiments.varwidth.collateral_report report`); per run `collateral/<concept>/s<seed>/<variant>/{grid.json, result.json}`; Pareto PNGs in [collateral/pareto/](results/varwidth/layers/collateral/pareto/); shape choice in [shape\_choice.json](results/varwidth/layers/collateral/shape_choice.json). Edit sets and the 2×2 top-N overlap (`_info.topN_overlap_rawY`): `collateral/<concept>/edits_s<seed>.pt`.
- E2 results: [integrity/report.md](results/varwidth/layers/collateral/integrity/report.md) (written by `python -m experiments.varwidth.integrity report`); per config `integrity/<concept>/s<seed>/*.json`, with the words, nonces and tokens in `meta.json`; the pilots in `integrity/Harry_Potter/_pilot1_s42/` and `_pilot2_s42/`.
- T3 labels: [integrity/judge/](results/varwidth/layers/collateral/integrity/judge/): blind batches `batch_*.json`, agent labels `labels_*.json` (3,641 items plus 358 duplicates; batches 21–25 are CINDER-RS), the id-to-config key `key.json`, and `pre_clarification/` (the first labels of batches 00–02, superseded after the brief's clarifications; 347 of 450 unchanged). Brief: [briefs/judge\_t3.md](results/varwidth/layers/collateral/briefs/judge_t3.md).
- Code: [erase.py](experiments/varwidth/erase.py), [erase\_v2.py](experiments/varwidth/erase_v2.py), [collateral.py](experiments/varwidth/collateral.py), [collateral\_diag.py](experiments/varwidth/collateral_diag.py), [collateral\_report.py](experiments/varwidth/collateral_report.py), [integrity.py](experiments/varwidth/integrity.py).

### References

- Belrose et al. LEACE: Perfect linear concept erasure in closed form. NeurIPS 2023.
- Deeb and Roger. Do unlearning methods remove information from language model weights? arXiv:2410.08827, 2024.
- Eldan and Russinovich. Who's Harry Potter? Approximate unlearning in LLMs. arXiv:2310.02238, 2023.
- Hendrycks et al. Measuring massive multitask language understanding. ICLR 2021.
- Lynch et al. Eight methods to evaluate robust unlearning in LLMs. arXiv:2402.16835, 2024.
- Suslik, Shafran and Geva. Don't forget your embeddings: robust knowledge erasure via precise editing of embeddings (EMBER). arXiv:2606.03695, 2026.
- Taori et al. Stanford Alpaca: an instruction-following LLaMA model. 2023.
