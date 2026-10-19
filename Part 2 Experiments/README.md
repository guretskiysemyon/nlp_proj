# Part 2 Experiments: CINDER vs EMBER (English fit, 6 concepts)

*Started 2026-10-18 evening, while the main CINDER δ search runs (`cinder/run_pipeline.sbatch`, jobs 1051382/3/6/7).
Scripts in `scripts/`; each writes its own folder with a `report.md`. EMBER = Part 1's features; CINDER = Ori's L123,
seed 42 (no feature for Nazism and Republic of Ireland, so no CINDER edit there).*

## Index

| # | Question | Script | Runs on | Status |
|---|---|---|---|---|
| X1 | Which tokens does each method edit; do the edits remove the same direction? | `x1_edit_sets.py` | CPU | **done** |
| X2 | How often do the edited tokens occur in ordinary text (collateral exposure)? | `x2_collateral.py` | CPU | **done** |
| X4 | Do native de/es/fr/ru concept words lie along each method's concept directions? | `x4_native_alignment.py` | CPU | **done** |
| X5 | Extrapolating the English features to native foreign tokens: what gets added, how much coverage? | `x5_extrapolation.py`, `x5b_build_extrapolated.py` | CPU | **done** |
| X6 | Erasure and side effects vs *edit size* (not δ); leak curves | `x6_strength_curves.py` | CPU | English **done**; leak plots after G1 |
| X7 | English train questions: does erasure happen only where an edited token appears? | `x7_question_split.py` | CPU | **done** |
| X8 | Same on the held-out questions in 5 languages at the picked δ | `x8_test_mechanism.py` | CPU | EMBER **done**; CINDER after the final test |
| G1 | Retention per language at **every** δ, for EMBER, CINDER, EMBER+X, CINDER+X | `g1_leak_curves.py` (+ `.sbatch`) | 1 GPU each | queued: 1051538–47 (plain), 1051577–86 (+X) |
| post | 5-language final test of CINDER at its pick, CINDER vs EMBER table, X6–X8 | `post_main.sbatch` | 1 GPU | queued 1051552 (after the main jobs) |
| final | Redraw X6, rerun X7–X9 when everything is done | `final_summary.sbatch` | 1 GPU | queued 1051587 (after all) |
| X9 | Does extrapolation reduce the leak? Retention per language at the picked δ, ± extrapolation | `x9_extrapolation_effect.py` | CPU | after G1 |

The G1 jobs wait for GPUs held by the main jobs (per-user limit) and start when those finish; they do not slow the
main run. Logs: `logs/slurm/p2x_*.out`.

## First results (2026-10-18)

**X8 (EMBER, Part 1 final test): the edit removes words, not knowledge.** Pooled over the 6 concepts, of the
questions the pristine model answered correctly, the edit turns 36–46% wrong when the question contains an edited
token, and 0–5% when it does not, in **every** language (en 36% / 5%, de 45% / 1%, es 46% / 1%, fr 43% / 3%).
The cross-lingual leak is therefore a coverage effect: a foreign question keeps its answer exactly when the edit
does not reach any of its tokens. X7 shows the same on the English train questions at every δ (untouched questions
do not change at all).

**X5: extrapolation fixes most of the coverage gap, for both methods.** Keeping the English features fixed and
solving coefficients for the native tokens adds the right words: Griechenland / griego / Grèce / Гре…,
pistola / fusil / pistolet / оружие, with some noise (trumpet, Tobacco, flight). Share of test questions reached:

| Concept | method | de | es | fr | ru |
|---|---|---|---|---|---|
| Gun | EMBER → EMBER+X | 0.04 → 0.28 | 0.08 → 0.62 | 0.12 → 0.60 | 0.00 → 0.60 |
| Gun | CINDER → CINDER+X | 0.04 → 0.32 | 0.08 → 0.66 | 0.12 → 0.60 | 0.00 → 0.54 |
| Culture of Greece | EMBER → EMBER+X | 0.18 → 0.82 | 0.04 → 0.56 | 0.08 → 0.90 | 0.00 → 0.52 |
| Culture of Greece | CINDER → CINDER+X | 0.22 → 0.82 | 0.06 → 0.60 | 0.12 → 0.92 | 0.00 → 0.52 |

(Full table, also Cannabis / Golf / Nazism / Ireland: `x5_extrapolation/built.csv`.) G1 measures whether this
turns into less leak (X9).

**X1 / X2: CINDER's wider features add neighbouring topics, which raises collateral exposure.** The tokens only
CINDER edits: Culture of Greece → Roman, Rome, Mediterranean, Italy, Persian, Ottoman; Golf → baseball, ball,
sport, archery; Gun → hit, struck, strikes, blow, attacked. In ordinary Wikipedia text, CINDER's Culture of Greece
tokens occur in 2.8% of sentences vs 0.56% for EMBER (5×); Gun 1.30% vs 0.92%; Golf 0.95% vs 0.80%; Cannabis
0.11% vs 0.19% (CINDER lower). Where both edit a token, the two edit directions have cosine 0.64–0.83.

**X6: at equal edit size, CINDER does not erase English better on the train questions.** QA_train accuracy at the
same relative edit size: about equal for Gun; CINDER higher (less erased) for Cannabis and Golf; Culture of Greece
better for CINDER at small edits, worse at large ones (`x6_strength_curves/report.md`, figures `english_*.png`).
The final picks (Qwen-judged, tomorrow) decide the headline comparison.

**X4: neither method's directions separate native foreign concept words well.** AUC 0.41–0.64, close to random
features of the same fit (null 0.43–0.65), for both methods; the top-scoring native words are the right ones for
Gun and Culture of Greece (Pistole, fusil, pistolet, griechisch…), mixed with noise elsewhere. The directions carry
some cross-lingual signal, but not a clean one.

## Notes

- Test-split numbers in G1 / X8 / X9 are descriptive: nothing is chosen on them.
- Extrapolated edits (`mf_outputs/extrap_*`) keep the original English edit exactly (checked) and add native tokens;
  they use foreign native *text* to choose tokens, the features stay English-only.

## Results after all jobs (2026-10-19)

All 30 jobs finished with exit code 0, no tracebacks. G1 reproduces Part 1's EMBER final test exactly at the picked
δ (60/60 cells). Full tables: `comparison_cinder_vs_ember.md`, `x8_test_mechanism/report.md`,
`x9_extrapolation_effect/report.md`.

**CINDER vs EMBER (seed 42, Part 1 protocol).** Retention on held-out QA_test (lower = more erased), mean over the 4
concepts CINDER edits: EMBER en 0.60 / de 0.72 / es 0.77 / fr 0.69; CINDER 0.66 / 0.78 / 0.84 / 0.76. Mean leak
(other languages minus en) about equal: +0.19 vs +0.18. CINDER erases English more only for Culture of Greece (0.46 vs
0.73), and then leaks more (+0.37 vs +0.13). Search score (harmonic with Alpaca) lower for CINDER on all 4 concepts;
no edit for Nazism and Republic of Ireland. MMLU unchanged for both (0.55).

**X8: both methods erase only the questions their edit reaches.** Correct answers lost on touched vs untouched
questions: EMBER 36–46% vs 0–5%, CINDER 30–31% vs 0–3%, in en, de, es and fr alike.

**X9: extrapolation lowers the leak at the same δ (descriptive, test split).** Mean leak, plain → +X: EMBER Cannabis
+0.14 → +0.09, Gun +0.42 → +0.32, Nazism +0.15 → +0.01, Ireland +0.23 → +0.07, Culture of Greece +0.13 → −0.06,
Golf +0.05 → +0.09; CINDER Gun +0.27 → +0.18, Culture of Greece +0.37 → +0.14, Cannabis and Golf about unchanged.
SimdomQA_test accuracy unchanged by extrapolation (within 0.02). The original English edit is kept, but some added
tokens also occur in English questions, so English retention moves too (EMBER Golf 0.75 → 0.64, Nazism 0.63 → 0.71);
part of a leak change can come from the English side. Not yet measured for +X: MMLU, Alpaca coherence, a δ chosen on the train split.

## Next: fair comparison on all 18 concepts (2026-10-19)

The 6-concept results above use Part 1's protocol (Alpaca in the search) and EMBER with the paper's judge. The
follow-up runs EMBER, EMBER + Ori's judge and CINDER on all 18 concepts with the same judge model (gemini-3.5),
the same grid, a pick without Alpaca and the 5-language test: `cinder/fair.py`, plan in `docs/Part 2/plan.md`
"Fair comparison (option B)", report `judge_rerun/Fair/report.md`.
