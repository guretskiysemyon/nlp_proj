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
| X10 | Does extrapolation need the features? EMBER+X vs a frequency rule without features, and foreign-only added tokens (train questions, coverage only) | `x10_extrapolation_ablation.py` | CPU | **done** 2026-10-19 |
| X11 | Which extrapolation works best? Real erasure of EMBER, EMBER+X, X_foreign, X_clean at the fair EMBER pick δ and its neighbour; QA_train + SimdomQA_train in 5 languages, MMLU (train items only; the test split is kept for confirming the choice) | `x11_extrap_erasure.py` (+ `x11.sbatch`) | 1 GPU per concept | **done** 2026-10-19 (jobs 1056580–1056585, all exit 0) |
| X12 | Token-choice control: E0 + as many added tokens as X, same edit norms, chosen by sentence frequency (F_own) or at random (R_own) from the same concept-only pool; real erasure at the fair EMBER pick δ, train questions, vs X11's EMBER / X | `x12_token_control.py` (+ `x12.sbatch`) | GPU, 2 jobs | **done** 2026-10-19 (jobs 1057464–65); `x12_token_control/report.md` |
| X13 | Fair run, held-out tests: two-level bootstrap CIs + Holm for the paired leak differences, per-question mechanism of +X, coverage vs leak over all arms, failure cases, added tokens of the 12 new concepts | `x13_fair_analysis.py` | CPU | **done** 2026-10-19 |
| X14 | Follow-up: foreign retention (co-primary), comparisons on concepts both arms edit, collateral on SimdomQA_test split by edited-token presence, exposure of the edited tokens in ordinary text; all arms incl. the native-text ones | `x14_followup.py` | CPU | **done** 2026-10-20 |
| ML | Native-text arms: EMBER fitted per language (rank 100, union of the edits) and jointly (rank 500), through the fair pipeline | `cinder/ml_arms.py`, `cinder/submit_ml.sh` | GPU chain | **done** 2026-10-20 |

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

## X10: does extrapolation need the features? (2026-10-19, EMBER, 6 concepts)

CPU, coverage only (no erasure), on the **train** questions of all 5 languages (`data/qa_lang`, the test split is not
used). Edit sets, all containing EMBER's English edit: X = EMBER+X as built and tested in X9; X_foreign = X without
the added tokens that occur in English text (English sentences of all 18 concepts); L = the same number of tokens
from the same concept-only pool, chosen by how many concept sentences contain them (no features); L_word = L with
whole-word tokens only; L_foreign = L_word from the foreign-only pool. Means over the 6 concepts (de/es/fr/ru
averaged; `simdom` = SimdomQA_train questions reached, `bg` = other concepts' native sentences reached, both
collateral):

| | added tokens | QA_train reached, en | foreign | SimdomQA_train reached, en | foreign | background, en | foreign |
|---|---|---|---|---|---|---|---|
| EMBER | 0 | 0.78 | 0.29 | 0.10 | 0.02 | 0.008 | 0.004 |
| X | 96 | 0.79 | 0.61 | 0.12 | 0.10 | 0.013 | 0.021 |
| X_foreign | 86 | 0.79 | 0.59 | 0.11 | 0.08 | 0.008 | 0.018 |
| L_word | 96 | 0.78 | 0.69 | 0.11 | 0.22 | 0.013 | 0.099 |
| L_foreign | 86 | 0.78 | 0.66 | 0.10 | 0.20 | 0.008 | 0.078 |

- **The features matter.** Without them, the frequency rule reaches somewhat more foreign questions but with 2–2.5×
  the similar-domain collateral and 4–5× the background exposure, in every concept except Nazism (similar). Per
  similar-domain question touched, X gains about twice as many concept questions (≈ 4–5 vs ≈ 2). The two sets share
  only 5–23% of their tokens: the frequency rule picks subword fragments (пу, tom, sem, ca, Der) that are "concept-only"
  only because they are missing from 300 neutral sentences. So X4's near-chance AUC (energy share over all native
  words) does not mean the loading is useless: its top-ranked tokens are mostly real concept words.
- **Foreign-only (dropping English-surface tokens) is a small, mixed gain.** It brings English collateral back to
  EMBER's level (background 0.013 → 0.008) and lowers foreign collateral (0.10 → 0.08) for a small coverage loss
  (0.61 → 0.59), but it is blunt: it removes noise (dies, coup, violence, mental, permit, costumes, Wales, Albania) and
  also good tokens that English shares (munitions, canon, balles, Kaiser, Versailles, Ryder, Tee).
- Correction to the earlier description: the clearly unrelated tokens (trumpet, Tobacco, phone) are from X5's
  per-language solve, which also re-solves the English tokens; the build actually tested (x5b) keeps the English rows
  fixed and does not add them.
- A better lexical rule (e.g. against a large background corpus) might narrow the gap; this compares against a plain
  frequency rule only.

## X11: which extrapolation works best? (2026-10-19, EMBER, 6 concepts, real erasure)

Each edit applied to Gemma at the fair run's EMBER pick δ and its grid neighbour; QA_train / SimdomQA_train in 5
languages, MMLU (50 train items). Test split not used. Chance-corrected retention vs the pristine model (lower = more
erased); `sim` = SimdomQA_train accuracy change (collateral). Means over the 6 concepts at the pick δ
(`x11_extrap_erasure/report.md`):

| | tokens | ret en | de | es | fr | ru | foreign mean | leak | sim en | sim foreign | MMLU / pristine |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMBER | 50 | 0.63 | 0.76 | 0.79 | 0.72 | 0.98 | 0.81 | 0.18 | −0.03 | −0.007 | 1.00 |
| X | 146 | 0.58 | 0.60 | 0.72 | 0.49 | 0.91 | 0.68 | 0.10 | −0.04 | −0.022 | 1.00 |
| X_foreign | 136 | 0.59 | 0.59 | 0.75 | 0.48 | 0.91 | 0.68 | 0.10 | −0.03 | −0.020 | 1.00 |
| X_clean | 137 | 0.58 | 0.60 | 0.73 | 0.50 | 0.95 | 0.70 | 0.11 | −0.04 | −0.018 | 1.00 |

- **Extrapolation works in real erasure, not only in coverage.** Foreign retention 0.81 → 0.68 and leak 0.18 → 0.10
  at the same δ; X erases at least as much foreign knowledge as EMBER in all 6 concepts, at both δ values. Largest
  gains: Culture of Greece de 0.81 → 0.36, Ireland fr 0.82 → 0.09, Nazism de/es/fr −0.19 to −0.26. Smallest: Gun at
  its weak pick δ 5 (0.99 → 0.96; at δ 10 0.95 → 0.84) and Golf (0.74 → 0.74).
- **Cost:** similar-domain accuracy falls by about 1.5 points more in the foreign languages (−0.007 → −0.022) and
  1 point in English; MMLU unchanged at the pick δ (one Golf neighbour 0.96). Added tokens also reach English
  questions: English retention 0.63 → 0.58.
- **The two filters do not beat plain X.** X_foreign − X: foreign retention +0.002 (better in 3 of 6 concepts);
  X_clean − X: +0.014 (better in none), with collateral lower by only 0.002–0.005. These gaps are below one question
  (50 questions per language). Plain X is the choice; X_clean is the option if collateral matters more.
- Not tested: Ori's naive-projection extrapolation (only the Z-fixed sparse solve, X5/X5b), CINDER+X with real
  erasure on the train split, a δ re-tuned for X, and the test-split confirmation.

## X13: analyses of the fair run (2026-10-19, held-out QA_test, 18 concepts, `x13_fair_analysis/report.md`)

Reliable cells only (pristine QA_test ≥ 0.40; drops Harry Potter ru and Valentine's Day ru). Primary test, fixed
before looking: leak over de/es/fr/ru, 18 concepts; sensitivity: unclipped retention, de/es/fr only.

| paired leak difference | mean | 95% CI (concepts × questions) | lower / higher | Wilcoxon p | Holm p |
|---|---|---|---|---|---|
| EMBER+X − EMBER | −0.059 | [−0.102, −0.023] | 12 / 2 | 0.001 | **0.007** |
| CINDER+X − CINDER | −0.039 | [−0.088, −0.004] | 10 / 4 | 0.020 | 0.10 |
| CINDER − EMBER | +0.041 | [−0.060, +0.137] | 9 / 9 | 0.39 | 1.0 |
| CINDER+X − EMBER+X | +0.061 | [−0.035, +0.150] | 6 / 12 | 0.11 | 0.43 |
| EMBER + Ori's judge − EMBER | +0.048 | [−0.006, +0.119] | 0 / 4 | – | 1.0 |
| CINDER − EMBER + Ori's judge | −0.007 | [−0.094, +0.081] | 11 / 7 | 0.73 | 1.0 |

- Only EMBER+X − EMBER survives the correction; CINDER+X − CINDER has a CI just below 0 but Holm p 0.10. The same
  under unclipped retention. The de/es/fr-only lean CINDER+X > EMBER+X (p 0.021) does not survive Holm (0.10): no
  method difference.
- **Mechanism (per question, pristine-correct questions):** extrapolation's extra erasure sits in the questions it
  newly reaches. Foreign questions with only an added token: lost 0.5% → 36.2% (EMBER+X), 0.5% → 23.7% (CINDER+X);
  with a parent token: 44.2% → 45.3%, 29.4% → 29.3%; with neither: 0.6% → 1.1%, 0.1% → 0.6%.
- **Coverage vs leak:** Spearman ρ = −0.58 over all 296 edited reliable (concept, foreign language) cells of the 5
  arms (−0.35 to −0.77 per arm); vs the English − L coverage gap +0.65. One mechanism for every arm.
- **Failures (`failures.md`):** foreign questions still answered after EMBER+X, about half with no edited token:
  answers that are numbers or dates (1955, 17 марта), or names split into pieces (St·aling·rad, R·ube·us Hag·rid,
  Б·роне·бой·ный). Russian keeps 319 of 358, 255 of them without an edited token.
- **Added tokens, 12 new concepts:** 52–83% whole words, 85–94% never in English text; mostly on topic (Casinos,
  cuarentena, mascarilla, Neutron, Plut, römischen), with clear noise in some (EMBER+X Harry Potter: SHIELD,
  Nintendo, Wii, Prada, Hitler; Pornography: Poverty, Punk, Emo).
