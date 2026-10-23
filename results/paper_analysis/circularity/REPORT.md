<!-- Saved by the lead from the WP agent's hand-back (a hook blocked the agent's own write); text unchanged. -->

# WP7: Is Contrastive Sharpening circular?

## Question

From `CINDER_improvements.md`:

- "CS may be partly circular at the feature level. The sharpening target is the concept's Wikipedia page, which contains the very tokens our purity metrics score. With a target that excludes those tokens [...] the CS gains mostly disappear [...] a disjoint target is needed for a paper, such as other pages about the concept or a held-out half of the page."
- "Tune η and γ on concepts outside the evaluation set."
- The ladder study found that "concept-only" (in S_C, never in S_N) admits generic words (get, win, heart), so purity must also be measured with a stricter label.

## Method (short; full detail in the appendix)

- **Concepts and fits.** The 18 EMBER paper concepts, seeds 42–44. RS-1% sparse MF (`fit_varwidth`, `VWConfig(method="fixed", refit=False)`, the L0/L1 config) is primary. GS (`method="rn"`) is secondary. Everything except the sharpening target is fixed: same V\*, same background whitening, η = 0.05, γ = 1.
- **Raw and page fits.** These reuse the cached L0/L1 (and GS) fits. A GPU refit of Ancient Rome, seed 42, reproduced the cached L1 fit exactly (same rel_err, 100% support agreement). My transform also reproduces the cached CS matrices to within 1e-11.
- **Seven conditions:**
  - `raw` (EMBER).
  - `page` (default CS).
  - `disjoint`: page sentences that are not near-duplicates of any S_C or S_N sentence.
  - `related`: 4 other Wikipedia pages, chosen by rule from the main page's Main/Further/See-also hatnotes, with S_C near-duplicates removed and the pool size-matched to the page.
  - `pagesub`: a random page-sentence subset as large as `disjoint` (a size control).
  - `leakfree`: page minus every V\* token type (the existing control).
  - `wrong`: the page of an unrelated paper concept, from a fixed derangement (HP→Golf, ...).
- **Number of fits.** 1,116 in total. This is 18 × 7 × 3 for RS-1%, the same again for GS, and 360 for the η×γ grid.
- **Metrics (judge-free, zero Gemini).** Pre-registered in `prereg.json` before any new-target fit was run.
  - The concept feature f\* is the feature with the most lexicon hits (concept_words_v1) among its top-20 source words, ties broken by ρ.
  - Primary metrics: the **ρ-rank** of f\*, and **strict purity**, the share of supp(f\*) that is strict.
  - A token is strict if it is concept-only AND (a lexicon word OR (n_SC ≥ 2 AND log(p_SC / p_Wiki) ≥ log 10)). p_Wiki comes from 43M tokens of 40k articles in the cached `wikimedia/wikipedia` shard.
  - Secondary metrics:
    - concept-only purity;
    - lexicon precision;
    - found@5 (f\* has ≥ 5 lexicon hits);
    - AUC of strict vs neutral tokens, scored by the |dense code| of f\* (`auc_S`);
    - the η-sweep's τ-coverage AUC;
    - **held-out-token** versions of `auc_S` and purity, computed only on tokens that are *not* types of the target, against raw on the same tokens.
- **Statistics.** The concept is the unit (n = 18, after averaging over seeds). Paired Wilcoxon tests, 10k-resample bootstrap 95% CIs over concepts, and Holm correction over the primary family (2 metrics × 6 targets). Retention is mean(T − raw) / mean(page − raw).

## Backed findings (RS-1%, n = 18 concepts × 3 seeds unless stated)

1. **The ρ-rank story does not generalize, for any target (page included).** The mean ρ-rank of f\* is 4.9 for raw (median 2.5) and 3.9 for page (median 1.3).
   - Page is better for 7 concepts and worse for 6 (p = 0.51). Disjoint (2.6, p = 0.21) and related (4.1, median 1.0, p = 0.17) are not significantly different either.
   - In raw, f\* already ranks first for 7/18 concepts. CS lifts the buried ones: Ancient Rome 22 → 2, Harry Potter 14 → 1, Halloween 15 → 1.3, AI 10 → 8. Disjoint does the same (2, 1, 3.3, 6.7).
   - But page *buries* others: Republic of Ireland 2.7 → 26, Golf 3.7 → 9.7, Uranium 3.7 → 8.3. The page target amplifies a broader direction, and f\* becomes a "countries" feature (Ireland, Britain, Italy...) or a "sports" feature (golf, baseball, basketball). Disjoint does not do this (1.0, 6.0, 4.0).
   - Seed-42 examples are in `examples_seed42.json`.

2. **Strict purity: the non-circular targets keep the gain; the wrong target destroys it.**

   | Target | Strict purity | Δ vs raw [95% CI] | Wilcoxon p | Holm | Better/worse |
   | --- | --- | --- | --- | --- | --- |
   | raw | .568 | | | | |
   | page | .620 | +.052 [.000, .107] | .088 | .62 | 11/5 |
   | disjoint | .628 | +.060 [.026, .096] | .003 | **.034** | 14/4 |
   | related | .630 | +.062 [.011, .114] | .009 | .081 | 14/4 |
   | pagesub | .614 | +.046 [.000, .094] | .044 | .35 | 13/4 |
   | leakfree | .504 | −.064 [−.139, .003] | .26 | 1 | 7/9 |
   | wrong | .123 | −.445 [−.507, −.380] | < 1e-4 | **1e-4** | 0/18 |

3. **AUC of strict vs neutral tokens: every on-topic target helps, and page helps most.**
   - Raw .576.
   - Page +.055 [.043, .069] (17/18). Related +.041 [.028, .057] (17/18). Disjoint +.037 [.021, .053] (15/18). Pagesub +.046.
   - Leak-free −.004. Wrong −.068 (17/18 worse).
   - Retention of the page gain: related 0.75 [0.58, 0.89], disjoint 0.66 [0.43, 0.84], pagesub 0.83 [0.62, 0.98].

4. **The gain reaches tokens the target never contained.** Here the metric is computed only on V\* tokens that are not types of the target.
   - Related: `auc_S` +.034 [.018, .053] (p = .0003, 16/18 better). Strict purity of the held-out part of supp(f\*) +.084 [.037, .136] (p = .007).
   - Disjoint: +.018 [.005, .031] and +.078 [.041, .120].
   - The targets do overlap the labels, but only in part. Related holds 35% of the concept-only tokens and 48% of the strict ones; disjoint holds 25% and 43%; page holds 89% and 92%.

5. **The η-sweep's coverage metric is the circular one.** The τ-coverage AUC (score = max |Y| over features with ρ ≥ 2, any concept-only token) improves only with page: +.033 (18/0). It *drops* with disjoint (−.018, 13/18 worse), related (−.025, 17/18 worse) and leak-free (−.022). The "coverage/AUC" gains of the original η sweep therefore do not survive a disjoint target. The f\*-level purity and AUC gains do.

6. **Dose-response (exploratory).**
   - Across all 108 (concept, target) pairs, the AUC gain tracks the share of strict tokens contained in the target (Spearman .61). This is driven mostly by wrong and leakfree.
   - Among the four on-topic targets, the strict-purity gain is unrelated to that share (ρ_s = .05, p = .66).
   - The AUC gain still grows with it (ρ_s = .39, p < .001; within-concept mean .50).
   - So the extra AUC that page gives over related and disjoint (≈ +.015) is the overlap-dependent part.

7. **Sharpening is concept-specific, not a generic regularizer.**
   - With the wrong concept's page, f\* is found@5 in only 2 of 54 runs (raw: 39/54), its ρ-rank falls to 39 (Holm p = .003), and its words become coordinates, IPA strings and fragments.
   - Leak-free is not neutral either: lexicon hits fall by 1.3 (p = .036) and found@5 falls to 0.52. Its target is ~540 rare page types; for Golf it produces a "put/placing" verb feature.

8. **The target steers which aspect of the concept is amplified.** For Harry Potter, page keeps the proper-name feature (Gryffindor, Slytherin, Dumbledore, Weasley; strict purity .77). Disjoint and related give a "magic" feature (magic, wizard, witchcraft, Hogwarts, Voldemort; .60–.62). Both rank first. Raw ranks 12th (.48).

9. **"Concept-only" vs strict.** The gap between concept-only purity and strict purity of f\* is the same in raw and page: .842 − .568 = .274 and .896 − .620 = .276. CS's f\* does not absorb more generic concept-only words than EMBER's. The strict label is conservative: Malfoy, which occurs once in S_C and is not in the lexicon, is not strict. The sensitivity label without the count rule (strict_any) shows the same direction but no significant gains (page +.048, disjoint +.035, related +.034; all p > .1).

10. **GS (secondary, the same 18 × 3 design).**
    - The strict-purity gain disappears for *every* target, including page: page −.001, disjoint −.003, related −.016. Leak-free is −.089 (Holm .056) and wrong −.467.
    - The AUC gains remain: page +.059, related +.044 (18/18), disjoint +.042, pagesub +.052. Retention is 0.74 and 0.72.
    - The ρ-rank is unchanged by any target except wrong.

11. **η × γ on the 15 concepts outside the tuning trio** (page target, 3 seeds, 9 configs).
    - Every config beats raw: strict purity .60–.64 vs .59, `auc_S` .61–.64 vs .58, median ρ-rank 1.0–1.3 vs 2.3.
    - The default (η = 0.05, γ = 1) is mid-pack: 6th of 9 by mean rank over 5 metrics (5.2). The best are (0.2, 1) at 4.2 and (0.05, 2) at 4.4.
    - Against the best config per metric, the default loses .024 strict purity [−.054, −.001] and .012 AUC [−.018, −.005] (p = .007, both against η = .05, γ = 2). The ρ-rank difference is not significant.
    - γ = 0.5 is the worst column on every metric. η matters little.

## Speculations (each with a test)

- **γ = 2 may look best only because of the leak.** A higher γ sharpens harder toward the page, which contains 92% of the strict tokens. *Test:* the same 3 × 3 grid with the `related` target. This is `--plan etagamma_related`, already implemented and about 25 GPU-minutes; it was skipped to save usage.
- **GS loses the purity gain because of width.** GS gives f\* a wider support whose tail is where CS and raw differ least. *Test:* compute strict purity on f\*'s top-48 tokens under GS, from the stored Y; no refit is needed.
- **Generic-word absorption may live elsewhere.** The ladder's absorption of generic concept-only words (finding 9) may sit in the other F_C features or in the edited token set, not in f\*. *Test:* the purity_C − purity_strict gap over all judged F_C features of the L1 runs (labels in `labels/<Slug>.pt`).
- **Pages that bury the concept.** Page buries Republic of Ireland, Golf and Uranium because broad co-topics (countries, sports, chemistry) dominate the page's second moment. *Test:* the overlap of the top enrichment directions of S_η with z_f\*, or with a "countries" feature.

## Paper paragraph

The sharpening target of CS is the concept's Wikipedia page, which also supplies the sentences whose tokens we score, so we re-ran the factorization of 18 concepts (3 seeds) with targets that do not contain the sampled sentences. Using sentences of the page that are not in S_C, or four other Wikipedia pages about the concept, raises the strict purity of the best concept feature as much as the page does (+0.06 vs +0.05 over EMBER's matrix; Wilcoxon p = 0.003 and 0.009, n = 18). These targets also keep 66–75% of the page's gain in concept-vs-neutral separation, and the gain holds on tokens that do not occur in the target. Another concept's page instead destroys the concept feature (strict purity 0.57 → 0.12; found in 2 of 54 runs), so the effect is specific to the concept and is not circular label leakage. Two earlier observations do not generalize, however: CS does not change the ρ-rank of the concept feature on average (it lifts Harry Potter, Ancient Rome and Halloween to first place but buries Republic of Ireland and Golf), and the coverage-based AUC used to tune CS improves only with the page target. The default η = 0.05, γ = 1 beats the original matrix on 15 held-out concepts but is not optimal; γ = 2 scores slightly higher.

## Appendix text

**Targets.** Every target is a set of token types (gemma-2-2b-it tokenizer, `add_special_tokens=False`, minus special, unused and byte ids).

- `page` is the existing target file.
- `disjoint` removes page sentences that match any S_C ∪ S_N sentence. A match is decided after casefolding and keeping [a-z0-9] words: an equal string, a shared word 8-gram, or a word-set Jaccard ≥ 0.5.
- `related` takes the first 4 hatnote targets ({{Main}}, {{Further}}, {{See also}}) of the main page in order of appearance. A page is skipped if it is anchored (`#`), list-like (List/Outline/Timeline/Index/Glossary/Bibliography of), the main page after redirects, or a disambiguation page, or if fewer than 30 of its sentences survive the same near-duplicate filter. The pool is subsampled (seed 0) to the main page's sentence count.
- `pagesub` is a seed-0 random subset of page sentences, as many as `disjoint` has.
- `leakfree` is `page` minus the V\* types.
- `wrong` is the page target of the concept assigned in `common.WRONG`.

Three concepts have a page target that is already largely disjoint, because S_C came from an older or different page: Gun (the target is "Firearm"; 410/412 page sentences are disjoint), Cannabis and Gambling. For these the page holds only 45–52% of the concept-only tokens. The transform is X = (E − μ_B) W S_η^{γ/2} with S_η = (1 − η) UᵀU/m + ηI over the whitened target types, the same code path as `prep.whitening_transform` (verified to within 1e-11 on three concepts).

**Metrics.**

- ρ_i = mean_C|Y_it| / (mean_N|Y_it| + 1e-8), the same as `ember.compute_embedding_stats`.
- The ρ-rank of f\* is 1 + #{i : ρ_i > ρ_f\*}.
- Purity is the share of supp(f\*) with the given label.
- The dense code is (ZᵀZ + 1e-4 I)⁻¹ZᵀA, the fit's own dense Y-solve.
- AUC is Mann–Whitney with ties counted 1/2.
- Strict tokens make up 245–509 per concept, about 19% of the concept-only tokens.

ρ values in CS fits are often ~10⁵ because f\* has no neutral token and the 1e-8 guard dominates. Ranks are still well defined (by concept mass), but log ρ and "dominance" ratios are not meaningful. The "ρ = 125" in the brief came from the older η-sweep code. With the `fit_varwidth` fits used here, HP's raw f\* ranks 12th (seed 42; mean 13.7), not 11th.

**Targets per concept.** The type counts are page / disjoint / related / pagesub / leakfree / wrong. The share of concept-only tokens contained in the target is given for page / disjoint / related / wrong.

| Concept | Page sents | Disjoint sents | Related sents | Types (p/d/r/s/l/w) | C share (p/d/r/w) | Strict n | Related pages | Wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Culture of Greece | 351 | 48 | 351 | 3221/703/2952/771/231/3760 | 1.00/0.11/0.25/0.19 | 409 | Mycenaean Greece; Greek colonisation; Hellenization; Byzantines | AI |
| Golf | 327 | 29 | 327 | 2126/423/2512/422/118/4410 | 0.98/0.08/0.34/0.27 | 287 | History of golf; Golf course; Rules of golf; Golf equipment | Nazism |
| Republic of Ireland | 545 | 248 | 545 | 3650/2217/3249/2157/1075/3124 | 0.98/0.31/0.33/0.18 | 267 | Names of the Irish state; History of the RoI; Irish Home Rule movement; Irish Civil War | HP |
| Ancient Rome | 710 | 414 | 710 | 4210/2964/4112/3051/1520/3422 | 0.99/0.42/0.45/0.20 | 343 | Founding of Rome; Roman Kingdom; Roman Republic; Punic Wars | COVID-19 |
| Baseball | 417 | 120 | 417 | 2433/996/2616/1076/346/2835 | 0.98/0.21/0.39/0.20 | 245 | Rules of baseball; Pitch (baseball); History of baseball; Origins of baseball | Heroin |
| Uranium | 336 | 31 | 336 | 2388/455/2398/494/113/4352 | 0.99/0.11/0.34/0.26 | 430 | Natural nuclear fission reactor; Uranium in the environment; Uranium mining; Uranium market | Pornography |
| Harry Potter | 403 | 101 | 403 | 3124/1236/3099/1138/532/2126 | 0.98/0.18/0.32/0.14 | 282 | Fictional universe of HP; HP influences and analogues; Legal disputes over the HP series; Religious debates over the HP series | Golf |
| Cannabis | 251 | 135 | 251 | 1983/1317/2242/1336/553/3143 | 0.52/0.22/0.32/0.18 | 399 | Etymology of cannabis; Chemical defenses in Cannabis; History of cannabis; Cannabis (drug) | WWII |
| Gambling | 116 | 32 | 116 | 1165/390/1113/394/157/3650 | 0.47/0.07/0.16/0.26 | 509 | Gambling in the UK; History of gambling in the UK; Gambling in Singapore; Gambling in the US | RoI |
| Gun | 412 | 410 | 412 | 2774/2731/2601/2772/1505/1983 | 0.45/0.44/0.40/0.14 | 347 | Handgun; Rifle; Shotgun; Carbine | Cannabis |
| Pornography | 680 | 380 | 473 | 4352/3067/3578/3029/1613/4210 | 0.96/0.40/0.37/0.25 | 335 | History of human sexuality; Sex industry; Pornography laws by region; Legality of child pornography | Ancient Rome |
| Valentine's Day | 343 | 26 | 307 | 2738/352/2972/378/108/2388 | 0.99/0.06/0.29/0.13 | 321 | Saint Valentine; The Cowherd and the Weaver Girl; VD in India; VD in Pakistan | Uranium |
| World War II | 539 | 239 | 539 | 3143/1855/4147/1929/719/1165 | 0.98/0.36/0.48/0.07 | 339 | Causes of WWII; Second Italo-Ethiopian War; Spanish Civil War; Second Sino-Japanese War | Gambling |
| Nazism | 700 | 414 | 700 | 4410/3182/4650/3216/1574/2738 | 0.94/0.43/0.41/0.17 | 385 | German question; German nationalism; Pan-Germanism; Unification of Germany | Valentine's Day |
| Heroin | 372 | 72 | 372 | 2835/953/2989/829/405/3221 | 0.99/0.15/0.30/0.16 | 383 | Heroin-assisted treatment; Opioid withdrawal; Drug rehabilitation; History of opium in China | Culture of Greece |
| Halloween | 291 | 32 | 291 | 2480/481/2507/543/199/2774 | 0.90/0.07/0.30/0.15 | 383 | Trick-or-treating; Halloween costume; Haunted attraction; Geography of Halloween | Gun |
| Artificial intelligence | 589 | 308 | 589 | 3760/2488/3864/2497/1233/2480 | 0.93/0.37/0.41/0.14 | 357 | Hardware for AI; Applications of AI; Chatbot; AI in healthcare | Halloween |
| COVID-19 pandemic | 630 | 329 | 569 | 3422/2321/3419/2276/1125/2433 | 0.99/0.41/0.40/0.18 | 299 | COVID-19 naming; Origin of SARS-CoV-2; COVID-19 pandemic in Hubei; COVID-19 pandemic by country and territory | Baseball |

**Full results.**

- All paired tests, CIs, retention ratios, held-out tests, dose-response correlations and the η×γ grid are in `analysis.json`.
- Per-concept seed means are in `per_concept_fixed.csv` and `per_concept_rn.csv`.
- Per-fit metrics are in `fits_metrics.csv`; there is one row per fit, and each row also holds the held-out metrics for every target.

**Caveats.**

- Disjoint targets are small for 6 concepts (26–48 sentences). Pagesub is the size control, and it behaves like page.
- The lexicon (Gemini-written) both identifies f\* and enters the strict label. The same lexicon is used in every condition.
- The η×γ grid uses the leak-prone page target.
- Only Gemma was studied.
- No erasure was run.

## Figures

- `figures/fig_targets_paired.{pdf,png,csv}` (6.8 in). Paired differences from raw for each target, RS-1%: Δlog₂ ρ-rank of f\*, Δ strict purity, Δ AUC strict-vs-neutral on all tokens, and the same AUC on tokens not contained in the target. Small symbols are concepts (seed means). Large symbols with bars show the mean and a bootstrap 95% CI over 18 concepts. The page held-out panel has few positives per concept (n = 14 concepts) and is noisy.
- `figures/fig_targets_paired_gs.{pdf,png,csv}`. The same figure for GS.
- `figures/fig_etagamma_heatmap.{pdf,png,csv}` (6.8 in). The η × γ grid on the 15 concepts outside the tuning trio (page target, 3 seeds): median ρ-rank, mean strict purity, mean AUC strict vs neutral, and mean rank over 5 metrics. The default is outlined, and the raw values are in the panel titles.
- `figures/fig_dose_response.{pdf,png,csv}` (3.3 in). The ΔAUC (target − raw) for each (concept, target) against the share of the concept's strict tokens that the target contains (`_gs` for GS).

## Sources

- Code is in `experiments/paper_analysis/circularity/`:
  - `common.py` (constants, `TransformBuilder`)
  - `texts.py` (targets; Wikipedia API, cached in `texts/`)
  - `wikifreq.py` (Wikipedia token frequencies)
  - `labels.py` (strict / lexicon / target-membership labels)
  - `metrics.py`
  - `fit.py` (fits, cached-fit import, plans `main` / `etagamma` / `etagamma_related`)
  - `analyze.py` (statistics and figures)
  - `run_all.sh` and `gpu_watch.sh` (launchers)
- Inputs:
  - `results/varwidth/prep/*_S300.pt`
  - cached fits `results/varwidth/<Slug>/{fixed,rn}_refit-False[_plain]/seed4{2,3,4}/fit.pt`
  - `experiments/whitening/data/gemma/{bg_wiki_stats.pt, defaults.json, targets/}`
  - `experiments/whitening/data/pages/`
  - `experiments/varwidth/j3_data/concept_words_v1.json`
  - the HF `wikimedia/wikipedia` shard 00040
- Pre-registration: `prereg.json`. It was written before any new-target fit and covers the conditions (including `pagesub`), the metrics (including the held-out-token metrics), the strict label and its threshold, the tests, and GS as a secondary analysis. The dose-response analysis was added afterwards and is exploratory.
- Compute: one local c-001 GPU (nvidia-smi GPU 2, bus 03), taken only once it was idle and released afterwards, plus one CPU-only SLURM job (`cl_pa_circ`, no `--gres`). Zero Gemini calls.
- Size: `fits/` holds 125 MB (sparse Y, metrics, float16 dense codes of the top-5 features). The lead may want to exclude it from git; every number above can be rebuilt from `fits_metrics.csv`.
