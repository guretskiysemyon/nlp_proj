# Number audit of the CINDER draft (paper/)

Audited 2026-10-22 against the current `sections/*.tex` (after the restructure: new intro paragraph on multilingual erasure,
five-issue §3 with the "Scope" row, rewritten §4.4 and §5.4, new WP5/WP6 paragraphs in §6). The appendix was checked with
lower priority: every `\expnum` was checked, and new App. H text only by spot checks. No `.tex` file was edited.
Where a number could be recomputed from raw result files without a GPU, it was (marked "recomputed").

**Source abbreviations.** CE = `CINDER_collateral_evaluation.md`; CI = `CINDER_improvements.md`;
L/ = `results/varwidth/layers/`; COL/ = `results/varwidth/layers/collateral/`; CF = `COL/confirm.md`;
LM = `COL/last_mile.md`; TB = `COL/twobytwo.md`; NE = `COL/nonce_edit.md`; ER = `L/erasure/`;
HP/ = `results/varwidth/Harry_Potter/`; WP1 = `results/paper_analysis/quiet_concept/`;
WP2 = `results/paper_analysis/feature_geometry/`; WP5 = `results/paper_analysis/extrapolation/`;
WP6 = `results/paper_analysis/edit_direction/REPORT.md`; ml: = worktree `../nlp_proj_ml/`;
P1 = `report/main.tex`; EMBERpaper = `2606.03695v2.md`.

## Summary

Counts over the 200 table rows: **OK 158, MISMATCH 13 (12 distinct issues; M4 covers rows 26–27), NOT FOUND 1,
AMBIGUOUS 28.**

### MISMATCH (fix before submission)

| # | Where | Problem |
|---|---|---|
| M1 | 01_introduction.tex:80 | "The first layer reads one direction, the mean of all embeddings, far more strongly than others". On Gemma, μ̂_B is read at 2.9×, but the lowest 1% of Σ_B eigendirections are read at 3.4× and low-variance tilts at 5.3× (WP6:68-69). Only on Llama (20×) is the claim clearly true. |
| M2 | 03_motivation.tex:22 | "only 2 of the next 11" is true only for the notebook fit (`ember_three_steps.ipynb` cell 42). In the fit behind Fig. 2 (`HP/fixed_refit-False_plain/seed42/tables.md`), ranks 20–30 hold **3 of 11** HP tokens (Ginny, Lupin, Remus). |
| M3 | 03_motivation.tex:22 | "Latin taxonomic suffixes, all of which EMBER would edit". The suffix tokens (cidae, lidae, ridae, Ceramby) are labelled neutral [N] (tables.md:31-43). EMBER edits only the 31 concept-only tokens, so it does not edit them. |
| M4 | 03_motivation.tex:29 | "EMBER finds [Ancient Rome] as zero, one or two features; its edited token sets overlap with Jaccard 0.26 over 20 seeds". The 20-seed sweep is not EMBER's configuration: each feature gets a random width drawn from U(0.5%, 1.5%) (`results/ancient_rome_seed_sweep/experiments.md`:8, 62-72). Its Jaccard is over all concept-feature tokens (C, N and B, empty runs included), not over edited sets. Under EMBER's fixed 1% width and judge, seeds 42–46 give 0 or 1 features (WP2/ancient_rome.md). |
| M5 | 03_motivation.tex:49 | "Alpaca coherence stays at or above 0.87". In the anchor, coherence at every chosen point is ≥ **0.92** across all rungs and **0.956–0.994** for EMBER (recomputed from ER/*/result.json). 0.867 is the minimum *fluency fraction* (Gun, exploratory rung L0W). |
| M6 | 05_experiments.tex:47 | "those defined after earlier results were seen are labelled post hoc". None of the post hoc analyses the paper quotes is labelled: the McNemar test (in no PREREG), ITT and gate 0.7/0.9 (am. 9), floor and generation damage (am. 9), word-free efficacy, E3 and context damage (am. 11), the nonce edit (am. 13), the 2×2 (am. 14, "post hoc", L/PREREG.md:552), and A3z (ml: RESULTS.md:105). |
| M7 | 05_experiments.tex:103 | "CINDER's advantage becomes significant when runs without an edit count as failures". Under ITT, CINDER is +0.123, p = 0.092, Holm 0.103, so **not significant**. Only CINDER-RS-z survives Holm (0.049) (CF:79-82). The tightened gate (0.9) does hold: +0.174, p = 0.004. Both are post hoc. |
| M8 | 06_analysis.tex:20 | "the raw Harry Potter feature has the largest norm of all features". It ranks **5th of 100** (≈96th percentile, 1.56% share) (WP2/norm_runs.csv, HP L0 s42; WP1 features_all.csv z2_rank 5). |
| M9 | appendix/A_ember.tex:28 | "all numbers here use the repaired scorer". The Part-1 leakage numbers (0.24, 0.41, +0.01, +0.16), quoted in the intro, §3, §4.4 and App. F, use the **original** 4-token scorer (P1:280). With unparseable answers dropped they are +0.19 [0.12, 0.27] and +0.11, and Russian is +0.31 (P1:226, 284-285). |
| M10 | appendix/C_global_sparsity.tex:24 | "no concept feature reached it in 116 runs over 24 concepts". This count is stale. Now **190 runs over 38 concepts** (results/varwidth/*/rn_refit-False/seed*/summary.json) and still 0 at the cap (widest: Culture of Greece, 120 of 156; recomputed). |
| M11 | appendix/C_global_sparsity.tex:58 | The row "Width, seed s.d." is mislabelled. The source's ±4.1 / 8.1 / 7.9 is the mean absolute deviation, "average \|seed width − mean\|" (`results/varwidth/overall_report_cap0.05.md`:6). |
| M12 | appendix/I_examples.tex:19-26 | The table and caption come from the notebook fits (`ember_three_steps.ipynb` cell 42, `ember_three_steps_whitened.ipynb` cell 48), not from the fits behind Fig. 2. In the Fig. 2 fits, the raw ranks 20–30 hold 3 of 11 HP tokens, and the CS feature has no neutral token: it ends with Cedric and "arnia" (Narnia), with no "NHP". So App. I contradicts Fig. 2(b) ("no neutral mass"). |

### NOT FOUND

| # | Where | Problem |
|---|---|---|
| N1 | 04_method.tex:45 and appendix/B:41 | "whitening alone … on Harry Potter yields no concept feature at all". No source states this, and the data contradict it. In WP1's whitening-only fits, a clean HP feature ranks **1st by ρ in 4 of 5 seeds** (45 concept-only / 1 neutral tokens; WP1 features_all.csv, cond `wo`). defaults.json (whitening only) gives ρ_max 82 and top-feature purity 0.94. The documented claim (CI:168) is weaker: whitening alone *lowers coverage and AUC* below the raw matrix. |

### Most important AMBIGUOUS items (wording, scope, method)

- Several claims cover only one concept, seed or method but read as general: §3:47 (0.6 and 120× hold for HP only, 40–120× across concepts); §3:50 (12/20 is EMBER on HP seed 42); §1:52 and Fig. 2 (rank 12 is seed 42 only; seeds 43 and 46 rank HP 1st on the raw matrix).
- Table 2 caption: "12 held-out concepts" is the scope of Eff./Coll. only. "Broken" covers the 9 matched concepts at one seed each, and "Alp." covers seed 42 only.
- "single CINDER fit" (§1:67, 77) means A3z, the CINDER-z edit, which was defined post hoc. A3, the mapped-back edit that §5.2 calls "multilingual CINDER", scores 0.14.
- §5.4:127: "the gain holds on the 20 held-out concepts" holds in direction only (+0.19, 7–3, p = 0.064 strict).
- §7:22: "2–3× more than subtraction" holds against EMBER and CINDER-z under E3. Against CINDER under E3 (0.14) it is only 1.3–1.6×.
- §6:19: the raw nearest neighbours at "35–54°, punctuation and plural nouns" appear only in the agent brief. WP2 gives 35.1° / 42.1°, with a rare-names neighbour (bedridden, Fulda, …).

## Resolutions of the eight items the lead flagged

1. **Did selection choose δ = 200 for HP seed 42 (Table 4)? Yes.** `ER/Harry_Potter/L0_s42/result.json` has delta 200 and n_edit 31. The paper's own Alpaca-in-grid tuning (`L0_s42_papergrid`) also picks 200 (H 0.727 vs 0.726; L/erasure.md:35). Median r at δ = 1 is **0.605**, recomputed from `COL/Harry_Potter/edits_s42.pt` (C) and `results/varwidth/prep/Harry_Potter_plain_S300.pt` (A) over the 31 tokens; median cos(c, e) is 0.62. At δ = 200, median r is 121 and ‖e′‖/‖e‖ is 120.5. "r ≈ 0.6" is correct and the `\note` in the Table 4 caption can go. All quoted generations match `COL/premise_generations.json` (keys `Harry Potter|ember_d1` and `|ember_d200`; prompts 0, 7 and 4). The Trump prompt is truncated without "[…]": the original reads "…President of the United States?".
2. **"6 of 10", "344 grid points", "≥ 0.87".** All come from the Oct-18 erasure anchor (`L/erasure/`): seed 42, EMBER's MC protocol, the cheaper MMLU-gated δ rule, Alpaca scored only at the chosen δ, and the 14 non-prompt paper concepts.
   - EMBER (L0) edits 10 of the 14. It chose δ = 200 for Baseball, COVID-19, Gambling, Gun, Harry Potter and Valentine's Day, δ = 100 for Cannabis and Republic of Ireland, 50 for Pornography and 10 for Golf. It made no edit for Halloween, Heroin, Nazism or Uranium (CE:21, CE:317; recomputed).
   - 344 = 43 grids × 8 δ over **all 7 ladder rungs** (L0, L0H, L0W, L1, L12, L12W, L123). The gate field is empty at all 344 points (recomputed). EMBER alone has 11 grids (88 points).
   - For the 0.87, see M5: coherence is ≥ 0.92 everywhere and ≥ 0.956 for EMBER.
3. **"12 of 20 … 0–1 of 20".** From the premise check (CE:338-344). At δ = 200, **EMBER on Harry Potter seed 42** breaks 12/20, and CINDER breaks 9/20. On the other concepts: Ancient Rome s44, EMBER 3/20 and CINDER 12/20; Valentine's Day s42, EMBER 7/20 and CINDER 9/20. At δ = 1, both methods break 0–1 of 20 on all three concepts. The answers were read by hand against the unedited answer (not blind labels), and a few of the 20 prompts contain no edited token (CE:338).
4. **HP feature rank and tail.**
   - Raw fit: feature 87 ranks **12th** of the top-20 candidates by ρ (3.91), so 11 generic features outrank it (`HP/fixed_refit-False_plain/seed42/summary.json`). The CS fit's feature 85 ranks **1st** (N = 0). The paper is right; CI:39/149 and `experiments/paper_analysis/BRIEF.md` ("11th") are off by one.
   - The rank is seed-dependent. Across raw seeds 42–46 it is 12, 1, 28, 48 and 1 (WP1 features_all.csv).
   - The first 19 tokens are all HP in both fits. For the next 11, see M2 and M12: 2 of 11 in the notebook fit, 3 of 11 in the summary.json fit.
5. **SVD rank: 5 → 49 is right** for HP under RS-1% (`WP1/spectra/Harry_Potter.json`, `WP1/logs/spectra_0.log`:4). Raw is 5 at seed 42 (4–5 across seeds), and CS is 49 at every seed. For comparison: whitening only 66, CS + GS 52–53, raw + GS 7–8. The "67" in `experiments/paper_analysis/BRIEF.md`:46 is not reproduced; the closest value is the whitening-only 66.
6. **AUC 0.69 → 0.80, higher in 38 / 38: OK** (L/layers.md:22: "plain 0.690, CW 0.797; CW higher for 38 / 38"). This is judge-free separability of the V\* columns along the mean-difference direction. The 38 concepts are 23 original + 15 held-out (L/SUMMARY.md:4).
7. **Multilingual.** Table 3: all 12 numbers OK (ml: results/ml/PREREG_RESULTS.md:47-49).
   - "21 vs 10 of 38": OK.
   - "0.23 vs 0.25, p = 0.78": OK. The p is the paired Wilcoxon over the 29 concepts either approach edits (15–13), which is equivalent to all 38 with zero pairs dropped (PREREG_RESULTS.md:64; RESULTS.md:160, 230).
   - "+0.17, 16 of 21, p = 0.007": OK. It is the pooled 38 concepts, the 21 that A3 edits, 16–5, Wilcoxon (PREREG_RESULTS.md:58). The held-out 20 alone give +0.19, 7–3, p = 0.064.
8. **Part-1 leakage: values OK, scorer caveat missing.** +0.24 [0.15, 0.32], Russian +0.41, SNMF +0.01 and SNMF+EMBER +0.16 are all in P1:224, 223, 75-77. They use the original scorer (see M9).

## Full table

| # | Paper file:line | Claim | Value in paper | Value in source | Source file:line | Status | Suggested fix |
|---|---|---|---|---|---|---|---|
| 1 | 01_introduction.tex:52 | generic features outranking HP feature | eleven | rank 12 of 100 by ρ at seed 42; ranks 1, 28, 48, 1 at seeds 43–46 | HP/fixed_refit-False_plain/seed42/summary.json; WP1/features_all.csv | AMBIGUOUS | add "(seed 42)", or note that the rank varies by seed |
| 2 | 01_introduction.tex:52 | tail holds Nestlé, Lego, Trump | — | ranks 35, 46, 32 (C tokens) | HP/fixed_refit-False_plain/seed42/tables.md | OK | — |
| 3 | 01_introduction.tex:63 | EMBER-edited models retain more in de/es/fr/ru | 0.24 | +0.24 [0.15, 0.32], original scorer | P1:224, 280 | AMBIGUOUS | note the scorer, or quote +0.19 (unparseable dropped) |
| 4 | 01_introduction.tex:64 | multilingual EMBER fit finds a concept feature | 10 of 38 | A2: 10 of 38 (seed 42) | ml: results/ml/PREREG_RESULTS.md:47 | OK | — |
| 5 | 01_introduction.tex:67 | one CINDER fit erases as well as 8 EMBER runs | tie | A3z vs A1: 0.23 vs 0.25, p 0.78; A3 0.14 | ml: RESULTS.md:160 | AMBIGUOUS | say "with the CINDER-z edit (A3z)"; A3z is post hoc |
| 6 | 01_introduction.tex:67 | one factorization and one judging pass instead of eight | 1 vs 8 | 1 vs 8 fits; 10 vs 80 judged candidates | ml: RESULTS.md:67-73 | OK | — |
| 7 | 01_introduction.tex:71 | tuning prefers edits up to ~100× the token | 100× | paper-rule median r = 100 (EMBER, CINDER-z); HP δ = 200 gives r ≈ 121 | CE:1167, 1170; recomputed | OK | — |
| 8 | 01_introduction.tex:75 | "every method moves from token-destroying edits" to r ≈ 1 | every | paper rule: CINDER and CINDER-RS already at r = 1, CINDER-RS-z at 30 | CE:1167-1171 | AMBIGUOUS | "methods that destroy tokens under the paper's rule move to r ≈ 1" |
| 9 | 01_introduction.tex:76 | runs where only one method edits | 14 vs 3 | 14 vs 3, exact McNemar p 0.0127 (recomputed from COL/*/s*/sub_{ember,cinder}/result.json); not pre-registered | user summary report l.133 | AMBIGUOUS | value OK; label post hoc, unit = run |
| 10 | 01_introduction.tex:76 | equal edit size gain | +0.15, 10 of 11 | +0.151, 10/1, Holm 0.012 | CF:26 | OK | — |
| 11 | 01_introduction.tex:76 | tie at own operating points | tie | +0.042, Holm 0.412 | CF:9 | OK | — |
| 12 | 01_introduction.tex:77 | single CINDER fit vs per-language EMBER | 0.23 vs 0.25 | A3z 0.231 vs A1 0.248 | ml: PREREG_RESULTS.md:48 | AMBIGUOUS | value OK; name A3z / CINDER-z |
| 13 | 01_introduction.tex:77 | concepts with features | 21 vs 10 of 38 | 21 vs 10 | ml: PREREG_RESULTS.md:47 | OK | — |
| 14 | 01_introduction.tex:77 | selection cost | one eighth | 8 vs 1 fits, 80 vs 10 judged candidates | ml: RESULTS.md:67-73 | OK | — |
| 15 | 01_introduction.tex:80 | first layer reads the mean "far more strongly than others" | — | Gemma μ̂_B 2.9×; lowest-1% eigendirections 3.4×; low-variance tilts 5.3×; Llama 20× | WP6:18, 68-69 | MISMATCH | "far more strongly than a random direction (2.9× on Gemma, 20× on Llama)" |
| 16 | 01_introduction.tex:80 | "mean of all embeddings" | — | μ_B = mean over 107,472 Wikipedia token types | bg_wiki_stats.pt (n) | AMBIGUOUS | "the mean of ordinary-text embeddings" |
| 17 | 01_introduction.tex:83 | one eighth of the selection cost | 1/8 | as row 14 | ml: RESULTS.md:67-73 | OK | — |
| 18 | 03_motivation.tex:9 | Fig. 2: top 12 of 100, seed 42; raw: 11 generic outrank; CS: rank 1, no neutral mass | — | raw rank 12; CS feature 85 rank 1, C = 46, N = 0, B = 2 | HP/*/seed42/summary.json | OK | — |
| 19 | 03_motivation.tex:14 | reproduction within four points | 4 | 48.4/83.2/55.1 vs 52.1/84.6/53.6 (max gap 3.7) | L/erasure.md:39 | OK | ours: seed 42 × 5 concepts; paper: 5 × 5 seeds |
| 20 | 03_motivation.tex:15 | five issues | 5 | 6 table rows, one of them our own design choice | Table 1 | OK | — |
| 21 | 03_motivation.tex:20 | eleven generic features (magic, writing, 3rd-person verbs, books) | — | features 22, 9, 67, 73 at ranks 1, 2, 3, 5 | HP/fixed_refit-False_plain/seed42/summary.json | OK | — |
| 22 | 03_motivation.tex:22 | first 19 tokens all HP | 19 | 19 (both fits) | HP/fixed_refit-False_plain/seed42/tables.md:1-19 | OK | — |
| 23 | 03_motivation.tex:22 | HP tokens among the next 11 | 2 of 11 | 3 of 11 in the Fig.-2 fit (Ginny 21, Lupin 22, Remus 30); 2 of 11 in the notebook fit | tables.md:20-30; ember_three_steps.ipynb cell 42 | MISMATCH | use one fit throughout; with summary.json, "3 of the next 11" |
| 24 | 03_motivation.tex:22 | Nestlé, Lego, Trump and taxonomic suffixes, "all of which EMBER would edit" | all | suffixes are [N]; EMBER edits only the 31 C tokens | tables.md:28-44; CE:24 | MISMATCH | "Nestlé, Lego and Trump, which EMBER edits, and Latin taxonomic suffixes" |
| 25 | 03_motivation.tex:27 | 1% = 48 tokens for HP | 48 | ⌈0.01·4,780⌉ = 48 | summary.json (vstar 4780) | OK | — |
| 26 | 03_motivation.tex:29 | AR: EMBER finds 0/1/2 features over 20 seeds | 0/1/2 | random per-feature widths U(0.5%, 1.5%), not EMBER's 1% | ancient_rome_seed_sweep/experiments.md:8, 38, 62-72 | MISMATCH | describe it as a variable-width sweep, or use the RS-1% seeds 42–46 (0 or 1 TJ features; WP2/ancient_rome.md) |
| 27 | 03_motivation.tex:29 | "edited token sets" Jaccard | 0.26 | 0.258, over concept-feature tokens incl. N/B and empty runs | experiments.md:45, 48 | MISMATCH | "the tokens of its concept features" |
| 28 | 03_motivation.tex:34 | heroin "addictive drugs", WWII "totalitarian regimes", VD; all at 0.85 | 0.85 | as stated | CI:247, 274, 278 | OK | — |
| 29 | 03_motivation.tex:35 | EMBER edits nothing | 19 of 60 | 19 of 60 (41 edited; recomputed) | CE:137; CF:16 | OK | — |
| 30 | 03_motivation.tex:35 | multilingual EMBER finds a feature | 10 of 38 | 10 of 38 | ml: PREREG_RESULTS.md:47 | OK | — |
| 31 | 03_motivation.tex:39 | retain more in de/es/fr/ru; Russian | 0.24; 0.41 | +0.24 [0.15, 0.32]; Δ_ru +0.41; original scorer | P1:223-224, 280 | AMBIGUOUS | add a scorer caveat or the robust values (+0.19; Russian +0.31) |
| 32 | 03_motivation.tex:39 | "whose tokens the English edit never touches" | — | 0% of Russian concept words edited | P1:294 | OK | — |
| 33 | 03_motivation.tex:41 | single EMBER multilingual fit on Llama | 10 of 38 | 10 of 38 | ml: PREREG_RESULTS.md:47 | OK | — |
| 34 | 03_motivation.tex:46 | δ = 200 chosen for | 6 of 10 | 6 of the 10 concepts EMBER edits among the 14 anchor concepts, seed 42 (recomputed) | CE:21, 317; ER/*/L0_s42/result.json | OK | add "(seed 42; 10 of 14 concepts get an edit)" |
| 35 | 03_motivation.tex:47 | c_t ≈ 0.6 of the token; edited row ≈ 120× | 0.6; 120× | HP: µ = 0.605 (recomputed), ‖e′‖/‖e‖ = 120.5; Baseball 0.275, Gun 0.347; 40–120× across concepts | CE:290, 298-302 | AMBIGUOUS | "On Harry Potter, …"; or "40–120×" |
| 36 | 03_motivation.tex:49 | MMLU gate never binds | 344 grid points | 0 of 344 (43 grids × 8 δ, all 7 rungs, recomputed); EMBER alone 88 points | CE:23, 328 | OK | add "across all ladder variants" |
| 37 | 03_motivation.tex:49 | Alpaca coherence at or above | 0.87 | coherence ≥ 0.920 (all rungs), 0.956–0.994 (EMBER); 0.867 = min fluency fraction | ER/*/result.json (recomputed); CE:23, 329 | MISMATCH | "at or above 0.92 (0.96 for EMBER)" |
| 38 | 03_motivation.tex:50 | broken prompts at δ = 200 | 12 of 20 | EMBER, HP s42: 12/20 (CINDER 9/20) | CE:342 | AMBIGUOUS | "for EMBER on Harry Potter" |
| 39 | 03_motivation.tex:50 | broken at δ = 1 | 0–1 of 20 | 0–1 per method on HP42, AR44, VD42 | CE:342-344 | OK | — |
| 40 | 03_motivation.tex:50 | the 20 prompts "use the edited words in other senses" | 20 | "Most use an edited token …; a few are concept-adjacent topics with no edited token" | CE:338 | AMBIGUOUS | "20 neutral prompts, most using the edited words in other senses" |
| 41 | 04_method.tex:33 | η = 0.05, γ = 1 | — | defaults.json eta 0.05, gamma 1.0 | experiments/whitening/data/gemma/defaults.json | OK | — |
| 42 | 04_method.tex:45 | whitening alone yields no HP concept feature | none | whitening-only HP feature ranks 1st by ρ in 4/5 seeds (45 C / 1 N); ρ_max 82, purity 0.94 | WP1/features_all.csv (cond wo); defaults.json | NOT FOUND | drop, or replace with CI:168 ("lowers coverage and AUC below the raw matrix") |
| 43 | 04_method.tex:57 | GS floor / cap | 0.2% / 3% | floor 0.002, cap 0.03 | experiments/varwidth/common.py:49-50 | OK | — |
| 44 | 04_method.tex:68 | lexicon size | 20 | n = 20 | CI:252 | OK | — |
| 45 | 04_method.tex:70 | TWJ = TJ ∨ EJ@5 | 5 | as stated | CI:255 | OK | — |
| 46 | 04_method.tex:81 | EMBER's edit energy along μ_B | 44% | μ̂_B share 0.44 (27 concepts) | WP6:53 | OK | — |
| 47 | 04_method.tex:82 | μ̂_B read strength | 2.9× Gemma; 20× Llama | 2.9×; 20.4× | WP6:18, 107 | OK | — |
| 48 | 04_method.tex:82 | μ term alone explains EMBER > CINDER read | — | ⊥μ: 1.31 vs 1.32, p 0.44 | WP6:53 | OK | — |
| 49 | 04_method.tex:88 | c^z = c + β μ_B | — | ≥ 0.9995 energy on μ̂_B, 137 Gemma runs | WP6:44 | OK | — |
| 50 | 04_method.tex:97 | 30 + 30 sentences, fewest giving full rank | 30 | smallest N in {20, 30, 40, 50} with rank 4096; N = 20 fails for 9 concepts | ml: experiments/ml/PLAN.md:37-39 | OK | "the fewest of {20, 30, 40, 50}" |
| 51 | 04_method.tex:104 | 8 factorizations + 8 judging passes vs 1 + 1 lexicon call | 8 vs 1 | as stated | ml: RESULTS.md:67-73 | OK | — |
| 52 | 04_method.tex:105 | evaluation costs the same | — | "Evaluation … costs the same for every approach" | ml: RESULTS.md:75 | OK | — |
| 53 | 05_experiments.tex:24 | collateral items / tokens | 3,501 / 1,026 | 3,501 / 1,026 | CE:1135; L/PREREG.md (am. 8 freeze note) | OK | — |
| 54 | 05_experiments.tex:24 | items "kept only if the unedited model answers them correctly" | — | held-out set: under both option orders | CE:1134 | OK | — |
| 55 | 05_experiments.tex:28 | gates | MMLU 0.9, collateral 0.8 | same (≥) | CE:403 | OK | — |
| 56 | 05_experiments.tex:34 | r grid | {0.05 … 100} | same 10 values | CE:457 | OK | — |
| 57 | 05_experiments.tex:35 | EMBER's Alpaca judge | 150 outputs per point | AlpacaEval 150 validation prompts | EMBERpaper:589 | OK | — |
| 58 | 05_experiments.tex:39 | 300 sentences; 50/50 questions | 300; 50/50 | n = 300; 50 val / 50 test | EMBERpaper:142, 585 | OK | — |
| 59 | 05_experiments.tex:40 | 12 held-out concepts × 5 seeds | 12; 60 | 12 × seeds 42–46 = 60 | CE:1128 | OK | — |
| 60 | 05_experiments.tex:41 | 18 + 20 multilingual concepts, pre-registered | 18 + 20 | as stated | ml: RESULTS.md:113-115 | OK | — |
| 61 | 05_experiments.tex:46 | unit = concept, Wilcoxon + Holm, bootstrap | — | true of the primary contrasts; McNemar (row 81) is run-level | CF:3 | OK | — |
| 62 | 05_experiments.tex:47 | post hoc analyses are labelled | — | none labelled (see M6) | L/PREREG.md:388, 429, 447, 552 | MISMATCH | add "(post hoc)" where used, or reword |
| 63 | 05_experiments.tex:69 | without collateral, selection goes to ~100× | 100× | median r = 100 for EMBER and CINDER-z only; CINDER-RS-z 30; CINDER and CINDER-RS 1 | CE:1167-1171 | AMBIGUOUS | "for EMBER and CINDER-z" |
| 64 | 05_experiments.tex:69 | with collateral, every method at r ≈ 1 | 1 | median r = 1 for all five | CE:1162-1166 | OK | — |
| 65 | 05_experiments.tex:70 | neutral prompts broken at destructive points | 24–34% | EMBER 0.24, CINDER-z 0.34 (9 matched concepts; 0.31 / 0.35 on all) | LM:36, 38; CE:1226-1227 | OK | add "(9 concepts with every configuration)" |
| 66 | 05_experiments.tex:70 | deletion floor | 0.23–0.28 | 0.23 (EMBER), 0.28 (CINDER-z) | LM:9, 12 | OK | — |
| 67 | 05_experiments.tex:70 | runs above the floor | 83%, 98% | 1 − 0.17, 1 − 0.02 | LM:9, 12; CE:145 | OK | — |
| 68 | 05_experiments.tex:70 | "these edits derail the surrounding prompt as well" | — | context-MC at r = 100: EMBER 0.38, CINDER-z 0.31 (dev) | CE:130 | OK | interpretation backed by F23 (development concepts) |
| 69 | 05_experiments.tex:71 | Alpaca coherence | 0.97–0.98 | 0.97–0.98 at all 66 points, seed 42 | LM:45-56; CE:147 | OK | add "(seed 42)" |
| 70 | 05_experiments.tex:72 | paper rule puts EMBER ahead of CINDER | — | H2 −0.134, 1 win / 10, Holm 0.056 | CF:43 | OK | not significant after Holm |
| 71 | 05_experiments.tex:82 | EMBER, paper: r, eff, coll | 100, 0.47, 0.73 | 100, 0.470, 0.725 (recomputed) | CE:1167 | OK | — |
| 72 | 05_experiments.tex:82 | EMBER, paper: broken, Alp | 0.24, 0.97 | 0.24 (9 matched), 0.97 (seed 42) | LM:38, 46 | OK | — |
| 73 | 05_experiments.tex:83 | EMBER, ours: r, eff, coll | 1, 0.30, 0.86 | 1, 0.295, 0.855 (recomputed; LM shows 0.29) | CE:1162; LM:9 | OK | — |
| 74 | 05_experiments.tex:83 | EMBER, ours: broken, Alp | 0.15, 0.97 | 0.15 (matched; 0.14 all), 0.97 | LM:39, 45 | OK | — |
| 75 | 05_experiments.tex:84 | CINDER-z, paper: r, eff, coll | 100, 0.68, 0.68 | 100, 0.682, 0.681 (recomputed) | CE:1170 | OK | — |
| 76 | 05_experiments.tex:84 | CINDER-z, paper: broken, Alp | 0.34, 0.97 | 0.34 (matched; 0.35 all), 0.97 | LM:36, 53 | OK | — |
| 77 | 05_experiments.tex:85 | CINDER-z, ours: r, eff, coll | 1, 0.39, 0.86 | 1, 0.390, 0.857 | CE:1165 | OK | — |
| 78 | 05_experiments.tex:85 | CINDER-z, ours: broken, Alp | 0.16, 0.98 | 0.16 (matched; 0.15 all), 0.98 | LM:37, 52 | OK | — |
| 79 | 05_experiments.tex:88 | Table 2 caption: "on 12 held-out concepts" | 12 | Eff/Coll: runs with an edit (41 / 52) over 5 seeds; Broken: 9 matched concepts, one seed each; Alp: seed 42 only | CE:1194-1195, 1229 | AMBIGUOUS | state the scope of Broken and Alp. |
| 80 | 05_experiments.tex:94 | runs with an edit | 41 vs 52 of 60 | 41, 52 (recomputed) | CF:16-17 | OK | — |
| 81 | 05_experiments.tex:95 | only-one-edits runs; McNemar | 14 vs 3; p 0.013 | 14 vs 3, p 0.0127 (recomputed); not pre-registered; concept-level Wilcoxon p 0.06–0.18 | user summary report l.133 | AMBIGUOUS | label post hoc and run-level |
| 82 | 05_experiments.tex:95 | EMBER's empty runs on Heroin, Nazism, Uranium | — | 4/5 each; also Halloween 3/5, Golf 2/5 | CE:137; recomputed | OK | — |
| 83 | 05_experiments.tex:98 | efficacy at r = 0.5 | 0.25–0.30 vs 0.20 | 0.252–0.297 vs 0.201, collateral 0.93 for all (recomputed) | CE:136 | OK | — |
| 84 | 05_experiments.tex:98 | CINDER gain at r = 0.5 | +0.15, 10 of 11, Holm 0.012 | +0.151, 10/1, Holm 0.012 (pre-registered secondary) | CF:26 | OK | — |
| 85 | 05_experiments.tex:99 | direction effect | +0.07, 10 of 11 | +0.073, 10/1, p 0.032, Holm 0.097; post hoc (am. 14) | TB:16; CE:1401 | AMBIGUOUS | add "(post hoc; Holm p = 0.10)" |
| 86 | 05_experiments.tex:99 | token-set effect at matched size | +0.02 | +0.016 (n.s.) | TB:17 | OK | — |
| 87 | 05_experiments.tex:102 | pre-registered primary | +0.04, Holm 0.41 | +0.042, Holm 0.412 | CF:9; L/PREREG.md am. 8 | OK | — |
| 88 | 05_experiments.tex:103 | significant when no-edit runs count as failures | significant | ITT CINDER +0.123, p 0.092, Holm 0.103 | CF:79 | MISMATCH | drop, or "only CINDER-RS-z survives Holm (0.049)" |
| 89 | 05_experiments.tex:103 | significant when the gate is tightened | significant | gate 0.9: +0.174, p 0.004 (survives Holm); post hoc | CF:79; CE:144 | OK | label post hoc |
| 90 | 05_experiments.tex:117 | A3z vs A1 | 0.23 vs 0.25 strict; 0.28 vs 0.25 ungated | 0.231 / 0.248; 0.282 / 0.252 | ml: PREREG_RESULTS.md:48-49 | OK | — |
| 91 | 05_experiments.tex:117 | wins–losses, p | 15–13, p 0.78 | 15–13, Wilcoxon 0.779 (n = 29 either-edit) | ml: PREREG_RESULTS.md:64 | OK | — |
| 92 | 05_experiments.tex:118 | pooled fits cost | 3 instead of 24 | A3zU 3 fits; A1U 8 × 3 = 24 | ml: RESULTS.md:189-190 | OK | — |
| 93 | 05_experiments.tex:118 | pooling helps CINDER-z | +0.08, 19–4, p < 0.001 | +0.082, 19–4, Wilcoxon 0.000 (pre-registered am. 1) | ml: results/ml/UNION_RESULTS.md:23 | OK | — |
| 94 | 05_experiments.tex:118 | pooling does not help A1 | +0.05, n.s. | +0.054, 14–11, p 0.578 | ml: UNION_RESULTS.md:22 | OK | — |
| 95 | 05_experiments.tex:118 | pooled still tie | — | 0.313 vs 0.302, p 0.751 | ml: UNION_RESULTS.md:18 | OK | — |
| 96 | 05_experiments.tex:121 | A2 erases as well as A3z where both edit | — | 0.424 vs 0.429, n = 10, p 0.73 | ml: PREREG_RESULTS.md:60 | OK | — |
| 97 | 05_experiments.tex:121 | coverage A2 vs CINDER | 10 / 38, 2 / 20 vs 21, 10 | 10, 2 vs 21, 10 | ml: PREREG_RESULTS.md:9, 47 | OK | — |
| 98 | 05_experiments.tex:121 | across 3 fit seeds | 9–12 vs 17–21 | A2 10/9/12, A3 21/17/20 | ml: results/ml/seeds/COVERAGE.md:158-162 | OK | — |
| 99 | 05_experiments.tex:122 | counting coverage, A3z beats A2 | +0.21, 15–4, p 0.011 | +0.213, 15–4, Wilcoxon 0.011 | ml: PREREG_RESULTS.md:62 | OK | — |
| 100 | 05_experiments.tex:127 | A3z − A3 | +0.17, 16 of 21, p 0.007 | +0.166, 16–5, Wilcoxon 0.007 (38 pooled) | ml: PREREG_RESULTS.md:58 | OK | — |
| 101 | 05_experiments.tex:127 | "the gain holds on the 20 held-out" | holds | +0.19, 7–3, p 0.064 strict (ungated +0.26, p 0.014) | ml: PREREG_RESULTS.md:20, 35; RESULTS.md:157 | AMBIGUOUS | "and points the same way on the 20 held-out concepts (+0.19, p = 0.06)"; A3z is post hoc |
| 102 | 05_experiments.tex:128 | Llama mean direction read | 20× | 20.4× | WP6:107 | OK | — |
| 103 | 05_experiments.tex:131 | non-English Latin-script languages erased at | ~0.6 of English | per-approach ratio 0.49 (A2) to 0.70 (A1, A3, A3z); 18-concept run only | ml: RESULTS.md:58-65 | AMBIGUOUS | "about 0.5–0.7 of that (18 concepts)" |
| 104 | 05_experiments.tex:131 | Thai little, Hindi almost nothing | — | th 0.09–0.19; hi 0.02–0.08 | ml: RESULTS.md:62-65 | OK | — |
| 105 | 05_experiments.tex:132 | Hindi/Thai tokens "never concept-only and never edited" | never | hi 5%, th 6% concept-only (Baseball); Tuberculosis A1U erases Hindi | ml: OBSERVATIONS.md:32, 171 | AMBIGUOUS | "rarely concept-only and almost never edited" |
| 106 | 05_experiments.tex:142 | fits per concept | 8/1/1/1 | as stated | ml: RESULTS.md:67-73 | OK | — |
| 107 | 05_experiments.tex:143 | concepts edited | 26/10/21/21 | 26/10/21/21 | ml: PREREG_RESULTS.md:47 | OK | — |
| 108 | 05_experiments.tex:144 | strict score | .25/.11/.14/.23 | .248/.113/.139/.231 | ml: PREREG_RESULTS.md:48 | OK | — |
| 109 | 05_experiments.tex:145 | ungated score | .25/.14/.14/.28 | .252/.143/.139/.282 | ml: PREREG_RESULTS.md:49 | OK | — |
| 110 | 06_analysis.tex:8 | 38 concepts × 5 seeds; 18 + 20, 15 added after freeze | 38; 15 | 23 original + 15 held-out | L/SUMMARY.md:4 | OK | — |
| 111 | 06_analysis.tex:12 | AUC up in every concept | 38; 0.69 → 0.80 | 0.690 → 0.797, 38/38 | L/layers.md:22 | OK | — |
| 112 | 06_analysis.tex:12 | HP feature 12th → 1st by ρ | 12 → 1 | 12 → 1 (seed 42) | HP/*/seed42/summary.json | OK | add "(seed 42)" |
| 113 | 06_analysis.tex:13 | SVD rank matching sparse-MF error | 5 → 49 | raw 5 (4–5 over seeds), CS 49 (all seeds) | WP1/spectra/Harry_Potter.json; WP1/logs/spectra_0.log:4 | OK | — |
| 114 | 06_analysis.tex:17 | AR concept share | 0.72 ± 0.15 → 0.92 ± 0.04 | 0.717 ± 0.151 → 0.922 ± 0.037 (excl. 3 empty runs) | ancient_rome_seed_sweep/experiments.md:29, 32 | OK | add that widths were random per feature in both arms |
| 115 | 06_analysis.tex:17 | AR neutral tokens | 13.8 → 1.7 | 13.75 → 1.65 | experiments.md:27 | OK | — |
| 116 | 06_analysis.tex:17 | AR token overlap | 0.26 → 0.69 | 0.258 → 0.685 | experiments.md:45-46 | OK | — |
| 117 | 06_analysis.tex:17 | every seed one AR feature | 20/20 | 20/20 | experiments.md:39 | OK | — |
| 118 | 06_analysis.tex:19 | CS HP nearest neighbour magic; others beyond | 68°; 82° | 67.9°; 82.4° | WP2/concept_nn.csv (HP L1 s42 fit) | OK | — |
| 119 | 06_analysis.tex:19 | raw neighbours: 35–54°, punctuation and plural nouns | 35–54° | 35.1° (1st), 42.1° (2nd); 1st-neighbour words: bedridden, Fulda, bedrock, seabed… | WP2/concept_nn.csv (HP L0 s42); only source for the wording: experiments/paper_analysis/BRIEF.md:43 | AMBIGUOUS | re-derive from WP2; "35–42°, to unrelated rare-name features" |
| 120 | 06_analysis.tex:20 | raw HP feature has the largest norm | largest | rank 5 of 100 (96th pct); CS rank 47–75 | WP2/norm_runs.csv; WP1/features_all.csv | MISMATCH | "one of the largest norms (5th of 100)" |
| 121 | 06_analysis.tex:24 | GS widths | 15 (Baseball) – 113 (Culture of Greece) | 15 – 113 (cap-5% sweep, 18 concepts) | overall_report_cap0.05.md:16, 22, 26 | OK | — |
| 122 | 06_analysis.tex:24 | width change across seeds | ±4 | ±4.1 mean abs. deviation | overall_report_cap0.05.md:6, 16 | OK | — |
| 123 | 06_analysis.tex:24 | "as clean as EMBER's" | 76 vs 75 of 90 | 76 vs 75; both on the CS matrix; Heroin untracked (effective n = 85) | overall_report_cap0.05.md:12, 16, 36 | AMBIGUOUS | "as clean as EMBER's 1% rule on the same sharpened matrix" |
| 124 | 06_analysis.tex:25 | fixed 2% clean runs | 31 of 90 | 31 | overall_report_cap0.05.md:13 | OK | — |
| 125 | 06_analysis.tex:25 | no concept feature reaches the cap | none | 0 of 190 runs, 38 concepts (cap 3%, recomputed) | results/varwidth/*/rn_refit-False/seed*/summary.json | OK | — |
| 126 | 06_analysis.tex:26 | raw GS neutral share vs CS | 11–44% vs 1–9% | AR 37%, HP 44%, VD 11% vs 2%, 1%, 9% (3 concepts) | results/varwidth/whitened_vs_plain.md:13, 23, 33 | OK | add "(3 concepts)" |
| 127 | 06_analysis.tex:30 | TWJ hits, 19 concepts | 82 of 91 vs 61 | 82 / 91 vs 61 (j3_eval) | CI:284-287, 294 | OK | L/layers.md:97-106 (other labelled pool) gives TJ 59 |
| 128 | 06_analysis.tex:30 | unrelated accepts | 1 of 110 | 1 of 110 | CI:42, 294 | OK | — |
| 129 | 06_analysis.tex:31 | strict-label gain | 21 → 7 | 82−61 = 21; 67−60 = 7 | CI:284-287, 296 | OK | — |
| 130 | 06_analysis.tex:39 | leak-free target, HP ρ_max | 125 → 14 | 125.07 → 14.11 | defaults.json (table, control_page_minus_vstar) | OK | — |
| 131 | 06_analysis.tex:44 | samples | 300; 30 per language | as stated | ml: RESULTS.md:12 | OK | — |
| 132 | 06_analysis.tex:45 | unsampled concept tokens per concept | 17 concepts; ~730 | 17; 734 (balanced) | WP5/summary.json (n_concepts; tokens.n_Cstar) | OK | — |
| 133 | 06_analysis.tex:45 | top-10 precision vs random | 0.18–0.27 vs 0.065 | cP10 0.177–0.276; base cS 0.065 | WP5/summary.json (ranking, base_rate) | OK | — |
| 134 | 06_analysis.tex:45 | "about a third are unrelated" | ~1/3 | U share 0.27–0.39 (frac/width/cutoff rules), 0.52–0.70 (argmax) | WP5/summary.json (admission) | OK | — |
| 135 | 06_analysis.tex:45 | recall of missed concept tokens | 4% | recall_frac 0.036–0.037 | WP5/recall_runs.csv | OK | — |
| 136 | 06_analysis.tex:46 | raw projection tracks rarity | Spearman 0.70 | ρ_rarity 0.705 (EMBER proj) | WP5/extra.json judge_free | OK | — |
| 137 | 06_analysis.tex:46 | largest \|cos\| of concept feature | 0.28 vs 0.64 | 0.281 (CINDER) vs 0.642 (EMBER) | WP5/mechanism_runs.csv | OK | — |
| 138 | 06_analysis.tex:47 | unsampled page tokens seen once | 87% | 0.874 | WP5/extra.json page_count_distribution_mean | OK | — |
| 139 | 06_analysis.tex:47 | at one occurrence in-sample no more precise | — | EMBER c1: 0.297 vs 0.307, p 0.97 | WP5/extra.json insample_vs_new | OK | — |
| 140 | 06_analysis.tex:51 | CINDER ≈ random; EMBER / CINDER-z ~twice random | ~1; ~2 | 1.18; 1.86 / 1.99 | WP6:53-54 | OK | — |
| 141 | 06_analysis.tex:54 | lowest 1% eigendirections | 3.4× | 3.4 | WP6:69 | OK | — |
| 142 | 06_analysis.tex:54 | ⊥μ read, 27 concepts | 1.31× vs 1.32× | 1.31 / 1.32, n = 27 | WP6:53 | OK | — |
| 143 | 06_analysis.tex:55 | EMBER centred Z read | 1.09× | 1.09 (slightly below CINDER's 1.18, p 1e-4) | WP6:56 | OK | — |
| 144 | 06_analysis.tex:57 | cos(c, e) with / without mean term | 0.50 vs 0.30; 0.31 vs 0.29 | same | WP6:80-82 | OK | — |
| 145 | 06_analysis.tex:62 | Fig. 3 caption: concepts with an edit, mean over seeds | — | EMBER 30, CINDER 31, both 27 concepts; Llama A3 21, A2 10 | WP6:25-26, 39 | OK | — |
| 146 | 07_token_analysis.tex:11 | word-free efficacy | 0.000 in 18 configs, incl. r = 100 | 9 run-variants × 2 points, 0.000 (by construction: untied head); post hoc am. 11 | CE:152, 1252-1257; LM:82-91 | OK | mention "by construction" and post hoc |
| 147 | 07_token_analysis.tex:13 | efficacy close to the deletion floor at our points | — | EMBER 0.29, CINDER 0.32 vs 0.23–0.28; CINDER-z 0.39 clearly above | LM:9-13; CE:145 | AMBIGUOUS | "except the uncentred variants" |
| 148 | 07_token_analysis.tex:16 | unreadable words at operating points | 0.49–0.61 | T3 destruction 0.49–0.61 (seed 42, blind labels) | CF:107-111 | OK | add "(seed 42)" |
| 149 | 07_token_analysis.tex:16 | concept-exclusive words | 0.65–0.81 | 0.65–0.81 | CF:107-111 | OK | — |
| 150 | 07_token_analysis.tex:16 | no significant difference | — | p ≥ 0.19 | CF:115-118 | OK | — |
| 151 | 07_token_analysis.tex:17 | copy-gated efficacy | 0.06–0.14 | 0.06–0.14 | LM:113-117 | OK | — |
| 152 | 07_token_analysis.tex:17 | CINDER ~2× EMBER under copy gate | 2× | 0.12 vs 0.06; paired Δ eff +0.047, p 0.30; H3v Holm 0.42; post hoc | CE:154, 1290 | AMBIGUOUS | add "(not significant; post hoc)" |
| 153 | 07_token_analysis.tex:17 | CINDER-z advantage "came from damage to the surrounding prompt" | — | context-MC −0.041 vs real word (p 0.024); causal link untested | CE:153 | AMBIGUOUS | "is consistent with damage to …" |
| 154 | 07_token_analysis.tex:22 | nonce edit vs copy-gated subtraction | 2–3× | NM 0.18–0.22 vs E3 EMBER 0.07, CINDER-z 0.06 (2.6–3.7×), CINDER 0.14 (1.3–1.6×) | NE:8-19; CE:158 | AMBIGUOUS | "2–3× EMBER's and CINDER-z's" |
| 155 | 07_token_analysis.tex:22 | substantial nonce floor reached | 2 of 12 concepts | 2 runs (Baseball NM-free-EMBER, Heroin NM-QA), different variants; seed 42 | CE:159, 1338 | OK | label post hoc (am. 13) |
| 156 | 07_token_analysis.tex:22 | model can no longer name the word | — | renamed 0.32–0.47 vs nonce 0.17 | NE:53-73 | OK | — |
| 157 | 07_token_analysis.tex:32 | δ = 1 has r ≈ 0.6; δ = 200 selected by H | 0.6; 200 | 0.605 (recomputed); δ = 200 (ER/Harry_Potter/L0_s42, L0_s42_papergrid) | CE:290; L/erasure.md:35 | OK | remove the \note |
| 158 | 07_token_analysis.tex:34 | Prince Harry outputs (δ = 1 / 200) | quoted | match ember_d1 / ember_d200, prompt 0 | COL/premise_generations.json | OK | — |
| 159 | 07_token_analysis.tex:36 | Lego outputs | quoted | match, prompt 7 | COL/premise_generations.json | OK | — |
| 160 | 07_token_analysis.tex:38 | Trump outputs | quoted | match, prompt 4; prompt text truncated without "[…]" | COL/premise_prompts.json | OK | add "[…]" or "of the United States" |
| 161 | 07_token_analysis.tex:41 | seed 42; all three tokens in the HP feature | — | ▁Harry 17, ▁Trump 32, ▁Lego 46 (C, edited) | HP/fixed_refit-False_plain/seed42/tables.md | OK | — |
| 162 | 09_conclusion.tex:9 | one fit matches per-language EMBER at 1/8 cost | — | A3z tie, p 0.78 | ml: RESULTS.md:230 | OK | — |
| 163 | 10_statements.tex:6 | 4-bit transformer, fp16 embeddings | — | NF4 + fp16 | ml: RESULTS.md:106 | OK | — |
| 164 | 10_statements.tex:15 | no human spot check | — | "[TBD, optional]" | CE:448 | OK | — |
| 165 | 10_statements.tex:18 | 11–12 concepts; differences < ~0.08 hard to resolve | 0.08 | "+0.08 to +0.13 reaches uncorrected, not Holm" | CE:175 | OK | — |
| 166 | 10_statements.tex:37 | Gemini 3.5 Flash-Lite as judge and lexicon generator | — | also generated and verified the held-out multilingual questions, and judged Alpaca coherence | ml: RESULTS.md:178; CE:254 | AMBIGUOUS | list all Gemini uses |
| 167 | appendix/A_ember.tex:8-12 | λ, s, stop rule, k, √d scaling | 1e-4; 0.01; 200 it, 0.5%, 20,000; k = 100/200 | tol 1e-4, patience 200, churn 0.005, max 20,000; √d in prep | experiments/varwidth/common.py:48-55; prep.py:91 | OK | — |
| 168 | appendix/A_ember.tex:16 | \|V\*\| for HP | 4,780 | 4,780 | HP/*/seed42/summary.json | OK | — |
| 169 | appendix/A_ember.tex:23 | re-judging flips | 230; 0.4% | 230; 0.004 | L/layers.md:72 | OK | — |
| 170 | appendix/A_ember.tex:26 | fidelity | 48.4/83.2/55.1 vs 52.1/84.6/53.6 | same | L/erasure.md:39 | OK | note seed 42 vs 5 seeds |
| 171 | appendix/A_ember.tex:27 | Alpaca-in-grid H | 0.727 vs 0.726 | 0.727 vs 0.726 | L/erasure.md:35 | OK | — |
| 172 | appendix/A_ember.tex:28 | all numbers use the repaired scorer | all | Part-1 leakage numbers use the original scorer | P1:280 | MISMATCH | add the exception, or re-quote the robust values |
| 173 | appendix/B_contrastive_sharpening.tex:9 | background types | 107,472 | n = 107,472 | experiments/whitening/data/gemma/bg_wiki_stats.pt | OK | — |
| 174 | appendix/B_contrastive_sharpening.tex:14 | HP page types | 3,124 | 3,124 | experiments/whitening/data/gemma/targets/Harry_Potter.json | OK | — |
| 175 | appendix/B_contrastive_sharpening.tex:15 | η and γ chosen by the sweep | 5 η values | only η swept, γ fixed at 1 | defaults.json chosen_by | AMBIGUOUS | "η was chosen by …; γ = 1 was fixed" |
| 176 | appendix/B_contrastive_sharpening.tex:41 | whitening alone finds no HP feature | — | see row 42 | WP1/features_all.csv | AMBIGUOUS | same fix as row 42 (counted there as NOT FOUND) |
| 177 | appendix/B_contrastive_sharpening.tex:62-64 | leak-free table (9 numbers per row) | — | all match | defaults.json | OK | — |
| 178 | appendix/C_global_sparsity.tex:23 | m_min, m_max on HP | 10, 144 | ⌈0.002·4780⌉ = 10, ⌈0.03·4780⌉ = 144 | common.py:49-50 | OK | — |
| 179 | appendix/C_global_sparsity.tex:24 | cap never reached | 116 runs / 24 concepts | 190 runs / 38 concepts, 0 at cap (recomputed) | results/varwidth/*/rn_refit-False/seed*/summary.json | MISMATCH | update to 190 / 38 |
| 180 | appendix/C_global_sparsity.tex:45 | Lasso tuned to K nonzeros; never clean AR in 5 seeds | — | λ bisected to ~K nonzeros; never clean | experiments/varwidth/lasso.py:1; CI:221 | OK | — |
| 181 | appendix/C_global_sparsity.tex:57,59-61 | widths, clean runs, neutral share, Jaccard | 20 numbers | all match | overall_report_cap0.05.md:12-16 | OK | clean /90 counts Heroin's 5 untracked runs as not clean |
| 182 | appendix/C_global_sparsity.tex:58 | "Width, seed s.d." | 0/0/4.1/8.1/7.9 | values match; statistic is mean abs. deviation | overall_report_cap0.05.md:6 | MISMATCH | rename "mean abs. deviation" |
| 183 | appendix/C_global_sparsity.tex:68 | raw GS 11–44% vs RS 15–25%; CS 1–9% | — | match (3 concepts) | whitened_vs_plain.md:9-33 | OK | — |
| 184 | appendix/D_two_way_judge.tex:39,41 | 23 × 5, 7,105 candidates; 19 concepts, 95 runs | — | as stated | CI:280 | OK | — |
| 185 | appendix/D_two_way_judge.tex:52-56 | judge table (30 numbers) | — | all match the j3_eval table | CI:284-288 | OK | L/layers.md:97-106 differs slightly (other pool) |
| 186 | appendix/D_two_way_judge.tex:63 | held-out 15: EJ@5, TWJ, TJ∨WJ+share, hits | .46/.26; .53/.56; .71/.89; 49 vs 33 of 75 | .462/.257; .527/.557; .713/.886; 49 vs 33 | L/layers.md:114-122 | OK | — |
| 187 | appendix/E_edit_space.tex:29,41-43 | ≤0.008 (137 runs); ≥0.9995 (137 / 21); β 0.66/0.56/0.05, 1.06; 0.917 vs 0.932 | — | all match | WP6:44-47, 87, 93, 106, 165 | OK | — |
| 188 | appendix/E_edit_space.tex:55-58 | read-decomposition table (16 numbers) | — | all match; rows use n = 27 / 30 / 31 concepts | WP6:53-57, 80 | OK | state the n per row |
| 189 | appendix/E_edit_space.tex:63-66 | +0.68 (27/27), p 0.44, 4.3 vs 4.4, Spearman .75/.63, deciles, +4.48/+4.34, 20.4×, 7% | — | all match | WP6:53, 59, 64, 69, 106-107 | OK | — |
| 190 | appendix/F_multilingual.tex:8-9 | leakage 0.24 [0.15, 0.32], 0.41; SNMF +0.01; SNMF+EMBER +0.16 | — | match; original scorer | P1:223-224, 280 | AMBIGUOUS | add the scorer caveat (see M9) |
| 191 | appendix/F_multilingual.tex:13-23 | NF4/fp16; 30 + 30; NLLB/MADLAD/LaBSE; 100 held-out sentences; eligibility 0.4; gates 0.9 | — | all match | ml: RESULTS.md:106; PLAN.md:31; results/ml/data/qa_v2/CLEANUP.md:10-17; OBSERVATIONS.md:18 | OK | gated δ sweep stops at the first gated δ (OBS:17), not stated |
| 192 | appendix/F_multilingual.tex:44 | seed union +0.08, 19–4, p < 0.001; pooled 0.30 vs 0.31 | — | A1U 0.302, A3zU 0.313 | ml: UNION_RESULTS.md:10, 23 | OK | — |
| 193 | appendix/F_multilingual.tex:48 | Baseball concept-only shares; edited-token question share | 32/13/6/5%; 0.94 → 0.00 | match (A2 token set; A3 set on QA_train) | ml: OBSERVATIONS.md:28, 32 | OK | note the two token sets |
| 194 | appendix/F_multilingual.tex:49 | Golf Thai feature covers English questions | 76% | 76% | ml: OBSERVATIONS.md:42-43 | OK | — |
| 195 | appendix/F_multilingual.tex:52-53 | seed counts 9–12 / 17–21; Jaccard ≈ 0.2; 6 gate failures | — | match | ml: COVERAGE.md:158-162; OBSERVATIONS.md:97; RESULTS.md:174 | OK | — |
| 196 | appendix/G_evaluation.tex:9,48 | 3,501 / 1,026; 10% duplicates; agreement 0.92–0.97 / 0.97–1.00 | — | match | CE:1135, 1137; CF:103; NE:87, 99; LM:17 | OK | — |
| 197 | appendix/G_evaluation.tex:60,70-71 | 2×2 cells .20/.39, .26/.46, .20/.39, .26/.48; runs | — | .201/.389, .256/.456, .204/.389, .263/.475; ee on 41 runs, others 38 | TB:9-12 | OK | "effects on the 38 runs with both sets" |
| 198 | appendix/G_evaluation.tex:76 | direction +0.073 / +0.097, 10 of 11, Holm .097/.055; token +0.016 / +0.024; collateral cost 0.02 | — | match; post hoc (am. 14) | TB:16-20; CE:1401 | OK | label post hoc |
| 199 | appendix/H_analyses.tex:31-38 | WP5 appendix (spot check): 734; 0.18–0.27 vs 0.065; 0.49–0.69; 15–22 at 0.22–0.25, ~37% unrelated; 3.9 (EMBER) admitted S; 0.28–0.30 at one occurrence | — | match where checked (n_Cstar 734; cP10; cPSA10 0.493–0.694; n_admit 15.3, prec 0.224–0.251, U 0.35–0.39; nS 3.86; S_concept_mean 0.290–0.297) | WP5/summary.json; WP5/extra.json | OK | not checked: 5,196 labels, κ 0.82, AUC 0.40–0.48 and 0.91 vs 0.67, 6.1, 13–15%, 48 → 59 |
| 200 | appendix/I_examples.tex:8,19-26 | HP feature token table; "2 of 11 … 11 of 11" | — | matches the notebook fits; the Fig. 2 fit gives 3 of 11 raw, and its CS feature has N = 0 (no NHP; ends Cedric, Narnia) | ember_three_steps*.ipynb cells 42/48; HP/*/seed42/tables.md | MISMATCH | regenerate from HP/*/seed42/tables.md, or cite the notebook fit and drop "no neutral mass" consistency claims |

Not audited as quantitative claims: App. J paths; `\pending{}` and `\expfill{}` slots without numbers. App. J:18, about 16 NIS (single-language
ladder, L/SUMMARY.md:9: 16.4) and about 17 NIS (multilingual, ml: RESULTS.md:183), is consistent with its sources. It leaves out the Gemini
spent on the collateral evaluation (Alpaca coherence: 8.3 + 5.7 NIS; CE:254, total 37.5 NIS shared).
