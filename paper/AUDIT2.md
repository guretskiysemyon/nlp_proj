# Second number audit of the CINDER draft (paper/)

Audited 2026-10-23 against the uncommitted working tree on top of `cc68637d`. Read-only: no `.tex`, figure or result
file was edited, no GPU job and no paid API was run.

**Scope.**
- A. Every number and quantitative claim in the main text (`sections/00`–`09`, with table cells, the figure captions
  and Fig. 1's TikZ boxes). Each was compared with the pre-trim copy in `drafts/sections_before_trim/`.
- B. Every `\expnum` and quantitative claim on the appendix lines added or changed since `cc68637d` (App. B, D, E, F,
  G, H, I, J), including the captions of `fig:integrity`, `fig:twobytwo`, `fig:read` and `fig:extrapolation`.
- C. Consistency between the main text and the appendices.
- D. The novelty and cost claims.
- Skipped: `\pending{WP6b ...}` slots, `10_statements.tex` (one line noted) and the AI-disclosure draft.

**Recomputed (CPU, read-only, `.venv/bin/python`).**
- Table 2 cells from `COL/*/s*/sub_*/result*.json` via `confirm_stats.load`; 14 vs 3 runs and McNemar p.
- The 11 `wordfree.json` files.
- Figure 3 inputs (confirm.json, last_mile.json) and the PNG.
- Integrity curves from the 255 per-run `grid.json` / `copygrid.json` files, which reproduce `integrity_data.json`
  exactly.
- Gate-row Holm p from uncorrected p; the leave-one-out denominator from `confirm_stats.py`.
- Nonce attainment for Heroin and Baseball from `last_mile/*/floor.json`.
- 2×2 ee point on 41 vs 38 runs.
- TJ rejections and held-out judge table: `paper/drafts/tj_rejections_heldout.py`, run read-only.
- Raw-fit `judge.json` verdicts for Heroin and WWII.
- Circularity Holm p and ρ-ranks (`circularity/analysis.json`); Tolkien and Gandalf positions
  (`robustness/features.parquet`).
- Multilingual: seed coverage split by concept set, dev-table means, TJ acceptance by lexicon-hit bin
  (`ml_geometry/features.csv.gz`), fit × judge cells, per-language ratios, Llama read medians.
- Gemini spend by tag (`results/gemini_usage.jsonl`, 37.50 NIS; `ml: gemini_usage.jsonl`, 17.11 NIS).
- GPU-hours by job prefix (`sacct -u oripetel -X`), and local multilingual worker minutes from
  `ml: results/ml/logs/worker_*local*.log` (50 units, 9,065 min).

**Source abbreviations.**
- As in `paper/AUDIT.md`: CE, CI, L/, COL/, CF, LM, TB, NE, ER, HP/, WP1, WP2, WP5, WP6, ml:, P1, EMBERpaper.
- Analysis reports under `results/paper_analysis/`: WP3 = `widths/`; WP4 = `robustness/`; WP7 = `circularity/`;
  WP8 = `llama_cs/`; WP9 = `ml_geometry/REPORT.md`; WP6b = `mean_edit/REPORT.md`.
- Collateral study: CSPL = `COL/coverage_split.md`; POC = `COL/poc.md`; PR = `L/PREREG.md`.
- Multilingual worktree: ml:PR = `ml: results/ml/PREREG_RESULTS.md`; ml:UR = `…/UNION_RESULTS.md`;
  ml:RT = `…/REPORT_tables.md`; ml:RES = `ml: experiments/ml/RESULTS.md`; ml:OBS = `…/OBSERVATIONS.md`;
  ml:PN = `…/PREREG_NEW.md`.
- Paper-side files: IDJ = `paper/figures/src/integrity_data.json`; JR = `results/varwidth/judge_rejected_examples.md`;
  TJR = recount with `paper/drafts/tj_rejections_heldout.py`.
- Line numbers refer to the current files.

## Summary

There are 279 checklist rows: **OK 207, MISMATCH 14, UNSUPPORTED or missing qualifier 14, AMBIGUOUS 44.** The 14
UNSUPPORTED rows are 12 distinct issues, because U2 covers rows 19, 115 and 119.

Nine of the 14 mismatches (M1–M9) are in the main text. Three of them (M1, M6, M8) and one unsupported claim (U11)
come from the 8-page condensation: a qualifier or comparison present in `drafts/sections_before_trim/` was dropped
or re-paired. M3 was already in the pre-trim text.

All 13 items of the first audit (M1–M12, N1) are resolved in the current text, except M6 (post hoc labelling), which
is only partly fixed and reappears as M4.

### MISMATCH (fix before submission)

| id | file:line | paper says | source says | source | suggested replacement (same length) |
|---|---|---|---|---|---|
| M1 | 01_introduction.tex:55 (also 03_motivation.tex:32) | "…only 10 of 38 concepts, since language identity carries far more energy than the concept" | Coverage is lost mostly at the judge. The raw fit has a lexicon-matching feature in 25–27/38 concepts per seed (CS: 23–28), but TJ rejects it. §5.3:104 says the same. The pre-trim intro had both clauses. | WP9:44, 63-67, 96 | "…only 10 of 38 concepts, since its judge rejects most concept features the fit finds." In §3:32: "…on Llama-3.1-8B, mostly at the judge; the eight language means carry 34 times the concept's energy, itself about 3 times quieter than in English." |
| M2 | 02_preliminaries.tex:34 | "…and robustness to relearning, normalized by the unedited model; their harmonic mean H selects δ" | H aggregates inverted concept accuracy, similar-domain, MMLU and both AlpacaEval scores. Relearning is measured only on the final configuration. | EMBERpaper:134, 148 | "…and robustness to relearning \citep{deeb2024unlearning}; the harmonic mean $H$ of all but relearning, normalized by the unedited model, selects δ…" |
| M3 | 03_motivation.tex:21 | "Removing the shared component leaves a pure but ordinary cluster, which the factorization finds less often than before (Fig. 2b)" | Only whitening gives this: seeded purity 0.86, 70th percentile value, found in 49% vs 56% raw. With μ̂_B projected out, the seeded feature has purity 0.27 and value at the 97th percentile, and is found more often (61%). App H:18-19 states these numbers. | WP1:56-60 | "Whitening removes the shared component but leaves a pure, ordinary cluster, which the factorization finds less often than before (Figure~\ref{fig:mechanism}b)" |
| M4 | 05_experiments.tex:27 | "analyses defined after earlier results were seen are marked post hoc, all others were pre-registered" | Several analyses are unmarked but post hoc or exploratory: see the note after this table. | PR:388, 429, 552; ml:PN:42-49 | "pre-registered contrasts are the primary, fixed-$r$ and $H$ ones and H1--H3; all others are post hoc." |
| M5 | 05_experiments.tex:46 (Table 2, EMBER, $H_3$) | Eff. 0.30, Coll. 0.86 | 0.2947 and 0.8547 over 41 runs (recomputed); LM:9 prints 0.29. CE:1162's 0.30 / 0.86 is a double rounding. | COL/*/s*/sub_ember/result.json; LM:9 | `\expnum{0.29}` and `\expnum{0.85}` |
| M6 | 05_experiments.tex:104 | "but EMBER's judge accepts only 5% of such candidates, against 30% for CINDER's" ("such" = lexicon-matching features) | 5% vs 30% holds only for candidates with 3–4 lexicon words. For lexicon-matching candidates (≥ 5 hits), TJ accepts 41–70% (A2) vs 48–93% (A3) by bin, 52% vs 63% pooled (recomputed). The pre-trim text had the qualifier. | WP9:67-73; ml_geometry/features.csv.gz | "but EMBER's judge accepts only 5% of those with 3--4 lexicon words, against 30% for CINDER's" |
| M7 | 05_experiments.tex:108 | "Hindi, whose words split into pieces that are never concept-only" | 5% (Baseball) and 4% (Gun) of Hindi question tokens are concept-only. App F:194 itself says 5%, and Tuberculosis A1U erases Hindi (App F:137). | ml:OBS:31-33; App F:194 | "…whose words split into pieces that are rarely concept-only" |
| M8 | 06_analysis.tex:31 | "reads EMBER's and CINDER-z's edits about twice as strongly as CINDER's" | Relative to a random direction: EMBER 1.86, CINDER-z 1.99, CINDER 1.18. That is 1.6–1.7× CINDER's, and about 2× a random direction. The pre-trim text compared with a random direction. | WP6:53-54 | "reads EMBER's and CINDER-z's edits at about twice a random direction, CINDER's at 1.2×" |
| M9 | 07_token_analysis.tex:10 | "efficacy 0.000 in all 18 configurations checked" | 11 run-variants × 2 points = 22 configurations, all 0.000 (11 `wordfree.json` files). The "9 run-variants / 18" in CE F36 predates the Golf s45 runs. | LM:82-91; COL/{Golf/s44, Golf/s45, Gun/s43}/sub_*/wordfree.json | "…in all \expnum{22} configurations checked…" |
| M10 | appendix/G_evaluation.tex:80 (tab:robust caption) | "…of the uncorrected p over the 12 omitted concepts" | The leave-one-out loop runs over the 11 concepts that have pairs. | experiments/varwidth/confirm_stats.py:93-100; CF:9 | "…over the 11 omitted concepts" |
| M11 | appendix/G_evaluation.tex:87 | "…did not hold, so we quote only held-out numbers in the main text" | The main text quotes development-concept results: §3:38 (HP, δ = 200, 12 of 20 prompts) and Table 4 (HP, seed 42). | CE:338-344 | "…did not hold, so the main text compares methods only on held-out concepts" |
| M12 | appendix/G_evaluation.tex:223 | "the free edits repeat the word in only 0.47–0.49 of the answers" | The echo rate is 0.47 (free), 0.49 (+QA) and **0.34** for the third free edit (EMBER's tokens). The 32–47% in the same sentence does include that edit. | NE:41-47 | "…in only 0.34–0.49 of the answers…" |
| M13 | appendix/J_reproducibility.tex:20, 42 | "its seed-42 fits ran on a workstation GPU that the accounting does not record"; Multilingual 320 GPU-h | 320 is the SLURM ml_* total (319.9, recomputed). About 151 GPU-h of evaluation also ran unrecorded on the local c-001 GPUs: 50 units, 31 at seed 42 and 19 union units. | sacct; ml: results/ml/logs/worker_local_gpu*.log, worker_u_local_*.log | "its seed-42 fits and about 150 GPU-hours of evaluation (50 units) ran on workstation GPUs that the accounting does not record". In the table, mark 320 with "≈ +150 local". |
| M14 | appendix/J_reproducibility.tex:43, 45, 48 | Analyses "10⁺", "⁺: one job still running"; Total 695 | `cl_pa_mean` has finished; SLURM analyses total 10.7 GPU-h. The rows sum to 696 (696.1). WP1, WP4, WP7, WP8 and part of WP6b also ran on unrecorded local GPUs (WP6b reports ~16.5 GPU-h, vs 10.5 on SLURM). | sacct; WP6b:3 | Analyses `\expnum{11}`, with the note "SLURM only; plus local GPU time"; Total `\expnum{696}`. |

**M4 detail.** Unmarked post hoc or exploratory analyses in the main text:
- §5.2:34 and Table 2's Broken column: prompt damage and the deletion floor (am. 9, "post hoc", PR:388).
- §7:10: word-free efficacy (am. 11, post hoc, PR:429).
- §7:11: the deletion-floor comparison (am. 9).
- §7:15: context damage (am. 11).
- §5.3:79 and :103: the A3z−A1 and coverage-counted A3z−A2 score contrasts (not among H1–H3, ml:PN:44-49).
- §5.3:104: the WP9 re-judging.
- §6: every analysis except circularity, which alone is called pre-registered.

### UNSUPPORTED or missing qualifier

| id | file:line | paper says | source says | source | suggested replacement (same length) |
|---|---|---|---|---|---|
| U1 | 00_abstract.tex:5 | "…to eight languages with a single fit, matching per-language EMBER with an eighth of the judge calls" | The tie needs the post hoc CINDER-z edit: A3z 0.23 vs A1 0.25, p 0.78, an exploratory contrast. With CINDER's own edit, A3 scores 0.14. The pre-registered non-inferiority test (H4, pooled seeds) is not met (CI lower bound −0.114 < −0.05). | ml:PR:48, 64; ml:UR:18; ml:PN:81-83; App F:136 | "…to eight languages with a single fit that, with the \cinderz{} edit, ties per-language \ember{} at an eighth of the judge calls." |
| U2 | 01_introduction.tex:56; 05_experiments.tex:79, :80 | "a single CINDER-z fit … erases as well as eight separate EMBER runs"; "ties per-language EMBER (A1) … (p = 0.78)"; "and the two still tie" | A3z−A1 is not among the pre-registered H1–H3. The one pre-registered non-inferiority test (H4, A3zU vs A1U) failed, which App F:136 states and the main text does not. | ml:PN:44-49, 81-83; ml:UR:3, 18 | 01:56: "…ties eight separate EMBER runs (no significant difference), with…". 05:79: "…ties per-language EMBER (A1; exploratory): …". 05:80: "…and the two still tie (non-inferiority not shown)." |
| U3 | 03_motivation.tex:27 | "clean Heroin and World War II features became 'pharmacological substances […] and addictive drugs' and '20th-century totalitarian regimes'" (§3 is about EMBER) | The quoted features come from the **sharpened** fits with RS-1% (`fixed_refit-False`), which App I:41 states. EMBER's raw fits fail the same way: Heroin s42 f56 "substances, medications, and illicit drugs […] pharmacology and addiction"; WWII s43 f47 "20th-century totalitarian political ideologies, regimes […]"; both False, 0.85 (recomputed). | JR:5-179; results/varwidth/{Heroin/…/seed42, World_War_II/…/seed43}/judge.json (`fixed_refit-False_plain`) | Quote the raw-fit descriptions: "Heroin and World War II features became ``substances, medications, and illicit drugs'' and ``20th-century totalitarian political ideologies''". Alternatively keep the quotes and add "(sharpened fit)". |
| U4 | 04_method.tex:26 | "When a page yields few directions above sampling noise, as on Llama, we sharpen only those (§B.3)" | The noise-edge rule was tested only in WP8's judge-free single-language fits. The reported Llama fits (multilingual A3) sharpen with the full $S_\eta$ over the pooled 8-language target, with no truncation. | WP8:168-198; App B:121-125; ml: experiments/ml/fit.py:29, 96 | "On Llama a single page yields few directions above sampling noise; our multilingual fits pool eight languages' pages (\S\ref{app:cs-llama})." |
| U5 | 05_experiments.tex:17 | "We keep EMBER's metrics (§2; …)" (§2 lists relearning) | Relearning is never tested, as the Limitations admit (10_statements.tex:13). | 10_statements.tex:13 | "We keep \ember{}'s metrics except relearning (\S\ref{sec:prelim}; …)" |
| U6 | 05_experiments.tex:26 (also Table 2 rows :45, :47 and caption :51; Fig. 3 x-label) | "we report EMBER's H" | The "paper" rule is max HM(efficacy, specificity) under an MMLU gate, with no Alpaca terms (App G:25). EMBER's H puts MMLU and both Alpaca scores inside the HM. The two were shown equivalent on HP only (H 0.727 vs 0.726). | App G:25; App A:27; EMBERpaper:134; CE F28 | "(besides our rule, we report \ember{}'s $H$ without Alpaca and a fixed $r = 0.5$)" |
| U7 | 05_experiments.tex:104 | "re-judged, TWJ alone lifts the raw fit from 10 to 18 concepts, and sharpening alone to 16 (seed 42; §F)" | This is a post hoc re-judging of the saved fits (WP9). It is counterfactual: no edit was evaluated (tab:ml-2x2 caption). | WP9:79-84; App F:178 | "…(seed 42, post hoc; \S\ref{app:ml})" |
| U8 | 06_analysis.tex:24 | "(+0.06 vs. +0.05; p = 0.003, 0.009; pre-registered)" | These p are uncorrected. The pre-registered test is Holm over the primary family: disjoint 0.034, related 0.081 (n.s.). The page's own gain is n.s. (p 0.088, Holm 0.62). | WP7:48-54; WP7/prereg.json ("unit_and_tests") | "(+0.06 vs. +0.05; Holm $p = 0.034$, $0.081$; pre-registered)" |
| U9 | 06_analysis.tex:27 | "an abstract concept is written in a register shared with its parent topics, not around a vocabulary of names" | Listed under Speculations, with a proposed test. The one probe run is null (proper-noun share does not predict success, p 0.58). | WP4:128-139 | "…their top features fall on parent topics and discourse words, perhaps a shared register (untested)." |
| U10 | 07_token_analysis.tex:15 | "CINDER-z's advantage disappears, since part of it came from damage to the surrounding prompt" | The held-out evidence is a post hoc association: context-MC −0.041 vs the real word (p 0.024) and −0.027 vs EMBER (p 0.10); CE F37 says "comes with". WP6b's mean-only edit supports the mechanism, but descriptively, on 3 development concepts. | LM:128, 135; CE F37; WP6b:7-11 | "…disappears, as part of it came with damage to the surrounding prompt (post hoc)" |
| U11 | 07_token_analysis.tex:19 | "removes 2–3× as much as EMBER's subtraction while keeping the words copyable" | 2–3× holds only against copy-gated EMBER (0.07). At its own operating point EMBER removes 0.29, more than the nonce edit's 0.18–0.22. App G:222 says "integrity-gated"; the pre-trim text had "at readable strength". | NE:8-11, 17; LM:9 | "removes 2--3× as much as copy-gated \ember{} while keeping the words copyable" |
| U12 | appendix/G_evaluation.tex:53 (unchanged; heads the new tab:robust) | "Every analysis plan was written into a pre-registration with amendments before its results existed" | Am. 2 was written after its results; ITT was computed while am. 9 was written; McNemar is in no pre-registration. | PR:90, 388-390 | "Analysis plans were written into a pre-registration and amendments, nearly all before their results existed." |

### AMBIGUOUS (wording, scope or a minor missing caveat; one line each, with fix)

These are the 44 checklist rows marked AMBIGUOUS. A minor qualifier lands here, not under U, when adding it would not change whether the claim is true. Bullets marked "not counted" are housekeeping notes outside the checklist.

Main text:
- **00:7 and 07:10.** "Revealed to block access": word-free efficacy is 0 by construction, since the output head is untied (CE F36). Fix in §7: add "(by construction: only input rows change)".
- **01:26 (Fig. 1) and 01:63 (2).** "One LLM call per concept" is true of the lexicon path only; TWJ still runs TJ, 2 calls per candidate. Fix: "(lexicon: one LLM call per concept)".
- **01:46.** "The mean of all embeddings": μ_B is the mean over 107,472 Wikipedia token types. Fix: "the mean of ordinary-text embeddings".
- **01:47.** "Without this component … clean but quiet cluster, of ordinary value": the data are for whitening, as in M3. Fix: "Whitened, the concept forms a clean but quiet cluster…".
- **01:58.** "Evaluating CINDER on Gemma and Llama exposed a blind spot": the collateral study ran on Gemma only. Fix: drop Llama from this sentence.
- **03:14.** "Tested each observation on 38 concepts and five seeds": the judge used 19 concepts, the destruction anchor 14 concepts at seed 42 (and 12 held-out), and the English-only edit the Llama study. Fix: "tested most observations on 38 concepts".
- **03:15.** "Five issues, which Figure 1 pairs with CINDER's changes": the rows are not one-to-one, since rarity and width both map to Decompose and Erase has no issue. Fix: "maps to".
- **03:20.** "42% of its norm": this is the squared norm. App H:56 gives 45% for the same quantity under WP2's identification rule. Fix: "42% of its squared norm", and use one number in both places.
- **03:38.** "20 prompts that use the edited words in other senses": most do, but a few contain no edited token (CE:338). Fix: "20 prompts, most using the edited words in other senses".
- **04:32.** "The exact projection onto the constraint": it is exact for the row-share score of Alg. 1. Fix: "…onto the constraint for this score".
- **04:59.** "Weakens the language directions about threefold": this is relative to an English background (enrichment 8.2 → 2.6); against the raw matrix the language share halves (WP9:48-57). Fix: "about threefold against an English background".
- **04:60.** "Wall-clock time … about the same": true on GPU (40 s vs 44 s), but 76 s vs 39–40 s on CPU (WP9:123-124). Fix: "GPU time".
- **05:27 vs 05:56.** The concept is stated as the unit, but McNemar is run-level. Fix: "(run-level exact McNemar …)".
- **05:33.** "Edits a hundred times the token's length": only EMBER and CINDER-z reach r = 100; CINDER-RS-z is at 30 and CINDER, CINDER-RS at 1 (CE F28). Fix: "…for EMBER and CINDER-z".
- **05:34.** "The edits derail the surrounding prompt": context damage at r = 100 is shown on development concepts only (CE F23). Fix: "likely derail".
- **05:35.** "Under H … EMBER ahead of CINDER": H2 −0.134, Holm 0.056, not significant (CF:43). Fix: add "(n.s.)".
- **05:60.** Δdir +0.07 is significant before Holm only (p 0.032, Holm 0.097; TB:16). Fix: add "Holm $p = 0.10$".
- **05:61.** "Secondary": this is the secondary endpoint of post hoc amendment 14 (PR:552), but it reads as pre-registered. Fix: "secondary endpoint, post hoc".
- **05:65.** "EMBER catches up with larger edits": both methods sit at median r = 1. Fix: "both move to r ≈ 1, where the direction no longer matters".
- **05:66.** Only gate 0.9 is quoted; at gate 0.7 no variant wins (Holm ≥ 0.27; tab:robust). Fix: add "(not at 0.7)".
- **05:93, :98 (Table 3).** Fit seed 42 is not stated, and A1's 26 includes Karate and Prime number, which have features but no admissible δ (ml:RES:140). Fix: add "fit seed 42" to the caption.
- **05:107.** "+0.17 (16 of 21)" is pooled over all 38 concepts, including the 18 on which A3z was defined. Fix: "(16 of 21, over 38 concepts, …)".
- **06:13.** "97% of the raw feature's excess norm": this is the excess squared norm, and the number is not in App H (WP2:97). Fix: add it to App H.2, or say "squared norm".
- **06:21.** "49 vs. 33 of 54": 49 is TJ ∨ WJ+share and 33 is TJ; the lexicon path itself scores EJ@5 15 and TWJ 35 (App D:80-84). Fix: "(TJ with source words 49, TJ 33 of 54 runs)".
- **07:10.** "Knowledge remains in the weights" is true by construction and comes from post hoc amendment 11. Fix: as for 00:7.
- **07:11.** "Efficacy lies close to the floor": the -z variants are clearly above it (0.39 / 0.38 vs 0.28 / 0.26; LM:9-13). Fix: "…except for the -z variants".
- **07:14.** The T3 numbers are seed 42 only (CF:101-118). Fix: add "(seed 42)".
- **07:15.** "About twice as much as EMBER": paired H3 +0.096, Holm 0.42, not significant (LM:106). Fix: add "(n.s.)".
- **07:19.** "2 of 12 concepts" is 2 runs from different variants (Baseball free-EMBER, Heroin +QA), seed 42. Optional fix: add "(seed 42)".
- **07:35** (OK row, housekeeping). Table 4's `\note`: both facts are verified (median r 0.605 at δ = 1; δ = 200 chosen in ER/Harry_Potter/L0_s42). Delete the `\note`.
- **10_statements.tex:22** (not counted; outside the scope; it depends on the pending WP6b sentence). "The mean term … erases mostly by damaging the prompt" rests on WP6b, which is descriptive and covers 3 development concepts only. Fix: state that scope.

Appendix:
- **E:14 (fig:read).** "Centring EMBER's dictionary brings its read strength down to CINDER's" holds on Gemma (1.09 vs 1.18). On Llama, centred A2 stays at 1.70 vs A3 1.24 (p 0.014; WP6:96-108). Fix: "down to near CINDER's".
- **F:48.** "Counting concepts that only one approach edits": the set is the concepts *either* approach edits (n 21, 29). Fix: "Over the concepts either approach edits".
- **F:163** (unchanged line). "Such features … 5% vs 30% at 3–4 lexicon words": this is correct for the bin, but "such features" refers to lexicon-matching ones (see M6). Fix: "candidates with 3–4 lexicon words".
- **G:80.** "12 held-out concepts": the Primary and gate rows use 11 concepts; only the All-60 row uses 12. Fix: "(11 concepts where both edit; 12 for all runs)".
- **G:109.** The development claim compares with the parent, but the held-out cell compares CINDER-z with EMBER. The verdict holds. Optional fix: "CINDER-z +.13 vs EMBER, CINDER +.04".
- **G:122.** Mean's copy, destruction and context numbers come from 3 labelled runs, which the text does not say (CE F15). Fix: add "(3 labelled runs)". The same applies to Noise at G:123.
- **G:153.** "Crosses its deletion floor": the curves are validation efficacy, the floor is test. Fix: "its (test) deletion floor".
- **G:160.** "On the pre-registered paired endpoint": the integrity rule itself is post hoc (am. 11). Fix: "On the paired endpoint of the primary analysis".
- **G:193 (twobytwo.pdf).** The ee point is the 41-run mean (0.201); on the 38 full-2×2 runs it is 0.192 (recomputed). Fix: plot ee on the 38 runs, or add "ee: 41 runs".
- **H:20.** "When a raw fit has a lexicon concept feature … median rank 2": the median conditional on existence is 1. The 2 vs 1 (p 0.92) is for the max-lexicon feature over all runs (WP1:65). Fix: "The raw fit's best lexicon feature usually ranks near the top (median 2 vs. 1 under CS)".
- **H:77.** "Keeps them at positions 16–18": there is no Hobbit in CS's top 20, and Tolkien and Gandalf sit at 16–20 over seeds (recomputed). Fix: "keeps only Tolkien and Gandalf, at positions 16–20".
- **H:97.** "For in-sample and extrapolated tokens alike": 0.59–0.71 at ten or more occurrences is in-sample only, since that bin of new tokens has fewer than 5 concepts. Fix: "…(in-sample); at one occurrence both are alike".
- **I:38.** The Trump prompt is truncated without "[…]"; the original ends "…President of the United States?". Fix: "…first elected President [\dots]?".
- **J:18.** "One GPU per job (one used two)" holds for 17–23 Oct, but the 3–8 Oct pass-2 jobs named in J:23 used 3 GPUs each. Fix: "(Table~\ref{tab:cost} jobs)".
- **J:20.** "Under a minute per concept for selection": only the fit stage was timed (WP9:152). Fix: "for the fits".
- **Cosmetic** (not counted). In Fig. 2 (mechanism.png), the panel (b) title is clipped at the right edge ("…sharpenec").

### Novelty and cost claims (item D)

| Claim | Where | Verdict | Evidence |
|---|---|---|---|
| "to our knowledge, the first multilingual erasure at the embedding level" | 01:56 | OK | EMBER leaves languages to future work (EMBERpaper:249). Our own Part-1 report fitted a joint 5-language EMBER and counted edited words but evaluated no erasure (P1:307-314); this matters only if Part-1 is cited. |
| "an eighth of the judge calls" | 00:5, 01:56, 01:63, 09:7 | OK | 160 vs 20 judge calls per concept, exactly 1/8 (WP9:126, 148). |
| "about a fifth of the factorization work" | 04:60 | OK | rows × iterations 11.0 M vs 2.4 M (1/4.7; A2 1/5.0) (WP9:122, 149). |
| "a third of the LLM cost" | 04:60 | OK | 0.0174 vs 0.055 NIS = 31%, or 12.5% with the lexicon cached (WP9:129, 150). |
| "wall-clock time and evaluation … cost about the same" | 04:60 | AMBIGUOUS | GPU fit 40 s vs 44 s, CPU 76 s vs 39–40 s; evaluation 2.5–3.5 GPU-h per unit, the same for all (WP9:123-124, 152). |
| "no LLM judge in the tuning loop" | 01:63 | OK | H3 uses MC accuracies and the MMLU gate; Alpaca is scored only after selection (seed 42); questions and labels were made in advance (PR:340-374; LM:41). |
| "(one LLM call per concept)" / "a lexicon judge that needs one LLM call per concept" | Fig. 1; 01:63 | OK for the lexicon / AMBIGUOUS in Fig. 1 | One lexicon call per concept, cached (ml: select.py:10; WP9:127). TWJ still runs TJ at 2 calls per candidate (CI:255). |
| "matching per-language EMBER" | 00:5 | UNSUPPORTED (U1) | The tie needs post hoc A3z; non-inferiority not shown. |

### Status of the first audit (paper/AUDIT.md)

All of these are resolved in the current text:
- M1: §6:32 now quotes 2.9× and 20× against a random direction.
- M2, M3, M12: §3 was trimmed, and App I:15 now labels its table as the notebook fit.
- M4: the Ancient Rome sweep is out of the main text.
- M5: 0.92.
- M7: the ITT claim moved to App G, with Holm 0.10.
- M8: App H:57 now says 3rd–6th.
- M9: App A:28 now has the exception.
- M10: App C:24 now says 190 / 38.
- M11: App C:66 now says MAD.
- N1: App B:45 now says whitening alone finds a feature.

**M6 (post hoc labels)** is only partly fixed. McNemar, gate 0.9, the 2×2, copy-gating, the nonce edit and A3z are labelled now; the rest are listed in M4 above.

## Full checklist

File prefixes: `00`–`09` = `sections/0x_*.tex`; `B`–`J` = `appendix/<letter>_*.tex`.

| # | file:line | claim | paper value | source value | source | verdict |
|---|---|---|---|---|---|---|
| 1 | 00:3 | concept tokens rare; rare tokens share a common direction | — | concept-only rarer in 35/38; μ̂_B projection falls with frequency | WP1:48-50 | OK |
| 2 | 00:4 | lexicon generated once per concept | 1 | one lexicon per concept | CI:252 | OK |
| 3 | 00:5 | embedding erasure "so far limited to English" | — | EMBER leaves languages to future work | EMBERpaper:249 | OK |
| 4 | 00:5 | single fit "matching per-language EMBER" | match | A3z (post hoc) .231 vs A1 .248, p .78; A3 .139; non-inferiority not shown | ml:PR:48, 64; ml:UR:18 | UNSUPPORTED (U1) |
| 5 | 00:5 | an eighth of the judge calls | 1/8 | 20 vs 160 | WP9:126 | OK |
| 6 | 00:6 | standard evaluation rewards destroying the words | — | F28, F32–F34 | CE:139-150 | OK |
| 7 | 00:7 | CINDER erases concepts EMBER misses; removes more at moderate size | — | 14 vs 3 runs; r = 0.5: +.151, Holm .012 | recomputed; CF:26 | OK |
| 8 | 00:7 | erasure "revealed to block access" | — | word-free efficacy 0 by construction (untied head) | CE F36 | AMBIGUOUS |
| 9 | 01:20-29 | Fig. 1: 1% per feature; ρ > τ, one-line summary; δ up to 200; one fit for eight languages | — | as stated | EMBERpaper:72, 469-472, 686 | OK |
| 10 | 01:26 | Fig. 1 "(one LLM call per concept)" | 1 | lexicon 1 call; TJ 2 calls per candidate | CI:255; WP9:126 | AMBIGUOUS |
| 11 | 01:46 | "the mean of all embeddings" | — | μ_B = mean of 107,472 Wikipedia token types | bg_wiki_stats.pt | AMBIGUOUS |
| 12 | 01:46 | HP feature collects Nestlé, Lego, Trump | — | C tokens at ranks 35, 46, 32 (raw, seed 42) | HP/fixed_refit-False_plain/seed42/tables.md | OK |
| 13 | 01:47 | "without this component … clean but quiet cluster, of ordinary value" | — | true of whitening (0.86, 70th pct); μ̂_B removed: purity .27, value 97th pct | WP1:58 | AMBIGUOUS |
| 14 | 01:48 | fixed width and summary judge lose more; nothing erased without a feature | — | 19/60 empty runs; tail misfit 15 + 15 of 38 | CE F27; WP3:117 | OK |
| 15 | 01:54 | EMBER leaves multilingual erasure to future work | — | as stated | EMBERpaper:249 | OK |
| 16 | 01:55 | per-language cost multiplies by the number of languages | ×8 | 8 fits, 8 judging passes (selection) | WP9:119, 126 | OK |
| 17 | 01:55 | single multilingual EMBER fit accepts a feature | 10 of 38 | 10 (seed 42) | ml:PR:47 | OK |
| 18 | 01:55 | "since language identity carries far more energy than the concept" | cause | loss mostly at the judge | WP9:44, 63-67, 96 | MISMATCH (M1) |
| 19 | 01:56 | CINDER-z fit "erases as well as" 8 EMBER runs | tie | 15–13, p .78, CI [−.156, +.110]; exploratory; H4 not met | ml:PR:64; ml:PN:44-49; ml:UR:18 | UNSUPPORTED (U2) |
| 20 | 01:56 | an eighth of the judge calls | 1/8 | 160 vs 20 | WP9:126, 148 | OK |
| 21 | 01:56 | first multilingual erasure at the embedding level | — | EMBER: future work; Part-1 joint fit evaluated no erasure | EMBERpaper:249; P1:307-314 | OK |
| 22 | 01:58 | tuning score prefers edits up to 100× | 100× | paper rule: median r 100 (EMBER, CINDER-z) | CE F28; recomputed | OK |
| 23 | 01:58 | blind spot found on Gemma "and Llama" | — | collateral study Gemma only | CE D17 | AMBIGUOUS |
| 24 | 01:59 | collateral questions, edit-size scale, word tests | — | as stated | App G:6-50 | OK |
| 25 | 01:60 | every method moves to r ≈ 1 | 1 | median r 1 for all five | recomputed (result.json) | OK |
| 26 | 01:61 | runs where only one method edits | 14 vs 3 | 14 vs 3; McNemar .0127 | recomputed | OK |
| 27 | 01:61 | gain at moderate equal size | +.15, 10 of 11 | +.151, 10/1, Holm .012 | CF:26 | OK |
| 28 | 01:61 | tie at own operating points | tie | +.042, Holm .412 | CF:9 | OK |
| 29 | 01:63 | (2) lexicon judge needs one LLM call per concept | 1 | as stated for the lexicon path | CI:252 | OK |
| 30 | 01:63 | (3) eight languages, single fit, 1/8 judge calls | — | as rows 19–20 | WP9:148 | OK |
| 31 | 01:63 | (4) no LLM judge in the tuning loop | — | H3 from MC accuracies + MMLU gate; Alpaca after selection | PR:340-374; LM:41 | OK |
| 32 | 02:16 | top 1% per row; ALS | 1% | as stated | EMBERpaper:72, 429 | OK |
| 33 | 02:23 | τ = 2; two-stage judge; confidence ≥ .85 | — | as stated | EMBERpaper:469-472, 498 | OK |
| 34 | 02:26-31 | edit concept tokens with Y ≠ 0 | — | as stated | EMBERpaper:110 | OK |
| 35 | 02:34 | relearning listed among the metrics of "their harmonic mean H" | — | H excludes relearning (final config only) | EMBERpaper:134, 148 | MISMATCH (M2) |
| 36 | 02:34 | δ grid {0.5, …, 200} on validation | — | {0.5, 1, 2, 5, 10, 50, 100, 200}; validation | EMBERpaper:591, 686 | OK |
| 37 | 03:9 | Fig. 2 caption, panels (a), (b) | — | medians raw (.24, 100), whitened (.86, 70), CS (.86, 91) | paper/figures/src/fig_mechanism.py; WP1/figures/*.csv | OK |
| 38 | 03:14 | reproduction within four points | 4 | max gap 3.7 | L/erasure.md:39 | OK |
| 39 | 03:14 | each observation tested on 38 concepts × 5 seeds | 38 × 5 | judge 19; destruction 14 anchor (s42) / 12 held-out; ML on Llama | CI:280; CE | AMBIGUOUS |
| 40 | 03:15 | five issues paired in Fig. 1 | 5 | 5 rows, not one-to-one | Fig. 1 | AMBIGUOUS |
| 41 | 03:19 | same energy; rarer in 35 of 38 | 35/38 | ratio 1.00; 35/38, p 1.7e-7 | WP1:36, 48 | OK |
| 42 | 03:20 | 42% of its norm along μ̂_B vs 12% | 42%; 12% | 42% of ‖z‖² vs 12% (37/38); App H:56 says 45% | WP1:38; WP2:96 | AMBIGUOUS |
| 43 | 03:20 | groups the concept with rare tokens EMBER edits | — | support 2.5 vs 23 per M; Nestlé/Lego/Trump edited | WP1:39; HP tables.md | OK |
| 44 | 03:21 | removing the shared component → pure, ordinary, found less often | — | only whitening (49% vs 56%); μ̂_B removed 61% | WP1:56-60; App H:18-19 | MISMATCH (M3) |
| 45 | 03:24 | 1% ≈ 48 tokens | 48 | ⌈.01·4,780⌉ = 48; RS widths 43–52 | HP summary.json; WP3:49 | OK |
| 46 | 03:24 | tail beyond the cut 15/38, before it 15/38 | 15; 15 | raw matrix 15 short, 15 long, 8 right | WP3:117; App C:74 | OK |
| 47 | 03:27 | Heroin/WWII descriptions, rejected like VD | quotes | verbatim, False .85, but from CS + RS-1% fits | JR; judge.json; App I:41 | UNSUPPORTED (U3) |
| 48 | 03:28 | EMBER edits nothing in 19 of 60 | 19/60 | 19 | CE F27; CF:16 | OK |
| 49 | 03:31 | Gemma retention gap de/es/fr/ru | .19–.24 | +.24 [.15, .32]; +.19 unparseable dropped; original scorer | P1:224, 226, 284; App F:8 | OK |
| 50 | 03:32 | single multilingual EMBER fit on Llama | 10 of 38 | 10 | ml:PR:47 | OK |
| 51 | 03:32 | language means 34× concept; concept 3× quieter | 34; 3 | 34× median; 2.9× | WP9:48, 51 | OK (framing: M1) |
| 52 | 03:35 | δ = 200 for 6 of the 10 concepts EMBER edits | 6/10 | 6/10 (anchor, seed 42, MMLU-gated H) | CE:317 | OK |
| 53 | 03:36 | edited row 40–120× longer, along −c_t | 40–120 | HP 120.5×; 40–120 across concepts | CE:302 | OK |
| 54 | 03:37 | MMLU gate never binds | 344 | 0 of 344 | CE:328 | OK |
| 55 | 03:37 | Alpaca coherence ≥ | .92 | ≥ .920 (EMBER ≥ .956) | ER/*/result.json (AUDIT.md M5) | OK |
| 56 | 03:38 | HP δ = 200: prompts broken; δ = 1 | 12 of 20; 0–1 | EMBER HP s42 12/20; 0–1 | CE:342 | OK |
| 57 | 03:38 | prompts "use the edited words in other senses" | all | most; a few lack an edited token | CE:338 | AMBIGUOUS |
| 58 | 04:14, 23 | Wikipedia background once per model; η .05, γ 1 | — | 107,472 types; defaults.json | bg_wiki_stats.pt; experiments/whitening/data/gemma/defaults.json | OK |
| 59 | 04:24-25 | gain = ratio^γ; whitening equalizes directions | — | g_j = (1−η)λ_j + η | WP1:32 | OK |
| 60 | 04:26 | "as on Llama, we sharpen only those" | — | rule only in WP8 judge-free fits; ML fits use full S_η | WP8:168-198; ml: experiments/ml/fit.py:29, 96 | UNSUPPORTED (U4) |
| 61 | 04:27 | T invertible; exact map-back | — | as stated | WP6:44-46 | OK |
| 62 | 04:31 | GS floor .2%, cap 3% | — | .002, .03 | experiments/varwidth/common.py:49-50 | OK |
| 63 | 04:32 | "exact projection onto the constraint" | — | exact for the row-share score | App C:12-16 | AMBIGUOUS |
| 64 | 04:38-40 | 20-word lexicon; EJ@5; TWJ; no call per feature | — | as stated | CI:252-255 | OK |
| 65 | 04:45 | LS dictionary independent of metric | — | Prop. 1 holds | WP6:90-94 | OK |
| 66 | 04:46 | μ_B share of EMBER's edit energy | 44% | .44 (27 concepts) | WP6:53 | OK |
| 67 | 04:52 | CINDER-z adds exactly a mean term | — | ≥ .9995 on μ̂_B, 137/137 runs | WP6:44 | OK |
| 68 | 04:58 | 30 + 30 sentences, fewest for full rank | 30 | smallest of {20, 30, 40, 50}; N = 20 fails for 9 | ml: experiments/ml/PLAN.md:37-39 | OK |
| 69 | 04:59 | language directions weakened ~3× | 3× | 8.16 (English bg) vs 2.55; vs raw, share halves | WP9:48-57 | AMBIGUOUS |
| 70 | 04:60 | one lexicon call covers all languages | 1 | 1 call per concept, cached | ml: experiments/ml/select.py:10; WP9:127 | OK |
| 71 | 04:60 | judge calls | 160 vs 20 | 80 vs 10 candidates × 2 | WP9:125-126 | OK |
| 72 | 04:60 | a fifth of the factorization work | 1/5 | 11.0 M vs 2.2–2.4 M rows × iterations | WP9:122, 149 | OK |
| 73 | 04:60 | a third of the LLM cost | 1/3 | .0174 / .055 = 31% | WP9:129, 150 | OK |
| 74 | 04:60 | wall-clock time about the same | — | GPU 40 s vs 44 s; CPU 76 vs 39–40 s | WP9:123-124 | AMBIGUOUS |
| 75 | 04:60 | evaluation dominates and costs the same | — | 2.5–3.5 GPU-h per unit, ~250× the fit | WP9:152 | OK |
| 76 | 05:12 | Llama in eight languages | 8 | en de fr it pt hi es th | ml:RES:3 | OK |
| 77 | 05:13 | 300 sentences; 50/50 questions | — | as stated | EMBERpaper:142, 585 | OK |
| 78 | 05:14 | dev HP/AR/VD; 12 concepts × 5 seeds | 12; 60 | as stated | PR:322-324 | OK |
| 79 | 05:14 | 18 + 20 multilingual, pre-registered protocol | 18 + 20 | as stated | ml:PN:1-13 | OK |
| 80 | 05:17 | "We keep EMBER's metrics" | — | relearning not tested | 10_statements.tex:13 | UNSUPPORTED (U5) |
| 81 | 05:17 | perplexity replaces Alpaca (ML) | — | perplexity gate on native text | ml: experiments/ml/PLAN.md:46-53 | OK |
| 82 | 05:18 | 3,501 items / 1,026 tokens; blind; frozen before scoring | — | as stated; frozen in 8423d03 | PR:369-370 | OK |
| 83 | 05:19 | integrity tests vs deletion and nonce references | — | as stated | App G:32-50 | OK |
| 84 | 05:20 | gates MMLU .9, collateral .8 | — | ≥ .9, ≥ .8 | PR:329 | OK |
| 85 | 05:21 | r grid {.05 … 100}; definition of r | — | 10 values | PR:150 | OK |
| 86 | 05:22 | EMBER's coherence: 150 Alpaca outputs per point | 150 | 150 validation prompts | EMBERpaper:589 | OK |
| 87 | 05:25 | A1/A2/A3/A3z definitions | — | as stated | ml:RES:9-16 | OK |
| 88 | 05:26 | "we report EMBER's H" | — | paper rule = HM(eff, spec) + MMLU gate, no Alpaca | App G:25; App A:27; EMBERpaper:134 | UNSUPPORTED (U6) |
| 89 | 05:27 | concept unit; Wilcoxon, Holm, bootstrap | — | true for contrasts; McNemar run-level | experiments/varwidth/confirm_stats.py | AMBIGUOUS |
| 90 | 05:27 | "marked post hoc, all others pre-registered" | — | several unmarked post hoc analyses | PR:388, 429, 552; ml:PN:44-49 | MISMATCH (M4) |
| 91 | 05:33 | without collateral, edits 100× the token | 100× | EMBER, CINDER-z 100; RS-z 30; CINDER, -RS 1 | CE F28 | AMBIGUOUS |
| 92 | 05:33 | with collateral, every method r ≈ 1 | 1 | median 1 for all five | recomputed | OK |
| 93 | 05:34 | neutral prompts broken at destructive points | 24–34% | .24 / .34 (9 matched concepts) | LM:36, 38 | OK (unmarked post hoc: M4) |
| 94 | 05:34 | deletion floor; runs above it | .23–.28; 83%, 98% | .23 / .28; 1 − .17, 1 − .02 | LM:9, 12 | OK (unmarked post hoc: M4) |
| 95 | 05:34 | edits "derail the surrounding prompt" | — | context damage at r = 100 on dev concepts only | CE F23 | AMBIGUOUS |
| 96 | 05:35 | Alpaca registers none of this | — | .97 at paper points (seed 42) | LM:46, 53 | OK |
| 97 | 05:35 | under H, EMBER ahead of CINDER | — | H2 −.134, 1/10, Holm .056 | CF:43 | AMBIGUOUS |
| 98 | 05:45 | Table 2 EMBER, H | 100 / .47 / .73 / .24 / .97 | 100 / .4696 / .7253 / .24 / .9745 | recomputed; LM:38, 46 | OK |
| 99 | 05:46 | Table 2 EMBER, H3 | 1 / .30 / .86 / .15 / .97 | 1 / .2947 / .8547 / .15 / .9737 | recomputed; LM:9 | MISMATCH (M5) |
| 100 | 05:47 | Table 2 CINDER-z, H | 100 / .68 / .68 / .34 / .97 | 100 / .6821 / .6807 / .34 / .9736 | recomputed; LM:36, 53 | OK |
| 101 | 05:48 | Table 2 CINDER-z, H3 | 1 / .39 / .86 / .16 / .98 | 1 / .3898 / .8568 / .16 / .9772 | recomputed; LM:37, 52 | OK |
| 102 | 05:51 | Table 2 caption scopes | — | as stated; Eff/Coll over 41 / 52 runs | LM:29-46 | OK |
| 103 | 05:56 | runs with an edit | 41 vs 52 | 41, 52 | recomputed | OK |
| 104 | 05:56 | only-one-edits; McNemar, post hoc | 14 vs 3; .013 | .0127; in no PREREG | recomputed | OK |
| 105 | 05:59 | r = .5 efficacy at the same collateral | .25–.30 vs .20 | .252–.297 vs .201; coll .926–.935 vs .933 | result_fixed05.json (recomputed) | OK |
| 106 | 05:59 | CINDER gain at r = .5 | +.15, 10 of 11, Holm .012 | +.151, 10/1, .012 | CF:26 | OK |
| 107 | 05:60 | 2×2: direction vs token set (post hoc) | +.07, 10 of 11; +.02 | +.073 (Holm .097), 10/1; +.016 | TB:16-17 | AMBIGUOUS |
| 108 | 05:61 | r = 1 direction effect; "secondary" | −.02, p .97 | −.016, p .966; secondary endpoint of post hoc am. 14 | TB:49; PR:552 | AMBIGUOUS |
| 109 | 05:65 | pre-registered primary | +.04, Holm .41 | +.042, .412 | CF:9 | OK |
| 110 | 05:65 | "EMBER catches up with larger edits" | — | both at median r 1 | recomputed; TB:42-49 | AMBIGUOUS |
| 111 | 05:66 | gate .9: all four beat EMBER significantly (post hoc) | — | Holm .012 / .012 / .008 / .014 | CSPL:27-30 | OK |
| 112 | 05:66 | gate .7 not mentioned | — | Holm ≥ .27 | CSPL:36-39 | AMBIGUOUS |
| 113 | 05:71 | Fig. 3: 11 concepts, four rules, bootstrap CI, stars | — | as stated; stars at fixed r for CINDER, -RS, -RS-z | paper/figures/src/fig_results.py; results.png | OK |
| 114 | 05:79 | A3z vs A1 strict / ungated | .23/.25; .28/.25 | .231/.248; .282/.252 | ml:PR:48-49 | OK |
| 115 | 05:79 | W–L, p | 15–13, .78 | 15–13, .779 (n 29); not among H1–H3 | ml:PR:64; ml:PN:44-49 | UNSUPPORTED (U2) |
| 116 | 05:80 | 3 fits instead of 24 | 3; 24 | as stated | ml:RES:189-191 | OK |
| 117 | 05:80 | pooling helps CINDER-z | +.08, 19–4, < .001 | +.082, 19–4 | ml:UR:23 | OK |
| 118 | 05:80 | pooling does not help A1 | +.05, n.s. | +.054, 14–11, .578 | ml:UR:22 | OK |
| 119 | 05:80 | "the two still tie" | tie | +.011 [−.114, +.137]; H4 non-inferiority not met | ml:UR:18; ml:PN:81-83 | UNSUPPORTED (U2) |
| 120 | 05:90 | fits per concept | 8/1/1/1 | 8/1/1/(A3's fit + k×k solve) | WP9:119 | OK |
| 121 | 05:91 | judge calls per concept | 160/20/20/20 | as stated | WP9:126 | OK |
| 122 | 05:92 | LLM cost per concept (NIS) | .055/.007/.017/.017 | .055/.0069/.0174/as A3 | WP9:129 | OK |
| 123 | 05:93 | concepts edited | 26/10/21/21 | as stated | ml:PR:47 | OK |
| 124 | 05:94-95 | strict / ungated scores | .25/.11/.14/.23; .25/.14/.14/.28 | .248/.113/.139/.231; .252/.143/.139/.282 | ml:PR:48-49 | OK |
| 125 | 05:98 | Table 3 caption | — | fit seed 42 unstated; A1's 26 incl. 2 with no admissible δ | ml:PR:3; ml:RES:140 | AMBIGUOUS |
| 126 | 05:103 | A2 erases as well as CINDER-z where it edits | — | .424 vs .429, n 10, p .73 | ml:PR:60 | OK |
| 127 | 05:103 | coverage | 10/38, 2/20 vs 21, 10 | as stated | ml:PR:9, 47 | OK |
| 128 | 05:103 | counting coverage, A3z beats A2 | +.21, 15–4, .011 | +.213, 15–4, .011; not among H1–H3 | ml:PR:62 | OK (qualifier: M4) |
| 129 | 05:104 | raw fit has a lexicon feature as often | 25–27 vs 23–28 | A2 25/26/27; A3 28/23/25 | WP9:64 | OK |
| 130 | 05:104 | judge accepts "5% of such candidates" vs 30% | 5%; 30% | 3–4-hit bin only; ≥ 5 hits 41–70% vs 48–93% | WP9:67-73 | MISMATCH (M6) |
| 131 | 05:104 | re-judged: 10 → 18; 16 (seed 42) | 10/18/16 | as stated; post hoc, counterfactual | WP9:79-84; App F:178 | UNSUPPORTED (U7) |
| 132 | 05:107 | A3z − A3 | +.17, 16 of 21, .007 | +.166, 16–5, .007 over 38 (incl. 18 dev) | ml:PR:58 | AMBIGUOUS |
| 133 | 05:107 | on the 20 held-out | +.19, 7–3, .064 | as stated | ml:PR:20 | OK |
| 134 | 05:107 | A3z defined after the development results | — | as stated | ml: experiments/ml/PLAN.md:86 | OK |
| 135 | 05:108 | other Latin-script languages ≈ .6 of English | .6 | .54–.72 (38 concepts) | ml:RT:69-72 | OK |
| 136 | 05:108 | Thai little; Hindi almost none | — | Thai .15–.38; Hindi .05–.09 | ml:RT:69-72 | OK |
| 137 | 05:108 | Hindi pieces "never concept-only" | never | 5% (Baseball), 4% (Gun) | ml:OBS:31-33; App F:194 | MISMATCH (M7) |
| 138 | 06:7 | 38 × 5; 15 added after freeze | — | 23 + 15 | L/SUMMARY.md:4 | OK |
| 139 | 06:11 | seeded feature 24% / 86% / 70th / 91st; 37 of 38 | — | as stated | WP1:58 | OK |
| 140 | 06:12 | full fits 56 / 49 / 69%; neutral 3.0 vs 4.5 | — | as stated | WP1:60-61 | OK |
| 141 | 06:13 | NN becomes magic for HP | — | magic at 68–75° | WP2:68, 72 | OK |
| 142 | 06:13 | 97% of excess norm along μ̂_B | 97% | 97% of excess ‖z‖², pooled; absent from App H | WP2:97 | AMBIGUOUS |
| 143 | 06:16 | 1% cut short on the sharpened matrix | 25 of 38 | 25/38 | WP3:48; App C:74 | OK |
| 144 | 06:16 | Spearman .67 vs .27; error 30 vs 42, p < .001; compressed range | — | .67 [.42, .84] vs .27; 30.2 vs 42.4, p .0009; slope .30 | WP3:53-57 | OK |
| 145 | 06:17 | GS keeps direction (cos .97); helps only after CS | — | .973; raw GS adds neutral tokens | WP3:77, 90-92 | OK |
| 146 | 06:20 | TWJ 82 of 91 vs 61; 1 of 110 unrelated; added mostly broader | — | as stated | CI:284-296 | OK |
| 147 | 06:21 | 15 later concepts: 49 vs 33 of 54 | — | 49 = TJ ∨ WJ+share, 33 = TJ (TWJ 35, EJ@5 15) | TJR; App D:80-84 | AMBIGUOUS |
| 148 | 06:24 | strict purity gain, other targets vs page | +.06 vs +.05 | +.060 / +.062 vs +.052 | WP7:48-54 | OK |
| 149 | 06:24 | "p = .003, .009; pre-registered" | — | uncorrected; pre-registered Holm .034, .081 | WP7:48-54; WP7/prereg.json | UNSUPPORTED (U8) |
| 150 | 06:24 | wrong page: found in 2 of 54 | 2/54 | 2/54 (raw 39) | WP7:79 | OK |
| 151 | 06:27 | 13 abstract; 0 of 65; 13 and 23 of 40 entity | — | as stated (CS+GS 1/65) | WP4:46 | OK |
| 152 | 06:27 | register explanation stated as fact | — | speculation; probe null, p .58 | WP4:128-139 | UNSUPPORTED (U9) |
| 153 | 06:28 | dilution hurts EMBER more; weak extrapolation | — | EMBER rank 4.1 → 6.9, CS top-10 80–93%; P@10 .18–.27 | WP4:101-109; WP5:58-63 | OK |
| 154 | 06:31 | read "about twice as strongly as CINDER's" | 2× | 1.86 / 1.99 vs 1.18 = 1.6–1.7× | WP6:53-54 | MISMATCH (M8) |
| 155 | 06:32 | 2.9× Gemma, 20× Llama; ⊥μ 1.31× vs 1.32× | — | 2.9; 20.4; 1.31 / 1.32 (n 27) | WP6:18, 53, 107 | OK |
| 156 | 07:10 | word-free efficacy 0.000 in N configurations | 18 | 22 (11 run-variants × 2) | LM:82-91; 11 wordfree.json files | MISMATCH (M9) |
| 157 | 07:10 | "knowledge remains in the weights" | — | by construction; post hoc am. 11 | CE F36; PR:429 | AMBIGUOUS |
| 158 | 07:11 | efficacy close to the deletion floor | — | .29–.32 vs .23–.28; -z .39 / .38 vs .28 / .26 | LM:9-13 | AMBIGUOUS |
| 159 | 07:14 | unreadable; exclusive; no significant difference | .49–.61; .65–.81 | as stated, p ≥ .19; seed 42 only | CF:101-118 | AMBIGUOUS |
| 160 | 07:15 | copy-gated efficacy cap (post hoc) | .06–.14 | .056–.136 | LM:113-117 | OK |
| 161 | 07:15 | CINDER ≈ 2× EMBER under copy gate | 2× | .12 vs .06; paired H3 +.096, Holm .42 | LM:106, 114 | AMBIGUOUS |
| 162 | 07:15 | CINDER-z advantage "came from" prompt damage | — | post hoc association; WP6b supports on 3 dev concepts | LM:128, 135; CE F37; WP6b:7-11 | UNSUPPORTED (U10) |
| 163 | 07:19 | nonce edit 2–3× "EMBER's subtraction", copyable | 2–3× | vs copy-gated EMBER .07; EMBER own point .29 | NE:8-11, 17; LM:9 | UNSUPPORTED (U11) |
| 164 | 07:19 | substantial nonce floor in 2 of 12 (post hoc) | 2/12 | Baseball (efree), Heroin (+QA); seed 42 | CE F40 | AMBIGUOUS |
| 165 | 07:28-32 | Table 4 cells (Harry, Lego) | quoted | match ember_d1 / ember_d200, prompts 0, 7 | COL/premise_generations.json | OK |
| 166 | 07:28, 35 | r ≈ .6 at δ = 1; δ = 200 selected; seed 42; tokens in the feature | — | .605; δ 200; C tokens | CE:290; ER/Harry_Potter/L0_s42/result.json | OK (delete the `\note`) |
| 167 | 09:6-8 | conclusion claims; 1/8 judge calls | — | as above | — | OK |
| 168 | B:90 | ρ-rank only changed by wrong target; 4.9 vs 3.9; AR 22 → 2, HP 14 → 1, Halloween 15 → 1.3 | — | 4.93 / 3.93, p .51; wrong Holm .003 | WP7:40-43; WP7/analysis.json | OK |
| 169 | D:65 | 75 runs; 54 with a P candidate; 70 P candidates | — | as stated | TJR | OK |
| 170 | D:80-84 | tab:twj-heldout, 30 cells | — | all match | TJR; L/layers.md:114-122 | OK |
| 171 | D:87 | caption | — | as stated | TJR | OK |
| 172 | D:92 | TJ rejects 33 of 70; TJ ∨ WJ+share recovers 25 | — | 33; 25 | TJR | OK |
| 173 | D:93-94 | five concepts; no-rejection and no-P lists | — | 30 of 33 in five concepts; lists match | TJR | OK |
| 174 | D:104-116 | tab:tj-rejections: Rej., Runs, confidences | — | all match | TJR (stage-1 dump) | OK |
| 175 | E:14 | fig:read, Gemma: centring brings EMBER to CINDER's | — | 1.09 vs 1.18 | WP6:56 | OK |
| 176 | E:14 | fig:read, Llama: same claim | — | centred A2 1.70 vs A3 1.24 (p .014) | WP6:96-108 | AMBIGUOUS |
| 177 | E:72 | cos(c, e) without μ̂_B | .31 vs .29, p .15 | as stated | WP6:82, 87 | OK |
| 178 | F:44 | dev run on 18; A3z defined post hoc | — | as stated | ml: experiments/ml/PLAN.md:79-86 | OK |
| 179 | F:45 | H1–H3 definitions | — | match | ml:PN:44-51 | OK |
| 180 | F:46 | H2 on 20: +.19, 7–3, .064; ungated +.26, 8–2, .014 | — | match | ml:PR:18-22, 35 | OK |
| 181 | F:47 | over 38: H2 +.166 [.054, .268] 16–5 .007; H1 −.051 3–7 .38; H3 −.005 5–4 .73 | — | match | ml:PR:56-60 | OK |
| 182 | F:48 | A3z − A2 +.213 15–4 .011; A3z − A1 −.023 15–13 .78 | — | match | ml:PR:62, 64 | OK |
| 183 | F:48 | "concepts that only one approach edits" | — | concepts either approach edits | ml:PR:62, 64 | AMBIGUOUS |
| 184 | F:60-79 | tab:ml-dev: 60 cells, counts, means | — | all match | ml:RT:9-26; ml:PR:85-87 | OK |
| 185 | F:82 | tab:ml-dev caption (seed 42; gate failures; ungated values) | — | match | ml:RT:19, 26 | OK |
| 186 | F:95-103 | tab:ml-prereg cells | — | all match | ml:PR:9-26, 35 | OK |
| 187 | F:106 | tab:ml-prereg caption | — | match | ml:PR:3 | OK |
| 188 | F:111-112 | seeds 1, 2 selection only; union 31/13/29; gap at every seed | — | match | ml: results/ml/seeds/COVERAGE.md:46-50 | OK |
| 189 | F:124-126 | tab:ml-seeds | — | all match (recomputed) | COVERAGE.md:5-44 | OK |
| 190 | F:134 | union amendment before evaluations; 52 units | 52 | 52 | ml:RES:191; ml:PN:60 | OK |
| 191 | F:135 | pooling values; Tuberculosis, The Matrix, HP, Cricket | — | match | ml:UR:22-23, 59; ml:RES:211-214 | OK |
| 192 | F:136 | pooled tie +.011 …; misses non-inferiority; both-edit +.082 16–10 .32 | — | match | ml:UR:18, 20 | OK |
| 193 | F:137 | Tuberculosis A1U erases Hindi | H_L .77 | .77 | ml:RES:218-219 | OK |
| 194 | F:149-154 | tab:ml-union | — | all match | ml:UR:9-11 | OK |
| 195 | F:174-178 | tab:ml-2x2 cells; counterfactual | — | match (recomputed) | WP9:83-84 | OK |
| 196 | F:184 | 10 → 16 with TJ; TWJ 18 vs 21 n.s. over seeds | — | +.05, 11–8, p .73 | WP9:87-89 | OK |
| 197 | F:185 | ≥ 2 scripts: 34% / 15% / none | — | match | WP9:106-112 | OK |
| 198 | F:163 | (unchanged) "such features … 5% vs 30%" | — | bin-specific (3–4 hits) | WP9:67-73 | AMBIGUOUS |
| 199 | G:54 | robustness rows post hoc, after the primary | — | am. 9 post hoc | PR:388 | OK |
| 200 | G:68-69 | Primary Δ and Holm | — | match | CF:9-12 | OK |
| 201 | G:70-71 | All 60 runs Δ and Holm | — | match | CF:79-82; CSPL:9-12 | OK |
| 202 | G:72-73 | Gate .9 Δ and Holm | — | match (Holm over 4, recomputed) | CF:79-82; CSPL:27-30 | OK |
| 203 | G:74-75 | Gate .7 Δ and Holm | — | match | CSPL:36-39 | OK |
| 204 | G:76-77 | leave-one-out ranges | — | match | CF:79-82 | OK |
| 205 | G:80 | LOO "over the 12 omitted concepts" | 12 | 11 | confirm_stats.py:93-100; CF:9 | MISMATCH (M10) |
| 206 | G:80 | caption "12 held-out concepts" | 12 | primary and gate rows 11 | CF:9-12, 79-82 | AMBIGUOUS |
| 207 | G:80 | caption: no-edit run scored as unedited; gates rerun | — | match | CF:75; PR:391-395 | OK |
| 208 | G:85-86 | 3 dev concepts, 8 concept-seeds; 12 / 60 held out | — | match | POC; PR:319-323 | OK |
| 209 | G:87 | 4 replicate, 1 in part, 3 not | 4/1/3 | matches the table | tab:devclaims | OK |
| 210 | G:87 | "only held-out numbers in the main text" | only | §3:38, Table 4 quote HP | CE:338-344 | MISMATCH (M11) |
| 211 | G:99 | devclaims row 1 | — | match | CE F1, F25, F28 | OK |
| 212 | G:101 | row 2 | — | match | CE F3; CF:26-29 | OK |
| 213 | G:103 | row 3 | — | match | CE F6; TB:16, 49 | OK |
| 214 | G:105 | row 4 | — | match | CE F5; CF:9 | OK |
| 215 | G:107 | row 5 | — | match | CE F18; CF:10-12 | OK |
| 216 | G:109 | row 6: dev vs parent, held-out vs EMBER | — | values match; comparator differs | CE F22; CF:11 | AMBIGUOUS |
| 217 | G:111 | row 7 | — | match, p .688 | CE F12, F17; CF:107-115 | OK |
| 218 | G:113 | row 8 | — | 1/10; 11/0 | CF:43, 45 | OK |
| 219 | G:116 | devclaims caption | — | match | POC; PR:322 | OK |
| 220 | G:121 | Mean/Noise defined following EMBER | — | match | EMBERpaper:152, 180 | OK |
| 221 | G:122 | Mean r .85–.95; coll .71 (.55–.78); copy 0, destr. .99, context .52–.54 | — | match; integrity from 3 labelled runs (unstated) | CE F7, F15 | AMBIGUOUS |
| 222 | G:123 | Noise median r 1; destruction .25–.35; leak .29 | — | match | POC; CE F12 | OK |
| 223 | G:136-141 | tab:baselines | — | match | POC:162-175 | OK |
| 224 | G:144 | tab:baselines caption | — | match | POC:3-6 | OK |
| 225 | G:153 | fig:integrity: 41–55 runs; validation; floors .29–.36 / .23–.28 | — | match | IDJ; LM:9-13 | OK |
| 226 | G:153 | crosses deletion floor at r .5–1; copy .8 → .5 | — | true; validation curve vs test floor | IDJ | AMBIGUOUS |
| 227 | G:153 | no nonce floor reached while copy > .5 | — | true | IDJ | OK |
| 228 | G:153 | star: 9 runs, copy .99, efficacy .18 | — | .989, .177 | COL/nonce_edit.json; NE:8 | OK |
| 229 | G:157 | E3 post hoc; 1,530 points | — | 6 × 255 | PR:429, 438 | OK |
| 230 | G:158 | E3 = collateral point in 16 of 255 | — | match | CE F38 | OK |
| 231 | G:160 | "pre-registered paired endpoint" | — | endpoint of post hoc am. 11 | PR:429, 443 | AMBIGUOUS |
| 232 | G:160 | E3 paired contrasts | — | match | LM:106-109; CSPL:45-48 | OK |
| 233 | G:161 | E3 all-runs contrasts | — | match | CSPL:45-48 | OK |
| 234 | G:162 | -z at r = .05 in 24/52 and 26/55 | — | match | CE F38 | OK |
| 235 | G:163 | Alpaca .97–.98 at E3 points (seed 42) | — | match | LM:47-56 | OK |
| 236 | G:176-180 | tab:e3 cells | — | match | LM:111-117, 47-56 | OK |
| 237 | G:183 | tab:e3 caption (H4 8–10 runs) | — | 8/10/10/10/10 | LM:47-56 | OK |
| 238 | G:193 | fig:twobytwo: post hoc; +.07 10/11; +.02; −.02 6/5 | — | match | PR:552-576; TB:16-17, 49 | OK |
| 239 | G:193 | plotted ee point | — | ee 41-run mean .201; .192 on the 38 full runs | TB:9; result_fixed05.json | AMBIGUOUS |
| 240 | G:219 | subspace rank ≤ 3; +QA | — | match | PR:498-512; CE F42 | OK |
| 241 | G:220 | nonce point definition; post hoc, seed 42 | — | match | NE:3; PR:447 | OK |
| 242 | G:221 | floors .36 / .26–.27; COVID .00; Porn .07–.11; Baseball .36 vs .31; Heroin .44 vs .49 | — | match; Heroin .444 ≥ .489 − .05 | COL/nonce_edit.json; last_mile/*/floor.json | OK |
| 243 | G:222 | free median r 3, 3.2, 1.6; 2–3× / 1.3–1.6× (integrity-gated) | — | match | NE:8-11, 17-18 | OK |
| 244 | G:222 | paired H3 +.20 (7 of 9, .055), +.26 (7–1, .039), +.28 (6–1, .047) | — | match | NE:28, 31, 34 | OK |
| 245 | G:223 | free edits repeat the word | .47–.49 | .47, .49, .34 | NE:41-47 | MISMATCH (M12) |
| 246 | G:223 | renamed 32–47% (nonce 17%) | — | .36 / .47 / .32; .17 | NE:54-62 | OK |
| 247 | G:224 | subspace leak .39 / .53 | — | match | NE:79, 81 | OK |
| 248 | G:237-247 | tab:nonce cells | — | match | NE:7-19, 79-97 | OK |
| 249 | G:250 | tab:nonce caption | — | match | NE:93-97 | OK |
| 250 | G:53 | (unchanged) every analysis plan before its results | every | am. 2 after results; ITT during am. 9; McNemar no PREREG | PR:90, 388-390 | UNSUPPORTED (U12) |
| 251 | H:18 | seeded feature .24 / .04 / .27; .86, 41%; 70th; 91st; 37/38 | — | match | WP1:58 | OK |
| 252 | H:19 | found 56/61/59/49/69%; 15/3 (p .011); 17/0 | — | match | WP1:60 | OK |
| 253 | H:20 | "when a raw fit has a lexicon feature … median rank 2" | 2 vs 1 | conditional median 1; 2 vs 1 = max-lexicon feature | WP1:65 | AMBIGUOUS |
| 254 | H:20 | HP raw ranks 12, 1, 28, 48, 1; CS first ×5 | — | match | WP1:66 | OK |
| 255 | H:53 | EMBER NN angle 50.6° → 74.0° (38/38) | — | match | WP2:69 | OK |
| 256 | H:57 | HP raw feature 3rd–6th by norm; .67–.80 | — | match | WP2:94, 99 | OK |
| 257 | H:76 | EMBER rank 4.1 → 6.9; 5.9 vs 5.7 (.96); CS top-10 80–93%, rank 1.8–3.3 | — | match | WP4:101-109 | OK |
| 258 | H:77 | intruder shares; CS_sc .23 | — | match | WP4:118-121 | OK |
| 259 | H:77 | CS at f = .5 keeps Tolkien, Gandalf, Hobbit at 16–18 | 16–18 | no Hobbit; Tolkien, Gandalf at 16–20 | WP4/features.parquet (recomputed) | AMBIGUOUS |
| 260 | H:94-95 | Spearman .70 vs .25; \|cos\| .28–.30 vs .64; 6 concepts, 4% | — | match | WP5:63, 67, 69 | OK |
| 261 | H:97 | 87% once (52% of concept-only V*) | — | match | WP5:86 | OK |
| 262 | H:97 | precision .28–.30 → .59–.71, "in-sample and extrapolated alike" | — | .59–.71 in-sample only | WP5:87-89; fig_precision_vs_pagefreq.json | AMBIGUOUS |
| 263 | H:106 | fig:extrapolation caption | — | match | WP5 figure JSON; paper/figures/src/fig_extrapolation.py | OK |
| 264 | I:38 | Trump prompt and answers | — | answers match; prompt truncated | COL/premise_prompts.json (HP[4]); premise_generations.json | AMBIGUOUS |
| 265 | I:41-43 | sharpened RS-1% fits; "not a member" at .85; threshold-independent | — | match | judge.json; experiments/varwidth/layer_eval.py:42-44 | OK |
| 266 | I:55-59 | tab:tj-examples: seeds, ρ 95 / 31 / 21, tokens, quotes | — | match | JR:5-179 | OK |
| 267 | J:18 | one GPU per job (one used two); ladder fits on CPU | — | true 17–23 Oct; 3–8 Oct pass-2 used 3 GPUs each | sacct | AMBIGUOUS |
| 268 | J:20 | 2.5–3.5 GPU-h per unit; 68/34/71/66 GPU-h | — | match | WP9:130, 135, 152 | OK |
| 269 | J:20 | "under a minute per concept for selection" | — | fit stage only measured | WP9:152 | AMBIGUOUS |
| 270 | J:20, 42 | only the seed-42 fits unrecorded; Multilingual 320 | 320 | SLURM 319.9; plus ~151 GPU-h local evaluation unrecorded | sacct; ml: results/ml/logs/worker_*local*.log | MISMATCH (M13) |
| 271 | J:21 | ladder Gemini: judging, anchor Alpaca | 7.5; 8.9 | 7.51; 8.90 | results/gemini_usage.jsonl | OK |
| 272 | J:21 | collateral Gemini: dev, last mile, E3 | 7.1 / 8.3 / 5.7 | 7.09 / 8.33 / 5.68 | results/gemini_usage.jsonl | OK |
| 273 | J:21 | multilingual Gemini: selection, questions, stability | 8.3 / 8.0 / .75 | 8.32 / 8.04 / .75 | WP9:145 | OK |
| 274 | J:22 | prices .30 / .25 and 2.50 USD per M; 3.7 NIS/USD | — | match | ember/gemini_budget.py; WP9:184 | OK |
| 275 | J:23 | ~540 GPU-h on 3–8 Oct; 446 for pass 2 | 540; 446 | 542.2; 446.4 | sacct | OK |
| 276 | J:40-42 | GPU-h 20 / 346 / 320; Gemini 16.4 / 21.1 / 17.1 | — | 19.8 / 345.7 / 319.9 (SLURM); 16.41 / 21.09 / 17.11 | sacct; gemini_usage.jsonl | OK |
| 277 | J:43, 45, 48 | Analyses 10⁺ "one job still running"; Total 695 | 10⁺; 695 | finished, 10.7; rows sum to 696 | sacct | MISMATCH (M14) |
| 278 | J:45 | Gemini total | 54.6 | 54.61 | gemini_usage.jsonl | OK |
| 279 | J:48 | 182 TITAN Xp + 164 RTX 2080 Ti | — | 181.6 + 164.1 | sacct | OK |

Not audited: the `\pending{WP6b …}` slots, the AI-disclosure draft, App. J paths, and appendix lines unchanged since
`cc68637d`. The exceptions are App. G:53 and F:163, checked here because the new text leans on them.

## Resolution (lead, 2026-10-23)

Applied: M1–M14 and U1–U12 (U3: §3 now quotes EMBER's raw-fit descriptions, verified in
`results/varwidth/{Heroin/…/seed42,World_War_II/…/seed43}/judge.json`; App I adds them; U4: App B.3 now states the
multilingual fits used the full S_η). AMBIGUOUS items applied except: Fig. 3's x-label keeps "EMBER's H" (defined in
§5.1 and the Table 1 caption as H without Alpaca); G:109 ordering note (verdict unchanged); mechanism.png clipping is
the PNG preview only (the PDF used by the paper is not clipped). Table 4's `\note` removed after verification
(r = 0.605 at δ = 1; δ = 200 selected in `layers/erasure/Harry_Potter/L0_s42_papergrid/result.json`).
Main text still ends on p. 8 (several sentences shortened to pay for the qualifiers).
