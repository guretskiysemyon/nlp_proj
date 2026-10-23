# CINDER paper (ACL format, EMNLP-style)

`main.tex` is the paper: ACL style files (`acl.sty`, `acl_natbib.bst`), identical to the official
acl-org template that EMBER (EMNLP 2026 Findings) uses; see `template/SOURCE.md`.
Build with `sh paper/build.sh` (Tectonic, on the share) or upload this folder to Overleaf.

## Structure (mirrors the EMBER paper)

| File | Section | Role |
| --- | --- | --- |
| `sections/01_introduction.tex` | 1 Introduction + Figure 1 (schematic) | EMBER's intro arc: setting, gap, hypothesis, method, evaluation, results, analysis, "to conclude" |
| `sections/02_preliminaries.tex` | 2 Preliminaries and Notation | EMBER's three stages and its evaluation, in its notation |
| `sections/03_motivation.tex` | 3 Why EMBER Misses Quiet Concepts + Figure 2, Table 1 | four observations: quiet concept, fixed width, judge, token destruction |
| `sections/04_method.tex` | 4 CINDER (4.1 CS + GS, 4.2 TWJ, 4.3 original-space edit, 4.4 multilingual) | one subsection per EMBER stage |
| `sections/05_experiments.tex` | 5 Experiments (setup with metrics, methods, results, multilingual) | **skeleton for the experiments pass** |
| `sections/06_analysis.tex` | 6 Embedding-Space Analysis | slots for the analysis agents (WP1-WP7) |
| `sections/07_token_analysis.tex` | 7 Token-Level Analysis + Table 4 (real generations) | counterpart of EMBER's Section 5 |
| `sections/08_related_work.tex` ... `10_statements.tex` | Related Work, Conclusion, Limitations, Ethics, AI disclosure | |
| `appendix/A_...J_...tex` | A EMBER, B CS, C GS, D TWJ, E edit space, F multilingual, G evaluation, H analyses, I examples, J reproducibility | low-level detail for each high-level section |

## Placeholder conventions (`\drafttrue` in `main.tex`)

- `\pending{...}` (orange): an embedding-space analysis result still being produced (WP1-WP8).
- `\expfill{...}` (blue box): a table, figure or paragraph for the experiments pass.
- `\expnum{...}` (blue number): a number copied from our internal reports; re-derive it from the source file
  before submission. With `\draftfalse` it prints plainly and the other macros print nothing.
- `\note{...}` (red): a note to the authors.

## Analysis work packages → where their results go

| WP | Results folder (`results/paper_analysis/`) | Paper slots |
| --- | --- | --- |
| WP1 quiet concept (variance, math, frequency) | `quiet_concept/` | §3 ¶1, §6 ¶1, App. B.2 (Proposition 1), App. H.1 |
| WP2 feature geometry, norm puzzle, splits | `feature_geometry/` | §6 ¶2, App. H.2 |
| WP3 widths and tails | `widths/` | §3 ¶2, §6 ¶3, App. C, App. H.3 |
| WP4 abstract concepts, contamination | `robustness/` | §6 ¶5, App. H.4 |
| WP5 extrapolation | `extrapolation/` | §6 ¶7, App. H.5 |
| WP6 edit direction, read strength (CINDER-z) | `edit_direction/` | §4.3, §6 ¶8, App. E |
| WP7 circularity, η/γ sensitivity | `circularity/` | §6 ¶6, App. B.3-B.4 |
| WP8 Llama single-language CS | `llama_cs/` | App. B.5 |

Sources of every number already in the text: `CINDER_improvements.md`, `CINDER_collateral_evaluation.md`,
`results/varwidth/layers/{SUMMARY.md,collateral/*.md}`, and on branch `orip-ml` `experiments/ml/RESULTS.md`,
`results/ml/PREREG_RESULTS.md`.
