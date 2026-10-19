# A2/A3 selection stability to the N=30 sample

2026-10-19 · `experiments/ml/stability.py` · robustness check, not part of the PREREG grid.

**Setup.** For Baseball, Culture of Greece, Gun, Harry Potter and Ancient Rome I drew 3 new N=30 samples with
`build_sets.subsample`, changing only its Random seed (1, 2, 3). The reference is the existing
`results/ml/data/sample/<slug>_N30.json`, which uses data `SEED = 0` (`experiments/ml/data/common.py`), not 42.
42 is the fit seed (`common.SEED`), and it is the same in every run here. For each sample I ran `fit.run`
(A2, A3) and `select.run` without changes (run_dir, load_sample and lexicon are redirected inside the stability
process; the lexicon is read from `results/ml/lexicons.json`). All fits ran on CPU (8 threads, about 25 s for A2 and
about 55 s for A3). The seed-0 reference runs were fit on GPU. Control "0cpu" refits the seed-0 sample on CPU to
measure device nondeterminism. Judge verdicts were reused when the top-20 token strings matched exactly.

Columns: **#F_C** = selected features for seeds 0 / 1 / 2 / 3 (0cpu); **#tok** = edited tokens
(concept-only, nonzero Y on a selected feature, as `evaluate.run_edit`); **J** = Jaccard of the edited token sets
for the pairs 0-1, 0-2, 0-3, 1-2, 1-3, 2-3 (`–` = both empty, `0` = one empty); **J(0,0cpu)** = same sample, GPU
vs CPU fit.

| Concept | App | #F_C 0/1/2/3 (0cpu) | #tok 0/1/2/3 (0cpu) | J pairs | J(0,0cpu) |
| --- | --- | --- | --- | --- | --- |
| Baseball | A2 | 1/1/3/2 (1) | 120/65/214/163 (120) | .23 .32 .42 .17 .16 .29 | 1.00 |
| Baseball | A3 | 2/3/3/1 (2) | 160/186/203/62 (160) | .16 .26 .14 .14 .12 .10 | 1.00 |
| Culture of Greece | A2 | 1/**0**/**0**/1 (1) | 90/0/0/74 (88) | 0 0 .37 – 0 0 | 0.85 |
| Culture of Greece | A3 | 1/**0**/1/2 (1) | 108/0/103/164 (102) | 0 .54 .24 0 0 .28 | 0.84 |
| Gun | A2 | 1/4/2/2 (2) | 99/232/158/127 (140) | .30 .37 .47 .26 .31 .43 | 0.68 |
| Gun | A3 | 5/5/8/4 (3) | 274/283/448/265 (168) | .18 .19 .20 .20 .19 .19 | 0.26 |
| Harry Potter | A2 | 1/**0**/1/1 (1) | 101/0/102/102 (101) | 0 .57 .59 0 0 .57 | 0.92 |
| Harry Potter | A3 | 1/1/1/3 (2) | 79/75/72/243 (162) | .21 .29 .12 .34 .12 .13 | 0.07 |
| Ancient Rome | A2 | **0**/**0**/1/**0** (0) | 0/0/58/0 (0) | – 0 – 0 – 0 | – |
| Ancient Rome | A3 | **0**/**0**/**0**/**0** (0) | 0/0/0/0 (0) | – – – – – – | – |

**Reconstruction error** (relative Frobenius error; seeds 0/1/2/3/0cpu):

| Concept | A2 sample | A2 full | A3 sample | A3 full |
| --- | --- | --- | --- | --- |
| Baseball | .9347 .9336 .9347 .9337 .9347 | .9630 .9624 .9630 .9627 .9626 | .9522 .9512 .9511 .9504 .9522 | .9790 .9793 .9791 .9788 .9792 |
| Culture of Greece | .9355 .9354 .9352 .9351 .9357 | .9627 .9629 .9627 .9621 .9628 | .9545 .9542 .9541 .9550 .9545 | .9800 .9797 .9801 .9799 .9802 |
| Gun | .9340 .9335 .9337 .9325 .9341 | .9623 .9618 .9620 .9616 .9619 | .9504 .9507 .9496 .9491 .9509 | .9780 .9783 .9783 .9781 .9785 |
| Harry Potter | .9356 .9353 .9355 .9352 .9352 | .9628 .9627 .9638 .9636 .9629 | .9557 .9557 .9549 .9551 .9557 | .9809 .9810 .9809 .9810 .9810 |
| Ancient Rome | .9349 .9347 .9347 .9352 .9346 | .9634 .9624 .9629 .9630 .9631 | .9534 .9546 .9543 .9542 .9534 | .9801 .9802 .9807 .9802 .9803 |

The error spread across samples is at most 0.0015, so reconstruction quality does not depend on the sample.

## Qualitative

- **Baseball:** every selected feature on every seed is a baseball feature (pitching, batting, positions,
  teams). A3 has a "pitching" feature on every seed. The features are different factorizations of the same
  vocabulary, and they overlap only partly in tokens (J 0.1–0.4).
- **Gun:** always 1–8 firearm or weapon features, all on topic. The token sets agree moderately (A2 J 0.26–0.47,
  A3 J about 0.19). A3 picks up broader "explosives / loading / ammunition" features on some seeds.
- **Harry Potter:** one HP-universe feature in most runs. A2 is empty on seed 1. A3 on seed 3 and on 0cpu
  picks broader "wizards / fantasy" features.
- **Culture of Greece:** a single feature at about the rho threshold (A2 rho 3–4). It drops out completely on
  seeds 1 (A2, A3) and 2 (A2). One A3 seed-3 feature was accepted by EJ only and is a generic "ancient
  Mediterranean civilizations" feature.
- **Ancient Rome:** effectively unselectable at N=30. One A2 feature on 1 of 4 seeds (rho 2.5, "ancient Rome").
  A3 is empty on all seeds. The closest miss is A3 seed 1, f79 "ancient Rome, Latin, Italian", judged
  non-member at confidence 0.75 with 3/5 EJ matches. Most Rome-adjacent candidates are "Romance-language
  morphology" features, which the judge rejects correctly.

## Device control

Refitting the same seed-0 sample on CPU instead of GPU changes the selection noticeably. Baseball is identical.
Culture of Greece and Harry Potter A2 keep the same feature with J 0.84–0.92. Gun A3 goes from 5 features to
3 (J 0.26), and Harry Potter A3 goes to different features (J 0.07). So a large part of the variation across
seeds is already present with a fixed sample: it comes from the fit itself (ALS from the same seed, but with
nonconvex, near-tied features), not only from which sentences are drawn.

## Cost

Gemini 0.256 NIS (738 calls; 40 selections). Wall time about 49 min, CPU only.
