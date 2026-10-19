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

## Extension: fit-seed variance at a fixed sample, and A1

Same 5 concepts, CPU only. I changed the fit seed by patching `fit.SEED` inside the stability process
(`stability.py vfit/vselect/vreport`). Variants: **ref** = the existing GPU run (sample 0, fit 42); **0cpu** = the
same on CPU; **fs1, fs2** = sample 0 with fit seed 1 or 2. For A1 the fit is on the full per-language sets, as in
`fit.run` for A1, and the merged edit set is the union over languages in `page_priority` order, as in
`evaluate.edits`. Gemini 0.494 NIS for this extension (cap 0.5); a guard stopped selection at the budget. Because
of the budget, A1 fs2 covers **Baseball only**.

### A2/A3: fit seed at a fixed sample

| Concept | App | #F_C ref/0cpu/fs1/fs2 | #tok ref/0cpu/fs1/fs2 | J ref-fs1, ref-fs2, fs1-fs2 | mean J fit (0cpu,fs1,fs2) | mean J sample (0cpu,s1,s2,s3) |
| --- | --- | --- | --- | --- | --- | --- |
| Baseball | A2 | 1/1/1/1 | 120/120/109/111 | .53 .55 .67 | 0.58 | 0.26 |
| Baseball | A3 | 2/2/1/2 | 160/160/54/194 | .18 .35 .09 | 0.20 | 0.15 |
| Culture of Greece | A2 | 1/1/**0**/1 | 90/88/0/71 | 0 .50 0 | 0.17 | 0.08 |
| Culture of Greece | A3 | 1/1/**0**/1 | 108/102/0/110 | 0 .69 0 | 0.21 | 0.18 |
| Gun | A2 | 1/2/1/4 | 99/140/111/213 | .48 .28 .38 | 0.36 | 0.35 |
| Gun | A3 | 5/3/2/6 | 274/168/133/348 | .20 .20 .18 | 0.21 | 0.20 |
| Harry Potter | A2 | 1/1/1/**0** | 101/101/96/0 | .67 0 0 | 0.23 | 0.29 |
| Harry Potter | A3 | 1/2/3/2 | 79/162/237/152 | .18 .11 .13 | 0.11 | 0.14 |
| Ancient Rome | A2 | 0/0/**1**/0 | 0/0/76/0 | 0 – 0 | 0.00 | 0.00 |
| Ancient Rome | A3 | 0/0/**1**/0 | 0/0/61/0 | 0 – 0 | 0.00 | – |

Mean pairwise J (pairs where both sides are empty are excluded): **fit seed 0.22 (28 pairs) vs sample 0.20 (50
pairs)**. Every comparison here is CPU against CPU. With fit seed 1, Ancient Rome gets a feature in both A2 and
A3: a "Latin language / classical literature" feature, judged a member by TJ. Reconstruction errors for fs1 and
fs2 are within 0.001 of ref (A2 sample .9339–.9357, full .9616–.9637; A3 sample .9503–.9556, full .9781–.9809).

### A1 (per-language EMBER, full sets)

#F_C per language, in the order en de fr it pt hi es th:

| Concept | ref (GPU, 42) | 0cpu (42) | fs1 | fs2 | merged #tok ref/0cpu/fs1/fs2 | J merged ref-0cpu, ref-fs1, ref-fs2, fs1-fs2 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseball | 1 0 1 1 1 0 0 3 | 1 0 1 1 1 0 **2** 3 | 1 0 0 2 1 0 2 1 | 2 0 1 2 3 0 1 1 | 108/129/108/148 | 0.84, 0.21, 0.24, 0.21 |
| Culture of Greece | 2 1 0 0 0 0 0 0 | same | 1 0 2 1 0 0 0 0 | – | 91/91/98/– | 1.00, 0.08, –, – |
| Gun | 3 1 1 1 1 0 1 3 | same | 3 1 1 1 1 0 0 0 | – | 169/169/114/– | 1.00, 0.32, –, – |
| Harry Potter | 1 1 1 0 1 0 1 0 | same | 0 1 0 1 1 0 1 0 | – | 84/84/81/– | 0.81, 0.38, –, – |
| Ancient Rome | 0 1 0 0 0 0 0 1 | same | 0 0 0 0 0 0 0 0 | – | 33/33/0/– | 1.00, 0.00, –, – |

- **Device:** A1 is nearly device-stable. The CPU refit reproduces the GPU fit to 4 decimals in most languages,
  98% of judge verdicts are reused, and merged J is 0.81–1.00. A2/A3 are not (J(0, 0cpu) as low as 0.07): the
  larger multilingual fit and the extrapolation amplify small numerical differences.
- **Fit seed:** A1 is just as seed-sensitive as A2/A3. Merged J against ref is 0.21–0.38 for Baseball, Gun and
  Harry Potter. Culture of Greece is at 0.08, where fs1 picked two off-topic fr features ("culinary traditions",
  "historical artifacts") that TJ accepted. Ancient Rome loses both of its features. Per language, J against ref
  ranges from 0 to 0.68, and the low-resource languages (es, th) flip between 0 and 3 features.

### Variance decomposition

At a fixed sample, changing only the ALS fit seed moves the edited-token set about as much as redrawing the
sample: mean pairwise J 0.22 vs 0.20. So for A2/A3 the fit seed is the dominant noise source, and the sample adds
little beyond it. A1 has no sample, yet its fit-seed variance is of the same size. Device nondeterminism (GPU vs
CPU) acts like a small fit-seed perturbation for A2/A3, but it is close to zero for A1.
