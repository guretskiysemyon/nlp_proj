# CINDER code in this repo: what we have, what is missing, decisions

> **Update 2026-10-18 (evening).** Ori's working folder (`/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj`)
> became readable. It holds newer code than `main`: the judge he actually uses (`overlap_judge.py`,
> `j3_words.py`, `j3_eval.py`, `judge_compare.py`, `layer_eval.py`, `j3_data/` with prompts, concept words and
> labels), the candidate rule "top 20 by ρ" in `select.py`, `run.py --fit-only/--reselect`, and his erasure
> script `erase.py`. All of `experiments/varwidth/` was copied into the repo. CINDER is now exactly his rung
> **L123**: config `rn_refit-False` (not the `refit=True` class default) + HJ = TJ ∨ LO@5. The lexicon
> judge written earlier from his document (`cinder/lexicon.py`) was removed. His seed-42 runs for the 6
> concepts were copied to `results/varwidth/<C>/rn_refit-False/seed42/` (refits reproduce them). His Gun
> target (Firearm page) replaces the rebuilt one. Not copied: `ember/gemini_budget.py` (a spend guard; the
> judge prompts are identical). §3 and §5 below are updated accordingly.

*Review 2026-10-18 of Ori's code as merged into `main` (commit `4c7cc7f`), for the current
plan: CINDER vs EMBER, fitted on English only, 6 concepts, then queried in 5 languages
([`plan.md`](plan.md), "Current plan"). Method description: Ori's "CINDER" document
(contrastive sharpening, global sparsity, two-way judge).*

## 1. Where the code is

| Path | What | Used for the current plan? |
|---|---|---|
| `experiments/whitening/sharpen.py` | background whitening + target sharpening; `Transform.unmap` maps directions back | **yes** (layer 1) |
| `experiments/whitening/background.py` | builds the English background statistics, concept pages and target token sets | yes, only if we rebuild the data (§3) |
| `experiments/whitening/eta_sweep.py`, `eps_select.py`, `compare_backgrounds.py` | how η, ε were chosen | reference |
| `experiments/whitening/background_ml.py`, `queue_ml.json` | 8-language background (no Russian), notebook queue | no (multilingual, Llama) |
| `experiments/varwidth/core.py`, `common.py` | the variable-width ALS loop; allocation rules `fixed`, `m1`–`m4`, `rn` (= GS-norm), `rnn` | **yes** (layer 2) |
| `experiments/varwidth/prep.py` | V*, token labels, plain or whitened matrix A, `T_inv` | **yes** |
| `experiments/varwidth/select.py` | ρ ≥ 2 filter + EMBER's two-stage judge (TJ), gemini-3.5-flash-lite | **yes** (judge, layer 3 without the lexicon) |
| `experiments/varwidth/run.py`, `queue.py`, `evaluate.py`, `overall_report.py`, … | one (concept, config, seed) run → `results/varwidth/…`; reports | yes (run.py) |
| `ember/judge_words.py` (+ `sys_prompt` in `ember/interpret_features.py`) | word-aware judge input (token → source words): the "WJ" variant | optional |
| `experiments/multilingual_hp/`, `experiments/resample_sentences/` | 8-language data (6 concepts), resampled English sentences, Wikipedia dump pages | no |
| `scripts/` (7 files by Ori) | notebook builders / patchers, seed sweeps, overnight runner | no |
| 25 notebooks in the repo root, `low_ram_demands/` (27), `variable_width/` (15) | the notebook versions of the same steps (English and multilingual, Gemma and Llama) | no (reference) |
| `experiments/whitening/Instructions.txt`, `experiments/varwidth/instructions_new.txt` | Ori's specs for the two layers (moved from the repo root on 2026-10-18) | reference |

## 2. What was checked

- **It runs here.** `prep.prepare("Cannabis", matrix="plain")` + `core.fit_varwidth(…, VWConfig(method="rn"))`
  on the CPU: prep 86 s, fit 11 s, converged in 456 iterations, feature widths 10–133 tokens.
- **Same data as EMBER.** Its V* for Cannabis is the 4,713 tokens with 1,803 concept / 2,326
  neutral / 584 both, exactly our EMBER fit's numbers (`ConceptDataset`, English `data/`).
- **Same judge as ours.** `select.py` uses `models/gemini-3.5-flash-lite`.
- **CINDER's edit can run through EMBER's edit code unchanged.** EMBER subtracts
  `δ · F G_iᵀ` from each concept-only token (`ember/erasure/embed_edit._erase_factored`). CINDER
  subtracts `δ · T⁻¹ (Z̃ y_i)`. With `F = (Z̃ᵀ T_inv)ᵀ` (each column mapped back with `T_inv`,
  row convention as in `sharpen.py`) and `G = Yᵀ`, the two are the same computation. So a
  converter into EMBER's file format makes the whole Part 1 pipeline (strength search,
  cross-lingual test) usable for CINDER.

## 3. What was missing, and how it was resolved (2026-10-18)

| Missing | Resolution |
|---|---|
| `experiments/whitening/data/gemma/` (background stats, `defaults.json`, targets; git-ignored by Ori) | **Rebuilt** with Ori's `background.py --models gemma`: the same seeded Wikipedia shard (train-00040, byte-identical to the copy in his cache), sampling stops at the 300k cap (107,472 token types). Check: the top-5 sharpening eigenvalues for Harry Potter, Ancient Rome and Valentine's Day equal the ones in his `reports/compare_backgrounds_gemma.md` (all 15 values). `defaults.json` written by hand with his documented η 0.05, ε_rel 10⁻⁴, γ 1. Targets for the 6 concepts: each concept's current English Wikipedia page (`build_all_targets.py`'s rule; for Gun the local raw page is "Firearm", so the "Gun" page is fetched). |
| CINDER's erasure code | Not needed: CINDER's features are converted to EMBER's file format and run through Part 1's pipeline (D6). Ori's `ly_erase` runs (1 Titan GPU, no 32B Alpaca judge) are not used. |
| The expansion judge (lexicon, EJ@t) | **Copied from Ori's folder**: `overlap_judge.py` (normalization, feature words), `j3_eval.j3_count`, `layer_eval.fc(cache, "HJ")`, concept words `j3_data/concept_words_v1.json` (prompt v1, 4 example concepts: Culture of Greece, Ancient Rome, Artificial intelligence, World War II). |
| Converter to EMBER's format; a way for the edit to read another folder | **Written** (`cinder/build_features.py`); `cinder.common.use_cinder_features()` points `ember.erasure.features.MF_OUTPUTS_ROOT` at `mf_outputs/cinder/` (nothing in `ember/` edited). |
| His results | Not needed: fits take seconds and were rerun (`results/varwidth/<concept>/rn/seed42/`). |

## 4. Differences from EMBER beyond the three layers

| | EMBER (paper's code, Part 1) | CINDER defaults (`VWConfig`) |
|---|---|---|
| allocation | fixed top 1% per feature | `method="m2"` by default; the doc's GS-norm is **`rn`** and must be set explicitly |
| support refit after allocation | no | **yes** (`refit=True`) |
| solver / stopping | `SparseMatrixFactorization`, patience 500 | Ori's ALS, patience 200 + support churn < 0.005 |
| per-feature cap | none | 0.2% floor, 3% cap (was 5% until 2026-10-17) |

A gain of "CINDER" over Part 1's EMBER therefore mixes the three layers with the refit and the
solver. Ori's own ladder starts from `fixed` + `plain` in his solver for this reason.

## 5. Decisions (taken 2026-10-18)

| # | Decision | Taken |
|---|---|---|
| D1 | CINDER variant | Ori's whole method, rung **L123**: CS + GS-norm (`rn`, refit off) + **HJ = TJ ∨ LO@5** on the top 20 by ρ (his code). We compare two methods, not layers. |
| D2 | Control for the solver / refit | None: Ori's solver is part of his method. |
| D3 | Background, η, γ, ε, targets | Ori's (rebuilt, see §3): wiki background, η 0.05, γ 1, ε_rel 10⁻⁴, page target. |
| D4 | Sharpening target | The concept's page (his default). Circularity caveat stays in the report. |
| D5 | Edit strengths | Measured: at equal δ CINDER's edit is half of EMBER's (ratio 1.93, inside the 0.5–2 band) → EMBER's 8 values (as Ori's `erase.py`); one extra step if the best δ is at a grid edge. |
| D6 | Erasure pipeline | Part 1's (fixed MC scoring, top 3 × 50 Alpaca with Qwen2.5-32B, same gates and pick rule). |
| D7 | Concepts | Cannabis, Gun, Nazism, Republic of Ireland, Culture of Greece, Golf. CINDER edits 4 (no feature for Nazism, Republic of Ireland). Culture of Greece is an example in his concept-word prompt (accepted by TJ anyway). |
| D8 | Model | Gemma-2-2b-it. |

## 6. Notebooks

The root notebooks, `low_ram_demands/` and `variable_width/` were **not moved**: they resolve paths
relative to the repo root (`Path.cwd()`, `sys.path.insert(0, "experiments/…")`), Ori's queue
(`queue_ml.json`) and patch scripts address them by path, and he is running from his own
copy. Proposal for Ori: move them under `notebooks/` (already in `.gitignore` for checkpoints),
update `queue_ml.json` and `scripts/`, and run them with the repo root as working directory.

## 7. Implementation (2026-10-18): `cinder/`

| Module | Does | Runs on |
|---|---|---|
| `build_features.py` | Ori's `run.py` (rn, refit off, whitened, seed 42; reuses an existing run) → his `layer_eval.build_cache` + `fc(..., "HJ")` → EMBER format (`mf_outputs/cinder/`) + `judge_rerun/Cinder/features/<C>/` (candidates, summary) | CPU + Gemini, ~1 min per concept |
| `edit_size.py` | ‖c_t‖/‖e_t‖ per edited token for both methods → `judge_rerun/Cinder/delta_grid.json` | CPU |
| `grid.py` | each δ: edit, fixed MC scoring (MMLU 50, QA_train, SimdomQA_train), records; fast screen → top 3; Alpaca completions of the top 3 | 1 GPU |
| `pick.py` | Qwen2.5-32B on the first 50 completions of the top 3; Part 1's metrics and pick rule | 3 GPUs |
| `final_test.py` | QA_test + SimdomQA_test in 5 languages, MMLU test, edit check; `summary.csv` with `crosslingual/summary_fixed.write_summary` | 1 GPU |
| `compare.py` | CINDER vs EMBER table → `judge_rerun/Cinder/comparison.md` | CPU |
| `run_gemma.sbatch`, `run_pick.sbatch` | Slurm wrappers (resources as Part 1's EMBER jobs) | |

The edit: EMBER subtracts δ·F′G′_tᵀ from each concept-only token with a nonzero coefficient on a
selected feature. CINDER's F = (Zᵀ T⁻¹)ᵀ (its directions mapped back to embedding space; T⁻¹ =
S^(−γ/2) W⁻¹) and G = Yᵀ, so `e_t − μ_B ≈ F G_tᵀ` and the same code applies. Checks: the Part 2 status in
`progress.md` (selection reproduces EMBER's picks on EMBER's data; edited-token
counts; exact restore; mapping error 5·10⁻⁶).


## 8. Lexicon judge: Ori's vs our reconstruction (2026-10-18)

Before Ori's code was available we wrote a lexicon judge from his document (prompt: rare, specific words).
Scored on his hand labels (`j3_data/labels.json`; his CS + GS-norm fits, 18 concepts × 5 seeds = 90 runs,
1,800 top-20 candidates; his 4 prompt examples and our example Harry Potter left out; every accept of our
lexicon is labelled, so the comparison is fair). P = specific to the concept, B = broader or adjacent.

| Rule | Accepts | Precision (P) | Runs hit, P | Runs hit, P+B |
|---|---|---|---|---|
| TJ (EMBER's judge) | 77 | 0.84 | 55 | 56 |
| EJ@5, Ori's lexicon | 82 | 0.72 | 58 | 75 |
| EJ@5, our lexicon | 47 | 0.98 | 46 | 46 |
| **TJ ∨ EJ@5, Ori's (CINDER L3)** | 104 | 0.69 | **61** | **76** |
| TJ ∨ EJ@5, ours | 82 | 0.85 | 60 | 61 |

Both two-way judges find a feature specific to the concept in the same number of runs (61 vs 60). Ori's adds
broader features (76 runs with P+B); ours is more precise and adds almost no broader ones. Neither dominates,
and t = 5 was chosen by Ori on these labels. On our 6 concepts at seed 42 the two give identical decisions, so
the choice does not affect the CINDER vs EMBER run; CINDER keeps Ori's judge. Our earlier Nazism feature came
from the fit settings (refit on, ρ ≥ 2 candidates), not from the judge.

## 9. Ori's notebook write-up ("Cinder-Contrastive-Ember", Harry Potter) vs the run (2026-10-18)

| His write-up | Our CINDER run |
|---|---|
| Whitening against the background alone finds no HP direction; whitening + sharpening toward the concept page does | Same: whitened matrix with sharpening (`whitening/data/gemma`, identical to his files; η 0.05, γ 1, ε 10⁻⁴, page target) |
| 300 sampled concept sentences + neutral sentences → V* | Same V* as EMBER (checked: identical token sets) |
| "Fixed budget" (global) sparsity instead of 1% per feature | His final version: `rn`, floor 0.2% / cap 3%, refit off |
| Stage-2 judge shown token → source words (idea 3) | Not in L123 (that is his "WJ" variant, which lost precision); L123 = TJ ∨ LO@5 |
| Erasure of the support's concept tokens, directions mapped back | Same rule as his `erase.py` (edited token sets identical) |
| Extrapolation to tokens outside V* (naive projection; sparse Y′ with Z fixed) | **Not used** — notebook experiments only, not in `erase.py`; a possible extension |

His two direction findings, recomputed on our concepts (`judge_rerun/Cinder/direction_geometry.csv`):
- **Orthogonality reproduces in the whitened space**: CINDER's concept features have median |cos| 0.01–0.02 and
  max |cos| 0.15–0.39 to the other 99 features, against 0.16–0.30 and 0.47–0.89 for EMBER's. **After mapping back
  to embedding space, where the edit acts**, they stay fairly separate for Cannabis, Gun and Golf (max 0.16–0.28)
  but not for Culture of Greece (max 0.58, median 0.29, like EMBER).
- **The smaller norm does not generalize**: his HP feature fell from the 94th to the 27th norm percentile; on our
  concepts it falls for Golf (75 → 20) and Culture of Greece (99 → 83), not for Cannabis (84 → 92) or Gun.
