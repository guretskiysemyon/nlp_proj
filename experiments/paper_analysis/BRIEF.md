# Paper analyses: shared brief for every analysis agent

Read this whole file before you start. Then read your own work package (WP) in the prompt that launched you.

## 1. The project in one page

**EMBER** (Suslik, Shafran and Geva 2026; `2606.03695v2.md` in the repo root, read §3 and App. A) erases a concept
from an LM by editing only its *input embeddings*, in three steps:

1. **Decompose.** Take the tokens V\* of 300 concept sentences S_C (the concept's Wikipedia page) and 300 neutral
   sentences S_N. Factorize A = E_{V\*}ᵀ (d × n) ≈ Z Y with sparse MF: k features (Gemma: k = 100), every row of Y
   keeps only its top 1% entries (row sparsity, "RS-1%"), ALS with unit-norm Y rows.
2. **Identify.** Score each feature by the mass ratio ρ_i = Σ_{t∈C}|Y_it| / Σ_{t∈N}|Y_it|; keep ρ > τ = 2; an LLM
   judge (two stages: describe the top tokens without the concept name, then decide membership from the
   description at confidence ≥ 0.85) gives the concept features F_C.
3. **Erase.** For each concept-only token t with Y_it ≠ 0 on some i ∈ F_C: e_t ← e_t − δ Σ_{i∈F_C} Y_it z_i.

**CINDER** (our method) changes steps 1–2 and keeps the edit:

- **CS, Contrastive Sharpening.** Whiten against a background of ordinary Wikipedia token types,
  ũ = W(e − μ_B), W = (Σ_B + εI)^{-1/2}; then sharpen toward the concept page:
  x = S_η^{γ/2} ũ with S_η = (1−η)·(1/m)Σ_j ũ_j ũ_jᵀ + ηI over the whitened page token types (η = 0.05, γ = 1).
  Factorize X instead of E. Map directions back exactly with T⁻¹ = W⁻¹ S_η^{−γ/2}.
- **GS, Global Sparsity.** One budget K = 1% of all entries of Y, chosen globally by the scale-free score
  |y_ij| / ‖y_i‖₂, with a per-feature floor 0.2% and cap 3%. A feature's width becomes an output of the fit.
- **TWJ, Two-Way Judge.** EMBER's judge OR a lexicon match: one LLM call per concept writes 20 concept-specific
  words (pre-processing); a feature is accepted if ≥ 5 of its top-20 source words are in the lexicon (EJ@5).
- **CINDER-z.** Keep CINDER's selection (Y, F_C) but re-solve the dictionary Z against the *original* embeddings
  (ridge least squares with Y fixed) and edit with that. **Correction (WP6, Oct 22):** the old explanation ("whitening
  inflates low-variance directions the model barely reads") is wrong. The CS fit is centred at μ_B; the raw re-solve is
  not, so CINDER-z = CINDER + β_t μ_B exactly, and the first layer reads μ̂_B ~2.9× (Gemma) / 20× (Llama) random. That
  mean term explains the whole read-gain gap (see `results/paper_analysis/edit_direction/REPORT.md`).

**What the motivation document observed on Harry Potter (HP), and what you are asked to test more broadly:**

- In EMBER's raw matrix the HP feature ranks **11th by ρ** (ρ = 3.9), behind generic high-variance topics
  (magic, writing, books, criticism...). After CS it ranks **1st** (ρ = 125), 2.4× ahead of the next (magic).
- Intuition: sparse MF minimizes squared error, so it spends features on high-variance directions; the concept's
  tokens are few and (speculation) rarer, "seen less in training", hence carry little variance.
- The raw HP feature's tail holds Nestlé, Lego, Trump, Beatles, taxonomic fragments; the CS feature's tail holds
  Gringotts, Hogsmeade, Quidditch, Hagrid (positions 20–30: 2/11 HP vs 11/11 HP).
- CS features are closer to orthogonal and their cosines are interpretable (HP–magic 68°, other neighbours > 82°);
  raw features are not (HP's nearest raw neighbours at 35–54° are unrelated: punctuation, plural nouns).
- **Norm puzzle:** in the raw fit the HP feature has the largest ‖z‖ (1.55% share, 94th percentile); after CS its
  ‖z‖ is ordinary (0.84%, 27th percentile). Unexplained.
- After CS the matrix factorizes "more sparsely": the truncated-SVD rank that matches the sparse-MF error is
  67 (CS) vs 5 (raw) on HP.
- Fixed widths: EMBER gives every feature ~48 tokens; Ancient Rome splits into 2 features under EMBER; GS gives it
  one feature with 56–103 tokens; Baseball 15, Culture of Greece 113.
- Extrapolation to tokens outside the 300 sampled sentences (fixed Z, re-solve sparse Y) brings a few good tokens
  and many unrelated ones; unsampled tokens are rarer on the page and noisier.
- Known caveats: (a) **circularity** — the sharpening target is the same page whose tokens the purity metrics
  score; with a target that excludes every matrix token ("leak-free") the CS gains mostly vanish; (b) the
  "concept-only" label (occurs in S_C, never in S_N) admits generic words (get, win, heart), and CS features absorb
  them; (c) η, γ were tuned on HP, Ancient Rome and Valentine's Day.

**Erasure results you should know but do not need to reproduce** (collateral-aware evaluation, 12 held-out
concepts × 5 seeds, Gemma): EMBER's own protocol (δ up to 200) rewards *token destruction* (r ≈ 100, edited rows
~100× their length); with a collateral term every method moves to r ≈ 1; CINDER edits in runs where EMBER finds
nothing (14 vs 3 discordant runs); at equal edit size (r = 0.5) CINDER's *direction* removes more (+0.15 H3v);
a 2×2 swap shows the gain comes from the direction, not the token set. Multilingual (Llama-3.1-8B, 8 languages):
one shared CINDER-z fit ties per-language EMBER at 1/8 the selection cost; CINDER finds concept features for ~2×
more concepts than multilingual EMBER. Sources: `CINDER_collateral_evaluation.md`, `CINDER_improvements.md`,
worktree `../nlp_proj_ml/experiments/ml/RESULTS.md`.

**Your job is steps 1–2 (the embedding space and the directions), not erasure.** We want the observations above
tested across many concepts and seeds, explained (mechanism / math), and reported honestly: if an observation
does not generalize, that is a result. Aim for clear, figure-ready evidence a sharp professor would accept.

## 2. Where things are (repo root `/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj`, branch `orip`)

- Python: `.venv/bin/python`, run modules from the repo root: `.venv/bin/python -m experiments.paper_analysis.<module>`.
- **Prep cache** `results/varwidth/prep/<Slug>_<plain|whitened>_S300.pt` (load with `torch.load(..., weights_only=False)`
  or `experiments.varwidth.prep.prepare(concept, matrix)`), keys: `A` (d × n; raw E_{V\*}ᵀ for `plain`, the CS
  transform Xᵀ for `whitened`), `vstar_ids`, `tokens`, `codes` (1 = concept-only C, 0 = neutral-only N, 2 = both B),
  and for whitened `T_matrix`, `T_inv` (d × d, row convention: X = (E − μ_B) T, directions back z = z̃ T_inv),
  `mu_B`, `eta`, `eps`, `n_target`. Gemma embeddings are ×√d (TransformerLens convention), float32.
- **Saved fits** `results/varwidth/<Slug>/<tag>/seed<N>/fit.pt` with `Z` (d × k), `Y` (k × n), `widths`,
  `rel_err`; `summary.json` (F_C, candidates with ρ and Gemini descriptions); `judge.json`; `layer_cache.json`.
  Tags: **EMBER = `fixed_refit-False_plain`** (L0), **CS + RS-1% = `fixed_refit-False`** (L1),
  **CS + GS = `rn_refit-False`** (L12; CINDER = L12 with judge HJ = TJ ∨ EJ@5),
  plain GS = `rn_refit-False_plain`, RS-2% = `fixed_sparsity-0.02_refit-False[_plain]`. Seeds 42–46 for 38 concepts
  (`experiments/varwidth/layer_eval.py` CONFIGS/RUNGS; concept lists in `experiments/varwidth/j3_eval.py`,
  `data/concept_sentences.json`, `data/concept_sentences_extra.json`). Check which (concept, tag, seed) exist.
- Fitting code: `experiments/varwidth/core.py` (`fit_varwidth(A, VWConfig, seed, device)`), `common.py`
  (`VWConfig`, `run_dir`), `prep.py`, `select.py` (ρ filter + judge), `run.py`. CS: `experiments/whitening/sharpen.py`,
  background stats `experiments/whitening/data/gemma/bg_wiki_stats.pt`, defaults `.../defaults.json`,
  page targets `.../targets/<Slug>.json`. A fit of one concept takes ~15–30 s on a GPU, a few minutes on CPU.
- Lexicons (20 concept words per concept): `experiments/varwidth/j3_data/concept_words_v1.json`.
  Hand labels of judged features (P/B/N): `experiments/varwidth/j3_data/labels*.json`.
- Erasure edits (for edit-direction work): `results/varwidth/layers/collateral/<Slug>/edits_s<seed>.pt`;
  see `experiments/varwidth/erase_v2.py` and §D16 of `CINDER_collateral_evaluation.md` (read gain, A3z).
- Wikipedia: internet works (Wikipedia API); `experiments/varwidth/new_concepts.py` shows how sentences were fetched.
  The HF cache (`HF_HOME=/home/morg/NLP_2526b/oripetel/.cache/huggingface`) has gemma-2-2b-it, Llama-3.1-8B-Instruct
  and `wikimedia/wikipedia`.

## 3. Rules

- **Write only inside** `experiments/paper_analysis/<your_wp>/` (code) and `results/paper_analysis/<your_wp>/`
  (outputs). Import existing modules; never edit files outside your folders. Do not `git commit` (the lead commits).
- **Gemini: zero by default.** Prefer judge-free metrics (ρ rank, concept-only share, lexicon hits, AUC) and your
  own blind labels: write the items to label without condition names, in shuffled order, label them, then unblind.
  If a WP truly needs the Gemini judge, use the existing cached client in `experiments/varwidth/select.py`
  (`JudgeCache`), stay under 0.25 NIS, and log what you spent.
- **Compute.** CPU first (`torch.set_num_threads(4)`; the login node c-001 has 16 cores shared by several agents).
  For many fits, submit CPU-only SLURM jobs (no `--gres`, so they do not count against the shared 12-GPU cap) with
  job names starting `cl_pa_`. A local GPU on c-001 is allowed only if `nvidia-smi` shows it idle, one GPU per
  agent at most. Never cancel jobs you did not start; never `scancel -u`. Other Claude sessions share this account
  (`cl_2x2_*`, `ml_*` jobs): leave them alone.
- **Seeds and statistics.** Use seeds 42–46 where fits exist; report per-concept values and paired comparisons
  (concept as the unit; Wilcoxon signed-rank, bootstrap 95% CI over concepts). State n.
- **Honesty.** Separate *backed findings* (with numbers and the script that produced them) from *speculations*
  (each with how to test it). Report failures and nulls plainly.
- **Time.** Aim to finish within ~3 hours of wall time. If something will take much longer, deliver what you have
  and describe the remaining step.

## 4. Deliverables (in `results/paper_analysis/<your_wp>/`)

1. `REPORT.md`, short and structured: Question (quote the TODO it answers) · Method · Backed findings (numbers,
   n, tests) · Speculations (+ test) · **Paper paragraph** (3–6 sentences for the main text, plain academic English,
   no hype) · **Appendix text** (the low-level detail: definitions, settings, full table) · Figures (path +
   caption) · Sources (scripts, inputs).
2. Figures as **PDF** (also PNG for quick viewing), matplotlib, single-column width 3.3 in (or 6.8 in for
   two-column), font size 8–9, no titles inside the plot (captions go in the paper). Colours: EMBER `#7F7F7F`,
   CS-only `#0072B2`, CINDER (CS+GS) `#D55E00`, CINDER-z `#E69F00`, other `#009E73`. Save the plotted data as CSV/JSON
   next to each figure.
3. Your final message to the lead: a ≤ 250-word summary of the backed findings with the key numbers, the figure
   paths, and anything the lead must know (caveats, surprises, contradictions with the brief).
