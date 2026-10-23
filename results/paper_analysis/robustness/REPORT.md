<!-- Saved by the lead from the WP agent's hand-back (a hook blocked the agent's own write); text unchanged. -->

# WP4 — Robustness: abstract concepts and contaminated concept sets

## Question

1. *"For some abstract, more delicate concept (like stoicism) which are the intersection of few 'stronger' topics (ancient rome, philosophy, art) – it may even miss to detect this somewhat delicate concept direction – Todo: make experiment and back up this statement and direction!"*
2. *"Is the sharp contrast between Harry Potter and Neutral tokens what allows us to detect HP direction? If we insert a very few tokens that 'fade' this sharp contrast (like tokens from 'magic' wikipedia page, or 'Lord of the Rings') will it damage the factorization badly? How much?"*

## Method (short)

- **Fits.** `experiments/varwidth/core.py` with the project configs:
  - EMBER = raw E, fixed RS-1%.
  - CS = CS + RS-1%.
  - CSGS = CS + GS (CINDER's factorization).
  - All with k = 100, no refit, seeds 42–46.
  - CS uses the project's wiki background and defaults (η 0.05, γ 1, ε_rel 1e-4), with the concept's own page as the sharpening target.
  - 615 fits in total (525 + 90 control).
  - The pipeline reproduces the project's prep exactly: V*, codes and A for Harry Potter and Uranium, max |ΔA| = 0. A GPU refit gave the project's CPU fit (identical support, max |ΔY| < 2e-6). So the 88 existing S = 300 fits were reused.
- **Part A concepts.** 13 abstract / intersectional concepts:
  - 8 new long pages fetched now: Romanticism, Utilitarianism, Baroque, Gothic architecture, Surrealism, Art Nouveau, Neoclassicism, Gnosticism.
  - Existentialism, already in the project.
  - All of these use |S_C| = 300.
  - Stoicism, Epicureanism, Impressionism and Neoplatonism have only 197–233 page sentences. They ran at |S_C| = 190, next to the 4 entity controls (Harry Potter, Baseball, Uranium, Golf) sub-sampled to 190 sentences.
  - The entity controls were also run at 300. The same 300 neutral sentences were used everywhere.
- **Part B.** For Harry Potter, Baseball and Ancient Rome, a fraction f ∈ {0, .1, .25, .5} of the 300 concept sentences was replaced by sentences from adjacent pages. Positions are fixed and nested in f, and |S_C| = 300 is kept.
  - Adjacent pages: HP — Magic (supernatural), The Lord of the Rings, The Chronicles of Narnia, Fantasy literature; Baseball — Softball, Cricket; Rome — Ancient Greece, Byzantine Empire.
  - Sentences naming the concept itself were dropped.
  - Conditions: EMBER, CS, CSGS and CS_sc (sharpening target = the contaminated S_C itself).
  - **Control**, added after seeing the first results: the same positions replaced by sentences of unrelated project concepts (Giraffe, Tuberculosis, Prime number, Flush toilet), run with EMBER and CS.
- **Judge-free blind labels (zero Gemini).** For every run, each of the top-10 features by ρ was shown as its 20 strongest tokens with source words (`piece[word]`). Condition, seed and f were hidden, items were shuffled within each concept, and exact duplicates were merged.
  - Each item was labelled P (specific to the concept), B (broader: a parent or neighbouring topic) or N (not the concept).
  - The annotators were independent Claude sub-agents that saw only the item files. Part A had 3094 items over 8 annotators; part B had 2952 new items over 8 annotators. Labels of identical items were reused.
  - A second annotator re-labelled a random 12% of part A (372 items): raw agreement 0.94, Cohen κ = 0.88 (3-way), κ = 0.96 for P vs not-P.
- **Metrics per run:**
  - has_P@k: a P feature among the top k by ρ (k = 1, 3, 10).
  - P rank: the ρ-rank of the best P feature. "None" means no P in the top 10, set to 11 in means.
  - Seed stability: mean pairwise Jaccard of the best P features' 20-token sets across seeds.
  - Lexicon hits: the 20 source words against 20 concept words. For new concepts the lexicons were written by the analysis agent before any fit (`concepts.py`).
  - Intruder share: the share of the P feature's top-20 tokens that occur only in the inserted sentences, never in the clean S_C.

## Backed findings

**A1. Abstract concepts rarely yield a concept-specific feature at the top, under every method.**

Over 65 runs per method (13 abstract (concept, S) units × 5 seeds), the top-ρ feature was P in 0/65 (EMBER), 0/65 (CS) and 1/65 (CSGS) runs. For the entity controls (8 units × 5) it was 13/40, 23/40 and 18/40. The top feature was B (a parent topic) in 41/65, 43/65 and 34/65 abstract runs.

At |S_C| = 300 (9 abstract vs 4 entity concepts, concept = unit):

| | P@1 | P@3 | P@10 [95% CI] | mean P rank (none = 11) | stability |
|---|---|---|---|---|---|
| abstract, EMBER | 0.00 | 0.07 | 0.44 [0.11, 0.78] | 8.5 | 0.64 |
| abstract, CS | 0.00 | 0.00 | 0.49 [0.20, 0.78] | 9.0 | 0.88 |
| abstract, CSGS | 0.00 | 0.18 | 0.67 | 7.2 | 0.75 |
| entity, EMBER | 0.40 | 0.50 | 0.60 | 5.6 | 0.50 |
| entity, CS | 0.65 | 0.65 | 0.70 | 4.4 | 0.62 |
| entity, CSGS | 0.45 | 0.45 | 0.65 | 5.9 | 0.54 |

- EMBER, abstract vs entity: P@3 0.07 vs 0.50, Mann–Whitney p = 0.043 (n = 9 vs 4). P@10 0.44 vs 0.60, p = 0.61 (not significant).
- At |S_C| = 190: P@10 is 0.25 under all three methods for the 4 short-page abstract concepts, vs 0.75 for the matched entity controls (MWU p = 0.17, n = 4 vs 4).
- Files: `stats_A.json`, `concepts_A.csv`, `table_A_per_concept.csv`.

**A2. Six of the 13 abstract concepts have no P feature in the top 10 in any of their 15 runs** (3 methods × 5 seeds): Stoicism, Epicureanism, Neoplatonism, Gnosticism, Gothic architecture, Art Nouveau.

- **Judge-free confirmation over all 100 features:** a feature holding ≥ 5 of the 20 concept words exists in 9% (EMBER), 11% (CS) and 6% (CSGS) of abstract runs, vs 95%, 100% and 100% of entity runs. The best feature's mean hits are 3.0 / 3.2 / 3.0 vs 7.4 / 7.9 / 7.9.
- **Stoicism (seed 42) shows the pattern.** The top features are emotions / passions, philosophy / religion, ethics / virtue, and discourse words (therefore, whether, claims, reason).
- The most Stoic feature is a spelling-based *sto-* feature: Stoic, Stoicism, stoa and neo-stoic mixed with stock, store and stand, with 4–7 lexicon hits. It sits at ρ-rank 39 (EMBER), 13 (CS) and 16 (CSGS).

**A3. Where EMBER's top features land (abstract concepts, top 10, B labels).** They fall on the parent topics, which were fixed before the fits:

| Concept | Where the B items fall |
|---|---|
| Gothic architecture | generic architecture, 17 of 22 |
| Art Nouveau | decorative arts / visual art, 26 of 30 |
| Neoplatonism | religion / mysticism, 13 |
| Gnosticism | Christianity / theology, 9 |
| Impressionism | painting, 11 |
| Surrealism | painting, 10 |
| Romanticism | literature / poetry, 6 |
| Existentialism | generic philosophy / ethics, 6 |
| Utilitarianism | generic philosophy / ethics, 5 |

The N items are discourse vocabulary: frequency and degree adverbs, connectives, generic abstract nouns and verbs.

**A4. CS does not recover abstract concepts.**

- **Paired over the 9 abstract concepts at S = 300:**
  - CS − EMBER P@10 = +0.04 (95% CI [−0.07, 0.20]; 1 up / 1 down; Wilcoxon p = 1.0).
  - CSGS − EMBER P@10 = +0.22 ([0.00, 0.56]; 2 up / 0 down, Baroque and Utilitarianism; p = 0.5).
  - CSGS − EMBER P@3 = +0.11 (4 up / 1 down, p = 0.38).
- **At S = 190:** all paired differences are 0.
- **What CS does change is stability:** when a P feature exists, it is more reproducible across seeds (Jaccard 0.88 vs 0.64 at S = 300).
- **For entity concepts CS behaves as the brief describes:** Harry Potter P@1 goes from 0.4 to 1.0 (S = 300) and from 0.0 to 0.8 (S = 190). Entity P@1 overall goes from 0.40 to 0.65.

**B1. Replacing up to half of S_C with adjacent-page sentences does not break CS. It degrades EMBER, but no more than unrelated replacement does.**

Pooled over 3 concepts × 5 seeds (n = 15 runs per cell):

| | f = 0 | 0.1 | 0.25 | 0.5 | unrelated control at 0.1 / 0.25 / 0.5 |
|---|---|---|---|---|---|
| EMBER P@10 | 0.80 | 0.67 | 0.73 | 0.53 | 0.93 / 0.67 / 0.60 |
| EMBER P@1 | 0.47 | 0.33 | 0.13 | 0.20 | 0.33 / 0.00 / 0.07 |
| EMBER mean P rank | 4.1 | 5.4 | 5.5 | 6.9 | 4.1 / 6.2 / 6.8 |
| CS P@10 | 1.00 | 0.93 | 0.80 | 0.93 | 0.87 / 0.93 / 1.00 |
| CS mean P rank | 1.3 | 1.9 | 3.3 | 1.8 | 2.7 / 2.5 / 1.5 |
| CSGS P@10 | 0.87 | 0.80 | 0.60 | 0.80 | – |
| CS_sc P@10 | 1.00 | 0.93 | 0.67 | 0.87 | – |

- **EMBER's rank loss is the same with adjacent and unrelated sentences:** mean rank over f > 0 is 5.9 vs 5.7 (MWU p = 0.96). So it reflects fewer concept sentences (dilution), not the fading of the contrast.
- **Baseball** is robust under every method: P@10 = 1 at all f.
- **Ancient Rome:** under EMBER, P@10 falls from 1.0 to 0.4 at f = 0.5, and the same happens with unrelated pages. Under CS it stays at 1.0, with rank 1–2.
- **Harry Potter:** under EMBER it is already weak at f = 0 (P@10 0.4; median rank 12 for the judge-free names feature; ρ 3.9). That feature falls to rank 43 (ρ 1.8) at f = 0.25–0.5.

**B2. What adjacency does specifically: neighbour tokens enter the concept feature.**

- **Intruder share of the P feature's top-20 tokens at f > 0:** EMBER 0.19 (adjacent) vs 0.04 (unrelated), MWU p = 3e-8. CS 0.11 vs 0.00, p = 1e-12.
- **It grows with f (f = 0.1 / 0.25 / 0.5):**
  - EMBER 0.14 / 0.20 / 0.25
  - CS 0.05 / 0.13 / 0.14
  - CSGS 0.03 / 0.08 / 0.11
  - CS_sc, with the contaminated target, 0.06 / 0.13 / 0.23. The clean own-page target cuts the intrusion at f = 0.5 by about 40% (0.14 vs 0.23).
- **Examples:**
  - EMBER's HP feature at f = 0.25, rank 43: Gryffindor, Dumbledore, Slytherin, Rowling, **Tolkien**, Weasley, Voldemort, Hermione, Snape, **Gandalf, Beowulf, Kipling, Tolkien**, Hogwarts, **Quixote, Hobbit**, …
  - CS's HP feature at f = 0.5, rank 1: the HP names, with Tolkien and Gandalf at positions 16–18.

**B3. CS's HP feature is not always formed.**

- When the HP names feature is present under CS, it is rank 1.
- It is absent from the top 10 in 1, 3 and 1 of 5 seeds at f = 0.1, 0.25 and 0.5 with adjacent pages, and in 2, 1 and 0 of 5 with unrelated pages.
- In the failing seeds no feature collects the names at all. They are scattered over punctuation and function-word features with ρ < 2, and the generic "magic / wizard" and "stories / tales" features take ranks 1–2.
- This all-or-nothing behaviour across seeds is not specific to adjacency: 10/15 adjacent runs have a P feature vs 12/15 unrelated runs.

**Mechanism probe (null result).**

- The share of capitalized (proper-noun-like) tokens among concept-only tokens does not predict P@10: Spearman ρ = 0.13, p = 0.58, n = 21 units (abstract 0.22 vs entity 0.19).
- Neither does the number of concept-only tokens: ρ = 0.12, p = 0.60.
- File: `mechanism_capshare.csv`.

## Speculations (each with a test)

- **Abstract pages are written in a register, not around a vocabulary of names.**
  - Their concept-only tokens are dominated by discourse and parent-field words that random neutral sentences lack. So the high-ρ slots go to register and parent-topic features.
  - The names (Zeno, Seneca, Plotinus) are too few, and they get absorbed into spelling-based or name-fragment features.
  - Test: replace the 300 random neutral sentences with 300 sentences from parent-topic pages (philosophy, art history). If the register features lose ρ, the concept feature should rise.
- **CS cannot help because the sharpening target is the same page.** It amplifies the page's register as much as its names.
  - Test: sharpen with a "page minus parent-topic page" target.
- **Dilution, not contrast, drives EMBER's loss on HP and Rome.**
  - Test: pure removal (|S_C| = 300 − n, nothing inserted).
- **The all-or-nothing HP feature under CS (B3) is an initialization basin.**
  - Test: more seeds (20) and an SVD initialization.

## Paper paragraph

We tested two robustness questions with blind, judge-free labels from two independent annotators (κ = 0.88). First, concepts that are intersections of broader topics (e.g., Stoicism, Gnosticism, Gothic architecture; 13 concepts, 5 seeds) almost never produce a concept-specific feature at the top of the ρ ranking. The top feature was concept-specific in 0/65 EMBER runs and 0/65 CS runs, against 13/40 and 23/40 for entity concepts, and for 6 of the 13 concepts no method placed a specific feature in the top 10 in any run; their top features cover the parent topics (architecture, religion, painting) and discourse vocabulary. Contrastive sharpening does not recover these concepts (paired gain in top-10 presence +0.04, not significant); only CS + GS gains nominally (+0.22, from 2 of 9 concepts). Second, when up to half of the concept sentences are replaced with sentences from adjacent pages (e.g., The Lord of the Rings for Harry Potter), the CS concept feature remains in the top 10 in 80–93% of runs at mean ρ-rank 1.8–3.3, while EMBER's mean rank falls from 4.1 to 6.9. The same fall occurs when the sentences come from unrelated pages, so it reflects dilution rather than a loss of contrast. What adjacency does cause is contamination of the feature itself: 14–25% of the top tokens of EMBER's concept feature and 5–14% of CS's come from the neighbouring topic (Tolkien, Gandalf), against at most 9% and 0% with unrelated pages.

## Appendix text

**Data.**
- New concepts were fetched from the English Wikipedia API on 2026-10-22 with the project's `page_paragraphs` / `split_sentences` (≥ 5 words, citations removed). They were sampled with `random.Random(42)`, keeping page order; revisions are in `data/pages/*.json`.
- Pages with < 300 sentences (Stoicism 204, Epicureanism 233, Impressionism 212, Neoplatonism 197) form the S = 190 group. The entity controls were sub-sampled to 190 of their 300 project sentences (`random.Random(42)`).
- Pages checked and rejected as too short: Transcendentalism 94, Rococo 219, Pragmatism 275, Mannerism 256, Scholasticism 109, Cubism 296, Platonism 104, Empiricism 192, Expressionism 119.
- CS targets are the token types of the whole page, minus special, unused and byte tokens. The builder reproduces the project's `targets/Harry_Potter.json`.

**Contamination.**
- The replaced positions are `perm[:n]`, where `perm` is a `random.Random(42)` permutation of 0..299 and n = round(300 f).
- Inserted sentences are interleaved round-robin over the adjacent pages after a seeded shuffle.
- Sentences containing a name form were removed: HP — potter, hogwarts, rowling; Baseball — baseball; Rome — rome, roman.
- Pool sizes: 1270 (HP), 1000 (Baseball), 626 (Rome).
- The neutral side and the V*/code construction are the project's (`ConceptDataset`, neutral seed 42, `build_vstar`).

**Fits.**
- VWConfig(method = fixed or rn, refit = False), rank 100, sparsity 0.01, GS floor 0.2% / cap 3%, tol 1e-4, patience 200.
- Run on one c-001 GPU or on CPU-only SLURM (`cl_pa_rob`); GPU and CPU fits are identical.

**ρ.** Mean |Y| over concept-only tokens divided by (mean |Y| over neutral-only tokens + 1e-8), as in `compute_embedding_stats`.

**Labels.**
- An item is the 20 strongest tokens of a feature, shown as `piece[word]` (the most frequent source word in S_C ∪ S_N).
- Annotator instructions: P = a reader would name the concept from the tokens; B = a parent or neighbouring topic, or generic field vocabulary; N = unrelated (punctuation, numbers, function words, another topic).
- Parent topics per concept were fixed in `concepts.py` before any fit.
- The lexicons of the 12 new concepts were written by the analysis agent before any fit, following `prompt_v1.txt`. The other concepts use `concept_words_v1.json`.

**Judge-free concept feature for part B (not used for conclusions).** The `cf_*` columns of `runs_B.csv` take the feature with the most lexicon hits. For HP it picks the generic magic feature, because the lexicon contains magic, spell and wand. A lexicon restricted to words absent from the adjacent pages is too small for Baseball (7 words).

**Per-concept results (part A).** The full table (P@10 / P@3 / median P rank) is in `table_A_per_concept.csv`.
- Abstract concepts with P@10 > 0 under some method (P@10 for EMBER / CS / CSGS):
  - Romanticism 1.0 / 1.0 / 1.0
  - Existentialism 1.0 / 1.0 / 1.0
  - Neoclassicism 1.0 / 1.0 / 1.0
  - Surrealism 1.0 / 0.8 / 1.0
  - Impressionism (190) 1.0 / 1.0 / 1.0
  - Utilitarianism 0.0 / 0.6 / 1.0
  - Baroque 0.0 / 0.0 / 1.0
- Their best P rank is 3–9, never 1, except one CSGS Impressionism run.
- Entity concepts:
  - Baseball: P rank 1 everywhere.
  - Harry Potter: EMBER P@10 0.4 (S = 300) and 0.6 (S = 190); CS 1.0, at rank 1.
  - Golf: P@10 0.8–1.0.
  - **Uranium:** no P feature under CS / CSGS; its features are labelled generic nuclear / chemistry (B). Under EMBER it has a P feature only at S = 190, in 2 of 5 seeds.

## Figures

- `figures/fig_A_rank.pdf` (.png, .csv) — *ρ-rank of the best concept-specific (P) feature in each run (5 seeds; blind labels of the top 10) for 13 abstract and 4 entity concepts: EMBER (grey circles), CS + RS-1% (blue squares), CS + GS (orange triangles). "None" = no P feature in the top 10. Entity controls are shown at |S_C| = 300 and at the |S_C| = 190 of the short-page abstract group.*
- `figures/fig_A_summary.pdf` — *Share of runs with a P feature at rank 1, ≤ 3 and ≤ 10, averaged over (concept, |S_C|) units (13 abstract, 8 entity).*
- `figures/fig_B_curves.pdf` — *Contamination: a fraction f of the 300 concept sentences is replaced by sentences from adjacent pages (solid) or unrelated pages (dashed, open markers). Rows: share of 5 seeds with a P feature in the top 10; mean ρ-rank of the best P feature (none = 11; axis inverted); share of its 20 strongest tokens that come only from the inserted sentences (mean over runs with a P feature).*

## Sources

**Code** (`experiments/paper_analysis/robustness/`):
- `concepts.py` — concept sets, lexicons, parent topics
- `fetch.py` — Wikipedia pages, sharpening targets
- `variants.py` — S_C variants, prep check, job list
- `pipeline.py` — V*, A, fit, compact save
- `run_fits.py`
- `import_project.py`
- `analyze.py` — features, blind export
- `metrics.py`
- `figures.py`

**Outputs** (`results/paper_analysis/robustness/`):
- `data/` — pages, targets, variants, meta
- `fits/` — 615 fits
- `features.parquet` — 61,500 feature rows
- `labels/` — batches, hidden keys, labels, second-annotator file
- `runs_A.csv`, `concepts_A.csv`, `stats_A.json`, `runs_A_bestlex.csv`, `table_A_per_concept.csv`
- `runs_B.csv`, `curves_B.csv`
- `mechanism_capshare.csv`
- `figures/`

**Inputs:**
- Project fits `results/varwidth/<Slug>/{fixed_refit-False_plain, fixed_refit-False, rn_refit-False}/seed42–46` for HP, Baseball, Uranium, Golf, Existentialism and Ancient Rome.
- Background stats `experiments/whitening/data/gemma/bg_wiki_stats.pt`.
- `data/neutral_sentences.json`.

Gemini spend: 0.
