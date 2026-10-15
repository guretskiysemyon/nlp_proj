# Cross-lingual erasure — results summary tables

**Snapshot date:** 2026-09-30 (`results/cross_lang_test/summary.csv` was last written 2026-09-15)
**Scope:** `google/gemma-2-2b-it`, rank 100, seed 42, multiple-choice QA, 8 concepts × 3 erasure methods × 5 languages.

This file condenses the raw results into a small number of tables, explains every table and every column, and
contains the code that produced all of them (§5). Every number below is recomputed from the raw files by that code;
none is typed in by hand. It complements [`progress_gemini.md`](progress_gemini.md), which is the narrative execution record (what
was run, what broke, design rationale). Where the two overlap, the numbers agree (§4 of `progress.md` was checked
against this file); the one deliberate difference is noted under Table 3.

**Contents**

1. [The setup in one paragraph](#1-the-setup-in-one-paragraph)
2. [Metric reference](#2-metric-reference)
3. [Tables](#3-tables) — Table 1 data checks · Table 2 per method · Table 3 per concept (overview) · Table 4 per language ·
   Table 5 what predicts leakage · Table 6 one table per concept (6.1–6.8) · Table 7 stability of the English denominators
4. [Caveats](#4-caveats)
5. [Code](#5-code)

---

## 1. The setup in one paragraph

For each concept, an erasure method removes that concept from the model by editing weights **once**, using
hyper-parameters the English grid search had already chosen. The edited model is then asked the concept's
multiple-choice questions in English and in four other languages (German `de`, Spanish `es`, French `fr`,
Russian `ru`). The question is whether the concept is also gone in the other languages, or whether it "leaks" back.

Three methods (called *configs* in the code) are compared:

| Method | What is edited | Why it matters here |
|---|---|---|
| `ember` | Input-embedding rows of the tokens that express the concept (EMBER). | The edit lives on **English** token embeddings, so other languages that use different tokens may be untouched. |
| `snmf` | MLP weights (`up_proj` rows and `down_proj` columns) in a range of layers, along SNMF feature directions. | The edit is inside the network, past the embedding layer, so it is not tied to specific input tokens. |
| `snmf_ember` | Both of the above. | The paper's headline configuration. |

Per concept there are 4 configs (the three above plus an unedited `pristine` model) × 5 languages. The `pristine`
model is used only to provide denominators (§2).

---

## 2. Metric reference

| Metric | Definition | Direction / range |
|---|---|---|
| **Accuracy** (`qa_acc`) | Share of the 50 `QA_test` multiple-choice questions answered with the correct letter. An unparseable answer counts as wrong. | 0–1; chance = 0.25 |
| **Pristine accuracy** (`qa_base`) | The same accuracy on the unedited model, **in the same language**. It is the denominator of retention. | 0–1 |
| **Retention** (`qa_frac`) | `clip((accuracy − 0.25) / (pristine accuracy − 0.25), 0, 1)` — the paper's Eq. 22. | 1 = concept knowledge fully intact; 0 = reduced to chance. **Lower = more erased.** Values at exactly 0 or 1 are saturated by the clip. |
| **Erasure** | `1 − retention`. | Higher = more erased |
| **EN retention** | Retention on the English questions. | As retention |
| **Δ (leak)** | `retention(language L) − retention(English)` for the **same method and concept**. | **Positive = leak** (the concept survives better in L than in English). ≈ 0 = equal erasure in both (or no erasure in either). Negative = L is *more* erased than English. |
| **Mean non-EN retention** | Average retention over the reliable non-English languages of a concept (3 for Harry Potter, 4 for the others). | As retention |
| **Mean Δ** | Average of Δ over the same languages. Identical to *mean non-EN retention − EN retention*. | As Δ |
| **Reliable / screened out** | A (concept, language) pair is screened out when the **pristine** model scores below 0.40 on it, because a weak baseline makes the ratio meaningless. Only Harry Potter/`ru` (0.30) is affected. | — |
| **Native wrapper** | The instruction text around each question (e.g. "Answer with the letter") is in the question's own language. An English-wrapper variant was also run; it is not used here because its effect is negligible (Table 1). | — |
| **Similar-domain QA retained** (`simdom_frac`) | The retention formula applied to 50 questions from a *neighbouring* topic that should **not** be erased (e.g. Roman history for the Greece concept), in English. | 1 = no collateral damage; **lower = more collateral damage** |
| **MMLU retained** (`mmlu_frac_en`) | English MMLU accuracy relative to the unedited model. One number per method, identical for every concept. | 1 = general ability intact |
| **Leak Δ ± SE** | Mean Δ over the 8 concepts ± the standard error *across concepts* (standard deviation ÷ √8). Concepts are treated as the independent units. | — |
| **Share of English erasure that transfers** | `mean over concepts of non-EN erasure ÷ mean over concepts of EN erasure`. A ratio of means, so it needs no threshold on weakly-erased concepts. The interval is a bootstrap: resample the 8 concepts with replacement 10,000 times (seed 0), take the 2.5th and 97.5th percentiles. | 1 = other languages are erased as much as English; 0 = no transfer |
| **Concepts with Δ > +0.05** | How many of the 8 concepts have a mean Δ above 0.05 (the "clear leak" threshold used in the code). | 0–8 |
| **Word overlap** | Per question: the share of the English question's distinct words (lower-cased, at least 4 characters) that also appear in its German/Spanish/French translation. Averaged over the 50 `QA_test` questions and the three languages. Russian is excluded (different script). | 0–1; higher = translations look more like English |
| **Tokens edited by EMBER** | `n_tokens_edited` in `best_embed.csv`: how many vocabulary tokens EMBER changes for the concept. | count |
| **EMBER δ** | EMBER's edit-strength hyper-parameter, chosen by the English grid (5–200). | — |
| **SNMF δ_in / δ_out @ layers** | Edit strength and layer range of the SNMF edit on MLP input weights (`up_proj`) and output weights (`down_proj`). | — |
| **Pearson r / partial r** | Linear correlation across the 8 concepts; the partial version removes the linear effect of a third variable. | −1 to +1 |
| **t (paired)** | Mean of the per-concept difference between two methods divided by its standard error, 7 degrees of freedom. | \|t\| ≈ 2.4 is the 5% two-sided threshold |

---

## 3. Tables

### Table 1 — Data checks

| Check | Value |
|---|---|
| Rows in summary.csv | 288 |
| Non-English cells (8 concepts x 3 methods x 4 languages) | 96 |
| ... of which reliable (pristine MC accuracy >= 0.40) | 93 |
| Screened-out (concept, language) pairs | Harry Potter/ru |
| Wrapper effect, native minus English scaffold (mean / mean absolute) | -0.014 / 0.089 |
| Median sampling SE of one retention value (binomial, 50 questions) | 0.21 |
| ... so the SE of one cell's Δ is roughly (x sqrt 2, independent noise assumed) | 0.30 |

**What it shows.** That the table contents below rest on a complete, clean file, and how much sampling noise to expect.
**Columns.** *Check* is the quantity and *Value* its value.
**How to read it.** `summary.csv` has 288 rows with no duplicate keys. Of the 96 non-English cells (8 concepts × 3
methods × 4 languages), 93 are reliable; the three dropped are Harry Potter/`ru` for each method. The wrapper effect is
essentially zero on average (−0.014) with a mean absolute swing of 0.089, which is why only the native wrapper is used.
The last two rows are the important warning: one retention value has a sampling standard error of about 0.21, so one
cell's Δ is uncertain by about ±0.30. **Individual cells are not interpretable; only averages over many cells are.**

### Table 2 — Per method, averaged over the 8 concepts

| Method | English retention | Non-English retention | Leak Δ (± SE over concepts) | Share of English erasure that transfers [95% CI] | Concepts with Δ > +0.05 | Similar-domain QA retained (English) | MMLU retained (English) |
|---|---|---|---|---|---|---|---|
| ember | 0.53 | 0.78 | +0.25 ± 0.07 | 0.47 [0.34–0.69] | 6/8 | 0.98 | 1.00 |
| snmf | 0.37 | 0.48 | +0.10 ± 0.04 | 0.84 [0.73–0.93] | 5/8 | 0.65 | 0.97 |
| snmf_ember | 0.30 | 0.51 | +0.22 ± 0.04 | 0.69 [0.59–0.79] | 7/8 | 0.83 | 0.96 |

**What it shows.** The headline comparison of the three methods: how well each erases in English, how well that erasure
carries over to other languages, and what it costs elsewhere.
**Columns.**
- *English retention* and *Non-English retention*: mean retention (lower = more erased) over the concepts; non-English
  uses the reliable languages only.
- *Leak Δ*: their difference (equivalently, the mean of the per-concept mean Δ), with the standard error across concepts.
- *Share of English erasure that transfers*: the ratio defined in §2, with its bootstrap interval. It corrects for the
  fact that a method which erases English harder has more room to "leak".
- *Concepts with Δ > +0.05*: in how many of the 8 concepts the leak is clearly positive.
- *Similar-domain QA retained* and *MMLU retained*: collateral damage (English). Lower = more damage.

**How to read it.** `ember` erases English least (0.53) and keeps the most knowledge elsewhere (0.78): only about 47% of its
English erasure transfers, against 84% for `snmf`. `snmf_ember` is in between (69%). The price of `snmf`'s better
transfer is collateral damage: it retains only 0.65 of similar-domain knowledge, against 0.98 for `ember` and 0.83 for
`snmf_ember`. MMLU barely moves for any method (0.96–1.00).

**Table 2b — Are the method differences larger than noise?** Paired across concepts (each concept is its own control):

| Comparison (difference in mean Δ) | Mean difference | SE | t (7 d.f.) | Concepts where first leaks more |
|---|---|---|---|---|
| ember − snmf | +0.15 | 0.07 | +2.0 | 6/8 |
| snmf_ember − snmf | +0.11 | 0.05 | +2.1 | 6/8 |
| ember − snmf_ember | +0.03 | 0.04 | +0.7 | 5/8 |

*Columns.* *Mean difference* is the average over concepts of (first method's mean Δ − second method's mean Δ); *SE* its
standard error; *t* the ratio; the last column counts concepts where the first method leaks more.
*Reading.* `ember` and `snmf_ember` each leak about 0.11–0.15 more than `snmf` (t ≈ 2.0–2.1, in 6 of 8 concepts). With
7 degrees of freedom the 5% two-sided threshold is |t| ≈ 2.4, so this is **suggestive, not conclusive**. `ember` and
`snmf_ember` cannot be told apart (+0.03 ± 0.04).

### Table 3 — Per concept, overview

| Concept | Pristine EN accuracy | Pristine non-EN accuracy (mean) | Word overlap | Tokens edited by EMBER | EMBER δ | ember: Δ (EN retention) | snmf: Δ (EN retention) | snmf_ember: Δ (EN retention) |
|---|---|---|---|---|---|---|---|---|
| Harry Potter | 0.88 | 0.52 | 0.158 | 39 | 10 | -0.04 (0.59) | -0.03 (0.62) | +0.05 (0.24) |
| World War II | 0.74 | 0.64 | 0.144 | 36 | 5 | +0.01 (0.88) | +0.17 (0.14) | +0.19 (0.47) |
| Halloween | 0.62 | 0.51 | 0.139 | 80 | 10 | +0.32 (0.51) | +0.28 (0.51) | +0.19 (0.41) |
| Uranium | 0.60 | 0.55 | 0.117 | 70 | 100 | +0.41 (0.26) | +0.02 (0.49) | +0.28 (0.00) |
| Golf | 0.80 | 0.61 | 0.177 | 70 | 10 | +0.29 (0.53) | +0.18 (0.16) | +0.15 (0.45) |
| Baseball | 0.64 | 0.55 | 0.154 | 79 | 100 | +0.13 (0.79) | +0.10 (0.38) | +0.19 (0.33) |
| Culture of Greece | 0.68 | 0.59 | 0.055 | 52 | 200 | +0.50 (0.35) | +0.10 (0.35) | +0.42 (0.30) |
| Gambling | 0.82 | 0.68 | 0.108 | 69 | 100 | +0.37 (0.37) | 0.00 (0.33) | +0.28 (0.16) |

**What it shows.** One row per concept: how hard the concept is for the unedited model, the properties of the edit, and the
leak for each method.
**Columns.**
- *Pristine EN accuracy* and *Pristine non-EN accuracy (mean)*: accuracy of the unedited model in English and averaged over
  the four other languages (Harry Potter's includes Russian at 0.30). A big gap means the model knows the concept much
  better in English.
- *Word overlap*: surface similarity between English questions and their translations (§2).
- *Tokens edited by EMBER* and *EMBER δ*: size and strength of the embedding edit.
- *method: Δ (EN retention)*: the mean leak over reliable languages, with the English retention in parentheses so you can
  see how deeply English was erased.

**How to read it.** Always read Δ together with the bracketed English retention. World War II `ember` has Δ ≈ +0.01 only
because English was barely erased (0.88): nothing was erased anywhere, so nothing could leak. Uranium `snmf_ember` erased
English completely (0.00), so any retention elsewhere counts as leak. Harry Potter is the one concept where no method leaks
clearly (Δ between −0.04 and +0.05).
*Note on overlap.* The overlap column is recomputed here from the translated QA files. It matches the values recorded in
`progress.md` §4.5 to within 0.004 (Halloween 0.139 vs 0.135, Baseball 0.154 vs 0.156, Culture of Greece 0.055 vs 0.056;
the other five are identical), because the original script was not kept and its tokenisation had to be reconstructed.

### Table 4 — Per language, averaged over concepts

| Language | ember | snmf | snmf_ember | All methods |
|---|---|---|---|---|
| de | +0.15 (n=8) | +0.10 (n=8) | +0.17 (n=8) | +0.14 |
| es | +0.24 (n=8) | +0.10 (n=8) | +0.14 (n=8) | +0.16 |
| fr | +0.19 (n=8) | +0.06 (n=8) | +0.13 (n=8) | +0.13 |
| ru | +0.47 (n=7) | +0.17 (n=7) | +0.49 (n=7) | +0.38 |

**What it shows.** Whether some languages leak more than others.
**Columns.** One column per method, plus all three pooled. Each entry is the mean Δ over the concepts with a reliable cell
for that language, with the number of concepts in brackets (Russian has 7, because Harry Potter/`ru` is screened out).
**How to read it.** German, Spanish and French leak moderately and similarly (+0.06 to +0.24). **Russian leaks most for the
methods that edit embeddings** (`ember` +0.47, `snmf_ember` +0.49) and much less for `snmf` (+0.17).

**Table 4b — Russian against the other three languages.**

| Method | Russian retention minus mean(de, es, fr) retention | Concepts where Russian retains more |
|---|---|---|
| ember | +0.25 | 7/7 |
| snmf | +0.06 | 6/7 |
| snmf_ember | +0.33 | 7/7 |

*Columns.* The average, over concepts, of (Russian retention − mean of de/es/fr retention) for the same method, and the
number of concepts where Russian retains more.
*Reading.* For `ember` and `snmf_ember`, Russian retains more knowledge than the Latin-script languages in every concept
(7/7), by 0.25 and 0.33. For `snmf` the gap is small (+0.06; 6/7). This is what one expects if the embedding edit is tied
to Latin-script tokens that Russian prompts never contain.

### Table 5 — What predicts how much a concept leaks?

| Method | Word overlap | Tokens edited | log10 EMBER δ | English retention |
|---|---|---|---|---|
| ember | -0.68 | +0.51 | +0.65 | -0.82 |
| snmf | +0.20 | n/a | n/a | -0.37 |
| snmf_ember | -0.91 | +0.15 | +0.76 | -0.31 |

Word overlap vs tokens edited: r = +0.10. For ember, partial r(overlap | tokens) = -0.85 and partial r(tokens | overlap) = +0.79.

**What it shows.** Pearson correlation, across the 8 concepts, between each candidate explanation and the concept's mean Δ
(Table 3), separately per method.
**Columns.**
- *Word overlap*: negative correlation = the more translations look like English, the less the concept leaks.
- *Tokens edited* and *log10 EMBER δ*: the breadth and strength of the embedding edit. "n/a" for `snmf`, which applies no
  embedding edit.
- *English retention*: how deeply English was erased. A negative value means the leak is larger where English was erased
  harder.

**How to read it.** With n = 8 a correlation must exceed about |0.71| to differ from zero at the 5% level, so treat the
**direction** as informative and the magnitudes as soft. The two methods with an embedding edit track word overlap
(−0.68 for `ember`, −0.91 for `snmf_ember`); plain `snmf` does not (+0.20). This fits the mechanism (surface form only
matters when tokens are edited). Two confounds limit that reading: EMBER δ was tuned per concept and also correlates with
the leak (+0.65, +0.76), and for `ember` the leak is larger wherever English was erased harder (−0.82). Overlap and tokens
edited are essentially independent (r = +0.10), and for `ember` both keep a sizeable partial correlation with the leak, so
neither explains the other away.

### Table 6 — One table per concept

Each table has one row per method. **How to read the columns:** *EN retention* is retention on English; each language
column gives the retention in that language followed by its Δ in brackets (Δ = that retention minus EN retention);
*Mean non-EN retention* and *Mean Δ* average over the reliable languages; *Similar-domain retained (EN)* is the collateral
damage measure (lower = more damage). "screened out" means the pristine model's accuracy in that language was below 0.40.
Under each table, the *pristine accuracies* are the denominators of the retention values, the *selected
hyper-parameters* are the ones the English grid chose (SNMF: δ_in / δ_out are the edit strengths on `up_proj` / `down_proj`
and the layer ranges follow the `@`), and the three bullets are generated from the table.

#### 6.1 Harry Potter

| Method | EN retention | de: retention (Δ) | es: retention (Δ) | fr: retention (Δ) | ru: retention (Δ) | Mean non-EN retention | Mean Δ | Similar-domain retained (EN) |
|---|---|---|---|---|---|---|---|---|
| ember | 0.59 | 0.46 (-0.13) | 0.45 (-0.13) | 0.74 (+0.15) | screened out | 0.55 | -0.04 | 1.00 |
| snmf | 0.62 | 0.51 (-0.11) | 0.76 (+0.14) | 0.48 (-0.14) | screened out | 0.58 | -0.03 | 0.87 |
| snmf_ember | 0.24 | 0.35 (+0.11) | 0.15 (-0.09) | 0.35 (+0.12) | screened out | 0.29 | +0.05 | 0.74 |

Pristine accuracy (the denominators): en 0.88 · de 0.62 · es 0.58 · fr 0.56 · ru 0.30 (screened out).

Selected hyper-parameters — ember: δ = 10 (39 tokens edited); snmf: δ_in = 7 @ layers 0–12, δ_out = 1 @ layers 0–8; snmf_ember: δ_in = 10 @ layers 0–8, δ_out = 1 @ layers 0–8, δ_embed = 10.

- Leakiest method: **snmf_ember** (mean Δ +0.05); least leaky: **ember** (-0.04).
- Strongest English erasure: **snmf_ember** (retention 0.24).
- Largest single-language leak: **fr** under **ember** (+0.15).

*Note.* Russian is screened out (pristine 0.30). Per-language Δ values have mixed signs (−0.14 to +0.15) and are within
the ±0.30 noise of a single cell, so there is no consistent leak.

#### 6.2 World War II

| Method | EN retention | de: retention (Δ) | es: retention (Δ) | fr: retention (Δ) | ru: retention (Δ) | Mean non-EN retention | Mean Δ | Similar-domain retained (EN) |
|---|---|---|---|---|---|---|---|---|
| ember | 0.88 | 0.77 (-0.11) | 0.87 (-0.01) | 0.90 (+0.02) | 1.00 (+0.12) | 0.88 | +0.01 | 1.00 |
| snmf | 0.14 | 0.12 (-0.03) | 0.36 (+0.22) | 0.28 (+0.14) | 0.48 (+0.34) | 0.31 | +0.17 | 0.58 |
| snmf_ember | 0.47 | 0.49 (+0.02) | 0.53 (+0.06) | 0.69 (+0.22) | 0.93 (+0.46) | 0.66 | +0.19 | 0.82 |

Pristine accuracy (the denominators): en 0.74 · de 0.68 · es 0.72 · fr 0.64 · ru 0.52.

Selected hyper-parameters — ember: δ = 5 (36 tokens edited); snmf: δ_in = 10 @ layers 0–25, δ_out = 1 @ layers 13–25; snmf_ember: δ_in = 10 @ layers 0–25, δ_out = 1 @ layers 0–8, δ_embed = 5.

- Leakiest method: **snmf_ember** (mean Δ +0.19); least leaky: **ember** (+0.01).
- Strongest English erasure: **snmf** (retention 0.14).
- Largest single-language leak: **ru** under **snmf_ember** (+0.46).

*Note.* `ember` barely erased English (0.88), so its Δ ≈ 0 says nothing about transfer. `snmf` erased English most
(0.14) but at a collateral cost (similar-domain retained 0.58); Russian is where both SNMF variants leak most.

#### 6.3 Halloween

| Method | EN retention | de: retention (Δ) | es: retention (Δ) | fr: retention (Δ) | ru: retention (Δ) | Mean non-EN retention | Mean Δ | Similar-domain retained (EN) |
|---|---|---|---|---|---|---|---|---|
| ember | 0.51 | 0.79 (+0.28) | 0.78 (+0.26) | 0.76 (+0.25) | 1.00 (+0.49) | 0.83 | +0.32 | 1.00 |
| snmf | 0.51 | 0.93 (+0.42) | 0.56 (+0.04) | 0.68 (+0.17) | 1.00 (+0.49) | 0.79 | +0.28 | 0.85 |
| snmf_ember | 0.41 | 0.52 (+0.11) | 0.41 (0.00) | 0.44 (+0.03) | 1.00 (+0.59) | 0.59 | +0.19 | 0.85 |

Pristine accuracy (the denominators): en 0.62 · de 0.54 · es 0.52 · fr 0.50 · ru 0.46.

Selected hyper-parameters — ember: δ = 10 (80 tokens edited); snmf: δ_in = 10 @ layers 0–25, δ_out = 7 @ layers 13–25; snmf_ember: δ_in = 7 @ layers 0–12, δ_out = 10 @ layers 13–25, δ_embed = 10.

- Leakiest method: **ember** (mean Δ +0.32); least leaky: **snmf_ember** (+0.19).
- Strongest English erasure: **snmf_ember** (retention 0.41).
- Largest single-language leak: **ru** under **snmf_ember** (+0.59).

*Note.* Russian retains everything (1.00) under all three methods. In the Latin-script languages `snmf_ember` leaks little
(0.00 to +0.11), whereas `ember` leaks +0.25 to +0.28.

#### 6.4 Uranium

| Method | EN retention | de: retention (Δ) | es: retention (Δ) | fr: retention (Δ) | ru: retention (Δ) | Mean non-EN retention | Mean Δ | Similar-domain retained (EN) |
|---|---|---|---|---|---|---|---|---|
| ember | 0.26 | 0.19 (-0.07) | 0.74 (+0.48) | 0.73 (+0.47) | 1.00 (+0.74) | 0.66 | +0.41 | 0.88 |
| snmf | 0.49 | 0.63 (+0.14) | 0.57 (+0.08) | 0.57 (+0.08) | 0.27 (-0.21) | 0.51 | +0.02 | 0.59 |
| snmf_ember | 0.00 | 0.00 (0.00) | 0.22 (+0.22) | 0.41 (+0.41) | 0.52 (+0.52) | 0.28 | +0.28 | 0.76 |

Pristine accuracy (the denominators): en 0.60 · de 0.52 · es 0.48 · fr 0.62 · ru 0.58.

Selected hyper-parameters — ember: δ = 100 (70 tokens edited); snmf: δ_in = 10 @ layers 0–25, δ_out = 1 @ layers 0–8; snmf_ember: δ_in = 4 @ layers 0–12, δ_out = 10 @ layers 13–25, δ_embed = 100.

- Leakiest method: **ember** (mean Δ +0.41); least leaky: **snmf** (+0.02).
- Strongest English erasure: **snmf_ember** (retention 0.00).
- Largest single-language leak: **ru** under **ember** (+0.74).

*Note.* `snmf_ember` erased English completely (0.00). Under `ember`, German is slightly more erased than English (−0.07)
while Spanish, French and Russian retain +0.47 to +0.74 more. Under `snmf`, Russian is more erased than English (−0.21).

#### 6.5 Golf

| Method | EN retention | de: retention (Δ) | es: retention (Δ) | fr: retention (Δ) | ru: retention (Δ) | Mean non-EN retention | Mean Δ | Similar-domain retained (EN) |
|---|---|---|---|---|---|---|---|---|
| ember | 0.53 | 0.90 (+0.38) | 0.91 (+0.38) | 0.46 (-0.06) | 1.00 (+0.47) | 0.82 | +0.29 | 1.00 |
| snmf | 0.16 | 0.41 (+0.25) | 0.26 (+0.09) | 0.27 (+0.10) | 0.43 (+0.26) | 0.34 | +0.18 | 0.17 |
| snmf_ember | 0.45 | 0.61 (+0.16) | 0.63 (+0.17) | 0.17 (-0.28) | 1.00 (+0.55) | 0.60 | +0.15 | 0.91 |

Pristine accuracy (the denominators): en 0.80 · de 0.66 · es 0.68 · fr 0.66 · ru 0.46.

Selected hyper-parameters — ember: δ = 10 (70 tokens edited); snmf: δ_in = 10 @ layers 0–12, δ_out = 10 @ layers 9–17; snmf_ember: δ_in = 1 @ layers 0–8, δ_out = 1 @ layers 9–17, δ_embed = 10.

- Leakiest method: **ember** (mean Δ +0.29); least leaky: **snmf_ember** (+0.15).
- Strongest English erasure: **snmf** (retention 0.16).
- Largest single-language leak: **ru** under **snmf_ember** (+0.55).

*Note.* `snmf` erases well but damages neighbouring knowledge severely (similar-domain retained 0.17).

#### 6.6 Baseball

| Method | EN retention | de: retention (Δ) | es: retention (Δ) | fr: retention (Δ) | ru: retention (Δ) | Mean non-EN retention | Mean Δ | Similar-domain retained (EN) |
|---|---|---|---|---|---|---|---|---|
| ember | 0.79 | 1.00 (+0.21) | 0.94 (+0.15) | 0.76 (-0.03) | 1.00 (+0.21) | 0.93 | +0.13 | 1.00 |
| snmf | 0.38 | 0.35 (-0.03) | 0.60 (+0.22) | 0.44 (+0.06) | 0.56 (+0.17) | 0.49 | +0.10 | 0.39 |
| snmf_ember | 0.33 | 0.68 (+0.34) | 0.43 (+0.10) | 0.28 (-0.05) | 0.70 (+0.37) | 0.52 | +0.19 | 0.67 |

Pristine accuracy (the denominators): en 0.64 · de 0.56 · es 0.60 · fr 0.50 · ru 0.52.

Selected hyper-parameters — ember: δ = 100 (79 tokens edited); snmf: δ_in = 1 @ layers 0–12, δ_out = 10 @ layers 0–8; snmf_ember: δ_in = 4 @ layers 0–25, δ_out = 7 @ layers 13–25, δ_embed = 100.

- Leakiest method: **snmf_ember** (mean Δ +0.19); least leaky: **snmf** (+0.10).
- Strongest English erasure: **snmf_ember** (retention 0.33).
- Largest single-language leak: **ru** under **snmf_ember** (+0.37).

*Note.* `ember` barely erased English (0.79). `snmf` again shows heavy collateral damage (similar-domain retained 0.39).

#### 6.7 Culture of Greece

| Method | EN retention | de: retention (Δ) | es: retention (Δ) | fr: retention (Δ) | ru: retention (Δ) | Mean non-EN retention | Mean Δ | Similar-domain retained (EN) |
|---|---|---|---|---|---|---|---|---|
| ember | 0.35 | 0.62 (+0.27) | 0.95 (+0.60) | 0.82 (+0.47) | 1.00 (+0.65) | 0.85 | +0.50 | 0.97 |
| snmf | 0.35 | 0.62 (+0.27) | 0.40 (+0.05) | 0.33 (-0.02) | 0.46 (+0.11) | 0.45 | +0.10 | 0.90 |
| snmf_ember | 0.30 | 0.52 (+0.22) | 0.91 (+0.60) | 0.64 (+0.33) | 0.80 (+0.50) | 0.72 | +0.42 | 0.93 |

Pristine accuracy (the denominators): en 0.68 · de 0.46 · es 0.68 · fr 0.58 · ru 0.66.

Selected hyper-parameters — ember: δ = 200 (52 tokens edited); snmf: δ_in = 1 @ layers 0–12, δ_out = 10 @ layers 0–8; snmf_ember: δ_in = 1 @ layers 0–8, δ_out = 1 @ layers 0–8, δ_embed = 200.

- Leakiest method: **ember** (mean Δ +0.50); least leaky: **snmf** (+0.10).
- Strongest English erasure: **snmf_ember** (retention 0.30).
- Largest single-language leak: **ru** under **ember** (+0.65).

*Note.* This concept has the lowest word overlap (Table 3) and the largest leak under `ember` (+0.50) with the strongest
EMBER δ (200). Its English retention on the test split (0.35) is much higher than the grid saw (0.02; Table 7).

#### 6.8 Gambling

| Method | EN retention | de: retention (Δ) | es: retention (Δ) | fr: retention (Δ) | ru: retention (Δ) | Mean non-EN retention | Mean Δ | Similar-domain retained (EN) |
|---|---|---|---|---|---|---|---|---|
| ember | 0.37 | 0.74 (+0.37) | 0.54 (+0.17) | 0.66 (+0.29) | 1.00 (+0.63) | 0.73 | +0.37 | 1.00 |
| snmf | 0.33 | 0.25 (-0.09) | 0.33 (0.00) | 0.40 (+0.07) | 0.35 (+0.02) | 0.33 | 0.00 | 0.86 |
| snmf_ember | 0.16 | 0.55 (+0.39) | 0.18 (+0.02) | 0.40 (+0.25) | 0.61 (+0.46) | 0.44 | +0.28 | 1.00 |

Pristine accuracy (the denominators): en 0.82 · de 0.78 · es 0.64 · fr 0.72 · ru 0.56.

Selected hyper-parameters — ember: δ = 100 (69 tokens edited); snmf: δ_in = 7 @ layers 0–12, δ_out = 4 @ layers 0–8; snmf_ember: δ_in = 1 @ layers 0–25, δ_out = 7 @ layers 0–8, δ_embed = 100.

- Leakiest method: **ember** (mean Δ +0.37); least leaky: **snmf** (0.00).
- Strongest English erasure: **snmf_ember** (retention 0.16).
- Largest single-language leak: **ru** under **ember** (+0.63).

*Note.* `snmf` transfers almost perfectly here (mean Δ 0.00), and `snmf_ember` keeps similar-domain retention at 1.00
while erasing English the most (0.16).

### Table 7 — How stable are the English denominators?

| Concept | English retention, train split (grid) | English retention, test split (evaluation) | Test − train |
|---|---|---|---|
| Harry Potter | 0.47 | 0.59 | +0.11 |
| World War II | 0.60 | 0.88 | +0.28 |
| Halloween | 0.64 | 0.51 | -0.12 |
| Uranium | 0.18 | 0.26 | +0.08 |
| Golf | 0.57 | 0.53 | -0.05 |
| Baseball | 0.68 | 0.79 | +0.11 |
| Culture of Greece | 0.02 | 0.35 | +0.33 |
| Gambling | 0.41 | 0.37 | -0.04 |
| **mean absolute difference** |  |  | 0.14 |

**What it shows.** For `ember`, the English retention of the chosen hyper-parameters on the grid's **train** split (the
data it was selected on) against the **test** split used in every other table.
**Columns.** *Train* and *test* are English retention (lower = more erased); *Test − train* is the gap; the last row is the
mean absolute gap.
**How to read it.** The selected edit looks stronger on the data it was chosen on than on the held-out data (mean absolute
gap 0.14; Culture of Greece 0.02 → 0.35, World War II 0.60 → 0.88). Since Δ is measured against the English test
retention, this adds noise to every Δ.

---

## 4. Caveats

- **Noise.** Each cell rests on 50 questions, so a single Δ has a standard error of about 0.3 (Table 1). The averages in
  Tables 2, 4 and 5 are the trustworthy quantities, and even those have only n = 8 concepts behind them.
- **Δ depends on the depth of English erasure.** It is a difference of ratios, so it is compressed when English is barely
  erased and inflated when English is fully erased. Table 2's "share of English erasure that transfers" is included to
  give a second view, but it inherits the same noise.
- **Only one model and seed.** Everything is Gemma-2-2B, seed 42. Llama-3.1-8B, which the paper also covers, was not run.
- **EMBER δ is not a controlled variable.** The grid picked values from 5 to 200 per concept, and δ correlates with the
  leak (Table 5), so cross-concept differences cannot be attributed to the concept alone.
- **Multiple-choice only.** Open-ended QA was dropped for the non-English languages, because the pristine open-ended
  accuracy is 0.04–0.24 (too low a denominator).
- **Word overlap is a proxy.** It measures question vocabulary, not whether the tokens EMBER actually edited appear in the
  prompt. The direct measure (`coverage_L`, `progress.md` §8.2) has not been computed yet.
- **Retention is clipped.** Many values sit at exactly 1.00 (Russian, mostly) or 0.00; there the true difference is
  hidden by the clip.

---

## 5. Code

The block below is the complete script that generated every table above (standard library only, Python ≥ 3.8; the
project's `.venv` is not needed). Run it **from the repository root**; it prints each table as Markdown, preceded by an
HTML comment with its name, and `--table NAME` prints a single one (`checks`, `method`, `paired`, `concepts`,
`language`, `russian`, `correlations`, `denominator`, `concept_<Name_with_underscores>`).

To extract and run it straight from this document:

```
sed -n '/^```python$/,/^```$/p' docs/crosslingual_results_summary.md | sed '1d;$d' > /tmp/crosslingual_summary.py
python3 /tmp/crosslingual_summary.py
```

How the main quantities are computed, in the order they appear in the code:

| Quantity | Function | Computation |
|---|---|---|
| Retention, EN retention | `Data.lang_ret`, `Data.en_ret` | `qa_frac` of the native-wrapper row; `None` if the row is not reliable |
| Δ for one cell | `Data.delta` | `lang_ret(L) − en_ret` for the same concept and method |
| Per-concept means | `Data.concept_stats` | mean of retention and of Δ over the reliable languages |
| Per-method means, SE | `t_method` | `mean` and `se` (= `stdev / sqrt(n)`) over the 8 concepts |
| Transfer and its CI | `transfer`, `t_method` | `mean(non-EN erasure) / mean(EN erasure)`; bootstrap over concepts, `N_BOOT = 10000`, `Random(0)` |
| Paired comparison | `t_paired` | per-concept difference of mean Δ; `mean / se` |
| Word overlap | `word_overlap` | per question, (number of EN words also found in the translation) ÷ (number of EN words), for words matching `\w{4,}`; then averaged |
| Correlations | `pearson`, `partial`, `t_correlations` | Pearson over concepts; partial via `(rxy − rxz·ryz) / √((1 − rxz²)(1 − ryz²))` |
| Sampling noise | `t_checks` | `sqrt(p(1−p)/n) / (pristine − 0.25)`, median over reliable cells |

```python
#!/usr/bin/env python3
"""Summarise the cross-lingual erasure experiment (Gemma-2-2B, MC QA, 8 concepts).

Run from the repository root with any Python >= 3.8 (standard library only):

    python3 crosslingual_summary.py                 # print every table
    python3 crosslingual_summary.py --table method  # print one table

Inputs
  results/cross_lang_test/summary.csv     one row per (concept, language, config, wrapper)
  results/ember/.../best_embed.csv        ember's selected hyper-parameters (+ train-split metrics)
  results/snmf*/.../top_hps_valid.csv     SNMF's validated top hyper-parameters
  data/qa_lang/mc_questions_<lang>.json   index-aligned translated QA files (for word overlap)
"""
import argparse
import csv
import json
import math
import random
import re
import statistics as st
from pathlib import Path

REPO = Path(".")
SUMMARY = REPO / "results/cross_lang_test/summary.csv"
MODEL_SUBDIR = "google_gemma-2-2b-it/rank100/seed42/train_mc"
QA_DIR = REPO / "data/qa_lang"

CONFIGS = ["ember", "snmf", "snmf_ember"]
LANGS = ["de", "es", "fr", "ru"]
CONCEPTS = ["Harry Potter", "World War II", "Halloween", "Uranium",
            "Golf", "Baseball", "Culture of Greece", "Gambling"]
GRID_DIR = {"ember": "ember", "snmf": "snmf", "snmf_ember": "snmf_ef"}
LEAK_THRESHOLD = 0.05      # a concept "clearly leaks" when its mean delta exceeds this
N_BOOT = 10_000            # bootstrap resamples (over concepts) for the transfer CI


# --------------------------------------------------------------------------- #
# formatting helpers                                                          #
# --------------------------------------------------------------------------- #
def f2(x):
    return f"{x:.2f}"


def sgn(x):
    """Signed 2-decimal number; never prints '-0.00'."""
    r = round(x, 2)
    return "0.00" if r == 0 else f"{r:+.2f}"


def table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(out)


def mean(v):
    v = list(v)
    return sum(v) / len(v)


def se(v):
    v = list(v)
    return st.stdev(v) / math.sqrt(len(v))


def pearson(x, y):
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def partial(x, y, z):
    """Correlation of x and y after removing the linear effect of z."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


# --------------------------------------------------------------------------- #
# loading                                                                     #
# --------------------------------------------------------------------------- #
def load_summary():
    rows = list(csv.DictReader(open(SUMMARY)))
    keys = [(r["concept"], r["language"], r["config"], r["wrapper"], r["qa_kind"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate (concept, language, config, wrapper, qa_kind)"
    return rows, {(r["concept"], r["language"], r["config"], r["wrapper"]): r for r in rows}


def read_csv(path):
    return list(csv.DictReader(open(path)))


def ember_best():
    """concept -> first best_embed.csv row (the file repeats rows; all identical per concept)."""
    out = {}
    for r in read_csv(REPO / "results" / GRID_DIR["ember"] / MODEL_SUBDIR / "best_embed.csv"):
        out.setdefault(r["concept"], r)
    return out


def snmf_best(config, concept):
    """Validated top row by harmonic_alpaca, exactly how crosslingual/best_hp.py picks it."""
    p = REPO / "results" / GRID_DIR[config] / MODEL_SUBDIR / concept.replace(" ", "_") / "top_hps_valid.csv"
    return max(read_csv(p), key=lambda r: float(r["harmonic_alpaca"]))


def word_overlap():
    """concept -> mean word overlap between each English QA_test question and its de/es/fr translation.

    Per question: share of the English question's distinct words (lower-cased, >= 4 characters)
    that also occur in the translated question. Averaged over questions and over de/es/fr.
    """
    toks = lambda s: set(re.findall(r"\w{4,}", s.lower()))
    qa = {l: json.load(open(QA_DIR / f"mc_questions_{l}.json")) for l in ["en", "de", "es", "fr"]}
    out = {}
    for c in CONCEPTS:
        vals = []
        for l in ["de", "es", "fr"]:
            for en_q, tr_q in zip(qa["en"][c]["QA_test"], qa[l][c]["QA_test"]):
                a, b = toks(en_q["q"]), toks(tr_q["q"])
                if a:
                    vals.append(len(a & b) / len(a))
        out[c] = mean(vals)
    return out


# --------------------------------------------------------------------------- #
# core numbers                                                                #
# --------------------------------------------------------------------------- #
class Data:
    def __init__(self):
        self.rows, self.R = load_summary()
        self.best = ember_best()
        self.overlap = word_overlap()

    # one native-wrapper cell
    def cell(self, c, k, L):
        return self.R[(c, L, k, "native")]

    def en_ret(self, c, k):
        return float(self.cell(c, k, "en")["qa_frac"])

    def reliable(self, c, L, k="ember"):
        return self.cell(c, k, L)["reliable"] == "True"

    def lang_ret(self, c, k, L):
        """Retention in language L, or None if that (concept, language) is screened out."""
        r = self.cell(c, k, L)
        return float(r["qa_frac"]) if r["reliable"] == "True" else None

    def delta(self, c, k, L):
        v = self.lang_ret(c, k, L)
        return None if v is None else v - self.en_ret(c, k)

    def concept_stats(self, c, k):
        """English retention, mean non-English retention and mean delta over reliable languages."""
        vals = [self.lang_ret(c, k, L) for L in LANGS if self.lang_ret(c, k, L) is not None]
        en = self.en_ret(c, k)
        return dict(en=en, non=mean(vals), d=mean(vals) - en, n=len(vals))

    def reliable_cells(self):
        return [(c, k, L) for c in CONCEPTS for k in CONFIGS for L in LANGS
                if self.lang_ret(c, k, L) is not None]

    def pristine_acc(self, c, L):
        return float(self.cell(c, "pristine", L)["qa_acc"])

    def simdom_en(self, c, k):
        return float(self.cell(c, k, "en")["simdom_frac"])

    def mmlu(self, c, k):
        return float(self.cell(c, k, "en")["mmlu_frac_en"])


def transfer(D, concepts, k):
    """Mean non-English erasure / mean English erasure (erasure = 1 - retention)."""
    en = mean(1 - D.concept_stats(c, k)["en"] for c in concepts)
    non = mean(1 - D.concept_stats(c, k)["non"] for c in concepts)
    return non / en


# --------------------------------------------------------------------------- #
# tables                                                                      #
# --------------------------------------------------------------------------- #
def t_checks(D):
    cells = [(c, k, L) for c in CONCEPTS for k in CONFIGS for L in LANGS]
    rel = D.reliable_cells()
    dropped = sorted({(c, L) for c in CONCEPTS for L in LANGS if not D.reliable(c, L)})
    wrap = [float(D.R[(c, L, k, "native")]["qa_frac"]) - float(D.R[(c, L, k, "english")]["qa_frac"])
            for (c, k, L) in rel]
    # binomial standard error of one retention value: sqrt(p(1-p)/n) / (pristine - chance)
    noise = []
    for (c, k, L) in rel:
        r = D.cell(c, k, L)
        acc, base, n = float(r["qa_acc"]), float(r["qa_base"]), int(r["n_qa"])
        noise.append(math.sqrt(acc * (1 - acc) / n) / (base - 0.25))
    rows = [
        ("Rows in summary.csv", len(D.rows)),
        ("Non-English cells (8 concepts x 3 methods x 4 languages)", len(cells)),
        ("... of which reliable (pristine MC accuracy >= 0.40)", len(rel)),
        ("Screened-out (concept, language) pairs", ", ".join(f"{c}/{L}" for c, L in dropped) or "none"),
        ("Wrapper effect, native minus English scaffold (mean / mean absolute)",
         f"{mean(wrap):+.3f} / {mean(abs(w) for w in wrap):.3f}"),
        ("Median sampling SE of one retention value (binomial, 50 questions)", f"{st.median(noise):.2f}"),
        ("... so the SE of one cell's Δ is roughly (x sqrt 2, independent noise assumed)",
         f"{st.median(noise) * math.sqrt(2):.2f}"),
    ]
    return table(["Check", "Value"], rows)


def t_method(D):
    body = []
    rng = random.Random(0)
    for k in CONFIGS:
        s = {c: D.concept_stats(c, k) for c in CONCEPTS}
        d = [s[c]["d"] for c in CONCEPTS]
        boots = sorted(transfer(D, [rng.choice(CONCEPTS) for _ in CONCEPTS], k) for _ in range(N_BOOT))
        lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
        body.append([
            k,
            f2(mean(s[c]["en"] for c in CONCEPTS)),
            f2(mean(s[c]["non"] for c in CONCEPTS)),
            f"{sgn(mean(d))} ± {se(d):.2f}",
            f"{transfer(D, CONCEPTS, k):.2f} [{lo:.2f}–{hi:.2f}]",
            f"{sum(x > LEAK_THRESHOLD for x in d)}/{len(d)}",
            f2(mean(D.simdom_en(c, k) for c in CONCEPTS)),
            f2(mean(D.mmlu(c, k) for c in CONCEPTS)),
        ])
    return table(["Method", "English retention", "Non-English retention", "Leak Δ (± SE over concepts)",
                  "Share of English erasure that transfers [95% CI]", f"Concepts with Δ > +{LEAK_THRESHOLD:.2f}",
                  "Similar-domain QA retained (English)", "MMLU retained (English)"], body)


def t_paired(D):
    body = []
    for a, b in [("ember", "snmf"), ("snmf_ember", "snmf"), ("ember", "snmf_ember")]:
        diff = [D.concept_stats(c, a)["d"] - D.concept_stats(c, b)["d"] for c in CONCEPTS]
        body.append([f"{a} − {b}", sgn(mean(diff)), f"{se(diff):.2f}", f"{mean(diff) / se(diff):+.1f}",
                     f"{sum(x > 0 for x in diff)}/{len(diff)}"])
    return table(["Comparison (difference in mean Δ)", "Mean difference", "SE", "t (7 d.f.)",
                  "Concepts where first leaks more"], body)


def t_concepts(D):
    body = []
    for c in CONCEPTS:
        b = D.best[c]
        row = [c, f2(D.pristine_acc(c, "en")), f2(mean(D.pristine_acc(c, L) for L in LANGS)),
               f"{D.overlap[c]:.3f}", b["n_tokens_edited"], f"{float(b['delta_embed']):g}"]
        for k in CONFIGS:
            s = D.concept_stats(c, k)
            row.append(f"{sgn(s['d'])} ({f2(s['en'])})")
        body.append(row)
    return table(["Concept", "Pristine EN accuracy", "Pristine non-EN accuracy (mean)", "Word overlap",
                  "Tokens edited by EMBER", "EMBER δ", "ember: Δ (EN retention)", "snmf: Δ (EN retention)",
                  "snmf_ember: Δ (EN retention)"], body)


def t_language(D):
    rel = D.reliable_cells()
    body = []
    for L in LANGS:
        row = [L]
        for k in CONFIGS:
            v = [D.delta(c, k, L) for (c, kk, LL) in rel if kk == k and LL == L]
            row.append(f"{sgn(mean(v))} (n={len(v)})")
        allv = [D.delta(c, k, L) for (c, k, LL) in rel if LL == L]
        row.append(sgn(mean(allv)))
        body.append(row)
    return table(["Language", "ember", "snmf", "snmf_ember", "All methods"], body)


def t_russian(D):
    body = []
    for k in CONFIGS:
        gaps, higher = [], 0
        for c in CONCEPTS:
            ru = D.lang_ret(c, k, "ru")
            if ru is None:
                continue
            other = mean(D.lang_ret(c, k, L) for L in ["de", "es", "fr"])
            gaps.append(ru - other)
            higher += ru > other
        body.append([k, sgn(mean(gaps)), f"{higher}/{len(gaps)}"])
    return table(["Method", "Russian retention minus mean(de, es, fr) retention",
                  "Concepts where Russian retains more"], body)


def t_correlations(D):
    O = [D.overlap[c] for c in CONCEPTS]
    N = [float(D.best[c]["n_tokens_edited"]) for c in CONCEPTS]
    G = [math.log10(float(D.best[c]["delta_embed"])) for c in CONCEPTS]
    body = []
    for k in CONFIGS:
        y = [D.concept_stats(c, k)["d"] for c in CONCEPTS]
        en = [D.concept_stats(c, k)["en"] for c in CONCEPTS]
        has_embed = k != "snmf"     # plain SNMF applies no embedding edit
        body.append([k, f"{pearson(O, y):+.2f}", f"{pearson(N, y):+.2f}" if has_embed else "n/a",
                     f"{pearson(G, y):+.2f}" if has_embed else "n/a", f"{pearson(en, y):+.2f}"])
    y = [D.concept_stats(c, "ember")["d"] for c in CONCEPTS]
    note = (f"Word overlap vs tokens edited: r = {pearson(O, N):+.2f}. For ember, partial r(overlap | tokens) = "
            f"{partial(O, y, N):+.2f} and partial r(tokens | overlap) = {partial(N, y, O):+.2f}.")
    return table(["Method", "Word overlap", "Tokens edited", "log10 EMBER δ", "English retention"], body) + "\n\n" + note


def t_denominator(D):
    body, diffs = [], []
    for c in CONCEPTS:
        tr, te = float(D.best[c]["qa_frac"]), D.en_ret(c, "ember")
        diffs.append(te - tr)
        body.append([c, f2(tr), f2(te), sgn(te - tr)])
    body.append(["**mean absolute difference**", "", "", f"{mean(abs(x) for x in diffs):.2f}"])
    return table(["Concept", "English retention, train split (grid)", "English retention, test split (evaluation)",
                  "Test − train"], body)


def hp_line(D, c):
    b = D.best[c]
    parts = [f"ember: δ = {float(b['delta_embed']):g} ({b['n_tokens_edited']} tokens edited)"]
    for k in ["snmf", "snmf_ember"]:
        r = snmf_best(k, c)
        d_emb = float(D.best[c]["delta_embed"]) if k == "snmf_ember" else 0.0
        s = (f"δ_in = {float(r['delta_in']):g} @ layers {r['layer_lo_in']}–{r['layer_hi_in']}, "
             f"δ_out = {float(r['delta_out']):g} @ layers {r['layer_lo_out']}–{r['layer_hi_out']}")
        if k == "snmf_ember":
            s += f", δ_embed = {d_emb:g}"
        parts.append(f"{k}: {s}")
    return "; ".join(parts)


def t_concept_detail(D, c):
    body = []
    for k in CONFIGS:
        s = D.concept_stats(c, k)
        row = [k, f2(s["en"])]
        for L in LANGS:
            v = D.lang_ret(c, k, L)
            row.append("screened out" if v is None else f"{f2(v)} ({sgn(v - s['en'])})")
        row += [f2(s["non"]), sgn(s["d"]), f2(D.simdom_en(c, k))]
        body.append(row)
    tbl = table(["Method", "EN retention", "de: retention (Δ)", "es: retention (Δ)", "fr: retention (Δ)",
                 "ru: retention (Δ)", "Mean non-EN retention", "Mean Δ", "Similar-domain retained (EN)"], body)

    pris = " · ".join(
        f"{L} {f2(D.pristine_acc(c, L))}" + ("" if D.reliable(c, L) else " (screened out)") for L in ["en"] + LANGS)
    stats = {k: D.concept_stats(c, k) for k in CONFIGS}
    lo = min(CONFIGS, key=lambda k: stats[k]["d"])
    hi = max(CONFIGS, key=lambda k: stats[k]["d"])
    best_en = min(CONFIGS, key=lambda k: stats[k]["en"])
    worst = max(((D.delta(c, k, L), k, L) for k in CONFIGS for L in LANGS if D.delta(c, k, L) is not None))
    facts = (f"- Leakiest method: **{hi}** (mean Δ {sgn(stats[hi]['d'])}); least leaky: **{lo}** "
             f"({sgn(stats[lo]['d'])}).\n"
             f"- Strongest English erasure: **{best_en}** (retention {f2(stats[best_en]['en'])}).\n"
             f"- Largest single-language leak: **{worst[2]}** under **{worst[1]}** ({sgn(worst[0])}).")
    return (tbl + "\n\n" + f"Pristine accuracy (the denominators): {pris}.\n\n"
            + f"Selected hyper-parameters — {hp_line(D, c)}.\n\n" + facts)


BLOCKS = [("checks", t_checks), ("method", t_method), ("paired", t_paired), ("concepts", t_concepts),
          ("language", t_language), ("russian", t_russian), ("correlations", t_correlations),
          ("denominator", t_denominator)] + \
         [("concept_" + c.replace(" ", "_"), (lambda D, c=c: t_concept_detail(D, c))) for c in CONCEPTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", help="print only this block (see BLOCKS)")
    args = ap.parse_args()
    D = Data()
    for name, fn in BLOCKS:
        if args.table and args.table != name:
            continue
        print(f"<!-- {name} -->\n{fn(D)}\n")


if __name__ == "__main__":
    main()
```
