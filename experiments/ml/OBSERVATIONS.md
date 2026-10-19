# Multilingual A1 / A2 / A3: observations and protocol notes

Running notes for presenting the multilingual comparison (Llama-3.1-8B-Instruct, NF4, 8 languages). Numbers are
validation-split (QA_train, MMLU val) unless stated; they are partial while the δ grid runs (started 2026-10-19).
Plan and fixed hyperparameters: [PLAN.md](PLAN.md).

## Protocol notes (deviations from the EMBER release)

The paper only says δ is chosen per concept by the harmonic score on validation (§C.2–C.3). The 0.9 gates come
from the authors' released code (`ember/erasure/methods/ember.py`, `grid_eval_kwargs`: `min_mmlu=0.90`,
`min_alpaca=0.9`). Ours differ in three places:

| | EMBER release | Here | Reason |
|---|---|---|---|
| Coherence gate | Alpaca instruction + fluency fractions ≥ 0.9 (LLM-judged) | perplexity ratio on held-out native text ≥ 0.9 | no Alpaca (time); judge-free; covers fluency, not instruction following |
| Gates | one language | every one of the 8 languages | per-language version of the same rule |
| δ sweep | all 8 δ, gated cells score 0 | stops at the first gated δ | time; assumes damage grows with δ (the paper's Fig. 11 shows monotone decline). A larger δ that would pass again is not seen |
| Eligibility | — | languages with unedited QA_train < 0.40 leave the score, keep their gates | near chance the normalized efficacy is noise (Part 1 cutoff) |

Hardware: the transformer runs in NF4 4-bit (Titan Xp cannot run fp16 Llama correctly); edited embeddings stay
fp16. The same setup is used for all approaches and baselines; absolute numbers need one bf16/fp16 confirmation.

## 1. Hindi (and mostly Thai) cannot be edited by construction

Hindi efficacy is 0.00 in every unit so far, in all three approaches. The edit never reaches the Hindi questions:

- Share of QA_train questions (question + options) containing at least one edited token, e.g. Baseball A3:
  en 0.94, de 0.86, fr 0.76, it 0.78, pt 0.42, es 0.32, th 0.22, **hi 0.00**. Over all units with edits, Hindi is
  0.00–0.10 except COVID-19 (0.32, Latin-script "COVID" inside Hindi text).
- Cause: EMBER edits only concept-only tokens (code 1: in concept sentences, not in neutral ones). Llama splits
  Hindi and Thai into short pieces shared with everyday text. For Baseball (A2 token set), of the token types in
  the questions: en 84% in V*, 32% concept-only; hi 99% in V*, **5%** concept-only; th 98% / 6%; de 92% / 13%.
  Gun: en 32%, de 11%, hi 4%, th 5% concept-only.
- So the rule that protects neutral text also excludes almost every Hindi token. This holds for A1, A2 and A3
  alike; it is a limit of embedding-only erasure with this tokenizer, not of the multilingual pipeline.
- Possible direction (not run): allow tokens with a high concept/neutral frequency ratio instead of
  concept-only, for languages whose tokenizer fertility is high.

## 2. Edits on shared subword tokens damage other languages

Golf A1 edits 21 tokens, all from the one feature A1 found (in Thai). At δ = 10 French MMLU fell to 0.52 of
baseline (all other languages ≥ 0.93), so δ = 5 was chosen. The Thai feature's tokens also cover 76% of the English
and 62% of the French questions: per-language fits still contain Latin-script pieces, and an edit on a shared piece
acts in every language that uses it.
Possible direction (not run): a shared-token guard that scales the edit by the token's frequency in other
languages' neutral text.

## 3. Strong erasure takes related knowledge with it

Baseball A2 at δ = 5: English efficacy 0.93, Italian 0.51, German 0.44, but English similar-domain accuracy falls
to 0.66 of baseline (other sports). The score includes this cost; it is the main specificity price so far.

## 4. Concepts without features

Selected features per concept (A1: languages with ≥ 1 feature, of 8; A2/A3: features):

| Concept | A1 | A2 | A3 |
|---|---|---|---|
| Halloween, Heroin, Valentine's Day | 0 | 0 | 0 |
| Golf, Republic of Ireland, World War II | 1 | 0 | 0 |
| Ancient Rome | 2 | 0 | 0 |
| Uranium | 2 | 0 | 2 |
| Cannabis | 4 | 0 | 4 |
| Nazism | 0 | 0 | 1 |
| Culture of Greece | 2 | 1 | 1 |
| Baseball | 5 | 1 | 2 |
| Harry Potter | 5 | 1 | 1 |
| COVID-19 pandemic | 6 | 1 | 2 |
| Gambling | 6 | 2 | 2 |
| Pornography | 7 | 1 | 3 |
| Artificial intelligence | 5 | 1 | 3 |
| Gun | 7 | 1 | 5 |

Three concepts have no edit in any approach and are reported as "no edit" (score 0), not tuned to rescue them.
A1 edits 15 concepts, A2 8, A3 11: A1 gets more concepts because any one of 8 per-language fits suffices, but
each A1 feature usually covers few languages.

## 5. Selection depends on the sample

Ancient Rome and Valentine's Day had A3 features at N = 40 that disappeared at N = 30. Seed study
([results/ml/stability/STABILITY.md](../../results/ml/stability/STABILITY.md); 5 concepts, the reference sample
(data seed 0) + 3 new N = 30 samples, fit + selection only, 0.26 NIS):

- Strong concepts (Baseball, Gun): every sample selects on-topic features, but which features and which tokens
  they edit varies (pairwise Jaccard of edited token sets 0.10–0.47).
- Weak concepts: one feature near ρ = 2 that comes and goes. Selection is empty on at least one sample for
  Culture of Greece (A2, A3), Harry Potter (A2) and Ancient Rome (A2, A3; A2 finds a Rome feature on 1 of 4).
  The N = 40 → 30 disappearance fits this pattern; it is not an effect of N.
- Part of it is the fit, not the sample: refitting the identical sample on CPU instead of GPU gives Jaccard
  0.26 (Gun A3) and 0.07 (Harry Potter A3) — near-tied features in the nonconvex fit.
- Reconstruction error barely moves (spread ≤ 0.0015).

Consequence: a single-concept A2/A3 result, especially an empty or one-feature selection, is one draw. The
comparison should be read over concepts, and ideally over several samples.

Fit seed vs sample (extension in the same file, 0.49 NIS): at a fixed sample, changing only the ALS fit seed moves
the A2/A3 edited-token set as much as redrawing the sample (mean pairwise Jaccard 0.22 vs 0.20, CPU vs CPU). A1,
which has no sample, has fit-seed variance of the same size (merged Jaccard vs the reference 0.0–0.38; Ancient Rome
loses both features, Culture of Greece picks off-topic French features). With fit seed 1, Ancient Rome gets an
on-topic feature in A2 and A3. GPU vs CPU barely moves A1 (Jaccard 0.81–1.0) but acts like a small seed change for
A2/A3. So the instability is EMBER's (the nonconvex sparse fit), shared by all three approaches; it is not
introduced by the sampling step. Seed-robust selection (union or vote over fit seeds) is a natural fix, not run.

## 6. A3 inverse map: verified

The CS transform is x = (e − μ_B) W S_η^{γ/2}; edits are mapped back with c = c_x S_η^{−γ/2} W^{−1}
(`unmap = Sm @ W_inv`, `evaluate.run_edit`). Checked numerically on the Baseball and Culture of Greece A3 fits:
round trip x·unmap + μ_B vs e has relative error 3.0e-8, and ‖unmap·M − I‖ / √d = 3.3e-8 (M = W·Sp). The mean
μ_B correctly drops out of the edit (c is a difference). The median edit norm is 7% of the embedding norm at δ = 1.

Reconstruction error, for scale (1%-sparse fits explain little variance by design): EMBER per language (A1 en)
0.92; A2 sample 0.93 → full 0.96 after extrapolation; A3 0.95 → 0.98 (in the whitened space, not comparable to A1/A2
directly). Extrapolating Y with Z fixed costs about 3 points of relative error.
