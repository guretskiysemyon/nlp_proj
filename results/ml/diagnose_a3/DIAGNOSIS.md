# Why A3 (ML-CINDER) erases less than A2 (ML-EMBER): diagnosis from saved artifacts

2026-10-19 · CPU only, no new evaluation, no Gemini. Script `experiments/ml/diagnose_a3.py`
(`main` -> edits.json, curves.json; `read` -> read_gain.json; `refitz` -> refit_z.json). Evals still running for
some units (AI, COVID, Gun, Pornography grids are partial); test scores only where done.

## Table (median over edited tokens; edits as in `evaluate.run_edit`, original space, δ = 1)

| Concept | n edited A2/A3 | ‖c‖/‖e‖ A2/A3 | cos(c,e) A2/A3 | layer-0 read gain of ĉ (× random dir) A2/A3/A3-refitZ | shared tok (Jacc.) | cos(c3,c2) shared | ‖c3‖/‖c2‖ shared | best δ A2/A3 | ‖δc‖/‖e‖ at best A2/A3 | test A2/A3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Harry Potter | 101/79 | .104/.064 | .143/.049 | 7.4/1.7/3.1 | 22 (.14) | .40 | .91 | 10/5 | 1.04/0.32 | .67/.24 |
| Baseball | 120/160 | .125/.076 | .187/.075 | 7.6/1.7/7.3 | 63 (.29) | .57 | .88 | 5/200 | 0.62/15.2 | .47/.34 |
| Culture of Greece | 90/108 | .086/.076 | .098/.070 | 2.6/1.6/8.6 | 50 (.34) | .67 | 1.15 | 200/50 | 17.3/3.8 | .28/.22 |
| Gambling | 134/114 | .083/.063 | .122/.062 | 4.7/1.2/5.1 | 46 (.23) | .50 | .96 | 10/100 | 0.83/6.3 | .49/.20 |
| Gun | 99/274 | .124/.065 | .165/.047 | 5.8/1.1/3.3 | 52 (.16) | .50 | 1.08 | – | – | – |
| Pornography | 101/239 | .077/.065 | .114/.051 | 6.8/1.1/5.5 | 53 (.19) | .72 | 1.00 | – | – | – |
| COVID-19 | 95/131 | .066/.063 | .088/.061 | 3.3/1.2/8.1 | 44 (.24) | .88 | 1.18 | – | – | – |
| AI | 149/311 | .124/.061 | .212/.059 | 5.9/1.3/5.4 | 63 (.16) | .72 | .82 | – | – | – |
| Uranium (A3 only) | –/160 | –/.062 | –/.051 | –/1.2/4.6 | | | | –/200 | –/12.3 | –/.15 |

Read gain = median ‖W_in (g ⊙ ĉ)‖ with W_in = [q;k;v;gate;up] of layer 0 and g its input RMSNorm weight, divided by
the same for random unit directions; for scale, token embeddings themselves score 3.1–3.2.
Share of c's energy in the top-100 background PCs: A2 0.17–0.42, A3 0.09–0.29.
Share of edited tokens with concept-set frequency ≤ 1: A2 0.31–0.48, A3 0.32–0.58.

Selection (all A3 selected features): every one is TJ-accepted (COVID f91 is the only EJ-only one), on topic,
higher ρ than A2 (17–41 vs 3–11). Top-norm tokens are the same kind as A2's (Dumbledore, pitcher, casinos, Greek...).

## δ curves (mean efficacy over eligible languages, train)

A3 saturates: it reaches ~0.08–0.10 by δ = 5 and stays at 0.10–0.14 up to δ = 200 (Baseball .095 → .137,
Gambling .081 → .093, HP .089 → .067, Greece .075 → .103, Cannabis .084 → .113, Uranium ≤ .034) while ‖δc‖ grows
40× and passes ‖e‖ by δ ≈ 15. Gates almost never fire (fluency ≥ 0.97, MMLU ≥ 0.92). A2 climbs steeply between
δ = 2 and 10 (Baseball en 0.18 → 0.93, Gambling en → 0.95, Pornography mean .09 → .61 at δ = 2) and then gates.
Per language at the chosen δ the A3 profile is flat and low (test en 0.07–0.29 vs A2 0.36–0.98); A3 is not stronger
in any language, the gap is largest in English, where A2 is strongest.

## Hypotheses, ranked

1. **Edit direction is one the model barely reads (whitening artifact). Confirmed, high confidence.**
   Whitening by the 8-language background makes low-variance embedding directions as large as high-variance ones,
   so the A3 features (fit in x-space) put their mass there; mapped back, ĉ is read by layer 0 at 1.1–1.7× a random
   direction vs 2.6–7.6× for A2 (every concept, no overlap between the ranges). It is also nearly orthogonal to the
   token (cos 0.05–0.075 vs 0.09–0.21), so it subtracts little of e_t. At large δ the embedding becomes ≈ −δc, a
   near-"silent" vector after RMSNorm: the token's meaning is partly removed (hence efficacy ~0.1) but the model
   reconstructs the concept from context, and nothing breaks (hence no gating). The smallest A2 gain (Greece 2.6)
   coincides with the smallest A2–A3 gap (.28 vs .22) and A2's own δ = 200 choice, consistent with the mechanism.
2. **Edit magnitude too small. Rejected.** ‖c‖/‖e‖ at δ = 1 is 0.6–1× A2's, the same order; on shared tokens
   ‖c3‖/‖c2‖ ≈ 0.9–1.2. At A3's chosen δ the edit is 4–15× ‖e‖ vs 0.6–1.0× for A2. It is the direction, not the size.
3. **Edits on rare/low-frequency tokens. Minor contributor, low-medium confidence.** A3 has somewhat more frequency-1
   tokens (median freq 1–2 vs 2) and larger, more diffuse token sets for Gun/AI/Pornography (GS-rn widths), but the
   high-norm tokens are the central concept words and overlap A2's. Would not explain saturation.
4. **Selection (TWJ/EJ) picks wrong features. Rejected.** All selected features are TJ-accepted, on topic.
5. Inverse map bug: already ruled out (OBSERVATIONS §6); refit below gives the same direction to cos 0.87–0.94.

## Suggested fix

Keep A3's support (CS for *what* to edit: ρ, features, GS masks, tokens) but build the edit in the original space:
re-solve Z with Y fixed, Z_o = argmin ‖E_V* − Z Y‖ (ridge, one k×k solve, seconds on CPU), c_t = Z_o[:,F_C] y_t.
Offline (`refitz`): the result stays at cos 0.87–0.94 with the current A3 edit, yet its layer-0 read gain returns
to 2.7–8.6× random (A2 range), and cos(c,e) rises to 0.07–0.14. The ~10% of the direction that the unmap drops is
the part the model reads. Alternative: c = c_x · Sm · W_inv projected onto the top background PCs, or the dual map
(W instead of W⁻¹) — untested.
Cost to test: CPU edit build (minutes, no Gemini, no refit of the factorization) + one δ grid + test per concept
on GPU, same as an A3 evaluate run (≈ 1 GPU-hour per concept with the early-stop grid); 4 concepts with finished
A2/A3 (HP, Baseball, Greece, Gambling) suffice for a first read. It would be a new approach variant (A3z),
needs a PLAN amendment before running. Expectation (medium confidence): A3z curves look like A2's (steep rise,
gating at δ 10–50) and scores near A2; whether CS selection then helps beyond A2 is open.
