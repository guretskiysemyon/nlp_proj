# Facts cut from the main text during the 8-page trim (to place in appendices)

Full pre-trim sections: `paper/drafts/sections_before_trim/`.

## App F (multilingual), from §3 "The edit stays in English"
- On Gemma, EMBER-edited models retain 0.19–0.24 more of the concept's knowledge in German, Spanish, French and
  Russian than in English, and 0.31–0.41 more in Russian, whose tokens the English edit never touches.

## App I (examples), from §3 "The concept hides behind rarity"
- EMBER's raw Harry Potter feature (seed 42): first 19 tokens all Harry Potter; further down Nestlé, Lego, Trump among
  unrelated names and taxonomic suffixes; feature norm at the 86th percentile.

## App G (evaluation), from §5.3 "At their own operating points"
- When runs without an edit count as failures, CINDER gains +0.12 H3 (Holm p = 0.10; post hoc) — not significant (audit M7).
- With a stricter collateral gate (0.9), all four CINDER variants beat EMBER significantly (post hoc). (still in main, short)

## App F (multilingual), from §5.4
- Coverage consistent across three fit seeds (A2 9–12 vs CINDER 17–21 of 38).
- Either half of CINDER's selection closes most of the gap, and together they close it fully (counterfactual coverage on the saved fits).
- CINDER's concept features span more scripts (34% use two or more, against none for EMBER's), although much of their non-Latin mass is generic words.
- On Llama, the first layer reads the mean direction 20× a random one, so even a small mean term changes the edit's effect.

## App H.1 (energy), from §6 "Sharpening makes the concept pure and valuable"
- Full fits: CS better than raw in 15 concepts, worse in 3.
- Two observations from Harry Potter do not generalize: the raw concept feature usually already ranks near the top by ρ
  (median 2; Harry Potter's 12th place at seed 42 varies from 1st to 48th across seeds — WP1: ranks 12, 1, 28, 48, 1),
  and no selection-free concept direction rises to the top of the spectrum.

## App H.2 (geometry), from §6 "Neighbours"
- Ancient Rome's neighbour is empire; top 0.5% of pairs semantically related (75–79th pct vs chance for EMBER; 38/38 concepts).
- Projecting out the mean direction alone opens EMBER's NN angle to 74°.
- Harry Potter's raw feature norm 3rd–6th of 100 features (first drew attention).
- Splits rare; parts distinct sub-topics (Rome + empire for Ancient Rome), which GS keeps apart.

## App C (widths), from §6 "Widths follow the data"
- GS widens 26 of the 28 features that are cut short; robust across 45 definitions of the tail; cuts broadest concepts short.
- Width varies by ~4 tokens across seeds when the same feature is found; unrelated to feature index.

## App D (TWJ), from §6
- On 15 concepts added after the lexicon prompt was fixed, the lexicon path alone generalizes poorly, while showing the
  judge each token's source word does generalize.

## App H.4 (robustness), from §6 "Abstract concepts" / "contamination"
- 4 entity concepts as controls; for six abstract concepts no method places a specific feature in the top 10 in any run;
  sharpening does not recover them (+0.04 top-10 presence, n.s.).
- Contamination: adjacent pages (The Lord of the Rings, Magic for HP); sharpened feature mean ρ-rank 1.8–3.3; EMBER's mean
  rank falls from 4.1 to 6.9, as much as with unrelated sentences: loss is dilution, not lost contrast. Neighbouring
  tokens (Tolkien, Gandalf) make up 14–25% of EMBER's concept feature, 5–14% of the sharpened one.

## App B (circularity), from §6 "The gain is not circular"
- Strict label = concept-only and either a lexicon word or strongly over-represented on the page relative to Wikipedia.
- Disjoint targets keep 66–75% of the gain in concept-vs-neutral separation.
- Another concept's page: strict purity 0.57 → 0.12.
- Two weaker signals were circular or did not generalize: the coverage measure used to choose η improves only with the
  page target; ρ-rank of the concept feature does not change on average (page lifts HP, Ancient Rome, Halloween; buries
  Republic of Ireland and Golf under broader co-topics).
- Default (η, γ) is mid-pack; γ = 2 slightly better.

## App H.5 (extrapolation), from §6
- 17 concepts, ~730 unsampled tokens each, blind labels; top-10 precision 0.18–0.27 vs 0.065 random; about a third unrelated;
  re-solving sparse codes recovers only 4% of the concept-specific tokens the sample missed.
- On EMBER's raw dictionary, projection mostly ranks token rarity (Spearman 0.70 with Wikipedia rarity), so a joint
  least-squares solve is needed; in sharpened space the concept feature is nearly orthogonal to the rest (largest |cos|
  0.28 vs 0.64) and the two agree.
- 87% of unsampled page tokens occur once on the page; at one occurrence the in-sample support is no more precise.

## App E (edit space), from §6 "First layer"
- 27 concepts for the 1.31× vs 1.32× comparison.
- The mean term is most of what aligns an edit with the token it changes (cosine 0.50 vs 0.30, but 0.31 vs 0.29 without
  it) — part of what makes uncentred edits strong is generic rather than conceptual.

## App F: place Table tab:ml-2x2 (saved in paper/drafts/tab_ml_2x2.tex); main text §5.4 cites it via \S\ref{app:ml}.
- The raw fit gives 14% of its candidates to single-language features, CINDER's none.
- All other approaches (A1, A2), like EMBER, edit uncentred.
- Hindi/Thai: limit of the concept-only rule and the tokenizer, shared by all approaches, not of the single fit.

## App G: integrity-gated rule details.

## (third pass) also cut from main
- §6: whitened-only finds the concept in 49%; neighbours ~94th percentile relatedness; orthogonality depends on metric
  (mean |cos| 0.08 with μ̂_B removed); widths: doubling budget moves tail by 3 tokens; GS on raw adds mostly neutral tokens;
  strict-label gain shrinks 21 → 7 runs; lexicon path alone generalizes poorly on 15 later concepts; every (η, γ) beats raw on
  15 concepts outside tuning set; contamination intrusion 5–14% vs 14–25%; first layer reads lowest 1% of Σ_B directions at
  3.4×; re-solving EMBER's dictionary on centred embeddings lowers read strength to 1.09×.
- §7: nonce-like edit removes 1.3–1.6× CINDER's.
- Table 4 (examples) dropped row: "In which year was Donald **Trump** first elected President?" δ=1: "[...] in 2016." δ=200: "Donald Trump". → App I.
- §6 contamination: sharpened feature stays in top 10 in 80–93% of runs (App H.4 already has it).
- §7 nonce: nonce-like edit removes 1.3–1.6× CINDER's; the model often can no longer name the word; "within one embedding
  row, the concept and the word's identity seem to share the same few directions" → App G.7 (app:nonce).
