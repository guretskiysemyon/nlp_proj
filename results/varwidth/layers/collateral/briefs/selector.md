# Brief: selector of the final collateral-question version

Three versions of a multiple-choice question set were written for one target concept: v0, v1 and v2. Each later
version revises the previous one using a reviewer's feedback. The set tests whether a small language model
(Gemma-2-2B-it) still knows ordinary facts involving particular tokens (pieces of its vocabulary) **outside** the
target concept. You choose the **one** version that will be used. You did not write or review any of them, so judge
them impartially. Later is not automatically better: revisions can over-correct or drift.

## Inputs
- The writer's brief, with the rules every item must meet:
  `/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/collateral/briefs/writer.md`
- `CONCEPT`, the `POOL` file, the three versions and the two feedback files (F0 on v0, F1 on v1). Paths are given
  below.
- `STATS`: objective statistics per version, computed by script:
  - `items` / `items_ok`: token check;
  - `base_acc`: the share of items the unedited model answers correctly;
  - `tokens_with_2plus_usable` and `tokens_with_4_usable`: multi tokens with at least 2 / 4 items that pass the token
    check and that the unedited model answers correctly;
  - `exclusive_but_common_in_wiki`: tokens marked exclusive whose frequency ratio suggests they are common outside
    the concept.

  Only items the unedited model answers correctly are used downstream, and at most 4 per token (2 for validation,
  2 for test). Tokens with fewer than 2 usable items are dropped.

## Rubric (in priority order)
1. **Neutrality.** No item gives a fan of the concept an advantage.
2. **The token is essential.** Each item hinges on the token, and the answer isn't given away by other words.
3. **Factual certainty.** Exactly one correct option, stable facts.
4. **Coverage.** Multi-meaning tokens with ≥ 2 usable items (from STATS), and correct token classes. Wrongly calling
   a token exclusive hides collateral damage. Wrongly calling it multi forces bad items.
5. **Answerability by a small model** (base_acc) and natural wording.

Sample at least 40 items per version across tokens, plus all items for tokens whose class or items changed between
versions, and read them against the rubric. Use the feedback files to know what each revision tried to fix, but
verify for yourself.

## Output
Write one JSON file at the path given below:

```json
{"concept": "<CONCEPT>", "chosen": "v0" | "v1" | "v2",
 "scores": {"v0": {"neutrality": 1-5, "essential": 1-5, "certainty": 1-5, "coverage": 1-5, "answerability": 1-5}, "v1": {...}, "v2": {...}},
 "reasons": "<5-10 sentences comparing the versions against the rubric>",
 "residual_problems": ["<item q_start or token: problem>", "..."]}
```

`residual_problems` lists problems you found in the **chosen** version. They will be dropped mechanically, so list
only clear violations (non-neutral, wrong fact, two correct options, token not essential).

Don't read any repository files other than those listed here.
