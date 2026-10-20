# Brief: reviewer of collateral-knowledge questions

A writer produced multiple-choice items that test whether a small language model (Gemma-2-2B-it) still knows
ordinary facts involving particular tokens (pieces of its vocabulary). The tokens come from texts about one target
concept. The items must test the token's uses **outside** that concept. You review the draft and give item-level
feedback. **You do not rewrite the draft.**

## Inputs
- `CONCEPT` and the `POOL` file (token list).
- The writer's brief: `results/varwidth/layers/collateral/briefs/writer.md`. It lists the rules the items must meet.
  Read it first.
- `DRAFT`: the writer's JSON file.
- The checker output for the draft. Run it yourself:
  `cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj && HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.varwidth.collateral tokcheck --draft <DRAFT>`

**Tokenization note (applies to everyone):** the model sees each question as `Question: <q>` and the correct option as `A. <answer>`, so a word at the very start of the question or answer follows a space and yields the leading-space token. Plain tokenizer tests without that prefix are misleading. The checker reproduces the real prompt and is authoritative on whether a token occurs.

## What to check
1. **Token classes.** Is each `exclusive` call right, i.e. does the token really have no common use outside the
   concept? Is each `multi` call right? Flag misclassified tokens in both directions.
2. **Per item:**
   - Is the fact true, and is there exactly one correct option?
   - Is it answerable with zero knowledge of the concept, and not about it? Flag anything a fan of the concept would
     find easier.
   - Is it well-known enough for a small model, and stable (no recent events, no changing numbers)?
   - Does the item hinge on the token? Is the token in the question or answer as written, and would the question
     make sense without it?
   - Are the distractors plausible and of the same type, with no giveaway (length, grammar, answer echoed in the
     question)?
   - Are the 4 items for a token varied, not paraphrases?
3. **Coverage.** Are pool tokens missing? Are there multi tokens with fewer than 4 items?

## Output
Write one JSON file at the path given below:

```json
{"concept": "<CONCEPT>",
 "overall": "<3-6 sentences: main problems and patterns>",
 "tokens": [
  {"token": "<exact string>", "class_ok": true, "class_comment": "",
   "items": [{"q_start": "<first 8 words of the item>", "verdict": "keep" | "fix" | "drop", "comment": "<what is wrong and how to fix; empty for keep>"}],
   "missing_items_comment": ""}
 ]}
```

Be specific and brief. Feedback should help the writer fix things, not restate the rules. Don't read any repository
files other than those listed here.
