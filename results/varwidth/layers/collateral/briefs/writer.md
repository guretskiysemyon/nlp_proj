# Brief: writer of collateral-knowledge questions

You write multiple-choice questions that test whether a small language model (Gemma-2-2B-it, a 2-billion-parameter
chat model) still knows ordinary facts that involve particular **tokens** (pieces of text from its vocabulary).
The tokens all come from texts about one **target concept**. Your questions must test the token's uses
**outside** that concept.

## Inputs
- `CONCEPT`: the target concept (given below).
- `POOL`: a JSON file with the token list, `{"concept", "tokens": [{"id", "token"}]}`. A token string is exact. A
  leading space (`" Harry"`) means the token starts a word. No leading space (`"Harry"`, `"otts"`) means it appears at
  the start of a text or inside a word. Pieces like `"ius"` or `"otts"` are word fragments: they occur only inside
  certain words, which you find with the checker (below).
- On a revision round you also get your previous draft and a reviewer's feedback.

## Task
For **every** token in the pool, decide its class:
- `"exclusive"`: the token has essentially no use outside the concept. Examples: a fragment that only occurs inside
  a name from the concept, or a word invented by the concept. Give no items, and a one-line `note` saying why.
- `"multi"`: the token has real uses outside the concept (a common first name, an ordinary noun, another famous
  person, place or brand, a fragment that occurs in many unrelated words). Write **exactly 4 items**.

Rules for each item:
1. **The token must be essential.** The exact token must occur in the question text (preferred) or in the correct
   answer, as the model sees them. The checker verifies this: an item fails if the tokenizer doesn't produce that
   token id. Leading-space tokens usually need the word preceded by a space, e.g. `" Harry"` in "Prince Harry …". A
   fragment like `"otts"` needs a word that the tokenizer actually splits to include it (try words, run the checker).
2. **Outside the concept.** The item must be answerable with **zero knowledge of the concept** and must not be
   about the concept, its creators, works, characters, adaptations, places or history. If knowing the concept would
   help answer it, it is not acceptable.
3. **One unambiguous, stable, well-known answer.** It should be general knowledge that a small chat model is likely
   to know (school-level facts, very famous people, places, works, common word meanings). Avoid trivia, numbers
   that change, recent events (after 2023), opinions and trick questions.
4. **Options.** `options` has 4 distinct strings and `options[0]` equals `correct_answer`. The 3 distractors are
   plausible, of the same type and length, and clearly wrong. No "all/none of the above". The options are shuffled
   later.
5. **Variety.** The 4 items for a token use different contexts (e.g. a person, a place, an idiom, a common-noun
   use, science, geography, sport). Don't write 4 near-paraphrases.
6. Write in English, keep each question to one sentence and under 30 words, and don't put the answer in the
   question.

## Output
Write one JSON file at the path given below, in exactly this format:

```json
{"concept": "<CONCEPT>",
 "tokens": [
  {"token": "<exact pool string>", "class": "multi", "note": "<one line: which outside uses>",
   "items": [{"q": "...", "correct_answer": "...", "options": ["<correct>", "...", "...", "..."], "context": "<2-4 words>"}]},
  {"token": "<exact pool string>", "class": "exclusive", "note": "<why>", "items": []}
 ]}
```

Every pool token appears exactly once.

**Tokenization note (applies to everyone):** the model sees each question as `Question: <q>` and the correct option as `A. <answer>`, so a word at the very start of the question or answer follows a space and yields the leading-space token. Plain tokenizer tests without that prefix are misleading. The checker reproduces the real prompt and is authoritative on whether a token occurs.

## Checker (run it; iterate until it's clean)

```
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.varwidth.collateral tokcheck --draft <your file>
```

It prints a summary (`ok_rate` = share of items passing) and every failing item with its reason. It also lists pool
tokens you haven't covered and strings that aren't pool tokens. Aim for `ok_rate` = 1.0 and no missing tokens. If a
multi token has no item that can contain it after honest tries, keep it `multi` with fewer items and say so in
`note`.

## Example items (format and spirit only; these are for an unrelated concept)
For the token `" Paris"` with a target concept "Eiffel Tower" (outside uses: Paris Hilton, the mythological Paris,
the city in other contexts):

```json
{"q": "In Greek mythology, whom did Paris abduct from Sparta, starting the Trojan War?", "correct_answer": "Helen", "options": ["Helen", "Penelope", "Andromache", "Cassandra"], "context": "mythology"}
{"q": "Paris Hilton is the great-granddaughter of the founder of which hotel chain?", "correct_answer": "Hilton Hotels", "options": ["Hilton Hotels", "Marriott", "Hyatt", "Four Seasons"], "context": "celebrity"}
```

In both, the token is in the question and the answer is a different fact. That's the preferred style.

## Don'ts
- Don't read any other files in the repository except the pool file, your own drafts, the feedback you're given
  and this brief. Don't look for results, other drafts, or information about how the tokens were selected.
- Don't write about the concept itself.
