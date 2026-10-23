# Brief: concept-knowledge questions that avoid a list of words

You write multiple-choice questions that test whether a language model knows facts about a concept, **without using
certain words**. The aim is to ask about the concept indirectly: by its facts, people, places, events, dates,
numbers and descriptions, never by the forbidden words.

**Input.** A concept, a list `avoid` of forbidden words (and word pieces), and a few `examples` of ordinary
questions about the concept, to show the style and difficulty only.

**Task.** Write **60 questions** about the concept. Each has 4 options, exactly one correct.
- **No forbidden word anywhere**: not in the question, not in any option. Case does not matter. Entries of 4+
  characters also rule out words that contain them ("pitch" rules out "pitcher"). Entries of 1–3 characters ("ho",
  "can", "her") rule out only that exact word, not longer words. When in doubt, rephrase. A tokenizer check afterwards
  drops any item that still contains a forbidden token, which is why you write 60.
- The question must need knowledge **of the concept** to answer: someone who knows nothing about the concept should
  be at chance. Do not make the answer guessable from the wording.
- Clear, well-known, stable facts that a small model that knows the concept answers correctly. Not trivia.
- Distractors plausible and of the same type as the answer; none of them also correct.
- Vary the facts: history, people, rules or mechanics, places, numbers, culture, consequences.
- Do not copy or paraphrase the examples.

**Example (concept Golf, forbidden words include golf, club, hole, green, tee, putt, course, ball):**
`{"q": "Which Scottish town, home to the Royal and Ancient, is often called the birthplace of the sport played at the Masters in Augusta?", "options": ["St Andrews", "Aberdeen", "Inverness", "Dundee"], "correct_answer": "St Andrews"}`
(This works only if none of those words is in your list; always check against your own list.)

**Output.** JSON: `{"<concept>": [{"q": …, "options": [4 strings], "correct_answer": <one of the options>}, … 60]}`.
Before writing, check by script the rule above (case-insensitive: substring for 4+ character entries, whole word for shorter ones)
in every question and option, and that each correct_answer is one of its options; rewrite any item that fails.
