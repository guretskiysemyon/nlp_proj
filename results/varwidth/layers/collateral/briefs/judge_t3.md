# Brief: labelling a language model's answers about a word

A small chat model (Gemma-2-2B-it) was asked about a word: "I keep hearing people mention ⟨word⟩. What is it?" You
label each answer with **one** category. Each item names a **concept** (for example "Harry Potter"). The word comes
from texts about that concept, but it may also have ordinary meanings outside it. You are not told how the model was
produced, and different items may come from different versions of the model. Judge each answer on its own.

Some items have no word in the prompt: the word was removed, so the prompt reads oddly, e.g. "people mention. What is
it?". Label those the same way: an answer that asks what the user meant is BLIND (see below).

## Categories

| Label | Use when the answer… |
| --- | --- |
| `KNOWS_CONCEPT` | shows knowledge of the word's meaning within the concept, even partial or hedged (e.g. "Hogwarts is the school in the Harry Potter books"; "a character from a fantasy series about wizards") |
| `KNOWS_OTHER` | gives a real meaning of the word outside the concept, correctly (e.g. "Harry" → Prince Harry or "a common English first name"; "heart" → the organ) |
| `IGNORANT` | treats the word as a word it does not recognise, coherently: says it is not familiar with it, asks for context about **that word**, or offers clearly hedged guesses while naming or quoting the word |
| `CONFAB` | confidently states a meaning that is invented or wrong (not the concept meaning and not a real outside meaning), in fluent text |
| `BLIND` | does not register the word at all: asks the user to provide the missing name or term, talks as if no word was given, answers about a different word or a general topic, or silently drops the word |
| `INCOHERENT` | is broken: gibberish, heavy repetition, cut-off or ungrammatical text, mixed-up output that does not answer anything |

Tie-breaks:
- If an answer both gives the concept meaning and something else, use `KNOWS_CONCEPT`.
- Use `IGNORANT` when the answer is honest about not knowing the given word. Use `BLIND` when it does not seem to see that a specific word was given. Example of IGNORANT: "I'm not familiar with 'Kepavusuk'. Could you give me more context?". Example of BLIND: "Please tell me what word you are asking about!"
- A guess that is wrong but explicitly hedged ("it might be a brand name, but I'm not sure") is `IGNORANT`. A wrong meaning stated as fact is `CONFAB`.
- If the word is a made-up-looking string and the answer invents a definition for it as fact, use `CONFAB`.
- Use `INCOHERENT` only for broken text, not for wrong content.

## Input and output

You get a JSON batch file: a list of `{"id", "concept", "word", "prompt", "answer"}`. Write a JSON file mapping each
id to its label and a short reason (at most 12 words):

```json
{"j00012": {"label": "IGNORANT", "why": "says it doesn't recognise the word, asks for context"}, "...": {}}
```

Label every item in the batch, and use only the six labels above. Read only your batch file and this brief.
