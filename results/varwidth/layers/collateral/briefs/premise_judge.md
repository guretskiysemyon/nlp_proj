# Brief: is the candidate answer still consistent with the reference?

Each item has a **prompt**, a **reference** answer (from the original model) and a **candidate** answer (from a
modified model). Both were cut off at about 80 tokens, so ignore unfinished last sentences.

Label each item with one of:
- **OK**: the candidate answers the same question with content consistent with the reference. Different wording,
  order, extra or missing detail, or a small slip in one minor fact is OK.
- **BROKEN**: any of
  - *wrong sense*: the key word of the prompt is read as something else (asked about Prince Harry, the answer is about
    the singer Prince; asked about a golf club as a nightclub, the answer is about something unrelated);
  - *wrong answer*: the main fact asked for contradicts the reference (a different person, place, number, or thing);
  - *lost word*: the answer does not register the key word (asks what is meant, treats it as missing or garbled, or
    answers a different question);
  - *incoherent*: broken text, loops, or nonsense.

Judge only consistency with the reference, not whether the reference is right. If the reference itself is wrong or
evasive and the candidate says the same, it is OK.

**Output.** JSON: `{"<id>": {"label": "OK" | "BROKEN", "why": "<at most 12 words>"}, …}` covering every id in your
batch. Read every item yourself; scripts only to read and write files.
