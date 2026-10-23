# Brief: neutral prompts for a generation check

You write short prompts to test whether a language model still handles ordinary words correctly in free text.

**Input.** A concept (for example "Golf") and an alphabetical list of tokens from the model's vocabulary. A leading
space in a token (" club") marks a word start; tokens without one may be word pieces or capitalized starts.

**Task.** Write **exactly 20 prompts** for the concept. Each prompt:
- uses one token from the list, verbatim as a whole word (or the word the piece clearly belongs to), in a meaning
  **unrelated to the concept** (for Golf: " club" as a nightclub or a book club, " green" as the colour, " Woods" as
  a forest or another person named Woods). Knowing nothing about the concept must not help;
- is a short, natural question or instruction with a clear, well-known, stable answer that a small model normally
  gets right in one or two sentences (who / what / where / which, or "name two …");
- avoids the concept itself and anything adjacent to it.

**Choice of tokens.** Use 20 different tokens. Prefer common words with an obvious everyday meaning outside the
concept. Skip tokens that only exist inside the concept (names, jargon) and pieces you cannot build a clean word from.

**Examples (another concept, Harry Potter):**
- "Who is Prince Harry, and who is his wife?" (token " Harry")
- "Who created the detective Sherlock Holmes, and who is Holmes's assistant?" (token " Sherlock")
- "Which US president was Harry Truman, and what major decision is he known for in 1945?" (token " Truman")

**Output.** JSON: `{"<concept>": [{"token": "<token from the list>", "prompt": "<prompt>"}, … 20 items]}`.
Check that every token is from the list and appears as a word in its prompt.
