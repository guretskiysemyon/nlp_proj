# MC questions for the 20 new concepts (PREREG_NEW.md)

English multiple-choice questions for the 20 concepts of `data/concept_sentences_extra.json`
(`ML_CONCEPTS=new`), in EMBER's format and split sizes (QA_train / QA_test / SimdomQA_train / SimdomQA_test,
50 items each, 4 options), translated into de fr it pt hi es th. They were added to
`results/ml/data/qa/mc_questions_<lang>.json` (all 8 languages) and `results/ml/data/qa_v2/mc_questions_<lang>.json`
(it pt hi th; identical content for the new concepts). The 18 original concepts' entries were not changed
(sha256 of each concept's JSON checked before and after). `data/mc_questions.json` is unchanged.

Code: `experiments/ml/evaldata/qa_new_{common,gen,translate,score,manifest}.py`, `qa_new_tr*.sh|sbatch`.
Caches and per-item records (verifier verdicts, drop reasons): `results/ml/data/qa_new/work/` (not committed).

## Procedure

1. **Open-ended Q/A** (Gemini `gemini-3.5-flash-lite`, the project's judge model, shared usage log):
   concept questions from the concept's English Wikipedia page (same revision as
   `data/concept_sentences_extra.json`, full scraped page), similar-domain questions on 3 adjacent topics
   per concept (table below), with no article. First round 130 QA / 118 Simdom per concept (pilot: 125 / 125).
2. **MC conversion** with EMBER's Figure 9 prompt (20 Q/A per call), 3 distractors per question.
3. **Blind verification** by a second Gemini pass (25 items per call, options in a seeded order, no key): the
   verifier answers each item and flags it as not ok (ambiguous, several or no defensible options, not
   answerable from general knowledge) and as about / not about the concept.
   An item is dropped when the verifier's answer differs from the key, when it is flagged not ok, or when its
   about-flag contradicts its set (QA must be about the concept, Simdom must not).
4. **Deterministic checks**: 4 distinct options (no near-duplicates: equal without spaces, plural forms,
   1-character differences in strings of 10+ characters), the key exactly once among the options, no
   'all/none of the above', no non-Latin letters (one generated option contained Japanese), the answer not
   contained in the question, and no near-duplicate question within the concept across both sets and all
   rounds (content-word Jaccard >= 0.6, or >= 0.4 with the same answer; the earlier item is kept).
   The splits are drawn from this de-duplicated pool, so train and test cannot share a question.
5. **Refill** until each set has >= 100 valid items. QA refills get only the article sentences no existing
   question covers, plus the list of answers already used. A first refill version listed the existing
   questions verbatim, and flash-lite copied them, which is why some near-duplicate counts are high
   (Tourism, French Revolution Simdom).
6. **Split**: `random.Random(42).sample(valid, 100)` per set (valid items in generation order), the first
   50 -> `*_train`, the next 50 -> `*_test`; options stored in a seeded order (evaluation re-permutes them
   language-aligned anyway, `evaldata.io.load_mc`).
7. **Translation**, v2 pipeline (`cleanup_v2.py`) in one pass: NLLB-200 3.3B (beam 4) for every segment, and
   short option lists translated in context; LaBSE scoring. Targets are segments whose best candidate has
   LaBSE < 0.75, lost a number, or (hi, th) has no target-script character, plus every option of an item
   with duplicate translated options. Targets get NLLB 4-best (beam 8), MADLAD-400 3B 4-best (beam 8) and
   MADLAD option lists. Choice: numbers kept > target script > LaBSE; duplicates -> next-best distinct
   candidate, else the English option. Same item order and option order as English, so the correct
   position (letter) is the English one in every language.

## Prompts

Open-ended, concept (`{avoid}` is empty in the first round; refills: see step 5):

```text
You are writing an evaluation set that tests whether a language model knows about the concept "{concept}".
Below is the text of the English Wikipedia article "{title}".

Write {n} open-ended questions about {concept}, each with its correct short answer.
Requirements:
- Every question must be about {concept} itself (its history, people, places, works, terminology, facts); most of them should be based on facts stated in the article.
- Answerable by a knowledgeable person without seeing the article: prefer notable, well-established facts over minor details. No page-specific trivia, no figures that differ between sources, no "according to the article".
- Factual and unambiguous: exactly one correct answer; no opinions, no facts that may have changed recently.
- Self-contained: each question names its context (mentions {concept} or the specific entity), so it is clear on its own.
- The answer must not appear in the question.
- Short answers: a name, term, number, date or short phrase (at most 6 words).
- Diverse: cover the different sections and aspects of the article; never two questions about the same fact.
- Mixed difficulty: about a third easy (widely known), a third medium, a third hard.
{avoid}
Return ONLY a compact JSON list (no indentation): [{{"q": "...", "a": "..."}}, ...]

Article:
{text}
```

Open-ended, similar domain:

```text
You are writing the "similar domain" control set of an evaluation about the concept "{concept}". The control questions test knowledge that a language model must KEEP when its knowledge of {concept} is removed, so they are close in domain but NOT about {concept}.

Write open-ended questions, each with its correct short answer, about these adjacent topics (number of questions per topic in brackets):
{topics}

Requirements:
- No question may be about {concept}, mention {concept}, or need any knowledge of {concept}. Also avoid facts shared with it: {shared}.
- Answerable by a knowledgeable person: prefer notable, well-established facts (general-knowledge quiz level, mixed difficulty).
- Factual and unambiguous: exactly one correct answer; no opinions, no facts that may have changed recently.
- Self-contained: each question names its context (the topic or the specific entity), so it is clear on its own.
- The answer must not appear in the question.
- Short answers: a name, term, number, date or short phrase (at most 6 words).
- Diverse: never two questions about the same fact.
{avoid}
Return ONLY compact JSON (no indentation), one key per topic, exactly as written above: {{"<topic>": [{{"q": "...", "a": "..."}}, ...], ...}}
```

MC conversion (EMBER Figure 9; changes: the output holds only the 3 distractors per question, and the
distinct / wrong / same-type and compact-JSON bullets are added; the pilot concepts French Revolution and
Giraffe were converted with an earlier version that returned the full `q`, `correct_answer`, `options`
objects as in Figure 9):

```text
I am going to provide you {n} open-ended questions. You need to:
- Create a multiple-choice version of each open-ended question.
- The correct answer should be the original answer; you should write 3 additional distractors.
- The distractors should be reasonable and plausible.
- Save each new question under the same key as the open-ended question. E.g., the fifth question of the QA_test set should be saved as the fifth question of the QA_test set in its MC version.
- Each MC question should have the following format: a list of its 3 distractors (its "q" and correct answer "a" are the open-ended ones: do not repeat them).
- your output should be in JSON format, with the same keys as the open-ended input: {{"<concept>": {{"<set name>": [["distractor 1", "distractor 2", "distractor 3"], ...]}}}}.
- The 3 distractors must be clearly wrong to an expert, different from each other and from the correct answer, and of the same type and similar length as the correct answer.
- Write the JSON compactly, without indentation.
The questions:
{questions}
```

Verification:

```text
You are checking a multiple-choice quiz. For every item, answer the question yourself (pick the single best option) and judge the item.
For every item give:
- the item id (a number)
- the letter (A, B, C or D) of the correct option
- ok: 0 if the question is ambiguous or opinion-based, has more than one defensible correct option or none, or cannot be answered from general knowledge; otherwise 1
- about: 1 if answering requires knowledge about {concept} (the question is about {concept} or about a person, event, work, place or fact that belongs to {concept}); otherwise 0
Return ONLY a JSON list with one string "<id>:<letter>:<ok>:<about>" per item, e.g. ["1:A:1:1", "2:C:1:0", ...]

Items:
{items}
```

## Similar-domain topics

Chosen to be adjacent but distinct, and to avoid the other 37 concepts of the 18 + 20 set (e.g. Cricket
does not use baseball, Plutonium does not use uranium, Robot does not use AI). Counts = items in the final
100 SimdomQA items.

| concept | topics (final count) | must avoid |
| --- | --- | --- |
| French Revolution | the Russian Revolution of 1917 (39); the English Civil War and the Glorious Revolution (36); the unification of Italy (Risorgimento) (25) | the 1789-1799 revolution in France, the storming of the Bastille, Louis XVI, Robespierre, Napoleon's rise to power (coup of 18 Brumaire) |
| Tuberculosis | cholera (35); malaria (34); HIV/AIDS (31) | tuberculosis, Mycobacterium, the BCG vaccine, TB co-infection |
| Cricket | tennis (36); rugby union (34); field hockey (30) | cricket, its players, grounds, formats and tournaments |
| Buddhism | Taoism (34); Hinduism (34); Jainism (32) | the Buddha, Buddhist schools, texts and concepts, Buddhist sites; concepts shared with Buddhism such as karma, samsara, nirvana or dharma |
| Plutonium | thorium (36); radium (34); polonium (30) | plutonium, the Manhattan Project, nuclear weapons, uranium |
| Judaism | Islam (36); Christianity (the New Testament and the churches) (33); Zoroastrianism (31) | Judaism, the Hebrew Bible / Old Testament, Abraham, Moses, Jerusalem's Temple, Jewish history and practice |
| Karate | judo (37); taekwondo (32); boxing (31) | karate, Okinawa's martial arts, kata, karate styles and masters |
| Existentialism | utilitarianism (35); British empiricism (Locke, Berkeley, Hume) (34); Stoicism (31) | existentialism and its thinkers (Kierkegaard, Nietzsche, Heidegger, Sartre, de Beauvoir, Camus), phenomenology, absurdism |
| Prime number | Euclidean geometry (polygons, circles, the Pythagorean theorem) (37); pi and irrational numbers (32); the Fibonacci sequence and the golden ratio (31) | prime numbers, primality, factorization into primes, divisibility |
| The Matrix | the original Star Wars trilogy (36); The Terminator films (34); Blade Runner (1982) (30) | The Matrix franchise, the Wachowskis, Keanu Reeves, bullet time |
| Louvre | the British Museum (36); the Museo del Prado (33); the Metropolitan Museum of Art (31) | the Louvre, Paris museums, the Mona Lisa and other works held by the Louvre |
| American Revolutionary War | the American Civil War (36); the Mexican-American War (32); the War of 1812 (32) | the American Revolution / War of Independence (1775-1783), the Declaration of Independence, George Washington's Continental Army |
| Robot | the automobile (35); electricity and household electronics (33); the personal computer (32) | robots, robotics, automation, androids, artificial intelligence, industrial robots |
| Sigmund Freud | B. F. Skinner and behaviorism (35); Jean Piaget and developmental psychology (34); Ivan Pavlov and classical conditioning (31) | Freud, psychoanalysis, Carl Jung, the unconscious, dream analysis |
| Giraffe | the zebra (39); the African elephant (35); the lion (26) | giraffes, the okapi |
| Surfing | scuba diving (36); snowboarding (32); skateboarding (32) | surfing, surfboards, waves and surf breaks, Hawaii's surf culture |
| Flush toilet | the electric light bulb (36); the refrigerator (36); the washing machine (28) | toilets, plumbing, sewage, sanitation, bathrooms |
| Cloud | volcanoes (37); Earth's seasons and climate zones (33); ocean tides and currents (30) | clouds, cloud types, fog, precipitation formation, the water cycle in the atmosphere |
| Doctor of Philosophy | medical school and the MD degree (35); secondary education (high school, the SAT, A-levels) (33); law school and the bar exam (32) | the PhD / doctorates, doctoral theses and dissertations, doctoral supervision, the history of the doctorate |
| Tourism | agriculture (36); the retail industry (34); international trade and shipping (30) | tourism, travel, hotels and hospitality, tourist destinations |

French Revolution: the pilot's first round also used 'the Napoleonic Wars (1803-1815)'. The verifier
counted all 42 such items as French Revolution (membership drop), so the refills used the Italian unification.
Flush toilet: the appliance topics produced many technical, ambiguous questions (125 of 296 Simdom items
flagged not ok); 101 valid items remained.

## Verification and drop rates

generated = open-ended Q/A produced (all rounds); valid = passed every check; drop rate = 1 - valid /
generated. Reasons can overlap ('unverified' = not sent to the verifier because a deterministic check
had already failed).

| concept | set | rounds | generated | valid | drop rate | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| French Revolution | QA | 1 | 126 | 118 | 6.3% | verifier_disagrees 6, verifier_not_ok 1, dup_options 1, unverified 1 |
| French Revolution | Simdom | 6 | 233 | 103 | 55.8% | near_dup_of 81, membership 42, verifier_disagrees 6, answer_in_question 2, unverified 2, verifier_not_ok 1 |
| Tuberculosis | QA | 2 | 136 | 101 | 25.7% | unverified 14, answer_in_question 9, verifier_disagrees 9, verifier_not_ok 6, key_not_once 5, dup_options 5, membership 4, near_dup_of 4 |
| Tuberculosis | Simdom | 1 | 120 | 108 | 10.0% | verifier_disagrees 7, verifier_not_ok 3, near_dup_of 1, answer_in_question 1, unverified 1 |
| Cricket | QA | 2 | 130 | 103 | 20.8% | verifier_not_ok 10, verifier_disagrees 9, near_dup_of 8, membership 3, key_not_once 1, dup_options 1, unverified 1 |
| Cricket | Simdom | 1 | 120 | 107 | 10.8% | verifier_disagrees 9, answer_in_question 3, unverified 3, verifier_not_ok 1 |
| Buddhism | QA | 2 | 134 | 106 | 20.9% | unverified 9, answer_in_question 8, verifier_disagrees 6, verifier_not_ok 5, near_dup_of 5, membership 3, dup_options 1 |
| Buddhism | Simdom | 1 | 120 | 107 | 10.8% | verifier_disagrees 6, verifier_not_ok 4, unverified 2, near_dup_of 1, answer_in_question 1, dup_options 1, membership 1 |
| Plutonium | QA | 3 | 162 | 103 | 36.4% | near_dup_of 22, membership 14, verifier_not_ok 14, verifier_disagrees 13, answer_in_question 3, unverified 3, list_option 1 |
| Plutonium | Simdom | 2 | 136 | 107 | 21.3% | verifier_disagrees 13, verifier_not_ok 12, near_dup_of 6, membership 1 |
| Judaism | QA | 1 | 129 | 115 | 10.9% | verifier_not_ok 4, membership 4, verifier_disagrees 4, key_not_once 1, dup_options 1, unverified 1, near_dup_of 1 |
| Judaism | Simdom | 1 | 120 | 108 | 10.0% | verifier_disagrees 10, verifier_not_ok 1, near_dup_of 1, answer_in_question 1, unverified 1 |
| Karate | QA | 3 | 156 | 101 | 35.3% | near_dup_of 22, membership 14, verifier_disagrees 10, unverified 6, verifier_not_ok 4, non_latin_text 3, dup_options 2, answer_in_question 1, key_not_once 1 |
| Karate | Simdom | 1 | 120 | 101 | 15.8% | verifier_disagrees 6, near_dup_of 5, verifier_not_ok 5, unverified 3, answer_in_question 2, key_not_once 1, dup_options 1, membership 1 |
| Existentialism | QA | 3 | 177 | 101 | 42.9% | verifier_not_ok 25, verifier_disagrees 24, near_dup_of 21, membership 9, unverified 7, answer_in_question 6, key_not_once 1, dup_options 1 |
| Existentialism | Simdom | 1 | 120 | 106 | 11.7% | verifier_disagrees 11, answer_in_question 2, unverified 2, verifier_not_ok 2, garbled_question 1 |
| Prime number | QA | 1 | 139 | 102 | 26.6% | membership 22, verifier_not_ok 11, near_dup_of 5, verifier_disagrees 3, answer_in_question 1, unverified 1 |
| Prime number | Simdom | 1 | 120 | 100 | 16.7% | near_dup_of 11, unverified 7, dup_options 4, answer_in_question 3, verifier_disagrees 2 |
| The Matrix | QA | 4 | 168 | 106 | 36.9% | near_dup_of 26, verifier_disagrees 24, membership 7, unverified 5, verifier_not_ok 4, answer_in_question 4, dup_options 1 |
| The Matrix | Simdom | 2 | 125 | 100 | 20.0% | verifier_disagrees 18, verifier_not_ok 7, answer_in_question 3, unverified 3, membership 1, near_dup_of 1 |
| Louvre | QA | 1 | 129 | 107 | 17.1% | verifier_disagrees 8, unverified 7, answer_in_question 5, verifier_not_ok 5, membership 4, dup_options 2 |
| Louvre | Simdom | 2 | 136 | 106 | 22.1% | verifier_disagrees 12, unverified 9, verifier_not_ok 9, answer_in_question 8, garbled_question 3, dup_options 1 |
| American Revolutionary War | QA | 1 | 130 | 106 | 18.5% | membership 8, verifier_disagrees 7, verifier_not_ok 7, unverified 4, dup_options 2, answer_in_question 2, key_not_once 1, near_dup_of 1 |
| American Revolutionary War | Simdom | 1 | 120 | 111 | 7.5% | verifier_disagrees 4, verifier_not_ok 4, answer_in_question 2, unverified 2 |
| Robot | QA | 2 | 139 | 100 | 28.1% | membership 15, verifier_disagrees 15, answer_in_question 4, unverified 4, near_dup_of 4, verifier_not_ok 3 |
| Robot | Simdom | 1 | 120 | 117 | 2.5% | answer_in_question 3, unverified 3 |
| Sigmund Freud | QA | 1 | 130 | 104 | 20.0% | verifier_disagrees 12, membership 7, near_dup_of 4, verifier_not_ok 3, unverified 3, dup_options 2, answer_in_question 1, key_not_once 1 |
| Sigmund Freud | Simdom | 1 | 120 | 100 | 16.7% | verifier_disagrees 7, verifier_not_ok 6, unverified 6, answer_in_question 5, near_dup_of 2, mc_malformed 1 |
| Giraffe | QA | 1 | 124 | 101 | 18.5% | verifier_disagrees 13, near_dup_of 5, unverified 4, dup_options 2, answer_in_question 2, membership 1 |
| Giraffe | Simdom | 1 | 126 | 110 | 12.7% | verifier_disagrees 12, membership 2, verifier_not_ok 2, none_key 1, non_latin_text 1 |
| Surfing | QA | 2 | 147 | 100 | 32.0% | verifier_disagrees 18, verifier_not_ok 13, unverified 9, near_dup_of 8, membership 7, answer_in_question 6, dup_options 2, key_not_once 1, non_latin_text 1 |
| Surfing | Simdom | 1 | 120 | 101 | 15.8% | verifier_disagrees 14, verifier_not_ok 3, near_dup_of 2, answer_in_question 2, unverified 2 |
| Flush toilet | QA | 5 | 209 | 101 | 51.7% | near_dup_of 43, verifier_disagrees 24, membership 20, verifier_not_ok 18, unverified 14, answer_in_question 8, dup_options 5, key_not_once 2, mc_malformed 1 |
| Flush toilet | Simdom | 5 | 296 | 101 | 65.9% | verifier_not_ok 177, verifier_disagrees 48, unverified 4, answer_in_question 2, near_dup_of 2, dup_options 1, mc_malformed 1 |
| Cloud | QA | 2 | 208 | 101 | 51.4% | unverified 45, near_dup_of 38, key_not_once 20, dup_options 20, mc_malformed 15, answer_in_question 10, verifier_disagrees 10, membership 8, list_option 5, verifier_not_ok 1 |
| Cloud | Simdom | 1 | 120 | 107 | 10.8% | near_dup_of 7, verifier_not_ok 3, verifier_disagrees 2, answer_in_question 1, unverified 1 |
| Doctor of Philosophy | QA | 4 | 222 | 102 | 54.1% | membership 38, near_dup_of 35, verifier_disagrees 28, unverified 14, verifier_not_ok 12, mc_malformed 10, answer_in_question 3, key_not_once 1, dup_options 1 |
| Doctor of Philosophy | Simdom | 1 | 120 | 116 | 3.3% | verifier_disagrees 4 |
| Tourism | QA | 7 | 442 | 104 | 76.5% | near_dup_of 248, verifier_not_ok 42, verifier_disagrees 33, membership 22, unverified 11, answer_in_question 8, key_not_once 3, dup_options 3 |
| Tourism | Simdom | 1 | 120 | 113 | 5.8% | verifier_disagrees 5, answer_in_question 2, unverified 2 |
| **all** | QA | | 3337 | 2082 | 37.6% | |
| **all** | Simdom | | 2732 | 2129 | 22.1% | |

Final: every concept has 50 items in each of QA_train, QA_test, SimdomQA_train and SimdomQA_test (4,000
items). No pair of a concept's 200 final questions has content-word Jaccard >= 0.6.
Correct-answer position in the stored English options: 0: 1073, 1: 1014, 2: 950, 3: 963.

## Translation quality (LaBSE per unique segment)

stage 1 = best NLLB / in-context candidate; final = chosen translation. Targets = segments re-translated with
NLLB / MADLAD n-best. Dup items = items with two identical translated options.

| lang | segments | targets | LaBSE stage 1 median / p10 / <0.75 | LaBSE final median / p10 / mean / <0.75 / <0.5 | chosen engine | dup items stage 1 -> final | English fallback options | numbers lost | no target script |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 16729 | 1471 | 0.916 / 0.777 / 8.2% | 0.926 / 0.828 / 0.922 / 1.5% / 0.1% | nllb 10819, ctx 4076, madlad_nbest 900, madlad_ctx 521, nllb_nbest 413 | 3 -> 0 | 0 | 0 | n/a |
| fr | 16729 | 2358 | 0.913 / 0.714 / 12.8% | 0.934 / 0.828 / 0.925 / 1.1% / 0.0% | nllb 9197, ctx 4184, madlad_nbest 1797, madlad_ctx 1282, nllb_nbest 269 | 6 -> 0 | 0 | 13 | n/a |
| it | 16729 | 1266 | 0.937 / 0.798 / 6.6% | 0.941 / 0.833 / 0.93 / 2.0% / 0.2% | nllb 11359, ctx 3890, madlad_nbest 652, nllb_nbest 436, madlad_ctx 392 | 5 -> 0 | 0 | 45 | n/a |
| pt | 16729 | 1146 | 0.924 / 0.803 / 6.1% | 0.928 / 0.834 / 0.924 / 1.7% / 0.1% | nllb 11214, ctx 4096, madlad_nbest 579, madlad_ctx 459, nllb_nbest 381 | 3 -> 0 | 0 | 14 | n/a |
| hi | 16729 | 3320 | 0.847 / 0.687 / 19.2% | 0.856 / 0.756 / 0.843 / 8.4% / 0.2% | nllb 10273, ctx 2717, nllb_nbest 1592, madlad_nbest 1172, madlad_ctx 975 | 13 -> 0 | 0 | 5 | 1 |
| es | 16729 | 2026 | 0.921 / 0.737 / 10.9% | 0.933 / 0.824 / 0.923 / 1.9% / 0.1% | nllb 8585, ctx 5472, madlad_nbest 1433, madlad_ctx 908, nllb_nbest 331 | 5 -> 0 | 0 | 48 | n/a |
| th | 16729 | 5207 | 0.816 / 0.628 / 29.0% | 0.84 / 0.738 / 0.83 / 11.4% / 0.4% | nllb 9248, nllb_nbest 2646, ctx 2167, madlad_nbest 1929, madlad_ctx 739 | 22 -> 0 | 0 | 2 | 2 |

LaBSE caveat (as in qa_v2/CLEANUP.md): a weak judge for single words and names, blind to register; no
Kiwi and no human check. For the 18 concepts, de / fr / es came from `data/qa_lang`; here they are machine
translated with the same pipeline as it / pt / hi / th.

## Alignment check

`ML_CONCEPTS=all python -m experiments.ml.rescore_v2 --check`: OK, 60,800 items over 8 languages (38 concepts), correct letters equal between qa/ and qa_v2/ everywhere, en/de/fr/es identical; `ML_CONCEPTS=new`: OK, 32,000 items, 0 it/pt/hi/th items differ between qa/ and qa_v2/ (identical content for the new concepts). `evaldata.io.load_mc` (correct letter taken from the English position, checked against each language's `correct_answer`) loads every item of every language. The 18 original concepts' JSON (sha256 per concept) is identical to the snapshot taken before the task (`work/hash_before.json`), and `git diff` of the 12 files is additions only.

## Cost

Gemini (tags `ml_qa_new_*` in `results/ml/gemini_usage.jsonl`, prices 0.25 / 2.50 USD per M
input / output tokens, 3.7 NIS/USD):

| stage | calls | input tok | output tok | NIS |
| --- | --- | --- | --- | --- |
| ml_qa_new_debug | 2 | 1459 | 1884 | 0.019 |
| ml_qa_new_mc | 448 | 406533 | 297968 | 3.132 |
| ml_qa_new_oe | 96 | 913542 | 284668 | 3.478 |
| ml_qa_new_verify | 325 | 570930 | 95150 | 1.408 |
| **total** | 871 | | | **8.04** |

About 2.0 NIS of this went into the pilot before two parser bugs were fixed: the first MC conversions failed to parse (the outermost bracket was misread, and the model sometimes dropped the set key or a closing brace), and the loop meanwhile generated 3 extra open-ended rounds for the 2 pilot concepts (about 1,400 Q/A, discarded without conversion). The 7 NIS checkpoint: the 4-concept measurement projected about 7.1 NIS in total, so the run went ahead with a hard stop near 7.6 NIS for the rest. QA refills on hard concepts (Tourism, Flush toilet, Cloud, Doctor of Philosophy) pushed the final total to about 8.0 NIS, within the 9 NIS task budget.

GPU: about 26 GPU-hours on TITAN Xp: NLLB stage 1 about 18 h (16,729 segments + 3,855 option lists per language at about 2.5-4 segments/s), stage 2 (NLLB/MADLAD 4-best, MADLAD lists) and LaBSE about 8 h. Until 19:10, 2 GPUs (local c-001 GPU 2 + SLURM 1059514); after that, up to 5 (local GPUs 0/2/3 + SLURM ml_qa_*), split by language and stage. Two small top-ups of stage 1 for replaced items ran on CPU (about 15 min).

Wall time: about 12 h (10:44 pilot start to 22:40 write), of which about 11 h was translation.

## Items not fully fixed

- none

## Deviations from how the 18 concepts' questions were built

- Generator: gemini-3.5-flash-lite (the project's judge model, on the shared usage log), not Gemini 3 Flash Thinking as in EMBER. The open-ended prompts are our own: Gur-Arieh et al.'s generation prompts are not in the repository. They follow the same recipe: concept questions from the concept's Wikipedia page, similar-domain questions on adjacent topics.
- MC conversion: EMBER's Figure 9 prompt with the output reduced to the 3 distractors per question (q and the key are the open-ended ones) plus two added bullets (distractors clearly wrong, distinct, same type; compact JSON). This was done to cut output tokens for the budget. The 2 pilot concepts (French Revolution, Giraffe) used the full Figure 9 output format.
- Quality control: instead of EMBER's sampled manual review, an automatic blind verification by a second Gemini pass plus deterministic checks (see Procedure). Items failing either are dropped and replaced, so every final item passed the verifier. That selects items Gemini answers correctly, a possible bias toward easier or more 'Gemini-known' facts; Llama's unedited accuracy is checked by the protocol's eligibility rule (>= 0.40).
- No human check of factual accuracy, of the similar-domain membership, or of the translations.
- de / fr / es are machine translations with the v2 pipeline. For the 18 concepts they came from data/qa_lang (pre-existing checked translations).
- Translation pipeline in one pass: stage-1 NLLB + in-context lists, then the v2 targets with NLLB/MADLAD 4-best and MADLAD lists. The v1 MADLAD beam-4 retranslation step is subsumed by MADLAD 4-best (beam 8). For the 18 concepts, v1 and v2 were built one after the other. Selection and target rules are identical (cleanup_v2.Chooser2, LaBSE < 0.75, numbers, target script, duplicates).
- Similar-domain topics avoid the other 37 concepts (EMBER's Nazism set, for example, includes Ancient Rome facts). French Revolution's Simdom set mixes the first-round Russian Revolution / English Civil War items with Italian unification refills, after the Napoleonic Wars topic was dropped (the verifier counted it as French Revolution).
- Option order stored in the files is a seeded shuffle (correct answer not always first), as in 7 of EMBER's 18 concepts. Evaluation re-permutes options language-aligned anyway (io.load_mc).
- After the split, three checks were added: generator self-corrections in the question ('...Corning—wait, which...'), distractors that came back as stringified lists, and 'None' as the key. The 7 final items they caught (Louvre Simdom x3, Cloud QA x4, Giraffe Simdom x1; see work/en_patch.json) were replaced in place by the first valid unused items of the same concept and set. The seed-42 split is otherwise unchanged.
- 'numbers lost' in the translation table counts correct renderings: Roman-numeral centuries (fr 'XIXe siècle', it 'XIX secolo', es 'siglo XIX'), decades ('anni \'30' for 1930s), and ordinals written as words (hi). It does not count missing facts.
