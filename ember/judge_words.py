"""What the Stage-1 judge sees for each token of a feature (JUDGE_FORMAT in the notebooks).

    counts = token_word_counts(tokenizer, sentences, langs)          # tid -> Counter{(lang, word): n}
    items  = judge_items(tids, tok_str, counts, fmt, multilingual)  # list of strings, one per token
    desc   = get_description(client, items, sys_prompt=desc_sys_prompt(fmt, multilingual))

Formats:
    "tokens"    bare token strings, as in the paper:               '▁Pot'
    "words"     token -> the words it was cut from, sorted         '▁Pot → Potter, pottery'
                multilingual: grouped by language (LANGS order)    '▁Pot → en: Potter, pottery; de: Potter'
    "top_word"  token -> its single most frequent word (language)  '▁Pot → Potter (en)'
                (the multilingual notebooks' format before JUDGE_FORMAT existed)

Words come from the sentences the tokens were taken from (S_C and S_N of V*), found with the tokenizer's offset
mapping: a token belongs to every whitespace word (pythainlp word for Thai) its character span overlaps. Llama folds
the leading space into the token and SentencePiece models mark it with '▁'; both are skipped before the overlap test.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ember.interpret_features import TOKENS_DESC_SYS

FORMATS = ("tokens", "words", "top_word")
PUNCT = ".,;:!?()[]{}\"'«»“”‘’„–—।॥"
LANG_ORDER = ("en", "es", "fr", "de", "it", "pt", "hi", "th")  # splitting.LANGS order
MAX_WORDS = 8  # words (or (lang, word) pairs) shown per token: the most frequent, then sorted


def word_spans(s: str, lang: str) -> List[Tuple[int, int]]:
    """(start, end) of each word in s: whitespace-delimited, or pythainlp words for Thai."""
    if lang != "th":
        return [m.span() for m in re.finditer(r"\S+", s)]
    from pythainlp.tokenize import word_tokenize as th_word_tokenize
    spans, pos = [], 0
    for w in th_word_tokenize(s, keep_whitespace=True):
        start = s.find(w, pos)
        if start < 0 or not w.strip():
            continue
        spans.append((start, start + len(w)))
        pos = start + len(w)
    return spans


def token_word_counts(tokenizer, sentences: Iterable[str], langs: Optional[Sequence[str]] = None,
                      default_lang: str = "en") -> Dict[int, Counter]:
    """tid -> Counter{(lang, word): occurrences} over `sentences` (langs[i] is the language of sentences[i])."""
    sentences = list(sentences)
    langs = list(langs) if langs is not None else [default_lang] * len(sentences)
    assert len(langs) == len(sentences)
    counts: Dict[int, Counter] = {}
    for s, lang in zip(sentences, langs):
        if not s:
            continue
        enc = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
        spans = word_spans(s, lang)
        for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
            piece = s[a:b]
            a += len(piece) - len(piece.lstrip().lstrip("▁"))  # leading space / '▁' is not part of the word
            if a >= b:
                continue
            if all(unicodedata.category(ch)[0] in "PSZ" for ch in s[a:b]):  # punctuation / symbols: shown as is
                word = s[a:b]
            else:
                words = [s[ws:we] for ws, we in spans if ws < b and we > a]
                word = ("" if lang == "th" else " ").join(words).strip(PUNCT) or s[a:b]
            counts.setdefault(int(tid), Counter())[(lang, word)] += 1
    return counts


def _casefold_key(w: str):
    return (w.casefold(), w)


def format_item(tok: str, counter: Optional[Counter], fmt: str, multilingual: bool,
                max_words: int = MAX_WORDS) -> str:
    """One token as the judge sees it (see the module docstring)."""
    assert fmt in FORMATS, fmt
    if fmt == "tokens" or not counter:
        return tok if fmt == "tokens" else f"{tok} → ?"
    if fmt == "top_word":
        (lang, word), _ = counter.most_common(1)[0]
        return f"{tok} → {word} ({lang})"
    if multilingual:
        pairs = [p for p, _ in counter.most_common(max_words)]
        extra = len(counter) - len(pairs)
        by_lang: Dict[str, set] = {}
        for lang, w in pairs:
            by_lang.setdefault(lang, set()).add(w)
        order = [l for l in LANG_ORDER if l in by_lang] + sorted(l for l in by_lang if l not in LANG_ORDER)
        body = "; ".join(f"{l}: {', '.join(sorted(by_lang[l], key=_casefold_key))}" for l in order)
    else:
        merged = Counter()
        for (_, w), n in counter.items():
            merged[w] += n
        words = [w for w, _ in merged.most_common(max_words)]
        extra = len(merged) - len(words)
        body = ", ".join(sorted(words, key=_casefold_key))
    return f"{tok} → {body}" + (f" (+{extra} more)" if extra > 0 else "")


def judge_items(tids: Sequence[int], tok_str, counts: Dict[int, Counter], fmt: str, multilingual: bool,
                max_words: int = MAX_WORDS) -> List[str]:
    """Judge strings for the token ids `tids` (in order). tok_str: dict or callable tid -> display string."""
    show = tok_str if callable(tok_str) else tok_str.__getitem__
    return [format_item(show(t), counts.get(int(t)), fmt, multilingual, max_words) for t in tids]


# -------------------- Stage-1 system prompts --------------------

ML_LANG_LINE = ("The tokens can come from any of these 8 languages: English, Spanish, French, German, Italian, "
                "Portuguese, Hindi, Thai.")

WORDS_DESC_SYS = """
You get tokens that represent a single feature vector (with some noise).
Each token is shown as: token → the words it was cut from, in alphabetical order ("+N more" means more words were left out).
Some tokens are word pieces rather than whole words; the words show what they stand for.
Infer the single most specific, cohesive concept shared by the relevant tokens.
Return only one concise sentence. No preface, no list, no caveats.

Example:
Tokens: ['▁tomorrow → tomorrow', '▁tonight → tonight', 'yester → yesterday, yesteryear', '▁today → today, todays']
Explanation of feature behavior: this vector is related to specific dates and times (e.g., today/tomorrow/yesterday).
"""

ML_WORDS_DESC_SYS = f"""
You get tokens that represent a single feature vector (with some noise).
Each token is shown as: token → the words it was cut from, grouped by language code and in alphabetical order
("+N more" means more words were left out).
{ML_LANG_LINE}
Some tokens are word pieces rather than whole words; the words show what they stand for.
Infer the single most specific, cohesive concept shared by the relevant tokens, across languages. Phrase it in English.
Return only one concise sentence. No preface, no list, no caveats.

Example:
Tokens: ['▁tomorrow → en: tomorrow', '▁dem → fr: demain', '▁mañana → es: mañana; pt: manhã', '▁heute → de: heute']
Explanation of feature behavior: this vector is related to specific days relative to now (e.g., today/tomorrow).
"""

ML_TOP_WORD_DESC_SYS = f"""
You get tokens that represent a single feature vector (with some noise).
Each token is shown as: token → the word it came from (language code).
{ML_LANG_LINE}
Some tokens are word pieces rather than whole words.
Infer the single most specific, cohesive concept shared by the relevant tokens, across languages. Phrase it in English.
Return only one concise sentence. No preface, no list, no caveats.

Example:
Tokens: ['▁tomorrow → tomorrow (en)', '▁demain → demain (fr)', '▁mañana → mañana (es)', '▁heute → heute (de)']
Explanation of feature behavior: this vector is related to specific days relative to now (e.g., today/tomorrow).
"""


def desc_sys_prompt(fmt: str, multilingual: bool) -> str:
    """Stage-1 system prompt for a JUDGE_FORMAT."""
    assert fmt in FORMATS, fmt
    if fmt == "tokens":
        return TOKENS_DESC_SYS
    if fmt == "top_word":
        return ML_TOP_WORD_DESC_SYS if multilingual else TOKENS_DESC_SYS.replace(
            "You get tokens", "You get tokens (each shown as: token → the word it came from (language code))")
    return ML_WORDS_DESC_SYS if multilingual else WORDS_DESC_SYS


def judge_cache_name(fmt: str, legacy_fmt: str, base: str = "judge") -> str:
    """judge.json for the notebook's original format (old caches stay valid), judge_<fmt>.json otherwise."""
    return f"{base}.json" if fmt == legacy_fmt else f"{base}_{fmt}.json"
