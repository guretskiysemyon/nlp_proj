"""J3 overlap judge: does a feature's vocabulary overlap the words one expects for the concept?

    cw  = concept_words("Heroin", n=10)                 # n words, one Gemini call per (concept, n), cached on disk
    fw  = feature_words(tids, counts, tok_str=...)       # 10 words from the feature row, no Gemini
    res = match(cw, fw)                                  # {"pairs", "n_pairs", "exact", "accept", "llm", ...}

concept_words: Gemini names n words specific to the concept (not to its broader category), cached in
results/varwidth/overlap/concept_words.json under "<concept>|<n>" (file-locked: several processes share it).
J3 is run for n in N_SIZES; 10 is the primary spec, 20 / 30 are sensitivity.
feature_words: for each token of the row in |y| order, its most frequent source word (counts from
ember.judge_words.token_word_counts, merged over languages; the token string if none), normalized, deduped.
match: by default the exact (normalized-equality) pairs only. llm=True adds one Gemini call returning
[concept_word, feature_word] pairs that are the same word, an inflection or a translation; validate_pairs keeps only
pairs of listed words, one-to-one, exact pairs first (so n_pairs >= exact). accept = n_pairs >= MIN_PAIRS.
"""
from __future__ import annotations

import ast
import fcntl
import json
import re
import unicodedata
from collections import Counter

from experiments.varwidth.common import RESULTS
from experiments.varwidth.select import get_client, save_json

OVERLAP_DIR = RESULTS / "overlap"
CW_PATH = OVERLAP_DIR / "concept_words.json"
N_WORDS = 10
N_SIZES = (10, 20, 30)
MIN_PAIRS = 4
CW_VERSION = 2      # same for CONCEPT_WORDS_PROMPT
MATCH_VERSION = 2   # bump when MATCH_PROMPT changes: invalidates cached match results

CONCEPT_WORDS_PROMPT = """Given the concept '{concept}', name the {n} words you would most expect to encounter in its \
Wikipedia page or in an article discussing it.
Choose words specific to this concept that would NOT typically appear for its broader category \
(e.g. for 'World War II' not 'war').
Include the concept's own name if it is a distinctive single word (e.g. 'uranium' for 'Uranium', 'valentine' for \
"Valentine's Day").
Single words, lowercase. Return ONLY a JSON list of {n} strings."""

MATCH_PROMPT = """You get two word lists.
List A (words expected for a concept): {a}
List B (words of a feature): {b}
Find the pairs [a, b] (a from list A, b from list B) where b is the SAME word as a, an inflection of a \
(plural, verb tense, possessive, ...; same lemma) or a direct translation of a into another language.
Everything else does NOT count, even if closely related: synonyms, broader or narrower terms, words sharing a root, \
prefix or substring, etymological relatives, derived words of a different lemma, compounds containing the word.
Not pairs: 'war'/'battle', 'opioid'/'opiate', 'heroin'/'hero', 'overdose'/'dose', 'addiction'/'dependence', \
'morphine'/'morphinan'. Pairs: 'drug'/'drugs', 'inject'/'injected', 'drug'/'droga'.
Use each word at most once. Copy the words exactly as written in the lists. Most lists have few or no pairs.
Return ONLY a JSON list of pairs, e.g. [["drug", "drugs"]], or [] if there are none."""


# ---------------------------------------------------------------- normalization
def _strip_punct(w: str) -> str:
    is_p = lambda ch: unicodedata.category(ch)[0] in "PSZC"  # noqa: E731
    i, j = 0, len(w)
    while i < j and is_p(w[i]):
        i += 1
    while j > i and is_p(w[j - 1]):
        j -= 1
    return w[i:j]


def singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):  # keep virus, cannabis, analysis
        return w[:-1]
    return w


def normalize(w: str) -> str:
    w = _strip_punct(str(w).casefold().lstrip("▁ ").replace("’", "'"))
    if w.endswith("'s"):
        w = w[:-2]
    return singular(_strip_punct(w))


def _is_word(w: str) -> bool:
    return len(w) >= 2 and any(ch.isalpha() for ch in w)   # pure numbers out: digit tokens map to random years


def _piece(word: str, tok: str | None) -> str:
    """'1961)—require' -> 'require': split whitespace-word joins at dashes / brackets, keep the piece holding the
    token (else the longest)."""
    parts = [p for p in re.split(r"[—–()\[\]/\"]+", word) if p.strip()]
    if len(parts) < 2:
        return word
    t = (tok or "").casefold().lstrip("▁ ").strip()
    hit = [p for p in parts if t and t in p.casefold()]
    return hit[0] if hit else max(parts, key=len)


def _dedupe(words) -> list:
    out, seen = [], set()
    for w in words:
        if _is_word(w) and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def parse_list(raw: str):
    """First JSON list in a model reply (``` fences, single quotes and stray text tolerated); [] if none."""
    s = raw.strip().strip("`")
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if not m:
        return []
    for load in (json.loads, ast.literal_eval):
        try:
            v = load(m.group(0))
            return v if isinstance(v, list) else []
        except Exception:  # noqa: BLE001
            pass
    return re.findall(r"[\"']([^\"']+)[\"']", m.group(0))


# ---------------------------------------------------------------- concept words
def concept_words(concept: str, n: int = N_WORDS, client=None, path=CW_PATH, force: bool = False) -> list:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = f"{concept}|{n}"
    with open(path.with_suffix(".lock"), "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        cache = json.loads(path.read_text()) if path.exists() else {}
        if n == N_WORDS and key not in cache and cache.get(concept, {}).get("v") == CW_VERSION:
            cache[key] = cache.pop(concept)                       # entry from before the "|n" keys
        if key in cache and cache[key].get("v") == CW_VERSION and not force:
            return cache[key]["words"]
        raw = (client or get_client()).generate(CONCEPT_WORDS_PROMPT.format(concept=concept, n=n))
        words = _dedupe(normalize(w) for w in parse_list(raw) if isinstance(w, str))[:n]
        cache[key] = {"words": words, "v": CW_VERSION, "raw": raw}
        save_json(path, cache)
        return words


# ---------------------------------------------------------------- feature words
def feature_words(tids, counts: dict, n: int = N_WORDS, tok_str=None) -> list:
    """First n distinct normalized source words of the tokens `tids` (in |y| order). tok_str: tid -> token string,
    used for tokens with no source word."""
    out, seen = [], set()
    for t in tids:
        c = counts.get(int(t))
        if c:
            merged = Counter()
            for (_, word), k in c.items():
                merged[word] += k
            word = _piece(merged.most_common(1)[0][0], tok_str(int(t)) if tok_str else None)
        elif tok_str is not None:
            word = tok_str(int(t))
        else:
            continue
        w = normalize(word)
        if _is_word(w) and w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) == n:
                break
    return out


# ---------------------------------------------------------------- matching
def exact_pairs(cw, fw) -> list:
    fset = {normalize(w) for w in fw}
    return [[w, w] for w in dict.fromkeys(normalize(w) for w in cw) if w in fset]


def validate_pairs(pairs, cw, fw) -> list:
    """Keep [a, b] with a in cw and b in fw (after normalization), each word used once, in order."""
    cset, fset = {normalize(w) for w in cw}, {normalize(w) for w in fw}
    out, used_a, used_b = [], set(), set()
    for p in pairs:
        if isinstance(p, dict):
            p = list(p.values())
        if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(isinstance(x, str) for x in p):
            continue
        a, b = normalize(p[0]), normalize(p[1])
        if a in cset and b in fset and a not in used_a and b not in used_b:
            used_a.add(a)
            used_b.add(b)
            out.append([a, b])
    return out


def match(cw, fw, client=None, llm: bool = False, min_pairs: int = MIN_PAIRS) -> dict:
    ex = exact_pairs(cw, fw)
    llm_pairs, raw = [], None
    if llm and cw and fw:
        raw = (client or get_client()).generate(MATCH_PROMPT.format(a=json.dumps(cw, ensure_ascii=False),
                                                                    b=json.dumps(fw, ensure_ascii=False)))
        llm_pairs = parse_list(raw)
    pairs = validate_pairs(ex + llm_pairs, cw, fw)
    return {"pairs": pairs, "n_pairs": len(pairs), "exact": len(ex), "accept": len(pairs) >= min_pairs,
            "llm": bool(llm), "v": MATCH_VERSION if llm else None, "raw": raw}
