"""Checks for overlap_judge.py.  Run:  pytest experiments/varwidth/test_overlap_judge.py  (or .venv/bin/python -m experiments.varwidth.test_overlap_judge, no pytest)"""
from collections import Counter

from experiments.varwidth.overlap_judge import exact_pairs, feature_words, match, normalize, parse_list, validate_pairs


def test_normalize():
    assert normalize("▁Drugs") == "drug"
    assert normalize(" Heroin,") == "heroin"
    assert normalize("injuries") == "injury"
    assert normalize("churches") == "church"
    assert normalize("addiction's") == "addiction"
    assert normalize("glass") == "glass"
    assert normalize("virus") == "virus" and normalize("cannabis") == "cannabis"
    assert normalize("gas") == "gas"          # len 3: kept
    assert normalize("...") == ""


def test_feature_words_dedupe_and_fallback():
    counts = {1: Counter({("en", "Drugs"): 5, ("en", "drugged"): 1}),
              2: Counter({("en", "drug"): 3}),                     # same word after normalization: dropped
              3: Counter({("en", ","): 9}),                         # punctuation: skipped
              4: Counter({("en", "Heroin"): 2, ("de", "heroin"): 1}),
              6: Counter({("en", "a"): 4})}                         # too short: skipped
    toks = {5: "▁opium", 7: "▁Opium"}
    assert feature_words([1, 2, 3, 4, 5, 6, 7], counts, tok_str=toks.get) == ["drug", "heroin", "opium"]
    assert feature_words([1, 2, 3, 4, 5], counts) == ["drug", "heroin"]   # no tok_str: tokens without words skipped
    assert feature_words([1, 4, 5], counts, n=2, tok_str=toks.get) == ["drug", "heroin"]


def test_feature_words_pieces_and_numbers():
    counts = {1: Counter({("en", "1961)—require"): 3}), 2: Counter({("en", "diacetylmorphine—this"): 2}),
              3: Counter({("en", "1999"): 4})}
    toks = {1: "▁require", 2: "▁this", 3: "9"}
    assert feature_words([1, 2, 3], counts, tok_str=toks.get) == ["require", "this"]


def test_feature_words_merges_languages():
    counts = {1: Counter({("en", "opioid"): 2, ("es", "opio"): 2, ("fr", "opio"): 1})}
    assert feature_words([1], counts) == ["opio"]


def test_validate_pairs():
    cw, fw = ["heroin", "opioid", "needle", "addiction"], ["heroin", "opioids", "syringe", "drug"]
    pairs = [["heroin", "heroin"], ["opioid", "opioids"], ["needle", "syringe"],
             ["morphine", "drug"],            # hallucinated concept word
             ["addiction", "addicts"],        # feature word not in the list
             ["addiction", "heroin"],         # feature word already used
             ["heroin", "drug"],              # concept word already used
             "junk", ["a"], {"a": "addiction", "b": "drug"}]
    assert validate_pairs(pairs, cw, fw) == [["heroin", "heroin"], ["opioid", "opioid"], ["needle", "syringe"],
                                             ["addiction", "drug"]]


def test_exact_and_match_without_llm():
    cw, fw = ["drug", "heroin", "opium"], ["drugs", "heroin", "the"]
    assert exact_pairs(cw, fw) == [["drug", "drug"], ["heroin", "heroin"]]
    r = match(cw, fw, llm=False, min_pairs=2)
    assert r["n_pairs"] == r["exact"] == 2 and r["accept"]


class FakeClient:
    def generate(self, prompt):
        return '```json\n[["opium", "opio"], ["opium", "the"], ["war", "battle"]]\n```'


def test_match_llm_pairs_validated():
    r = match(["drug", "heroin", "opium", "war"], ["drug", "heroin", "opio", "the"], FakeClient(), llm=True, min_pairs=3)
    assert r["pairs"] == [["drug", "drug"], ["heroin", "heroin"], ["opium", "opio"]]
    assert (r["n_pairs"], r["exact"], r["accept"]) == (3, 2, True)


def test_parse_list():
    assert parse_list("Sure: ['a', 'b']") == ["a", "b"]
    assert parse_list("no list") == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
