#!/usr/bin/env python3
"""Patch the existing notebooks for the multilingual-whitening work (idempotent; outputs of untouched cells are kept).

Multilingual notebooks (6 concepts x main/low-RAM):
    - cell 2: JUDGE_FORMAT (default "top_word" = the format these notebooks used, so cached verdicts stay valid),
      EXT_PER_LANG = 300 and EXT_TOPUP = "translated" for 3.2.h
    - 3.2.c / 3.2.d: judge input via ember/judge_words.py per JUDGE_FORMAT; judge cache per format
    - 3.2.h: the capped extended sets (extended_sets.build_concept_n / build_neutral_n / ext_paths) instead of the whole page
Whitened notebooks (Harry Potter / Valentine's Day / Ancient Rome x Gemma / Llama x main/low-RAM):
    - cell 2: JUDGE_FORMAT (default "tokens" = the format they used)
    - judge input via ember/judge_words.py when JUDGE_FORMAT = "words"; judge caches per format

A notebook that already contains "JUDGE_FORMAT" is skipped. Close the notebooks in VS Code before running.
Usage: python scripts/patch_notebooks_ml.py [--only PATH ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import nb_patch as P  # noqa: E402

ML_CONCEPTS = ["hp", "gambling", "gun", "halloween", "pornography", "valentines_day"]
WH_NAMES = ["whitened", "whitened_llama", "valentines_whitened", "valentines_whitened_llama", "ancient_rome_whitened",
            "ancient_rome_whitened_llama"]
ML_NOTEBOOKS = [f"ember_three_steps_{c}_multilingual_llama.ipynb" for c in ML_CONCEPTS] + \
               [f"low_ram_demands/{c}_multilingual_llama.ipynb" for c in ML_CONCEPTS]
WH_NOTEBOOKS = [f"ember_three_steps_{c}.ipynb" for c in WH_NAMES] + [f"low_ram_demands/{c}.ipynb" for c in WH_NAMES]

GEMINI_LINE = re.compile(r'^(GEMINI_MODEL = "models/[^\n]*\n)', re.M)

ML_CONFIG_ADD = '''JUDGE_FORMAT = "top_word"   # Stage-1 judge input: "top_word" (token → most frequent word (lang), this notebook's original
                            # format) | "tokens" (bare tokens, paper) | "words" (token → its source words by language, alphabetical)

# ---- 3.2.h extended set (fixed Z) ----------------------------------------
EXT_PER_LANG = 300       # sentences per language and side in S_C^ext / S_N^ext; None = whole page (the original sets)
EXT_TOPUP = "translated" # "translated": pages shorter than EXT_PER_LANG are filled with Gemini translations; None: native only
'''

WH_CONFIG_ADD = '''JUDGE_FORMAT = "tokens"   # Stage-1 judge input: "tokens" (bare tokens, paper; this notebook's original format)
                          # | "words" (token → the words it was cut from in S_C ∪ S_N, alphabetical)
'''

ML_JUDGE_FN = '''from ember.judge_words import desc_sys_prompt, judge_cache_name, judge_items, token_word_counts

jw_counts = token_word_counts(tokenizer, [s for s, _ in dataset], [sent_lang[s] for s, _ in dataset])  # tid -> Counter{(lang, word)}


def ml_judge_tokens(f: int, n: int = TOP_K_TOKENS) -> list:
    """Top-n support tokens of feature f by |Y|, formatted by JUDGE_FORMAT ("top_word": 'token → word (lang)', as before)."""
    mags = Y[f].abs()
    top = torch.argsort(mags, descending=True)[:n]
    top = top[mags[top] > 0].tolist()
    if JUDGE_FORMAT == "top_word":   # the original format, unchanged (tok_word from 3.2.a′)
        out = []
        for j in top:
            tid = vstar_ids[j]
            word, lang = tok_word.get(tid, (tok_str[tid], "?"))
            out.append(f"{tok_str[tid]} → {word} ({lang})")
        return out
    return judge_items([vstar_ids[j] for j in top], tok_str, jw_counts, JUDGE_FORMAT, multilingual=True)

'''

ML_EXT_CONCEPT = '''from extended_sets import build_concept_n, build_neutral_n, ext_paths
if EXT_PER_LANG is not None:   # capped sets, built once and cached under the concept's extended dir (no network calls)
    build_concept_n(CONCEPT, EXT_PER_LANG, EXT_TOPUP)
    build_neutral_n(CONCEPT, NEUTRAL_REP, EXT_PER_LANG, EXT_TOPUP)
_ext_files = ext_paths(CONCEPT, NEUTRAL_REP, EXT_PER_LANG, EXT_TOPUP)   # EXT_PER_LANG None: the original whole-page sets
CONCEPT_EXT_PATH, NEUTRAL_EXT_PATH = _ext_files["concept"], _ext_files["neutral"]
'''

ML_EXT_HP = '''if EXT_PER_LANG is None:   # the original whole-page sets
    NEUTRAL_EXT_PATH = NEUTRAL_EXT_TRANSLATED_PATH if NEUTRAL_SOURCE == "translated" else NEUTRAL_EXT_NATIVE_PATH
else:                      # capped sets, built once and cached under data/extended (no network calls)
    assert NEUTRAL_SOURCE == "translated", "the capped extended sets have translated neutral sentences only"
    from extended_sets import build_concept_n, build_neutral_n, ext_paths
    build_concept_n(CONCEPT, EXT_PER_LANG, EXT_TOPUP)
    build_neutral_n(CONCEPT, 0, EXT_PER_LANG, EXT_TOPUP)   # replicate 0 = data/neutral_sentences_translated.json, the V* neutral set
    _ext_files = ext_paths(CONCEPT, 0, EXT_PER_LANG, EXT_TOPUP)
    CONCEPT_EXT_PATH, NEUTRAL_EXT_PATH = _ext_files["concept"], _ext_files["neutral"]
'''

EXT_NOTE = """
**Capped sets (`EXT_PER_LANG`, `EXT_TOPUP` in cell 2).** By default $S_C^{ext}$ is, per language, the 40 $V^*$ concept sentences plus a seeded
sample of the concept page up to 300 sentences; pages shorter than that are filled with Gemini translations of the other languages' pages
(`experiments/multilingual_hp/topup_translated.py`, rows marked `source = "translated"`). $S_N^{ext}$ is, per language, the $V^*$ neutral
sentences plus translated neutral sentences, as many as the concept side, so $V^*_{ext}\\supseteq V^*$. `EXT_PER_LANG = None` restores the
whole-page sets described above."""

TR_PRINT = ('from collections import Counter\n'
            '_tr = Counter(r["lang"] for r in ext_concept_rows if r.get("source") == "translated")\n'
            'print(f"translated top-up in S_C^ext: {dict(_tr)} ({sum(_tr.values())} of {len(ext_concept_rows)})")\n')

WH_JUDGE_ADD = '''
from ember.judge_words import desc_sys_prompt, judge_cache_name, judge_items, token_word_counts

DESC_SYS = desc_sys_prompt(JUDGE_FORMAT, multilingual=False)   # = TOKENS_DESC_SYS for "tokens"
jw_counts = token_word_counts(tokenizer, [s for s, _ in dataset]) if JUDGE_FORMAT == "words" else {}


def judge_input(Ymat: torch.Tensor, f: int, activating_tokens) -> list:
    """What Stage 1 sees for feature f of Ymat: "tokens" = the top activating tokens (as before); "words" = token → source words."""
    if JUDGE_FORMAT == "tokens":
        return list(activating_tokens[:TOP_K_TOKENS])
    mags = Ymat[f].abs()
    top = torch.argsort(mags, descending=True)[:TOP_K_TOKENS]
    top = top[mags[top] > 0].tolist()
    return judge_items([vstar_ids[j] for j in top], lambda t: tokenizer.decode([t]), jw_counts, JUDGE_FORMAT,
                       multilingual=False)


cand["judge_tokens"] = [judge_input(Y, int(f), toks) for f, toks in zip(cand["feature"], cand["activating_tokens"])]
'''


def add_config(nb, text: str) -> None:
    i = P.find(nb, "TOP_K_TOKENS = 20", "code")
    s = P.src(nb["cells"][i])
    assert GEMINI_LINE.search(s), "GEMINI_MODEL line not found"
    P.set_source(nb, i, GEMINI_LINE.sub(lambda m: m.group(1) + text, s, count=1))


def patch_ml(nb, is_hp: bool) -> None:
    add_config(nb, ML_CONFIG_ADD)
    # 3.2.c: judge tokens per format
    i = P.find(nb, "def ml_judge_tokens(f: int")
    s = P.src(nb["cells"][i])
    start, end = s.index("def ml_judge_tokens"), s.index("cand = candidates[")
    P.set_source(nb, i, s[:start] + ML_JUDGE_FN + "\n" + s[end:])
    # judge cache per format
    if is_hp:
        i = P.find(nb, 'descriptions = globals().get("descriptions", {})')
        P.replace_in(nb, i, 'descriptions = globals().get("descriptions", {})\nverdicts = globals().get("verdicts", {})',
                     'if globals().get("_JUDGE_FORMAT_CACHED", "top_word") != JUDGE_FORMAT:   # never reuse another format\'s answers\n'
                     '    descriptions, verdicts = {}, {}\n'
                     'else:\n'
                     '    descriptions, verdicts = globals().get("descriptions", {}), globals().get("verdicts", {})\n'
                     '_JUDGE_FORMAT_CACHED = JUDGE_FORMAT')
    else:
        i = P.find(nb, 'JUDGE_PATH = RUN_DIR / "judge.json"')
        P.replace_in(nb, i, 'JUDGE_PATH = RUN_DIR / "judge.json"',
                     'JUDGE_PATH = RUN_DIR / judge_cache_name(JUDGE_FORMAT, "top_word")   # judge.json for the original format')
    # 3.2.d: system prompt per format
    i = P.find(nb, "ifx.TOKENS_DESC_SYS = ML_TOKENS_DESC_SYS")
    P.replace_in(nb, i, "ifx.TOKENS_DESC_SYS = ML_TOKENS_DESC_SYS  # get_description reads the module global at call time",
                 'DESC_SYS = ML_TOKENS_DESC_SYS if JUDGE_FORMAT == "top_word" else desc_sys_prompt(JUDGE_FORMAT, multilingual=True)\n'
                 "ifx.TOKENS_DESC_SYS = DESC_SYS  # get_description reads the module global at call time (describe() below too)")
    P.replace_in(nb, i, "print(ML_TOKENS_DESC_SYS.strip(),", 'print(f"JUDGE_FORMAT = {JUDGE_FORMAT!r}"); print(DESC_SYS.strip(),')
    # 3.2.h: capped extended sets
    if is_hp:
        i = P.find(nb, "NEUTRAL_EXT_PATH = NEUTRAL_EXT_TRANSLATED_PATH if NEUTRAL_SOURCE")
        P.replace_in(nb, i, 'NEUTRAL_EXT_PATH = NEUTRAL_EXT_TRANSLATED_PATH if NEUTRAL_SOURCE == "translated" else NEUTRAL_EXT_NATIVE_PATH\n',
                     ML_EXT_HP)
    else:
        i = P.find(nb, 'CONCEPT_EXT_PATH = concept_paths(CONCEPT)["extended"] / "concept_ext.json"')
        P.replace_in(nb, i, 'CONCEPT_EXT_PATH = concept_paths(CONCEPT)["extended"] / "concept_ext.json"\n'
                            'NEUTRAL_EXT_PATH = neutral_paths(CONCEPT, NEUTRAL_REP)["ext"]\n', ML_EXT_CONCEPT)
    P.replace_in(nb, i, 'print(f"S_C^ext = {len(ext_concept)}', TR_PRINT + 'print(f"S_C^ext = {len(ext_concept)}')
    j = P.find(nb, "### 3.2.h Extended vocabulary", "markdown")
    P.set_source(nb, j, P.src(nb["cells"][j]).rstrip("\n") + "\n" + EXT_NOTE)


def patch_wh(nb) -> None:
    add_config(nb, WH_CONFIG_ADD)
    i = P.find(nb, 'cand["judge_tokens"] = cand["activating_tokens"].apply(lambda toks: toks[:TOP_K_TOKENS])')
    s = P.src(nb["cells"][i])
    line = 'cand["judge_tokens"] = cand["activating_tokens"].apply(lambda toks: toks[:TOP_K_TOKENS])\n'
    tail = s[s.index(line) + len(line):]
    P.set_source(nb, i, s[:s.index(line)] + WH_JUDGE_ADD.lstrip("\n") + tail)
    i = P.find(nb, 'JUDGE_PATH = RUN_DIR / "judge.json"')
    P.replace_in(nb, i, 'JUDGE_PATH = RUN_DIR / "judge.json"',
                 'JUDGE_PATH = RUN_DIR / judge_cache_name(JUDGE_FORMAT, "tokens")   # judge.json for the original format')
    i = P.find(nb, 'JUDGE0_PATH = ORIG_DIR / "judge0.json"')
    P.replace_in(nb, i, 'JUDGE0_PATH = ORIG_DIR / "judge0.json"',
                 'JUDGE0_PATH = ORIG_DIR / judge_cache_name(JUDGE_FORMAT, "tokens", "judge0")')
    P.replace_in(nb, i, 'get_description(client, row["activating_tokens"][:TOP_K_TOKENS])',
                 'get_description(client, judge_input(Y0, f, row["activating_tokens"]), sys_prompt=DESC_SYS)')
    i = P.find(nb, 'get_description(client, row["judge_tokens"])')
    P.replace_in(nb, i, 'get_description(client, row["judge_tokens"])', 'get_description(client, row["judge_tokens"], sys_prompt=DESC_SYS)')
    for i in P.find_all(nb, 'get_description(client, feat_tokens.loc[f, "activating_tokens"][:TOP_K_TOKENS])'):
        P.replace_in(nb, i, 'get_description(client, feat_tokens.loc[f, "activating_tokens"][:TOP_K_TOKENS])',
                     'get_description(client, judge_input(Y, f, feat_tokens.loc[f, "activating_tokens"]), sys_prompt=DESC_SYS)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, help="patch only these notebooks (paths relative to the repo)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    todo = [(p, "ml") for p in ML_NOTEBOOKS] + [(p, "wh") for p in WH_NOTEBOOKS]
    if args.only:
        todo = [(p, k) for p, k in todo if p in set(args.only)]
    for rel, kind in todo:
        path = ROOT / rel
        if not path.exists():
            print(f"missing: {rel}")
            continue
        nb = P.load(path)
        if P.has(nb, "JUDGE_FORMAT"):
            print(f"already patched: {rel}")
            continue
        if kind == "ml":
            patch_ml(nb, is_hp="hp_multilingual" in rel)
        else:
            patch_wh(nb)
        if not args.dry_run:
            P.save(nb, path)
        print(f"{'checked' if args.dry_run else 'patched'}: {rel}")


if __name__ == "__main__":
    main()
