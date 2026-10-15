#!/usr/bin/env python3
"""Step 4b of the translation plan: fix term errors and dropped parentheses.

Two kinds of sentences are translated again (docs/Part 2/multilingual_features.md,
"Step 4 result"):
  terms   the sentences the step-4 judge marked ERROR (terms/judged.csv), with the judge's
          correct term given as a hint ("Terms: insufflation -> инсуффляция");
  parens  every pair whose English contains "(" but whose translation does not. The step-1
          prompt rule "no ... transliterations in brackets" made the translator drop
          parenthesised content (acronyms, original-language names, dates); the new prompt
          says to keep it.
A sentence can be in both.

Stages, as in retranslate_low.py:
  1. translate (this script, project .venv, Gemini --model) -> fix4b/candidates.json
  2. score     (score_translations.py --candidates, .venv-comet, GPU) -> candidates_scored.json
  3. apply     (this script with --apply). A candidate replaces the current translation only if it
     passes the step-1 checks, keeps "(" where that was the problem, contains every hinted
     term (stem match as in term_check.py; alternatives split on "/", "or", ...), and its
     Kiwi is at most --kiwi-tolerance below the current one. A small drop is tolerated because the
     fix restores content, which Kiwi does not always reward. Log: fix4b/replacements.csv.

Usage (no -I: it imports modules from this folder):
    .venv/bin/python experiments/multilingual_fit/fix_terms_parens.py
    sbatch experiments/multilingual_fit/run_score_translations.sbatch \
        --candidates experiments/multilingual_fit/data_translated/fix4b/candidates.json
    .venv/bin/python experiments/multilingual_fit/fix_terms_parens.py --apply
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import retranslate_low as rl
import term_check as tc
import translate_sentences as ts

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
csv.field_size_limit(sys.maxsize)

PROMPT = """Translate the numbered English sentences below into {lang_name}.

They are sentences from English Wikipedia{topic}. Rules:
- Translate each sentence faithfully and completely, as a single sentence, in the natural style of a {lang_name} Wikipedia article.
- Use the established {lang_name} forms of names, places, titles and terms (e.g. the usual {lang_name} name of a person, city, war, book or film), not word-for-word renderings.
- Write dates and numbers the way {lang_name} text does.
- Keep everything the English has in parentheses, in parentheses: abbreviations, original-language names and their glosses, dates and reign years, unit conversions, cross-references. Translate English prose inside them; keep names, abbreviations and numbers (or use their standard {lang_name} form).
- Add nothing that is not in the English: no explanations or notes of your own, no quotation marks around the whole sentence.
- Keep every sentence separate, even if it is a fragment; never merge or split sentences.
""" + rl.EXTRA_RULES + """- Some sentences are followed by a line "Terms:" giving the required {lang_name} rendering of a term. Use it, inflected as the grammar needs. The Terms line is an instruction, not part of the sentence.

Return a JSON array with one object per input sentence: {{"id": <number>, "text": "<translation>"}}.

{items}"""


def load_sets(data_dir):
    en = {c["concept"]: c["sentences"] for c in json.loads((REPO / "data" / "concept_sentences.json").read_text())}
    en[ts.NEUTRAL] = [n["sentence"] for n in json.loads((REPO / "data" / "neutral_sentences.json").read_text())]
    tr = {}
    for lang in ts.LANGS:
        tr[lang] = {c["concept"]: c["sentences"] for c in
                    json.loads((data_dir / f"concept_sentences_{lang}.json").read_text())}
        tr[lang][ts.NEUTRAL] = [n["sentence"] for n in
                                json.loads((data_dir / f"neutral_sentences_{lang}.json").read_text())]
    return en, tr


def select(data_dir):
    """{(lang, set, idx): {"en", "hints": [(en_term, term, kind)], "paren": bool}}"""
    en, tr = load_sets(data_dir)
    todo = defaultdict(lambda: {"hints": [], "paren": False})
    for r in csv.DictReader(open(data_dir / "terms" / "judged.csv")):
        if r["verdict"] == "ERROR" and r["correct_term"].strip():
            todo[(r["lang"], r["concept"], int(r["idx"]))]["hints"].append(
                (r["en_term"], r["correct_term"].strip(), r["kind"]))
    for lang in ts.LANGS:
        for name, sents in en.items():
            for i, s in enumerate(sents):
                if "(" in s and "(" not in tr[lang][name][i]:
                    todo[(lang, name, i)]["paren"] = True
    for (lang, name, i), d in todo.items():
        d["en"] = en[name][i]
    return todo


def alternatives(term):
    """'Фронт Стрезы / Стрезский фронт', 'cannabis (ou Cannabis)' -> each alternative."""
    term = re.sub(r"\((?:ou|or|o|oder|или)\s+([^)]*)\)", r"/ \1", term)
    return [a.strip() for a in re.split(r"\s+/\s+|/|\s+(?:or|ou|o|oder|или)\s+", term) if a.strip()]


def has_term(tr, term, lang, kind):
    return any(tc.contains_native(tr, a, lang, kind == "name") for a in alternatives(term))


def translate_group(args, key, lang, set_name, items):
    """items: [((lang, set, idx), info)] -> {idx: translation}. Chunks, then one by one."""
    topic = (" (random sentences on unrelated topics)" if set_name == ts.NEUTRAL else f' about "{set_name}"')
    out = {}

    def ask(sub):
        lines = []
        for n, (k, d) in enumerate(sub):
            lines.append(f"{n + 1}. {d['en']}")
            if d["hints"]:
                lines.append("   Terms: " + "; ".join(f'"{e}" -> "{t}"' for e, t, _ in d["hints"]))
        prompt = PROMPT.format(lang_name=ts.LANG_NAMES[lang], topic=topic, items="\n".join(lines))
        got = ts.call_gemini(args, key, prompt, f"fix4b/{lang}/{set_name}")
        return {sub[int(o["id"]) - 1][0][2]: o["text"].strip() for o in got
                if isinstance(o, dict) and "id" in o and 1 <= int(o["id"]) <= len(sub)}

    def good(d, t):
        return (t and not ts.problems(lang, d["en"], t) and ("(" in t or not d["paren"])
                and all(has_term(t, term, lang, kind) for _, term, kind in d["hints"]))

    for c in range(0, len(items), args.chunk_size):
        todo = items[c:c + args.chunk_size]
        best = {}
        for _ in range(args.retries):
            try:
                res = ask(todo)
            except Exception as e:
                print(f"  [{lang}/{set_name}] call failed: {e}", flush=True)
                continue
            for k, d in todo:
                if k[2] in res:
                    best.setdefault(k[2], res[k[2]])
                    if good(d, res[k[2]]):
                        out[k[2]] = res[k[2]]
            todo = [(k, d) for k, d in todo if k[2] not in out]
            if not todo:
                break
        for k, d in todo:  # one at a time; else keep the best attempt (apply decides)
            for _ in range(args.retries):
                try:
                    t = ask([(k, d)]).get(k[2])
                except Exception as e:
                    print(f"  [{lang}/{set_name}/{k[2]}] call failed: {e}", flush=True)
                    continue
                if t:
                    best[k[2]] = t
                if good(d, t):
                    out[k[2]] = t
                    break
            if k[2] not in out and k[2] in best:
                out[k[2]] = best[k[2]]
    return out


def stage_translate(args):
    todo = select(args.data_dir)
    groups = defaultdict(list)
    for k, d in todo.items():
        groups[(k[0], k[1])].append((k, d))
    n_t = sum(bool(d["hints"]) for d in todo.values())
    n_p = sum(d["paren"] for d in todo.values())
    print(f"{len(todo)} sentences: {n_t} with term hints, {n_p} with dropped parentheses", flush=True)

    rdir = args.data_dir / "fix4b"
    rdir.mkdir(exist_ok=True)
    args.out_dir = rdir
    key = ts.load_key()
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {g: pool.submit(translate_group, args, key, g[0], g[1], items) for g, items in groups.items()}
    cands = []
    for g, f in futs.items():
        new = f.result()
        for k, d in groups[g]:
            if k[2] in new:
                cands.append({"lang": k[0], "set": k[1], "idx": k[2], "en": d["en"], "tr": new[k[2]],
                              "hints": d["hints"], "paren": d["paren"]})
    (rdir / "candidates.json").write_text(json.dumps(
        {"model": args.model, "candidates": cands}, ensure_ascii=False, indent=1))
    print(f"{len(cands)} candidates of {len(todo)} ({time.time() - t0:.0f}s) -> {rdir / 'candidates.json'}")


def stage_apply(args):
    def accept(c, old_k):
        return (not ts.problems(c["lang"], c["en"], c["tr"])
                and ("(" in c["tr"] or not c["paren"])
                and all(has_term(c["tr"], term, c["lang"], kind) for _, term, kind in c["hints"])
                and c["kiwi"] >= old_k - args.kiwi_tolerance)
    rl.apply_candidates(args.data_dir, args.data_dir / "fix4b", accept)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=HERE / "data_translated")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--thinking-level", default="minimal")
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--kiwi-tolerance", type=float, default=0.02)
    args = ap.parse_args()
    stage_apply(args) if args.apply else stage_translate(args)


if __name__ == "__main__":
    main()
