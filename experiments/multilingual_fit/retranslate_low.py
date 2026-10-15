#!/usr/bin/env python3
"""Step 3 of the translation plan: re-translate low-scoring sentences, keep the better version.

Three stages (docs/Part 2/multilingual_features.md, "Translated sentence sets"):

  1. translate (this script, project .venv, Gemini): every main pair with COMET-Kiwi below
     --threshold in data_translated/quality/scores.csv is translated again with
     --model (default gemini-3.6-flash). The prompt adds rules for field terminology,
     quotations and titles. Output: data_translated/retranslate/candidates.json.
  2. score (score_translations.py --candidates, .venv-comet, GPU): Kiwi + LaBSE for every
     candidate -> candidates_scored.json.
  3. apply (this script with --apply): a candidate replaces the current translation only
     if its Kiwi is higher and it passes the step-1 checks. It rewrites
     the data files and the affected rows of quality/scores.csv, and logs every decision (old
     and new text, both scores) to retranslate/replacements.csv, so the change can be undone.

Usage (no -I: it imports translate_sentences.py from this folder):
    .venv/bin/python experiments/multilingual_fit/retranslate_low.py
    sbatch experiments/multilingual_fit/run_score_translations.sbatch \
        --candidates experiments/multilingual_fit/data_translated/retranslate/candidates.json
    .venv/bin/python experiments/multilingual_fit/retranslate_low.py --apply
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import translate_sentences as ts

HERE = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)

EXTRA_RULES = """- Use the standard terminology of the field in {lang_name}: the established {lang_name} terms for sports rules, positions, scoring and equipment, for firearms and their parts, for games and gambling, as {lang_name} specialist texts use them. If {lang_name} normally borrows the English term, use the borrowed form in {lang_name} spelling; never translate a technical term word for word.
- Translate quotations, verses, sayings and proverbs into {lang_name} too, faithfully (keep their literal meaning; do not substitute a different local saying). Old or Middle English quotations are translated into modern {lang_name}.
- For titles of books, films, games, parks and organisations, use the form {lang_name} Wikipedia uses: the established {lang_name} title if there is one, otherwise the original.
"""


def translate_group(args, key, lang, set_name, items):
    """items: [(idx, en)] -> {idx: new translation}. Chunks of --chunk-size, then one by one."""
    topic = (' (random sentences on unrelated topics)' if set_name == ts.NEUTRAL
             else f' about "{set_name}"')
    prompt_tpl = ts.PROMPT.replace("Return a JSON array", EXTRA_RULES + "\nReturn a JSON array")
    out = {}

    def ask(sub):
        text = "\n".join(f"{n + 1}. {en}" for n, (_, en) in enumerate(sub))
        prompt = prompt_tpl.format(lang_name=ts.LANG_NAMES[lang], topic=topic, items=text)
        got = ts.call_gemini(args, key, prompt, f"retranslate/{lang}/{set_name}")
        return {sub[int(o["id"]) - 1][0]: o["text"].strip() for o in got
                if isinstance(o, dict) and "id" in o and 1 <= int(o["id"]) <= len(sub)}

    for c in range(0, len(items), args.chunk_size):
        todo = items[c:c + args.chunk_size]
        for _ in range(args.retries):
            try:
                res = ask(todo)
            except Exception as e:
                print(f"  [{lang}/{set_name}] call failed: {e}", flush=True)
                continue
            for i, en in todo:
                if i in res and not ts.problems(lang, en, res[i]):
                    out[i] = res[i]
            todo = [(i, en) for i, en in todo if i not in out]
            if not todo:
                break
        for i, en in todo:  # one at a time; a candidate that still fails the checks is skipped
            for _ in range(args.retries):
                try:
                    t = ask([(i, en)]).get(i)
                except Exception as e:
                    print(f"  [{lang}/{set_name}/{i}] call failed: {e}", flush=True)
                    continue
                if t and not ts.problems(lang, en, t):
                    out[i] = t
                    break
    return out


def stage_translate(args):
    rows = [r for r in csv.DictReader(open(args.data_dir / "quality" / "scores.csv"))
            if r["source"] == "main" and float(r["kiwi"]) < args.threshold]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lang"], r["set"])].append((int(r["idx"]), r["en"]))
    print(f"{len(rows)} sentences below Kiwi {args.threshold} in {len(groups)} (lang, set) groups", flush=True)

    rdir = args.data_dir / "retranslate"
    rdir.mkdir(exist_ok=True)
    args.out_dir = rdir  # call_gemini logs usage to <out_dir>/usage.jsonl
    key = ts.load_key()
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {g: pool.submit(translate_group, args, key, g[0], g[1], items) for g, items in groups.items()}
    cands = []
    for (lang, set_name), f in futs.items():
        new = f.result()
        cands += [{"lang": lang, "set": set_name, "idx": i, "en": en, "tr": new[i]}
                  for i, en in groups[(lang, set_name)] if i in new]
    (rdir / "candidates.json").write_text(json.dumps(
        {"model": args.model, "threshold": args.threshold, "candidates": cands}, ensure_ascii=False, indent=1))
    print(f"{len(cands)} candidates of {len(rows)} ({time.time() - t0:.0f}s) -> {rdir / 'candidates.json'}")


def stage_apply(args):
    apply_candidates(args.data_dir, args.data_dir / "retranslate",
                     lambda c, old_k: c["kiwi"] > old_k and not ts.problems(c["lang"], c["en"], c["tr"]))


def apply_candidates(data_dir, rdir, accept):
    """Replace current translations by scored candidates where accept(candidate, old_kiwi).

    Reads <rdir>/candidates_scored.json (score_translations.py --candidates), rewrites the data
    files and the affected rows of quality/scores.csv, and logs every decision to
    <rdir>/replacements.csv. Shared by step 3 (this script) and step 4b (fix_terms_parens.py)."""
    scored = json.loads((rdir / "candidates_scored.json").read_text())
    scores_path = data_dir / "quality" / "scores.csv"
    with open(scores_path) as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    cur = {(r["lang"], r["set"], int(r["idx"])): r for r in rows if r["source"] == "main"}

    data = {}
    for lang in ts.LANGS:
        data[(lang, "c")] = json.loads((data_dir / f"concept_sentences_{lang}.json").read_text())
        data[(lang, "n")] = json.loads((data_dir / f"neutral_sentences_{lang}.json").read_text())

    log, n_rep = [], 0
    for c in scored["candidates"]:
        r = cur[(c["lang"], c["set"], c["idx"])]
        old_k, new_k = float(r["kiwi"]), c["kiwi"]
        if r["tr"] != c["old_tr"]:
            sys.exit(f"{c['lang']}/{c['set']}/{c['idx']}: data changed since scoring; rerun stages 1-2")
        take = bool(accept(c, old_k))
        log.append({"lang": c["lang"], "set": c["set"], "idx": c["idx"], "replaced": int(take),
                    "kiwi_old": f"{old_k:.4f}", "kiwi_new": f"{new_k:.4f}",
                    "labse_old": r["labse"], "labse_new": f"{c['labse']:.4f}",
                    "en": c["en"], "old": r["tr"], "new": c["tr"]})
        if not take:
            continue
        n_rep += 1
        if c["set"] == ts.NEUTRAL:
            data[(c["lang"], "n")][c["idx"]]["sentence"] = c["tr"]
        else:
            [s for s in data[(c["lang"], "c")] if s["concept"] == c["set"]][0]["sentences"][c["idx"]] = c["tr"]
        r.update(kiwi=f"{new_k:.4f}", labse=f"{c['labse']:.4f}", tr=c["tr"],
                 len_ratio=f"{len(c['tr']) / max(len(c['en']), 1):.2f}")

    for (lang, kind), d in data.items():
        name = "concept" if kind == "c" else "neutral"
        (data_dir / f"{name}_sentences_{lang}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
    with open(scores_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(rdir / "replacements.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"replaced {n_rep} of {len(log)} candidates; log -> {rdir / 'replacements.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="stage 3: replace where the candidate scores higher")
    ap.add_argument("--data-dir", type=Path, default=HERE / "data_translated")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--thinking-level", default="minimal")
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    stage_apply(args) if args.apply else stage_translate(args)


if __name__ == "__main__":
    main()
