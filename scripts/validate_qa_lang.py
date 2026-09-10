#!/usr/bin/env python3
"""Pre-flight check for the QA sets. Run before any GPU job.

Mirrors what the eval path actually requires, and runs the REAL
``prepare_mc_items`` so a latent AssertionError surfaces here rather than
half an hour into a Slurm job. Exits non-zero on any FAIL.
"""
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ember.local_datasets import MCQAItem                    # noqa: E402
from ember.evals.mc import prepare_mc_items                  # noqa: E402

LANGS = ["en", "de", "es", "fr", "ru"]
DATA, QA = ROOT / "data", ROOT / "data" / "qa_lang"

def path(kind, lang):
    return DATA / f"{kind}.json" if lang == "en" else QA / f"{kind}_{lang}.json"

fails, warns = [], []

def check(cond, msg):
    (fails if not cond else []).append(msg) if not cond else None

ref_counts = None
for lang in LANGS:
    p = path("mc_questions", lang)
    if not p.exists():
        fails.append(f"[{lang}] MISSING {p.relative_to(ROOT)}"); continue
    data = json.loads(p.read_text(encoding="utf-8"))
    counts, kept_items, n_raw = {}, [], 0
    bad_count = dup_opts = ws = empty = 0
    for concept, sets in data.items():
        for setname, items in sets.items():
            kept = 0
            for i, o in enumerate(items):
                n_raw += 1
                q = (o.get("q") or "").strip()
                a = (o.get("correct_answer") or "").strip()
                opts = o.get("options") or []
                if not isinstance(opts, list) or not all(isinstance(x, str) for x in opts):
                    fails.append(f"[{lang}] {concept}/{setname}/{i}: options not a list of str"); continue
                if len(opts) != 4:
                    bad_count += 1
                    fails.append(f"[{lang}] {concept}/{setname}/{i}: {len(opts)} options (must be 4)")
                if any(x != x.strip() for x in opts) or o.get("q") != q or o.get("correct_answer") != a:
                    ws += 1
                if len(set(x.strip() for x in opts)) != len(opts):
                    dup_opts += 1
                    fails.append(f"[{lang}] {concept}/{setname}/{i}: duplicate options {opts}")
                if not q or not a:
                    empty += 1
                if q and a and len(opts) >= 4:
                    kept += 1
                    kept_items.append(MCQAItem(concept=concept, subset=setname.split("_")[0],
                                               split=setname.split("_")[1], question=q,
                                               options=[x.strip() for x in opts], correct_answer=a))
            counts[(concept, setname)] = kept
    if ws:    fails.append(f"[{lang}] {ws} item(s) with untrimmed whitespace")
    if empty: fails.append(f"[{lang}] {empty} item(s) with empty q or correct_answer")

    try:
        prepare_mc_items(kept_items)
        prep = "prepare_mc_items OK"
    except AssertionError as e:
        prep = "prepare_mc_items RAISED"
        fails.append(f"[{lang}] prepare_mc_items: {str(e).splitlines()[0]}")

    if ref_counts is None:
        ref_counts = counts
        parity = "reference"
    else:
        diff = {k: (ref_counts.get(k), v) for k, v in counts.items() if ref_counts.get(k) != v}
        missing = set(ref_counts) - set(counts)
        if diff or missing:
            fails.append(f"[{lang}] structure differs from en: {len(diff)} mismatched, {len(missing)} missing")
        parity = "matches en" if not (diff or missing) else "MISMATCH"

    # Only duplicates WITHIN one (concept, set) matter: eval is per concept, and
    # stable_item_id keys on concept/subset/split too, so cross-cell repeats are fine.
    dupq = sum(v - 1 for v in Counter(
        (i.concept, i.subset, i.split, i.question) for i in kept_items).values() if v > 1)
    if dupq:
        warns.append(f"[{lang}] {dupq} question(s) duplicated within a single "
                     f"(concept, set) — inherited from English, graded twice, harmless")
    print(f"MC   {lang}: raw={n_raw} kept={len(kept_items)} cells={len(counts)} "
          f"| bad_count={bad_count} dup_opts={dup_opts} ws={ws} | {parity} | {prep}")

print()
for lang in LANGS:
    p = path("open_questions", lang)
    if not p.exists():
        fails.append(f"[{lang}] MISSING {p.relative_to(ROOT)}"); continue
    data = json.loads(p.read_text(encoding="utf-8"))
    n = kept = ws = 0
    for concept, sets in data.items():
        for setname, items in sets.items():
            for o in items:
                n += 1
                q, a = o.get("q"), o.get("a")
                if not isinstance(q, str) or not isinstance(a, str):
                    fails.append(f"[{lang}] open {concept}/{setname}: non-string q/a"); continue
                if q != q.strip() or a != a.strip(): ws += 1
                if q.strip() and a.strip(): kept += 1
    if ws: fails.append(f"[{lang}] open: {ws} item(s) with untrimmed whitespace")
    print(f"OPEN {lang}: raw={n} kept={kept} ws={ws}")

print()
for w in warns: print("WARN ", w)
for f in fails: print("FAIL ", f)
print(f"\n{'PASS — safe to run' if not fails else f'{len(fails)} FAILURE(S)'}")
sys.exit(1 if fails else 0)
