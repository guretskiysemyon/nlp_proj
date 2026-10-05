#!/usr/bin/env python3
"""One-off repair of the translated QA sets.

Fixes, per language:
  1. strip() on every q / a / correct_answer / option        (whitespace)
  2. option-count != 4                                        (crashes prepare_mc_items)
  3. duplicate options                                        (mis-grades: the correct
     answer maps to the LAST matching letter, so a model picking the first copy is
     scored wrong)
  4. one isolated mojibake in the German file

Applied to both the flat aggregates and the per-concept source files, so the defects
do not come back if the aggregates are regenerated.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA   = ROOT / "data" / "qa_lang"
LANGS = ["de", "es", "fr", "ru"]

# (lang, concept, set, item_idx, option_idx) -> replacement
# Every replacement targets a DISTRACTOR whose translation collided with another
# option. The English term each one renders is given in the comment.
DUP_FIX = {
 ("de","Baseball","SimdomQA_train",25,2): "Baseballschläger",        # Bat
 ("de","Baseball","SimdomQA_train",43,2): "Baseballschläger",        # Bat
 ("de","Gun","QA_train",8,3):             "Geräuschdämpfer",         # Silencer
 ("de","Halloween","SimdomQA_train",24,1):"Glocke der Freiheit",     # Freedom Bell
 ("de","Valentine's Day","SimdomQA_test",40,2): "Glucksen",          # Chuckle

 ("es","Gambling","QA_train",13,2):       "Valor facial",            # Face value
 ("es","Halloween","SimdomQA_train",24,1):"Campana de la Emancipación",  # Freedom Bell
 ("es","Valentine's Day","SimdomQA_test",48,0): "Boxing Day",        # Boxing Day

 ("fr","Baseball","SimdomQA_train",8,2):  "Ligne latérale",          # Sideline
 ("fr","Gambling","QA_train",12,3):       "Plaque",                  # Token
 ("fr","Gambling","QA_train",19,1):       "Donneur",                 # Dealer
 ("fr","Gun","SimdomQA_train",16,2):      "Proue",                   # Larboard (see note)
 ("fr","Gun","SimdomQA_train",29,1):      "Palettes",                # Vanes
 ("fr","Halloween","SimdomQA_train",11,1):"Trèfle à quatre feuilles",# Clover
 ("fr","Halloween","SimdomQA_test",34,2): "Papa Noël",               # Father Christmas

 ("ru","Baseball","SimdomQA_train",8,2):  "Линия аута",              # Sideline
 ("ru","Baseball","SimdomQA_train",43,1): "Теннисная ракетка",       # Racquet
 ("ru","Harry Potter","QA_train",18,3):   "Короста",                 # Scabbers
 ("ru","Uranium","SimdomQA_test",8,2):    "Оксид лития",             # Lithia
}

# (lang, concept, set, item_idx) -> (options, correct_answer)
# fr Harry Potter had 5 options: a hallucinated Chinese string prepended as the
# "correct" answer. Drop it; the remaining four align 1:1 with the English item.
COUNT_FIX = {
 ("fr","Harry Potter","QA_train",42): (
    ["Hongrois à pointes","Suédois à museau court",
     "Balle de feu chinoise","Vert gallois commun"],
    "Hongrois à pointes"),
}

MOJIBAKE = {"de": {"Pekannubkuchen": "Pekannusskuchen"}}

log = []

def fix_items(lang, concept, setname, items, is_mc):
    for i, o in enumerate(items):
        for k in ("q", "a", "correct_answer"):
            if isinstance(o.get(k), str) and o[k] != o[k].strip():
                o[k] = o[k].strip(); log.append(f"strip  {lang} {concept}/{setname}/{i} .{k}")
        for src, dst in MOJIBAKE.get(lang, {}).items():
            for k in ("q", "a", "correct_answer"):
                if isinstance(o.get(k), str) and src in o[k]:
                    o[k] = o[k].replace(src, dst); log.append(f"spell  {lang} {concept}/{setname}/{i} .{k}")
        if not is_mc:
            continue

        ck = (lang, concept, setname, i)
        if ck in COUNT_FIX:
            opts, corr = COUNT_FIX[ck]
            log.append(f"COUNT  {lang} {concept}/{setname}/{i}: {len(o['options'])} -> 4 options, "
                       f"correct {o['correct_answer']!r} -> {corr!r}")
            o["options"], o["correct_answer"] = list(opts), corr
            continue

        o["options"] = [x.strip() if isinstance(x, str) else x for x in o["options"]]
        for src, dst in MOJIBAKE.get(lang, {}).items():
            o["options"] = [x.replace(src, dst) if isinstance(x, str) else x for x in o["options"]]
        for j in range(len(o["options"])):
            dk = (lang, concept, setname, i, j)
            if dk in DUP_FIX:
                old = o["options"][j]
                o["options"][j] = DUP_FIX[dk]
                log.append(f"DUP    {lang} {concept}/{setname}/{i} opt[{j}]: {old!r} -> {DUP_FIX[dk]!r}")

def fix_aggregate(lang, kind):
    src = QA / "translated" / lang / f"{kind}.json"
    dst = QA / f"{kind}_{lang}.json"
    path = src if src.exists() else dst
    data = json.loads(path.read_text(encoding="utf-8"))
    for concept, sets in data.items():
        for setname, items in sets.items():
            fix_items(lang, concept, setname, items, kind == "mc_questions")
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return src, dst

def fix_per_concept(lang, kind):
    d = QA / "translated" / lang / kind
    if not d.is_dir():
        return 0
    n = 0
    for f in sorted(d.glob("*.json")):
        obj = json.loads(f.read_text(encoding="utf-8"))
        concept = obj.get("concept")
        for setname, items in obj.get("splits", {}).items():
            fix_items(lang, concept, setname, items, kind == "mc_questions")
        f.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n += 1
    return n

if __name__ == "__main__":
    moved = []
    for lang in LANGS:
        for kind in ("mc_questions", "open_questions"):
            src, dst = fix_aggregate(lang, kind)
            if src.exists() and src != dst:
                src.unlink() if False else moved.append(src)   # left for the shell to move
            fix_per_concept(lang, kind)
    for line in log:
        print(line)
    print(f"\n{len(log)} edits applied")
