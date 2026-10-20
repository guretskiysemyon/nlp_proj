#!/usr/bin/env python
"""Translate the new concepts' English MC items (work/en_final.json) into de fr it pt hi es th with the v2 pipeline,
then write them ADDITIVELY into the evaluation database.

    python -m experiments.ml.evaldata.qa_new_translate gen1    --langs ...   # GPU: NLLB beam 4 + NLLB option lists
    <comet venv> -I experiments/ml/evaldata/qa_new_score.py --engine nllb ctx --langs ...           # LaBSE
    python -m experiments.ml.evaldata.qa_new_translate targets --langs ...   # CPU: weak segments / lists
    python -m experiments.ml.evaldata.qa_new_translate gen2    --langs ...   # GPU: NLLB + MADLAD 4-best, MADLAD lists
    <comet venv> -I experiments/ml/evaldata/qa_new_score.py --engine nllb_nbest madlad_nbest madlad_ctx --langs ...
    python -m experiments.ml.evaldata.qa_new_translate build                 # CPU: work/mc_<lang>.json, report
    python -m experiments.ml.evaldata.qa_new_translate write                 # additive write + hash/alignment checks

The pipeline is the one that produced results/ml/data/qa_v2 (cleanup_v2.py), run in one pass on the new items:
  - stage 1 (= v1): every question / option segment with NLLB-200 3.3B (beam 4), and every short option list
    "a; b; c; d" in context (ctx) with NLLB; LaBSE of every candidate;
  - targets (= v2): segments whose best stage-1 candidate has LaBSE < 0.75, lost a number, or (hi, th) has no
    character of the target script, plus every option of an item whose options translate to duplicates;
  - stage 2 (= v2): for the targets NLLB 4-best (beam 8), MADLAD-400 3B 4-best (beam 8), and the item's option list
    by MADLAD; LaBSE;
  - choice over all candidates: numbers kept > target script (hi, th) > LaBSE (cleanup_v2.Chooser2); duplicate
    options -> next-best distinct candidate, else the English source of that option (listed in the report).
The v1 MADLAD-retranslation step (MADLAD beam 4 for LaBSE < 0.75) is subsumed by the MADLAD 4-best of stage 2.
Unlike the 18 concepts, de / fr / es are machine-translated here too (the 18 took them from data/qa_lang).
Items keep the English order and option order, so the correct position (and letter) is the English one.
Caches: results/ml/data/qa_new/work/tr/ (the v1 / v2 caches are not touched).
"""
from __future__ import annotations

import experiments.ml.evaldata.cleanup_v2 as CV2  # noqa: I001  (imports v2.py, which patches common's paths)

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Dict, List

import experiments.ml.evaldata.common as C
import experiments.ml.evaldata.v2 as V2
from experiments.ml.evaldata import build as B
from experiments.ml.evaldata.qa_new_common import (EN_FINAL, NLLB_CODE_ALL, QA_NEW, REPO, SPLITS, TR_LANGS, V2_LANGS,
                                                   WORK, new_concepts)
from experiments.ml.evaldata.tasks import CTX_MAX_WORDS, CTX_SEP

TR_WORK = WORK / "tr"
# redirect the shared evaldata machinery to the new caches (this process only)
C.WORK = TR_WORK
V2.WORK_V2 = TR_WORK
C.NLLB_CODE.update(NLLB_CODE_ALL)        # the same dict object is used by translate.py and cleanup_v2.py
QA_V1 = REPO / "results" / "ml" / "data" / "qa"
QA_V2 = REPO / "results" / "ml" / "data" / "qa_v2"


def en_items() -> dict:
    return json.loads(EN_FINAL.read_text(encoding="utf-8"))


def strings() -> List[str]:
    seen: Dict[str, None] = {}
    for sets in en_items().values():
        for items in sets.values():
            for it in items:
                for s in [it["q"], *it["options"]]:
                    seen.setdefault(s.strip(), None)
    return list(seen)


def segments() -> List[str]:
    seen: Dict[str, None] = {}
    for s in strings():
        for seg, _ in C.segment(s):
            if seg.strip() and C.needs_translation(seg):
                seen.setdefault(seg.strip(), None)
    return list(seen)


def option_lists() -> List[str]:
    seen: Dict[str, None] = {}
    for sets in en_items().values():
        for items in sets.values():
            for it in items:
                opts = [o.strip() for o in it["options"]]
                if CV2.short_list(opts) and any(C.needs_translation(o) for o in opts):
                    seen.setdefault(CTX_SEP.join(opts), None)
    return list(seen)


# --------------------------------------------------------------------------- stage 1 (GPU)
def cmd_gen1(args) -> None:
    from experiments.ml.evaldata import translate as T
    TR_WORK.mkdir(parents=True, exist_ok=True)
    segs, lists = segments(), option_lists()
    print(f"{len(segs)} segments, {len(lists)} option lists per language", flush=True)
    eng = T.Engine("nllb")
    for lang in args.langs:
        for name, todo in (("nllb", segs), ("ctx", lists)):
            path = TR_WORK / f"{name}_{lang}.jsonl"
            done = C.cache_map(path)
            T.run_batches(eng, [s for s in todo if s not in done], lambda b, l=lang: eng.translate(b, l),
                          lambda b, o, p=path: C.append_jsonl(p, [{"src": s, "tr": t} for s, t in zip(b, o)]),
                          f"{name}/{lang}", args.tok_budget, args.max_bs)
    print("DONE", flush=True)


# --------------------------------------------------------------------------- targets (CPU)
def cmd_targets(args) -> None:
    for lang in args.langs:
        ch = CV2.Chooser2(lang)
        segs: Dict[str, None] = {}
        for s in segments():
            best = ch.ranked(s)[0]
            if best["labse"] is None:
                raise SystemExit(f"{lang}: unscored candidate for {s!r}; run qa_new_score.py first")
            if best["labse"] < C.LABSE_THRESHOLD or not best["numbers_kept"] or not CV2.native_ok(lang, best["tr"]):
                segs.setdefault(s, None)
        lists: Dict[str, None] = {}
        dup_items = 0
        for sets in en_items().values():
            for items in sets.values():
                for it in items:
                    opts = [o.strip() for o in it["options"]]
                    tr = [ch.string(o)[0] for o in opts]
                    dup = len(set(tr)) < 4
                    dup_items += dup
                    if dup:
                        for o in opts:
                            for seg, _ in C.segment(o):
                                if seg.strip() and C.needs_translation(seg.strip()):
                                    segs.setdefault(seg.strip(), None)
                    if CV2.short_list(opts) and (dup or any(o in segs for o in opts)) \
                            and any(C.needs_translation(o) for o in opts):
                        lists.setdefault(CTX_SEP.join(opts), None)
        (TR_WORK / f"targets_{lang}.json").write_text(
            json.dumps({"segments": list(segs), "lists": list(lists)}, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{lang}: {len(segs)} target segments, {len(lists)} option lists, {dup_items} stage-1 dup items", flush=True)


def cmd_gen2(args) -> None:
    CV2.cmd_gen(args)   # reads TR_WORK/targets_<lang>.json, writes TR_WORK/{nllb_nbest,madlad_nbest,madlad_ctx}_<lang>


# --------------------------------------------------------------------------- build (CPU)
def _stats(xs: List[float]) -> dict:
    if not xs:
        return {}
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "median": round(statistics.median(xs), 3), "p10": round(q(0.10), 3),
            "mean": round(statistics.mean(xs), 3), "lt_0.75": round(sum(x < 0.75 for x in xs) / len(xs), 4),
            "lt_0.5": round(sum(x < 0.5 for x in xs) / len(xs), 4)}


def cmd_build(args) -> None:
    en = en_items()
    report = {}
    choice_rows = []
    for lang in args.langs:
        ch = CV2.Chooser2(lang)
        out, unfixed, n_dup1, fallback = {}, [], 0, 0
        for c, sets in en.items():
            out[c] = {}
            for sp in SPLITS:
                new_items = []
                for i, it in enumerate(sets[sp]):
                    q, _ = ch.string(it["q"])
                    first = [ch.string(o) for o in it["options"]]
                    n_dup1 += len({t for t, _ in first}) < 4
                    opts = [t for t, _ in B.dedup(ch, it["options"], first)]
                    en_o = [o.strip() for o in it["options"]]
                    ci = en_o.index(it["correct_answer"].strip())
                    for k in range(4):  # still duplicate: English source text for the non-correct one
                        if opts[k] in opts[:k] and en_o[k] not in en_o[:k]:
                            j = k if k != ci else opts.index(opts[k])
                            unfixed.append(f"{lang} {c}/{sp}[{i}]: option {j} kept in English ({en_o[j]!r}; "
                                           f"translations {opts})")
                            opts[j] = en_o[j]
                            fallback += 1
                    new_items.append({"q": q, "correct_answer": opts[ci], "options": opts})
                out[c][sp] = new_items
        (TR_WORK / f"mc_{lang}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        tg = json.loads((TR_WORK / f"targets_{lang}.json").read_text(encoding="utf-8"))["segments"]
        labse = [x["labse"] for x in ch.choice.values() if x["labse"] is not None]
        eng: Dict[str, int] = {}
        for x in ch.choice.values():
            eng[x["engine"]] = eng.get(x["engine"], 0) + 1
        stage1 = []
        for s in ch.choice:
            r1 = [x for x in ch.ranked(s) if x["engine"] in ("nllb", "ctx")]
            if r1 and r1[0]["labse"] is not None:
                stage1.append(r1[0]["labse"])
        report[lang] = {"segments": len(ch.choice), "targets": len(tg), "labse_stage1": _stats(stage1),
                        "labse_final": _stats(labse), "engines": eng, "dup_items_stage1": n_dup1,
                        "dup_items_final": sum(len(set(it["options"])) < 4 for s in out.values() for v in s.values()
                                               for it in v),
                        "english_fallback_options": fallback, "unfixed": unfixed,
                        "numbers_lost": sum(not x["numbers_kept"] for x in ch.choice.values()),
                        "no_script": sum(not CV2.native_ok(lang, x["tr"]) for x in ch.choice.values())}
        choice_rows += list(ch.choice.values())
        print(lang, json.dumps({k: v for k, v in report[lang].items() if k != "unfixed"}), flush=True)
    (TR_WORK / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    with (TR_WORK / "translation_choice.jsonl").open("w", encoding="utf-8") as f:
        for r in choice_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- additive write
def concept_hashes(path: Path) -> Dict[str, str]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {c: hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True).encode()).hexdigest() for c, v in d.items()}


def _clean_en(en: dict) -> dict:
    return {c: {sp: [{"q": it["q"], "correct_answer": it["correct_answer"], "options": it["options"]} for it in v]
                for sp, v in sets.items()} for c, sets in en.items()}


def cmd_write(args) -> None:
    new = {"en": _clean_en(en_items())}
    for lang in TR_LANGS:
        new[lang] = json.loads((TR_WORK / f"mc_{lang}.json").read_text(encoding="utf-8"))
    order = new_concepts()
    for lang, d in new.items():
        assert list(d) == order, lang
        for c in order:
            assert list(d[c]) == list(SPLITS) and all(len(d[c][sp]) == 50 for sp in SPLITS), (lang, c)
    targets = [(QA_V1, l) for l in ("en",) + TR_LANGS] + [(QA_V2, l) for l in V2_LANGS]
    before = {}
    for root, lang in targets:
        p = root / f"mc_questions_{lang}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        clash = set(d) & set(order)
        if clash and not args.replace_new:
            raise SystemExit(f"{p}: already has {sorted(clash)} (use --replace-new to overwrite only these)")
        before[p] = {c: h for c, h in concept_hashes(p).items() if c not in order}
        for c in order:
            d[c] = new[lang][c]
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
    ok = True
    for p, h in before.items():
        after = concept_hashes(p)
        same = all(after.get(c) == v for c, v in h.items())
        ok &= same
        print(f"{p.relative_to(REPO)}: {len(after)} concepts; existing {len(h)} unchanged: {same}")
    for lang in V2_LANGS:
        a, b = (json.loads((r / f"mc_questions_{lang}.json").read_text(encoding="utf-8")) for r in (QA_V1, QA_V2))
        ok &= all(a[c] == b[c] for c in order)
    if not ok:
        raise SystemExit("hash / v1-v2 identity check FAILED")
    print("hash check passed; qa/ and qa_v2/ identical for the new concepts (it, pt, hi, th)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen1", "targets", "gen2", "build", "write"])
    ap.add_argument("--langs", nargs="+", default=list(TR_LANGS))
    ap.add_argument("--models", nargs="+", default=["nllb", "madlad"])
    ap.add_argument("--tok-budget", type=int, default=1536)
    ap.add_argument("--max-bs", type=int, default=64)
    ap.add_argument("--replace-new", action="store_true", help="write: overwrite the 20 new concepts if present")
    args = ap.parse_args()
    if args.cmd == "gen2" and args.tok_budget == 1536:
        args.tok_budget, args.max_bs = 768, 32   # cleanup_v2 gen defaults (beam 8, 4 returned)
    t0 = time.time()
    {"gen1": cmd_gen1, "targets": cmd_targets, "gen2": cmd_gen2, "build": cmd_build, "write": cmd_write}[args.cmd](args)
    print(f"{args.cmd}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
