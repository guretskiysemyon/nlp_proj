#!/usr/bin/env python
"""v2 cleanup of the machine-translated MC sets (it, pt, hi, th) -> results/ml/data/qa_v2/.

v1 (results/ml/data/qa/mc_questions_<lang>.json, built 2026-10-18 23:53) is read-only here.
Problems left in v1: options of one item that translate to the same string, Hindi / Thai
strings left in English (LaBSE rewards a copy of the source), and low-LaBSE segments.

    python -m experiments.ml.evaldata.cleanup_v2 targets   # CPU: copy v1 caches, pick targets
    python -m experiments.ml.evaldata.cleanup_v2 gen       # GPU: NLLB + MADLAD n-best, MADLAD lists
    EVALDATA_V2=1 <comet venv> -I experiments/ml/evaldata/score.py \
        --engine nllb_nbest madlad_nbest madlad_ctx --metrics labse
    python -m experiments.ml.evaldata.cleanup_v2 build     # CPU: qa_v2 files + CLEANUP.md

Targets per language: every segment whose v1 choice has LaBSE < 0.75, lost a number, or (hi,
th) has no character of the target script; every option of an item with duplicate options.
For those, new candidates: 4-best of NLLB (beam 8) and of MADLAD (beam 8), and the item's
option list translated as one line by MADLAD (short options only). Choice over ALL candidates
(v1 ones included): numbers kept > target script present (hi, th) > LaBSE. Duplicates left
after that: the next-best distinct candidate; if none, the English source text (listed in
CLEANUP.md). Items keep their order, keys and option positions, so letters stay aligned.
"""
from __future__ import annotations

import experiments.ml.evaldata.v2 as V2  # noqa: I001  (must come first: patches paths)

import argparse
import json
import re
import shutil
import statistics
import time
from typing import Dict, List, Optional, Tuple

from experiments.ml.evaldata import build as B
from experiments.ml.evaldata.common import (LABSE_THRESHOLD, MT_LANGS, NLLB_CODE, REPO, SPLITS, append_jsonl,
                                            en_mc, load_jsonl, needs_translation, normalize_digits,
                                            numbers_kept, segment)
from experiments.ml.evaldata.tasks import CTX_MAX_WORDS, CTX_SEP

QA_V1 = REPO / "results" / "ml" / "data" / "qa"
SCRIPT = {"hi": re.compile(r"[ऀ-ॿ]"), "th": re.compile(r"[฀-๿]")}
NBEST, NBEAMS = 4, 8


def native_ok(lang: str, tr: str) -> bool:
    return lang not in SCRIPT or bool(SCRIPT[lang].search(tr))


def v1_choice() -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {l: {} for l in MT_LANGS}
    with (QA_V1 / "translation_choice.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["lang"]][r["src"]] = r
    return out


def v1_mc(lang: str) -> dict:
    return json.loads((QA_V1 / f"mc_questions_{lang}.json").read_text(encoding="utf-8"))


def short_list(opts: List[str]) -> bool:
    return all(len(o.split()) <= CTX_MAX_WORDS and ";" not in o and "\n" not in o for o in opts)


# --------------------------------------------------------------------------- targets
def cmd_targets(_args) -> None:
    V2.WORK_V2.mkdir(parents=True, exist_ok=True)
    for p in V2.WORK_V1.glob("*.jsonl"):
        dst = V2.WORK_V2 / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
    choice, en = v1_choice(), en_mc()
    for lang in MT_LANGS:
        segs: Dict[str, None] = {}
        for s, r in choice[lang].items():
            if (r["labse"] is not None and r["labse"] < LABSE_THRESHOLD) or not r["numbers_kept"] \
                    or not native_ok(lang, r["tr"]):
                segs.setdefault(s, None)
        lists: Dict[str, None] = {}
        dup_items = 0
        tr = v1_mc(lang)
        for c, sets in en.items():
            for sp, items in sets.items():
                for it, t in zip(items, tr[c][sp]):
                    opts = [o.strip() for o in it["options"]]
                    dup = len(set(t["options"])) < 4
                    dup_items += dup
                    if dup:
                        for o in opts:
                            for seg, _ in segment(o):
                                if seg.strip() and needs_translation(seg.strip()):
                                    segs.setdefault(seg.strip(), None)
                    if short_list(opts) and (dup or any(o in segs for o in opts)) \
                            and any(needs_translation(o) for o in opts):
                        lists.setdefault(CTX_SEP.join(opts), None)
        (V2.WORK_V2 / f"targets_{lang}.json").write_text(
            json.dumps({"segments": list(segs), "lists": list(lists)}, ensure_ascii=False, indent=0),
            encoding="utf-8")
        print(f"{lang}: {len(segs)} target segments, {len(lists)} option lists, {dup_items} v1 dup items")


# --------------------------------------------------------------------------- gen (GPU)
def _batches(texts: List[str], lens: Dict[str, int], budget: int, max_bs: int):
    i = 0
    while i < len(texts):
        j = i + 1
        while j < len(texts) and j - i < max_bs and (j - i + 1) * lens[texts[j]] <= budget:
            j += 1
        yield texts[i:j]
        i = j


def _run(model, tok, texts: List[str], prefix_fn, gen_kw, n_ret: int, write, tag: str,
         budget: int, max_bs: int) -> None:
    import torch
    if not texts:
        print(f"[{tag}] nothing to do", flush=True)
        return
    lens = {s: len(tok(prefix_fn(s))["input_ids"]) for s in texts}
    texts = sorted(texts, key=lambda s: lens[s])
    t0, n = time.time(), 0
    dev = next(model.parameters()).device
    def gen(batch, kw):
        enc = tok([prefix_fn(s) for s in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(dev)
        n_in = int(enc["attention_mask"].sum(1).max())
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=min(512, 2 * n_in + 16), num_return_sequences=n_ret, **kw)
        return [normalize_digits(t.strip()) for t in tok.batch_decode(out, skip_special_tokens=True)]

    def safe(batch, kw):
        """OOM: split the batch; a single long input falls back to a smaller beam, then is skipped
        (it is only an extra candidate). Retries happen outside the except block so the failed
        attempt's tensors are freed first."""
        try:
            return gen(batch, kw)
        except torch.cuda.OutOfMemoryError:
            pass
        torch.cuda.empty_cache()
        if len(batch) > 1:
            h = len(batch) // 2
            return safe(batch[:h], kw) + safe(batch[h:], kw)
        if kw.get("num_beams", 1) > n_ret:
            return safe(batch, {**kw, "num_beams": n_ret})
        print(f"  [{tag}] OOM, skipped: {batch[0][:80]!r}", flush=True)
        return [""] * n_ret

    for batch in _batches(texts, lens, budget, max_bs):
        dec = safe(batch, gen_kw)
        write(batch, [dec[k * n_ret:(k + 1) * n_ret] for k in range(len(batch))])
        n += len(batch)
        if n % 256 < len(batch) or n == len(texts):
            print(f"  [{tag}] {n}/{len(texts)} {n / (time.time() - t0):.1f}/s", flush=True)


def _split_list(src: str, out: str) -> Optional[List[str]]:
    srcs = src.split(CTX_SEP)
    t = out.replace("；", ";").strip().rstrip(".।").strip()
    parts = [p.strip() for p in t.split(";")]
    if len(parts) != len(srcs) and not any("," in x for x in srcs):
        parts = [p.strip() for p in t.replace("、", ",").split(",")]
    if len(parts) != len(srcs) or not all(parts):
        return None
    return [p[0].upper() + p[1:] if s[:1].isupper() and p[:1].islower() else p for s, p in zip(srcs, parts)]


def cmd_gen(args) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tg = {l: json.loads((V2.WORK_V2 / f"targets_{l}.json").read_text(encoding="utf-8")) for l in args.langs}

    def done_src(path):
        return {r["src"] for r in load_jsonl(path)}

    def nbest_writer(path):
        return lambda b, outs: append_jsonl(path, [{"src": s, "tr": t} for s, ts in zip(b, outs)
                                                   for t in dict.fromkeys(ts) if t])

    if "nllb" in args.models:
        tok = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True).cuda().eval()
        for l in args.langs:
            path = V2.WORK_V2 / f"nllb_nbest_{l}.jsonl"
            todo = [s for s in tg[l]["segments"] if s not in done_src(path)]
            kw = {"num_beams": NBEAMS, "forced_bos_token_id": tok.convert_tokens_to_ids(NLLB_CODE[l])}
            _run(model, tok, todo, lambda s: s, kw, NBEST, nbest_writer(path), f"nllb_nbest/{l}",
                 args.tok_budget, args.max_bs)
        del model
        torch.cuda.empty_cache()

    if "madlad" in args.models:
        tok = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/madlad400-3b-mt", torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True).cuda().eval()
        for l in args.langs:
            pre = lambda s, l=l: f"<2{l}> {s}"
            path = V2.WORK_V2 / f"madlad_nbest_{l}.jsonl"
            todo = [s for s in tg[l]["segments"] if s not in done_src(path)]
            _run(model, tok, todo, pre, {"num_beams": NBEAMS}, NBEST, nbest_writer(path), f"madlad_nbest/{l}",
                 args.tok_budget, args.max_bs)
            raw = V2.WORK_V2 / f"madlad_ctx_raw_{l}.jsonl"
            per = V2.WORK_V2 / f"madlad_ctx_{l}.jsonl"
            todo = [s for s in tg[l]["lists"] if s not in done_src(raw)]

            def write_lists(b, outs, raw=raw, per=per):
                rows = []
                for s, (o,) in zip(b, outs):
                    parts = _split_list(s, o)
                    if parts:
                        rows += [{"src": a, "tr": p} for a, p in zip(s.split(CTX_SEP), parts) if needs_translation(a)]
                append_jsonl(per, rows)
                append_jsonl(raw, [{"src": s, "tr": o} for s, (o,) in zip(b, outs)])

            _run(model, tok, todo, pre, {"num_beams": 4}, 1, write_lists, f"madlad_ctx/{l}",
                 args.tok_budget * 2, args.max_bs)
    print("DONE", flush=True)


# --------------------------------------------------------------------------- build
class Chooser2(B.Chooser):
    """Numbers kept > target script present > LaBSE (unscored candidates last)."""

    def ranked(self, s: str) -> List[dict]:
        if s not in self.cand["nllb"]:
            raise B.Missing(s)
        q = [{"engine": e, "tr": t, **self._q(e, s, t), "numbers_kept": numbers_kept(s, t)}
             for e in B.ENGINES for t in self.cand[e].get(s, [])]
        q = [x for x in q if x["tr"].strip()]
        return sorted(q, key=lambda x: (x["labse"] is not None, x["numbers_kept"],
                                        native_ok(self.lang, x["tr"]), x["labse"] or 0.0), reverse=True)


def _stats(xs: List[float]) -> str:
    if not xs:
        return "n/a"
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return (f"median {statistics.median(xs):.3f}, p10 {q(0.10):.3f}, mean {statistics.mean(xs):.3f}, "
            f"< 0.75: {sum(x < 0.75 for x in xs) / len(xs):.1%}, < 0.5: {sum(x < 0.5 for x in xs) / len(xs):.1%}")


def cmd_build(args) -> None:
    t0 = time.time()
    V2.QA_V2.mkdir(parents=True, exist_ok=True)
    en, before = en_mc(), v1_choice()
    rep = ["# QA v2 cleanup: it, pt, hi, th MC sets", "",
           "`results/ml/data/qa_v2/mc_questions_<lang>.json`: same schema, concept keys, splits, item order and",
           "option order as v1 (`results/ml/data/qa/`) and as en/de/es/fr, so item *i* and its correct letter",
           "stay aligned across languages; `correct_answer` is the translated option at the English correct",
           "position. en/de/es/fr are unchanged and not copied here.", "",
           "## What was done", "",
           "v1 = NLLB-200 3.3B per string + option lists translated in context (\"a; b; c; d\") + MADLAD-400",
           "for LaBSE < 0.75, chosen by LaBSE. Remaining v1 problems: items with two identical translated",
           "options, Hindi/Thai strings left in English (a copy of the source gets LaBSE ~1, so v1 preferred",
           "it), and segments with LaBSE < 0.75. v2 targets every segment that is low (LaBSE < 0.75), lost",
           "a number, or (hi, th) has no character of the target script, plus every option of a duplicate",
           "item. New candidates for them: 4-best of NLLB (beam 8), 4-best of MADLAD (beam 8), and the item's",
           "option list translated in one line by MADLAD. All candidates (v1 ones included) are ranked by",
           "numbers kept > target script present (hi, th) > LaBSE. Duplicates still left: next-best distinct",
           "candidate, else the English source text of that option (listed below).", "",
           "LaBSE caveat: cosine of LaBSE embeddings is a weak judge for single words and names (correct",
           "transliterations such as it 'Isocrate' score ~0.7), so 'LaBSE < 0.75' over-counts real errors,",
           "and it cannot see register or nuance. No Kiwi (gated model) and no human check.", ""]
    summary = ["## Summary", "", "| lang | items | items changed | strings changed | target segments | "
               "segments changed | dup items v1 -> v2 | English fallback options | no-script strings v1 -> v2 |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    dist = ["## LaBSE per unique segment, v1 -> v2", ""]
    unfixed: List[str] = []
    for lang in MT_LANGS:
        ch = Chooser2(lang)
        v1 = v1_mc(lang)
        out, n_items, items_changed, str_changed, dup_v1, dup_v2, fallback = {}, 0, 0, 0, 0, 0, 0
        for c, sets in en.items():
            out[c] = {}
            for sp, items in sets.items():
                new_items = []
                for i, (it, old) in enumerate(zip(items, v1[c][sp])):
                    q, _ = ch.string(it["q"])
                    tr_opts = B.dedup(ch, it["options"], [ch.string(o) for o in it["options"]])
                    opts = [t for t, _ in tr_opts]
                    ci = [o.strip() for o in it["options"]].index(it["correct_answer"].strip())
                    en_o = [o.strip() for o in it["options"]]
                    for k in range(4):  # still duplicate: English source for the non-correct one
                        if opts[k] in opts[:k] and en_o[k] not in en_o[:k]:
                            j = k if k != ci else opts.index(opts[k])
                            unfixed.append(f"{lang} {c}/{sp}[{i}]: option {j} kept in English "
                                           f"({en_o[j]!r}; translations {opts})")
                            opts[j] = en_o[j]
                            fallback += 1
                    ci = [o.strip() for o in it["options"]].index(it["correct_answer"].strip())
                    new = {"q": q, "correct_answer": opts[ci], "options": opts}
                    n_items += 1
                    dup_v1 += len(set(old["options"])) < 4
                    dup_v2 += len(set(opts)) < 4
                    ch_strings = (q != old["q"]) + sum(a != b for a, b in zip(opts, old["options"]))
                    str_changed += ch_strings
                    items_changed += ch_strings > 0
                    new_items.append(new)
                out[c][sp] = new_items
        assert all(list(out[c]) == list(SPLITS) for c in out)
        (V2.QA_V2 / f"mc_questions_{lang}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")
        tg = json.loads((V2.WORK_V2 / f"targets_{lang}.json").read_text(encoding="utf-8"))["segments"]
        b_all = [r["labse"] for r in before[lang].values() if r["labse"] is not None]
        a_all = [c_["labse"] for s, c_ in ch.choice.items() if c_["labse"] is not None]
        b_tg = [before[lang][s]["labse"] for s in tg if s in before[lang] and before[lang][s]["labse"] is not None]
        a_tg = [ch.choice[s]["labse"] for s in tg if s in ch.choice and ch.choice[s]["labse"] is not None]
        seg_changed = sum(1 for s in tg if s in ch.choice and s in before[lang] and ch.choice[s]["tr"] != before[lang][s]["tr"])
        eng = {}
        for s in tg:
            if s in ch.choice and s in before[lang] and ch.choice[s]["tr"] != before[lang][s]["tr"]:
                eng[ch.choice[s]["engine"]] = eng.get(ch.choice[s]["engine"], 0) + 1
        ns_b = sum(not native_ok(lang, r["tr"]) for r in before[lang].values())
        ns_a = sum(not native_ok(lang, c_["tr"]) for c_ in ch.choice.values())
        summary.append(f"| {lang} | {n_items} | {items_changed} | {str_changed} | {len(tg)} | {seg_changed} | "
                       f"{dup_v1} -> {dup_v2} | {fallback} | " + (f"{ns_b} -> {ns_a}" if lang in SCRIPT else "n/a") + " |")
        dist += [f"**{lang}** (changed target segments by engine: {eng})", "",
                 f"- all segments, v1: {_stats(b_all)}", f"- all segments, v2: {_stats(a_all)}",
                 f"- target segments, v1: {_stats(b_tg)}", f"- target segments, v2: {_stats(a_tg)}", ""]
        with (V2.QA_V2 / "translation_choice.jsonl").open("a" if lang != MT_LANGS[0] else "w", encoding="utf-8") as f:
            for c_ in ch.choice.values():
                f.write(json.dumps(c_, ensure_ascii=False) + "\n")
        print(f"{lang}: {items_changed}/{n_items} items changed, dup {dup_v1}->{dup_v2}, fallback {fallback}", flush=True)
    rep += summary + ["", "'items changed' = items with at least one string (question or option) different from v1.", ""]
    rep += dist
    rep += ["## Items not fully fixed", ""] + ([f"- {u}" for u in unfixed] or ["- none"]) + [""]
    rep += [f"Build wall time: {time.time() - t0:.0f} s (CPU). GPU wall time: see the ml_eval_v2 job logs.", ""]
    (V2.QA_V2 / "CLEANUP.md").write_text("\n".join(rep), encoding="utf-8")
    print("wrote", V2.QA_V2 / "CLEANUP.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["targets", "gen", "build"])
    ap.add_argument("--langs", nargs="+", default=list(MT_LANGS))
    ap.add_argument("--models", nargs="+", default=["nllb", "madlad"])
    ap.add_argument("--tok-budget", type=int, default=768)
    ap.add_argument("--max-bs", type=int, default=32)
    args = ap.parse_args()
    {"targets": cmd_targets, "gen": cmd_gen, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
