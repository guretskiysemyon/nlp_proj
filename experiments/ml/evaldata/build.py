#!/usr/bin/env python
"""Assemble the evaluation files from the translation and score caches.

    python -m experiments.ml.evaldata.build [--partial] [--no-report]

Writes
  results/ml/data/qa/mc_questions_<lang>.json   all 8 languages, keys / items / options in
        the order of data/mc_questions.json; ``correct_answer`` is the translated text of the
        same option position, so the correct letter after the language-aligned shuffle is
        unchanged.
  results/ml/data/mmlu/th.json                   NLLB(/MADLAD) translation of mmlu/en.json
  results/ml/data/qa/translation_choice.jsonl    per (lang, segment): chosen engine, scores, number check
  results/ml/data/qa/QUALITY.md                  quality summary (skipped with --no-report or if unscored)

Choice per segment: NLLB, unless a MADLAD retranslation exists (made only for a lost number
or a low score, common.needs_retranslation); then the candidate that keeps all numbers
wins, then the higher Kiwi (LaBSE while Kiwi is missing). Rebuilding after a new score
column is added may change the choice; no retranslation is needed.
With --partial, a concept is written for an MT language only when every one of its
segments is translated (the file then holds fewer concepts); without it, a missing
segment is an error.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import unicodedata
from typing import Dict, List, Optional, Tuple

from experiments.ml.evaldata.common import (KIWI_THRESHOLD, LABSE_THRESHOLD, LANGS, MMLU_DIR, MT_LANGS, QA_DIR, REPO,
                                            ENGINES, METRICS, SPLITS, candidates, en_mc, needs_translation,
                                            numbers_kept, scores, segment)




class Missing(Exception):
    pass


_END_PUNCT = (".", "!", "?", "।", ":", ";")


def clean(src: str, tr: str) -> str:
    """NLLB often appends ' .' to single words / short options: drop a final period (or
    Devanagari danda) when the English source has no final punctuation."""
    # NLLB writes Thai SARA AM as NIKHAHIT + SARA AA; recompose, then NFC
    t = unicodedata.normalize("NFC", tr.strip().replace("\u0e4d\u0e32", "\u0e33"))
    if not src.rstrip().endswith(_END_PUNCT + (")", '"', "'")):
        while t.endswith((".", "।")) and not t.endswith(".."):
            t = t[:-1].rstrip()
    return t or tr.strip()


class Chooser:
    """Picks the translation of each segment of one language and remembers why."""

    def __init__(self, lang: str):
        self.lang = lang
        self.cand = {e: candidates(e, lang) for e in ENGINES}
        self.choice: Dict[str, dict] = {}

    def _q(self, e: str, s: str, t: str) -> Dict[str, Optional[float]]:
        return {m: scores(m, e, self.lang).get((s, t)) for m in METRICS}

    def ranked(self, s: str) -> List[dict]:
        """All candidates of segment ``s``, best first: numbers kept, then Kiwi if every
        candidate has it, else LaBSE if every candidate has it, else NLLB first."""
        if s not in self.cand["nllb"]:
            raise Missing(s)
        q = [{"engine": e, "tr": t, **self._q(e, s, t), "numbers_kept": numbers_kept(s, t)}
             for e in ENGINES for t in self.cand[e].get(s, [])]
        for metric in ("kiwi", "labse"):
            if all(x[metric] is not None for x in q):
                return sorted(q, key=lambda x: (x["numbers_kept"], x[metric]), reverse=True)
        return q

    def seg(self, s: str, avoid: Tuple[str, ...] = ()) -> Dict[str, Optional[float]]:
        """Chosen translation of segment ``s`` -> {"tr", <metric>: value or None}; ``avoid``
        skips candidates equal to other options of the same item, if an alternative exists."""
        if s in self.choice and not avoid:
            return self.choice[s]
        q = self.ranked(s)
        best = next((x for x in q if clean(s, x["tr"]) not in avoid), q[0])
        rec = {"lang": self.lang, "src": s, **best, "n_candidates": len(q),
               "nllb": {k: v for k, v in q[0].items() if k != "engine"} if q[0]["engine"] == "nllb"
               else next({k: v for k, v in x.items() if k != "engine"} for x in q if x["engine"] == "nllb"),
               "retranslated": any(x["engine"] == "madlad" for x in q)}
        if not avoid:
            self.choice[s] = rec
        return rec

    def string(self, s: str, avoid: Tuple[str, ...] = ()) -> Tuple[str, Dict[str, Optional[float]]]:
        """Translated string and its length-weighted scores per metric (None if copied/unscored)."""
        out = []
        acc = {m: [0.0, 0] for m in METRICS}
        for piece, sep in segment(s.strip()):
            core = piece.strip()
            if not core or not needs_translation(core):
                out.append(piece + sep)
                continue
            c = self.seg(core, avoid)
            lead = piece[:len(piece) - len(piece.lstrip())]
            trail = piece[len(piece.rstrip()):]
            out.append(lead + clean(core, c["tr"]) + trail + sep)
            for m in METRICS:
                if c[m] is not None:
                    acc[m][0] += c[m] * len(core)
                    acc[m][1] += len(core)
        return "".join(out), {m: (v / w if w else None) for m, (v, w) in acc.items()}


def dedup(ch: Chooser, en_opts: List[str], tr_opts: List[tuple]) -> List[tuple]:
    """Two different English options with the same translation: retranslate the later one
    with its next-best candidate that differs from the other options (if any)."""
    tr_opts = list(tr_opts)
    for i in range(len(tr_opts)):
        others = tuple(t for j, (t, _) in enumerate(tr_opts) if j != i)
        if tr_opts[i][0] in others and en_opts[i].strip() not in [o.strip() for o in en_opts[:i]]:
            if any(tr_opts[i][0] == tr_opts[j][0] for j in range(i)):
                tr_opts[i] = ch.string(en_opts[i], avoid=others)
    return tr_opts


def build_mc(lang: str, ch: Optional[Chooser], partial: bool, rows: list) -> Tuple[dict, List[str]]:
    en = en_mc()
    if lang == "en":
        return en, []
    if lang not in MT_LANGS:
        src = json.loads((REPO / "data" / "qa_lang" / f"mc_questions_{lang}.json").read_text(encoding="utf-8"))
        return {c: {s: src[c][s] for s in en[c]} for c in en}, []
    out, problems = {}, []
    for concept, sets in en.items():
        new, new_rows, new_probs = {}, [], []
        try:
            for split, items in sets.items():
                new_items = []
                for i, it in enumerate(items):
                    q, sq = ch.string(it["q"])
                    tr_opts = [ch.string(o) for o in it["options"]]
                    tr_opts = dedup(ch, it["options"], tr_opts)
                    opts = [t for t, _ in tr_opts]
                    ci = [o.strip() for o in it["options"]].index(it["correct_answer"].strip())
                    if len(set(opts)) < 4:
                        new_probs.append(f"{lang} {concept}/{split}[{i}]: duplicate translated options {opts}")
                    new_items.append({"q": q, "correct_answer": opts[ci], "options": opts})
                    base = {"lang": lang, "set": "MC", "split": split, "concept": concept}
                    new_rows.append({**base, "kind": "q", **sq, "en": it["q"], "tr": q})
                    new_rows += [{**base, "kind": "opt", **so, "en": o, "tr": t}
                                 for o, (t, so) in zip(it["options"], tr_opts)]
                new[split] = new_items
        except Missing:
            if not partial:
                raise
            continue
        out[concept] = new
        rows += new_rows
        problems += new_probs
    return out, problems


def build_mmlu_th(ch: Chooser, rows: list) -> dict:
    en = json.loads((MMLU_DIR / "en.json").read_text(encoding="utf-8"))
    out, new_rows = {}, []
    for split, items in en.items():
        res = []
        for it in items:
            q, sq = ch.string(it["question"])
            base = {"lang": "th", "set": "MMLU", "split": split, "concept": it["subject"]}
            new_rows.append({**base, "kind": "q", **sq, "en": it["question"], "tr": q})
            tr_opts = dedup(ch, list(it["options"].values()), [ch.string(o) for o in it["options"].values()])
            opts = {}
            for (L, o), (t, so) in zip(it["options"].items(), tr_opts):
                opts[L] = t
                new_rows.append({**base, "kind": "opt", **so, "en": o, "tr": t})
            if len(set(opts.values())) < 4:
                print(f"WARNING th MMLU {it['id']}: duplicate translated options {opts}")
            res.append({"id": it["id"], "subject": it["subject"], "question": q, "options": opts,
                        "answer": it["answer"]})
        out[split] = res
    rows += new_rows
    return out


def _med(xs: List[float]) -> str:
    return f"{statistics.median(xs):.3f}" if xs else "n/a"


def report(rows: List[dict], choosers: Dict[str, Chooser], problems: List[str]) -> str:
    have_kiwi = any(r["kiwi"] is not None for r in rows)
    L = ["# Translation quality: MC (it, pt, hi, th) and Thai MMLU", "",
         "Machine translation from English with facebook/nllb-200-3.3B (fp16, beam 4); every question and",
         "option translated separately (long MMLU questions per line / sentence). Strings without letters",
         "(years, numbers, symbols) are copied verbatim and not scored; Devanagari / Thai digits in the",
         "output are mapped to ASCII. de / es / fr come from data/qa_lang (already checked); en is",
         "data/mc_questions.json.", "",
         "Quality columns (reference-free):",
         "- **kiwi**: COMET-Kiwi (Unbabel/wmt22-cometkiwi-da)" + ("" if have_kiwi else " -- *not computed yet* (gated model, license pending)"),
         "- **labse**: cosine of LaBSE embeddings of source and translation",
         "- **chrF-bt**: chrF (0-1) of the NLLB round-trip back-translation against the English source", "",
         f"Retranslation rule: a segment whose NLLB output lost a number of the source, or has Kiwi < {KIWI_THRESHOLD}",
         f"(LaBSE < {LABSE_THRESHOLD} where Kiwi is unavailable), is retranslated with google/madlad400-3b-mt",
         "(fp32, beam 4); the candidate that keeps all numbers wins, then the higher Kiwi (else LaBSE).",
         "String-level scores = length-weighted mean over the string's segments.", "",
         "## Medians per language and split (questions and options pooled, chosen translation)", "",
         "| lang | set | split | n strings | kiwi | labse | chrF-bt | share kiwi < 0.6 | share labse < 0.75 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        if r["labse"] is not None or r["kiwi"] is not None:
            groups.setdefault((r["lang"], r["set"], r["split"]), []).append(r)
    order = {l: i for i, l in enumerate(MT_LANGS)}
    for (lang, st, sp), rs in sorted(groups.items(), key=lambda x: (order[x[0][0]], x[0][1], SPLITS.index(x[0][2]) if x[0][2] in SPLITS else 9, x[0][2])):
        col = {m: [r[m] for r in rs if r[m] is not None] for m in METRICS}
        share = lambda xs, t: f"{sum(x < t for x in xs) / len(xs):.1%}" if xs else "n/a"
        L.append(f"| {lang} | {st} | {sp} | {len(rs)} | {_med(col['kiwi'])} | {_med(col['labse'])} | "
                 f"{_med(col['chrf_bt'])} | {share(col['kiwi'], KIWI_THRESHOLD)} | {share(col['labse'], LABSE_THRESHOLD)} |")
    L += ["", "## Retranslation per language (unique segments)", "",
          "| lang | segments | NLLB labse median | NLLB kiwi median | sent to MADLAD | MADLAD chosen | numbers lost (final) |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for lang, ch in choosers.items():
        cs = list(ch.choice.values())
        if not cs:
            continue
        nl = [c["nllb"]["labse"] for c in cs if c["nllb"]["labse"] is not None]
        nk = [c["nllb"]["kiwi"] for c in cs if c["nllb"]["kiwi"] is not None]
        rt = sum(c["retranslated"] for c in cs)
        md = sum(c["engine"] == "madlad" for c in cs)
        L.append(f"| {lang} | {len(cs)} | {_med(nl)} | {_med(nk)} | {rt} ({rt / len(cs):.1%}) | "
                 f"{md} ({md / len(cs):.1%}) | {sum(not c['numbers_kept'] for c in cs)} |")
    if problems:
        L += ["", "## Problems", ""] + [f"- {p}" for p in problems[:50]]
        if len(problems) > 50:
            L.append(f"- ... {len(problems) - 50} more")
    L += ["", "## 10 random examples per language (seed 0, MC and Thai MMLU strings)", ""]
    rng = random.Random(0)
    esc = lambda s: s.replace("|", "\\|").replace("\n", " ")[:300]
    f3 = lambda x: "-" if x is None else f"{x:.2f}"
    for lang in MT_LANGS:
        pool = [r for r in rows if r["lang"] == lang and r["labse"] is not None]
        if not pool:
            continue
        L += [f"### {lang}", "", "| where | kiwi | labse | chrF-bt | English | translation |",
              "| --- | --- | --- | --- | --- | --- |"]
        for r in rng.sample(pool, min(10, len(pool))):
            L.append(f"| {r['set']} {r['concept']} {r['split']} {r['kind']} | {f3(r['kiwi'])} | {f3(r['labse'])} | "
                     f"{f3(r['chrf_bt'])} | {esc(r['en'])} | {esc(r['tr'])} |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    QA_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    problems: List[str] = []
    choosers = {l: Chooser(l) for l in MT_LANGS}
    for lang in LANGS:
        mc, probs = build_mc(lang, choosers.get(lang), args.partial, rows)
        problems += probs
        for c, sets in mc.items():
            assert list(sets) == list(SPLITS), (lang, c)
        (QA_DIR / f"mc_questions_{lang}.json").write_text(json.dumps(mc, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
        print(f"mc_questions_{lang}.json: {len(mc)} concepts, "
              f"{sum(len(v) for s in mc.values() for v in s.values())} items")
    try:
        th = build_mmlu_th(choosers["th"], rows)
        (MMLU_DIR / "th.json").write_text(json.dumps(th, ensure_ascii=False, indent=1), encoding="utf-8")
        print("mmlu/th.json: " + ", ".join(f"{k}={len(v)}" for k, v in th.items()))
    except Missing:
        if not args.partial:
            raise
        print("mmlu/th.json: not all segments translated yet, skipped")
    with (QA_DIR / "translation_choice.jsonl").open("w", encoding="utf-8") as f:
        for ch in choosers.values():
            for c in ch.choice.values():
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    for p in problems:
        print("PROBLEM", p)
    if not args.no_report and any(r["labse"] is not None for r in rows):
        (QA_DIR / "QUALITY.md").write_text(report(rows, choosers, problems), encoding="utf-8")
        print("wrote QUALITY.md")


if __name__ == "__main__":
    main()
