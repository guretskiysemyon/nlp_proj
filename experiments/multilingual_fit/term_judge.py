#!/usr/bin/env python3
"""Step 4, judge stage: is a term mismatch from term_check.py a real translation error?

A mismatch only says the translation does not contain the native Wikipedia title; it may
still be right (inflected form the stem rule misses, accepted synonym such as "Cannabis" for
de "Hanf", abbreviation, or a title that names a broader article). A judge decides for each
mismatch: OK or ERROR, and for ERROR the term it should be.

Judge (--backend):
  gemini (default)  gemini-3.8-flash, thinking "low". It is not the model that translated
                    (3.5-flash-lite / 3.6-flash), and on 40 random mismatches it agreed
                    with gemini-3.1-pro-preview 40/40 at ~1/5 of the cost.
  qwen              Qwen2.5-32B-Instruct via judge_rerun/qwen_judge.py (GPU,
                    run_term_judge.sbatch). An independent model family, but far slower here
                    (a 32-prompt smoke test had not finished after 20 min on 3x RTX 2080 Ti).

Input:  <data-dir>/terms/matches.csv (rows with matched = 0)
Output: <data-dir>/terms/judged.csv  (the mismatch rows + verdict, correct_term, raw)
        <data-dir>/terms/judge_cache.jsonl (gemini: one answer per prompt, for resume)
        <data-dir>/terms/usage.jsonl       (gemini: tokens per call)

Usage:
    .venv/bin/python experiments/multilingual_fit/term_judge.py
"""
import argparse
import csv
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
LANG_NAMES = {"ru": "Russian", "de": "German", "es": "Spanish", "fr": "French"}

PROMPT = """You are checking terminology in a translation from English into {lang_name}.

English sentence: {en}
{lang_name} translation: {tr}

Term to check: "{en_term}"
Reference: the {lang_name} Wikipedia article for this term is titled "{native}". The title may be a broader or more formal name than what a sentence would normally use, so do not require it literally.

Is "{en_term}" rendered correctly in the translation? Answer OK if the translation uses the reference term, an inflected or shortened form of it, an established synonym or abbreviation, or another wording that is correct and natural in {lang_name} for this context. Answer ERROR only if the term is mistranslated (wrong meaning, word-for-word calque of a technical term, invented or non-standard word) or missing.

Reply in exactly two lines:
Reason: <one short sentence>
Verdict: OK  or  Verdict: ERROR | <the correct {lang_name} term>"""


def parse(resp):
    m = re.search(r"Verdict:\s*(OK|ERROR)\s*(?:\|\s*(.+))?", resp, re.I)
    if not m:
        return "UNPARSED", ""
    return m.group(1).upper(), (m.group(2) or "").strip().strip('"«»')


def gemini_judge(prompts, model, tdir, workers):
    """One answer per prompt; cached by prompt text in judge_cache.jsonl so a rerun resumes."""
    sys.path.insert(0, str(HERE))
    from translate_sentences import load_key
    key = load_key()
    cache_path, usage_path = tdir / "judge_cache.jsonl", tdir / "usage.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            d = json.loads(line)
            if d["model"] == model:
                cache[d["prompt"]] = d["answer"]
    lock = threading.Lock()

    def one(p):
        if p in cache:
            return cache[p]
        body = {"contents": [{"role": "user", "parts": [{"text": p}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 4096,
                                     "thinkingConfig": {"thinkingLevel": "low"}}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    resp = json.load(r)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                if getattr(e, "code", 503) in (429, 500, 502, 503, 504) and attempt < 7:
                    time.sleep(5 * 2 ** attempt)
                    continue
                raise
        u = resp.get("usageMetadata", {})
        answer = "".join(x.get("text", "") for x in resp["candidates"][0]["content"]["parts"]
                         if not x.get("thought"))
        with lock:
            with open(cache_path, "a") as f:
                f.write(json.dumps({"model": model, "prompt": p, "answer": answer}, ensure_ascii=False) + "\n")
            with open(usage_path, "a") as f:
                f.write(json.dumps({"model": model, "prompt": u.get("promptTokenCount", 0),
                                    "output": u.get("candidatesTokenCount", 0),
                                    "thinking": u.get("thoughtsTokenCount", 0)}) + "\n")
        return answer

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prompts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE / "data_translated")
    ap.add_argument("--backend", choices=["gemini", "qwen"], default="gemini")
    ap.add_argument("--judge-model", default=None,
                    help="default: gemini-3.8-flash (gemini) / Qwen/Qwen2.5-32B-Instruct (qwen)")
    ap.add_argument("--workers", type=int, default=4, help="gemini: parallel calls")
    ap.add_argument("--batch-size", type=int, default=16, help="qwen: generation batch")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N (smoke test)")
    args = ap.parse_args()
    model = args.judge_model or ("gemini-3.8-flash" if args.backend == "gemini" else "Qwen/Qwen2.5-32B-Instruct")
    tdir = args.data_dir / "terms"
    csv.field_size_limit(sys.maxsize)
    rows = [r for r in csv.DictReader(open(tdir / "matches.csv")) if r["matched"] == "0"]
    if args.limit:
        rows = rows[:args.limit]
    prompts = [PROMPT.format(lang_name=LANG_NAMES[r["lang"]], **r) for r in rows]
    print(f"{len(prompts)} mismatches to judge with {model}", flush=True)

    if args.backend == "gemini":
        out = gemini_judge(prompts, model, tdir, args.workers)
    else:
        from judge_rerun.qwen_judge import QwenJudge
        judge = QwenJudge(model_id=model)
        out = judge.generate_batch(prompts, max_new_tokens=100, batch_size=args.batch_size,
                                   checkpoint_path=tdir / ("judge_ckpt.json" if not args.limit else "judge_ckpt_smoke.json"))

    name = "judged.csv" if not args.limit else "judged_smoke.csv"
    with open(tdir / name, "w", newline="") as f:
        fields = ["lang", "concept", "idx", "kind", "en_term", "target", "native",
                  "verdict", "correct_term", "en", "tr", "raw"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, resp in zip(rows, out):
            verdict, term = parse(resp)
            w.writerow({**{k: r[k] for k in fields if k in r}, "verdict": verdict,
                        "correct_term": term, "raw": resp})
    verdicts = [parse(x)[0] for x in out]
    print({v: verdicts.count(v) for v in set(verdicts)}, f"-> {tdir / name}")


if __name__ == "__main__":
    main()
