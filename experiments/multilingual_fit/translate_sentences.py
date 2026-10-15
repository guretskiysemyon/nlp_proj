#!/usr/bin/env python3
"""Translate the English concept and neutral sentences (data/) into ru/de/es/fr with Gemini.

Option 6 in docs/Part 2/multilingual_features.md: the same 300 + 300 sentences in every
language, so a joint fit sees identical content and only the language differs. Sentence i
of a translated set is the translation of English sentence i (aligned).

Calls the Gemini REST API directly (the installed google.generativeai SDK cannot turn
thinking off). Sentences go in chunks of --chunk-size, numbered, and come back as JSON.
A chunk is accepted only if every id comes back, non-empty, in the right script, not a
copy of the English, and of plausible length; otherwise it is retried, and finally each
failing sentence is translated alone. Every accepted chunk is cached, so a rerun resumes.
It never touches data/.

Output (<out-dir>, default experiments/multilingual_fit/data_translated/):
    concept_sentences_<lang>.json   [{"concept","title","sentences"}]  (loader format,
                                     ember/local_datasets.py; title = "translated")
    neutral_sentences_<lang>.json   [{"sentence","title","url"}]
    cache/<lang>/<set>__<chunk>.json  per-chunk translations
    usage.jsonl                      tokens per call (prompt / output / thinking)

Usage:
    .venv/bin/python -I experiments/multilingual_fit/translate_sentences.py \
        --concepts "World War II" --no-neutral          # pilot
    .venv/bin/python -I experiments/multilingual_fit/translate_sentences.py   # everything
"""
import argparse
import json
import os
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
LANGS = ["ru", "de", "es", "fr"]
LANG_NAMES = {"ru": "Russian", "de": "German", "es": "Spanish", "fr": "French"}
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
NEUTRAL = "__neutral__"

PROMPT = """Translate the numbered English sentences below into {lang_name}.

They are sentences from English Wikipedia{topic}. Rules:
- Translate each sentence faithfully and completely, as a single sentence, in the natural style of a {lang_name} Wikipedia article.
- Use the established {lang_name} forms of names, places, titles and terms (e.g. the usual {lang_name} name of a person, city, war, book or film), not word-for-word renderings.
- Write dates and numbers the way {lang_name} text does.
- Add nothing: no explanations, notes, quotation marks around the whole sentence, or transliterations in brackets.
- Keep every sentence separate, even if it is a fragment; never merge or split sentences.

Return a JSON array with one object per input sentence: {{"id": <number>, "text": "<translation>"}}.

{items}"""

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {"id": {"type": "INTEGER"}, "text": {"type": "STRING"}},
        "required": ["id", "text"],
    },
}

_usage_lock = threading.Lock()


def load_key():
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No GEMINI_API_KEY in the environment or .env")


def call_gemini(args, key, prompt, tag):
    """One generateContent call; returns the parsed JSON list. Logs token usage."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 16384,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "thinkingConfig": {"thinkingLevel": args.thinking_level},
        },
    }
    req = urllib.request.Request(
        API.format(model=args.model, key=key),
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(5 * 2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 5:
                time.sleep(5 * 2 ** attempt)
                continue
            raise
    u = resp.get("usageMetadata", {})
    rec = {"tag": tag, "model": args.model, "prompt": u.get("promptTokenCount", 0),
           "output": u.get("candidatesTokenCount", 0), "thinking": u.get("thoughtsTokenCount", 0),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with _usage_lock, open(args.out_dir / "usage.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    cand = resp["candidates"][0]
    if cand.get("finishReason") != "STOP":
        raise RuntimeError(f"finishReason={cand.get('finishReason')}")
    text = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
    return json.loads(text)


def problems(lang, en, tr):
    """Reasons a translation is suspicious (empty list = accept)."""
    out = []
    if not isinstance(tr, str) or not tr.strip():
        return ["empty"]
    if "\n" in tr.strip():
        out.append("newline")
    letters = [c for c in tr if c.isalpha()]
    cyr = sum("Ѐ" <= c <= "ӿ" for c in letters) / max(len(letters), 1)
    if lang == "ru" and cyr < 0.5:
        out.append(f"cyrillic={cyr:.2f}")
    if lang != "ru" and cyr > 0.1:
        out.append(f"cyrillic={cyr:.2f}")
    mixed = [w for w in re.findall(r"\w+", tr) if re.search("[A-Za-z]", w) and re.search("[\u0400-\u04ff]", w)]
    if mixed:
        out.append(f"mixed_script={mixed}")
    if len(en) > 25 and tr.strip() == en.strip():
        out.append("copied")
    ratio = len(tr) / max(len(en), 1)
    if len(en) > 25 and not 0.6 <= ratio <= 2.2:
        out.append(f"len_ratio={ratio:.2f}")
    return out


def translate_chunk(args, key, lang, set_name, topic, chunk_id, sents):
    cache = args.out_dir / "cache" / lang / f"{re.sub(r'[^A-Za-z0-9]+', '_', set_name)}__{chunk_id:03d}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        if d["en"] == sents and d.get("model") == args.model:
            return d["tr"], d["flags"]
    topic_txt = f' about "{topic}"' if topic else " (random sentences on unrelated topics)"

    def ask(idx):
        # number the sentences 1..len(idx) in every call (a single sentence is always "1."),
        # then map the ids back to positions in the chunk
        items = "\n".join(f"{n + 1}. {sents[i]}" for n, i in enumerate(idx))
        prompt = PROMPT.format(lang_name=LANG_NAMES[lang], topic=topic_txt, items=items)
        got = call_gemini(args, key, prompt, f"{lang}/{set_name}/{chunk_id}")
        return {idx[int(o["id"]) - 1]: o["text"].strip() for o in got
                if isinstance(o, dict) and "id" in o and 1 <= int(o["id"]) <= len(idx)}

    tr = [None] * len(sents)
    todo = list(range(len(sents)))
    for _ in range(args.retries):
        try:
            res = ask(todo)
        except Exception as e:  # malformed JSON, truncated output, HTTP error
            print(f"  [{lang}/{set_name}/{chunk_id}] chunk call failed: {e}", flush=True)
            continue
        for i in todo:
            if i in res and not problems(lang, sents[i], res[i]):
                tr[i] = res[i]
        todo = [i for i in todo if tr[i] is None]
        if not todo:
            break
    flags = {}
    for i in todo:  # last resort: one sentence per call; keep the answer but flag it
        last = None
        for _ in range(args.retries):
            try:
                last = ask([i]).get(i)
            except Exception as e:
                print(f"  [{lang}/{set_name}/{chunk_id}] sentence {i} failed: {e}", flush=True)
                continue
            if last is not None and not problems(lang, sents[i], last):
                break
        tr[i] = last
        p = problems(lang, sents[i], last)
        if p:
            flags[str(i)] = p
    if any(t is None for t in tr):
        raise RuntimeError(f"{lang}/{set_name}/{chunk_id}: untranslated sentences remain")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"model": args.model, "en": sents, "tr": tr, "flags": flags}, ensure_ascii=False, indent=1))
    return tr, flags


def translate_set(args, key, pool, lang, set_name, topic, sents):
    chunks = [sents[i:i + args.chunk_size] for i in range(0, len(sents), args.chunk_size)]
    futs = [pool.submit(translate_chunk, args, key, lang, set_name, topic, j, c)
            for j, c in enumerate(chunks)]
    tr, flags = [], {}
    for j, f in enumerate(futs):
        t, fl = f.result()
        flags.update({str(j * args.chunk_size + int(k)): v for k, v in fl.items()})
        tr.extend(t)
    return tr, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concepts", nargs="+", default=None, help="default: all 18")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--no-neutral", action="store_true", help="skip the 300 neutral sentences")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--thinking-level", default="minimal")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=HERE / "data_translated")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    key = load_key()

    concepts = json.loads((REPO / "data" / "concept_sentences.json").read_text())
    if args.concepts:
        missing = set(args.concepts) - {c["concept"] for c in concepts}
        if missing:
            sys.exit(f"unknown concepts: {missing}")
        concepts = [c for c in concepts if c["concept"] in args.concepts]
    neutral = json.loads((REPO / "data" / "neutral_sentences.json").read_text())

    t0 = time.time()
    report = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for lang in args.langs:
            c_path = args.out_dir / f"concept_sentences_{lang}.json"
            c_out = {c["concept"]: c for c in json.loads(c_path.read_text())} if c_path.exists() else {}
            for c in concepts:
                tr, flags = translate_set(args, key, pool, lang, c["concept"], c["concept"], c["sentences"])
                c_out[c["concept"]] = {"concept": c["concept"], "title": "translated", "sentences": tr}
                report[f"{lang}/{c['concept']}"] = flags
                print(f"[{lang}] {c['concept']}: {len(tr)} sentences, {len(flags)} flagged "
                      f"({time.time() - t0:.0f}s)", flush=True)
            c_path.write_text(json.dumps(list(c_out.values()), ensure_ascii=False, indent=1))
            if not args.no_neutral:
                tr, flags = translate_set(args, key, pool, lang, NEUTRAL, None,
                                          [n["sentence"] for n in neutral])
                out = [{"sentence": t, "title": n["title"], "url": n["url"]} for t, n in zip(tr, neutral)]
                (args.out_dir / f"neutral_sentences_{lang}.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=1))
                report[f"{lang}/{NEUTRAL}"] = flags
                print(f"[{lang}] neutral: {len(tr)} sentences, {len(flags)} flagged", flush=True)

    rep_path = args.out_dir / "flags.json"
    old = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    old.update(report)
    rep_path.write_text(json.dumps(old, ensure_ascii=False, indent=1))
    print(f"done in {time.time() - t0:.0f}s; flagged sentences: {sum(len(v) for v in report.values())}")


if __name__ == "__main__":
    main()
