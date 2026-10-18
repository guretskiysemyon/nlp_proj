#!/usr/bin/env python3
"""LLM judge for the link sentences (fetch_link_sentences.py): is the concept a real part
of what the sentence says, or only a date / a list item / part of a name?

The link already fixes the sense (a link to "Héroïne" is the drug), so the judge does not
check the sense; it drops passing mentions such as "se casaron el día de San Valentín de
2009". build_concept_sets.py uses only the sentences judged relevant.

Gemini REST API as in translate_sentences.py (temperature 0, minimal thinking, JSON
schema), --chunk-size numbered sentences per call; a chunk is accepted only if every id
comes back with a boolean, otherwise it is retried. Resumable: sentences already judged
(same model) are skipped.

Output:
    wiki_raw/links_judged.json         {concept: {lang: {sentence: {"relevant", "reason", "model"}}}}
    wiki_raw/links_judge_usage.jsonl   tokens per call

Usage:
    .venv/bin/python -I experiments/multilingual_fit/judge_link_sentences.py
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_wiki_articles import safe  # noqa: E402
from translate_sentences import API, LANG_NAMES, load_key  # noqa: E402

HERE = Path(__file__).resolve().parent
LANGS = ["ru", "de", "es", "fr"]
# what each concept covers, for the judge (same scope as the English concept pages)
SCOPE = {
    "Ancient Rome": "ancient Roman civilisation (the Roman Kingdom, Republic and Empire)",
    "Artificial intelligence": "artificial intelligence as a field and a technology",
    "Baseball": "the sport of baseball",
    "COVID-19 pandemic": "the COVID-19 pandemic and the disease COVID-19",
    "Cannabis": "cannabis as a drug (marijuana, hashish, THC)",
    "Culture of Greece": "Greek culture, ancient and modern (arts, literature, religion, customs, language)",
    "Gambling": "gambling (betting, casinos, games of chance, lotteries)",
    "Golf": "the sport of golf",
    "Gun": "guns and firearms",
    "Halloween": "the holiday Halloween (31 October) and its customs",
    "Harry Potter": "the Harry Potter books and franchise (films, characters, the wizarding world)",
    "Heroin": "the drug heroin",
    "Nazism": "Nazism, the Nazi party and Nazi Germany",
    "Pornography": "pornography",
    "Republic of Ireland": "the Republic of Ireland (the state)",
    "Uranium": "the chemical element uranium",
    "Valentine's Day": "Valentine's Day (14 February) as a holiday and its customs",
    "World War II": "the Second World War",
}
PROMPT = """You check sentences for a dataset of sentences about one concept.

Concept: {concept}, i.e. {scope}.
Each sentence below comes from a {lang_name} Wikipedia article and contains a link to the {lang_name} article about this concept, so the concept is meant in this sense.

For each sentence, decide whether the concept is a real part of what the sentence says:
- relevant = true: the sentence says something about the concept itself, or about something centred on it: its history, customs, rules, use, effects, laws, trade or technology; people, groups, places or works known for it; events, products or episodes themed on it (e.g. a Halloween party, a Valentine's Day special, a heroin overdose, firearms making city walls useless).
- relevant = false: the concept is only a date or time reference ("they married on Valentine's Day 2009", "the album came out on Halloween"), only one item in a list of similar things with nothing said about it ("holidays such as Christmas, Easter and Halloween"), or only part of a name or title, and the sentence is about something else.

Judge every sentence. Return a JSON array with one object per sentence: {{"id": <number>, "relevant": true or false, "reason": "<at most 8 English words>"}}.

{items}"""
SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {"id": {"type": "INTEGER"}, "relevant": {"type": "BOOLEAN"},
                       "reason": {"type": "STRING"}},
        "required": ["id", "relevant", "reason"],
    },
}
_lock = threading.Lock()


def call(args, key, prompt, tag):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192,
                                 "responseMimeType": "application/json", "responseSchema": SCHEMA,
                                 "thinkingConfig": {"thinkingLevel": args.thinking_level}}}
    req = urllib.request.Request(API.format(model=args.model, key=key), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(5 * 2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < 5:
                time.sleep(5 * 2 ** attempt)
                continue
            raise
    u = resp.get("usageMetadata", {})
    with _lock, open(args.raw_dir / "links_judge_usage.jsonl", "a") as f:
        f.write(json.dumps({"tag": tag, "model": args.model, "prompt": u.get("promptTokenCount", 0),
                            "output": u.get("candidatesTokenCount", 0),
                            "thinking": u.get("thoughtsTokenCount", 0),
                            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    cand = resp["candidates"][0]
    if cand.get("finishReason") != "STOP":
        raise RuntimeError(f"finishReason={cand.get('finishReason')}")
    return json.loads("".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought")))


def judge_chunk(args, key, concept, lang, sents, tag):
    """{sentence: {"relevant", "reason", "model"}} for one chunk, or raises after retries."""
    items = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    prompt = PROMPT.format(concept=concept, scope=SCOPE[concept], lang_name=LANG_NAMES[lang], items=items)
    for attempt in range(args.retries):
        try:
            got = {o["id"]: o for o in call(args, key, prompt, tag)}
            if all(isinstance(got.get(i + 1, {}).get("relevant"), bool) for i in range(len(sents))):
                return {s: {"relevant": got[i + 1]["relevant"], "reason": got[i + 1]["reason"],
                            "model": args.model} for i, s in enumerate(sents)}
            print(f"  {tag}: missing ids, retry {attempt + 1}", flush=True)
        except (RuntimeError, json.JSONDecodeError, KeyError) as e:
            print(f"  {tag}: {e}, retry {attempt + 1}", flush=True)
    raise RuntimeError(f"{tag}: no complete answer after {args.retries} tries")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=HERE / "wiki_raw")
    ap.add_argument("--concepts", nargs="*", default=list(SCOPE))
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--thinking-level", default="minimal")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    key = load_key()

    out_path = args.raw_dir / "links_judged.json"
    judged = json.loads(out_path.read_text()) if out_path.exists() else {}
    jobs = []
    for concept in args.concepts:
        for lang in args.langs:
            f = args.raw_dir / lang / "links" / f"{safe(concept)}.json"
            if not f.exists():
                continue
            doc = json.loads(f.read_text())
            sents = list(dict.fromkeys(s for a in doc["articles"].values() for s in a["sentences"]))
            done = judged.get(concept, {}).get(lang, {})
            todo = [s for s in sents if done.get(s, {}).get("model") != args.model]
            for k in range(0, len(todo), args.chunk_size):
                jobs.append((concept, lang, todo[k:k + args.chunk_size], f"{lang}/{concept}/{k // args.chunk_size}"))
    print(f"{sum(len(j[2]) for j in jobs)} sentences in {len(jobs)} calls", flush=True)

    def run(job):
        concept, lang, sents, tag = job
        res = judge_chunk(args, key, concept, lang, sents, tag)
        with _lock:
            judged.setdefault(concept, {}).setdefault(lang, {}).update(res)
            out_path.write_text(json.dumps(judged, ensure_ascii=False, indent=1))
        return tag

    with ThreadPoolExecutor(args.workers) as pool:
        for tag in pool.map(run, jobs):
            print(f"  done {tag}", flush=True)

    for concept in judged:
        for lang, d in judged[concept].items():
            n = len(d)
            k = sum(v["relevant"] for v in d.values())
            print(f"[{lang}] {concept}: {k}/{n} relevant ({k / n:.0%})", flush=True)


if __name__ == "__main__":
    main()
