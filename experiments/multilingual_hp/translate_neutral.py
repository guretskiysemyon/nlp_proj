#!/usr/bin/env python3
"""Translated neutral sentence set: 320 English Wikipedia sentences, 40 per language.

The sentences are one random sentence from each of 320 random pages of the local English dump sample
(resample_sentences/fetch_en_sentences.neutral_sentences, seed SEED). Group i (40 sentences, in order)
belongs to LANGS key i: the en group stays English, the other 7 are translated by Gemini, one call per
language (re-asking only for missing ids). Each language is cached in data/translations/<lang>.json.

Output: data/neutral_sentences_translated.json
    [{"sentence", "lang", "en_sentence", "url", "title", "source": "translated"|"original"}]

Usage:
    python experiments/multilingual_hp/translate_neutral.py [--force]
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "resample_sentences"))
sys.path.insert(0, str(HERE.parent.parent))

from splitting import EN_DUMP_PAGES, LANGS, N_PER_LANG, NEUTRAL_TRANSLATED_PATH, SEED, TRANSLATIONS  # noqa: E402
from fetch_en_sentences import neutral_sentences  # noqa: E402

MODEL = "models/gemini-3.5-flash-lite"
MAX_ROUNDS = 3
MAX_CALL_RETRIES = 5

PROMPT = """Translate each English sentence below into {language}. The sentences come from random Wikipedia articles.
Write the way a {language} Wikipedia editor would: faithful and natural, no additions, omissions or notes.
Keep numbers and dates; write proper names the way {language} Wikipedia writes them (in {script} script where that is the norm).
Input: a JSON list of {{"id": int, "text": str}}.
Return ONLY a JSON list of {{"id": int, "translation": str}} with exactly the same ids.

Input:
{payload}"""

_client = None


def _model():
    global _client
    if _client is None:
        import dotenv

        dotenv.load_dotenv(HERE.parent.parent / ".env")
        from ember.interpret_features import SimpleGeminiClient

        _client = SimpleGeminiClient(model_name=MODEL)
    return _client.model


def _parse(text: str):
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    data = json.loads(m.group(0) if m else text)
    out = {}
    for item in data if isinstance(data, list) else []:
        # the model occasionally answers with the input key "text" instead of "translation"
        val = item.get("translation", item.get("text")) if isinstance(item, dict) else None
        if isinstance(val, str):
            try:
                out[int(item["id"])] = val.strip()
            except (TypeError, ValueError, KeyError):
                pass
    return out


def _call(prompt: str):
    for attempt in range(1, MAX_CALL_RETRIES + 1):
        try:
            resp = _model().generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return _parse(resp.text)
        except Exception as e:  # noqa: BLE001
            wait = 30 * attempt if ("429" in str(e) or "quota" in str(e).lower()) else 5 * attempt
            print(f"  [gemini] attempt {attempt}/{MAX_CALL_RETRIES} failed: {type(e).__name__}: {str(e)[:200]}; "
                  f"waiting {wait}s")
            if attempt < MAX_CALL_RETRIES:
                time.sleep(wait)
    return {}


def translate(texts, lang: str):
    """Translate texts (ids 0..len-1) into lang; re-ask only for missing/empty ids."""
    language, script = LANGS[lang]
    got, todo = {}, list(range(len(texts)))
    for rnd in range(1, MAX_ROUNDS + 1):
        payload = json.dumps([{"id": i, "text": texts[i]} for i in todo], ensure_ascii=False, indent=1)
        res = _call(PROMPT.format(language=language, script=script, payload=payload))
        # an answer equal to the English input is an echo, not a translation
        got.update({i: t for i, t in res.items() if i in todo and t and t != texts[i]})
        todo = [i for i in todo if not got.get(i)]
        print(f"  {lang} round {rnd}: {len(texts) - len(todo)}/{len(texts)} translated")
        if not todo:
            return [got[i] for i in range(len(texts))]
    raise RuntimeError(f"{lang}: no translation for ids {todo} after {MAX_ROUNDS} rounds")


def run(force: bool = False):
    langs = list(LANGS)
    pool = neutral_sentences(EN_DUMP_PAGES, len(langs) * N_PER_LANG, random.Random(SEED), set())
    TRANSLATIONS.mkdir(parents=True, exist_ok=True)
    out = []
    for gi, lang in enumerate(langs):
        group = pool[gi * N_PER_LANG:(gi + 1) * N_PER_LANG]
        if lang == "en":
            out += [{"sentence": g["sentence"], "lang": lang, "en_sentence": g["sentence"], "url": g["url"],
                     "title": g["title"], "source": "original"} for g in group]
            continue
        path = TRANSLATIONS / f"{lang}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() and not force else None
        if cached is not None and [c["en_sentence"] for c in cached] == [g["sentence"] for g in group]:
            print(f"{lang}: cached ({path})")
            rows = cached
        else:
            if cached is not None:
                print(f"{lang}: cache does not match the current English sample; re-translating")
            print(f"{lang}: translating {len(group)} sentences")
            trans = translate([g["sentence"] for g in group], lang)
            rows = [{"id": i, "en_sentence": g["sentence"], "translation": t, "url": g["url"], "title": g["title"]}
                    for i, (g, t) in enumerate(zip(group, trans))]
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        out += [{"sentence": r["translation"], "lang": lang, "en_sentence": r["en_sentence"], "url": r["url"],
                 "title": r["title"], "source": "translated"} for r in rows]
    NEUTRAL_TRANSLATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEUTRAL_TRANSLATED_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(out)} sentences -> {NEUTRAL_TRANSLATED_PATH}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="Re-translate even if data/translations/<lang>.json exists.")
    run(force=ap.parse_args().force)
