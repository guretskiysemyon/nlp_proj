#!/usr/bin/env python3
"""Translated top-up for the 300-per-language extended concept sets (extended_sets.build_concept_n).

A language whose concept page has fewer than PER_LANG sentences is filled up to PER_LANG with Gemini translations of
the concept page in the other languages:
    sources   the other languages' page sentences (pools/<lang>.json): English first, then the remaining languages by
              decreasing page size; each source language in a seeded random order. A surplus of SURPLUS is translated
              so that QA failures can be dropped without a second round.
    prompt    translate_neutral's Wikipedia-editor prompt, plus a glossary: the concept's title in the target
              language and, for English sources, the target-language Wikipedia titles of the English article's linked
              articles whose anchor text occurs in the batch (interlanguage links, Wikipedia API; cached).
    QA        (a) script: share of letters in the target script (hi: Devanagari, th: Thai, others: Latin) >= 0.8;
              (b) not an echo of the source; (c) length ratio target/source chars in [0.4, 2.5];
              (d) LaBSE cosine(source, translation), stored (scored by score_labse(); --labse-min drops low ones).
              Failures of (a)-(c) are re-translated once with RETRY_MODEL, then dropped.
Batches are cached in <concept extended dir>/translations_topup/<lang>_<batch>.json (reused while sources match).

Output: <concept extended dir>/topup_translated.json
    [{"sentence", "lang", "title", "source": "translated", "src_lang", "src_sentence", "labse", "model"}]
    (exactly PER_LANG - native count per language, where enough sources pass QA)
        <concept extended dir>/topup_report.json   per language: needed, translated, failed QA, LaBSE summary

Usage:
    python experiments/multilingual_hp/topup_translated.py --concept "Harry Potter" [--concept Gun ...] [--workers 8]
    python experiments/multilingual_hp/topup_translated.py --all
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "resample_sentences"))
sys.path.insert(0, str(HERE.parent.parent))

from splitting import CONCEPT_TITLES, LANGS, SEED, concept_paths  # noqa: E402
import translate_neutral  # noqa: E402

PER_LANG = 300
BATCH = 40
SURPLUS = 0.15
MODEL = "models/gemini-3.5-flash-lite"   # the repo's translator (translate_neutral.MODEL)
RETRY_MODEL = "models/gemini-3.6-flash"  # retranslate_low.py's model for hard cases
UA = "EMBER-Embedding-Erasure-course-project/0.1 (academic coursework)"
SCRIPT_RE = {"hi": re.compile(r"[ऀ-ॿ]"), "th": re.compile(r"[฀-๿]")}
LATIN_RE = re.compile(r"[A-Za-zÀ-ɏ]")
MIN_SCRIPT_FRAC = 0.8
LEN_RATIO = (0.4, 2.5)

PROMPT = """Translate each {src_language} sentence below into {language}. The sentences come from the Wikipedia article "{concept}".
Write the way a {language} Wikipedia editor would: faithful and natural, no additions, omissions or notes.
Keep numbers and dates; write proper names the way {language} Wikipedia writes them (in {script} script where that is the norm).
{glossary}Input: a JSON list of {{"id": int, "text": str}}.
Return ONLY a JSON list of {{"id": int, "translation": str}} with exactly the same ids.

Input:
{payload}"""


# ---------------------------------------------------------------- glossary (interlanguage links)
def _api(lang: str, **params):
    q = urllib.parse.urlencode({"format": "json", **params})
    req = urllib.request.Request(f"https://{lang}.wikipedia.org/w/api.php?{q}", headers={"User-Agent": UA})
    err = None
    for attempt in range(8):
        time.sleep(1.0)  # be polite: at most ~1 request per second
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            err = e
            wait = int(e.headers.get("Retry-After") or 0) or min(120, 10 * 2 ** attempt)
            print(f"  [wikipedia] HTTP {e.code}; waiting {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"wikipedia api failed: {err}")


def page_links(concept: str) -> dict:
    """{anchor text: English title} for the links of the concept's English article (raw/en.json HTML)."""
    html = json.loads((concept_paths(concept)["raw"] / "en.json").read_text())["html"]
    out = {}
    for href, title, anchor in re.findall(r'<a href="/wiki/([^"#:]+)"[^>]*title="([^"]+)"[^>]*>([^<]+)</a>', html):
        anchor = anchor.strip()
        if len(anchor) >= 3 and not anchor.isdigit():
            out.setdefault(anchor, title)
    return out


def glossary(concept: str, langs) -> dict:
    """{lang: {English title: native title}} via langlinks; cached in the concept's extended dir."""
    path = concept_paths(concept)["extended"] / "topup_glossary.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    titles = sorted(set(page_links(concept).values()) | {json.loads((concept_paths(concept)["raw"] / "en.json")
                                                                    .read_text())["title"]})
    for lang in langs:
        if lang == "en" or lang in cache:
            continue
        got = {}
        for i in range(0, len(titles), 50):
            res = _api("en", action="query", prop="langlinks", lllang=lang, lllimit="max", redirects=1,
                       titles="|".join(titles[i:i + 50]))
            for p in res.get("query", {}).get("pages", {}).values():
                for ll in p.get("langlinks", []):
                    got[p["title"]] = ll["*"]
        cache[lang] = got
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        print(f"  glossary {concept}/{lang}: {len(got)} of {len(titles)} linked titles have a {lang} article")
    return cache


def batch_glossary(texts, src_lang: str, lang: str, links: dict, gl: dict, concept_native: str) -> str:
    lines = [f'- the article "{concept_native}" is the {LANGS[lang][0]} article on this topic']
    if src_lang == "en" and lang != "en":
        joined = " ".join(texts)
        seen = set()
        for anchor, title in links.items():
            native = gl.get(lang, {}).get(title)
            if native and native not in seen and re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", joined):
                seen.add(native)
                lines.append(f'- "{anchor}" -> "{native}"')
                if len(lines) >= 40:
                    break
    return ("Glossary (titles used by " + LANGS[lang][0] + " Wikipedia; use them where they fit):\n"
            + "\n".join(lines) + "\n")


# ---------------------------------------------------------------- Gemini
_models = {}


def _gen(model_name: str, prompt: str):
    if model_name not in _models:
        translate_neutral._model()  # loads .env and configures the API key
        import google.generativeai as gai
        _models[model_name] = gai.GenerativeModel(model_name)
    for attempt in range(1, 6):
        try:
            resp = _models[model_name].generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return translate_neutral._parse(resp.text)
        except Exception as e:  # noqa: BLE001
            wait = 30 * attempt if ("429" in str(e) or "quota" in str(e).lower()) else 5 * attempt
            print(f"  [gemini {model_name}] attempt {attempt}/5 failed: {str(e)[:160]}; waiting {wait}s", flush=True)
            time.sleep(wait)
    return {}


def translate(texts, src_lang, lang, concept, gloss, model_name=MODEL):
    language, script = LANGS[lang]
    got, todo = {}, list(range(len(texts)))
    for _ in range(3):
        payload = json.dumps([{"id": i, "text": texts[i]} for i in todo], ensure_ascii=False, indent=1)
        res = _gen(model_name, PROMPT.format(src_language=LANGS[src_lang][0], language=language, script=script,
                                             concept=concept, glossary=gloss, payload=payload))
        got.update({i: t for i, t in res.items() if i in todo and t})
        todo = [i for i in todo if not got.get(i)]
        if not todo:
            break
    return [got.get(i, "") for i in range(len(texts))]


def qa_fail(src: str, tr: str, lang: str) -> str:
    """'' if the translation passes checks (a)-(c), else the reason."""
    if not tr.strip():
        return "empty"
    if tr.strip() == src.strip() and lang != "en":
        return "echo"
    letters = [c for c in tr if c.isalpha()]
    pat = SCRIPT_RE.get(lang, LATIN_RE)
    if not letters or sum(bool(pat.match(c)) for c in letters) / len(letters) < MIN_SCRIPT_FRAC:
        return "script"
    r = len(tr) / max(1, len(src))
    if not LEN_RATIO[0] <= r <= LEN_RATIO[1]:
        return "length"
    return ""


# ---------------------------------------------------------------- per concept
def pools(concept: str) -> dict:
    out = {}
    for lang in LANGS:
        p = concept_paths(concept)["pools"] / f"{lang}.json"
        if p.exists():
            out[lang] = json.loads(p.read_text())
    return out


def plan_sources(pl: dict, lang: str, n: int) -> list:
    """(src_lang, sentence) list of length n for target `lang`: English first, then by decreasing page size."""
    others = [l for l in pl if l != lang]
    others.sort(key=lambda l: (l != "en", -len(pl[l]["sentences"])))
    out = []
    for l in others:
        sents = list(pl[l]["sentences"])
        random.Random(f"{SEED}-topup-{lang}-{l}").shuffle(sents)
        out += [(l, s) for s in sents]
        if len(out) >= n:
            break
    return out[:n]


def _batch_job(concept, lang, b, items, gloss_args, tdir):
    path = tdir / f"{lang}_{b:03d}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if [(c["src_lang"], c["src_sentence"]) for c in cached] == [tuple(x) for x in items]:
            return lang, b, cached, True
    rows = []
    for src_lang in dict.fromkeys(l for l, _ in items):  # one call per source language in the batch
        idx = [i for i, (l, _) in enumerate(items) if l == src_lang]
        texts = [items[i][1] for i in idx]
        gloss = batch_glossary(texts, src_lang, lang, *gloss_args)
        trans = translate(texts, src_lang, lang, concept, gloss)
        retry = [k for k, t in enumerate(trans) if qa_fail(texts[k], t, lang)]
        if retry:
            again = translate([texts[k] for k in retry], src_lang, lang, concept, gloss, RETRY_MODEL)
            for k, t in zip(retry, again):
                if not qa_fail(texts[k], t, lang):
                    trans[k] = t
        for i, k in zip(idx, range(len(texts))):
            rows.append({"i": i, "src_lang": src_lang, "src_sentence": texts[k], "translation": trans[k],
                         "qa_fail": qa_fail(texts[k], trans[k], lang),
                         "model": RETRY_MODEL if k in retry and not qa_fail(texts[k], trans[k], lang) else MODEL})
    rows.sort(key=lambda r: r["i"])
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return lang, b, rows, False


def run_concept(concept: str, per_lang: int = PER_LANG, workers: int = 8, labse_min: float = 0.0):
    paths = concept_paths(concept)
    pl = pools(concept)
    need = {l: max(0, per_lang - len(pl[l]["sentences"])) for l in pl}
    need = {l: n for l, n in need.items() if n}
    print(f"[topup] {concept}: need {need}", flush=True)
    out_path = paths["extended"] / "topup_translated.json"
    if not need:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("[]")
        return []
    gl = glossary(concept, need)
    links = page_links(concept)
    tdir = paths["extended"] / "translations_topup"
    tdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for lang, n in need.items():
        src = plan_sources(pl, lang, int(n * (1 + SURPLUS)) + 1)
        for b in range((len(src) + BATCH - 1) // BATCH):
            jobs.append((lang, b, src[b * BATCH:(b + 1) * BATCH],
                         (links, gl, pl[lang]["title"])))
    translate_neutral._model()
    done = {}
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(_batch_job, concept, lang, b, items, ga, tdir) for lang, b, items, ga in jobs]
        for fut in as_completed(futs):
            lang, b, rows, cached = fut.result()
            done[(lang, b)] = rows
            print(f"  [{len(done)}/{len(jobs)}] {concept}/{lang} batch {b}: {'cached' if cached else 'translated'}",
                  flush=True)
    labse = score_labse([(r["src_sentence"], r["translation"]) for k in sorted(done) for r in done[k]
                         if not r["qa_fail"]])
    out, report = [], {}
    for lang, n in need.items():
        rows = [r for k in sorted(done) if k[0] == lang for r in done[k]]
        ok = [r for r in rows if not r["qa_fail"] and labse.get((r["src_sentence"], r["translation"]), 1.0) >= labse_min]
        keep = ok[:n]
        out += [{"sentence": r["translation"], "lang": lang, "title": pl[lang]["title"], "source": "translated",
                 "src_lang": r["src_lang"], "src_sentence": r["src_sentence"],
                 "labse": labse.get((r["src_sentence"], r["translation"])), "model": r["model"]} for r in keep]
        ls = sorted(x for x in (labse.get((r["src_sentence"], r["translation"])) for r in keep) if x is not None)
        fails = {}
        for r in rows:
            if r["qa_fail"]:
                fails[r["qa_fail"]] = fails.get(r["qa_fail"], 0) + 1
        report[lang] = {"needed": n, "translated": len(rows), "kept": len(keep), "qa_fail": fails,
                        "retried_model": sum(r["model"] == RETRY_MODEL for r in keep),
                        "labse_median": ls[len(ls) // 2] if ls else None,
                        "labse_p05": ls[int(0.05 * len(ls))] if ls else None,
                        "src_langs": {l: sum(r["src_lang"] == l for r in keep) for l in dict.fromkeys(r["src_lang"] for r in keep)}}
        if len(keep) < n:
            print(f"  WARNING {concept}/{lang}: only {len(keep)} of {n} passed QA", flush=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (paths["extended"] / "topup_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[topup] {concept}: wrote {len(out)} sentences -> {out_path}", flush=True)
    return out


# ---------------------------------------------------------------- LaBSE (plain transformers, no extra packages)
_labse = None


def score_labse(pairs, batch: int = 64) -> dict:
    """{(src, tr): cosine of LaBSE sentence embeddings} (sentence-transformers/LaBSE via transformers, CPU)."""
    global _labse
    if not pairs:
        return {}
    import torch
    from transformers import AutoModel, AutoTokenizer
    if _labse is None:
        name = "sentence-transformers/LaBSE"
        _labse = (AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval())
    tok, model = _labse

    def emb(texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=256, return_tensors="pt")
                out.append(torch.nn.functional.normalize(model(**enc).pooler_output, dim=-1))
        return torch.cat(out)

    pairs = list(dict.fromkeys(pairs))
    a, b = emb([p[0] for p in pairs]), emb([p[1] for p in pairs])
    cos = (a * b).sum(-1).tolist()
    return dict(zip(pairs, cos))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", nargs="+", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--per-lang", type=int, default=PER_LANG)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--labse-min", type=float, default=0.0)
    args = ap.parse_args()
    concepts = list(CONCEPT_TITLES) if args.all else args.concept
    for c in concepts:
        run_concept(c, args.per_lang, args.workers, args.labse_min)


if __name__ == "__main__":
    main()
