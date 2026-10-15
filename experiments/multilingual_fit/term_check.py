#!/usr/bin/env python3
"""Step 4 of the translation plan: do the translations use the native terms?

Reference: Wikipedia's own interlanguage links, written by people and independent of the
translation model. For each concept:
  1. terms: every [[link]] in the concept's English article (wiki_raw/en/<Concept>.json
     gives the title), with the anchor texts used for it, plus the article title itself.
     For each linked article, the titles of its ru/de/es/fr versions (redirects
     followed). Cached in <data-dir>/terms/<Concept>.json.
  2. match: every English surface form (anchor text or title) is resolved to the one
     article it most likely means, and kept only if it is that article's title (plural
     allowed) or, for names, its last part ("Hitler" -> Adolf Hitler) (resolve_forms). For every English concept sentence that
     contains a form (whole word, case-sensitive except the first letter), check whether the
     translation contains the native title. Lowercase forms are domain terms ("squeeze play",
     "house edge") and need every content word; capitalised forms are names and places and
     need one (Гитлер ~ Адольф Гитлер). A word matches if some word of the translation starts with its stem
     (the first ~60% of its letters; whole word if <= 4 letters), which tolerates
     inflection (жертвенного банта ~ Жертвенный бант). Words of <= 2 letters and
     function words are ignored; parenthesised disambiguators are dropped.

Output (<data-dir>/terms/):
    matches.csv   one row per (lang, concept, sentence, term): en_term, target, native, matched
    summary.csv   per (concept, lang): checks, matched share, distinct mismatched terms

A mismatch is not necessarily an error (synonym, abbreviation, rephrasing); the judge
stage decides. Usage:
    .venv/bin/python experiments/multilingual_fit/term_check.py            # both stages
    .venv/bin/python experiments/multilingual_fit/term_check.py --match-only
"""
import argparse
import csv
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LANGS = ["ru", "de", "es", "fr"]
UA = ("EMBER-Embedding-Erasure-course-project/0.1 "
      "(https://github.com/guretskiysemyon/nlp_proj; academic coursework)")
SKIP_NS = re.compile(r"^(file|image|category|wikipedia|wp|template|help|portal|special|talk|user|"
                     r"wiktionary|wikt|commons|s|q|b|n|v|voy|d|mw|meta|draft|module|media|"
                     r"[a-z]{2,3}(-[a-z]+)?)\s*:", re.I)
LINK = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|([^\[\]]*))?\]\]")
STOP = {"ru": {"и", "в", "во", "на", "по", "из", "для", "от", "до", "с", "со", "о", "об", "к", "у"},
        "de": {"der", "die", "das", "des", "dem", "den", "und", "von", "vom", "im", "in", "zu", "zur", "zum", "am", "an", "auf", "für", "mit", "ein", "eine"},
        "es": {"de", "del", "la", "las", "el", "los", "y", "en", "a", "al", "por", "para", "con", "un", "una"},
        "fr": {"de", "du", "des", "la", "le", "les", "et", "en", "à", "au", "aux", "pour", "par", "un", "une", "d", "l"}}


def api(lang, **params):
    params.update(format="json", formatversion="2")
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)
            time.sleep(0.3)
            return out
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(3 * 2 ** attempt)
            err = e
    raise RuntimeError(f"API failed for {url}: {err}")


def safe(concept):  # same file names as fetch_wiki_articles.py
    return re.sub(r"[^\w\-]+", "_", concept).strip("_")


def fetch_terms(concept, out_path):
    title = json.loads((HERE / "wiki_raw" / "en" / f"{safe(concept)}.json").read_text())["title"]
    wikitext = api("en", action="parse", page=title, prop="wikitext", redirects=1)["parse"]["wikitext"]
    (out_path.parent / f"{out_path.stem}.wikitext").write_text(wikitext)
    anchors = defaultdict(lambda: defaultdict(int))  # target -> anchor text -> times linked
    for target, anchor in LINK.findall(wikitext):
        target = target.strip().replace("_", " ")
        if not target or SKIP_NS.match(target):
            continue
        target = target[0].upper() + target[1:]
        anchor = re.sub(r"'{2,}", "", anchor).strip() if anchor else target
        anchors[target][anchor or target] += 1
    anchors[title][title] += 1

    titles = sorted(anchors)
    native = {}
    for b in range(0, len(titles), 50):
        batch = titles[b:b + 50]
        # one language per request: the API returns at most 500 langlinks per request in
        # total, so asking for all languages of 50 pages at once silently truncates
        for lang in LANGS:
            q = api("en", action="query", titles="|".join(batch), prop="langlinks",
                    lllang=lang, lllimit="max", redirects=1)["query"]
            alias = {t: t for t in batch}
            for n in q.get("normalized", []):
                alias[n["from"]] = n["to"]
            redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
            pages = {p["title"]: p.get("langlinks", []) for p in q.get("pages", [])}
            for t in batch:
                a = alias.get(t, t)
                for ll in pages.get(redir.get(a, a), []):
                    native.setdefault(t, {})[lang] = ll["title"]
    terms = [{"target": t, "anchors": dict(anchors[t]), "native": native.get(t, {})}
             for t in titles if native.get(t)]
    out_path.write_text(json.dumps({"concept": concept, "en_title": title, "terms": terms},
                                   ensure_ascii=False, indent=1))
    return terms


def bare(title):
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def resolve_forms(terms):
    """English surface form -> the one term it most likely means.

    A word can link to several articles ("baseball" -> Baseball, Baseball (ball)). Prefer the
    article whose title (without a disambiguator) is the form itself, the undisambiguated one first;
    otherwise the article the form links to most often."""
    cands = defaultdict(list)  # lowercased form -> [(exact title, no parens, count, form, term)]
    for t in terms:
        forms = dict(t["anchors"])
        forms.setdefault(bare(t["target"]), 0)
        for form, count in forms.items():
            form = bare(form)
            if len(form) < 3 or re.fullmatch(r"[\d\s,.–-]+", form):
                continue
            title = bare(t["target"]).lower()
            exact = form.lower().rstrip("s") == title.rstrip("s")
            # a form counts only if it names the article itself (a surname may stand for the
            # full name: "Hitler" -> Adolf Hitler). Otherwise "nucleus" -> Atomic nucleus,
            # "par" -> Glossary of golf or "Italy invaded France" -> Battle of the Alps would
            # demand a title the sentence does not mean.
            if not (exact or (form[0].isupper() and title.endswith(" " + form.lower()))):
                continue
            cands[form.lower()].append((exact, "(" not in t["target"], count, form, t))
    out = {}
    for key, cs in cands.items():
        exact, _, _, form, t = max(cs, key=lambda c: c[:3])
        out[form] = t
    return out


def form_pattern(form):
    """Case-sensitive; a lowercase term may also start with a capital (sentence start)."""
    if form[0].isupper():
        body = re.escape(form)
    else:
        body = f"[{re.escape(form[0].lower())}{re.escape(form[0].upper())}]" + re.escape(form[1:])
    return re.compile(r"(?<!\w)" + body + r"(?!\w)")


def stems(native, lang):
    native = re.sub(r"\s*\([^)]*\)", "", native).strip()
    out = []
    for w in re.findall(r"\w+", native.lower()):
        if len(w) <= 2 or w in STOP[lang] or w.isdigit():
            continue
        out.append(w if len(w) <= 3 else w[:3] if len(w) == 4 else w[:max(4, math.ceil(len(w) * 0.6))])
    return out


def contains_native(translation, native, lang, proper):
    """Common terms need every content word; names and places one (Гитлер ~ Адольф Гитлер)."""
    st = stems(native, lang)
    if not st:
        return None
    words = re.findall(r"\w+", translation.lower().replace("ё", "е"))
    hits = [any(w.startswith(s.replace("ё", "е")) for w in words) for s in st]
    return any(hits) if proper else all(hits)


def kept_unchanged(translation, form):
    """A name the translation keeps as written in English ("WHO", "Time")."""
    return re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", translation) is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE / "data_translated")
    ap.add_argument("--match-only", action="store_true", help="reuse cached terms, no Wikipedia calls")
    args = ap.parse_args()
    tdir = args.data_dir / "terms"
    tdir.mkdir(exist_ok=True)

    en = {c["concept"]: c["sentences"] for c in json.loads((REPO / "data" / "concept_sentences.json").read_text())}
    tr = {lang: {c["concept"]: c["sentences"] for c in
                 json.loads((args.data_dir / f"concept_sentences_{lang}.json").read_text())} for lang in LANGS}

    rows = []
    for concept in en:
        path = tdir / f"{safe(concept)}.json"
        if path.exists() or args.match_only:
            terms = json.loads(path.read_text())["terms"]
        else:
            terms = fetch_terms(concept, path)
            print(f"{concept}: {len(terms)} linked terms with a ru/de/es/fr article", flush=True)
        # longest forms first, so "squeeze play" is found before "play"
        pats = [(form, t, form_pattern(form)) for form, t in
                sorted(resolve_forms(terms).items(), key=lambda x: -len(x[0]))]
        for i, sent in enumerate(en[concept]):
            seen, taken = set(), []
            for form, t, pat in pats:
                m = pat.search(sent)
                if not m or t["target"] in seen or any(a <= m.start() < b for a, b in taken):
                    continue
                seen.add(t["target"])
                taken.append(m.span())
                proper = form[0].isupper()  # Wikipedia writes common nouns in lower case
                for lang in LANGS:
                    nat = t["native"].get(lang)
                    if not nat or "#" in nat:  # section links: no article title to compare
                        continue
                    ok = contains_native(tr[lang][concept][i], nat, lang, proper)
                    if ok is None:
                        continue
                    ok = ok or (proper and kept_unchanged(tr[lang][concept][i], m.group(0)))
                    rows.append({"lang": lang, "concept": concept, "idx": i, "kind": "name" if proper else "term",
                                 "en_term": m.group(0), "target": t["target"], "native": nat,
                                 "matched": int(ok), "en": sent, "tr": tr[lang][concept][i]})

    with open(tdir / "matches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["concept"], r["lang"])].append(r)
    with open(tdir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "lang", "kind", "checks", "matched_share", "mismatched_terms"])
        for (c, lang), rs in sorted(groups.items()):
            for kind in ("term", "name"):
                k = [r for r in rs if r["kind"] == kind]
                if k:
                    w.writerow([c, lang, kind, len(k), f"{sum(r['matched'] for r in k) / len(k):.3f}",
                                len({r["target"] for r in k if not r["matched"]})])
    for kind in ("term", "name"):
        k = [r for r in rows if r["kind"] == kind]
        print(f"{kind}: {len(k)} checks, matched {sum(r['matched'] for r in k) / len(k):.3f}")
    print(f"-> {tdir}")


if __name__ == "__main__":
    main()
