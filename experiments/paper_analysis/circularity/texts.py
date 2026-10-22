"""WP7 targets: the token-type sets that the sharpening step is built from, one file per concept.

    python -m experiments.paper_analysis.circularity.texts            # all 18 paper concepts (Wikipedia API, cached)

Targets (token types of the given sentences, add_special_tokens=False, minus excluded/special ids):
    page      experiments/whitening/data/gemma/targets/<Slug>.json["page"] (the target CS has always used)
    disjoint  page sentences that are not (near-)duplicates of any S_C or S_N sentence
    related   sentences of the first 4 qualifying pages named in the main page's {{Main}}/{{Further}}/{{See also}}
              hatnotes (order of appearance), near-duplicates of S_C / S_N removed; pooled and, if larger than the
              main page, randomly subsampled (seed 0) to the main page's sentence count
    leakfree  page minus every V* token type (the η-sweep control "page_minus_vstar")
    wrong     the page target of WRONG[concept] (an unrelated paper concept)
    pagesub   size control for "disjoint": a random subset (seed 0) of all page sentences with |disjoint| sentences

Near-duplicate rule (decided before any fit): a sentence matches S_C ∪ S_N if, after casefolding and keeping only
[a-z0-9] words, it equals one of them, shares any word 8-gram with one, or has word-set Jaccard >= 0.5 with one.
A related page qualifies if it is not the main page (after redirects), its title does not start with List of /
Lists of / Outline of / Timeline of / Index of / Glossary of / Bibliography of, the link has no #section anchor, it
is not a disambiguation page, and >= 30 sentences survive the near-duplicate filter.
"""
from __future__ import annotations

import random
import re
import time

from experiments.paper_analysis.circularity.common import (PAPER_CONCEPTS, ROOT, TEXTS, WH_DIR, WRONG, load_prep,
                                                         read_json, slug, whitening_module, write_json)

N_RELATED, MIN_SENTS, NGRAM, JACC = 4, 30, 8, 0.5
SKIP_PREFIX = ("List of", "Lists of", "Outline of", "Timeline of", "Index of", "Glossary of", "Bibliography of")
HATNOTE = re.compile(r"\{\{\s*(?:Main|Main article|Further|Further information|See also)\s*\|([^}]*)\}\}", re.I)
PAGES_DIR = ROOT / "experiments" / "whitening" / "data" / "pages"


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.casefold())


class DupIndex:
    def __init__(self, sents):
        self.exact, self.grams, self.sets = set(), set(), []
        self.inv: dict[str, set[int]] = {}
        for s in sents:
            w = words(s)
            self.exact.add(" ".join(w))
            self.grams.update(tuple(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1))
            ws = set(w)
            for x in ws:
                self.inv.setdefault(x, set()).add(len(self.sets))
            self.sets.append(ws)

    def why(self, s: str) -> str | None:
        w = words(s)
        if " ".join(w) in self.exact:
            return "exact"
        if any(tuple(w[i:i + NGRAM]) in self.grams for i in range(len(w) - NGRAM + 1)):
            return "ngram"
        ws = set(w)
        if not ws:
            return None
        cand = set().union(*(self.inv.get(x, set()) for x in ws))
        for j in cand:
            o = self.sets[j]
            if len(ws & o) / len(ws | o) >= JACC:
                return "jaccard"
        return None


def filter_dups(sents, idx: DupIndex):
    keep, drops = [], {"exact": 0, "ngram": 0, "jaccard": 0}
    for s in sents:
        r = idx.why(s)
        if r is None:
            keep.append(s)
        else:
            drops[r] += 1
    return keep, drops


def api_mod():
    import sys
    p = str(ROOT / "experiments" / "resample_sentences")
    if p not in sys.path:
        sys.path.append(p)
    import fetch_en_sentences as fe
    return fe


def main_page(concept: str) -> dict:
    """Sentences of the main page: the whitening pages cache, or (Ancient Rome) a fresh fetch cached under TEXTS."""
    p = PAGES_DIR / f"{slug(concept)}.json"
    if p.exists():
        return read_json(p)
    q = TEXTS / "pages" / f"{slug(concept)}.json"
    if q.exists():
        return read_json(q)
    fe = api_mod()
    title, revid, paras = fe.page_paragraphs(concept)
    page = {"concept": concept, "title": title, "revid": revid, "sentences": fe.split_sentences(paras),
            "fetched": time.strftime("%Y-%m-%d")}
    write_json(q, page)
    return page


def hatnote_titles(title: str) -> list[str]:
    fe = api_mod()
    d = fe.api(action="parse", page=title, prop="wikitext", redirects=1)
    wt = d["parse"]["wikitext"]
    out = []
    for g in HATNOTE.findall(wt):
        for part in g.split("|"):
            part = part.strip().replace("{{!", "").strip()
            if part and "=" not in part and part not in out:
                out.append(part)
    return out


def fetch_page(title: str) -> dict:
    """Cached action=parse of one page -> {title (resolved), revid, sentences, disambiguation}."""
    q = TEXTS / "related_raw" / f"{slug(title)}.json"
    if q.exists():
        return read_json(q)
    fe = api_mod()
    d = fe.api(action="parse", page=title, prop="text|revid|properties", redirects=1)
    props = d["parse"].get("properties", {})
    disamb = ("disambiguation" in props) if isinstance(props, dict) else any(
        p.get("name") == "disambiguation" for p in props)
    p = fe._ParagraphText()
    p.feed(d["parse"]["text"])
    out = {"requested": title, "title": d["parse"]["title"], "revid": d["parse"]["revid"],
           "disambiguation": bool(disamb), "sentences": fe.split_sentences(p.paragraphs)}
    write_json(q, out)
    return out


def related_pages(concept: str, main_title: str, idx: DupIndex) -> list[dict]:
    chosen, log = [], []
    for t in hatnote_titles(main_title):
        if len(chosen) == N_RELATED:
            break
        if "#" in t:
            log.append({"title": t, "skip": "anchor"})
            continue
        if t.startswith(SKIP_PREFIX):
            log.append({"title": t, "skip": "list-like"})
            continue
        try:
            pg = fetch_page(t)
        except Exception as e:  # noqa: BLE001
            log.append({"title": t, "skip": f"fetch failed: {e}"})
            continue
        if pg["title"] == main_title or pg["title"].startswith(SKIP_PREFIX):
            log.append({"title": t, "skip": f"resolves to {pg['title']}"})
            continue
        if pg["disambiguation"]:
            log.append({"title": t, "skip": "disambiguation"})
            continue
        if any(c["title"] == pg["title"] for c in chosen):
            log.append({"title": t, "skip": "duplicate"})
            continue
        keep, drops = filter_dups(pg["sentences"], idx)
        if len(keep) < MIN_SENTS:
            log.append({"title": t, "skip": f"{len(keep)} sentences after dedupe"})
            continue
        chosen.append({"title": pg["title"], "revid": pg["revid"], "n_raw": len(pg["sentences"]),
                       "n_kept": len(keep), "drops": drops, "sentences": keep})
        log.append({"title": t, "chosen": pg["title"]})
    return chosen, log


def type_ids(tok, ex, sents) -> list[int]:
    ids = set()
    for i in range(0, len(sents), 256):
        for seq in tok(sents[i:i + 256], add_special_tokens=False)["input_ids"]:
            ids.update(seq)
    return sorted(int(i) for i in ids if i not in ex)


def build(concept: str, tok, ex) -> dict:
    out_p = TEXTS / "targets" / f"{slug(concept)}.json"
    if out_p.exists():
        return read_json(out_p)
    from experiments.varwidth.prep import build_dataset
    ds = build_dataset(concept)
    s_c = [s for s, lbl in ds.data if lbl == concept]
    s_n = [s for s, lbl in ds.data if lbl != concept]
    assert len(s_c) == 300 and len(s_n) == 300, (len(s_c), len(s_n))
    idx = DupIndex(s_c + s_n)
    page = main_page(concept)
    page_ids = sorted(int(t) for t in read_json(WH_DIR / "targets" / f"{slug(concept)}.json")["page"])
    rebuilt = type_ids(tok, ex, page["sentences"])
    jac_page = len(set(rebuilt) & set(page_ids)) / len(set(rebuilt) | set(page_ids))
    disjoint, d_drops = filter_dups(page["sentences"], idx)
    rel, rel_log = related_pages(concept, page["title"], idx)
    pool = list(dict.fromkeys(s for r in rel for s in r["sentences"]))
    n_page = len(page["sentences"])
    rel_sents = pool if len(pool) <= n_page else random.Random(0).sample(pool, n_page)
    sub_sents = random.Random(0).sample(page["sentences"], len(disjoint))
    vstar = set(int(t) for t in load_prep(concept, "plain")["vstar_ids"])
    wrong_ids = sorted(int(t) for t in read_json(WH_DIR / "targets" / f"{slug(WRONG[concept])}.json")["page"])
    ids = {"page": page_ids, "disjoint": type_ids(tok, ex, disjoint), "related": type_ids(tok, ex, rel_sents),
           "leakfree": [t for t in page_ids if t not in vstar], "wrong": wrong_ids,
           "pagesub": type_ids(tok, ex, sub_sents)}
    meta = {"concept": concept, "main_title": page["title"], "main_revid": page["revid"], "n_page_sents": n_page,
            "page_ids_rebuild_jaccard": round(jac_page, 4), "n_disjoint_sents": len(disjoint),
            "disjoint_drops": d_drops, "n_related_pool": len(pool), "n_related_sents": len(rel_sents),
            "related_pages": [{k: v for k, v in r.items() if k != "sentences"} for r in rel],
            "related_log": rel_log, "wrong_concept": WRONG[concept], "n_pagesub_sents": len(sub_sents),
            "sizes": {k: len(v) for k, v in ids.items()}}
    write_json(TEXTS / "sentences" / f"{slug(concept)}.json",
               {"disjoint": disjoint, "related": rel_sents, "pagesub": sub_sents})
    res = {"meta": meta, "ids": ids}
    write_json(out_p, res, indent=None)
    return res


def main():
    wc = whitening_module()
    tok, ex = wc.tokenizer("gemma"), wc.excluded_ids("gemma")
    rows = []
    for c in PAPER_CONCEPTS:
        t0 = time.time()
        r = build(c, tok, ex)
        m = r["meta"]
        rows.append(m)
        print(f"{c:24s} page={m['n_page_sents']:4d} disjoint={m['n_disjoint_sents']:4d} related={m['n_related_sents']:4d}"
              f" ({len(m['related_pages'])} pages: {', '.join(p['title'] for p in m['related_pages'])}) "
              f"sizes={m['sizes']} rebuildJ={m['page_ids_rebuild_jaccard']} ({time.time() - t0:.0f}s)", flush=True)
    write_json(TEXTS / "targets_summary.json", rows)


if __name__ == "__main__":
    main()
