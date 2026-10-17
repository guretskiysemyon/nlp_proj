#!/usr/bin/env python3
"""Concept sentence sets for ru/de/es/fr from the downloaded Wikipedia articles.

Per (concept, language), from wiki_raw/ (fetch_wiki_articles.py and
fetch_related_articles.py), sentences split with sentences.py:
  1. main article: every usable sentence is eligible (the article guarantees the topic);
  2. related articles: only sentences that mention the concept (STEMS below), since a
     related article can drift into a neighbouring topic (other holidays, drugs, sports);
  3. link sentences (fetch_link_sentences.py): sentences of other articles that link to
     the main article, only those judged relevant (judge_link_sentences.py).
If the main article has >= n sentences, n are sampled from it; otherwise all of it is
used and the rest is sampled from the related sentences, then from the link sentences.
A pair with fewer than n eligible sentences keeps what it has. Fixed seed per
(concept, language); a pair that needs no link sentences is the same as without them.

Output (<out-dir>, default data_v2/):
    concept_sentences_<lang>.json   [{"concept", "title", "sentences", "sources", "kinds"}]
                                    (sources: the article each sentence came from; kinds:
                                    main / related / link; format readable by
                                    ember.local_datasets.ConceptDataset)
    concept_report.json             counts per (concept, language)
    removed_references.json         every eligible sentence dropped as a bibliography /
                                    reference line (sentences.py), with language, concept,
                                    kind, article, page and revision ids, the rule and the
                                    full source line

Usage:
    .venv/bin/python -I experiments/multilingual_fit/build_concept_sets.py
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_wiki_articles import safe  # noqa: E402
from sentences import usable_sentences  # noqa: E402

HERE = Path(__file__).resolve().parent
# Concept mention, case-insensitive, covering ru/de/es/fr word forms. Used only to filter
# related-article sentences.
STEMS = {
    "Ancient Rome": r"рим|\brom\b|röm|roma|romain|romano",
    "Artificial intelligence": r"интеллект|intelligenz|inteligencia|intelligence|\bии\b|\bki\b|\bia\b",
    "Baseball": r"бейсбол|baseball|béisbol",
    "COVID-19 pandemic": r"covid|коронавирус|sars-cov|pandemi|пандеми|coronavirus",
    "Cannabis": r"каннабис|марихуан|конопл|гашиш|cannabis|marihuana|marijuana|haschisch|hachís|thc|тгк",
    "Culture of Greece": r"грец|греч|эллин|griech|hellen|grieg|helén|grec|hellén",
    # whole words for azar / pari: not "rechazar", "aparición", or fr "paris"
    "Gambling": r"азарт|казино|ставк|букмекер|glücksspiel|casino|spielbank|wett|apuesta|\bazar\b|jeux? d.argent|"
                r"\bpari(?:er|eurs?|euses?|é|ons|ez|ant)?\b|hasard",
    "Golf": r"гольф|golf",
    "Gun": r"оружи|ружь|пистолет|винтовк|револьвер|waffe|gewehr|pistole|revolver|schuss|\barmas?\b|fusil|pistol|\barmes?\b",
    "Halloween": r"х[эе]лл?оуин|halloween",
    "Harry Potter": r"поттер|potter|хогвартс|hogwarts|poudlard",  # not Rowling: the author, not the books
    "Heroin": r"героин|heroin|hero[ií]na|héroïne",
    "Nazism": r"нацис|национал-социал|nationalsozial|nazi|hitler|гитлер|nsdap|нсдап",
    "Pornography": r"порн|porn",
    "Republic of Ireland": r"ирланд|irland|irlanda|irlande|irisch",
    "Uranium": r"уран|\buran|urani",
    "Valentine's Day": r"валентин|valent[ií]n",
    "World War II": r"войн|krieg|guerra|guerre",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=HERE / "wiki_raw")
    ap.add_argument("--out-dir", type=Path, default=HERE / "data_v2")
    ap.add_argument("--langs", nargs="*", default=["ru", "de", "es", "fr"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-links", action="store_true", help="main and related sentences only")
    args = ap.parse_args()

    main_report = json.loads((args.raw_dir / "report.json").read_text())
    judged_path = args.raw_dir / "links_judged.json"
    judged = {} if args.no_links or not judged_path.exists() else json.loads(judged_path.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report, removed = {}, []
    for lang in args.langs:
        entries = []
        for concept, stem in STEMS.items():
            main_doc = json.loads((args.raw_dir / lang / f"{safe(concept)}.json").read_text())
            seen, rm_seen = set(), set()

            def log_removed(found, kind, doc_title, doc):
                # eligible sentences dropped as references, once per pair, with their source
                for r in found:
                    if r["sentence"] not in rm_seen:
                        rm_seen.add(r["sentence"])
                        removed.append({"lang": lang, "concept": concept, "kind": kind,
                                        "article": doc_title, "pageid": doc.get("pageid"),
                                        "revid": doc.get("revid"), **r})

            found = []
            main_s = [(s, main_doc["title"], "main") for s in usable_sentences(main_doc["text"], lang, found)
                      if not (s in seen or seen.add(s))]
            log_removed(found, "main", main_doc["title"], main_doc)
            rx = re.compile(stem, re.I)
            rel_s = []
            for f in sorted((args.raw_dir / lang / "related" / safe(concept)).glob("*.json")):
                doc = json.loads(f.read_text())
                found = []
                rel_s += [(s, doc["title"], "related") for s in usable_sentences(doc["text"], lang, found)
                          if rx.search(s) and not (s in seen or seen.add(s))]
                log_removed([r for r in found if rx.search(r["sentence"])], "related", doc["title"], doc)
            link_s = []
            ok = judged.get(concept, {}).get(lang, {})
            links_path = args.raw_dir / lang / "links" / f"{safe(concept)}.json"
            if ok and links_path.exists():
                for t, a in json.loads(links_path.read_text())["articles"].items():
                    for s in a["sentences"]:
                        if not ok.get(s, {}).get("relevant") or s in seen:
                            continue
                        found = []  # fetched with an older sentences.py: check the reference rules again
                        usable_sentences(s, lang, found)
                        if found:
                            log_removed(found, "link", t, a)
                            continue
                        seen.add(s)
                        link_s.append((s, t, "link"))
            rng = random.Random(f"{args.seed}-{concept}-{lang}")
            if len(main_s) >= args.n:
                chosen = rng.sample(main_s, args.n)
            else:
                chosen = main_s + rng.sample(rel_s, min(len(rel_s), args.n - len(main_s)))
                if len(chosen) < args.n and link_s:
                    chosen += rng.sample(link_s, min(len(link_s), args.n - len(chosen)))
                rng.shuffle(chosen)
            entries.append({"concept": concept, "title": main_report[concept]["langs"][lang]["title"],
                            "sentences": [s for s, _, _ in chosen], "sources": [t for _, t, _ in chosen],
                            "kinds": [k for _, _, k in chosen]})
            n = {k: sum(kind == k for _, _, kind in chosen) for k in ("main", "related", "link")}
            report.setdefault(concept, {})[lang] = {
                "main_available": len(main_s), "related_mentioning_available": len(rel_s),
                "links_relevant_available": len(link_s), "used": len(chosen), "used_from_main": n["main"],
                "used_from_related": n["related"], "used_from_links": n["link"]}
            print(f"[{lang}] {concept}: {len(chosen)} ({n['main']} main + {n['related']} related "
                  f"+ {n['link']} link)")
        (args.out_dir / f"concept_sentences_{lang}.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=1))
    (args.out_dir / "concept_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    (args.out_dir / "removed_references.json").write_text(json.dumps(removed, ensure_ascii=False, indent=1))
    print(f"{len(removed)} eligible sentences dropped as references -> removed_references.json")


if __name__ == "__main__":
    main()
