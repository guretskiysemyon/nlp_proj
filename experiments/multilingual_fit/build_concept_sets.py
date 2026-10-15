#!/usr/bin/env python3
"""Concept sentence sets for ru/de/es/fr from the downloaded Wikipedia articles.

Per (concept, language), from wiki_raw/ (fetch_wiki_articles.py and
fetch_related_articles.py), sentences split with sentences.py:
  1. main article: every usable sentence is eligible (the article guarantees the topic);
  2. related articles: only sentences that mention the concept (STEMS below), since a
     related article can drift into a neighbouring topic (other holidays, drugs, sports).
If the main article has >= n sentences, n are sampled from it; otherwise all of it is
used and the rest is sampled from the related sentences. A pair with fewer than n
eligible sentences keeps what it has. Fixed seed per (concept, language).

Output (<out-dir>, default data_v2/):
    concept_sentences_<lang>.json   [{"concept", "title", "sentences", "sources"}]
                                    (sources: the article each sentence came from;
                                    format readable by ember.local_datasets.ConceptDataset)
    concept_report.json             counts per (concept, language)

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
    "Gambling": r"азарт|казино|ставк|букмекер|glücksspiel|casino|spielbank|wett|apuesta|azar|jeux? d.argent|pari|hasard",
    "Golf": r"гольф|golf",
    "Gun": r"оружи|ружь|пистолет|винтовк|револьвер|waffe|gewehr|pistole|revolver|schuss|\barmas?\b|fusil|pistol|\barmes?\b",
    "Halloween": r"х[эе]лл?оуин|halloween",
    "Harry Potter": r"поттер|potter|хогвартс|hogwarts|poudlard|роулинг|rowling",
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
    args = ap.parse_args()

    main_report = json.loads((args.raw_dir / "report.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for lang in args.langs:
        entries = []
        for concept, stem in STEMS.items():
            main_doc = json.loads((args.raw_dir / lang / f"{safe(concept)}.json").read_text())
            seen = set()
            main_s = [(s, main_doc["title"]) for s in usable_sentences(main_doc["text"], lang)
                      if not (s in seen or seen.add(s))]
            rx = re.compile(stem, re.I)
            rel_s = []
            for f in sorted((args.raw_dir / lang / "related" / safe(concept)).glob("*.json")):
                doc = json.loads(f.read_text())
                rel_s += [(s, doc["title"]) for s in usable_sentences(doc["text"], lang)
                          if rx.search(s) and not (s in seen or seen.add(s))]
            rng = random.Random(f"{args.seed}-{concept}-{lang}")
            if len(main_s) >= args.n:
                chosen = rng.sample(main_s, args.n)
            else:
                chosen = main_s + rng.sample(rel_s, min(len(rel_s), args.n - len(main_s)))
                rng.shuffle(chosen)
            entries.append({"concept": concept, "title": main_report[concept]["langs"][lang]["title"],
                            "sentences": [s for s, _ in chosen], "sources": [t for _, t in chosen]})
            n_main = sum(t == main_doc["title"] for _, t in chosen)
            report.setdefault(concept, {})[lang] = {
                "main_available": len(main_s), "related_mentioning_available": len(rel_s),
                "used": len(chosen), "used_from_main": n_main, "used_from_related": len(chosen) - n_main}
            print(f"[{lang}] {concept}: {len(chosen)} ({n_main} main + {len(chosen) - n_main} related)")
        (args.out_dir / f"concept_sentences_{lang}.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=1))
    (args.out_dir / "concept_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
