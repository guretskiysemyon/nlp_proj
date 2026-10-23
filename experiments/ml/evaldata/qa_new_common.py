"""Shared constants for the MC questions of the 20 new concepts (PREREG_NEW.md, ML_CONCEPTS=new).

Outputs: results/ml/data/qa_new/ (work caches, MANIFEST.md); the final items are added (additively) to
results/ml/data/qa/mc_questions_<lang>.json and, for it/pt/hi/th, results/ml/data/qa_v2/mc_questions_<lang>.json.

    qa_new_gen.py        English: Gemini open-ended Q/A -> MC (EMBER Figure 9) -> blind Gemini verification,
                         refill, seed-42 split -> work/en_final.json
    qa_new_translate.py  7 languages with the v2 pipeline (NLLB + ctx lists, LaBSE, NLLB/MADLAD n-best and
                         MADLAD lists for the weak ones, numbers > script > LaBSE, distinct options), additive write
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[3]
QA_NEW = REPO / "results" / "ml" / "data" / "qa_new"
WORK = QA_NEW / "work"
EN_FINAL = WORK / "en_final.json"
PAGES = Path("/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/experiments/whitening/data/pages")
SPLITS = ("QA_train", "QA_test", "SimdomQA_train", "SimdomQA_test")
PER_SPLIT = 50
TR_LANGS = ("de", "fr", "it", "pt", "hi", "es", "th")
V2_LANGS = ("it", "pt", "hi", "th")
NLLB_CODE_ALL = {"de": "deu_Latn", "fr": "fra_Latn", "it": "ita_Latn", "pt": "por_Latn", "hi": "hin_Deva",
                 "es": "spa_Latn", "th": "tha_Thai"}

#: similar-domain topics (adjacent to, but distinct from, the concept). Chosen 2026-10-20 to avoid the other 37
#: concepts of the 18 + 20 set where an obvious neighbour is one (Cricket != Baseball, Plutonium != Uranium,
#: Robot != Artificial intelligence, Tuberculosis != COVID-19, Judaism <-> Buddhism, ...). ``avoid``: facts the
#: concept shares with its neighbours that a similar-domain question must not touch.
SIMDOM: Dict[str, Dict[str, object]] = {
    # the pilot's first round also had "the Napoleonic Wars (1803-1815)": the verifier counted all of it as French
    # Revolution (membership drop 42/42), so the refills use the Italian unification instead
    "French Revolution": {"topics": ["the Russian Revolution of 1917", "the English Civil War and the Glorious Revolution",
                                     "the unification of Italy (Risorgimento)"],
                          "avoid": "the 1789-1799 revolution in France, the storming of the Bastille, Louis XVI, "
                                   "Robespierre, Napoleon's rise to power (coup of 18 Brumaire)"},
    "Tuberculosis": {"topics": ["malaria", "HIV/AIDS", "cholera"],
                     "avoid": "tuberculosis, Mycobacterium, the BCG vaccine, TB co-infection"},
    "Cricket": {"topics": ["rugby union", "tennis", "field hockey"],
                "avoid": "cricket, its players, grounds, formats and tournaments"},
    "Buddhism": {"topics": ["Hinduism", "Jainism", "Taoism"],
                 "avoid": "the Buddha, Buddhist schools, texts and concepts, Buddhist sites; concepts shared with "
                          "Buddhism such as karma, samsara, nirvana or dharma"},
    "Plutonium": {"topics": ["thorium", "radium", "polonium"],
                  "avoid": "plutonium, the Manhattan Project, nuclear weapons, uranium"},
    "Judaism": {"topics": ["Islam", "Christianity (the New Testament and the churches)", "Zoroastrianism"],
                "avoid": "Judaism, the Hebrew Bible / Old Testament, Abraham, Moses, Jerusalem's Temple, Jewish "
                         "history and practice"},
    "Karate": {"topics": ["judo", "taekwondo", "boxing"],
               "avoid": "karate, Okinawa's martial arts, kata, karate styles and masters"},
    "Existentialism": {"topics": ["Stoicism", "utilitarianism", "British empiricism (Locke, Berkeley, Hume)"],
                       "avoid": "existentialism and its thinkers (Kierkegaard, Nietzsche, Heidegger, Sartre, "
                                "de Beauvoir, Camus), phenomenology, absurdism"},
    "Prime number": {"topics": ["Euclidean geometry (polygons, circles, the Pythagorean theorem)",
                                "the Fibonacci sequence and the golden ratio", "pi and irrational numbers"],
                     "avoid": "prime numbers, primality, factorization into primes, divisibility"},
    "The Matrix": {"topics": ["the original Star Wars trilogy", "Blade Runner (1982)", "The Terminator films"],
                   "avoid": "The Matrix franchise, the Wachowskis, Keanu Reeves, bullet time"},
    "Louvre": {"topics": ["the British Museum", "the Metropolitan Museum of Art", "the Museo del Prado"],
               "avoid": "the Louvre, Paris museums, the Mona Lisa and other works held by the Louvre"},
    "American Revolutionary War": {"topics": ["the American Civil War", "the War of 1812", "the Mexican-American War"],
                                   "avoid": "the American Revolution / War of Independence (1775-1783), the "
                                            "Declaration of Independence, George Washington's Continental Army"},
    "Robot": {"topics": ["the automobile", "the personal computer", "electricity and household electronics"],
              "avoid": "robots, robotics, automation, androids, artificial intelligence, industrial robots"},
    "Sigmund Freud": {"topics": ["Ivan Pavlov and classical conditioning", "B. F. Skinner and behaviorism",
                                 "Jean Piaget and developmental psychology"],
                      "avoid": "Freud, psychoanalysis, Carl Jung, the unconscious, dream analysis"},
    "Giraffe": {"topics": ["the African elephant", "the lion", "the zebra"],
                "avoid": "giraffes, the okapi"},
    "Surfing": {"topics": ["skateboarding", "snowboarding", "scuba diving"],
                "avoid": "surfing, surfboards, waves and surf breaks, Hawaii's surf culture"},
    "Flush toilet": {"topics": ["the refrigerator", "the washing machine", "the electric light bulb"],
                     "avoid": "toilets, plumbing, sewage, sanitation, bathrooms"},
    "Cloud": {"topics": ["volcanoes", "ocean tides and currents", "Earth's seasons and climate zones"],
              "avoid": "clouds, cloud types, fog, precipitation formation, the water cycle in the atmosphere"},
    "Doctor of Philosophy": {"topics": ["medical school and the MD degree", "law school and the bar exam",
                                        "secondary education (high school, the SAT, A-levels)"],
                             "avoid": "the PhD / doctorates, doctoral theses and dissertations, doctoral "
                                      "supervision, the history of the doctorate"},
    "Tourism": {"topics": ["agriculture", "international trade and shipping", "the retail industry"],
                "avoid": "tourism, travel, hotels and hospitality, tourist destinations"},
}


def slug(concept: str) -> str:
    return re.sub(r"\W+", "_", concept).strip("_")


def new_concepts() -> List[str]:
    return [e["concept"] for e in json.loads((REPO / "data" / "concept_sentences_extra.json").read_text(encoding="utf-8"))]


def page_text(concept: str) -> tuple[str, int, str]:
    """(article text, revid, source): the full scraped page if present (same revision as
    data/concept_sentences_extra.json), else the 300 sentences of concept_sentences_extra.json."""
    extra = {e["concept"]: e for e in json.loads((REPO / "data" / "concept_sentences_extra.json").read_text(encoding="utf-8"))}[concept]
    p = PAGES / f"{slug(concept)}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("revid") == extra["revid"]:
            return " ".join(d["sentences"]), d["revid"], str(p)
    return " ".join(extra["sentences"]), extra["revid"], "data/concept_sentences_extra.json"


assert set(SIMDOM) == set(new_concepts()), set(SIMDOM) ^ set(new_concepts())
