"""WP4 (robustness): concept sets, adjacent pages, hand-written lexicons and output paths.

Part A (abstract / intersectional concepts): ABSTRACT_NEW are fetched from English Wikipedia by fetch.py (300 sampled
sentences, seed 42, page order kept, exactly as experiments/varwidth/new_concepts.py); ABSTRACT_OLD and ENTITY reuse
the project's data, prep and fits (results/varwidth/...).

Part B (contamination): a fraction f of the 300 concept sentences is replaced by sentences of ADJACENT pages.

LEXICONS (20 words per new concept) were written by the analysis agent BEFORE any fit of these concepts was run,
following experiments/varwidth/j3_data/prompt_v1.txt (name forms first, then well-known specific entities/terms,
avoid words of the broader category). The existing concepts use concept_words_v1.json (Gemini, made earlier).
PARENTS lists, also before any fit, the stronger topics each abstract concept is an intersection of; they define the
"B" (broader) label in the blind labelling.
"""
from __future__ import annotations

import json

from experiments.varwidth.common import ROOT, slug  # noqa: F401  (slug re-exported)

OUT = ROOT / "results" / "paper_analysis" / "robustness"
DATA = OUT / "data"
FITS = OUT / "fits"
PAGES = DATA / "pages"
TARGETS = DATA / "targets"
CONCEPTS_FILE = DATA / "concept_sentences_pa.json"      # new abstract concepts, same schema as data/concept_sentences*.json
CONTAM_FILE = DATA / "concept_sentences_contam.json"    # contaminated S_C sets, concept name kept, keyed by variant

SEEDS = (42, 43, 44, 45, 46)
N_SENTS = 300
SAMPLE_SEED = 42

# Pages with >= 300 sentences (the protocol's |S_C| = 300). Stoicism, Epicureanism, Impressionism and Neoplatonism
# (the first choices) have only 197-233 sentences: they form the SHORT group, fitted with |S_C| = 190 next to the
# entity controls sub-sampled to 190 sentences (random.Random(42) over the project's 300), so S is matched.
# Candidates checked for length (Oct 22 2026): Transcendentalism 94, Rococo 219, Pragmatism 275, Mannerism 256,
# Scholasticism 109, Cubism 296, Platonism 104, Empiricism 192, Expressionism 119 (too short); Surrealism 319,
# Art Nouveau 723, Neoclassicism 337, Gnosticism 494 (used); Romanesque architecture 480 (not used: twin of Gothic).
ABSTRACT_NEW = ["Romanticism", "Utilitarianism", "Baroque", "Gothic architecture", "Surrealism", "Art Nouveau",
                "Neoclassicism", "Gnosticism"]
ABSTRACT_SHORT = ["Stoicism", "Epicureanism", "Impressionism", "Neoplatonism"]
S_SHORT = 190
FALLBACK: list[str] = []
ABSTRACT_OLD = ["Existentialism"]                      # already in the project (data, prep, fits)
ENTITY = ["Harry Potter", "Baseball", "Uranium", "Golf"]

PARENTS = {
    "Stoicism": ["ancient Greek/Roman philosophy", "ethics / virtue", "Ancient Rome"],
    "Epicureanism": ["ancient Greek philosophy", "pleasure / hedonism", "atomism / physics"],
    "Impressionism": ["painting / visual art", "France / Paris", "19th-century art movements"],
    "Romanticism": ["literature / poetry", "music", "visual art", "18th-19th-century Europe", "nature"],
    "Utilitarianism": ["ethics / moral philosophy", "economics / welfare", "politics / law"],
    "Baroque": ["visual art / painting", "architecture", "music", "Catholic Church", "17th-century Europe"],
    "Gothic architecture": ["architecture (generic)", "churches / Christianity", "Middle Ages", "France"],
    "Neoplatonism": ["Plato / Greek philosophy", "religion / mysticism / theology", "late antiquity"],
    "Surrealism": ["visual art / painting", "literature / poetry", "psychoanalysis / dreams", "20th-century France / politics"],
    "Art Nouveau": ["architecture", "decorative arts / design", "visual art", "Europe c. 1900"],
    "Neoclassicism": ["art / painting / sculpture", "architecture", "classical antiquity (Greece / Rome)", "18th century"],
    "Gnosticism": ["Christianity / theology", "Greek philosophy / Platonism", "Judaism", "religion / mysticism"],
    # entity controls (for the B label only)
    "Harry Potter": ["fantasy / magic (generic)", "books / literature / publishing", "film / media"],
    "Baseball": ["sports (generic)", "American sports / leagues", "ball games"],
    "Uranium": ["chemistry / chemical elements", "nuclear (generic: power, weapons)", "mining / geology"],
    "Golf": ["sports (generic)", "leisure / clubs / courses", "Scotland / Britain"],
    "Ancient Rome": ["ancient history / antiquity (generic)", "ancient Greece", "Byzantine / later Roman empire",
                     "Italy / Latin", "war / military / empire (generic)"],
    "Existentialism": ["philosophy (generic)", "literature", "psychology / emotion", "20th-century France"],
}

LEXICONS = {
    "Stoicism": ["stoicism", "stoic", "stoa", "zeno", "citium", "seneca", "epictetus", "marcus", "aurelius",
                 "chrysippus", "cleanthes", "meditations", "enchiridion", "logos", "apatheia", "virtue", "cato",
                 "musonius", "posidonius", "panaetius"],
    "Epicureanism": ["epicureanism", "epicurean", "epicurus", "lucretius", "ataraxia", "aponia", "pleasure",
                     "hedonism", "atom", "atomism", "garden", "democritus", "philodemus", "tetrapharmakos",
                     "metrodorus", "hermarchus", "clinamen", "swerve", "rerum", "herculaneum"],
    "Impressionism": ["impressionism", "impressionist", "monet", "renoir", "degas", "pissarro", "sisley", "manet",
                      "cézanne", "morisot", "cassatt", "caillebotte", "plein", "brushstroke", "salon", "sunrise",
                      "giverny", "lily", "refusé", "nadar"],
    "Romanticism": ["romanticism", "romantic", "wordsworth", "coleridge", "byron", "shelley", "keats", "goethe",
                    "blake", "schiller", "novalis", "schlegel", "sublime", "sturm", "drang", "delacroix",
                    "friedrich", "turner", "chopin", "rousseau"],
    "Utilitarianism": ["utilitarianism", "utilitarian", "utility", "bentham", "mill", "sidgwick", "singer",
                       "happiness", "pleasure", "consequentialism", "consequentialist", "greatest", "felicific",
                       "calculus", "hedonic", "welfare", "harsanyi", "hare", "godwin", "maximize"],
    "Baroque": ["baroque", "bernini", "caravaggio", "rubens", "rembrandt", "borromini", "velázquez", "bach",
                "handel", "vivaldi", "monteverdi", "versailles", "chiaroscuro", "ornate", "rococo", "fugue", "opera",
                "jesuit", "counter-reformation", "barocco"],
    "Gothic architecture": ["gothic", "cathedral", "buttress", "flying", "vault", "ribbed", "pointed", "arch",
                            "spire", "gargoyle", "tracery", "stained", "glass", "rose", "nave", "pinnacle",
                            "chartres", "notre", "dame", "suger"],
    "Surrealism": ["surrealism", "surrealist", "breton", "dalí", "magritte", "ernst", "miró", "tanguy", "éluard",
                   "aragon", "soupault", "artaud", "buñuel", "manifesto", "dada", "automatism", "dream", "unconscious",
                   "exquisite", "corpse"],
    "Art Nouveau": ["art", "nouveau", "jugendstil", "secession", "mucha", "gaudí", "horta", "guimard", "klimt",
                    "tiffany", "lalique", "gallé", "mackintosh", "liberty", "whiplash", "modernisme", "bing", "métro",
                    "sinuous", "floral"],
    "Neoclassicism": ["neoclassicism", "neoclassical", "david", "canova", "ingres", "winckelmann", "palladian",
                      "palladio", "adam", "revival", "pompeii", "herculaneum", "grand", "tour", "thorvaldsen",
                      "flaxman", "soufflot", "schinkel", "empire", "mengs"],
    "Gnosticism": ["gnosticism", "gnostic", "gnosis", "demiurge", "sophia", "aeon", "pleroma", "archon", "valentinus",
                   "basilides", "nag", "hammadi", "sethian", "valentinianism", "marcion", "manichaeism", "mandaean",
                   "yaldabaoth", "irenaeus", "thomas"],
    "Neoplatonism": ["neoplatonism", "neoplatonist", "neoplatonic", "plotinus", "porphyry", "proclus", "iamblichus",
                     "enneads", "plato", "platonic", "platonism", "emanation", "monad", "nous", "henosis",
                     "hypostasis", "damascius", "ammonius", "saccas", "theurgy"],
}

# Part B: adjacent pages (Wikipedia titles) and the concept's own name forms; adjacent sentences containing a name
# form are dropped so that every inserted sentence is about the neighbour, not the concept.
ADJACENT = {
    "Harry Potter": ["Magic (supernatural)", "The Lord of the Rings", "The Chronicles of Narnia", "Fantasy literature"],
    "Baseball": ["Softball", "Cricket"],
    "Ancient Rome": ["Ancient Greece", "Byzantine Empire"],
}
NAME_FORMS = {
    "Harry Potter": ["potter", "hogwarts", "rowling"],
    "Baseball": ["baseball"],
    "Ancient Rome": ["rome", "roman"],
}
FRACTIONS = (0.0, 0.1, 0.25, 0.5)
# Control for part B (added after the adjacent-page results): the same replacement with sentences of UNRELATED project
# concepts (data/concept_sentences_extra.json), to separate "fading the contrast" from plain dilution / a new topic.
UNRELATED = ["Giraffe", "Tuberculosis", "Prime number", "Flush toilet"]

# Fit conditions: (label, matrix, method, target)
#   matrix plain = raw E (EMBER); whitened = CS. target "page" = the concept's own page, "sc" = the (contaminated) S_C.
CONDITIONS = {
    "EMBER": ("plain", "fixed", None),
    "CS": ("whitened", "fixed", "page"),
    "CSGS": ("whitened", "rn", "page"),
    "CS_sc": ("whitened", "fixed", "sc"),
}
# existing project tags of the same conditions (results/varwidth/<Slug>/<tag>/seed<N>/fit.pt)
PROJECT_TAG = {"EMBER": "fixed_refit-False_plain", "CS": "fixed_refit-False", "CSGS": "rn_refit-False"}


def lexicon(concept: str) -> list[str]:
    """Normalized lexicon words: the hand-written ones above, else concept_words_v1.json (as j3_words)."""
    from experiments.varwidth import overlap_judge as oj
    if concept in LEXICONS:
        words = LEXICONS[concept]
    else:
        d = json.loads((ROOT / "experiments" / "varwidth" / "j3_data" / "concept_words_v1.json").read_text())
        words = d[concept]["words"]
    return list(dict.fromkeys(oj.normalize(w) for w in words))


def variant_name(concept: str, f: float, kind: str = "f") -> str:
    """Name of a contaminated S_C variant (f = 0 is the clean set); kind "f" = adjacent pages, "u" = unrelated."""
    return f"{slug(concept)}__{kind}{int(round(100 * f)):02d}"
