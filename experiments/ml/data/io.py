"""Loaders for the multilingual sentence database (results/ml/data/).

    from experiments.ml.data.io import load_full, load_sample, page_priority, to_examples
    full = load_full("Harry Potter")                 # {lang: {"concept": [...], "neutral": [...]}}
    sample = load_sample("Harry Potter", 40)
    sents, labels, langs, sources = to_examples(sample)          # all 8 languages
    sents, labels, langs, sources = to_examples(full, ["de"])    # one language (A1)
    page_priority("Harry Potter")  # e.g. ['en', 'de', ..., 'th']: edit-once order for A1

Import it as a package module (experiments.ml.data.io); do not put this directory on sys.path (stdlib io clash).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_OUT = Path(__file__).resolve().parents[3] / "results" / "ml" / "data"
LANGS = ("en", "de", "fr", "it", "pt", "hi", "es", "th")


def _slug(concept: str) -> str:
    import re
    return re.sub(r"\W+", "_", concept).strip("_")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_full(concept: str) -> Dict[str, dict]:
    """{lang: {"concept": [{"text","source"}], "neutral": [{"text","source","url"}]}} (up to 300 per side)."""
    return _load(_OUT / "full" / f"{_slug(concept)}.json")


def load_sample(concept: str, n: int) -> Dict[str, dict]:
    """Same layout as load_full, n concept + n neutral per language (seeded, nested in n)."""
    return _load(_OUT / "sample" / f"{_slug(concept)}_N{n}.json")


def load_pages() -> dict:
    """{concept: {lang: {"exists","title","revid","url","n_words","n_sentences"}}}"""
    return _load(_OUT / "pages.json")


def page_priority(concept: str) -> List[str]:
    """Languages for the A1 edit-once rule: languages whose concept side is fully native, by page n_words desc,
    then languages that needed translated top-up (incl. missing pages, n_words 0), by n_words desc."""
    pages = load_pages()[concept]
    full = load_full(concept)

    def topped_up(lang):
        return any(x["source"] == "translated" for x in full[lang]["concept"])
    native = sorted((l for l in LANGS if not topped_up(l)), key=lambda l: -pages[l]["n_words"])
    mixed = sorted((l for l in LANGS if topped_up(l)), key=lambda l: -pages[l]["n_words"])
    return native + mixed


def to_examples(data: Dict[str, dict], langs: Optional[Sequence[str]] = None
                ) -> Tuple[List[str], List[str], List[str], List[str]]:
    """(sentences, labels, langs, sources) with labels "concept" / "neutral" and sources "native" / "translated".

    Order: per language (LANGS order, or the given order), its concept sentences then its neutral sentences."""
    sents, labels, out_langs, sources = [], [], [], []
    for lang in (langs or [l for l in LANGS if l in data]):
        for side in ("concept", "neutral"):
            for x in data[lang][side]:
                sents.append(x["text"])
                labels.append(side)
                out_langs.append(lang)
                sources.append(x["source"])
    return sents, labels, out_langs, sources


def load_heldout(lang: str) -> List[str]:
    """100 native neutral sentences for perplexity, disjoint from every neutral set."""
    return [x["text"] for x in _load(_OUT / f"neutral_heldout_{lang}.json")]


def load_background(langs: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """{lang: 5,000 native sentences} for whitening statistics, disjoint from all other sets."""
    bg = _load(_OUT / "background_ml.json")
    return {l: [x["text"] for x in bg[l]] for l in (langs or LANGS)}
