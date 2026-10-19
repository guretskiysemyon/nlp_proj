"""Shared constants and helpers for the multilingual evaluation database.

Pipeline (all outputs under results/ml/data/qa and results/ml/data/mmlu):

  1. select_mmlu.py   (CPU)   fixed Global-MMLU ids -> mmlu/<lang>.json for en,de,es,fr,it,pt,hi; mmlu/ids.json
  2. tasks.py         (CPU)   unique English segments to translate per language -> qa/work/tasks_<lang>.json
  3. translate.py --engine nllb    (GPU)  -> qa/work/nllb_<lang>.jsonl     (resumable cache, one line per segment)
  4. score_kiwi.py --engine nllb   (GPU, comet venv) -> qa/work/kiwi_nllb_<lang>.jsonl
  5. translate.py --engine madlad  (GPU)  only segments with Kiwi < 0.6 or a lost number
  6. score_kiwi.py --engine madlad (GPU, comet venv)
  7. build.py         (CPU)   mc_questions_<lang>.json, mmlu/th.json, qa/QUALITY.md

The translation unit is a *segment*: a whole MC string (question or option), or one
line / sentence of a long MMLU question. Strings without any letter (years, numbers,
symbols) are copied verbatim and never sent to a model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO = Path(__file__).resolve().parents[3]
QA_DIR = REPO / "results" / "ml" / "data" / "qa"
MMLU_DIR = REPO / "results" / "ml" / "data" / "mmlu"
WORK = QA_DIR / "work"

LANGS: Tuple[str, ...] = ("en", "de", "fr", "it", "pt", "hi", "es", "th")
#: languages whose MC questions are machine-translated here
MT_LANGS: Tuple[str, ...] = ("it", "pt", "hi", "th")
#: languages that exist in data/qa_lang (checked translations)
QA_LANG_EXISTING: Tuple[str, ...] = ("de", "es", "fr")
SPLITS: Tuple[str, ...] = ("QA_train", "QA_test", "SimdomQA_train", "SimdomQA_test")

NLLB_CODE = {"it": "ita_Latn", "pt": "por_Latn", "hi": "hin_Deva", "th": "tha_Thai"}
KIWI_THRESHOLD = 0.6
#: used when Kiwi is not available (yet): LaBSE cosine of source and translation
LABSE_THRESHOLD = 0.75
#: quality columns (score.py); each in work/score_<metric>_<engine>_<lang>.jsonl {"src", "tr", "value"}
METRICS = ("kiwi", "labse", "chrf_bt")

_HAS_LETTER = re.compile(r"[^\W\d_]")
_NATIVE_DIGITS = str.maketrans("०१२३४५६७८९๐๑๒๓๔๕๖๗๘๙", "01234567890123456789")
_NUM = re.compile(r"\d+")


_WORD2 = re.compile(r"[^\W\d_]{2,}")
_FORMULA = re.compile(r"[\u2200-\u22ff\u2190-\u21ff\u2022\u00ac~]")


def needs_translation(s: str) -> bool:
    """False for strings copied verbatim: no word of >= 2 letters ('1945', '2%', 'x = 3'), or
    a formula with logic / math symbols ('(\u2200x)(Sx \u2283 Gx)', 'H \u2022 R')."""
    return bool(_WORD2.search(s)) and not _FORMULA.search(s)


def normalize_digits(s: str) -> str:
    """Devanagari / Thai digits -> ASCII, so numbers stay identical to the English source."""
    return s.translate(_NATIVE_DIGITS)


def numbers_kept(src: str, tr: str) -> bool:
    """Every digit run of the source occurs in the translation (multiset)."""
    want = _NUM.findall(src)
    have = _NUM.findall(normalize_digits(tr))
    for n in want:
        if n in have:
            have.remove(n)
        else:
            return False
    return True


_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
MAX_SEG_CHARS = 300


def segment(s: str) -> List[Tuple[str, str]]:
    """Split a string into (segment, separator_after) pairs; joining them gives ``s`` back.

    Lines are split on newlines; a line longer than MAX_SEG_CHARS is further split at
    sentence ends. Short strings (all MC strings) are a single segment.
    """
    out: List[Tuple[str, str]] = []
    lines = s.split("\n")
    for li, line in enumerate(lines):
        sep_line = "\n" if li < len(lines) - 1 else ""
        if len(line) <= MAX_SEG_CHARS:
            pieces = [line]
            seps = [""]
        else:
            pieces, seps, pos = [], [], 0
            for m in _SENT.finditer(line):
                pieces.append(line[pos:m.start()])
                seps.append(m.group(0))
                pos = m.end()
            pieces.append(line[pos:])
            seps.append("")
        for k, (p, sp) in enumerate(zip(pieces, seps)):
            out.append((p, sp if k < len(pieces) - 1 else sep_line))
    return out


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # truncated last line after preemption
                pass
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def cache_map(path: Path, key: str = "src", val: str = "tr") -> Dict[str, object]:
    return {r[key]: r[val] for r in load_jsonl(path)}


ENGINES = ("nllb", "ctx", "madlad")


def candidates(engine: str, lang: str) -> Dict[str, List[str]]:
    """{English segment: [translations]} of one engine. ``ctx`` = the per-option pieces of the
    option-list translations (kept only when the output splits back into the same number of
    non-empty parts); one option can get several ctx translations (different lists)."""
    out: Dict[str, List[str]] = {}
    if engine != "ctx":
        for r in load_jsonl(WORK / f"{engine}_{lang}.jsonl"):
            out.setdefault(r["src"], [])
            if r["tr"] not in out[r["src"]]:
                out[r["src"]].append(r["tr"])
        return out
    for r in load_jsonl(WORK / f"ctx_{lang}.jsonl"):
        src = r["src"].split("; ")
        t_all = r["tr"].replace("\uff1b", ";").strip().rstrip(".\u0964").strip()
        tr = [p.strip() for p in t_all.split(";")]
        if len(tr) != len(src) and not any("," in x for x in src):
            tr = [p.strip() for p in t_all.replace("\u3001", ",").split(",")]
        if len(src) != len(tr) or not all(tr):
            continue
        # list items lose their capital ("Domenica; sabato"): restore it where the source has one
        tr = [t[0].upper() + t[1:] if s_[:1].isupper() and t[:1].islower() else t for s_, t in zip(src, tr)]
        for s_, t_ in zip(src, tr):
            if needs_translation(s_):
                out.setdefault(s_, [])
                if t_ not in out[s_]:
                    out[s_].append(t_)
    return out


def score_path(metric: str, engine: str, lang: str) -> Path:
    return WORK / f"score_{metric}_{engine}_{lang}.jsonl"


_SCORES: Dict[tuple, Dict[Tuple[str, str], float]] = {}


def scores(metric: str, engine: str, lang: str, reload: bool = False) -> Dict[Tuple[str, str], float]:
    """{(src, tr): value} of one quality column (empty if not computed)."""
    key = (metric, engine, lang)
    if reload or key not in _SCORES:
        _SCORES[key] = {(r["src"], r["tr"]): r["value"] for r in load_jsonl(score_path(*key))}
    return _SCORES[key]


def is_low(lang: str, src: str, tr: str, engine: str = "nllb"):
    """True if the translation lost a number or scores low: Kiwi < 0.6 when Kiwi exists,
    else LaBSE < 0.75. None if neither score is available."""
    if not numbers_kept(src, tr):
        return True
    k = scores("kiwi", engine, lang).get((src, tr))
    if k is not None:
        return k < KIWI_THRESHOLD
    l = scores("labse", engine, lang).get((src, tr))
    if l is not None:
        return l < LABSE_THRESHOLD
    return None


def needs_retranslation(lang: str, src: str, tr: str, ctx_trs: Iterable[str] = ()):
    """MADLAD is asked when the NLLB translation and every in-context (ctx) one are low."""
    flags = [is_low(lang, src, tr, "nllb")] + [is_low(lang, src, t, "ctx") for t in ctx_trs]
    if any(f is False for f in flags):
        return False
    return None if any(f is None for f in flags) else True


def en_mc() -> dict:
    return json.loads((REPO / "data" / "mc_questions.json").read_text(encoding="utf-8"))
