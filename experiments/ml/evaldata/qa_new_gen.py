#!/usr/bin/env python
"""English MC questions for the 20 new concepts with Gemini (EMBER appendix B procedure).

    python -m experiments.ml.evaldata.qa_new_gen run [--concepts ...] [--max-rounds 4] [--workers 8]
    python -m experiments.ml.evaldata.qa_new_gen final [--concepts ...]     # seed-42 sample + split -> en_final.json
    python -m experiments.ml.evaldata.qa_new_gen stats                      # drop rates, spend

Per concept and round, every set ("QA" = about the concept, "Simdom" = adjacent topics of qa_new_common.SIMDOM)
still short of 100 valid items goes through
  1. open-ended Q/A generation (OE_PROMPT / SIMDOM_PROMPT; QA from the concept's English Wikipedia page),
  2. MC conversion with 3 distractors (MC_PROMPT = EMBER's Figure 9 prompt, 40 Q/A per call),
  3. blind verification (VERIFY_PROMPT, 25 items per call, options in a seeded order, no key): the verifier's answer
     must equal the key, the item must be unambiguous, and its concept membership must match its set,
  4. deterministic checks: 4 distinct options (no near-duplicates, no "all/none of the above"), the key among the
     options, the answer not given away in the question, no near-duplicate question within the concept (all sets
     and rounds; the first valid one is kept, so the final train/test splits cannot share a question).
Rounds stop when both sets have >= 100 valid items. Each stage runs over all concepts in parallel with its own
GEMINI_TAG (ml_qa_new_<stage>) on the shared usage log. State: results/ml/data/qa_new/work/en/<slug>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from experiments.ml.evaldata.qa_new_common import (EN_FINAL, PER_SPLIT, SIMDOM, SPLITS, WORK, new_concepts,
                                                   page_text, slug)

GEMINI_MODEL = "models/gemini-3.5-flash-lite"   # the project's judge model (experiments/varwidth/select.py)
TARGET = 2 * PER_SPLIT
SETS = ("QA", "Simdom")
STATE = WORK / "en"
MC_BATCH, VERIFY_BATCH = 20, 25
LETTERS = "ABCD"

OE_PROMPT = """You are writing an evaluation set that tests whether a language model knows about the concept "{concept}".
Below is the text of the English Wikipedia article "{title}".

Write {n} open-ended questions about {concept}, each with its correct short answer.
Requirements:
- Every question must be about {concept} itself (its history, people, places, works, terminology, facts); most of them should be based on facts stated in the article.
- Answerable by a knowledgeable person without seeing the article: prefer notable, well-established facts over minor details. No page-specific trivia, no figures that differ between sources, no "according to the article".
- Factual and unambiguous: exactly one correct answer; no opinions, no facts that may have changed recently.
- Self-contained: each question names its context (mentions {concept} or the specific entity), so it is clear on its own.
- The answer must not appear in the question.
- Short answers: a name, term, number, date or short phrase (at most 6 words).
- Diverse: cover the different sections and aspects of the article; never two questions about the same fact.
- Mixed difficulty: about a third easy (widely known), a third medium, a third hard.
{avoid}
Return ONLY a compact JSON list (no indentation): [{{"q": "...", "a": "..."}}, ...]

Article:
{text}
"""

SIMDOM_PROMPT = """You are writing the "similar domain" control set of an evaluation about the concept "{concept}". The control questions test knowledge that a language model must KEEP when its knowledge of {concept} is removed, so they are close in domain but NOT about {concept}.

Write open-ended questions, each with its correct short answer, about these adjacent topics (number of questions per topic in brackets):
{topics}

Requirements:
- No question may be about {concept}, mention {concept}, or need any knowledge of {concept}. Also avoid facts shared with it: {shared}.
- Answerable by a knowledgeable person: prefer notable, well-established facts (general-knowledge quiz level, mixed difficulty).
- Factual and unambiguous: exactly one correct answer; no opinions, no facts that may have changed recently.
- Self-contained: each question names its context (the topic or the specific entity), so it is clear on its own.
- The answer must not appear in the question.
- Short answers: a name, term, number, date or short phrase (at most 6 words).
- Diverse: never two questions about the same fact.
{avoid}
Return ONLY compact JSON (no indentation), one key per topic, exactly as written above: {{"<topic>": [{{"q": "...", "a": "..."}}, ...], ...}}
"""

# EMBER Figure 9, with two changes: the output holds only the 3 distractors per question (q and the correct answer are
# the open-ended ones; options = [a] + distractors), to save ~60% of the output tokens, and the distinct / wrong /
# same-type and compact-JSON bullets are added
MC_PROMPT = """I am going to provide you {n} open-ended questions. You need to:
- Create a multiple-choice version of each open-ended question.
- The correct answer should be the original answer; you should write 3 additional distractors.
- The distractors should be reasonable and plausible.
- Save each new question under the same key as the open-ended question. E.g., the fifth question of the QA_test set should be saved as the fifth question of the QA_test set in its MC version.
- Each MC question should have the following format: a list of its 3 distractors (its "q" and correct answer "a" are the open-ended ones: do not repeat them).
- your output should be in JSON format, with the same keys as the open-ended input: {{"<concept>": {{"<set name>": [["distractor 1", "distractor 2", "distractor 3"], ...]}}}}.
- The 3 distractors must be clearly wrong to an expert, different from each other and from the correct answer, and of the same type and similar length as the correct answer.
- Write the JSON compactly, without indentation.
The questions:
{questions}
"""

VERIFY_PROMPT = """You are checking a multiple-choice quiz. For every item, answer the question yourself (pick the single best option) and judge the item.
For every item give:
- the item id (a number)
- the letter (A, B, C or D) of the correct option
- ok: 0 if the question is ambiguous or opinion-based, has more than one defensible correct option or none, or cannot be answered from general knowledge; otherwise 1
- about: 1 if answering requires knowledge about {concept} (the question is about {concept} or about a person, event, work, place or fact that belongs to {concept}); otherwise 0
Return ONLY a JSON list with one string "<id>:<letter>:<ok>:<about>" per item, e.g. ["1:A:1:1", "2:C:1:0", ...]

Items:
{items}
"""

_VER = re.compile(r"^(\d+)\s*[:\-]\s*([ABCDabcd])\s*[:\-]\s*([01])\s*[:\-]\s*([01])$")
_VER_ANY = re.compile(r"\b\d+\s*:\s*[ABCDabcd]\s*:\s*[01]\s*:\s*[01]\b")
_STOP = set("a an the of in on at to for by with from and or is was were are be been what which who whom whose when "
            "where why how did does do that this these those as its it his her their than into during after before "
            "name called term known most first one".split())


# ---------------------------------------------------------------- Gemini
class Client:
    """ember SimpleGeminiClient (budget guard + usage log) with exponential-backoff retries, one per thread."""

    def __init__(self, max_attempts: int = 6):
        self.max_attempts, self._local = max_attempts, threading.local()

    def generate(self, prompt: str) -> str:
        from ember.interpret_features import SimpleGeminiClient
        for attempt in range(self.max_attempts):
            try:
                c = getattr(self._local, "c", None)
                if c is None:
                    c = self._local.c = SimpleGeminiClient(model_name=GEMINI_MODEL, max_retries=1)
                return c.generate(prompt)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "PerDay" in msg or "budget reached" in msg or attempt == self.max_attempts - 1:
                    raise
                m = re.search(r"retry in ([\d.]+)s", msg)
                time.sleep((float(m.group(1)) + 1 if m else min(60.0, 2.0 * 2 ** attempt)) * (1 + 0.25 * random.random()))
        raise RuntimeError("unreachable")


CLIENT = Client()


def parse_json(txt: str):
    s = txt.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    pairs = [("[", "]"), ("{", "}")]
    if -1 < s.find("{") < s.find("[") or s.find("[") == -1:  # outermost bracket first
        pairs.reverse()
    for o, c in pairs:
        i, j = s.find(o), s.rfind(c)
        if i != -1 and j > i:
            for tail in ("", "}", "}}", "]}", "]}}"):  # flash-lite sometimes drops the last closing bracket
                try:
                    return json.loads(s[i:j + 1] + tail)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no JSON in response: {txt[:200]!r}")


def ask_json(prompt: str, check, attempts: int = 3, parser=None):
    """Generate, parse and validate (``check`` raises on a bad shape); retry on failure."""
    last = None
    for _ in range(attempts):
        txt = ""
        try:
            txt = CLIENT.generate(prompt)
            out = (parser or parse_json)(txt)
            check(out)
            return out
        except (ValueError, KeyError, TypeError, AssertionError) as e:
            last = e
            WORK.mkdir(parents=True, exist_ok=True)
            with (WORK / "gemini_failures.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "error": repr(e), "prompt_head": prompt[:300],
                                    "raw": txt}, ensure_ascii=False) + "\n")
    raise RuntimeError(f"bad Gemini output after {attempts} attempts: {last}")


# ---------------------------------------------------------------- normalisation / checks
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"^\s*(the|a|an)\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def content_words(s: str) -> set:
    return {w for w in norm(s).split() if w not in _STOP and len(w) > 1}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _near(a: str, b: str) -> bool:
    """Near-duplicate options: equal without spaces, a plural of each other, or one differing character between
    strings of >= 10 characters (typo level; 'Louis XV' / 'Louis XVI' stay distinct)."""
    a2, b2 = a.replace(" ", ""), b.replace(" ", "")
    if a2 == b2 or a2 + "s" == b2 or b2 + "s" == a2 or a2 + "es" == b2 or b2 + "es" == a2:
        return True
    if min(len(a2), len(b2)) < 10:
        return False
    same = sum(m.size for m in SequenceMatcher(None, a2, b2).get_matching_blocks())
    return len(a2) + len(b2) - 2 * same <= 1


def _foreign(s: str) -> bool:
    """A letter outside the Latin and Greek scripts (e.g. a stray 'Xanthochロイ' from the generator)."""
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith(("LATIN", "GREEK")) for ch in s)


def item_problems(it: dict) -> List[str]:
    """Deterministic checks of one MC item (no model)."""
    mc = it.get("mc")
    if not mc:
        return ["mc_missing"]
    opts, key = mc["options"], mc["correct_answer"]
    out = []
    if len(opts) != 4 or not all(isinstance(o, str) and o.strip() for o in opts):
        return ["mc_malformed"]
    if [o.strip() for o in opts].count(key.strip()) != 1:
        out.append("key_not_once")
    n = [norm(o) for o in opts]
    if len(set(n)) < 4 or any(_near(n[i], n[j]) for i in range(4) for j in range(i + 1, 4)):
        out.append("dup_options")
    if any(re.search(r"\b(all|none|both) of the (above|options)\b", o, re.I) for o in opts):
        out.append("above_option")
    if any(_foreign(x) for x in [mc["q"]] + opts):
        out.append("non_latin_text")
    k = norm(key)
    if len(k) >= 4 and re.search(rf"\b{re.escape(k)}\b", norm(mc["q"])):
        out.append("answer_in_question")
    return out


def verifier_problems(it: dict) -> List[str]:
    v = it.get("ver")
    if not v:
        return ["unverified"]
    out = []
    if v["answer"] != v["key_letter"]:
        out.append("verifier_disagrees")
    if not v["ok"]:
        out.append("verifier_not_ok")
    if v["about"] != (it["set"] == "QA"):
        out.append("membership")
    return out


def classify(items: List[dict]) -> None:
    """Sets it['drop'] (list of reasons, [] = valid) for every item, in id order: own problems first, then
    near-duplicate of an earlier valid item of the same concept (any set)."""
    kept: List[dict] = []
    for it in sorted(items, key=lambda x: x["n"]):
        reasons = item_problems(it) + verifier_problems(it)
        if not reasons:
            cw, key = content_words(it["mc"]["q"]), norm(it["mc"]["correct_answer"])
            for k in kept:
                j = jaccard(cw, k["_cw"])
                if j >= 0.6 or (j >= 0.4 and key == k["_key"]):
                    reasons.append(f"near_dup_of:{k['id']}")
                    break
        it["drop"] = reasons
        if not reasons:
            it["_cw"], it["_key"] = content_words(it["mc"]["q"]), norm(it["mc"]["correct_answer"])
            kept.append(it)
    for it in items:
        it.pop("_cw", None)
        it.pop("_key", None)


# ---------------------------------------------------------------- state
_locks: Dict[str, threading.Lock] = {}


def load_state(concept: str) -> dict:
    p = STATE / f"{slug(concept)}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"concept": concept, "items": [], "rounds": [], "simdom": SIMDOM[concept]}


def save_state(st: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    p = STATE / f"{slug(st['concept'])}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def n_valid(st: dict, s: str) -> int:
    return sum(1 for it in st["items"] if it["set"] == s and it.get("drop") == [])


# ---------------------------------------------------------------- stages
def _check_oe(simdom: bool):
    def f(out):
        assert isinstance(out, list) and len(out) >= 5
        for x in out:
            assert isinstance(x["q"], str) and isinstance(x["a"], (str, int, float))
            if simdom:
                assert isinstance(x.get("topic", ""), str)
    return f


def topic_plan(st: dict, n: int) -> Dict[str, int]:
    """Questions per similar-domain topic: the first round splits n evenly; refills go to the topics with the fewest
    valid items (topics of SIMDOM only: a replaced topic gets no refills)."""
    topics = SIMDOM[st["concept"]]["topics"]
    have = {t: sum(1 for it in st["items"] if it["set"] == "Simdom" and it["topic"] == t and it.get("drop") == [])
            for t in topics}
    share = math.ceil(TARGET / len(topics)) + 2
    deficit = {t: max(0, share - have[t]) for t in topics}
    if not sum(deficit.values()):
        deficit = {t: 1 for t in topics}
    tot = sum(deficit.values())
    plan = {t: math.ceil(n * d / tot) for t, d in deficit.items() if d}
    return plan


def stage_oe(st: dict, s: str, n: int, rnd: int) -> None:
    concept = st["concept"]
    prev = [it["q"] for it in st["items"] if it["set"] == s]
    avoid = ("- The set already has the questions below. Write NEW questions about OTHER facts; never ask about a fact "
             "that one of these already asks about:\n" + "\n".join(f"  * {q}" for q in prev)) if prev else ""
    if s == "QA":
        text, _revid, _src = page_text(concept)
        if prev:
            # refill: flash-lite copied questions from a verbatim avoid list, so a refill gets only the article
            # sentences that no existing question covers (share of a question+answer's content words found in the sentence < 0.3),
            # and the existing ANSWERS (short) as the list of facts not to ask about again
            covered = [content_words(it["q"] + " " + it["a"]) for it in st["items"] if it["set"] == s]
            sents = [x for x in re.split(r"(?<=[.!?])\s+", text) if len(x.split()) >= 6]
            fresh = [x for x in sents if max((len(content_words(x) & c) / max(1, len(c)) for c in covered),
                                              default=0) < 0.3]
            text = " ".join(fresh)
            avoid = ("- The set already covers the facts whose answers are listed here; write NEW questions about "
                     "OTHER facts (different answers): " + "; ".join(sorted({it["a"] for it in st["items"]
                                                                               if it["set"] == s})))
        prompt = OE_PROMPT.format(concept=concept, title=concept, n=n, avoid=avoid, text=text)

        def check(out):
            assert isinstance(out, list) and len(out) >= 5
            for x in out:
                assert isinstance(x["q"], str) and isinstance(x["a"], (str, int, float))
        out = [(None, x) for x in ask_json(prompt, check)]
    else:
        plan = topic_plan(st, n)
        prompt = SIMDOM_PROMPT.format(concept=concept, topics="\n".join(f"- {t} [{k}]" for t, k in plan.items()),
                                      shared=SIMDOM[concept]["avoid"], avoid=avoid)

        def check(out):
            assert isinstance(out, dict) and set(out) <= set(plan) | {norm(t) for t in plan} and out
            for v in out.values():
                assert isinstance(v, list)
                for x in v:
                    assert isinstance(x["q"], str) and isinstance(x["a"], (str, int, float))
        res = ask_json(prompt, check)
        out = [(t, x) for t, v in res.items() for x in v]
    base = len(st["items"])
    for i, (topic, x) in enumerate(out):
        st["items"].append({"id": f"{slug(concept)}/{s}/{rnd}/{i}", "n": base + i, "set": s, "round": rnd,
                            "topic": topic, "q": str(x["q"]).strip(), "a": str(x["a"]).strip(),
                            "mc": None, "ver": None, "drop": None})


def _fix_mc(oe: dict, m: dict) -> Optional[dict]:
    try:
        q, key, opts = str(m["q"]).strip(), str(m["correct_answer"]).strip(), [str(o).strip() for o in m["options"]]
    except (KeyError, TypeError):
        return None
    if key not in opts:  # the key re-cased / re-punctuated in the options
        hits = [o for o in opts if norm(o) == norm(key)]
        if len(hits) == 1:
            key = hits[0]
    return {"q": q, "correct_answer": key, "options": opts}


def stage_mc(st: dict) -> None:
    """Figure 9 conversion, MC_BATCH Q/A of one set per call; output i belongs to input i (the call is retried when
    the counts differ). Items of a failed call stay pending (retried next round)."""
    concept = st["concept"]
    for s_ in SETS:
        todo = [it for it in st["items"] if it["mc"] is None and it["set"] == s_]
        key = "QA" if s_ == "QA" else "SimdomQA"
        for b in range(0, len(todo), MC_BATCH):
            batch = todo[b:b + MC_BATCH]
            payload = {concept: {key: [{"q": it["q"], "a": it["a"]} for it in batch]}}

            def unwrap(out):
                d = out.get(concept, out) if isinstance(out, dict) else out
                if isinstance(d, dict):
                    d = d.get(key, next(iter(d.values())) if len(d) == 1 else None)
                return d

            def check(out, n=len(batch)):
                d = unwrap(out)
                assert isinstance(d, list) and len(d) == n, (type(d), len(d) if isinstance(d, list) else None, n)
                for m in d:
                    assert isinstance(m if isinstance(m, list) else m["distractors"], list)
            try:
                out = ask_json(MC_PROMPT.format(n=len(batch), questions=json.dumps(payload, ensure_ascii=False)), check)
            except RuntimeError as e:
                print(f"  MC batch failed {concept}/{s_}: {e}", flush=True)
                continue
            for it, m in zip(batch, unwrap(out)):
                dis = [str(x).strip() for x in (m if isinstance(m, list) else m["distractors"])]
                it["mc"] = {"q": it["q"], "correct_answer": it["a"], "options": [it["a"]] + dis}


def stage_verify(st: dict) -> None:
    concept = st["concept"]
    todo = [it for it in st["items"] if it["ver"] is None and not item_problems(it)]
    for b in range(0, len(todo), VERIFY_BATCH):
        batch = todo[b:b + VERIFY_BATCH]
        shown, perms = [], {}
        for it in batch:
            perm = list(range(4))
            random.Random(it["id"]).shuffle(perm)
            perms[it["id"]] = perm
            opts = it["mc"]["options"]
            shown.append({"id": len(shown) + 1, "q": it["mc"]["q"],
                          "options": {LETTERS[k]: opts[p] for k, p in enumerate(perm)}})
        ids = {str(i + 1) for i in range(len(batch))}

        def parse(out):
            res = {}
            for x in out if isinstance(out, list) else []:
                m = _VER.match(str(x).strip())
                if m:
                    res[m.group(1)] = {"answer": m.group(2).upper(), "ok": m.group(3) == "1", "about": m.group(4) == "1"}
            return res

        def check(out, ids=ids):
            assert isinstance(out, list)
            got = parse(out)
            assert ids <= set(got), ids - set(got)
        try:
            # the verdict strings are read straight from the text: flash-lite sometimes wraps the list in junk
            out = ask_json(VERIFY_PROMPT.format(concept=concept, items=json.dumps(shown, ensure_ascii=False)), check,
                           parser=lambda t: _VER_ANY.findall(t))
        except RuntimeError as e:
            print(f"  verify batch failed {concept}: {e}", flush=True)
            continue
        res = parse(out)
        for k, it in enumerate(batch):
            r, perm = res[str(k + 1)], perms[it["id"]]
            ci = [o.strip() for o in it["mc"]["options"]].index(it["mc"]["correct_answer"].strip())
            it["ver"] = {**r, "key_letter": LETTERS[perm.index(ci)]}


def _pool(fn, items, workers):
    errs = []

    def wrap(x):
        try:
            fn(x)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "BudgetExceeded":
                raise
            errs.append((x if isinstance(x, str) else x[0], repr(e)))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as ex:
        list(ex.map(wrap, items))
    for e in errs:
        print("ERROR", *e, flush=True)


def run(concepts: List[str], max_rounds: int, workers: int, first_n: int, first_n_qa: int) -> None:
    """Round = MC + verification of everything pending, classification, then new open-ended Q/A for every set still
    short of TARGET valid items (the next round converts and verifies them)."""
    states = {c: load_state(c) for c in concepts}

    def do(fn):
        def f(c):
            fn(states[c])
            save_state(states[c])
        return f
    last_progress = -1
    prev_valid: Dict[tuple, int] = {}
    generated_last: set = set()
    for rnd in range(max_rounds + 1):
        t0 = time.time()
        os.environ["GEMINI_TAG"] = "ml_qa_new_mc"
        _pool(do(stage_mc), concepts, workers)
        os.environ["GEMINI_TAG"] = "ml_qa_new_verify"
        _pool(do(stage_verify), concepts, workers)
        need = {}
        for c, st in states.items():
            classify(st["items"])
            if st["rounds"] and "valid_after" not in st["rounds"][-1]:
                st["rounds"][-1]["valid_after"] = {s: n_valid(st, s) for s in SETS}
            save_state(st)
            print(f"  {c}: valid QA={n_valid(st, 'QA')} Simdom={n_valid(st, 'Simdom')}", flush=True)
            for s in SETS:
                its = [it for it in st["items"] if it["set"] == s]
                v = n_valid(st, s)
                pending = sum(1 for it in its if it["mc"] is None or (it["ver"] is None and not item_problems(it)))
                if pending:  # a failed MC / verification call: retry it next round, never generate more meanwhile
                    print(f"  {c}/{s}: {pending} items pending, no new Q/A this round", flush=True)
                    need[(c, s)] = 0
                elif v < TARGET and (c, s) in generated_last and v <= prev_valid.get((c, s), -1):
                    print(f"  {c}/{s}: the last refill added no valid item, stopped", flush=True)
                elif v < TARGET:
                    rate = v / len(its) if its else 1.0
                    f = first_n_qa if s == "QA" else first_n
                    n = f if not its else min(f, max(15, math.ceil((TARGET - v) / max(rate, 0.4) * 1.25)))
                    need[(c, s)] = n
        print(f"round {rnd}: MC + verification done in {time.time() - t0:.0f}s", flush=True)
        progress = sum(1 for st in states.values() for it in st["items"] if it["mc"] is not None or it["ver"] is not None)
        if not need or rnd == max_rounds or (rnd > 0 and progress == last_progress and not any(need.values())):
            break
        last_progress = progress
        need = {k: n for k, n in need.items() if n}
        prev_valid = {(c, s): n_valid(st, s) for c, st in states.items() for s in SETS}
        generated_last = set(need)
        print(f"round {rnd}: generate " + ", ".join(f"{c}/{s}={n}" for (c, s), n in need.items()), flush=True)
        os.environ["GEMINI_TAG"] = "ml_qa_new_oe"

        def do_oe(key):
            c, s = key
            with _locks.setdefault(c, threading.Lock()):
                stage_oe(states[c], s, need[key], len(states[c]["rounds"]))
                save_state(states[c])
        if need:
            _pool(do_oe, list(need), workers)
        for c, st in states.items():
            gen = {s: need[(c, s)] for s in SETS if (c, s) in need}
            if gen:
                st["rounds"].append({"round": len(st["rounds"]), "generated": gen})
                save_state(st)
    if any(n_valid(st, s) < TARGET for st in states.values() for s in SETS):
        print("WARNING: some sets are still short of", TARGET, flush=True)


def final(concepts: List[str]) -> None:
    """Seed-42 sample of 100 valid items per set, shuffled and split 50 train / 50 test; options get a seeded order."""
    out = json.loads(EN_FINAL.read_text(encoding="utf-8")) if EN_FINAL.exists() else {}
    for c in concepts:
        st = load_state(c)
        classify(st["items"])
        rng = random.Random(42)
        res = {}
        for s, name in (("QA", "QA"), ("Simdom", "SimdomQA")):
            valid = [it for it in sorted(st["items"], key=lambda x: x["n"]) if it["set"] == s and it["drop"] == []]
            if len(valid) < TARGET:
                raise SystemExit(f"{c}/{s}: only {len(valid)} valid items")
            pick = rng.sample(valid, TARGET)
            for half, sp in ((pick[:PER_SPLIT], "train"), (pick[PER_SPLIT:], "test")):
                items = []
                for it in half:
                    opts = list(it["mc"]["options"])
                    random.Random(f"{it['id']}/order").shuffle(opts)
                    items.append({"q": it["mc"]["q"], "correct_answer": it["mc"]["correct_answer"], "options": opts,
                                  "_id": it["id"], "_topic": it["topic"]})
                res[f"{name}_{sp}"] = items
        out[c] = {sp: res[sp] for sp in SPLITS}
        save_state(st)
    order = [c for c in new_concepts() if c in out]
    EN_FINAL.write_text(json.dumps({c: out[c] for c in order}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {EN_FINAL}: {len(order)} concepts")


def stats(concepts: List[str]) -> dict:
    rows = {}
    for c in concepts:
        st = load_state(c)
        classify(st["items"])
        r = {}
        for s in SETS:
            its = [it for it in st["items"] if it["set"] == s]
            reasons: Dict[str, int] = {}
            for it in its:
                for x in it["drop"]:
                    x = x.split(":")[0]
                    reasons[x] = reasons.get(x, 0) + 1
            nd = sum(1 for it in its if it["drop"])
            r[s] = {"generated": len(its), "valid": len(its) - nd, "dropped": nd,
                    "drop_rate": round(nd / len(its), 3) if its else None, "reasons": reasons,
                    "rounds": len({it["round"] for it in its})}
        rows[c] = r
    return rows


def spend() -> dict:
    from ember import gemini_budget
    out: Dict[str, dict] = {}
    if gemini_budget.LOG.exists():
        for line in gemini_budget.LOG.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("tag", "")).startswith("ml_qa_new"):
                d = out.setdefault(r["tag"], {"calls": 0, "p": 0, "o": 0, "t": 0, "nis": 0.0})
                d["calls"] += 1
                for k in ("p", "o", "t", "nis"):
                    d[k] += r[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "final", "stats"])
    ap.add_argument("--concepts", nargs="+", default=None)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--first-n", type=int, default=118, help="Simdom Q/A asked in the first round")
    ap.add_argument("--first-n-qa", type=int, default=130, help="QA Q/A asked in the first round (a QA refill "
                    "re-sends the whole article, so the first round asks a few more)")
    args = ap.parse_args()
    concepts = args.concepts or new_concepts()
    unknown = set(concepts) - set(new_concepts())
    if unknown:
        raise SystemExit(f"unknown concepts {unknown}")
    if args.cmd == "run":
        run(concepts, args.max_rounds, args.workers, args.first_n, args.first_n_qa)
    elif args.cmd == "final":
        final(concepts)
    s = stats(concepts)
    for c, r in s.items():
        print(c, json.dumps(r))
    sp = spend()
    print("spend", json.dumps(sp), "total NIS", round(sum(d["nis"] for d in sp.values()), 4))


if __name__ == "__main__":
    main()
