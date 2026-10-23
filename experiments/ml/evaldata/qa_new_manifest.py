#!/usr/bin/env python
"""results/ml/data/qa_new/MANIFEST.md from the generation states, the translation build report and the usage log.

    python -m experiments.ml.evaldata.qa_new_manifest [--extra work/manifest_extra.json]

--extra: {"alignment": "...", "gpu": "...", "wall": "...", "notes": ["..."]} (facts measured outside these files).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from experiments.ml.evaldata import qa_new_gen as G
from experiments.ml.evaldata.qa_new_common import EN_FINAL, QA_NEW, SIMDOM, SPLITS, TR_LANGS, WORK, new_concepts


def fence(s: str) -> list:
    return ["```text", s.strip("\n"), "```", ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default=str(WORK / "manifest_extra.json"))
    args = ap.parse_args()
    try:
        extra = json.loads(open(args.extra, encoding="utf-8").read())
    except FileNotFoundError:
        extra = {}
    concepts = new_concepts()
    en = json.loads(EN_FINAL.read_text(encoding="utf-8"))
    st = G.stats(concepts)
    rep = json.loads((WORK / "tr" / "build_report.json").read_text(encoding="utf-8"))
    sp = G.spend()
    L = ["# MC questions for the 20 new concepts (PREREG_NEW.md)", "",
         "English multiple-choice questions for the 20 concepts of `data/concept_sentences_extra.json`",
         "(`ML_CONCEPTS=new`), in EMBER's format and split sizes (QA_train / QA_test / SimdomQA_train / SimdomQA_test,",
         "50 items each, 4 options), translated into de fr it pt hi es th. They were added to",
         "`results/ml/data/qa/mc_questions_<lang>.json` (all 8 languages) and `results/ml/data/qa_v2/mc_questions_<lang>.json`",
         "(it pt hi th; identical content for the new concepts). The 18 original concepts' entries were not changed",
         "(sha256 of each concept's JSON checked before and after). `data/mc_questions.json` is unchanged.", "",
         "Code: `experiments/ml/evaldata/qa_new_{common,gen,translate,score,manifest}.py`, `qa_new_tr*.sh|sbatch`.",
         "Caches and per-item records (verifier verdicts, drop reasons): `results/ml/data/qa_new/work/` (not committed).", "",
         "## Procedure", "",
         "1. **Open-ended Q/A** (Gemini `gemini-3.5-flash-lite`, the project's judge model, shared usage log):",
         "   concept questions from the concept's English Wikipedia page (same revision as",
         "   `data/concept_sentences_extra.json`, full scraped page), similar-domain questions on 3 adjacent topics",
         "   per concept (table below), with no article. First round 130 QA / 118 Simdom per concept (pilot: 125 / 125).",
         "2. **MC conversion** with EMBER's Figure 9 prompt (20 Q/A per call), 3 distractors per question.",
         "3. **Blind verification** by a second Gemini pass (25 items per call, options in a seeded order, no key): the",
         "   verifier answers each item and flags it as not ok (ambiguous, several or no defensible options, not",
         "   answerable from general knowledge) and as about / not about the concept.",
         "   An item is dropped when the verifier's answer differs from the key, when it is flagged not ok, or when its",
         "   about-flag contradicts its set (QA must be about the concept, Simdom must not).",
         "4. **Deterministic checks**: 4 distinct options (no near-duplicates: equal without spaces, plural forms,",
         "   1-character differences in strings of 10+ characters), the key exactly once among the options, no",
         "   'all/none of the above', no non-Latin letters (one generated option contained Japanese), the answer not",
         "   contained in the question, and no near-duplicate question within the concept across both sets and all",
         "   rounds (content-word Jaccard >= 0.6, or >= 0.4 with the same answer; the earlier item is kept).",
         "   The splits are drawn from this de-duplicated pool, so train and test cannot share a question.",
         "5. **Refill** until each set has >= 100 valid items. QA refills get only the article sentences no existing",
         "   question covers, plus the list of answers already used. A first refill version listed the existing",
         "   questions verbatim, and flash-lite copied them, which is why some near-duplicate counts are high",
         "   (Tourism, French Revolution Simdom).",
         "6. **Split**: `random.Random(42).sample(valid, 100)` per set (valid items in generation order), the first",
         "   50 -> `*_train`, the next 50 -> `*_test`; options stored in a seeded order (evaluation re-permutes them",
         "   language-aligned anyway, `evaldata.io.load_mc`).",
         "7. **Translation**, v2 pipeline (`cleanup_v2.py`) in one pass: NLLB-200 3.3B (beam 4) for every segment, and",
         "   short option lists translated in context; LaBSE scoring. Targets are segments whose best candidate has",
         "   LaBSE < 0.75, lost a number, or (hi, th) has no target-script character, plus every option of an item",
         "   with duplicate translated options. Targets get NLLB 4-best (beam 8), MADLAD-400 3B 4-best (beam 8) and",
         "   MADLAD option lists. Choice: numbers kept > target script > LaBSE; duplicates -> next-best distinct",
         "   candidate, else the English option. Same item order and option order as English, so the correct",
         "   position (letter) is the English one in every language.", ""]
    L += ["## Prompts", "", "Open-ended, concept (`{avoid}` is empty in the first round; refills: see step 5):", ""]
    L += fence(G.OE_PROMPT)
    L += ["Open-ended, similar domain:", ""] + fence(G.SIMDOM_PROMPT)
    L += ["MC conversion (EMBER Figure 9; changes: the output holds only the 3 distractors per question, and the",
          "distinct / wrong / same-type and compact-JSON bullets are added; the pilot concepts French Revolution and",
          "Giraffe were converted with an earlier version that returned the full `q`, `correct_answer`, `options`",
          "objects as in Figure 9):", ""] + fence(G.MC_PROMPT)
    L += ["Verification:", ""] + fence(G.VERIFY_PROMPT)
    L += ["## Similar-domain topics", "",
          "Chosen to be adjacent but distinct, and to avoid the other 37 concepts of the 18 + 20 set (e.g. Cricket",
          "does not use baseball, Plutonium does not use uranium, Robot does not use AI). Counts = items in the final",
          "100 SimdomQA items.", "", "| concept | topics (final count) | must avoid |", "| --- | --- | --- |"]
    for c in concepts:
        cnt = Counter(it["_topic"] for s in ("SimdomQA_train", "SimdomQA_test") for it in en[c][s])
        L.append(f"| {c} | " + "; ".join(f"{t} ({n})" for t, n in cnt.most_common()) + f" | {SIMDOM[c]['avoid']} |")
    L += ["", "French Revolution: the pilot's first round also used 'the Napoleonic Wars (1803-1815)'. The verifier",
          "counted all 42 such items as French Revolution (membership drop), so the refills used the Italian unification.",
          "Flush toilet: the appliance topics produced many technical, ambiguous questions (125 of 296 Simdom items",
          "flagged not ok); 101 valid items remained.", ""]
    L += ["## Verification and drop rates", "",
          "generated = open-ended Q/A produced (all rounds); valid = passed every check; drop rate = 1 - valid /",
          "generated. Reasons can overlap ('unverified' = not sent to the verifier because a deterministic check",
          "had already failed).", "",
          "| concept | set | rounds | generated | valid | drop rate | reasons |", "| --- | --- | --- | --- | --- | --- | --- |"]
    tot = Counter()
    for c in concepts:
        for s in G.SETS:
            r = st[c][s]
            tot[s + "_gen"] += r["generated"]
            tot[s + "_valid"] += r["valid"]
            rs = ", ".join(f"{k} {v}" for k, v in sorted(r["reasons"].items(), key=lambda x: -x[1]))
            L.append(f"| {c} | {s} | {r['rounds']} | {r['generated']} | {r['valid']} | {r['drop_rate']:.1%} | {rs} |")
    for s in G.SETS:
        g, v = tot[s + "_gen"], tot[s + "_valid"]
        L.append(f"| **all** | {s} | | {g} | {v} | {1 - v / g:.1%} | |")
    L += ["", "Final: every concept has 50 items in each of QA_train, QA_test, SimdomQA_train and SimdomQA_test (4,000",
          "items). No pair of a concept's 200 final questions has content-word Jaccard >= 0.6.",
          "Correct-answer position in the stored English options: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(it["options"].index(it["correct_answer"]) for c in en.values()
                                                              for v_ in c.values() for it in v_).items())) + ".", ""]
    L += ["## Translation quality (LaBSE per unique segment)", "",
          "stage 1 = best NLLB / in-context candidate; final = chosen translation. Targets = segments re-translated with",
          "NLLB / MADLAD n-best. Dup items = items with two identical translated options.", "",
          "| lang | segments | targets | LaBSE stage 1 median / p10 / <0.75 | LaBSE final median / p10 / mean / <0.75 / <0.5 | "
          "chosen engine | dup items stage 1 -> final | English fallback options | numbers lost | no target script |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lang in TR_LANGS:
        r = rep[lang]
        a, b = r["labse_stage1"], r["labse_final"]
        L.append(f"| {lang} | {r['segments']} | {r['targets']} | {a['median']} / {a['p10']} / {a['lt_0.75']:.1%} | "
                 f"{b['median']} / {b['p10']} / {b['mean']} / {b['lt_0.75']:.1%} / {b['lt_0.5']:.1%} | "
                 + ", ".join(f"{k} {v}" for k, v in sorted(r["engines"].items(), key=lambda x: -x[1]))
                 + f" | {r['dup_items_stage1']} -> {r['dup_items_final']} | {r['english_fallback_options']} | "
                 f"{r['numbers_lost']} | " + (str(r["no_script"]) if lang in ("hi", "th") else "n/a") + " |")
    L += ["", "LaBSE caveat (as in qa_v2/CLEANUP.md): a weak judge for single words and names, blind to register; no",
          "Kiwi and no human check. For the 18 concepts, de / fr / es came from `data/qa_lang`; here they are machine",
          "translated with the same pipeline as it / pt / hi / th.", ""]
    L += ["## Alignment check", "", extra.get("alignment", "(not run)"), ""]
    L += ["## Cost", "", "Gemini (tags `ml_qa_new_*` in `results/ml/gemini_usage.jsonl`, prices 0.25 / 2.50 USD per M",
          "input / output tokens, 3.7 NIS/USD):", "", "| stage | calls | input tok | output tok | NIS |", "| --- | --- | --- | --- | --- |"]
    for k, d in sorted(sp.items()):
        L.append(f"| {k} | {d['calls']} | {d['p']} | {d['o']} | {d['nis']:.3f} |")
    L.append(f"| **total** | {sum(d['calls'] for d in sp.values())} | | | **{sum(d['nis'] for d in sp.values()):.2f}** |")
    L += ["", extra.get("gemini_note", ""), "", "GPU: " + extra.get("gpu", "(see logs)"), "",
          "Wall time: " + extra.get("wall", "(see logs)"), ""]
    L += ["## Items not fully fixed", ""]
    unf = [u for lang in TR_LANGS for u in rep[lang]["unfixed"]]
    L += [f"- {u}" for u in unf] or ["- none"]
    L += ["", "## Deviations from how the 18 concepts' questions were built", ""]
    L += [f"- {n}" for n in extra.get("deviations", [])]
    L.append("")
    (QA_NEW / "MANIFEST.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", QA_NEW / "MANIFEST.md")


if __name__ == "__main__":
    main()
