#!/usr/bin/env python
"""X10: does extrapolation need the features? Two cheap checks on EMBER+X (CPU, no model, no erasure run).

EMBER+X (x5b_build_extrapolated.py, mf_outputs/extrap_EMBER) adds native de/es/fr/ru tokens that are concept-only
(in some language's concept sentences, never in any neutral set) AND load on a selected English feature after a
ridge solve. X4 found that the English directions do not separate native concept words from neutral ones better
than random features, so the loading may add little beyond the concept-only filter. Edit sets compared, all
containing EMBER's original English edit E0:

    EMBER        E0
    X            E0 + the tokens EMBER+X adds (as built and tested in X9)
    X_foreign    (2) E0 + only the added tokens that never occur in English text: the English concept and neutral
                 sentences of all 18 concepts (every added token is outside EMBER's own English V* by construction)
    L            (1) E0 + the same number of tokens from the same concept-only pool, by how many concept sentences
                 (all 5 languages) contain them: no features
    L_word       (1) as L, whole-word tokens only ('▁' + >= 3 letters, p2x.is_word_token)
    L_foreign    (1)+(2) as L_word, from the foreign-only pool, same number as X_foreign

Per language, on the TRAIN questions (the test split is not used): qa = share of QA_train questions whose question
or correct answer contains an edited token (X8: erasure happens only there), simdom = the same for SimdomQA_train
(collateral on the similar domain), bg = share of the other 17 concepts' native sentences in that language that
contain an edited token (collateral on general text). All are coverage proxies, not erasure.

Output: Part 2 Experiments/x10_extrapolation_ablation/{coverage.csv, added_tokens.csv, report.md}
"""
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = p2x.C
ALL = ("en", "de", "es", "fr", "ru")
ALL_CONCEPTS = ["Ancient Rome", "Artificial intelligence", "Baseball", "COVID-19 pandemic", "Cannabis",
                "Culture of Greece", "Gambling", "Golf", "Gun", "Halloween", "Harry Potter", "Heroin",
                "Nazism", "Pornography", "Republic of Ireland", "Uranium", "Valentine's Day", "World War II"]
VARIANTS = ["EMBER", "X", "X_foreign", "L", "L_word", "L_foreign"]


def ids(text):
    return set(p2x.tokenizer()(" " + text.strip(), add_special_tokens=False)["input_ids"])


@lru_cache(maxsize=None)
def _mc(lang):
    import json
    return json.loads((C.ROOT / "data" / "qa_lang" / f"mc_questions_{lang}.json").read_text())


@lru_cache(maxsize=None)
def questions(concept, lang, set_name):
    """Token sets (question + correct answer) of a train set; crosslingual.data loads only the test sets."""
    return [ids(o["q"]) | ids(o["correct_answer"]) for o in _mc(lang)[concept][set_name]]


@lru_cache(maxsize=None)
def concept_sentences(concept, lang):
    import edit_coverage as ec
    _, ds = ec.labels(concept, (lang,), str(p2x.DATA_V2))
    return [s for s, lbl in ds if lbl == concept]


@lru_cache(maxsize=None)
def background(concept, lang):
    """Token sets of the other 17 concepts' native sentences in `lang`."""
    return [ids(s) for o in ALL_CONCEPTS if o != concept for s in concept_sentences(o, lang)]


@lru_cache(maxsize=1)
def english_tokens():
    """Every token of the English concept + neutral sentences of all 18 concepts."""
    import edit_coverage as ec
    out = set()
    for o in ALL_CONCEPTS:
        out |= set(ec.labels(o, ("en",), str(p2x.DATA_V2))[0])
    return out


def share(sets, E):
    return float(np.mean([bool(s & E) for s in sets])) if sets else float("nan")


def main():
    import edit_coverage as ec
    from cinder.edit_size import edit_vectors
    from ember.erasure import features
    ec.TOK = p2x.tokenizer()
    special = set(p2x.tokenizer().all_special_ids)
    out = p2x.outdir("x10_extrapolation_ablation")
    cov, added, summ = [], [], []
    for c in p2x.CONCEPTS:
        E0 = set(edit_vectors(c, p2x.METHODS["EMBER"])[0])
        EX = set(edit_vectors(c, p2x.METHODS_X["EMBER+X"])[0])
        features.MF_OUTPUTS_ROOT = p2x.METHODS["EMBER"]
        pkl, _ = features._embedding_paths(C.MODEL_NAME, c, C.RANK, C.SEED)
        vstar = set(int(t) for t in features.load_embedding_payload(pkl)[2])
        codes_all, ds_all = ec.labels(c, ALL, str(p2x.DATA_V2))
        english = english_tokens()
        pool = {t for t, k in codes_all.items() if k == 1 and t not in special and t not in vstar}
        df = Counter(t for s, lbl in ds_all if lbl == c for t in ids(s) if t in pool)   # concept sentences per token
        add_x = EX - E0
        assert add_x <= pool, f"{c}: EMBER+X adds tokens outside the concept-only pool"
        add_xf = {t for t in add_x if t not in english}
        rank = sorted(pool, key=lambda t: (-df[t], t))
        words = [t for t in rank if p2x.is_word_token(p2x.tok_str([t])[0])]
        rank_f = [t for t in words if t not in english]
        add_l, add_lw, add_lf = set(rank[:len(add_x)]), set(words[:len(add_x)]), set(rank_f[:len(add_xf)])
        sets = {"EMBER": E0, "X": E0 | add_x, "X_foreign": E0 | add_xf, "L": E0 | add_l, "L_word": E0 | add_lw,
                "L_foreign": E0 | add_lf}
        summ.append({"concept": c, "n_E0": len(E0), "n_pool": len(pool), "n_pool_foreign": len(pool - english),
                     "n_added_X": len(add_x), "n_added_X_foreign": len(add_xf),
                     "X_vs_L_overlap": round(len(add_x & add_l) / max(len(add_x), 1), 2),
                     "X_vs_Lword_overlap": round(len(add_x & add_lw) / max(len(add_x), 1), 2),
                     "Xf_vs_Lf_overlap": round(len(add_xf & add_lf) / max(len(add_xf), 1), 2)})
        for v, E in sets.items():
            r = {"concept": c, "variant": v, "n_edit": len(E)}
            for L in ALL:
                r[f"qa_{L}"] = round(share(questions(c, L, "QA_train"), E), 3)
                r[f"simdom_{L}"] = round(share(questions(c, L, "SimdomQA_train"), E), 3)
                r[f"bg_{L}"] = round(share(background(c, L), E), 4)
            cov.append(r)
        for v, A in (("X", add_x), ("X_foreign", add_xf), ("L", add_l), ("L_word", add_lw), ("L_foreign", add_lf)):
            order = sorted(A, key=lambda t: (-df[t], t))
            for t in order:
                added.append({"concept": c, "variant": v, "token": p2x.tok_str([t])[0], "concept_sentences": df[t],
                              "foreign_only": t not in english, "in_X": t in add_x, "in_L": t in add_l})
        print(summ[-1], flush=True)

    Cv, Ad, S = pd.DataFrame(cov), pd.DataFrame(added), pd.DataFrame(summ)
    Cv.to_csv(out / "coverage.csv", index=False)
    Ad.to_csv(out / "added_tokens.csv", index=False)
    F = ["de", "es", "fr", "ru"]
    Cv["qa_foreign"] = Cv[[f"qa_{L}" for L in F]].mean(1)
    Cv["simdom_foreign"] = Cv[[f"simdom_{L}" for L in F]].mean(1)
    Cv["bg_foreign"] = Cv[[f"bg_{L}" for L in F]].mean(1)
    mean = Cv.groupby("variant", sort=False)[["n_edit", "qa_en", "qa_foreign", "simdom_en", "simdom_foreign", "bg_en",
                                             "bg_foreign"] + [f"qa_{L}" for L in F]].mean().round(3).reindex(VARIANTS)
    L_ = ["# X10: does extrapolation need the features? (EMBER, 6 concepts, train questions, coverage only)", "",
          "EMBER = original English edit; X = EMBER+X as built (x5b, tested in X9); X_foreign = only its added tokens "
          "that never occur in the English sentences of any of the 18 concepts; L = same number of tokens from the "
          "same concept-only pool, ranked by the number of concept sentences that contain them (no features); L_word "
          "= L with whole-word tokens only; L_foreign = L_word from the foreign-only pool, sized as X_foreign. qa_L / simdom_L = share of QA_train / SimdomQA_train questions in L "
          "with an edited token in the question or correct answer; bg_L = share of the other 17 concepts' native "
          "sentences in L with an edited token. Higher qa = more questions the edit can erase (X8); higher simdom / "
          "bg = more collateral. Coverage proxies only; no erasure is run.", "",
          "## Mean over the 6 concepts", "", p2x.md_table(mean.reset_index()), "",
          "## Sizes and overlap of the added sets", "",
          "`X_vs_L_overlap`: share of X's added tokens that the frequency rule also picks.", "", p2x.md_table(S), "",
          "## Per concept", "",
          p2x.md_table(Cv[["concept", "variant", "n_edit", "qa_en", "qa_de", "qa_es", "qa_fr", "qa_ru", "simdom_en",
                           "simdom_foreign", "bg_en", "bg_foreign"]]), "",
          "## Added tokens (most concept sentences first; * = foreign-only)", ""]
    for (c, v), g in Ad.groupby(["concept", "variant"], sort=False):
        L_.append(f"- {c} / {v} ({len(g)}): " + " ".join(f"{r.token}{'*' if r.foreign_only else ''}"
                                                          for r in g.head(30).itertuples()))
    (out / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_[:12]))


if __name__ == "__main__":
    main()
