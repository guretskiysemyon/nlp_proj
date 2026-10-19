#!/usr/bin/env python
"""Word lists for the J3 overlap judge, 20 x 20 variant: concept words from a prompt version, feature words from runs.

    python -m experiments.varwidth.j3_words concepts --version v1 [--concepts ...] [--n 20]
    python -m experiments.varwidth.j3_words features [--concepts ...] [--tag rn_refit-False]

concepts: one Gemini call per missing concept with j3_data/prompt_<version>.txt, normalized and deduped like
    overlap_judge.concept_words; merged into j3_data/concept_words_<version>.json ({concept: {"words", "raw"}}).
    v1 is the chosen prompt (name forms + well-known, specific words); v2 / v3 are the revisions that did not score better.
features: the first 20 distinct normalized source words (overlap_judge.feature_words, n = 20) of every rho candidate
    of every run with a judges.json; merged into results/varwidth/j3/fw20.json ({"concept|seed|feature": [words]}).
    No Gemini. The first 10 equal judges.json's feature_words.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from experiments.varwidth import overlap_judge as oj
from experiments.varwidth.common import RESULTS, ROOT, run_dir

DATA = ROOT / "experiments" / "varwidth" / "j3_data"
FW20 = RESULTS / "j3" / "fw20.json"
N = 20
SEEDS = range(42, 47)


def all_concepts() -> list:
    from experiments.varwidth.judge_report import concepts
    return concepts()


def concept_lists(version: str = "v1") -> dict:
    """{concept: [normalized words]} of j3_data/concept_words_<version>.json."""
    d = json.loads((DATA / f"concept_words_{version}.json").read_text())
    return {c: [oj.normalize(w) for w in r["words"]] for c, r in d.items()}


def gen_concepts(version: str, concepts: list, n: int = N, workers: int = 8) -> dict:
    from experiments.varwidth.select import get_client
    prompt = (DATA / f"prompt_{version}.txt").read_text()
    path = DATA / f"concept_words_{version}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    client = get_client()

    def one(c):
        raw = client.generate(prompt.format(concept=c, n=n))
        return c, {"words": oj._dedupe(oj.normalize(w) for w in oj.parse_list(raw) if isinstance(w, str))[:n], "raw": raw}

    with ThreadPoolExecutor(workers) as ex:
        for c, r in ex.map(one, [c for c in concepts if c not in out]):
            out[c] = r
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def gen_features(concepts: list, tag: str) -> dict:
    import torch

    from ember.judge_words import token_word_counts
    from experiments.varwidth.judge_compare import row_order
    from experiments.varwidth.prep import build_dataset, get_tokenizer, prepare

    out = json.loads(FW20.read_text()) if FW20.exists() else {}
    tok = get_tokenizer()
    tok_str = lambda t: tok.decode([t])  # noqa: E731
    for c in concepts:
        runs = [s for s in SEEDS if (run_dir(c, tag, s) / "judges.json").exists()
                and not any(k.startswith(f"{c}|{s}|") for k in out)]
        if not runs:
            continue
        vstar = prepare(c, "plain" if "_plain" in tag else "whitened", verbose=False)["vstar_ids"]
        counts = token_word_counts(tok, [s for s, _ in build_dataset(c).data])
        for s in runs:
            d = run_dir(c, tag, s)
            J = json.loads((d / "judges.json").read_text())
            Y = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)["Y"].float()
            for x in J["candidates"]:
                f = x["feature"]
                out[f"{c}|{s}|{f}"] = oj.feature_words([vstar[j] for j in row_order(Y, f)], counts, n=N, tok_str=tok_str)
        print(f"{c}: {len(runs)} runs", flush=True)
    FW20.parent.mkdir(parents=True, exist_ok=True)
    FW20.write_text(json.dumps(out, ensure_ascii=False))
    return out


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from experiments.varwidth.judge_compare import DEFAULT_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("concepts", "features"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    a = ap.parse_args()
    cs = a.concepts or all_concepts()
    if a.what == "concepts":
        out = gen_concepts(a.version, cs, a.n)
        for c in cs:
            print(f"{c:26} ({len(out[c]['words'])}) {', '.join(out[c]['words'])}")
    else:
        gen_features(cs, a.tag)


if __name__ == "__main__":
    main()
