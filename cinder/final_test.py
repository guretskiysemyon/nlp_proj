#!/usr/bin/env python
"""CINDER final test: the held-out questions in 5 languages at the best delta (GPU: Gemma only, no judge).

Same cells, prompts and scoring as the repaired EMBER run it is compared with
(results/crosslingual/google_gemma-2-2b-it/ember_fixed/, crosslingual/rerun_fixed.py "full" mode): QA_test and
SimdomQA_test in en/de/es/fr/ru with the baseline's items and shuffle, mc_fixed (32 new tokens,
parse_letter_fixed); English MMLU on the 1000-question test split; and the edit check (QA_train,
SimdomQA_train, 50 MMLU train, fixed scoring) against the grid row of the pick.

Output: results/crosslingual/google_gemma-2-2b-it/cinder/<Concept>/{en,de,es,fr,ru}.json, mmlu_test.json,
edit.json, then summary.csv (crosslingual/summary_fixed.write_summary, pristine from baseline_fixed/).

    python -m cinder.final_test --concepts Cannabis Gun ... [--languages en]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinder import common as C

XL_ROOT = C.ROOT / "results" / "crosslingual" / C.MODEL_SAFE
OUT_DIR = XL_ROOT / "cinder"


def resolve_pick(concept: str) -> dict:
    best = json.loads((C.PICK_OUT / C.safe(concept) / f"best_delta__{C.JUDGE_ID}.json").read_text())
    delta = float(best["best_delta_embed"])
    with (C.GRID_OUT / C.safe(concept) / "grid_rescored.csv").open(encoding="utf-8") as f:
        g = next(r for r in csv.DictReader(f) if float(r["delta_embed"]) == delta)
    search = {k: float(g[k]) for k in ("qa_acc", "simdom_acc", "mmlu_acc")}
    search.update({k: int(float(g[k])) for k in ("k_features_embed", "n_tokens_edited")})
    search["harmonic_alpaca"] = float(best["best_harmonic_alpaca"])
    return {"hp": {"delta_embed": delta}, "search": search}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concepts", nargs="+", default=C.CONCEPTS)
    ap.add_argument("--languages", nargs="+", default=["en", "de", "es", "fr", "ru"])
    ap.add_argument("--overwrite", action="store_true",
                    help="redo concepts whose edit.json exists (also needed to add languages to a done concept)")
    ap.add_argument("--summary", action="store_true",
                    help="also write summary.csv (cinder.compare does it too; parallel jobs should not)")
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")

    todo = []
    for c in a.concepts:
        if (OUT_DIR / C.safe(c) / "edit.json").exists() and not a.overwrite:
            print(f"[cinder final] {c!r}: done, skipping", flush=True)
            continue
        best_path = C.PICK_OUT / C.safe(c) / f"best_delta__{C.JUDGE_ID}.json"
        if not best_path.exists():
            print(f"[cinder final] {c!r}: no pick ({best_path} missing), skipping", flush=True)
            continue
        todo.append((c, resolve_pick(c)))
    if todo:
        run(todo, a.languages)
    from crosslingual.summary_fixed import headline, write_summary
    if a.summary and any(OUT_DIR.glob("*/edit.json")):
        df = write_summary(OUT_DIR, XL_ROOT / "baseline_fixed")
        for k, v in headline(df).items():
            print(f"{k}: {v}")
    return 0


def run(todo, langs) -> None:
    import torch
    from crosslingual.data import LANGUAGES, SETS
    from crosslingual.rerun_fixed import Generator, _write_json, full_cell
    from crosslingual.run_edited import _git_commit
    from ember.erasure import embed_edit, model_loader
    from ember.evals.mc import prepare_mc_items
    from ember.evals.mmlu import load_mmlu_items
    from ember.evals.model_wrap import ensure_wrapped_model
    from ember.local_datasets import load_mc_qa_items, load_mmlu_indices
    from judge_rerun.mc_fixed import evaluate_mc_fixed, evaluate_mmlu_fixed

    C.use_cinder_features()
    torch.manual_seed(C.SEED)
    hf_model, tokenizer = model_loader.load_hf_model(C.MODEL_NAME, "", dtype=model_loader.pick_dtype("bf16"))
    hf_model.eval()
    tm = ensure_wrapped_model(hf_model, tokenizer)
    gen = Generator(tm)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    mmlu_train = load_mmlu_items(load_mmlu_indices("train"))
    mmlu_test = load_mmlu_items(load_mmlu_indices("test"))
    commit = _git_commit()
    snap = embed_edit.snapshot(hf_model)

    for k, (concept, pick) in enumerate(todo, 1):
        t_job = time.time()
        cdir = OUT_DIR / C.safe(concept)
        try:
            info = embed_edit.apply_concept_embed_edit_factored(
                hf_model=hf_model, model_name=C.MODEL_NAME, concept_name=concept,
                delta_embed=pick["hp"]["delta_embed"], rank=C.RANK, seed=C.SEED, ratio_thresh=2.0)
            with torch.no_grad():
                check = {"mmlu_acc": evaluate_mmlu_fixed(tm, mmlu_train)["acc"]}
                for key, set_name in (("qa_acc", "qa_train"), ("simdom_acc", "simdom_train")):
                    items = prepare_mc_items(load_mc_qa_items(set_name, concept=concept), seed=C.SEED)
                    check[key] = evaluate_mc_fixed(tm, items)["acc"]
                mmlu = evaluate_mmlu_fixed(tm, mmlu_test)
                for lang in langs:
                    assert lang in LANGUAGES, lang
                    cell = {"concept": concept, "language": lang, "model": C.MODEL_NAME, "config": "cinder",
                            "prompt_lang": lang, "seed": C.SEED, "results": full_cell(gen, concept, lang, C.SEED),
                            "scoring": "full run with mc_fixed (32 tokens, new tokens only, parse_letter_fixed)"}
                    cell["still_invalid"] = [{"set": s, "index": r["index"], "model_raw": r["model_raw"]}
                                             for s in SETS for r in cell["results"][s]["records"] if not r["is_valid"]]
                    cell.update({"hp": pick["hp"], "gpu": gpu, "git_commit": commit, "mode": "full",
                                 "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    _write_json(cdir / f"{lang}.json", cell)
                    qa, sd = cell["results"]["QA_test"], cell["results"]["SimdomQA_test"]
                    print(f"[cinder final]   {concept!r} {lang}: QA_test={qa['accuracy']:.2f} "
                          f"(invalid {qa['invalid_rate']:.2f}) Simdom={sd['accuracy']:.2f}", flush=True)
        finally:
            embed_edit.restore(hf_model, snap)

        _write_json(cdir / "mmlu_test.json", {"concept": concept, "hp": pick["hp"], **mmlu})
        diffs = {key: round(check[key] - pick["search"][key], 4) for key in ("qa_acc", "simdom_acc", "mmlu_acc")}
        k_match = all(int(info.get(key, -1)) == pick["search"][key] for key in ("k_features_embed", "n_tokens_edited"))
        edit = {"concept": concept, "method": "cinder", "hp": pick["hp"], "mode": "full", "edit_info": info,
                "languages": list(langs),
                "search": pick["search"],
                "check": {"english_train": check, "diff_vs_search": diffs, "features_match": k_match,
                          "reproduces_search": k_match and all(v == 0 for v in diffs.values())},
                "mmlu_test": {"acc": mmlu["acc"], "invalid": mmlu["invalid_rate"], "n": mmlu["n"]},
                "fixed_run": True, "seconds": round(time.time() - t_job, 1), "git_commit": commit}
        _write_json(cdir / "edit.json", edit)
        flag = "OK" if edit["check"]["reproduces_search"] else "DIFFERS"
        print(f"[cinder final] {k}/{len(todo)} {concept!r}: edit check {flag} (diff vs search {diffs}) "
              f"MMLU test={mmlu['acc']:.3f} {time.time() - t_job:.0f}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
