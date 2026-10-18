#!/usr/bin/env python3
"""Cross-lingual runs with the fixed MC scoring and the repaired picks.

Stage B / C of the cross-lingual repair (docs/Part 1/crosslingual_next.md). Works on
``results/crosslingual/<model>/<method>_fixed/``, which ``crosslingual.reparse_saved``
creates from the original run (stage A: lowercase letters and the prompt-echo answer
re-read on the CPU). Per concept, the mode follows from the picks:

  regen  the repaired pick is the edit the original run used (same HP), so every valid
         answer stands. Only the answers still invalid after stage A are generated again,
         with the same prompt, 32 new tokens, new tokens only, parse_letter_fixed
         (judge_rerun/mc_fixed.py). Control: the first 4 generated tokens must equal the
         saved answer (greedy prefix); mismatches are counted, not hidden.
  full   the repaired pick is a different edit: every cell (5 languages x QA_test +
         SimdomQA_test) is generated again with the fixed scoring.

Both modes also re-score English MMLU (1000-question test split) with the fixed scoring,
saving every answer (``mmlu_test.json``), and run the edit check: the edited model's
QA_train / SimdomQA_train / 50-question MMLU, fixed scoring, against the repaired
search's grid row for that pick.

Wired: ``ember`` (picks: judge_rerun/Ember/pass2_rescored, grid:
judge_rerun/Ember/rescore_mc) and ``snmf_ember`` (picks:
judge_rerun/SNMF_EMBER/pass2_rescored, grid: judge_rerun/SNMF_EMBER/rescore_mc, or the
original grid row for a cell the repair did not re-score). The edit is "the same"
when the EMBER delta and the SNMF cell (delta / layers in and out) match. ``snmf``
needs no GPU (all its picks are unchanged and its invalid answers were lowercase
letters): crosslingual.reparse_saved alone fixes it. A concept whose ``edit.json`` already has
``fixed_run`` is skipped, so a preempted job can be resubmitted as is.

Usage:
    python -m crosslingual.rerun_fixed --method ember --concepts Uranium Gun
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosslingual.data import CONCEPTS, LANGUAGES, REPO_ROOT, SETS, load_mc_items
from crosslingual.prompts import build_mc_prompt
from crosslingual.run_edited import REGISTRY_NAME, RUN_REL, _git_commit, load_config

ROOT = REPO_ROOT / "results" / "crosslingual" / "google_gemma-2-2b-it"
EMBER_PICKS = REPO_ROOT / "judge_rerun" / "Ember" / "pass2_rescored" / RUN_REL / "best_deltas_summary.csv"
EMBER_RESCORED = REPO_ROOT / "judge_rerun" / "Ember" / "rescore_mc" / RUN_REL
SE_DIR = REPO_ROOT / "judge_rerun" / "SNMF_EMBER"
LETTERS = ("A", "B", "C", "D")
# The edit is the same iff these match (the SNMF cell and the EMBER delta).
EDIT_KEY = ("delta_embed", "delta_in", "layer_lo_in", "layer_hi_in", "delta_out", "layer_lo_out", "layer_hi_out")


def same_edit(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return all(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) < 1e-9 for k in EDIT_KEY)


def _safe(concept: str) -> str:
    from ember.utils import _safe_concept
    return _safe_concept(concept)


def resolve_pick(method: str, concept: str) -> Dict[str, Any]:
    """Repaired pick + the repaired search's (fixed-scoring) numbers for it.

    ``k_keys`` are the feature counts the edit check compares with the search."""
    if method == "snmf_ember":
        best = json.loads((SE_DIR / "pass2_rescored" / RUN_REL / _safe(concept)
                           / "best_hp__qwen2.5-32b-instruct.json").read_text(encoding="utf-8"))
        row = int(best["best_row"])
        resc = SE_DIR / "rescore_mc" / RUN_REL / _safe(concept) / "grid_rescored.csv"
        with resc.open(encoding="utf-8") as f:
            g = next((r for r in csv.DictReader(f) if int(r["row"]) == row), None)
        if g is None:  # a cell the repair did not re-score (no invalid answer): grid values stand
            from judge_rerun.rescore_snmf_grid import grid_path, load_cfg
            with grid_path(load_cfg(True), concept).open(encoding="utf-8") as f:
                g = list(csv.DictReader(f))[row]
        search = {k: float(g[k]) for k in ("qa_acc", "simdom_acc", "mmlu_acc")}
        search.update({k: int(float(g[k])) for k in ("k_features_mlp_in", "k_features_mlp_out")})
        hp = {k: v for k, v in best["best_hp"].items()}
        if abs(float(hp["delta_embed"]) - float(g["delta_embed"])) > 1e-9:
            raise ValueError(f"{concept!r}: pick delta {hp['delta_embed']} != grid row delta {g['delta_embed']}")
        return {"hp": hp, "search": search, "k_keys": ("k_features_mlp_in", "k_features_mlp_out")}
    if method != "ember":
        raise NotImplementedError(f"{method}: not wired yet")
    with EMBER_PICKS.open(encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["concept"] == concept)
    delta = float(row["best_delta_embed"])
    with (EMBER_RESCORED / _safe(concept) / "grid_rescored.csv").open(encoding="utf-8") as f:
        g = next(r for r in csv.DictReader(f) if float(r["delta_embed"]) == delta)
    search = {k: float(g[k]) for k in ("qa_acc", "simdom_acc", "mmlu_acc")}
    search.update({k: int(float(g[k])) for k in ("k_features_embed", "n_tokens_edited")})
    return {"hp": {"delta_embed": delta}, "search": search, "k_keys": ("k_features_embed", "n_tokens_edited")}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


class Generator:
    def __init__(self, tm: Any) -> None:
        self.tm = tm

    def answer(self, question: str, options: Dict[str, str], lang: str) -> Dict[str, Any]:
        from judge_rerun.mc_fixed import MC_MAX_NEW_TOKENS, OLD_MC_MAX_NEW_TOKENS, _decode, _generate_ids, parse_letter_fixed
        wrapped = self.tm.wrap_prompt(build_mc_prompt(question, options, lang))
        ids = _generate_ids(self.tm, [wrapped], MC_MAX_NEW_TOKENS)[0]
        raw = _decode(self.tm, ids)
        letter = parse_letter_fixed(raw)
        return {"model_raw": raw, "model_answer_text": raw, "model_letter": letter,
                "old4_raw": _decode(self.tm, ids[:OLD_MC_MAX_NEW_TOKENS]),
                "n_new_tokens": len(ids), "is_valid": letter in LETTERS}


def _finish_set(res: Dict[str, Any]) -> None:
    recs = res["records"]
    for r in recs:
        r["is_correct"] = r["is_valid"] and r["model_letter"] == r["correct_letter"]
    res["n"] = len(recs)
    res["accuracy"] = sum(r["is_correct"] for r in recs) / len(recs)
    res["invalid_rate"] = sum(not r["is_valid"] for r in recs) / len(recs)


def regen_cell(gen: Generator, cell: Dict[str, Any], lang: str) -> Dict[str, int]:
    """Re-generate only the still-invalid answers of a stage-A cell, in place."""
    todo = {(s["set"], s["index"]) for s in cell.get("still_invalid", [])}
    st = {"regenerated": 0, "now_valid": 0, "control_mismatch": 0}
    for s in SETS:
        for r in cell["results"][s]["records"]:
            if (s, r["index"]) not in todo:
                continue
            new = gen.answer(r["question"], r["options"], lang)
            match = new.pop("old4_raw").strip() == r["model_raw"].strip()
            st["regenerated"] += 1
            st["now_valid"] += int(new["is_valid"])
            st["control_mismatch"] += int(not match)
            r.update({"old_model_raw": r["model_raw"], "old_model_letter": r["model_letter"],
                      "old_is_correct": r["is_correct"], "regenerated": True,
                      "control_match": match, **new})
        _finish_set(cell["results"][s])
    cell["still_invalid"] = [{"set": s, "index": r["index"], "model_raw": r["model_raw"]}
                             for s in SETS for r in cell["results"][s]["records"] if not r["is_valid"]]
    cell["scoring"] = ("reparse_saved + invalid answers regenerated with mc_fixed "
                       "(32 tokens, new tokens only, parse_letter_fixed)")
    return st


def full_cell(gen: Generator, concept: str, lang: str, seed: int) -> Dict[str, Any]:
    results = {}
    for s in SETS:
        t0 = time.time()
        recs = []
        for it in load_mc_items(concept, lang, s, seed=seed):
            new = gen.answer(it["question"], it["options"], lang)
            new.pop("old4_raw")
            recs.append({**it, **new})
        results[s] = {"records": recs, "seconds": round(time.time() - t0, 2)}
        _finish_set(results[s])
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="ember", choices=("ember", "snmf_ember"))
    p.add_argument("--concepts", nargs="+", required=True)
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    bad = [c for c in args.concepts if c not in CONCEPTS]
    if bad:
        raise SystemExit(f"unknown concepts {bad}")

    src = ROOT / f"{args.method}_fixed"
    if not src.exists():
        raise SystemExit(f"{src} missing: run python -m crosslingual.reparse_saved --method {args.method}")
    plan = []
    for c in args.concepts:
        cdir = src / _safe(c)
        old = json.loads((cdir / "edit.json").read_text(encoding="utf-8"))
        if old.get("fixed_run") and not args.overwrite:
            print(f"[fixed] {c!r}: done, skipping", flush=True)
            continue
        pick = resolve_pick(args.method, c)
        mode = "regen" if same_edit(old["hp"], pick["hp"]) else "full"
        plan.append((c, pick, mode, old))
        print(f"[fixed] {args.method} {c!r}: {mode} (old hp {old['hp']} -> {pick['hp']})", flush=True)
    if not plan:
        return 0

    import torch
    from ember.erasure import model_loader
    from ember.erasure.methods import base as methods_base
    from ember.evals.mc import prepare_mc_items
    from ember.evals.mmlu import load_mmlu_items
    from ember.evals.model_wrap import ensure_wrapped_model
    from ember.local_datasets import load_mc_qa_items, load_mmlu_indices
    from judge_rerun.mc_fixed import evaluate_mc_fixed, evaluate_mmlu_fixed

    torch.manual_seed(args.seed)
    hf_model, tokenizer = model_loader.load_hf_model(args.model, "", dtype=model_loader.pick_dtype("bf16"))
    hf_model.eval()
    tm = ensure_wrapped_model(hf_model, tokenizer)
    gen = Generator(tm)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    mmlu_train = load_mmlu_items(load_mmlu_indices("train"))
    mmlu_test = load_mmlu_items(load_mmlu_indices("test"))
    commit = _git_commit()

    for k, (concept, pick, mode, old) in enumerate(plan, 1):
        t_job = time.time()
        cfg = load_config(args.method, args.model, pick["hp"])
        method = methods_base.get(REGISTRY_NAME[args.method])
        snap = method.snapshot(hf_model)
        cdir = src / _safe(concept)
        totals = {"regenerated": 0, "now_valid": 0, "control_mismatch": 0}
        try:
            method.on_concept_start(hf_model, concept, cfg)
            info = method.apply(hf_model, tokenizer, pick["hp"], concept, cfg)
            with torch.no_grad():
                check = {"mmlu_acc": evaluate_mmlu_fixed(tm, mmlu_train)["acc"]}
                for key, set_name in (("qa_acc", "qa_train"), ("simdom_acc", "simdom_train")):
                    items = prepare_mc_items(load_mc_qa_items(set_name, concept=concept), seed=args.seed)
                    check[key] = evaluate_mc_fixed(tm, items)["acc"]
                mmlu = evaluate_mmlu_fixed(tm, mmlu_test)
                for lang in LANGUAGES:
                    path = cdir / f"{lang}.json"
                    if mode == "regen":
                        cell = json.loads(path.read_text(encoding="utf-8"))
                        st = regen_cell(gen, cell, lang)
                        for key in totals:
                            totals[key] += st[key]
                    else:
                        cell = {"concept": concept, "language": lang, "model": args.model,
                                "config": args.method, "prompt_lang": lang, "seed": args.seed,
                                "results": full_cell(gen, concept, lang, args.seed),
                                "scoring": "full run with mc_fixed (32 tokens, new tokens only, parse_letter_fixed)"}
                        cell["still_invalid"] = [{"set": s, "index": r["index"], "model_raw": r["model_raw"]}
                                                 for s in SETS for r in cell["results"][s]["records"]
                                                 if not r["is_valid"]]
                    cell.update({"hp": pick["hp"], "gpu": gpu, "git_commit": commit, "mode": mode,
                                 "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    _write_json(path, cell)
                    qa, sd = cell["results"]["QA_test"], cell["results"]["SimdomQA_test"]
                    print(f"[fixed]   {concept!r} {lang}: QA_test={qa['accuracy']:.2f} "
                          f"(invalid {qa['invalid_rate']:.2f}) Simdom={sd['accuracy']:.2f} "
                          f"(invalid {sd['invalid_rate']:.2f})", flush=True)
        finally:
            method.restore(hf_model, snap)
            method.on_concept_end(hf_model, concept, cfg)

        _write_json(cdir / "mmlu_test.json", {"concept": concept, "hp": pick["hp"], **mmlu})
        diffs = {key: round(check[key] - pick["search"][key], 4) for key in ("qa_acc", "simdom_acc", "mmlu_acc")}
        k_match = all(int(info.get(key, -1)) == pick["search"][key] for key in pick["k_keys"])
        edit = {
            "concept": concept, "method": args.method, "hp": pick["hp"], "mode": mode,
            "old_hp": old["hp"], "edit_info": info, "search": pick["search"],
            "check": {"english_train": check, "diff_vs_search": diffs, "features_match": k_match,
                      "reproduces_search": k_match and all(v == 0 for v in diffs.values())},
            "mmlu_test": {"acc": mmlu["acc"], "invalid": mmlu["invalid_rate"], "n": mmlu["n"]},
            "old_mmlu_test": old.get("mmlu_test"),
            "regen": totals if mode == "regen" else None,
            "fixed_run": True, "seconds": round(time.time() - t_job, 1), "git_commit": commit,
        }
        _write_json(cdir / "edit.json", edit)
        flag = "OK" if edit["check"]["reproduces_search"] else "DIFFERS"
        print(f"[fixed] {k}/{len(plan)} {concept!r} [{mode}]: edit check {flag} (Δ vs search {diffs}) "
              f"MMLU test={mmlu['acc']:.3f} (was {(old.get('mmlu_test') or {}).get('acc')}) "
              f"{'regen ' + str(totals) if mode == 'regen' else ''} {time.time() - t_job:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
