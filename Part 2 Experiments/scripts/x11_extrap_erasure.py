#!/usr/bin/env python
"""X11: which extrapolation works best? Real erasure of the EMBER+X variants, on the TRAIN questions in 5 languages.

X10 compared the variants by coverage only. Here each edit is applied to Gemma and scored, so the choice rests on
erasure, not on a proxy. Variants (EMBER features, 6 concepts; all keep EMBER's English edit exactly):

    EMBER      no extrapolation (mf_outputs/, Part 1 = the fair run's EMBER features for these 6 concepts)
    X          EMBER+X as built by x5b (mf_outputs/extrap_EMBER), the version X9 tested
    X_foreign  X without the added tokens that occur in English text (English sentences of all 18 concepts)
    X_clean    X without the added tokens found in more than 0.2% of the other 17 concepts' native sentences
               in any language (collateral filter)

X_foreign / X_clean are X with the dropped tokens' G rows set to 0 (embed_edit only edits tokens with nonzero G),
written to mf_outputs/x11_<variant>/; `build` checks that each edit is exactly E0 + the kept added tokens.

Per concept, one GPU job, Gemma loaded once: the pristine model and every variant at the fair run's EMBER pick δ
(judge_rerun/Fair/ember/<concept>/pick.json) and the next grid value up (down for 200): MMLU (50 train items) and
QA_train + SimdomQA_train in en/de/es/fr/ru with the cross-lingual prompt and fixed scoring (crosslingual.rerun_fixed).
The test split is not touched: it stays for confirming the chosen variant.

    python "Part 2 Experiments/scripts/x11_extrap_erasure.py" build            # CPU
    sbatch "Part 2 Experiments/scripts/x11.sbatch" Golf Gun                    # GPU
    python "Part 2 Experiments/scripts/x11_extrap_erasure.py" report           # CPU

Output: Part 2 Experiments/x11_extrap_erasure/{build.json, <Concept>/<condition>.json, results.csv, report.md}
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402

C = p2x.C
LANGS = ["en", "de", "es", "fr", "ru"]
SETS = ("QA_train", "SimdomQA_train")
VARIANTS = ["EMBER", "X", "X_foreign", "X_clean"]
BG_MAX = 0.002
ROOTS = {"EMBER": p2x.METHODS["EMBER"], "X": p2x.METHODS_X["EMBER+X"],
         "X_foreign": C.ROOT / "mf_outputs" / "x11_X_foreign", "X_clean": C.ROOT / "mf_outputs" / "x11_X_clean"}
OUT = Path(os.environ.get("X11_OUT", p2x.OUT / "x11_extrap_erasure"))   # X11_OUT: redirect (smoke tests)
GRID = list(C.EMBER_GRID)
SMOKE = int(os.environ.get("X11_SMOKE", "0"))      # > 0: only that many items per set


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def deltas(concept):
    pick = float(json.loads((C.ROOT / "judge_rerun" / "Fair" / "ember" / C.safe(concept) / "pick.json").read_text())["delta"])
    i = GRID.index(pick)
    return [pick, GRID[i + 1] if i + 1 < len(GRID) else GRID[i - 1]]


# ---------------------------------------------------------------- build (CPU)
def build():
    import torch
    import x10_extrapolation_ablation as X10
    import edit_coverage as ec
    import numpy as np
    from cinder.edit_size import edit_vectors
    from ember.erasure import features
    ec.TOK = p2x.tokenizer()
    english = X10.english_tokens()
    rows = []
    for c in p2x.CONCEPTS:
        E0 = set(edit_vectors(c, ROOTS["EMBER"])[0])
        EX = set(edit_vectors(c, ROOTS["X"])[0])
        add = EX - E0
        bg = {t: max(float(np.mean([t in s for s in X10.background(c, L)])) for L in LANGS) for t in add}
        drop = {"X_foreign": {t for t in add if t in english}, "X_clean": {t for t in add if bg[t] > BG_MAX}}
        features.MF_OUTPUTS_ROOT = ROOTS["X"]
        pkl, _ = features._embedding_paths(C.MODEL_NAME, c, C.RANK, C.SEED)
        rel = Path(pkl).relative_to(ROOTS["X"])                    # <model>/pickles/rank100/seed42/<c>/embedding/...
        src_concept = ROOTS["X"] / rel.parents[1]                   # .../pickles/rank100/seed42/<c>
        r = {"concept": c, "deltas": deltas(c), "n_E0": len(E0), "n_added_X": len(add)}
        for v, D in drop.items():
            for kind in ("pickles", "csvs", "interpretations"):
                s = Path(str(src_concept).replace("/pickles/", f"/{kind}/"))
                d = ROOTS[v] / Path(str(rel.parents[1]).replace("pickles", kind))
                if d.exists():
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            payload = features._load_nmf_pickle(pkl)
            V = [int(t) for t in payload["vprime_token_ids"]]
            G = torch.as_tensor(payload["nmf"].G_).detach().float().cpu().clone()
            G[[i for i, t in enumerate(V) if t in D]] = 0.0
            payload["nmf"].G_ = G
            payload["source"] = f"{payload.get('source', '')}; x11 {v}: {len(D)} added tokens zeroed"
            import pickle
            with open(ROOTS[v] / rel, "wb") as fh:
                pickle.dump(payload, fh)
            got = set(edit_vectors(c, ROOTS[v])[0])
            assert got == E0 | (add - D), f"{c}/{v}: edit is not E0 + kept added tokens"
            r[f"n_dropped_{v}"] = len(D)
            r[f"dropped_{v}"] = p2x.tok_str(sorted(D, key=lambda t: -bg[t]))
        rows.append(r)
        log(c, {k: v for k, v in r.items() if not k.startswith("dropped")})
    OUT.mkdir(parents=True, exist_ok=True)
    p2x.save_json(OUT / "build.json", rows)


# ---------------------------------------------------------------- run (GPU)
def score(m, gen, concept):
    import crosslingual.data as cd
    from crosslingual.rerun_fixed import _finish_set
    from judge_rerun.mc_fixed import evaluate_mmlu_fixed
    cd.SETS = tuple(dict.fromkeys(cd.SETS + SETS))               # the loader allows only the test sets by default
    res = {"mmlu": evaluate_mmlu_fixed(m.tm, m.mmlu_train)}
    for L in LANGS:
        for s in SETS:
            items = cd.load_mc_items(concept, L, s, seed=C.SEED)
            items = items[:SMOKE] if SMOKE else items
            recs = []
            for it in items:
                new = gen.answer(it["question"], it["options"], L)
                new.pop("old4_raw")
                recs.append({**it, **new})
            cell = {"records": recs}
            _finish_set(cell)
            res[f"{L}/{s}"] = cell
    return res


def run(concepts):
    import torch
    from cinder.fair import Model
    from crosslingual.rerun_fixed import Generator
    from ember.erasure import features
    m = Model("ember")
    gen = Generator(m.tm)
    for c in concepts:
        od = OUT / C.safe(c)
        od.mkdir(parents=True, exist_ok=True)
        conds = [("pristine", None)] + [(v, d) for v in VARIANTS for d in deltas(c)]
        for v, d in conds:
            path = od / (f"{v}.json" if d is None else f"{v}_d{d:g}.json")
            if path.exists():
                continue
            t0 = time.time()
            info = None
            if d is not None:
                features.MF_OUTPUTS_ROOT = ROOTS[v]
                info = m.edit(c, d)
            try:
                with torch.no_grad():
                    res = score(m, gen, c)
            finally:
                m.restore()
            acc = {k: round(r["acc"] if k == "mmlu" else r["accuracy"], 4) for k, r in res.items()}
            p2x.save_json(path, {"concept": c, "variant": v, "delta": d, "edit_info": info, "accuracy": acc,
                                 "results": res, "seconds": round(time.time() - t0, 1), "smoke": SMOKE})
            log(f"{c} {v} δ={d}: tokens={None if info is None else info['n_tokens_edited']} mmlu {acc['mmlu']:.2f} "
                + " ".join(f"{L} {acc[f'{L}/QA_train']:.2f}/{acc[f'{L}/SimdomQA_train']:.2f}" for L in LANGS)
                + f" ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- report (CPU)
def report():
    import numpy as np
    import pandas as pd
    from crosslingual.run_edited import _retention
    rows = []
    for c in p2x.CONCEPTS:
        od = OUT / C.safe(c)
        if not (od / "pristine.json").exists():
            continue
        base = json.loads((od / "pristine.json").read_text())["accuracy"]
        for v in VARIANTS:
            for i, d in enumerate(deltas(c)):
                p = od / f"{v}_d{d:g}.json"
                if not p.exists():
                    continue
                J = json.loads(p.read_text())
                acc = J["accuracy"]
                r = {"concept": c, "variant": v, "delta": d, "which": "pick" if i == 0 else "neighbour",
                     "n_tokens": J["edit_info"]["n_tokens_edited"], "mmlu": acc["mmlu"],
                     "mmlu_frac": round(acc["mmlu"] / base["mmlu"], 3) if base["mmlu"] else None}
                for L in LANGS:
                    ret = _retention(acc[f"{L}/QA_train"], base[f"{L}/QA_train"])
                    r[f"ret_{L}"] = None if ret is None else round(ret, 3)
                    r[f"sim_{L}"] = round(acc[f"{L}/SimdomQA_train"] - base[f"{L}/SimdomQA_train"], 3)
                fr = [r[f"ret_{L}"] for L in LANGS[1:] if r[f"ret_{L}"] is not None]
                r["ret_foreign"] = round(float(np.mean(fr)), 3) if fr else None
                r["leak"] = None if r["ret_foreign"] is None or r["ret_en"] is None else round(r["ret_foreign"] - r["ret_en"], 3)
                r["sim_foreign"] = round(float(np.mean([r[f"sim_{L}"] for L in LANGS[1:]])), 3)
                rows.append(r)
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "results.csv", index=False)
    head = ["# X11: which extrapolation works best? (EMBER, 6 concepts, real erasure, train questions)", "",
            "ret_L = chance-corrected retention of QA_train in L vs the pristine model (lower = more erased); "
            "leak = mean(ret de/es/fr/ru) - ret_en; sim_L = SimdomQA_train accuracy change vs pristine (closer to 0 = "
            "less collateral); mmlu_frac = MMLU (50 train items) / pristine. δ = the fair run's EMBER pick and its "
            "neighbour on the grid. Variants: see the script docstring.", ""]
    if D.empty:
        (OUT / "report.md").write_text("\n".join(head + ["No results yet."]) + "\n")
        return
    cols = ["ret_en", "ret_de", "ret_es", "ret_fr", "ret_ru", "ret_foreign", "leak", "sim_en", "sim_foreign", "mmlu_frac"]
    L_ = list(head)
    for which in ("pick", "neighbour"):
        sub = D[D.which == which]
        agg = sub.groupby("variant")[cols].mean().round(3).reindex([v for v in VARIANTS if v in set(sub.variant)])
        agg.insert(0, "concepts", sub.groupby("variant").size().reindex(agg.index).values)
        L_ += [f"## Mean over concepts, δ = {which}", "", p2x.md_table(agg.reset_index()), ""]
        piv = sub.pivot(index="concept", columns="variant")
        for v in VARIANTS[2:] + ["EMBER"]:
            if {"X", v} <= set(sub.variant):
                d_r = (piv["ret_foreign"][v] - piv["ret_foreign"]["X"]).dropna()
                d_s = (piv["sim_foreign"][v] - piv["sim_foreign"]["X"]).dropna()
                L_.append(f"- {v} − X ({which}): ret_foreign {d_r.mean():+.3f} (lower in {int((d_r < 0).sum())} of "
                          f"{len(d_r)}), sim_foreign {d_s.mean():+.3f}")
        L_.append("")
    L_ += ["## Per concept", "", p2x.md_table(D[["concept", "variant", "delta", "which", "n_tokens"] + cols]), ""]
    (OUT / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("build", "run", "report"))
    ap.add_argument("--concepts", nargs="+", default=p2x.CONCEPTS)
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    {"build": build, "report": report}.get(a.stage, lambda: run(a.concepts))()


if __name__ == "__main__":
    main()
