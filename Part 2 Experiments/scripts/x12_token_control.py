#!/usr/bin/env python
"""X12: is extrapolation's gain about WHICH tokens it adds, or just about editing more tokens? (EMBER, 6 concepts)

X adds about 2-3x EMBER's tokens. Control edits with the same number of added tokens and the same edit sizes, but
tokens chosen without the features (all keep EMBER's English edit E0 exactly):

    F_own   E0 + the top whole-word concept-only tokens by number of concept sentences (X10's L_word rule)
    R_own   E0 + random whole-word tokens from the same concept-only pool (seed 42)

Each control token is edited along its OWN component on EMBER's selected features (dense ridge solve with Z fixed,
as X5's extrapolation before the sparsity rule), rescaled so that the edit norms ||c_t|| equal X's added tokens'
norms (largest to the first-chosen token). So the controls match X in token count and edit mass; only the token
choice differs. Read against X11's EMBER and X at the same δ (the fair run's EMBER pick), same items and scoring:
X11's pristine / EMBER / X files are reused, only the 2 controls run on the GPU. Train split only.

    python "Part 2 Experiments/scripts/x12_token_control.py" build        # CPU
    sbatch "Part 2 Experiments/scripts/x12.sbatch" Golf Gun               # GPU
    python "Part 2 Experiments/scripts/x12_token_control.py" report       # CPU

Output: Part 2 Experiments/x12_token_control/{build.json, <Concept>/<variant>_d<δ>.json, results.csv, report.md}
"""
import argparse
import json
import math
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import x11_extrap_erasure as X11  # noqa: E402

C = p2x.C
LANGS = X11.LANGS
VARIANTS = ["F_own", "R_own"]
ROOTS = {v: C.ROOT / "mf_outputs" / f"x12_{v}" for v in VARIANTS}
OUT = Path(os.environ.get("X12_OUT", p2x.OUT / "x12_token_control"))


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def pick_delta(concept):
    return X11.deltas(concept)[0]


# ---------------------------------------------------------------- build (CPU)
def build():
    import numpy as np
    import pandas as pd
    import torch
    import edit_coverage as ec
    import x10_extrapolation_ablation as X10
    from collections import Counter
    from cinder.edit_size import edit_vectors
    from ember.erasure import features
    from ember.utils import collect_feature_rows_for_embeddings
    from experiments.varwidth.common import VWConfig
    from experiments.varwidth.prep import TokenizerOnly, get_tokenizer
    ec.TOK = p2x.tokenizer()
    E = p2x.embeddings()
    special = set(p2x.tokenizer().all_special_ids)
    ridge = VWConfig(method="rn", refit=False).ridge
    rows = []
    for c in p2x.CONCEPTS:
        t0, _ = edit_vectors(c, X11.ROOTS["EMBER"])
        tx, cx = edit_vectors(c, X11.ROOTS["X"])
        E0, add_x = set(t0), [t for t in tx if t not in set(t0)]
        norm_x = sorted((float(cx[tx.index(t)].norm()) for t in add_x), reverse=True)
        features.MF_OUTPUTS_ROOT = X11.ROOTS["EMBER"]
        pkl0, pot = features._embedding_paths(C.MODEL_NAME, c, C.RANK, C.SEED)
        F0, _, v0 = features.load_embedding_payload(pkl0)
        feat = features.select_embed_feature_ids(pd.read_csv(pot), ratio_thresh=2.0)
        en_codes, _ = ec.labels(c, ("en",), str(p2x.DATA_V2))
        codes_all, ds_all = ec.labels(c, tuple(LANGS), str(p2x.DATA_V2))
        pool = {t for t, k in codes_all.items() if k == 1 and t not in special and t not in set(int(v) for v in v0)}
        assert set(add_x) <= pool, f"{c}: X adds tokens outside the pool"
        df = Counter(t for s, lbl in ds_all if lbl == c for t in X10.ids(s) if t in pool)
        words = [t for t in sorted(pool, key=lambda t: (-df[t], t)) if p2x.is_word_token(p2x.tok_str([t])[0])]
        rng = np.random.default_rng(C.SEED)
        chosen = {"F_own": words[:len(add_x)],
                  "R_own": [words[i] for i in rng.choice(len(words), size=len(add_x), replace=False)]}
        # own component on the selected features: ridge solve with Z = F0 fixed (X5's dense step)
        Z = torch.as_tensor(F0).double()
        Gm = Z.T @ Z + ridge * torch.eye(Z.shape[1], dtype=torch.float64)
        Fsel = Z[:, feat]
        features.MF_OUTPUTS_ROOT = X11.ROOTS["X"]
        pklx, _ = features._embedding_paths(C.MODEL_NAME, c, C.RANK, C.SEED)
        rel = Path(pklx).relative_to(X11.ROOTS["X"])
        src_concept = X11.ROOTS["X"] / rel.parents[1]
        r = {"concept": c, "delta": pick_delta(c), "n_E0": len(E0), "n_added": len(add_x),
             "n_pool_words": len(words)}
        for v, toks in chosen.items():
            for kind in ("pickles", "csvs", "interpretations"):
                s = Path(str(src_concept).replace("/pickles/", f"/{kind}/"))
                d = ROOTS[v] / Path(str(rel.parents[1]).replace("pickles", kind))
                if d.exists():
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            payload = features._load_nmf_pickle(pklx)
            V = [int(t) for t in payload["vprime_token_ids"]]
            pos = {t: i for i, t in enumerate(V)}
            G = torch.as_tensor(payload["nmf"].G_).detach().float().cpu().clone()
            G[[pos[t] for t in add_x]] = 0.0                      # drop X's added tokens
            Y = torch.linalg.solve(Gm, Z.T @ E[torch.tensor(toks)].T)  # (k, n) own coefficients
            for j, (t, target) in enumerate(zip(toks, norm_x)):
                g = Y[feat, j]
                n_own = float((Fsel @ g).norm())
                assert n_own > 1e-9, f"{c}/{v}: token {t} has no component on the selected features"
                row = torch.zeros(G.shape[1])
                row[feat] = (g * (target / n_own)).float()
                G[pos[t]] = row
            payload["nmf"].G_ = G
            payload["source"] = f"{payload.get('source', '')}; x12 {v}: X's added tokens -> {len(toks)} control tokens"
            with open(ROOTS[v] / rel, "wb") as fh:
                pickle.dump(payload, fh)
            label = {t: ("both" if k == 2 else c if k == 1 else "Neutral") for t, k in en_codes.items()}
            label.update({t: ("both" if codes_all[t] == 2 else c if codes_all[t] == 1 else "Neutral")
                          for t in V if t not in label and t in codes_all})
            fr = collect_feature_rows_for_embeddings(G, V, label, TokenizerOnly(get_tokenizer()), C.MODEL_SAFE, c,
                                                     C.RANK, max_activating=100000)
            csv = Path(str(ROOTS[v] / rel).replace("/pickles/", "/csvs/")).with_name("token_features.csv")
            pd.DataFrame(fr).to_csv(csv, index=False)
            tv, cv = edit_vectors(c, ROOTS[v])
            assert set(tv) == E0 | set(toks), f"{c}/{v}: edit is not E0 + the control tokens"
            got = sorted((float(cv[tv.index(t)].norm()) for t in toks), reverse=True)
            assert np.allclose(got, norm_x, rtol=1e-4), f"{c}/{v}: edit norms do not match X's"
            r[f"overlap_{v}_X"] = round(len(set(toks) & set(add_x)) / len(add_x), 3)
            r[f"tokens_{v}"] = p2x.tok_str(toks[:30])
        r["tokens_X"] = p2x.tok_str(sorted(add_x, key=lambda t: -float(cx[tx.index(t)].norm()))[:30])
        rows.append(r)
        log(c, {k: v for k, v in r.items() if not k.startswith("tokens")})
    OUT.mkdir(parents=True, exist_ok=True)
    p2x.save_json(OUT / "build.json", rows)


# ---------------------------------------------------------------- run (GPU)
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
        d = pick_delta(c)
        for v in VARIANTS:
            path = od / f"{v}_d{d:g}.json"
            if path.exists():
                continue
            t0 = time.time()
            features.MF_OUTPUTS_ROOT = ROOTS[v]
            info = m.edit(c, d)
            try:
                with torch.no_grad():
                    res = X11.score(m, gen, c)
            finally:
                m.restore()
            acc = {k: round(r["acc"] if k == "mmlu" else r["accuracy"], 4) for k, r in res.items()}
            p2x.save_json(path, {"concept": c, "variant": v, "delta": d, "edit_info": info, "accuracy": acc,
                                 "results": res, "seconds": round(time.time() - t0, 1), "smoke": X11.SMOKE})
            log(f"{c} {v} δ={d}: tokens={info['n_tokens_edited']} mmlu {acc['mmlu']:.2f} "
                + " ".join(f"{L} {acc[f'{L}/QA_train']:.2f}/{acc[f'{L}/SimdomQA_train']:.2f}" for L in LANGS)
                + f" ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- report (CPU)
def report():
    import numpy as np
    import pandas as pd
    from crosslingual.run_edited import _retention
    rows = []
    for c in p2x.CONCEPTS:
        d = pick_delta(c)
        x11 = X11.OUT / C.safe(c)
        base = json.loads((x11 / "pristine.json").read_text())["accuracy"]
        files = {"EMBER": x11 / f"EMBER_d{d:g}.json", "X": x11 / f"X_d{d:g}.json",
                 **{v: OUT / C.safe(c) / f"{v}_d{d:g}.json" for v in VARIANTS}}
        for v, p in files.items():
            if not p.exists():
                continue
            J = json.loads(p.read_text())
            acc = J["accuracy"]
            r = {"concept": c, "variant": v, "delta": d, "n_tokens": J["edit_info"]["n_tokens_edited"],
                 "mmlu_frac": round(acc["mmlu"] / base["mmlu"], 3)}
            for L in LANGS:
                ret = _retention(acc[f"{L}/QA_train"], base[f"{L}/QA_train"])
                r[f"ret_{L}"] = None if ret is None else round(ret, 3)
                r[f"sim_{L}"] = round(acc[f"{L}/SimdomQA_train"] - base[f"{L}/SimdomQA_train"], 3)
            r["ret_foreign"] = round(float(np.mean([r[f"ret_{L}"] for L in LANGS[1:]])), 3)
            r["leak"] = round(r["ret_foreign"] - r["ret_en"], 3)
            r["sim_foreign"] = round(float(np.mean([r[f"sim_{L}"] for L in LANGS[1:]])), 3)
            rows.append(r)
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "results.csv", index=False)
    order = ["EMBER", "X"] + VARIANTS
    cols = ["ret_en", "ret_de", "ret_es", "ret_fr", "ret_ru", "ret_foreign", "leak", "sim_en", "sim_foreign",
            "mmlu_frac"]
    agg = D.groupby("variant")[["n_tokens"] + cols].mean().round(3).reindex([v for v in order if v in set(D.variant)])
    agg.insert(0, "concepts", D.groupby("variant").size().reindex(agg.index).values)
    L_ = ["# X12: token-choice control for extrapolation (EMBER, 6 concepts, real erasure, train questions)", "",
          "Same δ (the fair run's EMBER pick) for all. X = EMBER+X; F_own / R_own = EMBER + the same number of added "
          "tokens with the same edit norms, chosen by sentence frequency / at random from the same concept-only pool, "
          "each edited along its own component on the selected features. ret_L = chance-corrected retention of "
          "QA_train (lower = more erased); sim_L = SimdomQA_train accuracy change vs pristine (collateral). EMBER, X "
          "and pristine from X11 (same items and scoring).", "", "## Mean over concepts", "",
          p2x.md_table(agg.reset_index()), ""]
    piv = D.pivot(index="concept", columns="variant")
    for v in VARIANTS:
        if {"X", v} <= set(D.variant):
            for col in ("ret_foreign", "ret_en", "sim_foreign", "sim_en"):
                dd = (piv[col][v] - piv[col]["X"]).dropna()
                L_.append(f"- {v} − X, {col}: {dd.mean():+.3f} (lower in {int((dd < 0).sum())}, higher in "
                          f"{int((dd > 0).sum())} of {len(dd)})")
    L_ += ["", "## Per concept", "", p2x.md_table(D[["concept", "variant", "n_tokens"] + cols]), ""]
    B = json.loads((OUT / "build.json").read_text())
    L_ += ["## Added tokens (first 30)", ""]
    for b in B:
        for v in ["X"] + VARIANTS:
            L_.append(f"- {b['concept']} / {v}: " + " ".join(b[f"tokens_{v}"]))
    (OUT / "report.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_[:20]))


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
