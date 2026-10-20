#!/usr/bin/env python
"""Part 2 arms whose features are fitted on native text (docs/Part 2/plan.md, "Earlier design": arms A and B),
run through the fair comparison (cinder/fair.py) like every other arm.

    ml_joint  arm A: one EMBER factorization of en+ru+de+es+fr together (data/ + experiments/multilingual_fit/
              data_v2), rank JOINT_RANK (500), seed 42, EMBER's two-stage judge with gemini-3.5-flash-lite.
              The fit's own features, labels and edit, unchanged (mf_outputs/ml_joint/: symlinks).
    ml_sep    arm B: one fit per language at rank 100: en = the fair run's EMBER (paper's factorization + the
              gemini-3.5 judge, mf_outputs/ember35), de / es / fr / ru = EMBER on that language's native sentences
              alone, same judge. The edit is the union of the five EMBER edits, each token edited ONCE: by English
              if English edits it (so EMBER's English edit is kept exactly), otherwise by the foreign fit with the
              largest ||c_t||. Written as one factorization in EMBER's format (mf_outputs/ml_sep/): F = the kept
              columns of the five fits side by side, G = each edited token's row in its source fit's block (zeros in
              the others), so EMBER's edit code reproduces every token's c_t exactly (checked).

Both are written under rank100/ (the path key cinder.fair uses); the joint fit's own rank is in build json.
Fits: cinder/run_ml_fit.sbatch -> mf_outputs/rank_search/multi_language/<langs>/ (as run_rank_search.sbatch).

    python -m cinder.fair build-ml [--concepts ...]     # CPU, after the fits
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

from cinder import common as C

JOINT_TAG = "en+ru+de+es+fr"
JOINT_RANK = int(os.environ.get("ML_JOINT_RANK", "500"))
FOREIGN = ["de", "es", "fr", "ru"]
RS = C.ROOT / "mf_outputs" / "rank_search" / "multi_language"
ROOT_JOINT = C.ROOT / "mf_outputs" / "ml_joint"
ROOT_SEP = C.ROOT / "mf_outputs" / "ml_sep"
KINDS = ("pickles", "csvs", "interpretations")


def log(*a):
    import time
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def fit_dir(tag, rank, kind, concept) -> Path:
    return RS / tag / C.MODEL_SAFE / kind / f"rank{rank}" / f"seed{C.SEED}" / C.safe(concept) / "embedding"


def arm_dir(root, kind, concept) -> Path:
    return root / C.MODEL_SAFE / kind / f"rank{C.RANK}" / f"seed{C.SEED}" / C.safe(concept) / "embedding"


def fit_ready(tag, rank, concept) -> bool:
    return ((fit_dir(tag, rank, "pickles", concept) / "embedding.pkl").exists()
            and (fit_dir(tag, rank, "csvs", concept) / "token_features.csv").exists()
            and (fit_dir(tag, rank, "interpretations", concept) / "potential_features.csv").exists())


def kept(pot_csv: Path):
    """[(feature, metric_score)] the edit selects (metric_score >= 2, as cinder.fair.kept_features)."""
    import pandas as pd
    df = pd.read_csv(pot_csv)
    return sorted((int(f), float(m)) for f, m in zip(df.get("feature", []), df.get("metric_score", []))
                  if float(m) >= 2.0)


def is_cyrillic(s: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in s)


def tok_strs(ids):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    return [tok.decode([t]) for t in ids]


# ---------------------------------------------------------------- arm A: joint fit
def build_joint(concept) -> dict:
    from cinder import fair as F
    from cinder.edit_size import edit_vectors
    if not fit_ready(JOINT_TAG, JOINT_RANK, concept):
        return {"concept": concept, "arm": "ml_joint", "missing": True}
    for kind in KINDS:
        link = arm_dir(ROOT_JOINT, kind, concept)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(fit_dir(JOINT_TAG, JOINT_RANK, kind, concept), target_is_directory=True)
    k = F.kept_features("ml_joint", concept)
    tids = edit_vectors(concept, ROOT_JOINT)[0] if k else []
    s = tok_strs(tids)
    r = {"concept": concept, "arm": "ml_joint", "missing": False, "fit_rank": JOINT_RANK, "kept": k,
         "n_edit": len(tids), "n_cyrillic": sum(map(is_cyrillic, s)), "edited": s}
    log(f"ml_joint {concept}: rank {JOINT_RANK}, kept {k}, {len(tids)} tokens edited ({r['n_cyrillic']} Cyrillic)")
    return r


# ---------------------------------------------------------------- arm B: per-language fits, union of the edits
def source_edit(root, concept):
    """(kept [(f, score)], F[:, kept] float, {tid: G row on kept}, {tid: c_t}) of one fit, as the edit builds them."""
    import pandas as pd
    import torch
    from ember.erasure import embed_edit, features
    from cinder.edit_size import edit_vectors
    features.MF_OUTPUTS_ROOT = root
    pkl, pot = features._embedding_paths(C.MODEL_NAME, concept, C.RANK, C.SEED)
    ks = kept(pot)
    if not ks:
        return ks, None, {}, {}
    Fd, G, vp = features.load_embedding_payload(pkl)
    feat = features.select_embed_feature_ids(pd.read_csv(pot), ratio_thresh=2.0)
    assert feat == [f for f, _ in ks], (concept, root, feat, ks)
    ctok = embed_edit._build_concept_token_set(C.MODEL_NAME, concept, C.RANK, C.SEED, vp)
    Gp = torch.as_tensor(G[:, feat]).float()
    tids, rows = embed_edit._eligible_tids_rows(vp, Gp, ctok or set())
    t2, cm = edit_vectors(concept, root)
    assert sorted(t2) == sorted(tids), (concept, root)
    cvec = dict(zip(t2, cm))
    return ks, torch.as_tensor(Fd[:, feat]).float(), {t: Gp[r] for t, r in zip(tids, rows)}, cvec


def build_sep(concept) -> dict:
    import pandas as pd
    import torch
    from cinder import fair as F
    from cinder.edit_size import edit_vectors
    from ember.utils import collect_feature_rows_for_embeddings
    from experiments.varwidth.prep import TokenizerOnly, get_tokenizer
    srcs = [("en", F.ROOTS["ember"])] + [(L, RS / L) for L in FOREIGN]
    missing = [L for L in FOREIGN if not fit_ready(L, C.RANK, concept)]
    if missing:
        return {"concept": concept, "arm": "ml_sep", "missing": True, "missing_langs": missing}
    parts = {L: source_edit(root, concept) for L, root in srcs}
    # each token once: English first (keeps EMBER's edit), else the foreign fit with the largest ||c_t||
    choice = {t: "en" for t in parts["en"][3]}
    for L in FOREIGN:
        for t, c in parts[L][3].items():
            if t in choice and choice[t] == "en":
                continue
            if t not in choice or float(c.norm()) > float(parts[choice[t]][3][t].norm()):
                choice[t] = L
    dirs = {kind: arm_dir(ROOT_SEP, kind, concept) for kind in KINDS}
    for d in dirs.values():
        if d.is_symlink():
            d.unlink()
        d.mkdir(parents=True, exist_ok=True)
    pot_rows, blocks, off = [], [], {}
    col = 0
    for L, _ in srcs:
        ks, Fk, _, _ = parts[L]
        off[L] = col
        if ks:
            blocks.append(Fk)
            for j, (f, m) in enumerate(ks):
                pot_rows.append({"model": C.MODEL_NAME, "concept": concept, "rank": C.RANK, "layer": 0,
                                 "feature": col + j, "metric_score": m, "source_lang": L, "source_feature": f})
            col += len(ks)
    r = {"concept": concept, "arm": "ml_sep", "missing": False,
         "kept": {L: [f for f, _ in parts[L][0]] for L, _ in srcs},
         "n_edit_by_fit": {L: len(parts[L][3]) for L, _ in srcs},
         "n_edit_chosen_from": {L: sum(v == L for v in choice.values()) for L, _ in srcs}}
    pot_cols = ["model", "concept", "rank", "layer", "feature", "metric_score", "source_lang", "source_feature"]
    pd.DataFrame(pot_rows, columns=pot_cols).to_csv(dirs["interpretations"] / "potential_features.csv", index=False)
    if not choice:
        for f in (dirs["pickles"] / "embedding.pkl", dirs["csvs"] / "token_features.csv"):
            f.unlink(missing_ok=True)
        log(f"ml_sep {concept}: no fit keeps a feature -> no edit")
        return {**r, "n_edit": 0}
    Fm = torch.cat(blocks, 1)                                    # (d, sum k)
    V = sorted(choice)
    G = torch.zeros(len(V), col)
    for i, t in enumerate(V):
        L = choice[t]
        g = parts[L][2][t]
        G[i, off[L]:off[L] + len(g)] = g
    with open(dirs["pickles"] / "embedding.pkl", "wb") as fh:
        pickle.dump({"nmf": SimpleNamespace(F_=Fm, G_=G), "vprime_token_ids": V,
                     "source": "ml_sep: union of the en (ember35) / de / es / fr / ru EMBER edits, each token once"}, fh)
    fr = collect_feature_rows_for_embeddings(G, V, {t: concept for t in V}, TokenizerOnly(get_tokenizer()),
                                             C.MODEL_SAFE, concept, C.RANK, max_activating=100000)
    pd.DataFrame(fr).to_csv(dirs["csvs"] / "token_features.csv", index=False)
    # check: the merged edit gives every chosen token its source fit's c_t, EMBER's English edit included
    tx, cx = edit_vectors(concept, ROOT_SEP)
    pos = {t: i for i, t in enumerate(tx)}
    assert set(tx) == set(V), f"ml_sep {concept}: edited tokens differ from the chosen set"
    for t in V:
        ref = parts[choice[t]][3][t]
        assert torch.allclose(cx[pos[t]], ref, rtol=0, atol=1e-6), f"ml_sep {concept}: c_t differs for {t}"
    s = tok_strs(V)
    r.update({"n_edit": len(V), "n_added_to_english": len(V) - len(parts["en"][3]),
              "english_edit_kept": all(choice[t] == "en" for t in parts["en"][3]),
              "n_cyrillic": sum(map(is_cyrillic, s)),
              "added": [x for x, t in zip(s, V) if choice[t] != "en"]})
    log(f"ml_sep {concept}: kept {r['kept']}, edit {len(parts['en'][3])} (en) -> {len(V)} tokens "
        f"(from {r['n_edit_chosen_from']}; {r['n_cyrillic']} Cyrillic); every c_t reproduced")
    return r


def build(concepts=None):
    from cinder import fair as F
    rows = []
    for c in concepts or F.ALL_CONCEPTS:
        for fn, root, arm in ((build_joint, ROOT_JOINT, "ml_joint"), (build_sep, ROOT_SEP, "ml_sep")):
            try:
                r = fn(c)
            except Exception as e:  # noqa: BLE001  one bad concept must not block the others' erasure jobs
                for kind in KINDS:                    # no half-written arm: its erasure job fails loudly instead
                    d = arm_dir(root, kind, c)
                    if d.is_symlink():
                        d.unlink()
                    elif d.exists():
                        import shutil
                        shutil.rmtree(d)
                r = {"concept": c, "arm": arm, "missing": True, "error": repr(e)}
                log(f"{arm} {c}: ERROR {e!r}")
            if r.get("missing"):
                log(f"{r['arm']} {c}: MISSING fit(s) {r.get('missing_langs', JOINT_TAG)} -> not built "
                    f"(its erasure job will fail)")
            rows.append(r)
    out = F.FAIR / "ml_build.json"
    old = {(r["concept"], r["arm"]): r for r in json.loads(out.read_text())} if out.exists() else {}
    old.update({(r["concept"], r["arm"]): r for r in rows})
    C.write_json(out, list(old.values()))
    log(f"wrote {out} ({sum(not r.get('missing') for r in rows)} of {len(rows)} built)")
