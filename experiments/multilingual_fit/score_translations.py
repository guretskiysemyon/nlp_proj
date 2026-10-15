#!/usr/bin/env python3
"""Step 2 of the translation plan: a quality score for every (English, translation) pair.

Two reference-free scores per pair:
  kiwi   COMET-Kiwi (Unbabel/wmt22-cometkiwi-da), the standard MT quality-estimation
         model; ~0-1, higher = better translation of the source.
  labse  cosine similarity of LaBSE sentence embeddings of the English and the
         translation; low = meaning drifted or content was dropped.

Needs unbabel-comet, which pins protobuf < 5 and so cannot go into the project .venv
(the Gemini SDK there needs protobuf 5). Run it from the separate environment
/home/morg/NLP_2526b/semyong/.venv-comet (see run_score_translations.sbatch).

Output (<data-dir>/quality/):
    scores.csv     one row per pair: source, lang, set, idx, kiwi, labse, len_ratio, en, tr
    summary.csv    per (source, lang, set): n, mean / p05 / min of both scores
    worst.csv      per language, the lowest-kiwi --worst-frac of the main data (step 3 input)

Usage:
    .venv-comet/bin/python -I experiments/multilingual_fit/score_translations.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LANGS = ["ru", "de", "es", "fr"]
NEUTRAL = "__neutral__"


def load_pairs(source, data_dir, concepts_only=None):
    """[(source, lang, set, idx, en, tr)] for every translated sentence in data_dir."""
    en = {c["concept"]: c["sentences"] for c in json.loads((REPO / "data" / "concept_sentences.json").read_text())}
    en[NEUTRAL] = [n["sentence"] for n in json.loads((REPO / "data" / "neutral_sentences.json").read_text())]
    rows = []
    for lang in LANGS:
        c_path = data_dir / f"concept_sentences_{lang}.json"
        sets = {c["concept"]: c["sentences"] for c in json.loads(c_path.read_text())} if c_path.exists() else {}
        n_path = data_dir / f"neutral_sentences_{lang}.json"
        if n_path.exists() and not concepts_only:
            sets[NEUTRAL] = [n["sentence"] for n in json.loads(n_path.read_text())]
        for name, tr in sets.items():
            if concepts_only and name not in concepts_only:
                continue
            assert len(tr) == len(en[name]), f"{source}/{lang}/{name}: length mismatch"
            rows += [(source, lang, name, i, e, t) for i, (e, t) in enumerate(zip(en[name], tr))]
    return rows


def score_both(src, mt, batch_size):
    """(kiwi, labse) arrays for aligned lists of English sources and translations."""
    from sentence_transformers import SentenceTransformer
    labse = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
    e = labse.encode(src, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    t = labse.encode(mt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    labse_scores = (e * t).sum(1)
    del labse
    print("LaBSE done", flush=True)

    from comet import download_model, load_from_checkpoint
    kiwi = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
    out = kiwi.predict([{"src": s, "mt": m} for s, m in zip(src, mt)],
                       batch_size=batch_size, gpus=1, progress_bar=False)
    print("COMET-Kiwi done", flush=True)
    return np.array(out.scores), labse_scores


def score_candidates(args):
    """Score retranslate_low.py candidates; also record the translation they would replace."""
    d = json.loads(args.candidates.read_text())
    cands = d["candidates"]
    current = {(r[1], r[2], r[3]): r[5] for r in load_pairs("main", args.data_dir)}
    kiwi, labse = score_both([c["en"] for c in cands], [c["tr"] for c in cands], args.batch_size)
    for c, k, l in zip(cands, kiwi, labse):
        c.update(kiwi=float(k), labse=float(l), old_tr=current[(c["lang"], c["set"], c["idx"])])
    out = args.candidates.with_name("candidates_scored.json")
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE / "data_translated")
    ap.add_argument("--compare-dir", type=Path, default=HERE / "data_translated_flash_pilot",
                    help="also score this dir's World War II set, for comparison (skipped if missing)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--worst-frac", type=float, default=0.02)
    ap.add_argument("--candidates", type=Path, default=None,
                    help="score retranslate_low.py candidates instead -> <same dir>/candidates_scored.json")
    args = ap.parse_args()

    if args.candidates:
        score_candidates(args)
        return

    rows = load_pairs("main", args.data_dir)
    if args.compare_dir and args.compare_dir.exists():
        rows += load_pairs(args.compare_dir.name, args.compare_dir, concepts_only={"World War II"})
    print(f"{len(rows)} pairs", flush=True)
    src = [r[4] for r in rows]
    mt = [r[5] for r in rows]

    kiwi_scores, labse_scores = score_both(src, mt, args.batch_size)

    q = args.data_dir / "quality"
    q.mkdir(exist_ok=True)
    with open(q / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "lang", "set", "idx", "kiwi", "labse", "len_ratio", "en", "tr"])
        for r, k, l in zip(rows, kiwi_scores, labse_scores):
            w.writerow([r[0], r[1], r[2], r[3], f"{k:.4f}", f"{l:.4f}", f"{len(r[5]) / max(len(r[4]), 1):.2f}", r[4], r[5]])

    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r[0], r[1], r[2]), []).append(i)
    with open(q / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "lang", "set", "n", "kiwi_mean", "kiwi_p05", "kiwi_min", "labse_mean", "labse_p05", "labse_min"])
        for (s, lang, name), idx in sorted(groups.items()):
            k, l = kiwi_scores[idx], labse_scores[idx]
            w.writerow([s, lang, name, len(idx), *(f"{v:.4f}" for v in
                        (k.mean(), np.percentile(k, 5), k.min(), l.mean(), np.percentile(l, 5), l.min()))])

    with open(q / "worst.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lang", "set", "idx", "kiwi", "labse", "en", "tr"])
        for lang in LANGS:
            idx = [i for i, r in enumerate(rows) if r[0] == "main" and r[1] == lang]
            idx = sorted(idx, key=lambda i: kiwi_scores[i])[:max(1, round(len(idx) * args.worst_frac))]
            for i in idx:
                r = rows[i]
                w.writerow([lang, r[2], r[3], f"{kiwi_scores[i]:.4f}", f"{labse_scores[i]:.4f}", r[4], r[5]])
    print(f"wrote {q}/scores.csv, summary.csv, worst.csv")


if __name__ == "__main__":
    main()
