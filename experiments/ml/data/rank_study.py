"""Sample-size study: |V*| and numerical rank of E_V* per concept and N (results/ml/data/SAMPLE.md).

V* = unique non-special Llama-3.1 token ids of the N concept + N neutral sample sentences of all 8 languages,
tokenized like TransformerLens to_tokens (BOS + tokens, BOS then dropped as special). E_V* = those rows of
model.embed_tokens.weight (only that tensor is read from the safetensors shard, float32). Rank via singular values,
tol = max(sv) * max(shape) * eps(float32). Rule: smallest N with |V*| >= 1.25 * d and rank = d for every concept.
With ML_CONCEPTS=new (or all) N is not re-chosen: N = 30 is checked, rows go to sample_study_<set>.json and a
"New concepts" section at the end of SAMPLE.md (the 18-concept part above it is kept).

    python -m experiments.ml.data.rank_study      (SLURM: experiments/ml/data/rank.sbatch)
"""
from __future__ import annotations

import json
import os
import time

import torch

from experiments.ml.data.common import FULL, LANGS, OUT, SAMPLE_NS, concepts, dump_json, load_json
from experiments.ml.data.io import load_full, load_sample, to_examples

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def embed_matrix():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    snap = snapshot_download(MODEL, allow_patterns=["*.json"])
    idx = json.load(open(f"{snap}/model.safetensors.index.json"))
    shard = idx["weight_map"]["model.embed_tokens.weight"]
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(MODEL, shard)
    with safe_open(path, framework="pt") as f:
        return f.get_tensor("model.embed_tokens.weight").float()


def vocab_ids(tok, sents, special):
    ids = set()
    for s in sents:
        ids.update([tok.bos_token_id] + tok.encode(s, add_special_tokens=False))
    return sorted(ids - special)


def rank_of(E, dev):
    M = E.to(dev)
    sv = torch.linalg.svdvals(M)
    tol = sv.max() * max(M.shape) * torch.finfo(torch.float32).eps
    return int((sv > tol).sum()), float(sv.min()), float(sv.max())


def main():
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    special = set(tok.all_special_ids) | set(tok.added_tokens_decoder.keys())
    W = embed_matrix()
    d = W.shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"embed {tuple(W.shape)} loaded in {time.time() - t0:.0f}s; device {dev}", flush=True)
    rows = {}
    for c in concepts():
        rows[c] = {}
        for n in SAMPLE_NS:
            sents, *_ = to_examples(load_sample(c, n))
            v = vocab_ids(tok, sents, special)
            r, smin, smax = rank_of(W[v], dev)
            rows[c][n] = {"V": len(v), "rank": r, "sv_min": smin, "sv_max": smax, "n_sent": len(sents)}
        sents, *_ = to_examples(load_full(c))
        v = vocab_ids(tok, sents, special)
        rows[c]["full"] = {"V": len(v), "n_sent": len(sents)}
        print(c, rows[c], flush=True)
    chosen = next((n for n in SAMPLE_NS if all(rows[c][n]["V"] >= 1.25 * d and rows[c][n]["rank"] == d
                                               for c in rows)), None)
    which = os.environ.get("ML_CONCEPTS", "orig")
    if which == "orig":
        dump_json(OUT / "sample_study.json", {"d": d, "rows": rows, "chosen_N": chosen})
        write_md(rows, d, chosen)
    else:  # N is frozen at FROZEN_N (PREREG_NEW.md): check it, do not re-choose; SAMPLE.md's 18-concept part is kept
        dump_json(OUT / f"sample_study_{which}.json", {"d": d, "rows": rows, "frozen_N": FROZEN_N,
                                                        "smallest_N_passing": chosen})
        append_md(rows, d)


FROZEN_N = 30
NEW_HEADER = "## New concepts (PREREG_NEW.md)"


def append_md(rows, d):
    """Replace (or add) the "New concepts" section at the end of SAMPLE.md; everything before it is kept."""
    summary = load_json(FULL / "_summary.json")
    ok = {c: r[FROZEN_N]["V"] >= 1.25 * d and r[FROZEN_N]["rank"] == d for c, r in rows.items()}
    L = [NEW_HEADER, "",
         f"The 20 concepts of data/concept_sentences_extra.json, same rule and procedure. N is frozen at {FROZEN_N}; "
         f"the check is |V\\*| >= {int(1.25 * d)} and rank {d} at N = {FROZEN_N} for every concept. "
         + (f"**All {len(rows)} pass at N = {FROZEN_N}.**" if all(ok.values()) else
            f"**Fail at N = {FROZEN_N}: {', '.join(c for c, v in ok.items() if not v)}.**"), "",
         "| Concept | " + " | ".join(f"N={n} \\|V\\*\\| / rank" for n in SAMPLE_NS) + " | full \\|V\\*\\| |",
         "| --- | " + " | ".join("---" for _ in SAMPLE_NS) + " | --- |"]
    for c, r in rows.items():
        L.append(f"| {c} | " + " | ".join(f"{r[n]['V']} / {r[n]['rank']}" for n in SAMPLE_NS)
                 + f" | {r['full']['V']} |")
    L.append("| min | " + " | ".join(f"{min(rows[c][n]['V'] for c in rows)} / {min(rows[c][n]['rank'] for c in rows)}"
                                     for n in SAMPLE_NS) + f" | {min(rows[c]['full']['V'] for c in rows)} |")
    L += ["", "### New concepts: concept sentences per (concept, language), native + translated (300 target)", "",
          "| Concept | " + " | ".join(LANGS) + " |", "| --- | " + " | ".join("---" for _ in LANGS) + " |"]
    for c in rows:
        L.append(f"| {c} | " + " | ".join(
            (f"{v['native']}+{v['translated']}t" if v["page_exists"] else f"no page: {v['translated']}t")
            for v in (summary[c][l] for l in LANGS)) + " |")
    path = OUT / "SAMPLE.md"
    old = path.read_text(encoding="utf-8")
    head = old.split("\n" + NEW_HEADER)[0].rstrip("\n")
    path.write_text(head + "\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def write_md(rows, d, chosen):
    summary = load_json(FULL / "_summary.json")
    L = ["# Sample size for A2 / A3", "",
         f"Rule (PLAN.md): smallest N in {{{', '.join(map(str, SAMPLE_NS))}}} such that for every concept "
         f"|V\\*| >= 1.25 * {d} = {int(1.25 * d)} and rank(E_V\\*) = {d}.",
         "V\\* = unique non-special Llama-3.1-8B-Instruct token ids of the N concept + N neutral sample sentences "
         "of all 8 languages (BOS + tokens, as TransformerLens). E_V\\* = rows of model.embed_tokens.weight (float32); "
         "rank = #singular values > max(sv) * max(shape) * eps.", "",
         f"**Chosen N = {chosen}**" if chosen else "**No N in the grid satisfies the rule.**", "",
         "| Concept | " + " | ".join(f"N={n} \\|V\\*\\| / rank" for n in SAMPLE_NS) + " | full \\|V\\*\\| |",
         "| --- | " + " | ".join("---" for _ in SAMPLE_NS) + " | --- |"]
    for c, r in rows.items():
        L.append(f"| {c} | " + " | ".join(f"{r[n]['V']} / {r[n]['rank']}" for n in SAMPLE_NS)
                 + f" | {r['full']['V']} |")
    L.append("| min | " + " | ".join(f"{min(rows[c][n]['V'] for c in rows)} / {min(rows[c][n]['rank'] for c in rows)}"
                                     for n in SAMPLE_NS) + f" | {min(rows[c]['full']['V'] for c in rows)} |")
    L += ["", "## Concept sentences per (concept, language): native + translated (full set, 300 target)", "",
          "| Concept | " + " | ".join(next(iter(summary.values())).keys()) + " |",
          "| --- | " + " | ".join("---" for _ in next(iter(summary.values()))) + " |"]
    for c, s in ((c, s) for c, s in summary.items() if c in rows):
        L.append(f"| {c} | " + " | ".join(
            (f"{v['native']}+{v['translated']}t" if v["page_exists"] else f"no page: {v['translated']}t")
            for v in s.values()) + " |")
    (OUT / "SAMPLE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
