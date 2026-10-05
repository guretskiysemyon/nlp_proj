#!/usr/bin/env python3
"""Probe: does a concept's embedding-track subspace (fit from English tokens
only, see ember/train_mf_features.py) still pick out the right words in other
languages?

Reuses the already-trained embedding-track NMF directions for a concept
(mf_outputs/.../embedding/embedding.pkl), restricted to the subset already
validated as concept-specific (interpret_features.py's potential_features.csv),
and scores every vocab token by what fraction of its own embedding energy
falls inside that subspace. No forward pass / generation involved -- just an
orthonormalization of a handful of vectors and one matmul against the
embedding matrix, so this runs in seconds once the model is loaded.

Usage:
    python experiments/crosslingual_embed_projection.py --concept Golf
    python experiments/crosslingual_embed_projection.py --concept Golf --words-json my_words.json

--words-json expects a JSON list of {"lang": "fr", "text": "golf"} objects.
Without it, a small built-in example list is used (meaningful only for
--concept Golf).
"""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from ember.utils import (
    _safe_concept, _safe_model_name, get_embedding_matrix, get_pipeline_path,
)


def _hf_login():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        try:
            login(token=token)
        except (ValueError, RuntimeError):
            pass

DEFAULT_WORDS = [
    {"lang": "en", "text": "golf"},
    {"lang": "fr", "text": "golf"},
    {"lang": "de", "text": "Golf"},
    {"lang": "es", "text": "golf"},
    {"lang": "ru", "text": "гольф"},
    {"lang": "ja", "text": "ゴルフ"},
    {"lang": "zh", "text": "高尔夫"},
    {"lang": "ar", "text": "جولف"},
    # a couple of unrelated control words, so you have a feel for "normal" scores
    {"lang": "en", "text": "banana"},
    {"lang": "fr", "text": "banane"},
]


def load_concept_subspace(outdir: str, model_safe: str, rank: int, seed: int, concept: str, ratio_thresh: float):
    concept_safe = _safe_concept(concept)
    pkl_path = get_pipeline_path(outdir, model_safe, "pickles", rank, concept_safe, "embedding", "embedding.pkl", seed=seed)
    stats_path = get_pipeline_path(outdir, model_safe, "csvs", rank, concept_safe, "embedding", "stats_embed.csv", seed=seed)

    if not Path(pkl_path).exists():
        raise FileNotFoundError(
            f"No embedding-track pickle for '{concept}' at {pkl_path}. "
            "Run train_mf_features.py --concepts <concept> --skip-mlp first."
        )

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    F_ = payload["nmf"].F_.detach().float()  # (d_model, rank)

    # Prefer the Gemini-validated concept-specific subset (interpret_features.py's
    # potential_features.csv), since ratio_abs>=thresh alone pulls in a lot of
    # generic distributional directions (numbers, "commonly", transition verbs,
    # ...) that differ between concept/neutral sentences without meaning the
    # concept -- those dominate a naive sum and bury the real signal (see run
    # notes / README). Falls back to the blunter ratio_abs threshold if no
    # interpretation pass has been run for this concept yet.
    base_dir = Path(outdir) / model_safe
    potential_path = base_dir / "interpretations" / f"rank{rank}" / f"seed{seed}" / concept_safe / "embedding" / "potential_features.csv"
    if potential_path.exists():
        pot = pd.read_csv(potential_path)
        passing = sorted(pot["feature"].astype(int).unique().tolist())
        source = f"potential_features.csv ({potential_path})"
    else:
        stats = pd.read_csv(stats_path)
        passing = stats[stats["ratio_abs"] >= ratio_thresh]["feature"].astype(int).tolist()
        source = f"ratio_abs>={ratio_thresh} on stats_embed.csv (no interpretation pass found -- this is noisier)"
    if not passing:
        raise RuntimeError(f"No embedding-track features selected for '{concept}' (source: {source}).")
    return F_, passing, source


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", default="Golf")
    ap.add_argument("--rank", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", default="google/gemma-2-2b-it")
    ap.add_argument("--outdir", default="mf_outputs")
    ap.add_argument("--ratio-thresh", type=float, default=2.0,
                     help="Same threshold interpret_features.py uses to call a feature 'concept-passing'.")
    ap.add_argument("--words-json", default=None)
    ap.add_argument("--top-k", type=int, default=15, help="English sanity-check tokens to print.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    words = json.loads(Path(args.words_json).read_text()) if args.words_json else DEFAULT_WORDS

    _hf_login()
    model_safe = _safe_model_name(args.model_name)
    F_, passing_features, source = load_concept_subspace(
        args.outdir, model_safe, args.rank, args.seed, args.concept, args.ratio_thresh)
    print(f"[{args.concept}] subspace = {len(passing_features)}/{F_.shape[1]} features {passing_features}, "
          f"from {source}")

    print(f"Loading {args.model_name} ...")
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device, dtype=torch.bfloat16)
    model.eval()

    # Orthonormalize the K' concept directions, then score every vocab token by
    # what fraction of ITS OWN embedding energy lies in that subspace:
    #   score(tok) = ||E[tok] @ Q||^2 / ||E[tok]||^2  in [0, 1]
    # This is scale-invariant, unlike a raw dot product (or going through W_U,
    # which is what train_mf_features.py's projection_top_tokens does) --
    # raw dot products here turn out to be dominated by a handful of
    # huge-norm "glitch" tokens (OCR-era long-s spellings, unused specials)
    # that have nothing to do with the concept, for ANY direction, not just
    # this one. Normalizing by each token's own norm removes that confound.
    # Done on CPU: it's one matmul over the full vocab, cheap either way, and
    # keeping a float32 copy of E alongside the GPU-resident model is what
    # was blowing out 12GB cards.
    E = get_embedding_matrix(model).detach().float().cpu()  # (|vocab|, d_model)
    V = F_[:, passing_features].cpu()                        # (d_model, K')
    Q, _ = torch.linalg.qr(V)                                # (d_model, K'), orthonormal columns
    with torch.no_grad():
        proj_energy = (E @ Q).pow(2).sum(dim=1)   # (|vocab|,)
        total_energy = E.pow(2).sum(dim=1).clamp_min(1e-12)
        score = proj_energy / total_energy         # (|vocab|,) in [0, 1]

    vocab_size = score.shape[0]
    sorted_idx = torch.argsort(score, descending=True)
    rank_of = {int(tok): r for r, tok in enumerate(sorted_idx.tolist())}

    print(f"\nTop {args.top_k} English vocab tokens for this subspace:")
    for tok_id in sorted_idx[:args.top_k].tolist():
        print(f"  {model.to_single_str_token(tok_id)!r:15s} score={score[tok_id].item():.4f}")

    print(f"\n{'lang':4s} {'text':10s} {'token(s)':30s} {'best_score':>10s} {'pct_rank':>9s}")
    for w in words:
        ids = model.to_tokens(w["text"], prepend_bos=False)[0].tolist()
        toks_str = [model.to_single_str_token(i) for i in ids]
        best_id = max(ids, key=lambda i: score[i].item())
        pct = 100.0 * (1.0 - rank_of[best_id] / vocab_size)
        print(f"{w['lang']:4s} {w['text']:10s} {str(toks_str):30s} {score[best_id].item():10.4f} {pct:8.2f}%")

    print(
        "\nNote: for multi-token words the line above takes the single best-scoring sub-token "
        "(printed under 'token(s)'), which is a crude proxy -- if a non-Latin script splits the "
        "word into many pieces, none individually meaning the whole word, treat a low score here "
        "as inconclusive rather than as evidence the concept doesn't transfer."
    )


if __name__ == "__main__":
    main()
