"""Input-embedding rows (gemma-2-2b-it, x sqrt(d), float32, as prep.load_input_embeddings) of every new token.

    .venv/bin/python -m experiments.paper_analysis.extrapolation.embrows

Reads tokens/<Slug>.json (tokens.py) and writes tokens/<Slug>_Enew.pt {"ids": (m,) long, "E": (m, d) f32} so the
scoring step never loads the full 256k x 2304 matrix.
"""
from __future__ import annotations

import json

import torch

from experiments.paper_analysis.extrapolation.common import TOKDIR


def main():
    from experiments.varwidth.prep import load_input_embeddings
    E = load_input_embeddings()
    for p in sorted(TOKDIR.glob("*.json")):
        d = json.loads(p.read_text())
        ids = torch.tensor(d["new"], dtype=torch.long)
        torch.save({"ids": ids, "E": E.index_select(0, ids).clone()}, p.with_name(p.stem + "_Enew.pt"))
        print(f"[embrows] {d['concept']}: {len(ids)} rows", flush=True)


if __name__ == "__main__":
    main()
