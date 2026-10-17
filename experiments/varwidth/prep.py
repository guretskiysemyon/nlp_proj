"""Per-concept data for the variable-width study: V*, token labels and the matrix A the fit runs on.

Reproduces cells 3.1.a-3.1.c' and 3.2.a of low_ram_demands/ancient_rome_whitened_varnnz.ipynb (whitened) and
ancient_rome_varnnz.ipynb (plain), gemma-2-2b-it only:
    E           input embeddings read from the checkpoint, x sqrt(d) (TL's W_E for Gemma), float32
    dataset     ConceptDataset(concept, neutral_sample_seed=42); with n_concept_sents < all, a seeded random subset
                (random.Random(sub_seed), file order kept) of the concept sentences, neutral sentences unchanged
    V*          sorted unique non-special token ids of S_C u S_N (left-padded batches of 5, as the notebooks)
    plain       A = E_V*^T (d, n)
    whitened    A = X^T, X = (E_V* - mu_B) W S_eta^(gamma/2): background "wiki", target "page", gamma 1,
                eta / eps_rel from experiments/whitening/data/gemma/defaults.json

Cached at results/varwidth/prep/<slug>_<matrix>_S<n>.pt.

    python -m experiments.varwidth.prep --concept "Ancient Rome" --matrix whitened [--S 300] [--sub-seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.varwidth.common import RESULTS, ROOT, slug

MODEL_NAME = "google/gemma-2-2b-it"
MODEL_SLUG = "gemma"
NEUTRAL_SAMPLE_SEED = 42
BACKGROUND, TARGET, GAMMA = "wiki", "page", 1.0
WH_DIR = ROOT / "experiments" / "whitening" / "data" / MODEL_SLUG
PREP_DIR = RESULTS / "prep"
FULL_S = 300


def prep_path(concept: str, matrix: str, n_concept_sents: int = FULL_S, sub_seed: int = 0) -> Path:
    sub = f"_sub{sub_seed}" if (n_concept_sents < FULL_S and sub_seed != 0) else ""
    return PREP_DIR / f"{slug(concept)}_{matrix}_S{n_concept_sents}{sub}.pt"


class TokenizerOnly:
    """The parts of a HookedTransformer that 3.1-3.2 use (same as the notebook's low-RAM cell 10)."""

    def __init__(self, tok):
        self.tokenizer = tok

    def to_tokens(self, prompts, padding_side="left"):
        self.tokenizer.padding_side = padding_side
        return self.tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]

    def to_str_tokens(self, ids):
        return self.tokenizer.batch_decode(torch.as_tensor(ids).reshape(-1, 1), clean_up_tokenization_spaces=False)


_TOK = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _TOK = tok
    return _TOK


def load_input_embeddings(model_name: str = MODEL_NAME) -> torch.Tensor:
    """model.embed_tokens.weight straight from the checkpoint, float32, x sqrt(d) for Gemma (= TL W_E)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from safetensors import safe_open
    try:
        index = json.loads(Path(hf_hub_download(model_name, "model.safetensors.index.json")).read_text())
        shard = index["weight_map"]["model.embed_tokens.weight"]
    except EntryNotFoundError:
        shard = "model.safetensors"
    with safe_open(hf_hub_download(model_name, shard), framework="pt") as f:
        E = f.get_tensor("model.embed_tokens.weight").to(torch.float32)
    if "gemma" in model_name.lower():
        E.mul_(torch.tensor(math.sqrt(E.shape[1]), dtype=torch.float32))
    return E


def build_dataset(concept: str, n_concept_sents: int = FULL_S, sub_seed: int = 0):
    from ember.local_datasets import ConceptDataset
    ds = ConceptDataset(concept_name=concept, neutral_sample_seed=NEUTRAL_SAMPLE_SEED)
    c_idx = [i for i, (_, lbl) in enumerate(ds.data) if lbl == concept]
    if n_concept_sents < len(c_idx):
        keep = set(random.Random(sub_seed).sample(c_idx, n_concept_sents))
        ds.data = [p for i, p in enumerate(ds.data) if p[1] != concept or i in keep]
    elif n_concept_sents > len(c_idx):
        raise ValueError(f"{concept!r} has only {len(c_idx)} concept sentences, asked for {n_concept_sents}")
    return ds


def build_vstar(concept: str, dataset, tok) -> tuple[list[int], dict]:
    from ember.utils import build_token_label_codes, get_special_token_ids
    from llm_utils.activation_generator import extract_token_ids_sample_ids_and_labels
    tl_model = TokenizerOnly(tok)
    act_generator = SimpleNamespace(model=tl_model, data_device="cpu")
    token_ids_all, _, _ = extract_token_ids_sample_ids_and_labels(dataset, act_generator)
    special = get_special_token_ids(tl_model)
    vstar_ids = sorted({int(t) for t in token_ids_all.tolist()} - special)
    token_to_code = build_token_label_codes(concept, dataset, tok)
    return vstar_ids, token_to_code, special


def whitening_transform(concept: str, E: torch.Tensor, special: set) -> dict:
    """Notebook cell 16 (3.1.c'): background whitener + target sharpener. Returns the Transform pieces."""
    sys.path.insert(0, str(ROOT / "experiments" / "whitening"))
    from sharpen import BackgroundWhitener, Sharpener, Transform
    defaults = json.loads((WH_DIR / "defaults.json").read_text())
    eta, eps_rel = defaults["eta"], defaults["eps_rel"]
    bg_path = WH_DIR / f"bg_{BACKGROUND}_stats.pt"
    targets_path = WH_DIR / "targets" / f"{slug(concept)}.json"
    if not targets_path.exists():
        raise FileNotFoundError(f"{targets_path} missing (build it with experiments/varwidth/prep.py --build-target)")
    stats_bg = torch.load(bg_path, map_location="cpu")
    stats_bg = {k: (v.double() if torch.is_tensor(v) else v) for k, v in stats_bg.items()}
    bw = BackgroundWhitener(stats_bg, eps_rel)
    vocab = E.shape[0]
    targets = json.loads(targets_path.read_text())
    target_ids = sorted({int(t) for t in targets[TARGET] if 0 <= int(t) < vocab} - set(special))
    E_target = E.index_select(0, torch.tensor(target_ids)).double()
    U_target = bw.whiten(E_target)
    sh = Sharpener(U_target, eta, GAMMA) if GAMMA > 0 else None
    T = Transform(bw, sh)
    m_tilde = U_target.mean(0)
    m_X = m_tilde if sh is None else m_tilde @ sh.Sp
    T_inv = bw.W_inv if sh is None else sh.Sm @ bw.W_inv
    return {"T": T, "m_X": m_X, "T_inv": T_inv, "eta": eta, "eps_rel": eps_rel, "eps": float(bw.eps),
            "n_target": len(target_ids)}


def prepare(concept: str, matrix: str = "whitened", n_concept_sents: int = FULL_S, sub_seed: int = 0,
            force: bool = False, verbose: bool = True) -> dict:
    assert matrix in ("whitened", "plain"), matrix
    path = prep_path(concept, matrix, n_concept_sents, sub_seed)
    if path.exists() and not force:
        return torch.load(path, weights_only=False)
    t0 = time.time()
    tok = get_tokenizer()
    dataset = build_dataset(concept, n_concept_sents, sub_seed)
    vstar_ids, token_to_code, special = build_vstar(concept, dataset, tok)
    codes = np.array([token_to_code.get(t, 0) for t in vstar_ids])
    E = load_input_embeddings()
    E_vstar = E.index_select(0, torch.tensor(vstar_ids)).detach().float()   # (n, d)
    out = {"concept": concept, "matrix": matrix, "n_concept_sents": n_concept_sents, "sub_seed": sub_seed,
           "model": MODEL_NAME, "vstar_ids": vstar_ids, "codes": codes, "token_to_code": token_to_code,
           "tokens": tok.convert_ids_to_tokens(vstar_ids),
           "n_neutral_sents": sum(lbl == "Neutral" for _, lbl in dataset)}
    if matrix == "plain":
        out["A"] = E_vstar.T.contiguous()
    else:
        tr = whitening_transform(concept, E, special)
        E_X = tr["T"].forward(E_vstar).float()
        out["A"] = E_X.T.contiguous()
        out.update({"m_X": tr["m_X"], "T_matrix": tr["T"].matrix, "T_inv": tr["T_inv"],
                    "mu_B": tr["T"].bw.mu, "background": BACKGROUND, "target": TARGET, "gamma": GAMMA,
                    "eta": tr["eta"], "eps_rel": tr["eps_rel"], "eps": tr["eps"], "n_target": tr["n_target"]})
    del E
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(out, tmp)
    tmp.replace(path)
    if verbose:
        print(f"[prep] {concept} {matrix} S={n_concept_sents}: |V*|={len(vstar_ids)} "
              f"C/N/B={(codes == 1).sum()}/{(codes == 0).sum()}/{(codes == 2).sum()}  A {tuple(out['A'].shape)}  "
              f"({time.time() - t0:.0f}s) -> {path}", flush=True)
    return out


# ---------------------------------------------------------------- target file for concepts without one (HP)
def build_target_page(concept: str, page_source: Path) -> Path:
    """experiments/whitening/background.py build_pages() + build_targets() for one concept, "page" key only.

    "page_minus_wiki" needs bg_wiki_types.json, which this checkout lacks, so it is not written.
    """
    wdir = ROOT / "experiments" / "whitening"
    for p in (wdir, ROOT / "experiments" / "resample_sentences", ROOT / "experiments" / "multilingual_hp"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import importlib.util
    spec = importlib.util.spec_from_file_location("whitening_common", wdir / "common.py")
    wc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wc)
    from fetch_en_sentences import _ParagraphText, split_sentences

    page_out = wc.DATA / "pages" / f"{slug(concept)}.json"
    if page_out.exists():
        page = wc.read_json(page_out)
    else:
        raw = wc.read_json(page_source)
        assert raw["lang"] == "en" and raw["title"] == concept, (raw["lang"], raw["title"])
        p = _ParagraphText()
        p.feed(raw["html"])
        sents = split_sentences(p.paragraphs)
        page = {"concept": concept, "title": raw["title"], "revid": raw["revid"], "sentences": sents}
        wc.write_json(page_out, page)
        print(f"[page] {concept}: '{raw['title']}' rev {raw['revid']}: {len(sents)} sentences -> {page_out}")
    ex = wc.excluded_ids(MODEL_SLUG)
    tok = wc.tokenizer(MODEL_SLUG)
    ids = set()
    for seq in tok(page["sentences"], add_special_tokens=False)["input_ids"]:
        ids.update(seq)
    pg = sorted(int(i) for i in ids if i not in ex)
    out = WH_DIR / "targets" / f"{slug(concept)}.json"
    wc.write_json(out, {"page": pg}, indent=None)
    print(f"[target] {concept}: page = {len(pg)} ids -> {out} (page_minus_wiki not written: no bg_wiki_types.json)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--matrix", choices=["whitened", "plain"], default="whitened")
    ap.add_argument("--S", type=int, default=FULL_S)
    ap.add_argument("--sub-seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--build-target", type=Path, default=None,
                    help="raw page json (lang/title/revid/html) to build the whitening target file from first")
    a = ap.parse_args()
    if a.build_target is not None:
        build_target_page(a.concept, a.build_target)
    prepare(a.concept, a.matrix, a.S, a.sub_seed, force=a.force)


if __name__ == "__main__":
    main()
