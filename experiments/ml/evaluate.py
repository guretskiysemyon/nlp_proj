#!/usr/bin/env python
"""Edit + evaluation stage (GPU): Llama-3.1-8B-Instruct in NF4 on one Titan Xp; embeddings stay fp16 and are edited.

    python -m experiments.ml.evaluate --concept "Harry Potter" --approach A1 A2 A3

Per approach: edits c_t from the fit + selection (A1 merged across languages, edit once, by page priority), then
the delta grid on the train split and the test split at the chosen delta (PLAN.md "Evaluation and delta selection").
Per language L: efficacy = 1 - qa_frac, specificity = HM(mmlu_frac, simdom_frac), fluency = min(1, ppl_base/ppl_edit)
on held-out native text, H_L = HM of the three; score = mean_L H_L, 0 if any language has mmlu_frac or fluency < 0.9.
Outputs run_dir/eval.json (grid rows saved after every delta, so a preempted job resumes); baselines in
results/ml/baselines/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import types

import torch

from experiments.ml.common import (DATA, DELTA_GRID, LANGS, MIN_FRAC, MODEL, RESULTS, run_dir, save_json, slug)

LETTERS = ("A", "B", "C", "D")
BATCH, MAX_NEW = 32, 32
BASE_DIR = RESULTS / "baselines"


# ---------------------------------------------------------------- model
def load_model():
    stub = types.ModuleType("bitsandbytes.nn.triton_based_modules")
    for n in ("StandardLinear", "SwitchBackLinear", "SwitchBackLinearGlobal", "SwitchBackLinearVectorwise"):
        setattr(stub, n, type(n, (), {}))
    sys.modules["bitsandbytes.nn.triton_based_modules"] = stub
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import os
    src = os.environ.get("LLAMA_LOCAL") or MODEL     # a local-disk copy of the snapshot loads far faster than NFS
    tok = AutoTokenizer.from_pretrained(src)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(src, quantization_config=q, device_map={"": 0}).eval()
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    return model, tok


class Editor:
    """Writes e_t - delta * c_t into the embedding rows; restore() puts the originals back."""

    def __init__(self, model):
        self.W = model.get_input_embeddings().weight
        self.ids, self.orig, self.C = None, None, None

    def set_edit(self, edit: dict):
        self.restore()
        self.ids = torch.tensor(sorted(edit), dtype=torch.long, device=self.W.device)
        self.C = torch.stack([torch.as_tensor(edit[t]) for t in sorted(edit)]).float().to(self.W.device) \
            if edit else None
        self.orig = self.W.data[self.ids].clone() if edit else None

    @torch.no_grad()
    def apply(self, delta: float):
        if self.C is None:
            return
        self.W.data[self.ids] = (self.orig.float() - delta * self.C).to(self.W.dtype)

    @torch.no_grad()
    def restore(self):
        if self.orig is not None:
            self.W.data[self.ids] = self.orig
        self.ids = self.orig = self.C = None


# ---------------------------------------------------------------- data
def _perm(concept, set_name, i, seed=42):
    key = f"{concept}|{set_name}|{i}|{seed}".encode()
    p = list(range(4))
    random.Random(int(hashlib.sha1(key).hexdigest()[:8], 16)).shuffle(p)
    return p


def mc_items(concept, lang, set_name):
    data = json.loads((DATA / "qa" / f"mc_questions_{lang}.json").read_text(encoding="utf-8"))
    out = []
    for i, o in enumerate(data[concept][set_name]):
        opts = [x.strip() for x in o["options"]]
        p = _perm(concept, set_name, i)
        by = {L: opts[j] for L, j in zip(LETTERS, p)}
        correct = next(L for L, j in zip(LETTERS, p) if j == opts.index(o["correct_answer"].strip()))
        out.append({"question": o["q"].strip(), "options": by, "correct": correct})
    return out


def mmlu_items(lang, split):
    rows = json.loads((DATA / "mmlu" / f"{lang}.json").read_text(encoding="utf-8"))
    rows = rows[split] if isinstance(rows, dict) else [r for r in rows if r.get("split", split) == split]
    return [{"question": r["question"], "options": r["options"], "correct": r["answer"]} for r in rows]


def heldout(lang):
    rows = json.loads((DATA / f"neutral_heldout_{lang}.json").read_text(encoding="utf-8"))
    return [r if isinstance(r, str) else r["text"] for r in rows]


# ---------------------------------------------------------------- scoring
def _prompt(tok, item, lang):
    from experiments.ml.prompts import build_mc_prompt
    msg = [{"role": "user", "content": build_mc_prompt(item["question"], item["options"], lang)}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def mc_acc(model, tok, items, lang):
    from judge_rerun.mc_fixed import parse_letter_fixed
    prompts = [_prompt(tok, it, lang) for it in items]
    right = valid = 0
    for s in range(0, len(prompts), BATCH):
        enc = tok(prompts[s:s + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
        for it, row in zip(items[s:s + BATCH], out[:, enc.input_ids.shape[1]:]):
            letter = parse_letter_fixed(tok.decode(row, skip_special_tokens=True))
            valid += letter in LETTERS
            right += letter == it["correct"]
    n = len(items)
    return {"acc": right / n, "invalid": 1 - valid / n, "n": n}


@torch.no_grad()
def ppl(model, tok, sents):
    nll = cnt = 0.0
    for s in range(0, len(sents), BATCH):
        enc = tok(sents[s:s + BATCH], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, :-1].float()
        tgt = enc.input_ids[:, 1:]
        m = enc.attention_mask[:, 1:].bool()
        lp = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        nll += float(lp[m].sum())
        cnt += float(m.sum())
    return math.exp(nll / cnt)


def mc_norm(acc, base):
    return min(1.0, max((acc - 0.25) / (base - 0.25), 0.0)) if base > 0.25 else 0.0


def hm(xs):
    return 0.0 if any(x <= 0 for x in xs) else len(xs) / sum(1 / x for x in xs)


# ---------------------------------------------------------------- baselines
def lang_baseline(model, tok, split):
    """MMLU and perplexity of the unedited model, per language (concept-independent)."""
    path = BASE_DIR / f"lang_{split}.json"
    b = json.loads(path.read_text()) if path.exists() else {}
    for L in LANGS:
        if L not in b:
            b[L] = {"mmlu": mc_acc(model, tok, mmlu_items(L, "val" if split == "train" else "test"), L)["acc"],
                    "ppl": ppl(model, tok, heldout(L))}
            save_json(path, b)
    return b


def concept_baseline(model, tok, concept, split):
    path = BASE_DIR / f"{slug(concept)}_{split}.json"
    b = json.loads(path.read_text()) if path.exists() else {}
    for L in LANGS:
        if L not in b:
            b[L] = {"qa": mc_acc(model, tok, mc_items(concept, L, f"QA_{split}"), L)["acc"],
                    "simdom": mc_acc(model, tok, mc_items(concept, L, f"SimdomQA_{split}"), L)["acc"]}
            save_json(path, b)
    return b


def score(model, tok, concept, split, base_c, base_l):
    per = {}
    for L in LANGS:
        qa = mc_acc(model, tok, mc_items(concept, L, f"QA_{split}"), L)
        sd = mc_acc(model, tok, mc_items(concept, L, f"SimdomQA_{split}"), L)
        mm = mc_acc(model, tok, mmlu_items(L, "val" if split == "train" else "test"), L)
        pp = ppl(model, tok, heldout(L))
        eff = 1 - mc_norm(qa["acc"], base_c[L]["qa"])
        mmf, sdf = mc_norm(mm["acc"], base_l[L]["mmlu"]), mc_norm(sd["acc"], base_c[L]["simdom"])
        flu = min(1.0, base_l[L]["ppl"] / pp)
        per[L] = {"qa": qa["acc"], "simdom": sd["acc"], "mmlu": mm["acc"], "ppl": pp, "invalid_qa": qa["invalid"],
                  "efficacy": eff, "mmlu_frac": mmf, "simdom_frac": sdf, "specificity": hm([mmf, sdf]),
                  "fluency": flu, "H": hm([eff, hm([mmf, sdf]), flu])}
    gated = any(per[L]["mmlu_frac"] < MIN_FRAC or per[L]["fluency"] < MIN_FRAC for L in LANGS)
    return {"per_lang": per, "gated": gated, "score": 0.0 if gated else sum(per[L]["H"] for L in LANGS) / len(LANGS)}


# ---------------------------------------------------------------- edits
def run_edit(fit_path, sel_path) -> dict:
    """{token id: c_t (d,) in the original embedding space} for the concept-only tokens touched by F_C."""
    fit = torch.load(fit_path, weights_only=False)
    F_C = json.loads(sel_path.read_text())["F_C"]
    if not F_C:
        return {}
    Y, Z = fit["Y"].float()[F_C], fit["Z"].float()[:, F_C]
    codes = torch.as_tensor(fit["codes"])
    cols = torch.nonzero((Y != 0).any(0) & (codes == 1)).flatten()
    C = (Z @ Y[:, cols]).T
    if fit.get("unmap") is not None:
        C = C @ fit["unmap"]
    return {int(fit["vstar_ids"][int(j)]): C[i] for i, j in enumerate(cols)}


def edits(concept, approach):
    if approach != "A1":
        d = run_dir(concept, approach)
        return run_edit(d / "fit.pt", d / "selection.json"), {}
    from experiments.ml.data.io import page_priority
    merged, src = {}, {}
    for L in page_priority(concept):           # highest priority first: a token keeps the first language's edit
        d = run_dir(concept, "A1", L)
        for t, c in run_edit(d / "fit.pt", d / "selection.json").items():
            if t not in merged:
                merged[t], src[t] = c, L
    return merged, src


# ---------------------------------------------------------------- main
def evaluate(model, tok, editor, concept, approach, force=False):
    out_path = run_dir(concept, approach) / "eval.json"
    res = json.loads(out_path.read_text()) if out_path.exists() and not force else {}
    if res.get("done"):
        return res
    edit, src = edits(concept, approach)
    res.update({"concept": concept, "approach": approach, "n_edited": len(edit),
                "edited_by_lang": {L: sum(v == L for v in src.values()) for L in LANGS} if src else None})
    grid = res.setdefault("grid", {})
    bc, bl = concept_baseline(model, tok, concept, "train"), lang_baseline(model, tok, "train")
    editor.set_edit(edit)
    for delta in (DELTA_GRID if edit else []):
        if str(delta) in grid:
            continue
        editor.apply(delta)
        grid[str(delta)] = score(model, tok, concept, "train", bc, bl)
        save_json(out_path, res)
        print(f"[{approach} {concept}] delta={delta} score={grid[str(delta)]['score']:.3f} "
              f"gated={grid[str(delta)]['gated']}", flush=True)
    best = max((float(d) for d in grid if grid[d]["score"] > 0), key=lambda d: (grid[str(d)]["score"], -d),
               default=0.0)
    res["best_delta"] = best
    editor.apply(best) if best else editor.restore()
    btc, btl = concept_baseline(model, tok, concept, "test"), lang_baseline(model, tok, "test")
    res["test"] = score(model, tok, concept, "test", btc, btl)
    res["done"] = True
    editor.restore()
    save_json(out_path, res)
    print(f"[{approach} {concept}] best delta={best} test score={res['test']['score']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, nargs="+")
    ap.add_argument("--approach", nargs="+", default=["A1", "A2", "A3"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    model, tok = load_model()
    editor = Editor(model)
    for c in a.concept:
        for ap_ in a.approach:
            evaluate(model, tok, editor, c, ap_, a.force)


if __name__ == "__main__":
    main()
