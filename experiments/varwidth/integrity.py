#!/usr/bin/env python
"""Experiment E2, token integrity (PREREG amendment 4): is an edited token still a readable, copyable, inert symbol,
or is it destroyed? Applies to every edited token, concept-exclusive ones included.

    python -m experiments.varwidth.integrity run --concept C --seed S [--pilot]
    python -m experiments.varwidth.integrity judge-export      # blind batches for the agent judges
    python -m experiments.varwidth.integrity report            # -> results/varwidth/layers/collateral/integrity/report.md

Reference behaviours, all on the unedited model: the real word ("orig", knows the concept), a matched nonce word
("nonce", the target: an unknown word), and the word deleted ("del", what "just removing the token" gives).
Tests per edited word w (its most frequent source word in the concept sentences, placed in concept-neutral carriers):
  T1 copy      "Repeat exactly: <carrier with w>" -> output contains w (judge-free)
  T2 context   NLL of a fixed, concept-free continuation after a carrier with w; pass when the excess over orig is
               within the nonce budget (+0.05 nats); also deletion-likeness = excess / deletion excess (judge-free)
  T2b ctx-MC   an unrelated MMLU question after a turn that mentions w; accuracy vs orig and nonce (judge-free)
  T3 ignorance "I keep hearing people mention <w>. What is it?" -> echo of w (judge-free) and a blind agent label
               (KNOWS_CONCEPT / KNOWS_OTHER / IGNORANT / CONFAB / BLIND / INCOHERENT)
  T4 static    edited rows: norm z-score vs the vocabulary, cosine to the nearest real token row (white-box
               fingerprint; no generation)
Configs: each E1 variant at its E1-selected point (recomputed from the frozen grids with the same rule), EMBER and
CINDER at r in SWEEP, and the Mean / Noise baselines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import replace

import numpy as np
import torch

from experiments.varwidth.collateral import OUT, POC, seeds
from experiments.varwidth.common import slug

IOUT = OUT / "integrity"
SWEEP = [0.25, 1.0, 10.0, 100.0]
PILOT_CONFIGS = ["sub:ember@1.0", "sub:ember@100.0", "mean:ember@sel"]
SEL_VARIANTS = ["sub:ember", "sub:cinder", "sub:ember_judge", "sub:1a", "sub:1b", "pow2.0:cinder",
                "mean:ember", "mean:cinder", "noise:ember", "noise:cinder"]
MAX_WORDS = 60
MARGIN = 0.05

COPY = ['Yesterday my cousin kept talking about {slot} while we cooked dinner.',
        'The note on the fridge just said {slot} and nothing more.']
CTX = [("Yesterday my cousin kept talking about {slot}",
        " while we cooked dinner, and afterwards we watched the evening news and went to bed early."),
       ("On the train I overheard two people mention {slot}",
        ". Then the conductor announced that the next station would be delayed by ten minutes because of maintenance work."),
       ("She wrote {slot} on a sticky note",
        " and left it on the kitchen table next to a bowl of apples and a half-empty cup of coffee."),
       ("The word {slot} came up in the meeting",
        ", but most of the discussion was about the budget for next year and the new office schedule."),
       ("I searched for {slot} online",
        " for a few minutes before giving up, closing my laptop, and going outside for a walk in the park.")]
ASK = "I keep hearing people mention {slot}. What is it?"
CTX_MC = "I just read an article that mentioned {slot}. Anyway, an unrelated question.\n\n{q}"


# ---------------------------------------------------------------- words, slots, nonces
def source_words(concept, tok, ids):
    """{token id: (word, slot)}: the token's most frequent source word in the concept sentences and the slot string
    (w or "w") whose tokenization after a space contains the token id; tokens with no such slot are left out."""
    from ember.judge_words import token_word_counts
    from experiments.varwidth.prep import build_dataset
    sents = [s for s, lbl in build_dataset(concept).data if lbl == concept]
    counts = token_word_counts(tok, sents)
    out = {}
    for t in ids:
        c = counts.get(int(t))
        if not c:
            continue
        merged = Counter()
        for (_, word), k in c.items():
            merged[word] += k
        for word, _ in merged.most_common(3):
            word = word.strip()
            if not word or len(word) > 40:
                continue
            for slot in (word, f'"{word}"'):
                if int(t) in tok(f"about {slot}", add_special_tokens=False)["input_ids"]:
                    out[int(t)] = (word, slot)
                    break
            if int(t) in out:
                break
    return out


SYL_C, SYL_V = "bdfgklmnprstvz", "aeiou"


def nonce(word, tok, banned, salt=0):
    """A pronounceable non-word of similar length and the same capitalization, whose tokens avoid `banned` ids."""
    h = int(hashlib.sha1(f"{word}|{salt}".encode()).hexdigest()[:8], 16)
    rng = random.Random(h)
    for _ in range(200):
        n = max(4, len(re.sub(r"\W", "", word)) + rng.choice([-1, 0, 1]))
        s = "".join(rng.choice(SYL_C) + rng.choice(SYL_V) for _ in range((n + 1) // 2))[:n]
        if rng.random() < 0.5:
            s += rng.choice(SYL_C)
        s = s.capitalize() if word[:1].isupper() else s
        ids = tok(f"about {s}", add_special_tokens=False)["input_ids"]
        if not banned.intersection(ids):
            return s
    raise RuntimeError(f"no nonce for {word}")


def fill(t, slot):
    return re.sub(r"\s+", " ", t.replace("{slot}", slot)).replace(" .", ".").replace(" ,", ",").strip()


# ---------------------------------------------------------------- measurements
@torch.no_grad()
def cont_nll(model, tok, prefix, cont):
    ids_p = tok(prefix, add_special_tokens=True)["input_ids"]
    ids_c = tok(cont, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids_p + ids_c], device=model.device)
    logits = model(x).logits[0, len(ids_p) - 1:-1].float()
    return float(torch.nn.functional.cross_entropy(logits, x[0, len(ids_p):]).item())


def measure(model, w, tok, words, slots, mmlu, with_ask=True):
    """Per-word tests for one model state; `slots` maps word -> slot string ('' = deletion)."""
    from ember.evals.mmlu import evaluate_mmlu_generation
    recs = {wd: {"slot": slots[wd]} for wd in words}
    copy_p = [(wd, i, f'Repeat the following sentence exactly, and nothing else: "{fill(c, slots[wd])}"')
              for wd in words for i, c in enumerate(COPY)]
    outs = w.generate_multiple([p for *_, p in copy_p], max_new_tokens=40, do_sample=False, batch_size=16)
    for (wd, i, _), o in zip(copy_p, outs):
        target = slots[wd].strip('"')
        recs[wd].setdefault("copy", []).append(bool(target) and target in o)
        recs[wd].setdefault("copy_out", []).append(o)
    for wd in words:
        recs[wd]["nll"] = [cont_nll(model, tok, fill(p, slots[wd]), c) for p, c in CTX]
    items = []
    for k, wd in enumerate(words):
        for j in range(2):
            it = mmlu[(2 * k + j) % len(mmlu)]
            items.append((wd, replace(it, question=CTX_MC.format(slot=slots[wd], q=it.question).replace("  ", " "))))
    _, _, mrecs = evaluate_mmlu_generation(w, [it for _, it in items])
    for (wd, _), r in zip(items, mrecs):
        recs[wd].setdefault("ctxmc", []).append(bool(r["is_correct"]))
    if with_ask:
        outs = w.generate_multiple([fill(ASK, slots[wd]) for wd in words], max_new_tokens=80, do_sample=False,
                                   batch_size=16)
        for wd, o in zip(words, outs):
            target = slots[wd].strip('"')
            recs[wd]["ask_out"] = o
            recs[wd]["echo"] = bool(target) and target.lower() in o.lower()
    return recs


@torch.no_grad()
def vocab_norm_stats(W, scale, chunk=32768):
    n = torch.cat([(W[i:i + chunk].float() * scale).norm(dim=1) for i in range(0, W.shape[0], chunk)])
    return float(n.mean()), float(n.std())


@torch.no_grad()
def static(W, scale, ids, new_rows, vocab_stats, chunk=32768):
    """Norm z-score of the edited rows (TL-scaled space) and cosine to the nearest other real token row."""
    mu, sd = vocab_stats
    z = ((new_rows.norm(dim=1) - mu) / sd).cpu().numpy()
    q = torch.nn.functional.normalize(new_rows.to(W.device).float(), dim=1)
    best = torch.full((len(ids),), -2.0, device=W.device)
    idt = torch.tensor(ids, device=W.device)
    for i in range(0, W.shape[0], chunk):
        sims = q @ torch.nn.functional.normalize(W[i:i + chunk].float(), dim=1).T
        own = (idt >= i) & (idt < i + chunk)
        if own.any():
            sims[own.nonzero().squeeze(1), idt[own] - i] = -2.0
        best = torch.maximum(best, sims.max(1).values)
    return {"z_gt3": float((z > 3).mean()), "z_median": float(np.median(z)),
            "nn_cos_median": float(best.median().item())}


# ---------------------------------------------------------------- configs
def selected_param(G, key, alpha):
    from experiments.varwidth.collateral_report import select
    rows = G.get(key)
    if not rows:
        return None
    if key[2].startswith("proj:"):
        rows = [x for x in rows if x["param"] == float(alpha[key[2].split(":")[1]])]
    b = select(rows)
    return None if b is None else b["param"]


def configs(concept, seed, pilot=False):
    """[(name, variant, param)]."""
    from experiments.varwidth.collateral_report import grids
    G = grids()
    sc = OUT / "shape_choice.json"
    alpha = json.loads(sc.read_text())["best_alpha"] if sc.exists() else {"ember": 1.0, "cinder": 1.0}
    out = []
    names = PILOT_CONFIGS if pilot else [f"{v}@sel" for v in SEL_VARIANTS] + \
        [f"{v}@{r}" for v in ("sub:ember", "sub:cinder") for r in SWEEP]
    for nm in names:
        v, p = nm.split("@")
        if (concept, seed, v) not in G:
            continue
        if p == "sel":
            par = selected_param(G, (concept, seed, v), alpha)
            if par is None and not v.startswith("mean:"):
                continue
            out.append((nm, v, par))
        else:
            out.append((nm, v, float(p)))
    return out


def run(concept, seed, pilot=False):
    from ember.evals.mmlu import load_mmlu_items
    from ember.local_datasets import load_mmlu_indices
    from experiments.varwidth.erase_v2 import setup
    d = IOUT / slug(concept) / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    base, sets, model, w, ed, E = setup(concept, seed)
    tok = w.tokenizer
    mmlu = load_mmlu_items(load_mmlu_indices("train"))
    cfgs = configs(concept, seed, pilot)
    all_ids = sorted({t for _, v, _ in cfgs for t in E.comp(v)[0]})
    sw = source_words(concept, tok, all_ids)
    banned = set(all_ids)
    # one entry per slot form (" Harry" -> Harry, "Harry" -> "Harry" are different tokens): key = slot string
    words = sorted({sw[t][1] for t in sw})
    rng = random.Random(seed)
    non = {k: nonce(k.strip('"'), tok, banned) for k in words}
    meta = {"concept": concept, "seed": seed, "n_tokens": len(all_ids), "n_words": len(words),
            "no_slot_tokens": [tok.decode([t]) for t in all_ids if t not in sw],
            "words": {k: {"nonce": non[k], "tokens": [tok.decode([t]) for t in sw if sw[t][1] == k]} for k in words}}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    ed.restore()
    W = model.get_input_embeddings().weight
    vocab_stats = vocab_norm_stats(W, ed.scale)
    refs = {"orig": {k: k for k in words},
            "nonce": {k: (f'"{non[k]}"' if k.startswith('"') else non[k]) for k in words},
            "del": {k: "" for k in words}}
    for name, slots in refs.items():
        p = d / f"ref_{name}.json"
        if not p.exists():
            p.write_text(json.dumps(measure(model, w, tok, words, slots, mmlu), ensure_ascii=False))
            print(concept, seed, "ref", name, flush=True)
    for name, v, par in cfgs:
        p = d / f"{name.replace(':', '_')}.json"
        if p.exists():
            continue
        ids, new, rr = E.rows(v, par)
        ed.set_rows(ids, new)
        mine = sorted({sw[t][1] for t in ids if t in sw})
        if len(mine) > MAX_WORDS:
            mine = sorted(rng.sample(mine, MAX_WORDS))
        recs = measure(model, w, tok, mine, {k: k for k in mine}, mmlu)
        st = static(W, ed.scale, ids, new.float(), vocab_stats)
        p.write_text(json.dumps({"config": name, "variant": v, "param": par, "r": rr, "n_edit": len(ids),
                                 "words": mine, "static": st, "recs": recs}, ensure_ascii=False))
        print(concept, seed, name, f"r={rr:.3g}", summarize(d, name)["summary"], flush=True)
    ed.restore()


# ---------------------------------------------------------------- summaries
def summarize(d, name):
    c = json.loads((d / f"{name.replace(':', '_')}.json").read_text())
    R = {k: json.loads((d / f"ref_{k}.json").read_text()) for k in ("orig", "nonce", "del")}
    ws = c["words"]
    exc, bud, dele, ok = [], [], [], []
    for wd in ws:
        for k in range(len(CTX)):
            e = c["recs"][wd]["nll"][k] - R["orig"][wd]["nll"][k]
            b = R["nonce"][wd]["nll"][k] - R["orig"][wd]["nll"][k]
            dl = R["del"][wd]["nll"][k] - R["orig"][wd]["nll"][k]
            exc.append(e)
            bud.append(b)
            dele.append(dl)
            ok.append(e <= max(b, 0.0) + MARGIN)

    def m(key, src=None):
        src = src or c["recs"]
        v = [x for wd in ws for x in (src[wd][key] if isinstance(src[wd][key], list) else [src[wd][key]])]
        return float(np.mean(v)) if v else float("nan")

    s = {"n_words": len(ws), "copy": m("copy"), "copy_orig": m("copy", R["orig"]), "copy_nonce": m("copy", R["nonce"]),
         "echo": m("echo"), "echo_orig": m("echo", R["orig"]), "echo_nonce": m("echo", R["nonce"]),
         "ctx_pass": float(np.mean(ok)) if ok else float("nan"), "ctx_excess": float(np.mean(exc)) if exc else float("nan"),
         "nonce_budget": float(np.mean(bud)) if bud else float("nan"),
         "del_excess": float(np.mean(dele)) if dele else float("nan"),
         "ctxmc": m("ctxmc"), "ctxmc_orig": m("ctxmc", R["orig"]), "ctxmc_nonce": m("ctxmc", R["nonce"]),
         **{f"static_{k}": v for k, v in c["static"].items()}}
    s["del_likeness"] = s["ctx_excess"] / s["del_excess"] if s["del_excess"] and s["del_excess"] > 0 else float("nan")
    return {"config": name, "variant": c["variant"], "param": c["param"], "r": c["r"], "summary":
            {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}}


def main():
    from dotenv import load_dotenv

    from experiments.varwidth.common import ROOT
    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run",))
    ap.add_argument("--concept")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.what == "run":
        run(a.concept, a.seed, a.pilot)


if __name__ == "__main__":
    main()
