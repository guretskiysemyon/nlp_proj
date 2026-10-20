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
  T2 context   recall of a concrete detail of the same sentence (chat generation, string match; valid when the
               references recall >= 0.9; amendment 4c); answer-NLL excess over the nonce as a secondary diagnostic;
               del_likeness = copy output identical to the deleted-word copy
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

from experiments.varwidth.collateral import ALL, EXPANSION, OUT, POC, seeds
from experiments.varwidth.common import slug

IOUT = OUT / "integrity"
SWEEP = [0.25, 1.0, 10.0, 100.0]
PILOT_CONFIGS = ["sub:ember@1.0", "sub:ember@100.0", "mean:ember@sel"]
SEL_VARIANTS = ["sub:ember", "sub:cinder", "sub:ember_judge", "sub:1a", "sub:1b", "pow2.0:cinder", "sub:cinder_rs", "sub:cinder_z", "sub:cinder_rs_z",
                "mean:ember", "mean:cinder", "noise:ember", "noise:cinder"]
MAX_WORDS = 60
MARGIN = 0.1     # nats on the answer tokens (amendment 4b)

COPY = ['Yesterday my cousin kept talking about {slot} while we cooked dinner.',
        'The note on the fridge just said {slot} and nothing more.']
# T2 (amendments 4b, 4c): recall of a concrete, concept-free detail of the same sentence. Primary: chat generation,
# string match ("recall"); secondary: NLL of the answer after a raw "Answer:" (the pilot found it poorly calibrated).
CTX = [("Yesterday my cousin kept talking about {slot} while we cooked pasta.", "What did we cook?", "pasta", ("pasta",)),
       ("On Tuesday two people on the train mentioned {slot}, and then the train stopped in Lyon.",
        "In which city did the train stop?", "Lyon", ("lyon",)),
       ("She wrote {slot} on a sticky note and stuck it on the green fridge.", "What color was the fridge?", "green",
        ("green",)),
       ("The word {slot} came up in the meeting that started at nine o'clock.", "At what time did the meeting start?",
        "nine", ("nine", "9")),
       ("My brother Daniel searched for {slot} online before walking the dog.", "What is my brother's name?", "Daniel",
        ("daniel",))]
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
        recs[wd]["nll"] = [cont_nll(model, tok, f"{fill(sn, slots[wd])} Question: {q} Answer:", " " + a)
                           for sn, q, a, _ in CTX]
    rec_p = [(wd, i, f"{fill(sn, slots[wd])} Question: {q} Answer with one word.")
             for wd in words for i, (sn, q, _, _) in enumerate(CTX)]
    outs = w.generate_multiple([p for *_, p in rec_p], max_new_tokens=8, do_sample=False, batch_size=16)
    for (wd, i, _), o in zip(rec_p, outs):
        recs[wd].setdefault("recall", []).append(any(k in o.lower() for k in CTX[i][3]))
    items = []
    for k, wd in enumerate(words):
        for j in range(2):
            it = mmlu[(2 * k + j) % len(mmlu)]
            items.append((wd, replace(it, question=CTX_MC.format(slot=slots[wd], q=it.question).replace("  ", " "))))
    _, _, mrecs = evaluate_mmlu_generation(w, [it for _, it in items])
    for (wd, _), r in zip(items, mrecs):
        recs[wd].setdefault("ctxmc", []).append(bool(r["is_correct_generation"]))
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
def static(W, scale, ids, new_rows, vocab_stats, chunk=32768, exclude=None):
    """Norm z-score of the edited rows (TL-scaled space) and cosine to the nearest other real token row."""
    mu, sd = vocab_stats
    z = ((new_rows.norm(dim=1) - mu) / sd).cpu().numpy()
    q = torch.nn.functional.normalize(new_rows.to(W.device).float(), dim=1)
    best = torch.full((len(ids),), -2.0, device=W.device)
    ex = torch.tensor(sorted(set(exclude or ids)), device=W.device)
    for i in range(0, W.shape[0], chunk):
        sims = q @ torch.nn.functional.normalize(W[i:i + chunk].float(), dim=1).T
        m = ex[(ex >= i) & (ex < i + chunk)] - i
        if len(m):
            sims[:, m] = -2.0                         # no edited row counts as a "real token" neighbour
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
    if concept in EXPANSION:                      # amendment 8: integrity at the selected point only (the endpoint;
        from experiments.varwidth.collateral import EXP_VARIANTS   # @0.5 dropped for time, Oct 20)
        names = [f"{v}@sel" for v in EXP_VARIANTS]
    else:
        names = None
    names = names or (PILOT_CONFIGS if pilot else [f"{v}@sel" for v in SEL_VARIANTS] + \
        [f"{v}@{r}" for v in ("sub:ember", "sub:cinder", "sub:cinder_rs", "sub:cinder_z", "sub:cinder_rs_z")
         for r in SWEEP] + \
        [f"{v}@paper" for v in ("sub:ember", "sub:cinder", "sub:cinder_rs", "sub:cinder_z", "sub:cinder_rs_z", "sub:1b")])
    for nm in names:
        v, p = nm.split("@")
        if (concept, seed, v) not in G:
            continue
        if p == "paper":                              # amendment 7: the paper-style rule's point
            from experiments.varwidth.erase_v2 import select_paper
            b = select_paper(G[(concept, seed, v)])
            if b is None:
                continue
            out.append((nm, v, b["param"]))
        elif p == "sel":
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
        have = json.loads(p.read_text()) if p.exists() else {}
        todo = [k for k in words if k not in have]          # top-up when a new variant brings new words
        if todo:
            have.update(measure(model, w, tok, todo, {k: slots[k] for k in todo}, mmlu))
            p.write_text(json.dumps(have, ensure_ascii=False))
            print(concept, seed, "ref", name, f"+{len(todo)} words", flush=True)
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
        st = static(W, ed.scale, ids, new.float(), vocab_stats, exclude=all_ids)
        p.write_text(json.dumps({"config": name, "variant": v, "param": par, "r": rr, "n_edit": len(ids),
                                 "words": mine, "static": st, "recs": recs}, ensure_ascii=False))
        print(concept, seed, name, f"r={rr:.3g}", summarize(d, name)["summary"], flush=True)
    ed.restore()


# ---------------------------------------------------------------- summaries
def summarize(d, name):
    c = json.loads((d / f"{name.replace(':', '_')}.json").read_text())
    R = {k: json.loads((d / f"ref_{k}.json").read_text()) for k in ("orig", "nonce", "del")}
    ws = c["words"]
    exc, bud, dele = [], [], []
    for wd in ws:
        for k in range(len(CTX)):
            exc.append(c["recs"][wd]["nll"][k] - R["nonce"][wd]["nll"][k])        # secondary: answer-NLL excess vs nonce
            bud.append(R["nonce"][wd]["nll"][k] - R["orig"][wd]["nll"][k])
            dele.append(R["del"][wd]["nll"][k] - R["orig"][wd]["nll"][k])
    ok = [x for wd in ws for x in c["recs"][wd]["recall"]]

    def m(key, src=None):
        src = src or c["recs"]
        v = [x for wd in ws for x in (src[wd][key] if isinstance(src[wd][key], list) else [src[wd][key]])]
        return float(np.mean(v)) if v else float("nan")

    s = {"n_words": len(ws), "copy": m("copy"), "copy_orig": m("copy", R["orig"]), "copy_nonce": m("copy", R["nonce"]),
         "echo": m("echo"), "echo_orig": m("echo", R["orig"]), "echo_nonce": m("echo", R["nonce"]),
         "ctx_pass": float(np.mean(ok)) if ok else float("nan"), "recall_orig": m("recall", R["orig"]),
         "recall_nonce": m("recall", R["nonce"]), "recall_del": m("recall", R["del"]),
         "ctx_excess": float(np.mean(exc)) if exc else float("nan"),
         "nonce_budget": float(np.mean(bud)) if bud else float("nan"),
         "del_excess": float(np.mean(dele)) if dele else float("nan"),
         "ctxmc": m("ctxmc"), "ctxmc_orig": m("ctxmc", R["orig"]), "ctxmc_nonce": m("ctxmc", R["nonce"]),
         **{f"static_{k}": v for k, v in c["static"].items()}}
    norm = lambda x: re.sub(r"\s+", " ", x).strip().strip('"').strip()  # noqa: E731
    dl = [norm(c["recs"][wd]["copy_out"][i]) == norm(R["del"][wd]["copy_out"][i]) for wd in ws for i in range(len(COPY))]
    s["del_likeness"] = float(np.mean(dl)) if dl else float("nan")   # copy output == the deleted-word reference
    return {"config": name, "variant": c["variant"], "param": c["param"], "r": c["r"], "summary":
            {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}}


# ---------------------------------------------------------------- judge export / import, report
JUDGE_RUNS = [("Harry Potter", 42), ("Ancient Rome", 44), ("Valentine's Day", 42)] + \
    [(c, 42) for c in EXPANSION]                  # amendment 8: seed 42 of every expansion concept
JDIR = IOUT / "judge"
BATCH = 150


def judge_export():
    """Blind T3 batches: every (config or reference, word) answer of the judge runs, shuffled, ids only; the key
    (id -> run, config, word) is kept apart. 10% of the items appear a second time in another batch (agreement)."""
    items = []
    for c, s in JUDGE_RUNS:
        d = IOUT / slug(c) / f"s{s}"
        files = sorted(d.glob("*.json"))
        for f in files:
            if f.name == "meta.json":
                continue
            j = json.loads(f.read_text())
            recs = j["recs"] if "recs" in j else j
            name = j.get("config", f.stem)
            for k, r in recs.items():
                if "ask_out" in r:
                    shown = r["slot"].strip('"') if r["slot"] else k.strip('"')
                    items.append({"concept": c, "seed": s, "config": name, "key": k, "word": shown,
                                  "prompt": fill(ASK, r["slot"]), "answer": r["ask_out"]})
    rng = random.Random(0)
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = f"j{i:05d}"
    dup = rng.sample(items, len(items) // 10)
    JDIR.mkdir(parents=True, exist_ok=True)
    (JDIR / "key.json").write_text(json.dumps({it["id"]: {k: it[k] for k in ("concept", "seed", "config", "key")}
                                               for it in items}, ensure_ascii=False))
    blind = [{"id": it["id"], "concept": it["concept"], "word": it["word"], "prompt": it["prompt"],
              "answer": it["answer"]} for it in items]
    second = [{**b, "id": b["id"] + "d"} for b in rng.sample(blind, len(dup))]
    batches = [blind[i:i + BATCH] for i in range(0, len(blind), BATCH)]
    for j, b2 in enumerate(second):                       # duplicates go into a different batch than the original
        batches[(j * 7 + 3) % len(batches)].append(b2)
    for j, b in enumerate(batches):
        rng.shuffle(b)
        (JDIR / f"batch_{j:02d}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1))
    print(f"{len(items)} items (+{len(second)} duplicates) -> {len(batches)} batches in {JDIR}")


def judge_export_new():
    """Incremental export (amendment 5): answers not yet in key.json get new ids and new batches; ids and batches
    already labelled are unchanged. 10% of the new items are duplicated into a different new batch."""
    key = json.loads((JDIR / "key.json").read_text())
    seen = {(k["concept"], k["seed"], k["config"], k["key"]) for k in key.values()}
    items = []
    for c, s in JUDGE_RUNS:
        d = IOUT / slug(c) / f"s{s}"
        for f in sorted(d.glob("*.json")):
            if f.name == "meta.json" or f.name.startswith("ref_"):
                continue
            j = json.loads(f.read_text())
            for k, r in j["recs"].items():
                if "ask_out" in r and (c, s, j["config"], k) not in seen:
                    items.append({"concept": c, "seed": s, "config": j["config"], "key": k,
                                  "word": r["slot"].strip('"'), "prompt": fill(ASK, r["slot"]), "answer": r["ask_out"]})
    if not items:
        print("nothing new")
        return
    rng = random.Random(1)
    rng.shuffle(items)
    start = max(int(i[1:].rstrip("d")) for i in key) + 1
    for n, it in enumerate(items):
        it["id"] = f"j{start + n:05d}"
        key[it["id"]] = {k: it[k] for k in ("concept", "seed", "config", "key")}
    (JDIR / "key.json").write_text(json.dumps(key, ensure_ascii=False))
    blind = [{"id": it["id"], "concept": it["concept"], "word": it["word"], "prompt": it["prompt"],
              "answer": it["answer"]} for it in items]
    nb = max(1, -(-len(blind) // BATCH))
    batches = [blind[i::nb] for i in range(nb)]
    first = max(int(f.stem.split("_")[1]) for f in JDIR.glob("batch_*.json")) + 1
    for j, b in enumerate(batches):
        others = [x for jj, bb in enumerate(batches) if jj != j for x in bb]
        if others:
            b += [{**x, "id": x["id"] + "d"} for x in rng.sample(others, min(len(others), len(b) // 10))]
        rng.shuffle(b)
        (JDIR / f"batch_{first + j:02d}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1))
    print(f"{len(items)} new items -> batches {first}..{first + nb - 1}")


def labels():
    """{id: label} merged over labels_*.json (first value wins; duplicates kept under their own ids)."""
    out = {}
    for f in sorted(JDIR.glob("labels_*.json")):
        for k, v in json.loads(f.read_text()).items():
            out.setdefault(k, v["label"] if isinstance(v, dict) else v)
    return out


GRACEFUL, LEAK, DESTR = {"IGNORANT", "KNOWS_OTHER"}, {"KNOWS_CONCEPT"}, {"BLIND", "INCOHERENT"}


def token_class(concept):
    """{slot-key word: 'multi' | 'exclusive'} from the chosen question-set version (via the token strings)."""
    sel = json.loads((OUT / "drafts" / slug(concept) / "selection.json").read_text())["chosen"]
    d = json.loads((OUT / "drafts" / slug(concept) / f"{sel}.json").read_text())
    return {e["token"]: e["class"] for e in d["tokens"]}


def report():
    L = ["# E2 token integrity: report", "",
         "Pre-registered in PREREG amendment 4. References on the unedited model: orig (real word), nonce (matched "
         "non-word: the target), del (word removed). ctx_pass = share of (word, carrier) whose continuation-NLL excess "
         "= recall of a same-sentence detail (chat generation; valid only when orig, nonce and del recall ≥ 0.9); ctx_excess = answer-NLL excess over the nonce (secondary); del_likeness = share of copy outputs identical to the copy output with the word deleted.", ""]
    lab, key = labels(), (json.loads((JDIR / "key.json").read_text()) if (JDIR / "key.json").exists() else {})
    rates = {}
    for i, k in key.items():
        if i in lab:
            rates.setdefault((k["concept"], k["seed"], k["config"]), Counter())[lab[i]] += 1
    if lab:
        batch_of = {it["id"]: int(f.stem.split("_")[1]) for f in JDIR.glob("batch_*.json")
                    for it in json.loads(f.read_text())}
        agent = lambda i: batch_of.get(i, -1) // 3      # noqa: E731  (3 consecutive batches per labelling agent;
        #                                                 new batches from judge_export_new are labelled by other agents)
        dups = [(i, i + "d") for i in key if i + "d" in lab and i in lab and agent(i) != agent(i + "d")]
        agree = np.mean([lab[a] == lab[b] for a, b in dups]) if dups else float("nan")
        destr = lambda x: x in DESTR                     # noqa: E731
        agree_d = np.mean([destr(lab[a]) == destr(lab[b]) for a, b in dups]) if dups else float("nan")
        L += [f"Agent labels: {sum(1 for i in key if i in lab)}/{len(key)} items labelled. Agreement on {len(dups)} "
              f"duplicates labelled by different agents: exact label {agree:.2f}; destruction yes/no {agree_d:.2f}.", ""]
    cols = ["r", "n_words", "copy", "ctx_pass", "del_likeness", "ctx_excess", "ctxmc", "echo", "static_z_gt3",
            "static_nn_cos_median"]
    L += ["| Concept | Seed | Config | " + " | ".join(cols) + " | graceful | leak | destruction |",
          "|" + "---|" * (len(cols) + 6)]
    agg = {}
    for c, s in [(c, s) for c in ALL for s in seeds(c)]:
        d = IOUT / slug(c) / f"s{s}"
        if not (d / "ref_orig.json").exists():
            continue
        R = {k: json.loads((d / f"ref_{k}.json").read_text()) for k in ("orig", "nonce", "del")}
        for k in ("orig", "nonce", "del"):
            ws = list(R[k])
            cp = np.mean([x for w in ws for x in R[k][w]["copy"]])
            ec = np.mean([R[k][w]["echo"] for w in ws])
            mc = np.mean([x for w in ws for x in R[k][w]["ctxmc"]])
            rc = rates.get((c, s, f"ref_{k}"), Counter())
            n = sum(rc.values())
            gl = (f"{sum(rc[x] for x in GRACEFUL) / n:.2f} | {sum(rc[x] for x in LEAK) / n:.2f} | "
                  f"{sum(rc[x] for x in DESTR) / n:.2f}") if n else "— | — | —"
            L.append(f"| {c} | {s} | ref:{k} | — | {len(ws)} | {cp:.2f} | — | — | — | {mc:.2f} | {ec:.2f} | — | — | {gl} |")
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("ref_") or f.name == "meta.json":
                continue
            sm = summarize(d, json.loads(f.read_text())["config"])
            x = {**sm["summary"], "r": sm["r"]}
            rc = rates.get((c, s, sm["config"]), Counter())
            n = sum(rc.values())
            gl = (f"{sum(rc[x2] for x2 in GRACEFUL) / n:.2f} | {sum(rc[x2] for x2 in LEAK) / n:.2f} | "
                  f"{sum(rc[x2] for x2 in DESTR) / n:.2f}") if n else "— | — | —"
            L.append(f"| {c} | {s} | {sm['config']} | " + " | ".join(
                f"{x[k]:.3g}" if isinstance(x.get(k), (int, float)) and x.get(k) is not None else "—" for k in cols)
                + f" | {gl} |")
            agg.setdefault(sm["config"], []).append(x)
    L += ["", "## Means over concept-seeds", "", "| Config | runs | " + " | ".join(cols[1:]) + " |", "|" + "---|" * (len(cols) + 1)]
    for cfg, xs in sorted(agg.items()):
        L.append(f"| {cfg} | {len(xs)} | " + " | ".join(
            f"{np.nanmean([x[k] for x in xs if isinstance(x.get(k), (int, float))]):.3g}" for k in cols[1:]) + " |")
    if lab:
        C = {}
        for i, k in key.items():
            if i in lab:
                C.setdefault(k["config"], Counter())[lab[i]] += 1
        L += ["", "## T3 agent labels by config (judge runs: Harry Potter 42, Ancient Rome 44, Valentine's Day 42)", "",
              "graceful = IGNORANT + KNOWS_OTHER; leak = KNOWS_CONCEPT; destruction = BLIND + INCOHERENT.", "",
              "| Config | n | graceful | leak | confab | destruction | KC | KO | IGN | CONFAB | BLIND | INCOH |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        def order(cfg):
            return (0 if cfg.startswith("ref_") else 1, cfg)
        for cfg in sorted(C, key=order):
            x = C[cfg]
            n = sum(x.values())
            L.append(f"| {cfg} | {n} | {sum(x[l] for l in GRACEFUL) / n:.2f} | {sum(x[l] for l in LEAK) / n:.2f} | "
                     f"{x['CONFAB'] / n:.2f} | {sum(x[l] for l in DESTR) / n:.2f} | {x['KNOWS_CONCEPT']} | "
                     f"{x['KNOWS_OTHER']} | {x['IGNORANT']} | {x['CONFAB']} | {x['BLIND']} | {x['INCOHERENT']} |")
    # R11: exclusive vs multi-meaning words (class of the word's tokens in the chosen question-set version; a word
    # counts as multi if any of its tokens is multi)
    split = {}
    for c, s in [(c, s) for c in ALL for s in seeds(c)]:
        d = IOUT / slug(c) / f"s{s}"
        if not (d / "meta.json").exists():
            continue
        cls_tok = token_class(c)
        meta = json.loads((d / "meta.json").read_text())["words"]
        wcls = {k: ("multi" if any(cls_tok.get(t) == "multi" for t in v["tokens"]) else
                    "exclusive" if any(cls_tok.get(t) == "exclusive" for t in v["tokens"]) else "unclassified")
                for k, v in meta.items()}
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("ref_") or f.name == "meta.json":
                continue
            j = json.loads(f.read_text())
            for k, r in j["recs"].items():
                g = split.setdefault((j["config"], wcls.get(k, "unclassified")), {"copy": [], "echo": [], "destr": []})
                g["copy"] += r["copy"]
                g["echo"].append(r["echo"])
        if lab:
            for i, kk in key.items():
                if i in lab and kk["concept"] == c and kk["seed"] == s and not kk["config"].startswith("ref_"):
                    g = split.setdefault((kk["config"], wcls.get(kk["key"], "unclassified")), {"copy": [], "echo": [], "destr": []})
                    g["destr"].append(lab[i] in DESTR)
    L += ["", "## R11: exclusive vs multi-meaning words (means over all runs; destruction from the judge runs)", "",
          "| Config | class | n words | copy | echo | T3 destruction (n) |", "|---|---|---|---|---|---|"]
    for (cfg, cl), g in sorted(split.items()):
        if cl == "unclassified" or not g["echo"]:
            continue
        dn = len(g["destr"])
        L.append(f"| {cfg} | {cl} | {len(g['echo'])} | {np.mean(g['copy']):.2f} | {np.mean(g['echo']):.2f} | "
                 + (f"{np.mean(g['destr']):.2f} ({dn})" if dn else "—") + " |")
    (IOUT / "report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {IOUT / 'report.md'}")


def main():
    from dotenv import load_dotenv

    from experiments.varwidth.common import ROOT
    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "judge-export", "judge-export-new", "report"))
    ap.add_argument("--concept")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.what == "run":
        run(a.concept, a.seed, a.pilot)
    elif a.what == "judge-export":
        judge_export()
    elif a.what == "judge-export-new":
        judge_export_new()
    else:
        report()


if __name__ == "__main__":
    main()
