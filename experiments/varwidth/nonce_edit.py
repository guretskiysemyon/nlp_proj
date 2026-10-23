#!/usr/bin/env python
"""Nonce-matching edit (PREREG amendment 13): learn only the edited rows so that each concept token behaves like a fresh
nonce word (unknown but intact): it stays copyable, keeps its other meanings, and stops carrying the concept.

    python -m experiments.varwidth.nonce_edit run --concept C --seed S [--variants nm:cinder ...] [--pilot]
    python -m experiments.varwidth.nonce_edit e2 --concept C --seed S       # judge-free E2 + T3 answers at the point
    python -m experiments.varwidth.nonce_edit report

run: training data and unedited targets (cached per edit set), then for each variant the knob path w_f = 0.5 -> 2 -> 8
(each knob continues from the previous one), a validation row per knob (grid_nm.json, rows_nm.pt), the E3-style
selection and a test pass at the selected knob (result_nm.json).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.varwidth.collateral import EXPANSION, OUT, seeds
from experiments.varwidth.common import slug
from experiments.varwidth.erase import hm, mc_norm, save

NM = {"nm:cinder": ("cinder_z", "sub"), "nm:free": ("cinder_z", "free"), "nm:ember": ("ember", "sub"),
      "nm:efree": ("ember", "free"),          # nm:efree: amendment 13 addendum (token set without the subspace)
      "nm:qa": ("cinder_z", "freeqa")}        # nm:qa: addendum 2 (+ self-written concept questions, lower lr)
KNOBS = [0.05, 0.15, 0.5, 2.0]
STEPS = {0.05: 120, 0.15: 60, 0.5: 60, 2.0: 60}   # path-following steps per knob (pilot; amendment 13 addendum)
LR = {"sub": 0.05, "free": 0.01, "freeqa": 0.003}
QA_PROMPT = "Write 3 short quiz questions about {c} that use the word '{w}'. Write one question per line, nothing else."
BATCH, TOPK, WIN, L2 = 16, 64, 16, 1e-3
MAX_SENT = 6                                  # concept sentences per token
DEF_PROMPTS = ["Explain what {w} means.", "What do you know about {w}?", "Describe {w} in one sentence."]
COPY_TRAIN = ["My friend wrote {w} on the whiteboard before the lesson.",
              "The last word of the message was {w}, followed by a smiley.",
              "Please file the folder labelled {w} in the top drawer.",
              "He said {w} twice and then left the room."]
COPY_INSTR = 'Copy this sentence word for word: "{s}"'
RETAIN_PROMPTS = ["Write one sentence that uses the word '{w}' in a sense unrelated to {c}. Reply with the sentence only.",
                  "Write a short everyday sentence containing the word '{w}', not about {c}. Reply with the sentence "
                  "only.",
                  "Give one example sentence with the word '{w}' from a context that has nothing to do with {c}. Reply "
                  "with the sentence only."]


# ---------------------------------------------------------------- helpers
def nm_dir(concept, seed, variant):
    return OUT / slug(concept) / f"s{seed}" / variant.replace(":", "_")


@torch.no_grad()
def greedy_ids(model, tok, prompts_ids, max_new, bs=16):
    """Greedy continuation token ids (eos stripped) for already tokenized prompts (lists of ids)."""
    out = []
    pad = tok.pad_token_id
    for i in range(0, len(prompts_ids), bs):
        chunk = prompts_ids[i:i + bs]
        L = max(len(x) for x in chunk)
        ids = torch.tensor([[pad] * (L - len(x)) + x for x in chunk], device=model.device)
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in chunk], device=model.device)
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=pad)[:, L:].tolist()
        for row in g:
            cut = [t for t in row]
            for stop in (tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")):
                if stop in cut:
                    cut = cut[:cut.index(stop)]
            out.append(cut)
    return out


def logits_at(model, ids, att, pos):
    """Soft-capped logits (fp32) at the given (row, position) pairs."""
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    rows = torch.tensor([p[0] for p in pos], device=ids.device)
    cols = torch.tensor([p[1] for p in pos], device=ids.device)
    lg = model.lm_head(h[rows, cols]).float()
    cap = getattr(model.config, "final_logit_softcapping", None)
    return cap * torch.tanh(lg / cap) if cap else lg


def batches(seqs, bs):
    for i in range(0, len(seqs), bs):
        yield seqs[i:i + bs]


def pad_batch(chunk, pad, device):
    L = max(len(x["ids"]) for x in chunk)
    ids = torch.tensor([x["ids"] + [pad] * (L - len(x["ids"])) for x in chunk], device=device)
    att = torch.tensor([[1] * len(x["ids"]) + [0] * (L - len(x["ids"])) for x in chunk], device=device)
    pos = [(b, p) for b, x in enumerate(chunk) for p in x["pos"]]
    return ids, att, pos


# ---------------------------------------------------------------- training data
def word_split(text, a, b):
    """The whole word around the character span [a, b): (start, end)."""
    while a > 0 and (text[a - 1].isalnum() or text[a - 1] in "'-"):
        a -= 1
    while b < len(text) and (text[b].isalnum() or text[b] in "'-"):
        b += 1
    return a, b


def eval_texts(concept):
    """Normalised question texts of every evaluation set of the concept (qa, simdom, collateral)."""
    from ember.local_datasets import load_mc_qa_items
    out = [it.question for sn in ("qa_train", "qa_test", "simdom_train", "simdom_test")
           for it in load_mc_qa_items(sn, concept)]
    for qf in ("data/collateral_questions_expansion.json",):
        d = json.loads(open(qf).read()).get(concept, {})
        out += [it["q"] for v in d.get("tokens", {}).values() for sp in ("val", "test") for it in v[sp]]
    return [norm_q(x) for x in out]


def norm_q(x):
    return re.sub(r"[^a-z0-9 ]", "", x.lower()).strip()


def near_dup(q, refs):
    import difflib
    a = norm_q(q)
    wa = set(a.split())
    for b in refs:
        wb = set(b.split())
        if a == b or (wa and wb and (len(wa & wb) / len(wa | wb) >= 0.6 or len(wa & wb) / min(len(wa), len(wb)) >= 0.8)) or \
                difflib.SequenceMatcher(None, a, b).ratio() >= 0.8:
            return True
    return False


def build(concept, seed, setname, model, w, sets, qa=False):
    """Examples {kind: [{ids, pos, (target ids, pos)}]} for one edit set; targets are filled by add_targets."""
    from experiments.varwidth.integrity import nonce, source_words, token_class
    from experiments.varwidth.prep import build_dataset
    tok = w.tokenizer
    ids_set = [int(t) for t in sets[setname]["ids"]]
    S = set(ids_set)
    sw = source_words(concept, tok, ids_set)
    cls = token_class(concept)
    rng = random.Random(seed)
    non = {}

    def nonce_of(word):
        if word not in non:
            non[word] = nonce(word, tok, S)
        return non[word]
    ex = {"forget": [], "copy": [], "retain": []}
    # (i) concept sentences: the whole word holding the token replaced by a nonce in the target
    sents = [s for s, lbl in build_dataset(concept).data if lbl == concept]
    rng.shuffle(sents)
    per = {t: 0 for t in ids_set}
    for s in sents:
        enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
        for p, (t, (a, b)) in enumerate(zip(enc["input_ids"], enc["offset_mapping"])):
            if t not in S or per[t] >= MAX_SENT or b <= a:
                continue
            wa, wb = word_split(s, a + (1 if s[a] == " " else 0), b)
            word = s[wa:wb]
            if not word or not re.search(r"[A-Za-z]", word):
                continue
            n = nonce_of(word)
            n = n.capitalize() if word[:1].isupper() else n.lower()
            pre, suf = s[:wa], s[wb:]
            P = [tok.bos_token_id] + (tok(pre.rstrip(" "), add_special_tokens=False)["input_ids"] if pre.strip() else [])
            sp = " " if pre.endswith(" ") else ""
            Wt = tok(sp + word, add_special_tokens=False)["input_ids"]
            Wn = tok(sp + n, add_special_tokens=False)["input_ids"]
            Sf = tok(suf, add_special_tokens=False)["input_ids"][:WIN]
            if t not in Wt or not Sf or set(Wn) & S:
                continue
            k = len(Sf)
            ex["forget"].append({"tok": t, "ids": P + Wt + Sf, "pos": list(range(len(P) + len(Wt) - 1, len(P) + len(Wt) - 1 + k)),
                                 "tids": P + Wn + Sf, "tpos": list(range(len(P) + len(Wn) - 1, len(P) + len(Wn) - 1 + k))})
            per[t] += 1
    # (ii) definitional prompts: teacher-force the unedited model's answer to the nonce prompt
    words = {t: sw[t][1] for t in sw}
    dp = []
    for t, slot in words.items():
        n = nonce_of(slot.strip('"'))
        n = f'"{n}"' if slot.startswith('"') else n
        for q in DEF_PROMPTS:
            dp.append((t, tok(w.wrap_prompt(q.format(w=slot)), add_special_tokens=True)["input_ids"],
                       tok(w.wrap_prompt(q.format(w=n)), add_special_tokens=True)["input_ids"]))
    resp = greedy_ids(model, tok, [x[2] for x in dp], 16)
    for (t, Pt, Pn), r in zip(dp, resp):
        if len(r) < 2:
            continue
        ex["forget"].append({"tok": t, "ids": Pt + r[:-1], "pos": list(range(len(Pt) - 1, len(Pt) - 1 + len(r) - 1)),
                             "tids": Pn + r[:-1], "tpos": list(range(len(Pn) - 1, len(Pn) - 1 + len(r) - 1))})
    if qa:   # addendum 2: self-written concept questions; target = the unedited answer to the nonce version
        refs = eval_texts(concept)
        qp = [tok(w.wrap_prompt(QA_PROMPT.format(c=concept, w=slot.strip('"'))), add_special_tokens=True)["input_ids"]
              for t, slot in words.items()]
        gq = greedy_ids(model, tok, qp, 90)
        qs, n_drop = [], 0
        for (t, slot), r in zip(words.items(), gq):
            for line in tok.decode(r, skip_special_tokens=True).split("\n"):
                q = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip()
                word = slot.strip('"')
                if len(q) < 12 or "?" not in q or word.lower() not in q.lower():
                    continue
                if near_dup(q, refs):
                    n_drop += 1
                    continue
                n = nonce_of(word)
                qn = re.sub(re.escape(word), n, q, flags=re.I)
                qs.append((t, tok(w.wrap_prompt(q), add_special_tokens=True)["input_ids"],
                           tok(w.wrap_prompt(qn), add_special_tokens=True)["input_ids"], q))
        resp = greedy_ids(model, tok, [x[2] for x in qs], 24)
        for (t, Pt, Pn, _), r in zip(qs, resp):
            if len(r) >= 2 and t in Pt:
                ex["forget"].append({"tok": t, "kind": "qa", "ids": Pt + r[:-1],
                                     "pos": list(range(len(Pt) - 1, len(Pt) - 1 + len(r) - 1)), "tids": Pn + r[:-1],
                                     "tpos": list(range(len(Pn) - 1, len(Pn) - 1 + len(r) - 1))})
        ex["_qa_meta"] = [{"q": x[3], "tok": x[0]} for x in qs] + [{"n_dropped_overlap": n_drop}]
    # copy: new carriers and instruction; the unedited model's answer on the true token is the target
    cp = [(t, tok(w.wrap_prompt(COPY_INSTR.format(s=c.format(w=slot))), add_special_tokens=True)["input_ids"])
          for t, slot in words.items() for c in COPY_TRAIN]
    resp = greedy_ids(model, tok, [x[1] for x in cp], 32)
    for (t, Pt), r in zip(cp, resp):
        if len(r) >= 2:
            ex["copy"].append({"tok": t, "ids": Pt + r[:-1], "pos": list(range(len(Pt) - 1, len(Pt) - 1 + len(r) - 1))})
    # retain: multi-meaning tokens, sentences the unedited model writes with the word outside the concept
    multi = [t for t, slot in words.items() if cls.get(tok.decode([t])) == "multi"]
    rp = [(t, tok(w.wrap_prompt(q.format(w=words[t].strip('"'), c=concept)), add_special_tokens=True)["input_ids"])
          for t in multi for q in RETAIN_PROMPTS]
    resp = greedy_ids(model, tok, [x[1] for x in rp], 40)
    for (t, _), r in zip(rp, resp):
        sent = tok.decode(r, skip_special_tokens=True).strip().strip('"')
        ids = [tok.bos_token_id] + tok(sent, add_special_tokens=False)["input_ids"]
        if t not in ids:
            continue
        p = ids.index(t)
        ps = list(range(p, min(len(ids) - 1, p + WIN)))
        if ps:
            ex["retain"].append({"tok": t, "ids": ids[:ps[-1] + 1], "pos": ps})
    return ex, {"n_tokens": len(ids_set), "n_words": len(words), "n_multi": len(multi), "nonces": non}


@torch.no_grad()
def add_targets(model, tok, ex):
    """Top-k log-probs (and the rest mass) of the unedited model at each example's target positions."""
    for kind, xs in ex.items():
        seqs = [{"ids": x.get("tids", x["ids"]), "pos": x.get("tpos", x["pos"])} for x in xs]
        k0 = 0
        for chunk in batches(seqs, 16):
            ids, att, pos = pad_batch(chunk, tok.pad_token_id, model.device)
            lp = F.log_softmax(logits_at(model, ids, att, pos), -1)
            v, i = lp.topk(TOPK, -1)
            rest = (1 - v.exp().sum(-1)).clamp_min(1e-8).log()
            j = 0
            for b, x in enumerate(chunk):
                m = len(x["pos"])
                xs[k0 + b]["t_lp"] = v[j:j + m].half().cpu()
                xs[k0 + b]["t_idx"] = i[j:j + m].int().cpu()
                xs[k0 + b]["t_rest"] = rest[j:j + m].half().cpu()
                j += m
            k0 += len(chunk)


def kl_topk(lg, t_lp, t_idx, t_rest):
    """KL(target || model) with the target's top-k and the rest mass."""
    lq = F.log_softmax(lg, -1)
    q_top = lq.gather(1, t_idx.long())
    q_rest = (1 - q_top.exp().sum(-1)).clamp_min(1e-8).log()
    p = t_lp.float().exp()
    pr = t_rest.float().exp()
    return ((p * (t_lp.float() - q_top)).sum(-1) + pr * (t_rest.float() - q_rest)).mean()


# ---------------------------------------------------------------- the edit
class RowDelta(torch.nn.Module):
    """Trainable Δ for the edited rows, in units of each row's norm; 'sub' restricts it to the span of the set's
    concept components (rank <= number of concept features)."""

    def __init__(self, ids, E, C, mode):
        super().__init__()
        self.ids = list(ids)
        self.register_buffer("En", E.norm(dim=1, keepdim=True).float())
        if mode == "sub":
            U, s, Vh = torch.linalg.svd(C.float(), full_matrices=False)
            k = int((s > s[0] * 1e-4).sum())
            self.register_buffer("B", Vh[:k].contiguous())
            self.coef = torch.nn.Parameter(torch.zeros(len(ids), k))
        else:
            self.B = None
            self.coef = torch.nn.Parameter(torch.zeros(len(ids), E.shape[1]))

    def delta(self):
        d = self.coef @ self.B if self.B is not None else self.coef
        return d * self.En                      # TL-scaled space

    def l2(self):
        return (self.coef @ self.B if self.B is not None else self.coef).pow(2).sum(1).mean()


def install_hook(model, rd, scale):
    """Adds Δ_t / scale to the embedding output wherever an edited token occurs (before Gemma's sqrt(d) normalizer)."""
    lut = torch.full((model.get_input_embeddings().weight.shape[0],), -1, dtype=torch.long, device=model.device)
    lut[torch.tensor(rd.ids, device=model.device)] = torch.arange(len(rd.ids), device=model.device)

    def hook(mod, inp, out):
        j = lut[inp[0]]
        m = j >= 0
        if not m.any():
            return out
        d = (rd.delta() / scale).to(out.dtype)
        out = out.clone()
        out[m] = out[m] + d[j[m]]
        return out
    return model.get_input_embeddings().register_forward_hook(hook)


def train_path(model, tok, ex, rd, mode, scale, log, steps=None):
    """Path over the knobs; yields (knob, Δ rows) after each knob's steps, with the hook removed and the model back in
    eval mode (the caller evaluates with the rows written)."""
    steps = steps or STEPS
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([rd.coef], lr=LR[mode])
    rng = random.Random(0)
    pools = {k: v for k, v in ex.items() if v}
    t0 = time.time()
    for wf in KNOBS:
        h = install_hook(model, rd, scale)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        try:
            for step in range(steps[wf]):
                loss_parts = {}
                tot = 0.0
                for kind, xs in pools.items():
                    chunk = rng.sample(xs, min(BATCH, len(xs)))
                    ids, att, pos = pad_batch(chunk, tok.pad_token_id, model.device)
                    lg = logits_at(model, ids, att, pos)
                    t_lp = torch.cat([x["t_lp"] for x in chunk]).to(model.device)
                    t_idx = torch.cat([x["t_idx"] for x in chunk]).to(model.device)
                    t_rest = torch.cat([x["t_rest"] for x in chunk]).to(model.device)
                    l = kl_topk(lg, t_lp, t_idx, t_rest)
                    loss_parts[kind] = float(l)
                    tot = tot + (wf if kind == "forget" else 1.0) * l
                tot = tot + L2 * rd.l2()
                opt.zero_grad()
                tot.backward()
                opt.step()
                if step % 40 == 0 or step == steps[wf] - 1:
                    log(f"wf={wf} step={step} " + " ".join(f"{k}={v:.3f}" for k, v in loss_parts.items()) +
                        f" |d|={rd.l2().sqrt().item():.3f} [{time.time() - t0:.0f}s]")
        finally:
            h.remove()
            model.gradient_checkpointing_disable()
            model.eval()
            model.config.use_cache = True
        yield wf, rd.delta().detach().double().cpu()


# ---------------------------------------------------------------- evaluation
def copy_eval(w, words):
    """T1 copy on the evaluation carriers (as integrity / copygrid): {word: [bool, bool]}."""
    from experiments.varwidth.integrity import COPY, fill
    prompts = [(wd, f'Repeat the following sentence exactly, and nothing else: "{fill(c, wd)}"')
               for wd in words for c in COPY]
    outs = w.generate_multiple([p for _, p in prompts], max_new_tokens=40, do_sample=False, batch_size=32)
    recs = {}
    for (wd, _), o in zip(prompts, outs):
        recs.setdefault(wd, []).append(wd.strip('"') in o)
    return recs


def eval_words(concept, seed, tok, ids):
    from experiments.varwidth.integrity import MAX_WORDS, source_words
    sw = source_words(concept, tok, sorted(ids))
    words = sorted({sw[t][1] for t in sw})
    if len(words) > MAX_WORDS:
        words = sorted(random.Random(seed).sample(words, MAX_WORDS))
    return words


def run(concept, seed, variants, qfile, pilot=False):
    from experiments.varwidth.erase import Evals
    from experiments.varwidth.erase_v2 import add_coll, coll_score, load_questions, score_val, setup
    base, sets, model, w, ed, E = setup(concept, seed)
    tok = w.tokenizer
    bc, bg = base[concept], base["_global"]
    ev = Evals(concept, 0)
    Q = load_questions(concept, qfile)
    cache = {}
    for v in variants:
        setname, mode = NM[v]
        st = sets[setname]
        d = nm_dir(concept, seed, v)
        if not st["ids"] or (d / "result_nm.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        logf = open(d / ("pilot.log" if pilot else "train.log"), "a")

        def log(msg):
            print(f"{concept} s{seed} {v} {msg}", flush=True)
            logf.write(msg + "\n")
            logf.flush()
        qa = mode == "freeqa"
        ck = (setname, qa)
        if ck not in cache:
            dp = OUT / slug(concept) / f"s{seed}" / f"_nm_data_{setname}{'_qa' if qa else ''}.pt"
            if dp.exists():
                cache[ck] = torch.load(dp, weights_only=False)
            else:
                ed.restore()
                ex, info = build(concept, seed, setname, model, w, sets, qa)
                meta = ex.pop("_qa_meta", None)
                add_targets(model, tok, ex)
                if meta is not None:
                    info["qa"] = meta
                cache[ck] = (ex, info)
                torch.save(cache[ck], dp.with_suffix(".tmp"))
                dp.with_suffix(".tmp").rename(dp)
            ex, info = cache[ck]
            log(f"data: " + " ".join(f"{k}={len(x)}" for k, x in ex.items()) +
                f" info={ {k: v2 for k, v2 in info.items() if k not in ('nonces', 'qa')} }" +
                (f" qa_questions={len(info['qa']) - 1} dropped={info['qa'][-1]}" if 'qa' in info else ""))
        ex, info = cache[ck]
        ids = [int(t) for t in st["ids"]]
        Eo = ed.orig_scaled(ids).double().cpu()
        rd = RowDelta(ids, Eo, st["C"].double(), mode).to(model.device)
        words = eval_words(concept, seed, tok, ids)
        rows, saved = [], {}
        ed.restore()
        for wf, D in train_path(model, tok, ex, rd, mode, ed.scale, log,
                                None):
            new = Eo + D
            r = float(np.median((D.norm(dim=1) / Eo.norm(dim=1)).numpy()))
            saved[wf] = new.half()
            if pilot:
                ed.set_rows(ids, new)
                cr = copy_eval(w, words)
                log(f"PILOT wf={wf} r={r:.3g} copy={np.mean([b for x in cr.values() for b in x]):.2f}")
                ed.restore()
                continue
            ed.set_rows(ids, new)
            row = {"variant": v, "param": wf, "r": r, "n_edit": len(ids)}
            score_val(ev, w, bc, bg, Q, row, ids)
            cr = copy_eval(w, words)
            row["copy"] = float(np.mean([b for x in cr.values() for b in x]))
            row["copy_recs"] = cr
            row["H3v"] = hm([row["efficacy"], row["specificity"], row.get("collateral", 0.0)])
            ed.restore()
            rows.append(row)
            save(d / "grid_nm.json", rows)
            log(f"wf={wf} r={r:.3g} eff={row['efficacy']:.2f} spec={row['specificity']:.2f} "
                f"coll={row.get('collateral', float('nan')):.2f} copy={row['copy']:.2f} mmlu={row['mmlu_frac']:.2f}")
        torch.save({"ids": ids, "rows": saved}, d / "rows_nm.pt")
        if pilot:
            continue
        ok = [x for x in rows if x["mmlu_frac"] >= 0.9 and x.get("collateral", 0) >= 0.8 and x["copy"] >= 0.9]
        best = sorted(ok, key=lambda x: (-x["H3v"], x["r"]))[0] if ok else None
        res = {"concept": concept, "seed": seed, "variant": v, "rule": "e3", "set": setname, "mode": mode}
        if best is None:
            res.update(param=None, r=0.0, n_edit=0)
            ed.restore()
        else:
            ed.set_rows(ids, saved[best["param"]].double())
            res.update(param=best["param"], r=best["r"], n_edit=len(ids), copy=best["copy"], copy_recs=best["copy_recs"])
        res["qa_acc"], res["simdom_acc"] = ev.mc_acc(w, "qa_test"), ev.mc_acc(w, "simdom_test")
        res["mmlu_acc"] = ev.mmlu_acc(w, "test")
        res["qa_frac"] = mc_norm(res["qa_acc"], bc["qa_test"])
        res["simdom_frac"] = mc_norm(res["simdom_acc"], bc["simdom_test"])
        res["mmlu_frac"] = mc_norm(res["mmlu_acc"], bg["mmlu_test"])
        res["efficacy"] = 1 - res["qa_frac"]
        res["specificity"] = hm([res["mmlu_frac"], res["simdom_frac"]])
        add_coll(res, coll_score(w, Q, "test"), ids)
        ed.restore()
        save(d / "result_nm.json", res)
        log(f"TEST param={res['param']} eff={res['efficacy']:.2f} spec={res['specificity']:.2f} "
            f"coll={res['collateral']:.2f}")
        logf.close()


def e2(concept, seed):
    """At each variant's selected point: judge-free E2 + T3 answers (integrity/<c>/s<seed>/nm_<v>@sel.json, picked up
    by integrity.judge_export_new for seed 42) and the neutral-prompt generations (last_mile/<c>/gen_nm.json)."""
    from ember.evals.mmlu import load_mmlu_items
    from ember.local_datasets import load_mmlu_indices
    from experiments.varwidth.erase_v2 import setup
    from experiments.varwidth.integrity import IOUT, measure, nonce
    from experiments.varwidth.last_mile import LM
    base, sets, model, w, ed, E = setup(concept, seed)
    tok = w.tokenizer
    mmlu = load_mmlu_items(load_mmlu_indices("train"))
    d = IOUT / slug(concept) / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    gf = LM / slug(concept) / "gen_nm.json"
    gen = json.loads(gf.read_text()) if gf.exists() else {}
    P = json.loads((LM / "prompts.json").read_text())[concept]
    for v in NM:
        rp = nm_dir(concept, seed, v) / "result_nm.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        if res.get("param") is None:
            continue
        R = torch.load(nm_dir(concept, seed, v) / "rows_nm.pt", weights_only=False)
        ids, new = R["ids"], R["rows"][res["param"]].double()
        words = eval_words(concept, seed, tok, ids)
        # reference top-up for words the amendment 8 configs did not cover
        for name in ("orig", "nonce", "del"):
            p = d / f"ref_{name}.json"
            have = json.loads(p.read_text()) if p.exists() else {}
            todo = [k for k in words if k not in have]
            if todo:
                ed.restore()
                slots = {k: (k if name == "orig" else "" if name == "del" else
                             (f'"{nonce(k.strip(chr(34)), tok, set(ids))}"' if k.startswith('"')
                              else nonce(k, tok, set(ids)))) for k in todo}
                have.update(measure(model, w, tok, todo, slots, mmlu))
                p.write_text(json.dumps(have, ensure_ascii=False))
        out = d / f"{v.replace(':', '_')}@sel.json"
        if not out.exists():
            ed.set_rows(ids, new)
            recs = measure(model, w, tok, words, {k: k for k in words}, mmlu)
            out.write_text(json.dumps({"config": f"{v}@sel", "variant": v, "param": res["param"], "r": res["r"],
                                       "n_edit": len(ids), "words": words, "static": {}, "recs": recs},
                                      ensure_ascii=False))
            print(concept, seed, v, "e2 done", flush=True)
        key = f"{v}@nm"
        if key not in gen:
            ed.set_rows(ids, new)
            gen[key] = {"seed": seed, "param": res["param"], "r": res["r"],
                        "gens": w.generate_multiple(P, max_new_tokens=80, do_sample=False, batch_size=10)}
            gf.write_text(json.dumps(gen, ensure_ascii=False, indent=1))
        ed.restore()


def gen_export():
    """Blind OK/BROKEN batches (last_mile/judge_nm/) for the NM generations, against the unedited answers."""
    from experiments.varwidth.last_mile import LM
    J = LM / "judge_nm"
    J.mkdir(parents=True, exist_ok=True)
    P = json.loads((LM / "prompts.json").read_text())
    items = []
    for c in EXPANSION:
        f = LM / slug(c) / "gen_nm.json"
        if not f.exists():
            continue
        g = json.loads(f.read_text())
        ref = json.loads((LM / slug(c) / "gen.json").read_text())["unedited"]["gens"]
        for name, x in g.items():
            for i, p in enumerate(P[c]):
                items.append({"concept": c, "config": name, "i": i, "prompt": p, "reference": ref[i],
                              "candidate": x["gens"][i]})
    rng = random.Random(5)
    rng.shuffle(items)
    key = {}
    for n, it in enumerate(items):
        it["id"] = f"n{n:05d}"
        key[it["id"]] = {k: it[k] for k in ("concept", "config", "i")}
    dup = [dict(it, id=it["id"] + "d") for it in rng.sample(items, len(items) // 10)]
    nb = max(1, -(-len(items) // 250))
    bs = [items[i::nb] for i in range(nb)]
    for j, x in enumerate(dup):
        bs[(int(x["id"][1:6]) % nb + 1 + j % max(1, nb - 1)) % nb if nb > 1 else 0].append(x)
    for b, its in enumerate(bs):
        rng.shuffle(its)
        (J / f"batch_{b}.json").write_text(json.dumps(
            [{k: it[k] for k in ("id", "prompt", "reference", "candidate")} for it in its], ensure_ascii=False, indent=1))
    (J / "key.json").write_text(json.dumps(key))
    print(f"{len(items)} items + {len(dup)} duplicates -> {nb} batches in {J}")


def report():
    from scipy.stats import wilcoxon

    from experiments.varwidth.collateral_report import renorm
    from experiments.varwidth.integrity import DESTR, GRACEFUL, IOUT, JDIR, LEAK, labels, summarize, token_class
    from experiments.varwidth.last_mile import LM
    L = ["# Nonce-matching edit (PREREG amendment 13)", "",
         "Seed 42. Selection as E3 (validation: MMLU ≥ 0.9, collateral ≥ 0.8, copy ≥ 0.9, max H3v); test split. "
         "Nonce point: test efficacy ≥ per-run nonce floor − 0.05, copy ≥ 0.9, test collateral ≥ 0.8.", ""]
    R, B = {}, {}
    for c in EXPANSION:
        fl = LM / slug(c) / "floor.json"
        fl = json.loads(fl.read_text()) if fl.exists() else {}
        for v, (setname, _) in NM.items():
            f = nm_dir(c, 42, v) / "result_nm.json"
            if f.exists():
                R[(c, 42, v)] = json.loads(f.read_text())
                fk = f"42|sub:{setname}"
                R[(c, 42, v)]["floor_nonce"] = (1 - mc_norm(fl[fk]["nonce_acc"], fl["unedited"])) if fk in fl else None
        for bv in ("sub:ember", "sub:cinder", "sub:cinder_z"):
            f = OUT / slug(c) / "s42" / bv.replace(":", "_") / "result_e3.json"
            if f.exists():
                B[(c, 42, bv)] = json.loads(f.read_text())
    for k, r in R.items():
        if not r.get("n_edit"):
            continue
        kk = (k[0], k[1], f"sub:{NM[k[2]][0]}")
        rr = {kk: dict(r, variant=kk[2])}
        renorm(rr)
        r["collateral_rn"], r["H3v_rn"] = rr[kk]["collateral_rn"], rr[kk]["H3v_rn"]
    if B:
        renorm(B)
    cls_cache = {}

    def exc_copy(c, r):
        if c not in cls_cache:
            cls_cache[c] = token_class(c)
        meta = IOUT / slug(c) / "s42" / "meta.json"
        meta = json.loads(meta.read_text())["words"] if meta.exists() else {}
        out = {"exclusive": [], "multi": []}
        for wd, b in (r.get("copy_recs") or {}).items():
            toks = meta.get(wd, {}).get("tokens", [])
            cl = "multi" if any(cls_cache[c].get(t) == "multi" for t in toks) else \
                "exclusive" if any(cls_cache[c].get(t) == "exclusive" for t in toks) else None
            if cl:
                out[cl].extend(b)
        return {k: (float(np.mean(x)) if x else float("nan")) for k, x in out.items()}
    L += ["| Variant | runs with a set | edits (passes the gates) | nonce point reached | test efficacy | nonce floor | "
          "collateral (rn) | copy | copy, exclusive | specificity | H3v | median r |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {"variants": {}}
    for v in NM:
        xs = [(k, r) for k, r in R.items() if k[2] == v]
        ed_ = [(k, r) for k, r in xs if r.get("n_edit")]
        att = [r for k, r in ed_ if r["floor_nonce"] is not None and r["efficacy"] >= r["floor_nonce"] - 0.05
               and r.get("copy", 0) >= 0.9 and r["collateral_rn"] >= 0.8]
        ec = [exc_copy(k[0], r)["exclusive"] for k, r in ed_]
        m = lambda key: float(np.mean([r[key] for k, r in ed_])) if ed_ else float("nan")  # noqa: E731
        fn = [r["floor_nonce"] for k, r in ed_ if r["floor_nonce"] is not None]
        res["variants"][v] = {"n_set": len(xs), "n_edit": len(ed_), "n_attain": len(att), "eff": m("efficacy"),
                              "floor": float(np.mean(fn)) if fn else float("nan"), "coll": m("collateral_rn"),
                              "copy": m("copy") if ed_ else float("nan"), "copy_exc": float(np.nanmean(ec)) if ec else
                              float("nan"), "spec": m("specificity"), "H3v": m("H3v_rn"),
                              "r": float(np.median([r["r"] for k, r in ed_])) if ed_ else float("nan")}
        x = res["variants"][v]
        L.append(f"| {v} | {x['n_set']} | {x['n_edit']} | {x['n_attain']} | {x['eff']:.2f} | {x['floor']:.2f} | "
                 f"{x['coll']:.2f} | {x['copy']:.2f} | {x['copy_exc']:.2f} | {x['spec']:.2f} | {x['H3v']:.2f} | "
                 f"{x['r']:.2g} |")
    L += ["", "E3 baselines (copy-gated collateral rule) on the same runs:", "",
          "| Baseline | edits | test efficacy | collateral (rn) | H3v | median r |", "|---|---|---|---|---|---|"]
    for bv in ("sub:ember", "sub:cinder", "sub:cinder_z"):
        xs = [r for k, r in B.items() if k[2] == bv and r.get("n_edit")]
        if xs:
            L.append(f"| {bv} (E3) | {len(xs)} | {np.mean([r['efficacy'] for r in xs]):.2f} | "
                     f"{np.mean([r['collateral_rn'] for r in xs]):.2f} | {np.mean([r['H3v_rn'] for r in xs]):.2f} | "
                     f"{np.median([r['r'] for r in xs]):.2g} |")
    L += ["", "Paired test H3v, NM variant − E3 baseline (unit = concept; runs where both edit):", "",
          "| Contrast | concepts | mean Δ | wins/losses | Wilcoxon p |", "|---|---|---|---|---|"]
    res["paired"] = {}
    for v in NM:
        for bv in ("sub:ember", "sub:cinder_z"):
            dd = [R[(c, 42, v)]["H3v_rn"] - B[(c, 42, bv)]["H3v_rn"] for c in EXPANSION
                  if R.get((c, 42, v), {}).get("n_edit") and B.get((c, 42, bv), {}).get("n_edit")]
            if len(dd) >= 1:
                p = float(wilcoxon(dd).pvalue) if len(dd) >= 2 and any(x != 0 for x in dd) else float("nan")
                res["paired"][f"{v}-{bv}"] = {"n": len(dd), "mean": float(np.mean(dd)), "p": p}
                L.append(f"| {v} − {bv} (E3) | {len(dd)} | {np.mean(dd):+.3f} | {sum(x > 0 for x in dd)}/"
                         f"{sum(x < 0 for x in dd)} | {p:.3f} |")
    # judge-free E2 vs references and baselines
    L += ["", "Judge-free E2 at the selected point (means over runs; references over the same runs' words):", "",
          "| Config | runs | copy | echo | context-MC | answer-NLL excess vs nonce |", "|---|---|---|---|---|---|"]
    agg = {}
    for c in EXPANSION:
        d = IOUT / slug(c) / "s42"
        for name in [f"{v}@sel" for v in NM] + ["sub:ember@sel", "sub:cinder_z@sel"]:
            if (d / f"{name.replace(':', '_')}.json").exists():
                s = summarize(d, name)["summary"]
                agg.setdefault(name, []).append(s)
    for name, ss in agg.items():
        L.append(f"| {name} | {len(ss)} | {np.mean([s['copy'] for s in ss]):.2f} | {np.mean([s['echo'] for s in ss]):.2f} | "
                 f"{np.mean([s['ctxmc'] for s in ss]):.2f} | {np.mean([s['ctx_excess'] for s in ss]):+.2f} |")
    if agg:
        ss = [s for v in agg.values() for s in v]
        L.append(f"| ref_orig / ref_nonce | – | {np.mean([s['copy_orig'] for s in ss]):.2f} / "
                 f"{np.mean([s['copy_nonce'] for s in ss]):.2f} | {np.mean([s['echo_orig'] for s in ss]):.2f} / "
                 f"{np.mean([s['echo_nonce'] for s in ss]):.2f} | {np.mean([s['ctxmc_orig'] for s in ss]):.2f} / "
                 f"{np.mean([s['ctxmc_nonce'] for s in ss]):.2f} | 0 |")
    # "renamed": the ASK answer drops the word and names a capitalised non-dictionary string instead
    dic = {x.strip().lower() for x in open("/usr/share/dict/words", errors="ignore")}
    ren = {}
    for c in EXPANSION:
        d = IOUT / slug(c) / "s42"
        for name in [f"{v}@sel" for v in NM] + ["sub:ember@sel", "sub:cinder_z@sel"]:
            f = d / f"{name.replace(':', '_')}.json"
            if not f.exists():
                continue
            j = json.loads(f.read_text())
            refs = {k: json.loads((d / f"ref_{k}.json").read_text()) for k in ("orig", "nonce")}
            for src, recs, tag in [(j["recs"], j["recs"], name)] + [(None, refs[k], f"ref_{k}") for k in refs]:
                for wd in j["words"]:
                    r = recs.get(wd)
                    if not r or "ask_out" not in r:
                        continue
                    o = r["ask_out"]
                    novel = [x for x in re.findall(r"\b[A-Z][a-z]{3,}\b", o) if x.lower() not in dic
                             and x.lower() not in wd.lower() and x.lower() not in c.lower()]
                    ren.setdefault(tag if src is not None else f"{tag} (words of {name})", []).append(
                        (not r.get("echo")) and bool(novel))
    if ren:
        L += ["", "Renamed rate: the answer to the T3 prompt does not contain the word and names a capitalised "
              "non-dictionary string instead (string check):", "", "| Config | answers | renamed |", "|---|---|---|"]
        res["renamed"] = {}
        for k in sorted(ren):
            res["renamed"][k] = float(np.mean(ren[k]))
            L.append(f"| {k} | {len(ren[k])} | {np.mean(ren[k]):.2f} |")
    # T3 labels
    key = json.loads((JDIR / "key.json").read_text()) if (JDIR / "key.json").exists() else {}
    agent = {**{b: "r1a" for b in (58, 59)}, **{b: "r1b" for b in (60, 61, 62)}, **{b: "r1c" for b in (63, 64, 65)},
             **{b: "r1d" for b in (66, 67)}}         # amendment 13 round 1 labellers (coordinator's split)
    if (JDIR / "nm_round2_agents.json").exists():
        agent.update({int(k): v for k, v in json.loads((JDIR / "nm_round2_agents.json").read_text()).items()})
    batch_of = {it["id"]: int(f.stem.split("_")[1]) for f in JDIR.glob("batch_*.json")
                if int(f.stem.split("_")[1]) >= 58 for it in json.loads(f.read_text())}
    lab = labels()
    from collections import Counter
    cnt = {}
    for i, k in key.items():
        if k["concept"] in EXPANSION and k["seed"] == 42 and i in lab and (k["config"].startswith("nm:") or
                                                                          k["config"] in ("sub:ember@sel", "sub:cinder_z@sel")
                                                                          or k["config"].startswith("ref_")):
            cnt.setdefault(k["config"], Counter())[lab[i]] += 1
    if any(k.startswith("nm:") for k in cnt):
        L += ["", "T3 labels at the selected point (seed 42, blind agents):", "",
              "| Config | n | graceful | leak | confab | destruction |", "|---|---|---|---|---|---|"]
        for cfg, cn in sorted(cnt.items()):
            n = sum(cn.values())
            L.append(f"| {cfg} | {n} | {sum(cn[x] for x in GRACEFUL) / n:.2f} | {sum(cn[x] for x in LEAK) / n:.2f} | "
                     f"{cn['CONFAB'] / n:.2f} | {sum(cn[x] for x in DESTR) / n:.2f} |")
        res["t3"] = {k: dict(v) for k, v in cnt.items()}
        dups = [(i, i + "d") for i in key if i in batch_of and i + "d" in batch_of and i in lab and i + "d" in lab
                and agent.get(batch_of[i]) and agent.get(batch_of[i + "d"]) and
                agent[batch_of[i]] != agent[batch_of[i + "d"]]]
        if dups:
            ag = (float(np.mean([lab[a] == lab[b] for a, b in dups])),
                  float(np.mean([(lab[a] in DESTR) == (lab[b] in DESTR) for a, b in dups])))
            res["t3_agreement"] = {"n": len(dups), "exact": ag[0], "destruction": ag[1]}
            L.append(f"\nCross-agent agreement on {len(dups)} NM duplicates: exact {ag[0]:.2f}, destruction {ag[1]:.2f}.")
    # generation damage
    J = LM / "judge_nm"
    if (J / "key.json").exists():
        k2 = json.loads((J / "key.json").read_text())
        l2 = {}
        for f in sorted(J.glob("labels_*.json")):
            for k, x in json.loads(f.read_text()).items():
                l2.setdefault(k, x["label"] if isinstance(x, dict) else x)
        if l2:
            L += ["", "Generation damage on the neutral prompts (blind OK/BROKEN vs the unedited answer):", "",
                  "| Config | prompts | broken |", "|---|---|---|"]
            g = {}
            for i, k in k2.items():
                if i in l2:
                    g.setdefault(k["config"], []).append(l2[i] == "BROKEN")
            for cfg, xs in sorted(g.items()):
                L.append(f"| {cfg} | {len(xs)} | {np.mean(xs):.2f} |")
            res["gen"] = {k: float(np.mean(v)) for k, v in g.items()}
            bo = {it["id"]: f.stem for f in J.glob("batch_*.json") for it in json.loads(f.read_text())}
            dd = [(i, i + "d") for i in k2 if i in l2 and i + "d" in l2 and bo.get(i) != bo.get(i + "d")]
            if dd:
                res["gen_agreement"] = {"n": len(dd), "exact": float(np.mean([l2[a] == l2[b] for a, b in dd]))}
                L.append(f"\nAgreement on {len(dd)} cross-batch duplicates: {res['gen_agreement']['exact']:.2f}.")
            # baseline broken rates from amendment 9 (collateral-rule points; different seed for 4 concepts)
            lm = json.loads((OUT / "last_mile.json").read_text()) if (OUT / "last_mile.json").exists() else {}
            if lm.get("gen"):
                L.append("\nFor reference (amendment 9, collateral-rule points): " + ", ".join(
                    f"{k} {v['broken']:.2f}" for k, v in sorted(lm["gen"].items())) + ".")
    (OUT / "nonce_edit.md").write_text("\n".join(L) + "\n")
    (OUT / "nonce_edit.json").write_text(json.dumps(res, indent=1, default=str))
    print("\n".join(L))


def main():
    from dotenv import load_dotenv

    from experiments.varwidth.common import ROOT
    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "e2", "report", "gen-export"))
    ap.add_argument("--concept")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", nargs="*", default=list(NM))
    ap.add_argument("--qfile", default="data/collateral_questions_expansion.json")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.what == "run":
        run(a.concept, a.seed, a.variants, a.qfile, a.pilot)
    elif a.what == "e2":
        e2(a.concept, a.seed)
    elif a.what == "gen-export":
        gen_export()
    else:
        report()


if __name__ == "__main__":
    main()
