#!/usr/bin/env python
"""WP6b mean_edit, GPU stages. Imports the collateral pipeline (erase_v2, integrity, last_mile) and writes only under
results/paper_analysis/mean_edit/. Existing grids / tests of CINDER, CINDER-z, EMBER and Noise are read from
results/varwidth/layers/collateral/ (read only).

    python -m experiments.paper_analysis.mean_edit.run pilot                       # timing + reproduction check
    python -m experiments.paper_analysis.mean_edit.run all --concept C --seed S    # every stage of one run
    python -m experiments.paper_analysis.mean_edit.run floor --concept C           # nonce / deletion floors

Stages of `all` (priority order; each stage skips what exists): grid (own-r MEAN-C / EMBER-c / MEAN-E), copy (T1 at
grid points with r <= 2.5), final (test passes at the collateral-rule and E3 points), integ (E2 judge-free tests),
dgrid (delta-matched MEAN-Cd / MEAN-Ed).
"""
from __future__ import annotations

import argparse
import functools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments.paper_analysis.mean_edit.sets import CINDER_RUNS, EMBER_RUNS, install
from experiments.varwidth.common import ROOT, slug

OUT = ROOT / "results" / "paper_analysis" / "mean_edit"
RUNS = OUT / "runs"
INTEG = OUT / "integrity"
COLL = ROOT / "results" / "varwidth" / "layers" / "collateral"
IOUT_OLD = COLL / "integrity"
QFILE = ROOT / "data" / "collateral_questions.json"
DGRID = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
SWEEP = [0.25, 1.0, 10.0, 100.0]
RMAX_COPY = 2.5
COPY_GATE = 0.9

FAM_C = {"new": ["sub:mean_c"], "old": ["sub:cinder", "sub:cinder_z"], "noise": ["noise:cinder"], "d": ["sub:mean_cd"],
         "words_from": "sub_cinder_z@1.0"}
FAM_E = {"new": ["sub:ember_c", "sub:mean_e"], "old": ["sub:ember"], "noise": ["noise:ember"], "d": ["sub:mean_ed"],
         "words_from": "sub_ember@1.0"}


ONLY = {"family": None}        # the queue worker restricts the stages to one family ("C" or "E")


def families(concept, seed):
    out = []
    if (concept, seed) in CINDER_RUNS and ONLY["family"] in (None, "C"):
        out.append(FAM_C)
    if (concept, seed) in EMBER_RUNS and ONLY["family"] in (None, "E"):
        out.append(FAM_E)
    return out


def is_new(v):
    return v.split(":")[1] in ("mean_c", "ember_c", "mean_e", "mean_cd", "mean_ed")


def vdir(concept, seed, v, mine=True):
    """Directory of a variant: this WP's folder (mine=True) or the existing collateral folder."""
    base = RUNS if mine else COLL
    return base / slug(concept) / f"s{seed}" / v.replace(":", "_")


def grid_rows(concept, seed, v):
    p = vdir(concept, seed, v, mine=is_new(v)) / "grid.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------ model and patches
@functools.lru_cache(maxsize=1)
def model_once():
    from experiments.varwidth.erase import load_model
    model, w = load_model()
    W = model.get_input_embeddings().weight
    pristine = W.detach().clone()
    return model, w, pristine


def patch():
    """Extended edit sets, cached model, outputs in this WP, delta-matched grids."""
    import experiments.varwidth.erase_v2 as ev2
    install()
    ev2.load_model = lambda: model_once()[:2]
    ev2.out_dir = lambda c, s, v: vdir(c, s, v, mine=True)
    orig_gp = ev2.grid_params
    ev2.grid_params = lambda v, pilot=False: DGRID if v.split(":")[1] in ("mean_cd", "mean_ed") else orig_gp(v, pilot)
    return ev2


def reset_embeddings():
    """Restore the whole input-embedding matrix from the pristine copy (guards against a stage that died mid-edit)."""
    model, _, pristine = model_once()
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(pristine)


def setup(concept, seed):
    ev2 = patch()
    reset_embeddings()
    return ev2.setup(concept, seed)


# ------------------------------------------------------------------------------------------------ stages
def st_grid(concept, seed, delta=False):
    ev2 = patch()
    vs = [v for f in families(concept, seed) for v in (f["d"] if delta else f["new"])]
    reset_embeddings()
    t0 = time.time()
    ev2.grid(concept, seed, None, str(QFILE), False, vs)
    print(f"[grid{'-d' if delta else ''}] {concept} {seed} {vs} done in {time.time() - t0:.0f}s", flush=True)


def words_of_set(concept, seed, tok, ids):
    """The copygrid word list (last_mile.copygrid): source words of the set's tokens, <= MAX_WORDS seeded sample."""
    from experiments.varwidth.integrity import MAX_WORDS, source_words
    sw = source_words(concept, tok, sorted(ids))
    words = sorted({sw[t][1] for t in sw})
    if len(words) > MAX_WORDS:
        words = sorted(random.Random(seed).sample(words, MAX_WORDS))
    return words


def st_copy(concept, seed):
    from experiments.varwidth.integrity import COPY, fill
    base, sets, model, w, ed, E = setup(concept, seed)
    tok = w.tokenizer
    t0 = time.time()
    for f in families(concept, seed):
        for v in f["new"] + f["old"] + f["noise"]:
            rows = grid_rows(concept, seed, v)
            if not rows:
                print(f"[copy] {concept} {seed} {v}: no grid", flush=True)
                continue
            out_f = vdir(concept, seed, v) / "copygrid.json"
            out = json.loads(out_f.read_text()) if out_f.exists() else {}
            ids0 = E.comp(v)[0]
            words = words_of_set(concept, seed, tok, ids0)
            prompts = [(wd, f'Repeat the following sentence exactly, and nothing else: "{fill(c, wd)}"')
                       for wd in words for c in COPY]
            for x in rows:
                if x["r"] > RMAX_COPY:
                    continue
                k = f"{x['param']:g}"
                if k in out:
                    continue
                ids, new, rr = E.rows(v, x["param"])
                ed.set_rows(ids, new)
                outs = w.generate_multiple([p for _, p in prompts], max_new_tokens=40, do_sample=False, batch_size=32)
                ed.restore()
                recs = {}
                for (wd, _), o in zip(prompts, outs):
                    recs.setdefault(wd, []).append(wd.strip('"') in o)
                out[k] = {"param": x["param"], "r": rr, "copy": float(np.mean([b for r in recs.values() for b in r])),
                          "recs": recs, "words": words}
                save(out_f, out)
                print(f"[copy] {concept} {seed} {v} {k} r={rr:.3g} copy={out[k]['copy']:.2f} [{time.time() - t0:.0f}s]",
                      flush=True)
    ed.restore()


def e3_rows(concept, seed, v, rows):
    cg_f = vdir(concept, seed, v) / "copygrid.json"
    if not cg_f.exists():
        return None
    cg = json.loads(cg_f.read_text())
    ok = {x["param"] for x in cg.values() if x["copy"] >= COPY_GATE}
    return [x for x in rows if x["param"] in ok and x["r"] <= RMAX_COPY]


def points(concept, seed, v):
    """{rule: selected param or None} for the collateral rule ('sel') and E3 ('e3')."""
    import experiments.varwidth.erase_v2 as ev2
    rows = grid_rows(concept, seed, v)
    if not rows:
        return {}
    out = {}
    b = ev2.select([dict(x) for x in rows])
    out["sel"] = None if b is None else b["param"]
    er = e3_rows(concept, seed, v, rows)
    if er is not None:
        b = ev2.select([dict(x) for x in er])
        out["e3"] = None if b is None else b["param"]
    return out


def test_pass(ev, w, ed, E, bc, bg, Q, v, param):
    """The test pass of erase_v2.final (no Alpaca): qa_test, simdom_test, MMLU test 1000, collateral test."""
    from experiments.varwidth.erase import hm, mc_norm
    from experiments.varwidth.erase_v2 import add_coll, coll_score
    res = {"variant": v}
    if param is None:
        ed.restore()
        res.update(param=None, r=0.0, n_edit=0)
    else:
        ids, new, rr = E.rows(v, param)
        ed.set_rows(ids, new)
        res.update(param=param, r=rr, n_edit=len(ids))
    res["qa_acc"], res["simdom_acc"] = ev.mc_acc(w, "qa_test"), ev.mc_acc(w, "simdom_test")
    res["mmlu_acc"] = ev.mmlu_acc(w, "test")
    res["qa_frac"] = mc_norm(res["qa_acc"], bc["qa_test"])
    res["simdom_frac"] = mc_norm(res["simdom_acc"], bc["simdom_test"])
    res["mmlu_frac"] = mc_norm(res["mmlu_acc"], bg["mmlu_test"])
    res["efficacy"] = 1 - res["qa_frac"]
    res["specificity"] = hm([res["mmlu_frac"], res["simdom_frac"]])
    add_coll(res, coll_score(w, Q, "test"), E.comp(v)[0])
    res["H3v"] = hm([res["efficacy"], res["specificity"], res["collateral"]])
    ed.restore()
    return res


def st_final(concept, seed):
    from experiments.varwidth.erase import Evals
    from experiments.varwidth.erase_v2 import load_questions
    base, sets, model, w, ed, E = setup(concept, seed)
    bc, bg = base[concept], base["_global"]
    ev = Evals(concept, 0)
    Q = load_questions(concept, str(QFILE))
    t0 = time.time()
    for f in families(concept, seed):
        for v in f["new"] + f["old"]:
            pts = points(concept, seed, v)
            d = vdir(concept, seed, v)
            done = {}
            if not is_new(v):        # the collateral-rule test pass exists in the collateral folder
                old = json.loads((vdir(concept, seed, v, mine=False) / "result.json").read_text())
                assert old["param"] == pts.get("sel"), f"{v}: reselected point differs from result.json"
                done[old["param"]] = old
            for rule in ("sel", "e3"):
                if rule not in pts:
                    continue
                fn = d / f"result_{rule}.json"
                if fn.exists():
                    done.setdefault(json.loads(fn.read_text())["param"], json.loads(fn.read_text()))
                    continue
                par = pts[rule]
                if par in done:
                    res = {k: x for k, x in done[par].items() if k not in ("rule",)}
                    res.update(rule=rule, reused=True)
                else:
                    res = test_pass(ev, w, ed, E, bc, bg, Q, v, par)
                    res.update(concept=concept, seed=seed, rule=rule)
                    done[par] = res
                save(fn, res)
                print(f"[final] {concept} {seed} {v} {rule} param={par} eff={res['efficacy']:.2f} "
                      f"coll={res['collateral']:.2f} H3v={res['H3v']:.2f} [{time.time() - t0:.0f}s]", flush=True)
    ed.restore()


def integ_configs(concept, seed):
    """[(name, variant, param, family)]: new variants at sel, e3 and the sweep; existing ones at e3 when that point
    has no existing E2 file."""
    out = []
    old_dir = IOUT_OLD / slug(concept) / f"s{seed}"
    for f in families(concept, seed):
        for v in f["new"]:
            pts = points(concept, seed, v)
            out += [(f"{v}@{r}", v, r, f) for r in SWEEP]
            seen = set(SWEEP)
            for rule in ("sel", "e3"):            # a selected point equal to a sweep point reuses that file
                if pts.get(rule) is not None and pts[rule] not in seen:
                    out.append((f"{v}@{rule}", v, pts[rule], f))
                    seen.add(pts[rule])
        for v in f["old"]:
            pts = points(concept, seed, v)
            if pts.get("e3") is None:
                continue
            par = pts["e3"]
            have = [nm for nm, p in ((f"{v}@sel", pts.get("sel")),) + tuple((f"{v}@{r}", r) for r in SWEEP)
                    if p == par and (old_dir / f"{nm.replace(':', '_')}.json").exists()]
            if not have:
                out.append((f"{v}@e3", v, par, f))
    return out


def st_integ(concept, seed):
    from ember.evals.mmlu import load_mmlu_items
    from ember.local_datasets import load_mmlu_indices
    from experiments.varwidth.integrity import measure, static, vocab_norm_stats
    base, sets, model, w, ed, E = setup(concept, seed)
    tok = w.tokenizer
    mmlu = load_mmlu_items(load_mmlu_indices("train"))
    d = INTEG / slug(concept) / f"s{seed}"
    old_dir = IOUT_OLD / slug(concept) / f"s{seed}"
    all_ids = sorted({int(t) for k, st in sets.items() if not k.startswith("_") for t in st["ids"]})
    ed.restore()
    W = model.get_input_embeddings().weight
    vstats = vocab_norm_stats(W, ed.scale)
    t0 = time.time()
    for name, v, par, fam in integ_configs(concept, seed):
        p = d / f"{name.replace(':', '_')}.json"
        if p.exists():
            continue
        words = json.loads((old_dir / f"{fam['words_from']}.json").read_text())["words"]
        ids, new, rr = E.rows(v, par)
        ed.set_rows(ids, new)
        recs = measure(model, w, tok, words, {k: k for k in words}, mmlu)
        st = static(W, ed.scale, ids, new.float(), vstats, exclude=all_ids)
        ed.restore()
        save(p, {"config": name, "variant": v, "param": par, "r": rr, "n_edit": len(ids), "words": words,
                 "static": st, "recs": recs})
        cp = np.mean([x for wd in words for x in recs[wd]["copy"]])
        mc = np.mean([x for wd in words for x in recs[wd]["ctxmc"]])
        print(f"[integ] {concept} {seed} {name} r={rr:.3g} copy={cp:.2f} ctxmc={mc:.2f} [{time.time() - t0:.0f}s]",
              flush=True)
    ed.restore()


def st_floor(concept):
    """Nonce / deletion floors of efficacy for CINDER's and EMBER's token sets (last_mile.floor, also on qa_train)."""
    import copy as _copy

    from ember.evals.mc import evaluate_mc_generation, prepare_mc_items
    from ember.local_datasets import load_mc_qa_items
    from experiments.varwidth.collateral import edit_sets, seeds
    from experiments.varwidth.last_mile import unreadable
    reset_embeddings()
    _, w, _ = model_once()
    tok = w.tokenizer
    out_f = OUT / "floors" / f"{slug(concept)}.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    for split in ("qa_train", "qa_test"):
        items = prepare_mc_items(load_mc_qa_items(split, concept), seed=42)
        if f"unedited|{split}" not in out:
            c, _, _ = evaluate_mc_generation(w, items)
            out[f"unedited|{split}"] = c / len(items)
        for s in seeds(concept):
            sets = edit_sets(concept, s)
            for v in ("cinder", "ember"):
                ids = set(int(t) for t in sets[v]["ids"])
                k = f"{s}|{v}|{split}"
                if not ids or k in out:
                    continue
                res = {"n_edit": len(ids)}
                for mode in ("nonce", "del"):
                    mod, touched = [], 0
                    for it in items:
                        q, n = unreadable(it.question, tok, ids, mode, ids)
                        opts = {L: unreadable(o, tok, ids, mode, ids)[0] for L, o in it.options_by_letter.items()}
                        touched += n > 0
                        x = _copy.copy(it)
                        x.question, x.options_by_letter = q, opts
                        x.correct_text = opts[it.correct_letter]
                        mod.append(x)
                    c, _, _ = evaluate_mc_generation(w, mod)
                    res[f"{mode}_acc"] = c / len(mod)
                    res["q_touched"] = touched / len(mod)
                out[k] = res
                save(out_f, out)
                print("[floor]", concept, k, res, flush=True)


def pilot():
    """Time one new grid point and re-evaluate one existing point (CINDER-z, r = 1, Harry Potter 42)."""
    from experiments.varwidth.erase import Evals
    from experiments.varwidth.erase_v2 import load_questions, score_val
    c, s = "Harry Potter", 42
    t0 = time.time()
    base, sets, model, w, ed, E = setup(c, s)
    t_load = time.time() - t0
    bc, bg = base[c], base["_global"]
    ev = Evals(c, 0)
    Q = load_questions(c, str(QFILE))
    res = {"load_s": t_load}
    for v, p in (("sub:cinder_z", 1.0), ("sub:mean_c", 1.0)):
        t1 = time.time()
        ids, new, rr = E.rows(v, p)
        ed.set_rows(ids, new)
        row = {"variant": v, "param": p, "r": rr, "n_edit": len(ids)}
        score_val(ev, w, bc, bg, Q, row, ids)
        ed.restore()
        row["seconds"] = time.time() - t1
        res[v] = {k: x for k, x in row.items() if k != "coll_tok"}
        print(json.dumps(res[v]), flush=True)
    old = next(x for x in grid_rows(c, s, "sub:cinder_z") if x["param"] == 1.0)
    res["reproduction"] = {k: (old[k], res["sub:cinder_z"][k]) for k in ("efficacy", "specificity", "collateral",
                                                                          "mmlu_acc", "qa_acc", "simdom_acc", "r")}
    res["reproduction_exact"] = all(abs(a - b) < 1e-9 for a, b in res["reproduction"].values())
    save(OUT / "pilot.json", res)
    print(json.dumps({k: res[k] for k in ("load_s", "reproduction", "reproduction_exact")}), flush=True)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pilot", "all", "grid", "copy", "final", "integ", "dgrid", "floor"))
    ap.add_argument("--concept")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--stages", nargs="*", default=["grid", "copy", "final", "integ", "dgrid"])
    a = ap.parse_args()
    if a.what == "pilot":
        pilot()
        return
    if a.what == "floor":
        st_floor(a.concept)
        return
    stages = a.stages if a.what == "all" else [a.what]
    fn = {"grid": lambda: st_grid(a.concept, a.seed), "copy": lambda: st_copy(a.concept, a.seed),
          "final": lambda: st_final(a.concept, a.seed), "integ": lambda: st_integ(a.concept, a.seed),
          "dgrid": lambda: st_grid(a.concept, a.seed, delta=True)}
    for st in stages:
        t0 = time.time()
        fn[st]()
        print(f"=== stage {st} {a.concept} {a.seed}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
