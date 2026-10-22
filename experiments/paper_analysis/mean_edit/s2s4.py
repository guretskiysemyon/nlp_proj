#!/usr/bin/env python
"""WP6b mean_edit, CPU follow-ups of the WP6 report.

    python -m experiments.paper_analysis.mean_edit.s2s4 s2    # read gain under modified layer-0 norm weights
    python -m experiments.paper_analysis.mean_edit.s2s4 s4    # forward-pass sensitivity (float32, CPU)

S2. WP6 read gain (|[W_q; W_k; W_v](g * u_hat)| relative to the mean over 500 random unit directions, torch seed 0;
median over edited tokens) recomputed with (a) the actual g = 1 + input_layernorm weight, (b) g = 1, (c) g with its 8
largest coordinates set to 0. The random baseline is recomputed under each g. Edit rows from WP6's run_edits (EMBER,
EMBER-c, CINDER, CINDER-z) on every WP6 run; the MEAN edits have the single direction mu_B-hat.

S4. gemma-2-2b-it in float32. 50 tokens: 5 runs with both families x 5 tokens sampled from CINDER's edit set and 5
from EMBER's (seed 0). Each token id sits in the first E2 copy carrier at the token-id level (BOS + "Yesterday my
cousin kept talking about" + t + " while we cooked dinner."). The input row of t is perturbed by -s |e_t| u (TL-scaled
space; u the unit edit direction, or one of 4 random unit directions) at s in {0.01, 0.25, 1}:
  l0_write  |delta(h1 - h0)| / |delta h0| at the token position (what layer 0 writes in response)
  l1_tot    |delta h1| / |delta h0| at the token position
  l1_after  mean over later positions of |delta h1| / |delta h0| (what reaches the context through layer 0)
  kl_tok    KL(p_orig || p_edit) of the next-token distribution at the token position
  kl_after  mean KL over the later positions
Each is also given relative to the mean over the token's 4 random directions at the same s. The linear proxy is WP6's
read gain of the same unit direction.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from experiments.paper_analysis.mean_edit.run import OUT, save

S4_RUNS = [("Harry Potter", 42), ("Harry Potter", 43), ("Ancient Rome", 44), ("Valentine's Day", 42),
           ("Valentine's Day", 46)]
SCALES = [0.01, 0.25, 1.0]
N_RAND = 4
PREFIX = "Yesterday my cousin kept talking about"
SUFFIX = " while we cooked dinner."


# ------------------------------------------------------------------------------------------------ S2
def s2():
    from experiments.paper_analysis.edit_direction.common import SEEDS, background, concepts, layer0, run_edits
    torch.set_num_threads(4)
    L = layer0()
    g0, Wq = L["g"].double(), L["qkv"].double()
    d = g0.numel()
    top8 = torch.argsort(g0, descending=True)[:8]
    gz = g0.clone()
    gz[top8] = 0.0
    G = {"actual": g0, "g1": torch.ones(d, dtype=torch.float64), "top8_zero": gz}
    torch.manual_seed(0)
    R = torch.randn(500, d, dtype=torch.float64)

    def raw(U, g):
        U = U.double()
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-30)
        return (Wq @ (g * U).T).norm(dim=0)

    base = {k: float(raw(R, g).mean()) for k, g in G.items()}

    def gain(U):
        return {k: (raw(U, g) / base[k]).numpy() for k, g in G.items()}

    mu = background()["mu"]
    mh = (mu / mu.norm())[None]
    out = {"top8_coords": top8.tolist(), "top8_g": g0[top8].tolist(), "g_median": float(g0.median()),
           "random_base": base, "mu_hat": {k: float(v[0]) for k, v in gain(mh).items()},
           "mu_hat_energy_top8": float((mh[0, top8] ** 2).sum()), "runs": []}
    t0 = time.time()
    for concept, sl in concepts():
        for seed in SEEDS:
            try:
                run = run_edits(concept, sl, seed)
            except Exception as e:          # noqa: BLE001
                print("skip", concept, seed, e, flush=True)
                continue
            if run is None or not run["sets"]:
                continue
            rec = {"concept": concept, "seed": seed}
            for v in ("ember", "ember_c", "cinder", "cinder_z"):
                if v in run["sets"]:
                    gg = gain(run["sets"][v]["C"])
                    rec[v] = {k: float(np.median(x)) for k, x in gg.items()}
                    C = run["sets"][v]["C"].double()
                    Cp = C - (C @ mh[0])[:, None] * mh
                    gp = gain(Cp)
                    rec[v + "_perp_mu"] = {k: float(np.median(x)) for k, x in gp.items()}
            out["runs"].append(rec)
        print(f"s2 {concept} [{time.time() - t0:.0f}s]", flush=True)
    # concept means
    summ = {}
    for v in ("ember", "ember_c", "cinder", "cinder_z", "ember_perp_mu", "cinder_perp_mu", "cinder_z_perp_mu",
              "ember_c_perp_mu"):
        for k in G:
            per_c = {}
            for r in out["runs"]:
                if v in r:
                    per_c.setdefault(r["concept"], []).append(r[v][k])
            if per_c:
                summ.setdefault(v, {})[k] = {"mean": float(np.mean([np.mean(x) for x in per_c.values()])),
                                             "n_concepts": len(per_c)}
    gaps = {}
    for a, b in (("ember", "cinder"), ("cinder_z", "cinder"), ("ember", "ember_c")):
        for k in G:
            per = {}
            for r in out["runs"]:
                if a in r and b in r:
                    per.setdefault(r["concept"], []).append(r[a][k] - r[b][k])
            m = [np.mean(x) for x in per.values()]
            gaps.setdefault(f"{a}-{b}", {})[k] = {"mean": float(np.mean(m)), "n_concepts": len(m),
                                                  "wins": int(sum(x > 0 for x in m))}
    out["summary"], out["gaps"] = summ, gaps
    save(OUT / "s2_read_gain.json", out)
    print(json.dumps({"mu_hat": out["mu_hat"], "gaps": gaps}, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ S4
def s4_tokens():
    """[(concept, seed, family, token id, {direction name: unit row (d,)}, e_t (d,))]."""
    from experiments.paper_analysis.mean_edit.sets import all_sets
    from experiments.varwidth.collateral import load_fit
    rng = np.random.default_rng(0)
    out = []
    for c, s in S4_RUNS:
        sets = all_sets(c, s)
        A = load_fit(c, "L0", s)[0]["A"].double()
        mu = load_fit(c, "L12", s)[0]["mu_B"].double()
        mh = mu / mu.norm()
        for fam, names in (("C", ("cinder", "cinder_z", "mean_c")), ("E", ("ember", "ember_c", "mean_e"))):
            st = sets[names[0]]
            pick = rng.choice(len(st["ids"]), size=min(5, len(st["ids"])), replace=False)
            for j in pick:
                dirs = {}
                for nm in names:
                    row = sets[nm]["C"][j].double()
                    dirs[nm] = row / row.norm()          # the MEAN row is +-mu_B-hat (sign of beta_t)
                e = A[:, st["cols"][j]]
                out.append((c, s, fam, int(st["ids"][j]), dirs, e))
    return out


@torch.no_grad()
def s4():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.paper_analysis.edit_direction.common import read_gain
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    name = "google/gemma-2-2b-it"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    emb = model.get_input_embeddings().weight
    d = emb.shape[1]
    sq = d ** 0.5
    pre = tok(PREFIX, add_special_tokens=True)["input_ids"]
    suf = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    toks = s4_tokens()
    g = torch.Generator().manual_seed(0)
    res = []
    t0 = time.time()
    for k, (c, s, fam, t, dirs, e) in enumerate(toks):
        ids = torch.tensor(pre + [t] + suf)
        p = len(pre)
        x0 = emb[ids].clone()                                  # HF rows; the model multiplies by sqrt(d)
        assert torch.allclose(x0[p].double() * sq, e, rtol=1e-3, atol=1e-2), "TL row mismatch"
        rnd = torch.randn(N_RAND, d, generator=g, dtype=torch.float64)
        rnd = rnd / rnd.norm(dim=1, keepdim=True)
        names = list(dirs) + [f"rand{i}" for i in range(N_RAND)]
        U = torch.stack([dirs[n] for n in dirs] + list(rnd))
        batch, meta = [x0], [("orig", 0.0)]
        en = float(e.norm())
        for si in SCALES:
            for n, u in zip(names, U):
                x = x0.clone()
                x[p] = x[p] - (si * en * u / sq).float()
                batch.append(x)
                meta.append((n, si))
        X = torch.stack(batch)
        o = model(inputs_embeds=X, output_hidden_states=True)
        hs = o.hidden_states
        h0, h1 = hs[0].double(), hs[1].double()
        lp = torch.log_softmax(o.logits.double(), dim=-1)
        rg = read_gain(U.float())
        for b, (n, si) in enumerate(meta):
            if b == 0:
                continue
            dh0 = (h0[b, p] - h0[0, p]).norm()
            w0 = ((h1[b, p] - h0[b, p]) - (h1[0, p] - h0[0, p])).norm()
            l1 = (h1[b, p] - h1[0, p]).norm()
            after = ((h1[b, p + 1:] - h1[0, p + 1:]).norm(dim=-1)).mean()
            P0 = lp[0].exp()
            kl = (P0 * (lp[0] - lp[b])).sum(-1)
            i = names.index(n)
            res.append({"concept": c, "seed": s, "family": fam, "token": t, "tok_str": tok.decode([t]),
                        "dir": n, "scale": si, "dh0": float(dh0), "l0_write": float(w0 / dh0),
                        "l1_tot": float(l1 / dh0), "l1_after": float(after / dh0), "kl_tok": float(kl[p]),
                        "kl_after": float(kl[p + 1:].mean()), "proxy_qkv": float(rg["qkv"][i]),
                        "proxy_all": float(rg["all"][i]),
                        "cos_dir_e": float(torch.nn.functional.cosine_similarity(U[i], e, dim=0))})
        print(f"s4 {k + 1}/{len(toks)} {c} {s} {fam} {tok.decode([t])!r} [{time.time() - t0:.0f}s]", flush=True)
    save(OUT / "s4_forward.json", {"scales": SCALES, "n_rand": N_RAND, "rows": res})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("s2", "s4"))
    a = ap.parse_args()
    {"s2": s2, "s4": s4}[a.what]()


if __name__ == "__main__":
    main()
