#!/usr/bin/env python
"""Overall comparison over all concepts: fixed 1%, fixed 2%, m1, m3, rn — whitened, no refit, seeds 42-46.

    python -m experiments.varwidth.overall_report

Per (concept, config): judged (F_C after tau + judge) and judge-free (concept direction tracked from the mean fixed-1%
concept direction of that concept) metrics, as in focus_report.py. Writes results/varwidth/overall_report.md with
(1) a summary per config over concepts, (2) one row per concept per metric.
"""
import itertools
import json

import numpy as np
import pandas as pd
import torch

from experiments.varwidth.common import RESULTS, ROOT, VWConfig, slug

SEEDS = range(42, 47)
CONFIG_SPECS = [("fixed 1%", "fixed", {}), ("fixed 2%", "fixed", {"sparsity": 0.02}), ("m1", "m1", {}), ("m3", "m3", {}),
                ("rn", "rn", {})]


def configs(cap: float) -> list:
    """(name, run tag) for the no-refit configs at row cap `cap` (fixed ignores the cap)."""
    return [(name, VWConfig(method=m, refit=False, cap=cap, **kw).tag()) for name, m, kw in CONFIG_SPECS]


CONFIGS = configs(VWConfig().cap)
PURE_N = 5


def load(concept, tag, seed):
    d = RESULTS / slug(concept) / tag / f"seed{seed}"
    if not (d / "summary.json").exists() or not (d / "fit.pt").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    f = torch.load(d / "fit.pt", map_location="cpu", weights_only=False)
    Z = f["Z"].float()
    return {"s": s, "Zn": Z / Z.norm(dim=0, keepdim=True).clamp_min(1e-12), "Y": f["Y"].float(), "rel_err": float(f["rel_err"])}


def top_fc(r):
    s = r["s"]
    if not s["F_C"]:
        return None
    rho = {int(c["feature"]): c["ratio_abs"] for c in s["candidates"]}
    return max(s["F_C"], key=lambda i: rho.get(int(i), -1))


def concept_rows(concept):
    prep = RESULTS / "prep" / f"{slug(concept)}_whitened_S300.pt"
    if not prep.exists():
        return []
    codes = torch.load(prep, weights_only=False)["codes"]
    runs = {name: [r for r in (load(concept, tag, s) for s in SEEDS) if r] for name, tag in CONFIGS}
    ref = [r["Zn"][:, top_fc(r)] for r in runs["fixed 1%"] if top_fc(r) is not None]
    if not ref:  # no fixed-1% run kept the concept: fall back to any judged concept feature of any config
        ref = [r["Zn"][:, top_fc(r)] for rs in runs.values() for r in rs if top_fc(r) is not None]
    u = torch.linalg.svd(torch.stack(ref, 1), full_matrices=False)[0][:, 0] if ref else None
    rows = []
    for name, _ in CONFIGS:
        rs = runs[name]
        row = {"concept": concept, "config": name, "runs": len(rs)}
        if rs:
            row["judge_kept"] = sum(bool(r["s"]["F_C"]) for r in rs)
            row["share"] = np.mean([r["s"]["C"] / r["s"]["n_distinct_tokens"] for r in rs if r["s"]["n_distinct_tokens"]] or [np.nan])
            row["rel_err"] = np.mean([r["rel_err"] for r in rs])
        if rs and u is not None:
            W, N, Z, S = [], [], [], []
            for r in rs:
                i = int((u @ r["Zn"]).abs().argmax())
                sup = torch.nonzero(r["Y"][i]).flatten().numpy()
                W.append(len(sup)); N.append(int((codes[sup] == 0).sum())); Z.append(r["Zn"][:, i]); S.append(set(sup.tolist()))
            cos = [float((a @ b).abs()) for a, b in itertools.combinations(Z, 2)]
            jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(S, 2) if a | b]
            m = float(np.mean(W))
            row.update(width=m, dev=float(np.mean([abs(w - m) for w in W])), wmin=min(W), wmax=max(W),
                       N=float(np.mean(N)), neutral_pct=100 * sum(N) / max(sum(W), 1), pure=sum(n <= PURE_N for n in N),
                       dir_mean=float(np.mean(cos)) if cos else np.nan, dir_worst=min(cos) if cos else np.nan,
                       jaccard=float(np.mean(jac)) if jac else np.nan, core=len(set.intersection(*S)))
        rows.append(row)
    return rows


def md(df: pd.DataFrame) -> list:
    out = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return out


def main():
    import argparse
    global CONFIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=VWConfig().cap, help="row cap of the variable-width configs (default: VWConfig default)")
    cap = ap.parse_args().cap
    CONFIGS = configs(cap)
    concepts = [e["concept"] for e in json.loads((ROOT / "data" / "concept_sentences.json").read_text())]
    df = pd.DataFrame([r for c in concepts for r in concept_rows(c)])
    df.to_csv(RESULTS / ("overall_runs.csv" if cap == VWConfig().cap else f"overall_runs_cap{cap:g}.csv"), index=False)
    names = [n for n, _ in CONFIGS]
    have = df[df["runs"] > 0]
    L = [f"# Overall comparison: all concepts, whitened, no refit (seeds 42–46, row cap {cap:.0%})", "",
         f"Concepts with runs: {have['concept'].nunique()} of {len(concepts)}. Configs: fixed 1% (the paper), fixed 2%, "
         "m1 contribution pooling, m3 entropy widths, rn row-normalized pooling.", "",
         "- **Judge kept**: seeds whose F_C (after τ = 2 + Gemini judge) is non-empty. **Share**: C / distinct F_C tokens.",
         "- Judge-free (concept direction tracked from the mean fixed-1% concept direction of each concept): **width** in "
         "tokens, **typical deviation** = average |seed width − mean| in tokens, **pure** = seeds with ≤ 5 neutral "
         "tokens, **direction worst** = lowest pairwise |cos z| between seeds, **Jaccard** = token-set overlap between seeds.", ""]

    # (1) summary per config over concepts
    g = have.groupby("config")
    summ = pd.DataFrame({
        "Config": names,
        "Concepts": [int(g.get_group(n)["concept"].nunique()) if n in g.groups else 0 for n in names],
        "Judge kept (all seeds)": [f"{int(g.get_group(n)['judge_kept'].sum())}/{int(g.get_group(n)['runs'].sum())}" if n in g.groups else "—" for n in names],
        "Pure (all seeds)": [f"{int(g.get_group(n)['pure'].sum())}/{int(g.get_group(n)['runs'].sum())}" if n in g.groups else "—" for n in names],
        "Mean width": [f"{g.get_group(n)['width'].mean():.0f}" if n in g.groups else "—" for n in names],
        "Width range over concepts": [f"{g.get_group(n)['width'].min():.0f}–{g.get_group(n)['width'].max():.0f}" if n in g.groups else "—" for n in names],
        "Mean typical deviation (tokens)": [f"±{g.get_group(n)['dev'].mean():.1f}" if n in g.groups else "—" for n in names],
        "Mean neutral share": [f"{g.get_group(n)['neutral_pct'].mean():.1f}%" if n in g.groups else "—" for n in names],
        "Mean direction worst": [f"{g.get_group(n)['dir_worst'].mean():.3f}" if n in g.groups else "—" for n in names],
        "Concepts with worst < 0.8": [int((g.get_group(n)['dir_worst'] < 0.8).sum()) if n in g.groups else "—" for n in names],
        "Mean Jaccard": [f"{g.get_group(n)['jaccard'].mean():.3f}" if n in g.groups else "—" for n in names],
        "Mean concept share": [f"{g.get_group(n)['share'].mean():.3f}" if n in g.groups else "—" for n in names],
    })
    L += ["## Summary per method (over concepts)", ""] + md(summ) + [""]

    # (2) per-concept tables
    def pivot(col, fmt):
        t = have.pivot(index="concept", columns="config", values=col).reindex(columns=names)
        t = t.loc[[c for c in concepts if c in t.index]]
        return t.map(lambda v: "—" if pd.isna(v) else fmt(v)).reset_index().rename(columns={"concept": "Concept"})

    sections = [("Concept width, tokens (typical deviation)", None),
                ("Pure seeds (of 5)", ("pure", lambda v: f"{int(v)}/5")),
                ("Judge kept the concept (of 5)", ("judge_kept", lambda v: f"{int(v)}/5")),
                ("Neutral share of the concept feature", ("neutral_pct", lambda v: f"{v:.1f}%")),
                ("Direction worst pair", ("dir_worst", lambda v: f"{v:.2f}")),
                ("Token Jaccard across seeds", ("jaccard", lambda v: f"{v:.2f}"))]
    for title, spec in sections:
        L += [f"## {title}", ""]
        if spec is None:
            t = have.assign(cell=have.apply(lambda r: "—" if pd.isna(r.get("width")) else f"{r['width']:.0f} (±{r['dev']:.0f})", axis=1))
            t = t.pivot(index="concept", columns="config", values="cell").reindex(columns=names).fillna("—")
            t = t.loc[[c for c in concepts if c in t.index]].reset_index().rename(columns={"concept": "Concept"})
        else:
            t = pivot(*spec)
        L += md(t) + [""]
    out = RESULTS / ("overall_report.md" if cap == VWConfig().cap else f"overall_report_cap{cap:g}.md")
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
