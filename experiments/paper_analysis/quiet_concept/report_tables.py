"""Markdown tables for REPORT.md (per-concept appendix table and the step decomposition).

    python -m experiments.paper_analysis.quiet_concept.report_tables
Writes tables/appendix_table.md and tables/decomposition.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.paper_analysis.quiet_concept.common import OUT, concepts


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def main():
    A = pd.read_csv(OUT / "tables" / "feature_rank_by_concept_cond.csv")
    T = pd.read_csv(OUT / "tables" / "spectra_by_concept.csv").set_index("concept")
    P = pd.read_csv(OUT / "frequency" / "per_concept.csv").set_index("concept")
    piv = lambda v: A.pivot(index="concept", columns="cond", values=v)  # noqa: E731
    found, rk, z2, sn = piv("found_rate"), piv("rank_raw_med"), piv("z2_pct_med"), piv("supp_N")
    paper = set(concepts("paper"))
    head = ("| concept | C/N | found raw/⊥μ̂/ctr/wo/CS/GS (of 5) | ρ-rank raw/ctr/wo/CS | ‖z‖² pct raw/ctr/CS | "
            "neutral tok. raw/CS | SVD rank raw/ctr/wo/CS | lex-dir rank raw/⊥μ/wo/CS | seeded purity wo/CS | "
            "seeded value pct wo/CS | g₁ | log f C/N |")
    L = [head, "|" + "---|" * 12]
    for c in concepts("all"):
        t = T.loc[c]
        L.append(
            f"| {c}{'' if c in paper else ' †'} | {int(t.n_C)}/{int(t.n_N)} | "
            + "/".join(fmt(found.loc[c, k] * 5) for k in ("raw", "rawp", "ctr", "wo", "cs", "gs")) + " | "
            + "/".join(fmt(rk.loc[c, k]) for k in ("raw", "ctr", "wo", "cs")) + " | "
            + "/".join(fmt(z2.loc[c, k]) for k in ("raw", "ctr", "cs")) + " | "
            + "/".join(fmt(sn.loc[c, k]) for k in ("raw", "cs")) + " | "
            + "/".join(fmt(t.get(f"svdrank_{k}")) for k in ("raw", "ctr", "wo", "cs")) + " | "
            + "/".join(fmt(t[f"{k}_lex_pca_rank"]) for k in ("raw", "rawp", "wo", "cs")) + " | "
            + "/".join(fmt(t[f"{k}_seeded_purity_C"], 2) for k in ("wo", "cs")) + " | "
            + "/".join(fmt(t[f"{k}_seeded_value_pct_{k}"]) for k in ("wo", "cs")) + " | "
            + f"{t.g1:.1f} | {P.loc[c, 'lf_C']:.2f}/{P.loc[c, 'lf_N']:.2f} |")
    (OUT / "tables" / "appendix_table.md").write_text("\n".join(L) + "\n")
    D = pd.read_csv(OUT / "tables" / "decomposition.csv").set_index("matrix")
    names = {"raw": "raw (EMBER)", "rawp": "raw ⊥μ̂_B", "ctr": "centred", "wo": "whiten only", "cs": "CS",
             "gs": "CS+GS"}
    rows = [("runs with a lexicon concept feature (%)", "runs_with_concept_feature", 100, 0),
            ("runs where it ranks 1st by ρ (%)", "runs_rank1", 100, 0),
            ("ρ rank of the max-lexicon feature", "rho_rank_of_max_lexicon_feature", 1, 1),
            ("neutral-only tokens in it", "neutral_tokens_in_feature", 1, 1),
            ("its ‖z‖² percentile", "z2_percentile", 1, 0),
            ("lexicon direction: PCA rank", "lex_dir_pca_rank", 1, 1),
            ("lexicon direction: energy / average direction", "lex_dir_x_avg", 1, 1),
            ("lexicon direction: concept-only share of top-m", "lex_dir_topm_purity", 1, 2),
            ("seeded feature: purity", "seeded_purity", 1, 2),
            ("seeded feature: value percentile", "seeded_value_pct", 1, 0),
            ("spectrum participation ratio", "spectrum_participation_ratio", 1, 0),
            ("top PC share of energy", "top1_pc_share", 1, 3),
            ("SVD rank matching sparse-MF error", "svd_matched_rank", 1, 0),
            ("concept/neutral energy per token", "C_over_N_token_energy", 1, 2)]
    cols = [k for k in ("raw", "rawp", "ctr", "wo", "cs", "gs") if k in D.index]
    L = ["| metric (median over 38 concepts) | " + " | ".join(names[k] for k in cols) + " |",
         "|---|" + "---|" * len(cols)]
    for lab, key, mul, nd in rows:
        L.append(f"| {lab} | " + " | ".join(fmt(D.loc[k, key] * mul if pd.notna(D.loc[k, key]) else None, nd)
                                           for k in cols) + " |")
    (OUT / "tables" / "decomposition.md").write_text("\n".join(L) + "\n")
    print((OUT / "tables" / "decomposition.md").read_text())


if __name__ == "__main__":
    main()
