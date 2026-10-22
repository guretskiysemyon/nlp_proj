"""Shared paths and constants for WP2 (feature geometry)."""
from __future__ import annotations

from experiments.varwidth.common import RESULTS as VW_RESULTS
from experiments.varwidth.common import ROOT

OUT = ROOT / "results" / "paper_analysis" / "feature_geometry"
SEEDS = tuple(range(42, 47))
CONFIGS = {"L0": "fixed_refit-False_plain", "L1": "fixed_refit-False", "L12": "rn_refit-False"}
NAMES = {"L0": "EMBER (raw, RS-1%)", "L1": "CS + RS-1%", "L12": "CINDER (CS + GS)"}
COLORS = {"L0": "#7F7F7F", "L1": "#0072B2", "L12": "#D55E00", "z": "#E69F00", "other": "#009E73"}
LO_T = 5          # EJ@5: >= 5 of the top-20 source words in the concept lexicon
LO_MIN_CONCEPT = 3  # judge-free concept feature: argmax (lo, rho) over the k features, kept if lo >= this
RUNS = VW_RESULTS
SPACES = ("fit", "raw", "cs", "w", "rawc")


def load_features():
    import pandas as pd
    return pd.read_parquet(OUT / "features.parquet")


def load_pairs(concept: str, cfg: str, seed: int) -> dict:
    import numpy as np

    from experiments.varwidth.common import slug
    with np.load(OUT / "pairs" / f"{slug(concept)}__{cfg}__s{seed}.npz") as z:
        return {k: z[k] for k in z.files}


def full(tri, k: int = 100, diag: float = 1.0):
    """k x k symmetric matrix from an upper triangle (offset 1)."""
    import numpy as np
    M = np.zeros((k, k), dtype=np.float64)
    iu = np.triu_indices(k, 1)
    M[iu] = tri
    M = M + M.T
    np.fill_diagonal(M, diag)
    return M


def concept_feature(g, how: str = "lex"):
    """Judge-free concept feature of one run (rows of features.parquet for one (concept, cfg, seed)).
    lex: argmax (lo, rho) if lo >= LO_MIN_CONCEPT; tj: the F_C feature (summary.json) with the largest lo, rho;
    p: hand label P with the largest lo, rho. Returns the feature id or None."""
    if how == "lex":
        r = g.sort_values(["lo", "rho"], ascending=False).iloc[0]
        return int(r.feature) if r.lo >= LO_MIN_CONCEPT else None
    sub = g[g.in_fc] if how == "tj" else g[g.label == "P"]
    if sub.empty:
        return None
    return int(sub.sort_values(["lo", "rho"], ascending=False).iloc[0].feature)


def paired(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison over concepts (b - a): mean diff, bootstrap 95% CI, Wilcoxon signed-rank p, n."""
    import numpy as np
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = b[m] - a[m]
    if len(d) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p = float(wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else float("nan")
    return {"n": int(len(d)), "mean_a": float(a[m].mean()), "mean_b": float(b[m].mean()), "diff": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "wilcoxon_p": p,
            "n_b_gt_a": int((d > 0).sum())}


def boot_ci(x, n_boot: int = 10000, seed: int = 0) -> list:
    import numpy as np
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    return plt


def savefig(fig, name: str):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)


def text_emb_path():
    """MiniLM embeddings (embed_text.py): the local run's file, else the SLURM run's (same content)."""
    for n in ("text_emb.npz", "text_emb_slurm.npz"):
        if (OUT / n).exists():
            return OUT / n
    raise FileNotFoundError(OUT / "text_emb.npz")
