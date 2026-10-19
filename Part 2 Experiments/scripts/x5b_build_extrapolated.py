#!/usr/bin/env python
"""X5b: extrapolated edits "EMBER+X" and "CINDER+X" in EMBER's file format, for the G1 leak-curve jobs (CPU).

Same features, same English edit: every token of the original V* keeps its original coefficients (G rows). Added:
tokens of the native de/es/fr/ru concept and neutral sentences (data_v2) that are not in V*, with coefficients from
X5's extrapolation (Z fixed, dense ridge solve, the method's sparsity rule; budget over the whole union set, as Ori's
extrapolate()). Labels for added tokens: concept-only = in some language's concept sentences and never in any neutral
set of the five languages. The edit (embed_edit) then also reaches native foreign tokens loading on the concept
feature. Written to mf_outputs/extrap_<method>/ (git-ignored).

Output: Part 2 Experiments/x5_extrapolation/built.csv (edited tokens per method, with the language coverage)
"""
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pickle  # noqa: E402
import torch  # noqa: E402
from x5_extrapolation import extrapolate, fit_pieces  # noqa: E402

C = p2x.C
ALL = ("en", "de", "es", "fr", "ru")


def main():
    import edit_coverage as ec
    from ember.erasure import features
    from ember.utils import collect_feature_rows_for_embeddings, get_pipeline_path
    from experiments.varwidth.prep import TokenizerOnly, get_tokenizer
    ec.TOK = p2x.tokenizer()
    E = p2x.embeddings()
    special = set(p2x.tokenizer().all_special_ids)
    rows = []
    for c in p2x.CONCEPTS:
        en_codes, _ = ec.labels(c, ("en",), str(p2x.DATA_V2))
        all_codes, _ = ec.labels(c, ALL, str(p2x.DATA_V2))
        for m in ("EMBER", "CINDER"):
            pieces = fit_pieces(c, m)
            if pieces is None:
                continue
            Z, transform, sel = pieces
            features.MF_OUTPUTS_ROOT = p2x.METHODS[m]
            pkl, pot = features._embedding_paths(C.MODEL_NAME, c, C.RANK, C.SEED)
            F0, G0, v0 = features.load_embedding_payload(pkl)
            v0set = set(v0)
            new = sorted(t for t in all_codes if t not in special and t not in v0set)
            V = list(v0) + new
            Y = extrapolate(Z, transform(E[torch.tensor(V)]), m)          # (k, |V|), budget over the union
            G = torch.cat([torch.as_tensor(G0).float(), Y[:, len(v0):].T.contiguous()], 0)
            label = {t: ("both" if code == 2 else c if code == 1 else "Neutral") for t, code in en_codes.items()}
            label.update({t: ("both" if all_codes[t] == 2 else c if all_codes[t] == 1 else "Neutral") for t in new})
            root = p2x.ROOT / "mf_outputs" / f"extrap_{m}"

            def path(kind, fname):
                return get_pipeline_path(str(root), C.MODEL_SAFE, kind, C.RANK, C.safe(c), "embedding", fname, seed=C.SEED)

            with open(path("pickles", "embedding.pkl"), "wb") as fh:
                pickle.dump({"nmf": SimpleNamespace(F_=torch.as_tensor(F0).float(), G_=G), "vprime_token_ids": V,
                             "source": f"{m}+X: original V* rows + extrapolated native de/es/fr/ru rows"}, fh)
            fr = collect_feature_rows_for_embeddings(G, V, label, TokenizerOnly(get_tokenizer()), C.MODEL_SAFE, c,
                                                     C.RANK, max_activating=100000)
            pd.DataFrame(fr).to_csv(path("csvs", "token_features.csv"), index=False)
            shutil.copy2(pot, path("interpretations", "potential_features.csv"))
            p2x.edit.cache_clear()
            tids, _ = __import__("cinder.edit_size", fromlist=["edit_vectors"]).edit_vectors(c, root)
            Ex, E0 = set(tids), set(p2x.edit(c, m)[0])
            r = {"concept": c, "method": f"{m}+X", "n_edit_original": len(E0), "n_edit": len(Ex),
                 "original_kept": len(E0 & Ex) == len(E0), "n_new_tokens_in_V": len(new)}
            for L in ALL:
                r[f"q_cov_{L}"] = round(float(np.mean([bool((q | a) & Ex) for q, a in ec.questions(c, L)])), 2)
            rows.append(r)
            print(r, flush=True)
    out = p2x.outdir("x5_extrapolation")
    pd.DataFrame(rows).to_csv(out / "built.csv", index=False)
    print(p2x.md_table(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()
