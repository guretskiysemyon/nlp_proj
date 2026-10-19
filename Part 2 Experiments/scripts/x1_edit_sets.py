#!/usr/bin/env python
"""X1: which tokens each method edits, and do the two edits remove the same thing? (CPU, no model)

Per concept: the edited token sets of EMBER and CINDER (exactly as embed_edit selects them), their overlap
(Jaccard), the tokens only one method edits (largest edits first), and the kind of token (whole-word start '▁abc',
word piece, other). For the tokens both edit: cosine between the two edit vectors c_t, and each method's relative
edit size ||c_t|| / ||e_t|| and cos(c_t, e_t) (how much of the token's own direction the edit removes).

Output: Part 2 Experiments/x1_edit_sets/{summary.csv, tokens.csv, shared_vectors.csv, report.md}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2x  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


def kind(piece: str) -> str:
    if p2x.is_word_token(piece):
        return "word"
    if piece.startswith("▁"):
        return "short/other word"
    return "piece"


def main():
    out = p2x.outdir("x1_edit_sets")
    E = p2x.embeddings()
    summ, toks, shared = [], [], []
    for c in p2x.CONCEPTS:
        ed = {}
        for m in p2x.METHODS:
            tids, Cm = p2x.edit(c, m)
            ed[m] = {t: Cm[i] for i, t in enumerate(tids)}
            for t, v in ed[m].items():
                e = E[t]
                piece = p2x.tok_str([t])[0]
                toks.append({"concept": c, "method": m, "token_id": t, "token": piece, "kind": kind(piece),
                             "rel_edit": float(v.norm() / e.norm()),
                             "cos_with_token": float(torch.nn.functional.cosine_similarity(v, e, dim=0))})
        a, b = set(ed["EMBER"]), set(ed["CINDER"])
        both = a & b
        for t in sorted(both):
            va, vb, e = ed["EMBER"][t], ed["CINDER"][t], E[t]
            shared.append({"concept": c, "token": p2x.tok_str([t])[0],
                           "cos_EMBER_CINDER": float(torch.nn.functional.cosine_similarity(va, vb, dim=0)),
                           "rel_EMBER": float(va.norm() / e.norm()), "rel_CINDER": float(vb.norm() / e.norm())})
        sh = pd.DataFrame([r for r in shared if r["concept"] == c])
        summ.append({"concept": c, "EMBER": len(a), "CINDER": len(b), "both": len(both),
                     "only_EMBER": len(a - b), "only_CINDER": len(b - a),
                     "jaccard": round(len(both) / max(len(a | b), 1), 2),
                     "median_cos_shared": round(float(sh.cos_EMBER_CINDER.median()), 2) if len(sh) else None})
    S, T, SH = pd.DataFrame(summ), pd.DataFrame(toks), pd.DataFrame(shared)
    S.to_csv(out / "summary.csv", index=False)
    T.to_csv(out / "tokens.csv", index=False)
    SH.to_csv(out / "shared_vectors.csv", index=False)

    L = ["# X1: edited token sets and edit vectors, EMBER vs CINDER", "",
         "Edit sets as embed_edit applies them (EMBER: Part 1 features; CINDER: L123, seed 42). "
         "`median_cos_shared`: median cosine between the two methods' edit vectors on the tokens both edit.", "",
         p2x.md_table(S), "", "## Kind of edited token (share)", ""]
    kinds = T.groupby(["concept", "method"]).kind.value_counts(normalize=True).unstack(fill_value=0).round(2)
    L += [p2x.md_table(kinds.reset_index()), "", "## Relative edit size and alignment with the token (medians)", ""]
    med = T.groupby(["concept", "method"])[["rel_edit", "cos_with_token"]].median().round(3).reset_index()
    L += [p2x.md_table(med), "", "## Tokens only one method edits (largest edits first, up to 25)", ""]
    for c in p2x.CONCEPTS:
        tc = T[T.concept == c]
        if tc.empty:
            continue
        sets = {m: set(tc[tc.method == m].token_id) for m in p2x.METHODS}
        L.append(f"**{c}**")
        for m, o in (("EMBER", "CINDER"), ("CINDER", "EMBER")):
            only = tc[(tc.method == m) & ~tc.token_id.isin(sets[o])].sort_values("rel_edit", ascending=False)
            L.append(f"- only {m} ({len(only)}): " + ", ".join(f"`{t}`" for t in only.token.head(25)))
        L.append("")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
