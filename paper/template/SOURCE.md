# ACL style files: where they came from

Downloaded 2026-10-22 from the official *ACL style repository:

- Repo: https://github.com/acl-org/acl-style-files
- Commit: `d5adc823ff0f80f98c80405ca0ab66c68e684409` (2026-06-29, "mention template for XeLaTeX in README.md"), the `master` HEAD on the download date
- Raw URLs: `https://raw.githubusercontent.com/acl-org/acl-style-files/d5adc823ff0f80f98c80405ca0ab66c68e684409/<file>`
- The course template (https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj) names this repo as its source.
- ARR (https://aclrollingreview.org/cfp) and EMNLP 2026 (https://2026.emnlp.org/calls/main_conference_papers/) both point to this repo.

| file | last changed upstream | sha256 |
|---|---|---|
| acl.sty | 24272820f6c4 (2025-08-19) | 19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de |
| acl_natbib.bst | 0dabbc62a723 (2026-06-29) | 6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e |
| acl_latex.tex | 0eb94793d53c (2025-08-19) | 339c9ee9705c1767d44ef24b85365c0bb8619ecc3fcf66172bf1356292389614 |
| acl_lualatex.tex | | 0d9987ba833331a996f9abcd1a03eebba7d7ae331145793e7ddb6b5ae8db27b3 |
| custom.bib | | d76ccb30ddceb70c9e1ad0be3f43dfe301ad5765aab2c960fc830ef67bf8232a |
| anthology.bib.txt | | 2b78d2d9aeda62e14c4e46099e8225b5fc116387d8e0a54aad776485e249ceff |
| README.md, formatting.md | | (upstream docs) |

## How these compare with `paper/acl.sty` and `paper/acl_natbib.bst`

Both files are byte-identical to the copies here (same sha256). So `paper/` already uses the current official version, including the June 2026 bst fix for "and others" author lists.

## The venue of the EMBER paper (arXiv:2606.03695)

The paper was accepted at **Findings of EMNLP 2026**:

- arXiv v2 (2026-08-30) has the Comments field "Accepted at EMNLP Findings 2026". The arXiv API returns the same comment. v1 (2026-06-02) had no comment.
- In v1, the main text runs 8 pages, with Limitations at the top of p. 9. That is the ARR submission length. In v2, the Conclusion ends on p. 9 and Limitations starts there, which matches the 9-page camera-ready limit for long papers.
- v2 is A4 and set in Times (NimbusRomNo9L) and Inconsolata, as `acl.sty` sets it up. It has no line numbers and no anonymization, so it was built with the `final` option. It has unnumbered "Limitations" and "Ethical Considerations" sections before the References, as ARR requires.
- The GitHub repo README gives no venue, only an arXiv `@misc` BibTeX. The paper is not yet in the ACL Anthology (EMNLP 2026 runs Oct 24-29, 2026) or on OpenReview. Semantic Scholar was rate-limited.

EMNLP 2026 uses ARR and these style files. Its limits are 8 pages for a long paper at submission and 9 pages camera-ready. The Limitations section is mandatory and does not count toward the limit. Ethical Considerations, references and appendices do not count either.

## Preamble for our paper

```latex
\documentclass[11pt]{article}
\usepackage[final]{acl}      % non-anonymous camera-ready; [review] = anonymous + line numbers; [preprint] = final + page numbers
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}
...
\bibliography{references}    % acl.sty sets \bibliographystyle{acl_natbib} itself
```

## Compile test (tectonic 0.15.0, scratch copies only)

- `acl_latex.tex` (review) builds a 4-page A4 PDF with line numbers. With `[final]` it builds with author names and no line numbers.
- A copy of `paper/` with these style files swapped in compiles, but only after one fix. `main.tex` fails with `LaTeX Error: Command \hm already defined` because `\usepackage{bm}` already defines `\hm`. This happens with the old and new style files alike, since they are identical. In the scratch copy, `\newcommand{\hm}` was changed to `\renewcommand{\hm}`. The cleaner fix is to rename the macro, for example to `\HM`.
- After that fix it builds (draft mode), with these leftovers from work in progress:
  - 28 citation keys missing from `references.bib`
  - 5 `app:*` labels in appendices that are not written yet
  - 9 small overfull boxes
