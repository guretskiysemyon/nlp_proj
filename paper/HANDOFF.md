# Paper handoff state (2026-10-23, after commit cc68637d)

## Done
- Paper in `paper/` (ACL = EMBER's EMNLP 2026 Findings template), EMBER-style structure, built with `sh paper/build.sh`.
  Title: "From EMBER to CINDER: Sharpen the Concept, Burn It Once in Eight Languages, Spare the Word".
  Authors: Semyon Guretskiy, Ori Petel, Mary Arav.
- All motivation-doc TODOs answered by WP1–WP9 (reports in `results/paper_analysis/<wp>/REPORT.md`) and integrated.
- Audit (`paper/AUDIT.md`): 12 mismatches + 1 unsupported claim fixed (up to the WP1–WP9 integration).

## Remaining (user asked: do all end to end)
1. **WP6b mean-edit test — DONE**: S1 supported (mean term erases by breaking the prompt; centred edits stay at the
   deletion floor). Integrated: App. E (new paragraphs + `app:mean` subsection + `fig:mean` = `figures/mean_curves.pdf`,
   copied from `results/paper_analysis/mean_edit/fig_mean_curves.pdf`), §5.2 (r = 1 plateau sentence), §7 (mean-only
   sentence), Limitations ("Multilingual evaluation": no collateral term in the ML study; A3z gain may be prompt damage).
   REPORT.md saved from the agent hand-back (hook blocked its write). Dated answers added to CE D16 and CI.
   Commit: `experiments/paper_analysis/mean_edit/*.py,*.sh` + force-add small results (exclude runs/, integrity/, queue/,
   logs/, slurm/).
2. **Experiments fill — DONE**: agent filled App. D, F, G, H, I, J; Figure 3 (`figures/results.pdf`, panel a only) placed
   in §5.2; 2x2 panel (`figures/twobytwo.pdf`) and the efficacy-vs-copy figure (`figures/integrity.pdf`) placed in App. G
   (`fig:twobytwo`, `fig:integrity`); §3 judge sentence fixed (rejections were "not a member" at confidence 0.85, from the
   sharpened RS fit; the threshold played no part). Agent's note: ~540 GPU-h on 3–8 Oct not attributable by job name (App. J).
3. **Trim main text to 8 pages — DONE (2026-10-23)**: main text ends on p. 8 with the draft abstract, the `\pending`
   WP6b lines and the Figure 3 placeholder. Pre-trim text: commit cc68637d (local copy `paper/drafts/sections_before_trim/`, not committed). Table 1 (diagnosis) removed
   (duplicated Fig. 1; text in commit cc68637d); Fig. 4 (read strength) moved to App. E; multilingual 2x2 table
   placed in App. F (`tab:ml-2x2`). Facts cut from the main text: `drafts/moved_to_appendix.md`; all now placed in the appendices.
   §7 efficacy-vs-copy figure goes to the appendix (one-line `\expfill` pointer in §7). Space budget: ~0 lines spare —
   anything added to the main text must be paid for.
4. **Re-audit — DONE**: `paper/AUDIT2.md` (279 rows; 14 MISMATCH, 12 UNSUPPORTED, 44 AMBIGUOUS); all fixes applied,
   see its "Resolution" section.
5. **Abstract — DONE** (`sections/00_abstract.tex`). `\draftfalse` builds cleanly (0 undefined refs, main text ends
   p. 8); the source keeps `\drafttrue` because the AI-disclosure reflection (`sections/10_statements.tex`, two `\note`s)
   must be written by the team. Flip to `\draftfalse` for submission.

## Open questions to the user (unanswered)
- Is the "cheaper judge" the lexicon path (one Gemini call per concept, before judging)? Paper currently says so.
- EMBER section order (Experiments before analysis sections) — kept.

## Conventions
- `\expnum{}` = number from reports (blue in draft); `\pending{}` analysis slot; `\expfill{}` experiment slot; `\note{}`.
- Figures: matplotlib serif, Okabe–Ito palette (EMBER #7F7F7F, CS #0072B2, CINDER #D55E00, CINDER-z #E69F00,
  other #009E73), scripts in `paper/figures/src/`.
- Share SLURM: QOS cap 12 GPUs across our sessions; job names `cl_*`; never cancel others' jobs.
