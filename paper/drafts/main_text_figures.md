# Main-text figures: files and captions (experiments pass, 2026-10-23)

Both figures are single-column (3.3 in wide), matplotlib, serif (Liberation Serif, the Times clone on this machine),
Okabe–Ito colours with a fixed encoding: EMBER grey; orange = CINDER's mapped-back edit; amber = the original-space
edit (-z); open markers / dashed lines = EMBER's row sparsity instead of GS (the -RS variants). Every number in the
captions below comes from the files named under "Data"; wrap them in `\expnum{}` as elsewhere.

---

## (a) Figure 3, Section 5.3 (`fig:results`)

- PDF: `paper/figures/results.pdf` (PNG preview `paper/figures/results.png`)
- Script: `paper/figures/src/fig_results.py`
- Data: `results/varwidth/layers/collateral/confirm.json` (rules `collateral|H3v_rn`, `paper|H3v_rn`,
  `fixed05|H3v_rn`), `last_mile.json` (`e3`), `twobytwo.json` (`result_fixed05.json`, `result_fixed1.json`)

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/results.pdf}
  \caption{...}
  \label{fig:results}
\end{figure}
```

**Caption.**
\cinder{}'s direction removes more at equal edit size, while at their own operating points the methods tie.
(a)~Paired test $\Hthree$ difference to \ember{} for each \cinder{} variant on the 11 held-out concepts where both
edit (dots: concepts, mean over seeds; marker and bar: mean and 95\% cluster-bootstrap CI; $*$: Holm $p < 0.05$
within a rule), at the point chosen by our rule (collateral, primary), by the paper's rule (no collateral term),
at a fixed edit size $r = 0.5$, and by the integrity-gated rule (copy $\geq 0.9$; post hoc).
Only the fixed edit size separates the methods (\cinder{} \expnum{$+0.15$}, \expnum{10 of 11} concepts, Holm
$p = \expnum{0.012}$); under the paper's rule \cinder{} falls behind (\expnum{$-0.06$}) and \cinderz{} leads
(\expnum{$+0.12$}) by deleting the words.
(b)~Test efficacy of the four direction $\times$ token-set cells (post hoc): at $r = 0.5$, switching to \cinder{}'s
direction adds \expnum{$+0.07$} (\expnum{10 of 11} concepts) and switching to \cinder{}'s tokens \expnum{$+0.02$}, so
the two lines nearly coincide; at $r = 1$ the direction effect vanishes (\expnum{$-0.02$}, \expnum{6}/\expnum{5}).
Colour: mapped-back edit (orange) or original-space edit (-z, amber); open markers: \ember{}'s row sparsity (-RS).

Numbers behind the caption (for the audit):
- fixed r = 0.5, CINDER − EMBER: +0.151 [+0.080, +0.226], 10/1, Holm p 0.012 (`confirm.md`, "Secondary: fixed r = 0.5").
  Holm p < 0.05 at fixed r = 0.5 also for CINDER-RS (0.037) and CINDER-RS-z (0.012); CINDER-z 0.054. No other rule has
  a Holm p < 0.05 (collateral 0.074–0.41; paper rule on H3v 0.055–0.20; integrity-gated 0.42–0.75).
- paper rule, H3v: CINDER −0.061, CINDER-RS −0.089, CINDER-z +0.115, CINDER-RS-z +0.049 (`confirm.md`, "Secondary:
  paper rule, H3v_rn"). Panel (a) plots H3v at the paper-rule point, not the paper's H2.
- 2×2 at r = 0.5: cells ee 0.201, 1b 0.256, cross 0.204, 1a 0.263; Δ_dir +0.073 [+0.012, +0.150], 10/1, Holm 0.097;
  Δ_tok +0.016 (`twobytwo.md`). At r = 1: ee 0.274, 1b 0.267, cross 0.280, 1a 0.289; Δ_dir −0.016, 6/5, p 0.97.
- The 2×2 is PREREG amendment 14; the main text calls it post hoc, as here.

---

## (b) Section 7 figure: efficacy against word integrity

- PDF: `paper/figures/integrity.pdf` (PNG preview `paper/figures/integrity.png`)
- Script: `paper/figures/src/fig_integrity.py` (plotted values saved to `paper/figures/src/integrity_data.json`)
- Data: per-run `results/varwidth/layers/collateral/<Concept>/s<seed>/sub_<variant>/{grid,copygrid}.json`
  (validation efficacy and copy at each grid point; copy was run for r ≤ 2 only, amendment 11),
  `last_mile.json` (`floor`), `nonce_edit.json` (`variants/nm:free`)

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/integrity.pdf}
  \caption{...}
  \label{fig:integrity}
\end{figure}
```

**Caption.**
Removing more costs the words.
Mean efficacy against the share of edited words the model can still copy, along the edit-size grid
$r = 0.05, \dots, 2$ (labelled on \ember{}'s curve), over the held-out runs where each method edits
(\expnum{41}--\expnum{55} of 60; validation questions).
Grey bands: the efficacy of the unedited model when the edited words are replaced by their nonces (nonce floor,
\expnum{0.29}--\expnum{0.36}) or deleted (deletion floor, \expnum{0.23}--\expnum{0.28}) in the test questions, over
the five methods' token sets; dotted line: the integrity gate (copy $\geq 0.9$).
Every method crosses its deletion floor between $r = 0.5$ and $1$, where copy falls from about \expnum{0.8} to
\expnum{0.5}, and none reaches its nonce floor while more than half of its words remain copyable.
Star: the free nonce-matching edit on \cinder{}'s tokens (post hoc; seed 42, \expnum{9} runs, test split), which
keeps \expnum{0.99} of the words copyable at efficacy \expnum{0.18}.

Numbers behind the caption (for the audit; validation means over runs, `integrity_data.json`):
- copy / efficacy at r = 0.5 and 1: EMBER 0.82 / 0.15 and 0.49 / 0.26; CINDER 0.79 / 0.26 and 0.57 / 0.32;
  CINDER-RS 0.80 / 0.22 and 0.56 / 0.30; CINDER-z 0.77 / 0.24 and 0.49 / 0.36; CINDER-RS-z 0.78 / 0.20 and 0.49 / 0.32.
- floors (test, unedited model; `last_mile.md`): nonce 0.29 (EMBER's sets), 0.36 (CINDER, CINDER-z), 0.35 (-RS sets);
  deletion 0.23, 0.28, 0.26.
- The only point at or above its own nonce floor is at r = 2 (EMBER 0.33 vs 0.29, copy 0.11; CINDER-z 0.40 vs 0.36,
  copy 0.12; CINDER-RS-z 0.38 vs 0.35, copy 0.15); CINDER-z at r = 1 is 0.357 vs 0.361 (copy 0.49).
- Caveat stated in the caption: the curves are validation efficacy, the floors and the star are test efficacy
  (the test split is scored only at selected points). The star is seed 42 only, the curves seeds 42–46.
- §7 currently says "App. figure: efficacy vs. copy along r"; if the figure moves to the appendix, the same PDF and
  caption work there.
