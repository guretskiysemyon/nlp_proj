## 4 Metrics

**Accuracy and normalization.** We use four-option multiple-choice (MC) accuracy, scored with greedy decoding and no judge, so the
score is identical in every language. Unparseable answers count as wrong. As in Suslik et al. (2026), accuracies are normalized
against the unedited model and corrected for the 0.25 chance level:

$$\phi(a;a_0)=\operatorname{clip}\!\left(\frac{a-0.25}{a_0-0.25},\,0,\,1\right),$$

where $a$ is the edited and $a_0$ the unedited accuracy. A value of 1 means unchanged and 0 means chance.

**Selection metrics (English).** We follow the EMBER paper without change, except for the judge:

- efficacy $=1-\phi(\text{QA}_{\text{train}})$;
- specificity $=\operatorname{HM}\big(\phi(\text{MMLU}),\,\phi(\text{SimDom}_{\text{train}})\big)$, which covers general and neighbouring-topic damage;
- coherence $=\operatorname{HM}$ of the AlpacaEval instruction-following and fluency scores, each divided by the unedited model's score and capped at 1;
- selection score $H=\operatorname{HM}(\text{efficacy},\text{specificity},\text{coherence})$.

The harmonic mean ($\operatorname{HM}$) is dominated by its smallest term, so an edit cannot score well by breaking one axis. We use the paper's gates
(MMLU fraction $\ge 0.9$; for EMBER also both judge fractions $\ge 0.9$). Because Gemini is paid, coherence is scored by
Qwen2.5-32B-Instruct instead (Appendix B). The first selection pass ranks cells by the judge-free $\operatorname{HM}(\text{efficacy},\text{specificity})$ and the second pass judges only the top three.

**Retention and leak (cross-lingual).** For language $L$, on the held-out $\text{QA}_{\text{test}}$,

$$r_L=\phi\big(\mathrm{acc}_L;\ \mathrm{acc}^{\text{unedited}}_L\big),\qquad \Delta_L=r_L-r_{\text{en}}.$$

Lower $r_L$ means stronger erasure, and $\Delta_L>0$ means the concept survives better in $L$ than in English. The paper has no
cross-lingual measure, so both are ours. We normalize by the unedited accuracy *in the same language* because Gemma knows less
in Russian (0.52) than in English (0.73), and only the per-language ratio separates "erased" from "never known". We report
$\Delta_L$ as a difference rather than a ratio because $r_{\text{en}}$ can be near 0. A concept–language pair is dropped when the unedited
accuracy is below 0.40 (2 of 90 pairs), since with 50 questions the denominator is then mostly noise. The mean leak averages a
concept's reliable languages first, then concepts.

**Collateral damage.** $\phi$ on $\text{SimDom}_{\text{test}}$ in each language, and English MMLU accuracy on the 1,000-question test split. MMLU is not translated, so
specificity in a language is in-language SimDom plus English MMLU.

**Invalid answers.** We report the share of unparseable answers per language next to retention, because a broken answer is scored
as wrong and so looks like erasure (Section 5.4).

**Statistics.** The unit of analysis is the concept. We report 95% bootstrap intervals over concepts (10,000 resamples) and compare
methods by paired per-concept differences with a bootstrap interval and a paired $t$-test.
