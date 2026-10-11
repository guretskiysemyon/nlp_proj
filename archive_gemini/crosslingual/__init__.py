"""Cross-lingual evaluation of concept erasure.

A self-contained entrypoint that measures whether an erasure edit made on the
English pipeline still holds when the same concept is probed in another
language. Nothing in :mod:`ember` is modified or imported *from* here in the
other direction: this package depends on ``ember``, never the reverse.

The matrix evaluated per concept is::

    config   in {pristine, ember, snmf, snmf_ember}   # EDIT time (language-independent)
    language in {en, de, es, fr, ru}                  # EVAL time
    wrapper  in {english, native}                     # non-English only
    qa_kind  in {mc, open}                            # both, always
    subset   in {QA_test, SimdomQA_test}              # both, always

The edit is applied exactly once per config; only the prompt text and the
grading data change across the inner loops.
"""

__all__ = [
    "data", "prompts", "evaluate", "baseline", "best_hp", "results_io",
]
