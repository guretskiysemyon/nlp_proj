"""Redirect the evaldata pipeline to results/ml/data/qa_v2 (import this FIRST).

The v1 files under results/ml/data/qa are being read by evaluation workers and must not
change, and common.py / io.py are imported by those workers, so the v2 cleanup patches
the paths at runtime instead of editing common.py:

    import experiments.ml.evaldata.v2  # noqa: F401  (before any other evaldata import)

v1 caches are copied (read-only source) into qa_v2/work by cleanup_v2.py ``targets``.
New candidate engines (generic {"src", "tr"} caches, several "tr" per "src" allowed):
  nllb_nbest    NLLB-200 3.3B, beam 8, 4 returned hypotheses, low / untranslated segments
  madlad_nbest  MADLAD-400 3B, the same
  madlad_ctx    MADLAD option lists "a; b; c; d" split back into per-option rows
"""
from __future__ import annotations

import experiments.ml.evaldata.common as C

QA_V2 = C.REPO / "results" / "ml" / "data" / "qa_v2"
WORK_V1 = C.WORK
WORK_V2 = QA_V2 / "work"
NEW_ENGINES = ("nllb_nbest", "madlad_nbest", "madlad_ctx")

C.WORK = WORK_V2
C.QA_DIR = QA_V2
C.ENGINES = tuple(C.ENGINES) + NEW_ENGINES
