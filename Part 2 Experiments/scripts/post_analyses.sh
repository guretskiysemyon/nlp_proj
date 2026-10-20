#!/bin/bash
# CPU analyses rerun after the main jobs (each script skips nothing; all are cheap). Missing scripts are skipped.
cd /home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure
S="Part 2 Experiments/scripts"
for x in x6_strength_curves x7_question_split x8_test_mechanism; do
  if [[ -f "$S/$x.py" ]]; then echo "[post] $x"; /home/morg/NLP_2526b/semyong/.local/bin/uv run python "$S/$x.py" || echo "[post] $x FAILED"; fi
done
