#!/bin/bash
# Translation of the new concepts' MC items for some languages, all GPU/CPU steps in order (resumable: every step
# skips what its cache already has). Environment (HF_HOME with NLLB / MADLAD / LaBSE, CUDA_VISIBLE_DEVICES) is set by
# the caller: qa_new_tr.sbatch (SLURM) or qa_new_tr_local.sh (c-001).
#   bash experiments/ml/evaldata/qa_new_steps.sh de fr it pt
# STEPS (default "gen1 rest"): gen1 = NLLB + option lists (GEN1_PHASES="nllb ctx" by default); rest = LaBSE, targets,
# NLLB/MADLAD n-best + MADLAD lists, LaBSE. WAIT_FOR=<file>: wait until that file exists before "rest"
# (used when another process translates the option lists of the same language). DONE_MARK=<file>: touched at the end.
set -euo pipefail
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml
PY=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.venv/bin/python
COMET_PY=/home/morg/NLP_2526b/semyong/.venv-comet/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L="$*"
STEPS=${STEPS:-gen1 rest}
echo "=== $(date) host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} langs=$L steps=$STEPS"
if [[ " $STEPS " == *" gen1 "* ]]; then
  $PY -m experiments.ml.evaldata.qa_new_translate gen1 --langs $L --tok-budget 4096 --max-bs 192 --phases ${GEN1_PHASES:-nllb ctx}
fi
if [[ " $STEPS " == *" rest "* ]]; then
  if [ -n "${WAIT_FOR:-}" ]; then until [ -e "$WAIT_FOR" ]; do sleep 30; done; echo "found $WAIT_FOR $(date)"; fi
  $COMET_PY -I experiments/ml/evaldata/qa_new_score.py --engine nllb ctx --langs $L
  $PY -m experiments.ml.evaldata.qa_new_translate targets --langs $L
  $PY -m experiments.ml.evaldata.qa_new_translate gen2 --langs $L
  $COMET_PY -I experiments/ml/evaldata/qa_new_score.py --engine nllb_nbest madlad_nbest madlad_ctx --langs $L
fi
[ -n "${DONE_MARK:-}" ] && touch "$DONE_MARK"
echo "=== ALL DONE $(date)"
