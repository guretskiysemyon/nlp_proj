#!/bin/bash
# Translation of the new concepts' MC items for some languages, all GPU/CPU steps in order (resumable: every step
# skips what its cache already has). Environment (HF_HOME with NLLB / MADLAD / LaBSE, CUDA_VISIBLE_DEVICES) is set by
# the caller: qa_new_tr.sbatch (SLURM) or qa_new_tr_local.sh (c-001).
#   bash experiments/ml/evaldata/qa_new_tr.sh de fr it pt
set -euo pipefail
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml
PY=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.venv/bin/python
COMET_PY=/home/morg/NLP_2526b/semyong/.venv-comet/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L="$*"
echo "=== $(date) host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} langs=$L"
$PY -m experiments.ml.evaldata.qa_new_translate gen1 --langs $L --tok-budget 4096 --max-bs 192
$COMET_PY -I experiments/ml/evaldata/qa_new_score.py --engine nllb ctx --langs $L
$PY -m experiments.ml.evaldata.qa_new_translate targets --langs $L
$PY -m experiments.ml.evaldata.qa_new_translate gen2 --langs $L
$COMET_PY -I experiments/ml/evaldata/qa_new_score.py --engine nllb_nbest madlad_nbest madlad_ctx --langs $L
echo "=== ALL DONE $(date)"
