#!/bin/bash
# One concept end to end on one local GPU of c-001 (no SLURM): fit -> select -> evaluate, all three approaches.
#   bash experiments/ml/run_local.sh <gpu> "<concept>" [approaches...]
# STAGES="fit select" runs a subset. Logs to results/ml/logs/local_<slug>.log. Each stage skips work already saved, so a rerun resumes.
set -euo pipefail
GPU=$1; CONCEPT=$2; shift 2
APPROACHES=${*:-A1 A2 A3}
ROOT=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml
cd $ROOT
set -a; source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.env; set +a
export HF_HOME=/home/morg/NLP_2526b/oripetel/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=$GPU
export LLAMA_LOCAL=/tmp/claude-21644/-home-morg-NLP-2526b-oripetel-nlp-proj-nlp-proj/c5e716de-a1fa-45a5-9267-5521d11b2fe1/scratchpad/llama
export GEMINI_USAGE_LOG=$ROOT/results/ml/gemini_usage.jsonl GEMINI_BUDGET_NIS=30 GEMINI_PRICE_IN_USD=0.25 GEMINI_PRICE_OUT_USD=2.50
export PYTHONPATH=/home/morg/NLP_2526b/oripetel/.cache/pylib_ml
PY=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.venv/bin/python
SLUG=$(echo "$CONCEPT" | sed -E 's/[^[:alnum:]]+/_/g; s/^_|_$//g')
LOG=$ROOT/results/ml/logs/local_${SLUG}.log
mkdir -p $(dirname $LOG)
{
  echo "=== $(date) host=$(hostname) gpu=$GPU concept=$CONCEPT approaches=$APPROACHES"
  for st in ${STAGES:-fit select evaluate}; do
    case $st in
      fit) $PY -m experiments.ml.fit --concept "$CONCEPT" --approach $APPROACHES --device cuda ;;
      select) $PY -m experiments.ml.select --concept "$CONCEPT" --approach $APPROACHES ;;
      evaluate) $PY -m experiments.ml.evaluate --concept "$CONCEPT" --approach $APPROACHES ;;
    esac
  done
  echo "=== done $(date)"
} >> $LOG 2>&1
