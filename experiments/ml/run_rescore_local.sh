#!/bin/bash
# One v2 re-score queue worker on a local c-001 GPU: bash experiments/ml/run_rescore_local.sh <gpu> [--approach ...]
GPU=$1; shift
ROOT=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml
cd $ROOT
set -a; source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.env; set +a
export HF_HOME=/home/morg/NLP_2526b/oripetel/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/morg/NLP_2526b/oripetel/.cache/pylib_ml
export LLAMA_LOCAL=/tmp/claude-21644/-home-morg-NLP-2526b-oripetel-nlp-proj-nlp-proj/c5e716de-a1fa-45a5-9267-5521d11b2fe1/scratchpad/llama
mkdir -p results/ml/logs
/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.venv/bin/python -m experiments.ml.rescore_v2 "$@" >> results/ml/logs/rescore_local_gpu$GPU.log 2>&1
