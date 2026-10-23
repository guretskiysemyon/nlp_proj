#!/bin/bash
# qa_new_tr.sh on one local c-001 GPU, models from the fast local HF cache.
#   bash experiments/ml/evaldata/qa_new_tr_local.sh <gpu> <langs...>
set -euo pipefail
GPU=$1; shift   # nvidia-smi index or GPU UUID; on c-001 CUDA skips the broken GPU 1, so pass the UUID
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml
set -a; source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.env; set +a
export HF_HOME=${LOCAL_HF:-/tmp/claude-21644/-home-morg-NLP-2526b-oripetel-nlp-proj-nlp-proj/c5e716de-a1fa-45a5-9267-5521d11b2fe1/scratchpad/hf}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/home/morg/NLP_2526b/oripetel/.cache/pylib_ml
umask 002
bash experiments/ml/evaldata/qa_new_tr.sh "$@"
