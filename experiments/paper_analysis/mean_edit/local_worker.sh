#!/bin/bash
# Queue worker on c-001 GPU 0 (only when nvidia-smi shows it idle).
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj
set -a; source .env; set +a
export HF_HOME=/home/morg/NLP_2526b/oripetel/.cache/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python -m experiments.paper_analysis.mean_edit.queue_worker work
