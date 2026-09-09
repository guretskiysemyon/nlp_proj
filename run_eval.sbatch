#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=eval_%j.out

cd /home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure

set -a
source .env
set +a

export HF_HOME=/home/morg/NLP_2526b/semyong/.cache/huggingface
export UV_CACHE_DIR=/home/morg/NLP_2526b/semyong/.cache/uv
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/morg/NLP_2526b/semyong/.local/bin/uv run python -m ember.run_eval --config configs/ember_gemma.yaml \
    --concepts "Harry Potter" --train-eval mc
