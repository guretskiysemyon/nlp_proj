#!/bin/bash
# Titan Xp slowness diagnosis (judge_rerun/titan_diag.py). Usage: sbatch ... [--load]
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=rtx_diag4_%j.out
set -euo pipefail
cd /home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure
export HF_HOME=/home/morg/NLP_2526b/semyong/.cache/huggingface
export UV_CACHE_DIR=/home/morg/NLP_2526b/semyong/.cache/uv
export HF_HUB_DISABLE_XET=1
echo "node=${SLURMD_NODENAME:-?} CVD=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi --query-gpu=index,name,memory.used --format=csv
/home/morg/NLP_2526b/semyong/.local/bin/uv run python -m judge_rerun.rtx_diag4 "$@"
