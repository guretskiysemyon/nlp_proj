#!/bin/bash
# Titan Xp slowness diagnosis (judge_rerun/titan_diag.py). Usage: sbatch ... [--load]
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:titan:3
#SBATCH --exclude=s-002
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=titan_diag2_%j.out
set -euo pipefail
cd /home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure
export HF_HOME=/home/morg/NLP_2526b/semyong/.cache/huggingface
export UV_CACHE_DIR=/home/morg/NLP_2526b/semyong/.cache/uv
export HF_HUB_DISABLE_XET=1
echo "node=${SLURMD_NODENAME:-?} CVD=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi --query-gpu=index,name,memory.used --format=csv
/home/morg/NLP_2526b/semyong/.local/bin/uv run python -m judge_rerun.titan_diag2 "$@"
