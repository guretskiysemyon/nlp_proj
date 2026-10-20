#!/bin/bash
# CINDER best delta (cinder/pick.py): Qwen2.5-32B judges the first 50 Alpaca completions of each concept's
# top 3 deltas, then Part 1's selection. RTX 2080 Ti x3, as judge_rerun/run_pass2_ember_rescored.sbatch.
#
# Usage:
#   sbatch cinder/run_pick.sbatch --concepts Cannabis Gun Nazism "Republic of Ireland" "Culture of Greece" Golf
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/slurm/cinder_pick_%j.out

set -euo pipefail
umask 002

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-/home/morg/NLP_2526b/semyong/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/uv}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

UV="${UV:-/home/morg/NLP_2526b/semyong/.local/bin/uv}"

echo "[cinder pick] args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python -m cinder.pick "$@"
