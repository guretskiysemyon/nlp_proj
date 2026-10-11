#!/bin/bash
# Pristine cross-lingual MC baseline (crosslingual/run_baseline.py).
# Arguments are forwarded to the python script; with none, it runs all
# 18 concepts x 5 languages. Finished cells are skipped, so a killed
# (preempted) job can simply be resubmitted.
#
# Usage:
#   sbatch crosslingual/run_baseline.sbatch --concepts "Harry Potter" --languages en de es fr ru
#   sbatch crosslingual/run_baseline.sbatch
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=cl_baseline_%j.out

set -euo pipefail
umask 002   # group-writable outputs, so a teammate can resume the run

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

echo "[run_baseline] host=$(hostname) args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$UV" run python -m crosslingual.run_baseline "$@"
