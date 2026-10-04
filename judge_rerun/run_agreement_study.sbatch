#!/bin/bash
# Qwen-vs-Gemini agreement study on Alpaca coherence (judge_rerun/agreement_study.py).
# Re-judges Gemini's already-scored completions with Qwen, same text, and
# compares item-by-item. Run this BEFORE spending GPU time on the full
# 8-delta x 18-concept grid.
#
# Usage:
#   sbatch judge_rerun/run_agreement_study.sbatch --split train
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:2
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=agreement_study_%j.out

set -euo pipefail

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

echo "[run_agreement_study] args: $*"
"$UV" run python -m judge_rerun.agreement_study "$@"
