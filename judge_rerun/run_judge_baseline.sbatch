#!/bin/bash
# Judge the pristine-model baseline's Alpaca completions with Qwen
# (judge_rerun/pass2_judge_baseline.py). 300 judge calls (150 instruct +
# 150 fluency), batched. Requires judge_rerun/pass1_baseline.py to have
# already run.
#
# Usage:
#   sbatch judge_rerun/run_judge_baseline.sbatch
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:2
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=judge_baseline_%j.out

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

echo "[run_judge_baseline] args: $*"
"$UV" run python -m judge_rerun.pass2_judge_baseline "$@"
