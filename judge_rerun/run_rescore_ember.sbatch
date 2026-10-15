#!/bin/bash
# Re-score the EMBER delta grid's MC/MMLU with the fixed evaluator
# (judge_rerun/rescore_ember_grid.py, judge_rerun/mc_fixed.py): 32-token budget,
# fixed letter parser, per-question records saved. Gemma only, no judge.
#
# Usage:
#   sbatch judge_rerun/run_rescore_ember.sbatch --concept Uranium --all-deltas
#   sbatch judge_rerun/run_rescore_ember.sbatch --concept "Harry Potter" Golf
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=rescore_ember_%j.out

set -euo pipefail
umask 002

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

if [[ $# -eq 0 ]]; then
  set -- --concept Uranium --all-deltas
fi

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

echo "[run_rescore_ember] args: $*"
"$UV" run python -m judge_rerun.rescore_ember_grid "$@"
