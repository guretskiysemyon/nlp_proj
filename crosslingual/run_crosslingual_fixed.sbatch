#!/bin/bash
# Cross-lingual runs with the fixed MC scoring and the repaired picks
# (crosslingual/rerun_fixed.py): regenerate only invalid answers where the pick is
# unchanged, full run where it changed; English MMLU test + edit check. 1 GPU.
#
# Usage:
#   sbatch crosslingual/run_crosslingual_fixed.sbatch --method ember --concepts Uranium Gun
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=cl_fixed_%j.out

set -euo pipefail
umask 002

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

if [[ $# -eq 0 ]]; then
  set -- --method ember --concepts Uranium
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

echo "[run_crosslingual_fixed] args: $*"
"$UV" run python -m crosslingual.rerun_fixed "$@"
