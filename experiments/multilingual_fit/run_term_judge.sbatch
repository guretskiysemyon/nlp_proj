#!/bin/bash
# Runs experiments/multilingual_fit/term_judge.py: Qwen2.5-32B (4-bit, judge_rerun/qwen_judge.py)
# decides whether each term mismatch from term_check.py is a real translation error.
# Step 4 of the translation plan in docs/Part 2/multilingual_features.md. Same GPU setup
# as judge_rerun/run_pass2_snmf.sbatch (3x RTX 2080 Ti). Resumable (checkpoint per batch).
#
# Usage:
#   sbatch experiments/multilingual_fit/run_term_judge.sbatch [--limit 32]
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=term_judge_%j.out

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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/triton}"

UV="${UV:-/home/morg/NLP_2526b/semyong/.local/bin/uv}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python experiments/multilingual_fit/term_judge.py --backend qwen "$@"
