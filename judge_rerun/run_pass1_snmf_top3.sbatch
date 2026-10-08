#!/bin/bash
# Runs judge_rerun/pass1_snmf_top3.py: per concept, reuse an archived 144-cell
# SNMF grid (if one exists) or run it fresh, narrow to the top-3 HP cells by
# the judge-free `harmonic` score, and generate (never judge) their first 50
# Alpaca completions.
#
# Only needs the small model-under-erasure (gemma-2-2b-it), not the 32B
# judge, so 1 GPU is enough -- but it still needs bf16 tensor-core support,
# so it's pinned to the same geforce_rtx_2080 nodes (s-004/s-005) as the
# Qwen-judge jobs: a Titan X/Xp smoke test on c-009 hit
# "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED" during generation (no native
# bf16 matmul), the same root cause already documented in
# docs/ember_search.md for the titan judge nodes.
#
# Usage:
#   sbatch judge_rerun/run_pass1_snmf_top3.sbatch --concept "Golf"
#   sbatch judge_rerun/run_pass1_snmf_top3.sbatch \
#       --concept "Culture of Greece" "Golf" "Halloween" ...
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=pass1_snmf_top3_%j.out

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

if [[ $# -eq 0 ]]; then
  set -- --concept "Golf"
fi

echo "[run_pass1_snmf_top3] args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python -m judge_rerun.pass1_snmf_top3 "$@"
