#!/bin/bash
# CINDER GPU stages that need only Gemma (one RTX 2080, as Part 1's EMBER grid and cross-lingual runs):
#   grid        cinder/grid.py        delta grid with the fixed MC scoring + Alpaca completions of the top 3
#   final_test  cinder/final_test.py  QA_test / SimdomQA_test in 5 languages + MMLU test at the best delta
#
# Usage (one concept per job runs them in parallel):
#   sbatch cinder/run_gemma.sbatch grid --concepts Cannabis
#   sbatch cinder/run_gemma.sbatch final_test --concepts Cannabis Gun
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/cinder_gemma_%j.out

set -euo pipefail
umask 002

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

STAGE="${1:?usage: sbatch cinder/run_gemma.sbatch grid|final_test [args]}"
shift
case "$STAGE" in
  grid|final_test) ;;
  *) echo "unknown stage $STAGE (grid | final_test)"; exit 2 ;;
esac

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

echo "[cinder $STAGE] args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python -m "cinder.$STAGE" "$@"
