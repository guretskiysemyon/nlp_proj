#!/bin/bash
# Same as run_pass2_delta_search.sbatch, but without pinning the GPU model.
# run_pass2_delta_search.sbatch requests geforce_rtx_2080 specifically, which
# only exists on 2 of the 5 studentkillable nodes -- when those are busy, jobs
# queue even if other nodes (titan, etc.) are idle. This variant requests any
# 3 GPUs, so it can also land on those other nodes. Test script to confirm
# pass2_delta_search.py actually runs on a different GPU model before relying
# on it.
#
# Usage (quick validation on a few deltas, on an untried concept):
#   sbatch --time=05:00:00 judge_rerun/run_pass2_delta_search_anygpu.sbatch \
#       --concept "Golf" --deltas 0.5 10.0 200.0
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=pass2_delta_%j.out

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

echo "[run_pass2_delta_search_anygpu] args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python -m judge_rerun.pass2_delta_search \
    --judge-model "Qwen/Qwen2.5-32B-Instruct" \
    --judge-batch-size 4 \
    --fluency-prompt original \
    "$@"
