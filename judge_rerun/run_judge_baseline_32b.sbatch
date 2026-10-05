#!/bin/bash
# Judge the pristine-model baseline's Alpaca completions with Qwen2.5-32B-
# Instruct (judge_rerun/pass2_judge_baseline.py --judge-model). 3 GPUs:
# 32B in 4-bit NF4 is ~16-18GB of weights alone, so 2x11GB would leave
# almost no headroom for batched KV cache/activations (see judge_rerun/
# qwen_judge.py's adaptive batch-halving, which exists for exactly this).
#
# Usage:
#   sbatch judge_rerun/run_judge_baseline_32b.sbatch
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=judge_baseline_32b_%j.out

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

echo "[run_judge_baseline_32b] args: $*"
"$UV" run python -m judge_rerun.pass2_judge_baseline \
    --judge-model "Qwen/Qwen2.5-32B-Instruct" \
    --judge-batch-size 4 \
    --fluency-prompt original \
    "$@"
