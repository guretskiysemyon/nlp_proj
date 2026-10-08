#!/bin/bash
# Benchmark the Qwen-32B judge on TITAN Xp nodes (judge_rerun/titan_judge_bench.py):
# GEMM speed per dtype, then judge 8 Harry Potter delta-1.0 completions under
# fp32/fp16/bf16 compute and compare speed + scores to the RTX 2080 Ti run.
#
# s-002 is excluded: CUDA failed to initialize there (jobs 981331/981347 ran
# on CPU). The first run quantizes from the 62 GB bf16 checkpoint on NFS (slow,
# 2-4 h) and saves an ~18 GB NF4 copy to QUANT_DIR; later runs load that.
#
# Usage:
#   sbatch judge_rerun/run_titan_judge_bench.sbatch
#   sbatch judge_rerun/run_titan_judge_bench.sbatch --compute-dtypes fp32 --batch-sizes 1 4 8
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:titan:3
#SBATCH --exclude=s-002
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=titan_bench_%j.out

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
QUANT_DIR="${QUANT_DIR:-/home/morg/NLP_2526b/semyong/.cache/qwen2.5-32b-instruct-nf4}"

echo "[run_titan_judge_bench] node=${SLURMD_NODENAME:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} args: $*"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
"$UV" run python -m judge_rerun.titan_judge_bench \
    --judge-model "Qwen/Qwen2.5-32B-Instruct" \
    --quantized-dir "$QUANT_DIR" \
    "$@"
