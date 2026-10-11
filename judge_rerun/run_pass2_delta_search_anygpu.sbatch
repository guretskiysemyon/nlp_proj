#!/bin/bash
# Runs pass2_delta_search.py on the GeForce RTX 2080 Ti nodes (s-004, s-005).
# The filename is historical: this used to request any 3 GPUs, but on the titan
# nodes (s-002/s-003/s-006) the Qwen-32B judge was ~40x slower (TITAN Xp has no
# native bf16; 4-bit compute dtype is bf16), and on s-002 CUDA failed to
# initialize so the job silently ran on CPU. Pinned to rtx_2080 until an fp16
# variant is validated on titan.
#
# Usage (quick validation on a few deltas, on an untried concept):
#   sbatch --time=05:00:00 judge_rerun/run_pass2_delta_search_anygpu.sbatch \
#       --concept "Golf" --deltas 0.5 10.0 200.0
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=pass2_delta_%j.out

set -euo pipefail

# Everything this job creates is group-writable (shared `benohayon` group),
# so a teammate can resume or take over the run.
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
