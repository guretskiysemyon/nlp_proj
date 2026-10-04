#!/bin/bash
# Pass 2 of the EMBER delta search (judge_rerun/pass2_delta_search.py): judge
# one concept's saved Alpaca completions with Qwen2.5-32B and pick the best
# delta. 3 GPUs, since 32B in 4-bit NF4 is ~16-18GB of weights alone.
#
# Budget: ~300 judge calls per delta at ~32s/call ~= 1.2h/delta, so all 8
# deltas is ~21h (hence the 24h default). pass2_delta_search.py checkpoints
# after every delta, so a time-limit kill or a preemption only loses the delta
# in flight -- just resubmit the identical command to continue.
#
# Usage:
#   # quick end-to-end validation on a few deltas (override the time limit):
#   sbatch --time=05:00:00 judge_rerun/run_pass2_delta_search.sbatch \
#       --concept "Harry Potter" --deltas 0.5 10.0 200.0
#   # then the full grid; already-scored deltas are skipped:
#   sbatch judge_rerun/run_pass2_delta_search.sbatch --concept "Harry Potter"
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
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
  set -- --concept "Harry Potter"
fi

echo "[run_pass2_delta_search] args: $*"
"$UV" run python -m judge_rerun.pass2_delta_search \
    --judge-model "Qwen/Qwen2.5-32B-Instruct" \
    --judge-batch-size 4 \
    --fluency-prompt original \
    "$@"
