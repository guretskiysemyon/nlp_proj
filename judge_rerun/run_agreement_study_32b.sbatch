#!/bin/bash
# Qwen2.5-32B-Instruct vs Gemini agreement study on Alpaca coherence
# (judge_rerun/agreement_study.py --judge-model). Same text, same 150 items as
# the 14B study (agreement_study_train.json), so the two judges are directly
# comparable. Uses the ORIGINAL fluency prompt: the 32B was already within the
# +-0.15 gate on the pristine baseline without the Qwen-strengthened rewrite.
#
# Time: 300 judge calls at ~28-31 s/call (see judge_baseline_32b_968008.out) is
# ~2.5 h, plus 18 s (warm) to ~1h08 (cold NFS) just to load the weights.
# Per-chunk checkpoints (judge_rerun/agreement_ckpt/) survive a preempt/requeue.
#
# Usage:
#   sbatch judge_rerun/run_agreement_study_32b.sbatch
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=agreement_study_32b_%j.out

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

echo "[run_agreement_study_32b] args: $*"
"$UV" run python -m judge_rerun.agreement_study \
    --split train \
    --judge-model "Qwen/Qwen2.5-32B-Instruct" \
    --judge-batch-size 4 \
    --fluency-prompt original \
    "$@"
