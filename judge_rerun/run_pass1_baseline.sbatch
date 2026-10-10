#!/bin/bash
# Pristine-model baseline for the decoupled EMBER delta search
# (judge_rerun/pass1_baseline.py). No embedding edit applied -- computes
# MMLU + the 150 Alpaca-train completions once, then QA/SimDom MC accuracy
# for all 18 concepts. No judge involved; Pass 2 scores the saved Alpaca
# completions afterward.
#
# Usage:
#   sbatch judge_rerun/run_pass1_baseline.sbatch
#   sbatch judge_rerun/run_pass1_baseline.sbatch --overwrite
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=pass1_baseline_%j.out

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

echo "[run_pass1_baseline] args: $*"
"$UV" run python -m judge_rerun.pass1_baseline "$@"
