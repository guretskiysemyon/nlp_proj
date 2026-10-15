#!/bin/bash
# SNMF Pass 1 on the re-scored grid (judge_rerun/pass2_snmf_rescored.py --stage generate):
# first 50 Alpaca answers + loop check for cells new to the top 3. Gemma only, 1 GPU.
#
# Usage:
#   sbatch judge_rerun/run_pass2_snmf_rescored_gen.sbatch --concept "COVID-19 pandemic"
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=pass2_snmf_gen_%j.out

set -euo pipefail
umask 002

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

if [[ $# -eq 0 ]]; then
  set -- --concept Uranium
fi

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

echo "[run_pass2_snmf_rescored_gen] args: $*"
"$UV" run python -m judge_rerun.pass2_snmf_rescored --stage generate "$@"
