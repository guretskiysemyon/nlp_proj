#!/bin/bash
# Pass 1 of the decoupled EMBER delta search (judge_rerun/pass1_ember_delta.py).
# Generates + scores MMLU/QA/SimDom and saves Alpaca completions for ONE
# concept's 8 EMBER deltas, without touching any judge. Submitting as an
# sbatch job (rather than running `uv run` interactively) sidesteps the
# UV_CACHE_DIR / disk-quota issue hit on c-010: these exports send uv's
# cache and the HF cache to the project's own large filesystem instead of
# the near-full ~/.cache on the small one.
#
# Usage:
#   sbatch judge_rerun/run_pass1.sbatch --concept "Harry Potter"
#   sbatch judge_rerun/run_pass1.sbatch --concept "Golf" --overwrite
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=pass1_ember_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

# Everything after the sbatch script path is forwarded here as "$@" and
# passed straight to the python script (e.g. --concept "Golf" --overwrite),
# matching the --concepts "$c" pattern scripts/submit_all.sh already uses.
if [[ $# -eq 0 ]]; then
  set -- --concept "Harry Potter"
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

echo "[run_pass1] args: $*"
"$UV" run python -m judge_rerun.pass1_ember_delta "$@"
