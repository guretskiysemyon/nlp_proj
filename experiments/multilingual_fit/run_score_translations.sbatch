#!/bin/bash
# Runs experiments/multilingual_fit/score_translations.py (COMET-Kiwi + LaBSE scores for
# every translated sentence; step 2 of the translation plan in
# docs/Part 2/multilingual_features.md). One GPU is enough (~2.3 GB model, fp32).
#
# Uses the separate environment /home/morg/NLP_2526b/semyong/.venv-comet, not the
# project .venv: unbabel-comet pins protobuf < 5, which would break the Gemini SDK there.
#
# Usage:
#   sbatch experiments/multilingual_fit/run_score_translations.sbatch
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=score_translations_%j.out

set -euo pipefail
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
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PY="${PY:-/home/morg/NLP_2526b/semyong/.venv-comet/bin/python}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PY" -I experiments/multilingual_fit/score_translations.py "$@"
