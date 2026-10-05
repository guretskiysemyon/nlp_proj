#!/bin/bash
# Smoke test for judge_rerun/smoke_gemma_judge.py -- does google/gemma-4-12B-it
# in 4-bit fit on an 11 GB RTX 2080 Ti, follow the project's own Alpaca
# judge prompts, and produce parseable "Rating: [[n]]" scores.
#
# Usage:
#   sbatch judge_rerun/run_smoke_gemma.sbatch
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=smoke_gemma_%j.out

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
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# google/gemma-4-12B-it is an omni (text+image+audio+video) checkpoint whose
# tokenizer config needs transformers>=5.0 -- the project's main .venv is
# pinned to transformers==4.53.3 for everything else, so this uses a separate
# throwaway venv (judge_rerun/.venv_gemma4, not part of uv.lock) instead of
# "uv run". See docs/last_step.md for how it was built.
GEMMA4_PY="${GEMMA4_PY:-$REPO_DIR/judge_rerun/.venv_gemma4/bin/python}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$GEMMA4_PY" -m judge_rerun.smoke_gemma_judge
