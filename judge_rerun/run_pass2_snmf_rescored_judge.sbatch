#!/bin/bash
# SNMF Pass 2 on the re-scored grid (judge_rerun/pass2_snmf_rescored.py --stage judge):
# judges only the top-3 cells that have no Qwen score yet (their first 50 answers, from
# --stage generate), reuses the old Qwen scores for the rest, writes final picks to
# judge_rerun/SNMF/pass2_rescored/. RTX 2080 Ti x3.
#
# Usage:
#   sbatch judge_rerun/run_pass2_snmf_rescored_judge.sbatch --concept "COVID-19 pandemic"
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=pass2_snmf_judge_rescored_%j.out

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
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/uv}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

UV="${UV:-/home/morg/NLP_2526b/semyong/.local/bin/uv}"

echo "[run_pass2_snmf_rescored_judge] args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python -m judge_rerun.pass2_snmf_rescored --stage judge \
    --judge-model "Qwen/Qwen2.5-32B-Instruct" \
    --judge-batch-size 4 \
    --fluency-prompt original \
    "$@"
