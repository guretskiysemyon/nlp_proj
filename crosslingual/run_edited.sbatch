#!/bin/bash
# Edited-model cross-lingual MC runs (crosslingual/run_edited.py).
# Arguments are forwarded to the python script; with none, it runs all
# methods (ember, snmf, snmf_ember) x 18 concepts x 5 languages. Finished (method,
# concept) pairs are skipped, so a killed (preempted) job can simply be
# resubmitted. Needs the pristine baseline (run_baseline.sbatch) first.
#
# Usage:
#   sbatch crosslingual/run_edited.sbatch --methods ember snmf --concepts "Harry Potter"
#   sbatch crosslingual/run_edited.sbatch --methods snmf_ember
#   sbatch crosslingual/run_edited.sbatch
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=cl_edited_%j.out

set -euo pipefail
umask 002   # group-writable outputs, so a teammate can resume the run

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

echo "[run_edited] host=$(hostname) args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$UV" run python -m crosslingual.run_edited "$@"
