#!/bin/bash
# X11: real erasure of the extrapolation variants on the train questions in 5 languages (x11_extrap_erasure.py).
# One or more concepts per job on one RTX 2080, Gemma loaded once; finished conditions are skipped on resubmit.
#
#   sbatch "Part 2 Experiments/scripts/x11.sbatch" Golf Gun
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/x11_%j.out

set -euo pipefail
umask 002

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"
[[ $# -ge 1 ]] || { echo "usage: sbatch x11.sbatch <concept> [<concept> ...]"; exit 2; }

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

echo "[x11] concepts: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python "Part 2 Experiments/scripts/x11_extrap_erasure.py" run --concepts "$@"
