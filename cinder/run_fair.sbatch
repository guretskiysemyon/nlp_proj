#!/bin/bash
# Fair EMBER vs CINDER comparison (cinder/fair.py), one method and one or more concepts per job on one RTX 2080
# (Gemma loaded once per job):
# delta grid (MMLU / QA_train / SimdomQA_train, fixed MC scoring, no Alpaca) -> pick -> final test in 5 languages.
# The grid resumes from grid.csv, so a killed job can just be resubmitted.
#
# Usage:
#   sbatch cinder/run_fair.sbatch ember "Golf"
#   sbatch cinder/run_fair.sbatch cinder "Valentine's Day"
#   sbatch cinder/run_fair.sbatch ember_hj Uranium Heroin
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/fair_%j.out

set -euo pipefail
umask 002

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

METHOD="${1:?usage: sbatch cinder/run_fair.sbatch ember|cinder|ember_hj <concept> [<concept> ...]}"
shift
[[ $# -ge 1 ]] || { echo "usage: sbatch cinder/run_fair.sbatch <method> <concept> [<concept> ...]"; exit 2; }

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

echo "[fair] method=$METHOD concepts: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python -m cinder.fair run --method "$METHOD" --concepts "$@"
