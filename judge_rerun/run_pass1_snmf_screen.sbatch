#!/bin/bash
# Runs judge_rerun/pass1_snmf_screen.py: for each concept, walk its full
# 144-cell SNMF grid by the judge-free harmonic, generating the first 50
# Alpaca answers per cell, skipping duplicates (identical answers) and cells
# whose answers loop ("Roman Roman Roman ..."), and stop at the first 3
# distinct non-looping cells, which go to Pass 2
# (judge_rerun/run_pass2_snmf.sbatch). Used for Ancient Rome only, whose plain
# top-3 cells all broke generation -- see docs/Part 1/snmf_search.md.
#
# Same node pinning as run_pass1_snmf_top3.sbatch: RTX 2080 only, because the
# model loads in bf16, which the Titan nodes cannot run. 1 GPU (gemma-2-2b-it
# only, no judge). ~3 min per grid row generated, plus the model load.
#
# Usage:
#   sbatch judge_rerun/run_pass1_snmf_screen.sbatch --concept "Ancient Rome"
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=pass1_snmf_screen_%j.out

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

if [[ $# -eq 0 ]]; then
  echo "usage: sbatch judge_rerun/run_pass1_snmf_screen.sbatch --concept \"<Concept>\" ..." >&2
  exit 2
fi

echo "[run_pass1_snmf_screen] args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python -m judge_rerun.pass1_snmf_screen "$@"
