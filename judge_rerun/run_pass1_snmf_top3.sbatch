#!/bin/bash
# Runs judge_rerun/pass1_snmf_top3.py: for each concept, reuse the archived
# 144-cell judge-free SNMF grid if it exists, or run it fresh (topk=3, so
# only 3 rows are ever written); then generate (never judge) the first 50
# Alpaca completions for those top-3 HP cells.
#
# Pinned to the RTX 2080 nodes (s-004/s-005), same reason as
# run_pass1.sbatch / run_pass2_delta_search_anygpu.sbatch: the model load
# defaults to bf16, which the Titan nodes (s-002/s-003/s-006, and the
# interactive c-009 Titan X node) cannot run -- confirmed directly with a
# CUBLAS_STATUS_NOT_SUPPORTED error during an interactive smoke test here.
# Only needs 1 GPU: this is the small model-under-erasure (gemma-2-2b-it),
# not the Qwen-32B judge.
#
# Usage (one job, several concepts back-to-back, one model load):
#   sbatch judge_rerun/run_pass1_snmf_top3.sbatch \
#       --concept "Culture of Greece" "Golf" "Heroin" "Nazism" "Pornography" "World War II"
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=pass1_snmf_top3_%j.out

set -euo pipefail

# semyong/maryarav/oripetel all share `benohayon` as their primary group, but
# the default umask (022) creates new dirs/files group-READ-ONLY -- exactly
# what made Culture of Greece's and Golf's earlier EMBER pass2 output dirs
# unwritable by a teammate. umask 002 makes everything this job creates
# group-writable from the start, so anyone sharing that group can resume or
# take over running it.
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
  set -- --concept "Golf"
fi

echo "[run_pass1_snmf_top3] args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python -m judge_rerun.pass1_snmf_top3 "$@"
