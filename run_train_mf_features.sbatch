#!/bin/bash
# Runs ember.train_mf_features (MLP SNMF + embedding Sparse MF tracks).
# All arguments are passed straight through to the Python module.
#
# Pinned to the RTX 2080 nodes (s-004/s-005): the model load defaults to
# bf16, which the Titan nodes cannot run.
#
# The model is loaded in fp32 (~13.5 GiB for gemma-2-2b-it), which does not fit
# on one 11 GB card, so its layers are split evenly across 4 GPUs (--n-devices 4;
# 2 GPUs ran out of memory on the logits). n_devices need not divide n_layers.
#
# Usage:
#   sbatch run_train_mf_features.sbatch \
#       --concepts "World War II" --languages en+ru+de+es+fr
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=train_mf_features_%j.out

set -euo pipefail

# Make everything this job creates group-writable (shared `benohayon` group).
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
# Shared Triton cache: it already holds the compiled cuda_utils.so. Without it
# Triton recompiles that helper on the node and fails there (no Python.h).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/triton}"

UV="${UV:-/home/morg/NLP_2526b/semyong/.local/bin/uv}"

if [[ $# -eq 0 ]]; then
  set -- --concepts "World War II" --languages en+ru+de+es+fr
fi

echo "[run_train_mf_features] args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
# Defaults come first so arguments passed to sbatch override them.
"$UV" run python -m ember.train_mf_features --n-devices 4 "$@"

# sbatch run_train_mf_features.sbatch --concepts "World War II" --languages en+ru+de+es+fr

