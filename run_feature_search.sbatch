#!/bin/bash
# Feature search for one concept: ember.train_mf_features (matrix factorization, MLP SNMF +
# embedding Sparse MF tracks), then ember.interpret_features (Gemini filter ->
# potential_features.csv). Same setup as run_train_mf_features.sbatch (4x RTX 2080, fp32
# model split across the GPUs).
#
# Usage:
#   sbatch run_feature_search.sbatch OUTDIR LANGS [DATA_DIR] [CONCEPT]
# e.g.
#   sbatch run_feature_search.sbatch mf_outputs/native en+ru+de+es+fr experiments/multilingual_fit/data_v2
#   sbatch run_feature_search.sbatch mf_outputs/en_rerun en
# Use a fresh OUTDIR per data set: train_mf_features skips a track whose output exists.
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/feature_search_%j.out

set -euo pipefail
umask 002

OUTDIR="$1"; LANGS="$2"; DATA_DIR="${3:-experiments/multilingual_fit/data_v2}"; CONCEPT="${4:-World War II}"

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export HF_HOME="${HF_HOME:-/home/morg/NLP_2526b/semyong/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/uv}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/triton}"
UV="${UV:-/home/morg/NLP_2526b/semyong/.local/bin/uv}"

# multilingual runs write under OUTDIR/multi_language/<langs>/ (see train_mf_features.py)
if [[ "$LANGS" == "en" ]]; then INTERP_OUT="$OUTDIR"; else INTERP_OUT="$OUTDIR/multi_language/$LANGS"; fi

echo "[feature_search] concept='$CONCEPT' langs=$LANGS data=$DATA_DIR outdir=$OUTDIR"
echo "[feature_search] git $(git rev-parse --short HEAD), uncommitted: $(git status --short | wc -l) files"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$UV" run python -m ember.train_mf_features --n-devices 4 --concepts "$CONCEPT" \
    --languages "$LANGS" --multilingual-data-dir "$DATA_DIR" --outdir "$OUTDIR"

echo "[feature_search] factorization done; Gemini filter -> $INTERP_OUT"
"$UV" run python -m ember.interpret_features --concepts "$CONCEPT" --outdir "$INTERP_OUT"
