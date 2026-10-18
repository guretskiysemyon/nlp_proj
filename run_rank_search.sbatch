#!/bin/bash
# Rank search for the EMBER (embedding) track of a multilingual fit. For every rank:
# ember.train_mf_features --skip-mlp (embedding Sparse MF only), then ember.interpret_features
# --tracks embedding (Gemini filter -> potential_features.csv). At the end
# experiments/multilingual_fit/analyze_rank_search.py compares the ranks.
# Same GPU setup as run_feature_search.sbatch (4x RTX 2080, fp32 model split across the GPUs);
# without the MLP track a fit takes seconds to minutes, not ~2 h.
#
# Usage:
#   sbatch run_rank_search.sbatch [RANK ...]        (default ranks: 100 200 300 400 600 800)
# Optional settings, from the environment:
#   OUTDIR    (default mf_outputs/rank_search)  use a fresh one per data set
#   LANGS     (default en+ru+de+es+fr)
#   DATA_DIR  (default experiments/multilingual_fit/data_v2)
#   CONCEPT   (default "World War II")
#   SEEDS     (default "42")
# e.g.
#   sbatch run_rank_search.sbatch 100 200 400 800
#   env SEEDS="42 43 44" sbatch run_rank_search.sbatch
#   env CONCEPT=Cannabis sbatch run_rank_search.sbatch
# Re-running is safe: finished fits and judge calls are skipped, so new ranks / seeds are added.
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:4
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=rank_search_%j.out

set -euo pipefail
umask 002

RANKS="${*:-100 200 300 400 600 800}"
OUTDIR="${OUTDIR:-mf_outputs/rank_search}"
LANGS="${LANGS:-en+ru+de+es+fr}"
DATA_DIR="${DATA_DIR:-experiments/multilingual_fit/data_v2}"
CONCEPT="${CONCEPT:-World War II}"
SEEDS="${SEEDS:-42}"

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

echo "[rank_search] concept='$CONCEPT' langs=$LANGS ranks='$RANKS' seeds='$SEEDS' data=$DATA_DIR outdir=$OUTDIR"
echo "[rank_search] git $(git rev-parse --short HEAD), uncommitted: $(git status --short | wc -l) files"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for SEED in $SEEDS; do
  # shellcheck disable=SC2086
  "$UV" run python -m ember.train_mf_features --n-devices 4 --skip-mlp --concepts "$CONCEPT" \
      --languages "$LANGS" --multilingual-data-dir "$DATA_DIR" --outdir "$OUTDIR" \
      --seed "$SEED" --ranks $RANKS
  for RANK in $RANKS; do
    echo "[rank_search] Gemini filter: rank=$RANK seed=$SEED -> $INTERP_OUT"
    "$UV" run python -m ember.interpret_features --tracks embedding --concepts "$CONCEPT" \
        --outdir "$INTERP_OUT" --rank "$RANK" --seed "$SEED"
  done
done

"$UV" run python experiments/multilingual_fit/analyze_rank_search.py --root "$INTERP_OUT" \
    --langs "$LANGS" --data-dir "$DATA_DIR" --concept "$CONCEPT"
