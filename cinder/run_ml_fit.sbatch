#!/bin/bash
# Native-text EMBER fits for cinder/ml_arms.py: one language set, one rank, the given concepts (default: all 18).
# ember.train_mf_features --skip-mlp (embedding track only) on experiments/multilingual_fit/data_v2 (English from
# data/), then ember.interpret_features --tracks embedding (gemini-3.5-flash-lite), as run_rank_search.sbatch.
# Output: mf_outputs/rank_search/multi_language/<LANGS>/. Finished fits and judge calls are skipped, so a killed
# job can just be resubmitted.
#
# Usage:
#   sbatch cinder/run_ml_fit.sbatch de 100                       # all 18 concepts
#   sbatch cinder/run_ml_fit.sbatch en+ru+de+es+fr 500 Golf Gun
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/ml_fit_%j.out

set -euo pipefail
umask 002

LANGS="${1:?usage: sbatch cinder/run_ml_fit.sbatch <langs> <rank> [<concept> ...]}"
RANK="${2:?usage: sbatch cinder/run_ml_fit.sbatch <langs> <rank> [<concept> ...]}"
shift 2
if [[ $# -ge 1 ]]; then
  CONCEPTS=("$@")
else
  CONCEPTS=("Culture of Greece" "Golf" "Republic of Ireland" "Ancient Rome" "Baseball" "Uranium" "Harry Potter"
            "Cannabis" "Gambling" "Gun" "Pornography" "Valentine's Day" "World War II" "Nazism" "Heroin"
            "Halloween" "Artificial intelligence" "COVID-19 pandemic")
fi
OUTDIR="mf_outputs/rank_search"
DATA_DIR="experiments/multilingual_fit/data_v2"

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export HF_HOME="${HF_HOME:-/home/morg/NLP_2526b/semyong/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/uv}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/triton}"
UV="${UV:-/home/morg/NLP_2526b/semyong/.local/bin/uv}"

echo "[ml_fit] langs=$LANGS rank=$RANK concepts=${#CONCEPTS[@]} data=$DATA_DIR outdir=$OUTDIR"
echo "[ml_fit] git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$UV" run python -m ember.train_mf_features --n-devices 4 --skip-mlp --concepts "${CONCEPTS[@]}" \
    --languages "$LANGS" --multilingual-data-dir "$DATA_DIR" --outdir "$OUTDIR" --seed 42 --ranks "$RANK"
"$UV" run python -m ember.interpret_features --tracks embedding --concepts "${CONCEPTS[@]}" \
    --outdir "$OUTDIR/multi_language/$LANGS" --rank "$RANK" --seed 42 --max-workers 3
echo "[ml_fit] done"
