#!/bin/bash
# The whole CINDER chain for ONE concept, seed 42 (docs/Part 2/plan.md, "Current plan"):
#   1. features    cinder/build_features.py  Ori's L123 fit + judge -> EMBER format (reuses an existing fit/judge run)
#   2. grid        cinder/grid.py            8 deltas, fixed MC scoring, Alpaca completions of the top 3   (Gemma, GPU 0)
#   3. pick        cinder/pick.py            Qwen2.5-32B (4-bit) on the top 3, best delta                  (all 3 GPUs)
#   4. final test  cinder/final_test.py      held-out questions at the best delta, languages $LANGS        (Gemma, GPU 0)
# A concept whose judge accepts no feature stops after step 1 (no edit). Every step skips what is already on disk,
# so a preempted job can be resubmitted as is.
#
# Usage:
#   sbatch --job-name=cinder_Golf cinder/run_pipeline.sbatch "Golf"
#   LANGS="en de es fr ru" sbatch cinder/run_pipeline.sbatch "Golf"    # later: the other languages
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/slurm/cinder_pipeline_%j.out

set -euo pipefail
umask 002

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

CONCEPT="${1:?usage: sbatch cinder/run_pipeline.sbatch <concept>}"
LANGS="${LANGS:-en}"

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
PY=("$UV" run python)
GPU0="${CUDA_VISIBLE_DEVICES%%,*}"        # Gemma runs on one GPU, as in Part 1

echo "[cinder pipeline] concept=$CONCEPT langs=$LANGS host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "[cinder pipeline] 1/4 features"
"${PY[@]}" -m cinder.build_features --concepts "$CONCEPT"

POT=$("${PY[@]}" -c "import sys; from cinder import common as C; print(C.potential_csv(sys.argv[1]))" "$CONCEPT")
if [[ $("${PY[@]}" -c "import pandas as pd, sys; print(len(pd.read_csv(sys.argv[1])))" "$POT") == 0 ]]; then
  echo "[cinder pipeline] $CONCEPT: the two-way judge accepted no feature -> no edit; stopping after step 1"
  CUDA_VISIBLE_DEVICES="$GPU0" "${PY[@]}" -m cinder.grid --concepts "$CONCEPT"   # writes top3.json "skipped"
  exit 0
fi

echo "[cinder pipeline] 2/4 grid"
CUDA_VISIBLE_DEVICES="$GPU0" "${PY[@]}" -m cinder.grid --concepts "$CONCEPT"

echo "[cinder pipeline] 3/4 pick (Qwen judge)"
"${PY[@]}" -m cinder.pick --concepts "$CONCEPT"

echo "[cinder pipeline] 4/4 final test ($LANGS)"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPU0" "${PY[@]}" -m cinder.final_test --concepts "$CONCEPT" --languages $LANGS

echo "[cinder pipeline] $CONCEPT done"
