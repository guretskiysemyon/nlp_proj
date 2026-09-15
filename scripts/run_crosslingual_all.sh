#!/bin/bash
# Thin driver: run the cross-lingual evaluation once per concept.
#
# One process per concept is deliberate (crosslingual/run.py takes a single
# --concept) so the concepts can be parallelized as separate jobs. Run it
# directly for a sequential sweep, or as a SLURM array where each task picks
# its own concept out of CONCEPTS via SLURM_ARRAY_TASK_ID:
#
#   sbatch --array=0-3 scripts/run_crosslingual_all.sh
#   ./scripts/run_crosslingual_all.sh --dry-run
#
# Any extra arguments are forwarded verbatim to crosslingual/run.py, so
# --dry-run / --configs / --languages / --force all work here.
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=crosslingual_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

# The agreed 4-concept set. Keep the spellings byte-identical to the QA JSON keys.
CONCEPTS=(
  "Harry Potter"
  "Golf"
  "Nazism"
  "COVID-19 pandemic"
)

# Forwarded to every invocation; override from the environment if needed.
MODEL="${MODEL:-google/gemma-2-2b-it}"
RANK="${RANK:-100}"
SEED="${SEED:-42}"
TRAIN_EVAL="${TRAIN_EVAL:-mc}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUT_DIR="${OUT_DIR:-results/cross_lang_test}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-/home/morg/NLP_2526b/semyong/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"

run_one() {
  local concept="$1"; shift
  echo "=============================================================="
  echo "[run_crosslingual_all] concept=${concept}"
  echo "=============================================================="
  "$PYTHON" crosslingual/run.py \
    --concept "$concept" \
    --model "$MODEL" --rank "$RANK" --seed "$SEED" \
    --train-eval "$TRAIN_EVAL" \
    --results-root "$RESULTS_ROOT" \
    --out-dir "$OUT_DIR" \
    "$@"
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # One array task per concept, matching the SLURM_ARRAY_TASK_ID pattern
  # ember/evals/gemini.py already assumes for its usage log.
  idx="$SLURM_ARRAY_TASK_ID"
  if (( idx < 0 || idx >= ${#CONCEPTS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=$idx out of range (0..$(( ${#CONCEPTS[@]} - 1 )))" >&2
    exit 2
  fi
  run_one "${CONCEPTS[$idx]}" "$@"
else
  for concept in "${CONCEPTS[@]}"; do
    run_one "$concept" "$@"
  done
fi
