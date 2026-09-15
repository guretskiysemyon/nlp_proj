#!/bin/bash
# Produce the erasure hyper-parameters that crosslingual/run.py consumes.
#
# This runs ONLY the removal/erasure search -- the grid (Step 2) and the top-K
# validate stage (Step 3) -- and stops there: the grid-only YAMLs set
# run_tests_after_train: false and relearning.enabled: false, so no final-test
# or relearning work is done. Outputs per concept:
#
#   results/<method[_ef]>/<model>/rank100/seed42/train_mc/<Concept>/hps.csv
#                                                        .../top_hps.csv
#                                                        .../top_hps_valid.csv
#   results/ember/.../train_mc/best_embed.csv          (ember arm only)
#
# Arm order matters: ember runs first because snmf_ember's embedding step reads
# best_embed.csv (it would otherwise run the ember grid inline itself).
#
# Everything is --train-eval mc. MC drives the HP selection for this project;
# the open-question format is not used for the grid.
#
# Usage:
#   ./scripts/run_erasure_grid.sh                       # all arms, all concepts
#   ./scripts/run_erasure_grid.sh --arms ember          # one arm
#   sbatch --array=0-3 scripts/run_erasure_grid.sh      # one concept per task
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=erasure_grid_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

# Keep these byte-identical to the QA JSON keys.
# Default only; scripts/submit_all.sh always passes --concepts explicitly.
# Batch 1 (done): Harry Potter, World War II, Halloween, Uranium
# Batch 2 (next): the list below -- see scripts/run_experiment.sh for why.
CONCEPTS=(
  "Golf"
  "Baseball"
  "Culture of Greece"
  "Gambling"
)

# ember must precede snmf_ember: it writes best_embed.csv.
ARMS=(ember snmf snmf_ember)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) shift; IFS=' ' read -r -a ARMS <<< "$1"; shift ;;
    --concepts) shift; IFS='|' read -r -a CONCEPTS <<< "$1"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

TRAIN_EVAL="${TRAIN_EVAL:-mc}"
RANK="${RANK:-100}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"

# ember.run_erasure calls load_dotenv() itself, so GEMINI_API_KEY / HF_TOKEN
# come from .env. Exported here too for the HF cache settings below.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export HF_HOME="${HF_HOME:-/home/morg/NLP_2526b/semyong/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/triton}"

run_arm_concept() {
  local arm="$1" concept="$2"
  echo "=============================================================="
  echo "[erasure_grid] arm=${arm} concept=${concept} train_eval=${TRAIN_EVAL}"
  echo "=============================================================="
  "$PYTHON" -m ember.run_erasure \
    --config "configs/${arm}_gemma_gridonly.yaml" \
    --concepts "$concept" \
    --train-eval "$TRAIN_EVAL" \
    --rank "$RANK" --seed "$SEED"
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="$SLURM_ARRAY_TASK_ID"
  if (( idx < 0 || idx >= ${#CONCEPTS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=$idx out of range (0..$(( ${#CONCEPTS[@]} - 1 )))" >&2
    exit 2
  fi
  for arm in "${ARMS[@]}"; do
    run_arm_concept "$arm" "${CONCEPTS[$idx]}"
  done
else
  for arm in "${ARMS[@]}"; do
    for concept in "${CONCEPTS[@]}"; do
      run_arm_concept "$arm" "$concept"
    done
  done
fi

echo "[erasure_grid] done. Next: ./scripts/run_crosslingual_all.sh --dry-run"
