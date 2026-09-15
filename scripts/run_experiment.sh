#!/bin/bash
# One entrypoint for the whole cross-lingual experiment, for ONE concept.
#
#   Phase 1  find the parameters   (the paper's Stages 1-3, per erasure arm)
#   Phase 2  erase and compare     (crosslingual/run.py: 4 configs x 5 languages)
#
# Nothing in ember/ is modified. Phase 1 runs the project's own pipeline
# unchanged via ember.run_erasure; the grid-only YAMLs simply stop it after the
# validate stage, because the paper's Stage 4 (English final test + relearning)
# is replaced here by the cross-lingual test, which evaluates English through
# the same code path as every other language.
#
# Both phases are resumable: the grid skips HP cells already in hps.csv, and
# the cross-lingual step skips cells that already have output. If a job hits
# the SLURM time limit, just resubmit the same command and it continues.
#
# Usage:
#   ./scripts/run_experiment.sh --concept "Harry Potter"
#   ./scripts/run_experiment.sh --concept "Golf" --arms "ember"     # one arm
#   ./scripts/run_experiment.sh --concept "Golf" --phase grids      # params only
#   ./scripts/run_experiment.sh --concept "Golf" --phase eval       # compare only
#   ./scripts/run_experiment.sh --concept "Golf" --dry-run          # check, no GPU
#   sbatch --array=0-3 scripts/run_experiment.sh                    # one task per concept
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiment_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

# The agreed concept set; index with SLURM_ARRAY_TASK_ID. Spellings must match
# the QA JSON keys byte-for-byte.
# BATCH 1 -- DONE (plan section 9 tokenization-contrast set):
#   Harry Potter, World War II, Halloween, Uranium
# Results in results/cross_lang_test/summary.csv. See docs/progress.md section 4.
#
# BATCH 2 -- chosen to BREAK the confound left by batch 1. In batch 1, surface
# overlap and EMBER's edited-token count moved together, so both explanations of
# the leakage fit equally well (docs/progress.md section 4.4):
#   H1 surface-form: leak ~ absence of the edited English tokens in the L prompt
#   H2 edit-breadth: leak ~ how many tokens EMBER edited (r=0.93 over batch 1)
# These four make H1 and H2 predict OPPOSITE things, so the batch separates them.
#
#   concept            overlap  emb_feats  H1 predicts  H2 predicts
#   Golf                 0.177      2        LOW          HIGH
#   Baseball             0.156      3        LOW          HIGHEST
#   Culture of Greece    0.056      1        HIGH         LOW
#   Gambling             0.108      3        mid          HIGH      (safety anchor)
#
# overlap = mean word overlap between the English question and its de/es/fr
# translation (QA files are index-aligned); emb_feats = rows in the concept's
# interpretations/.../embedding/potential_features.csv at tau=2.0.
CONCEPTS=(
  "Golf"
  "Baseball"
  "Culture of Greece"
  "Gambling"
)

# ember must come before snmf_ember: it writes the best_embed.csv that the
# snmf_ember arm's embedding step reads (Stage 1 feeding the ensemble).
ARMS=(ember snmf snmf_ember)

CONCEPT=""
PHASE="all"
DRY_RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --concept)  shift; CONCEPT="$1"; shift ;;
    --arms)     shift; IFS=' ' read -r -a ARMS <<< "$1"; shift ;;
    --phase)    shift; PHASE="$1"; shift ;;
    --dry-run)  DRY_RUN="--dry-run"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CONCEPT" ]]; then
  idx="${SLURM_ARRAY_TASK_ID:-0}"
  if (( idx < 0 || idx >= ${#CONCEPTS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=$idx out of range (0..$(( ${#CONCEPTS[@]} - 1 )))" >&2
    exit 2
  fi
  CONCEPT="${CONCEPTS[$idx]}"
fi

case "$PHASE" in all|grids|eval) ;; *) echo "--phase must be all|grids|eval" >&2; exit 2 ;; esac

# MC everywhere: the grid selects hyper-parameters on multiple-choice, which
# the paper reports as the more challenging setting and the one that transfers
# to open-ended (appendix E), not the other way round.
TRAIN_EVAL="${TRAIN_EVAL:-mc}"

# Phase 2 evaluates MC only. Gemma-2-2B's open-ended QA accuracy in de/es/fr/ru
# is 0.04-0.24 even on the pristine model -- too close to zero to be a usable
# denominator for a retention ratio, so every non-English open-ended cell would
# be screened out anyway (spec 6). MC pristine accuracy holds up (0.56-0.62 for
# de/es/fr), so that is where the cross-lingual signal is. Override with
# QA_KINDS="mc open" to evaluate both.
QA_KINDS="${QA_KINDS:-mc}"
RANK="${RANK:-100}"
SEED="${SEED:-42}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OUT_DIR="${OUT_DIR:-results/cross_lang_test}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"

# ember.run_erasure and crosslingual/run.py both call load_dotenv() themselves;
# sourcing here only makes HF_TOKEN / cache paths available to this shell.
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

banner() { echo; echo "=============================================================="; echo "$1"; echo "=============================================================="; }

banner "[experiment] concept=${CONCEPT}  phase=${PHASE}  arms=${ARMS[*]}  train_eval=${TRAIN_EVAL}"

# ---------------------------------------------------------------------------
# Phase 1: find the parameters (paper Stages 1-3)
# ---------------------------------------------------------------------------
if [[ "$PHASE" == "all" || "$PHASE" == "grids" ]]; then
  for arm in "${ARMS[@]}"; do
    banner "[experiment] PHASE 1 - grid: arm=${arm} concept=${CONCEPT}"
    if [[ -n "$DRY_RUN" ]]; then
      echo "[dry-run] would run: ember.run_erasure --config configs/${arm}_gemma_gridonly.yaml --concepts '${CONCEPT}' --train-eval ${TRAIN_EVAL}"
      continue
    fi
    "$PYTHON" -m ember.run_erasure \
      --config "configs/${arm}_gemma_gridonly.yaml" \
      --concepts "$CONCEPT" \
      --train-eval "$TRAIN_EVAL" \
      --rank "$RANK" --seed "$SEED"
  done
fi

# ---------------------------------------------------------------------------
# Phase 2: erase and compare across languages
# ---------------------------------------------------------------------------
if [[ "$PHASE" == "all" || "$PHASE" == "eval" ]]; then
  banner "[experiment] PHASE 2 - cross-lingual evaluation: concept=${CONCEPT}"
  # Only evaluate the configs whose parameters this run actually produced.
  CONFIGS=(pristine)
  for arm in "${ARMS[@]}"; do CONFIGS+=("$arm"); done
  # shellcheck disable=SC2086
  "$PYTHON" crosslingual/run.py \
    --concept "$CONCEPT" \
    --configs "${CONFIGS[@]}" \
    --qa-kinds $QA_KINDS \
    --model google/gemma-2-2b-it --rank "$RANK" --seed "$SEED" \
    --train-eval "$TRAIN_EVAL" \
    --results-root "$RESULTS_ROOT" \
    --out-dir "$OUT_DIR" \
    $DRY_RUN
fi

banner "[experiment] done: ${CONCEPT}"
echo "summary -> ${OUT_DIR}/summary.csv"
