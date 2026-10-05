#!/bin/bash
# Gemma 4 12B-it as a local judge: the same checks the Qwen judges got, plus a
# negative-control sensitivity check (the baseline alone is nearly all-2 under
# Gemini, so it cannot show the judge can tell damaged text from good text).
#
#   1. agreement_study   -- Gemini's exact 150 scored Alpaca records re-judged
#                           (item-level agreement; same files as the 14B/32B studies)
#   2. pass2_judge_baseline -- the 150 pristine Pass-1 completions (the real
#                           alp_instr_base / alp_flu_base denominators)
#   3. judge_sensitivity -- clean vs mismatched-instruction / scrambled / repeated
#
# Runs in a SEPARATE venv (transformers 5.x; the project's uv.lock pins 4.53.3,
# which predates Gemma 4):  /home/morg/NLP_2526b/semyong/.venvs/gemma4_judge
# Original (verbatim) fluency prompt throughout, like the 32B run.
#
# Usage:
#   sbatch judge_rerun/run_gemma4_judge_eval.sbatch
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:2
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=gemma4_judge_eval_%j.out

set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-/home/morg/NLP_2526b/semyong/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/external/snmf"

PY="${PY:-/home/morg/NLP_2526b/semyong/.venvs/gemma4_judge/bin/python}"
JUDGE="${JUDGE:-google/gemma-4-12B-it}"

echo "[gemma4_eval] judge=$JUDGE python=$PY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[gemma4_eval] === 1/3 agreement study ==="
"$PY" -u -m judge_rerun.agreement_study --split train --judge-model "$JUDGE" \
    --judge-batch-size 4 --fluency-prompt original

echo "[gemma4_eval] === 2/3 pristine baseline ==="
"$PY" -u -m judge_rerun.pass2_judge_baseline --judge-model "$JUDGE" \
    --judge-batch-size 4 --fluency-prompt original

echo "[gemma4_eval] === 3/3 sensitivity (negative controls) ==="
"$PY" -u -m judge_rerun.judge_sensitivity --judge-model "$JUDGE" \
    --judge-batch-size 4 --fluency-prompt original

echo "[gemma4_eval] done"
