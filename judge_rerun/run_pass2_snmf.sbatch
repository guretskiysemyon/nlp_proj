#!/bin/bash
# Runs pass2_snmf_judge.py (SNMF Pass 2: judge the top-3 HP cells x 50 Alpaca
# completions from pass1_snmf_top3.py, pick the best cell) with the adopted
# Qwen2.5-32B judge. Same node pinning and GPU count as
# run_pass2_delta_search_anygpu.sbatch (EMBER Pass 2): rtx_2080 only, since the
# titan nodes run the 32B judge ~40x slower.
#
# ~50 min per cell (3 cells = ~2.5 h per concept), so up to ~8 concepts fit the
# 24h limit; 3-4 per job keeps a preemption cheap. A finished concept is
# skipped and a half-judged one resumes, so after a kill resubmit the same
# command. Put each concept in only one job at a time.
#
# Usage:
#   sbatch judge_rerun/run_pass2_snmf.sbatch --concept "Golf" "Heroin" "Nazism"
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=pass2_snmf_%j.out

set -euo pipefail

# Everything this job creates is group-writable (shared `benohayon` group),
# so a teammate can resume or take over the run.
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
  echo "usage: sbatch judge_rerun/run_pass2_snmf.sbatch --concept \"<Concept>\" ..." >&2
  exit 2
fi

echo "[run_pass2_snmf] args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$UV" run python -m judge_rerun.pass2_snmf_judge \
    --judge-model "Qwen/Qwen2.5-32B-Instruct" \
    --judge-batch-size 4 \
    --fluency-prompt original \
    "$@"
