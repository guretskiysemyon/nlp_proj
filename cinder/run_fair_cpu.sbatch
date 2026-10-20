#!/bin/bash
# A CPU stage of cinder/fair.py as a short job (this cluster has no CPU-only partition), so it can sit in a
# dependency chain: build-ml after the fits, report after the erasure jobs.
#
# Usage:
#   sbatch cinder/run_fair_cpu.sbatch build-ml
#   sbatch cinder/run_fair_cpu.sbatch report
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/fair_cpu_%j.out

set -euo pipefail
umask 002
REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export HF_HOME="${HF_HOME:-/home/morg/NLP_2526b/semyong/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/uv}"
export CUDA_VISIBLE_DEVICES=""
UV="${UV:-/home/morg/NLP_2526b/semyong/.local/bin/uv}"
echo "[fair_cpu] python -m cinder.fair $*"
"$UV" run python -m cinder.fair "$@"
