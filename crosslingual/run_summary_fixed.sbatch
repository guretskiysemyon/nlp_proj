#!/bin/bash
# Stage D of the cross-lingual repair (crosslingual/summary_fixed.py): summary.csv of
# <method>_fixed against baseline_fixed, and the old-vs-new headline table (in the log).
# CPU only.
#
# Usage:
#   sbatch crosslingual/run_summary_fixed.sbatch --method snmf_ember
#
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=cl_summary_fixed_%j.out

set -euo pipefail
umask 002
REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/morg/NLP_2526b/semyong/.cache/uv}"
UV="${UV:-/home/morg/NLP_2526b/semyong/.local/bin/uv}"
echo "[run_summary_fixed] args: $*"
"$UV" run python -m crosslingual.summary_fixed "$@"
