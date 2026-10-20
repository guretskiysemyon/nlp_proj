#!/bin/bash
# G1 leak curves for one (method, concept): see g1_leak_curves.py. One RTX 2080, Gemma only.
#   sbatch "Part 2 Experiments/scripts/g1_leak_curves.sbatch" CINDER Golf
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/p2x_g1_%j.out
set -euo pipefail
umask 002
cd /home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure
set -a; source .env; set +a
export HF_HOME=/home/morg/NLP_2526b/semyong/.cache/huggingface HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/morg/NLP_2526b/semyong/.local/bin/uv run python "Part 2 Experiments/scripts/g1_leak_curves.py" --method "$1" --concept "$2"
