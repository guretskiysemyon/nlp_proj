#!/bin/bash
# Steps 3.1-3.2 for one (notebook, seed) pair; see scripts/run_seed_sweep.py.
#   sbatch --array=0-4 --export=ALL,KIND=plain    scripts/seed_sweep.sbatch
#   sbatch --array=0-4 --export=ALL,KIND=whitened scripts/seed_sweep.sbatch
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=results/ancient_rome_seed_sweep/logs/%x_%A_%a.out

umask 002
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj

SEED=$((42 + SLURM_ARRAY_TASK_ID))   # task 0 -> seed 42

set -a; source .env; set +a
export HF_HOME=/home/morg/NLP_2526b/oripetel/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PWD/.venv/bin:$PATH

echo "host=$(hostname) kind=$KIND seed=$SEED"
python scripts/run_seed_sweep.py "$KIND" "$SEED"
