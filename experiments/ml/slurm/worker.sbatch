#!/bin/bash
#SBATCH -J ml_worker
#SBATCH -p studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 23:00:00
#SBATCH --requeue
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m experiments.ml.worker
