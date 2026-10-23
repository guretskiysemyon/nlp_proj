#!/bin/bash
#SBATCH -J ml_seeds
#SBATCH -p studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o results/ml/logs/%x_%j.out
# python -m experiments.ml.seeds with the given args (e.g. --seed 1 --apps A1 --device cuda), ML_CONCEPTS=all
source experiments/ml/slurm/header.sh
export ML_CONCEPTS=all
python -m experiments.ml.seeds "$@"
