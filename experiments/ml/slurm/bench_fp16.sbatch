#!/bin/bash
#SBATCH -J ml_bench_fp16
#SBATCH -p studentkillable
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 00:40:00
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
python -m experiments.ml.bench --mode fp16
