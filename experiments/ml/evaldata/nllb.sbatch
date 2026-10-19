#!/bin/bash
# NLLB-200 3.3B translation (resumable, per-segment cache). The model is copied from HF to
# node-local /tmp first (NFS loads are slow). Each language's cache must be written by ONE job.
# Usage: sbatch -J ml_eval_nllb_x experiments/ml/evaldata/nllb.sbatch --langs hi [--phases ...]
#SBATCH -J ml_eval_nllb
#SBATCH -p studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
source experiments/ml/evaldata/fetch_local.sh facebook/nllb-200-3.3B
python -m experiments.ml.evaldata.translate --engine nllb "$@"
