#!/bin/bash
# Runs a sequence of qa_new_steps.sh calls (given as one bash command string) on one SLURM GPU.
#   sbatch -J ml_qa_<x> experiments/ml/evaldata/qa_new_multi.sbatch "STEPS=rest bash experiments/ml/evaldata/qa_new_steps.sh es"
#SBATCH -p studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
source experiments/ml/evaldata/fetch_local.sh facebook/nllb-200-3.3B google/madlad400-3b-mt sentence-transformers/LaBSE
eval "$1"
