#!/bin/bash
# qa_new_steps.sh on one SLURM GPU; models copied to node-local /tmp first (fetch_local.sh).
#   sbatch -J ml_qa_<x> experiments/ml/evaldata/qa_new_steps.sbatch <langs...>
#SBATCH -J ml_qa_steps
#SBATCH -p studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 16:00:00
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
source experiments/ml/evaldata/fetch_local.sh facebook/nllb-200-3.3B google/madlad400-3b-mt sentence-transformers/LaBSE
bash experiments/ml/evaldata/qa_new_steps.sh "$@"
