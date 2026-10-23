#!/bin/bash
# v2 cleanup, GPU part: NLLB + MADLAD n-best and MADLAD option lists for the targets (resumable).
#SBATCH -J ml_eval_v2_gen
#SBATCH -p studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
source experiments/ml/evaldata/fetch_local.sh facebook/nllb-200-3.3B google/madlad400-3b-mt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m experiments.ml.evaldata.cleanup_v2 gen "$@"
