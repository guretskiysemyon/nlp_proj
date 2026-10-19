#!/bin/bash
# MADLAD-400 3B retranslation of the segments whose NLLB output has Kiwi < 0.6 or lost a number.
# fp32 over 2 GPUs (T5 is unstable in fp16). Resumable.
#SBATCH -J ml_eval_madlad
#SBATCH -p studentkillable
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
python -m experiments.ml.evaldata.translate --engine madlad "$@"
