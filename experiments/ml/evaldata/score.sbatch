#!/bin/bash
# Quality columns of the cached translations (LaBSE, round-trip chrF, COMET-Kiwi when licensed).
# Usage: sbatch experiments/ml/evaldata/score.sbatch --engine nllb|madlad [--metrics labse chrf_bt kiwi]
# Runs in the comet environment (unbabel-comet pins protobuf < 5). Resumable.
#SBATCH -J ml_eval_score
#SBATCH -p studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
/home/morg/NLP_2526b/semyong/.venv-comet/bin/python -I experiments/ml/evaldata/score.py "$@"
