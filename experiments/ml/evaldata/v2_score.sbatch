#!/bin/bash
# v2 cleanup: LaBSE for the new candidates (comet venv), then the CPU build of qa_v2.
#SBATCH -J ml_eval_v2_score
#SBATCH -p studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
source experiments/ml/evaldata/fetch_local.sh sentence-transformers/LaBSE
EVALDATA_V2=1 /home/morg/NLP_2526b/semyong/.venv-comet/bin/python -I experiments/ml/evaldata/score.py \
    --engine nllb_nbest madlad_nbest madlad_ctx --metrics labse
python -m experiments.ml.evaldata.cleanup_v2 build
