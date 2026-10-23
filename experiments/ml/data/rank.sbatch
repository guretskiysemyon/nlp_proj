#!/bin/bash
#SBATCH --job-name=ml_data_rank
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --exclude=s-004,s-005
#SBATCH --output=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml/results/ml/logs/ml_data_rank_%j.log
source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml/experiments/ml/slurm/header.sh
python -m experiments.ml.data.rank_study
