#!/bin/bash
#SBATCH --job-name=cl_xbase
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=s-005
#SBATCH --output=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/logs/%x_%j.out
source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/header.sh
export GEMINI_TAG=collateral OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv
D=results/varwidth/layers/collateral/drafts
python -m experiments.varwidth.collateral basecheck --base $D/base_answers_expansion.json --draft $DRAFTS
