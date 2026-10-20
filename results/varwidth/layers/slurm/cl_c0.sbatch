#!/bin/bash
#SBATCH --job-name=cl_c0
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --exclude=s-005,s-004
#SBATCH --output=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/logs/%x_%j.out
source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/header.sh
export GEMINI_TAG=collateral OMP_NUM_THREADS=4
python -m experiments.varwidth.collateral premise
python -m experiments.varwidth.erase baselines --concepts "Ancient Rome" --alpaca-n 50
