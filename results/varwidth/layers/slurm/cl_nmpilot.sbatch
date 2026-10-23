#!/bin/bash
#SBATCH --job-name=cl_nmpilot
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=s-003,s-005
#SBATCH --output=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/logs/%x_%A_%a.out
source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/header.sh
export OMP_NUM_THREADS=4
python -m experiments.varwidth.nonce_edit run --concept Baseball --seed 42 --variants nm:free nm:cinder --pilot
