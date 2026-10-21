#!/bin/bash
#SBATCH --job-name=cl_floor
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --exclude=s-005
#SBATCH --output=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/logs/%x_%A_%a.out
source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/header.sh
export GEMINI_TAG=collateral OMP_NUM_THREADS=4
C="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/varwidth/layers/slurm/cl_lm.lines)"
echo "$C"
python -m experiments.varwidth.last_mile floor --concept "$C"
