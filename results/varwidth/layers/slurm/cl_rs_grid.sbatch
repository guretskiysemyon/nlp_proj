#!/bin/bash
#SBATCH --job-name=cl_rsgrid
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=s-005
#SBATCH --output=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/logs/%x_%A_%a.out
source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/header.sh
export GEMINI_TAG=collateral OMP_NUM_THREADS=4
sleep $(( (SLURM_ARRAY_TASK_ID % 6) * 60 ))   # stagger model loads from NFS
IFS='|' read -r C S G <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/varwidth/layers/slurm/cl_grid_rs.lines)"
echo "$C seed=$S group=$G"
python -m experiments.varwidth.erase_v2 grid --concept "$C" --seed "$S" --group "$G"
