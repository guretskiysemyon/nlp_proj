#!/bin/bash
#SBATCH --job-name=cl_2x2_fixed05
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
sleep $(( (SLURM_ARRAY_TASK_ID % 6) * 30 ))
IFS='|' read -r C S <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/varwidth/layers/slurm/cl_2x2.lines)"
echo "$C seed=$S"
python -m experiments.varwidth.erase_v2 final --concept "$C" --seed "$S" --qfile data/collateral_questions_expansion.json --rule fixed05
