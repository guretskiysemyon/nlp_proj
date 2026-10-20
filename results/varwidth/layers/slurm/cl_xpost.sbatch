#!/bin/bash
#SBATCH --job-name=cl_xpost
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=s-005
#SBATCH --output=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/logs/%x_%A_%a.out
source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/header.sh
export GEMINI_TAG=collateral OMP_NUM_THREADS=4
sleep $(( (SLURM_ARRAY_TASK_ID % 6) * 60 ))
# amendment 8, per run: phase-2 collateral on validation (no-op if present), the three final arms, E2 integrity
IFS='|' read -r C S <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/varwidth/layers/slurm/cl_x.lines)"
Q=data/collateral_questions_expansion.json
echo "$C seed=$S"
set -e
python -m experiments.varwidth.erase_v2 grid --concept "$C" --seed "$S" --variants sub:ember sub:cinder sub:cinder_rs sub:cinder_z sub:cinder_rs_z --qfile $Q
for R in collateral fixed05 paper; do python -m experiments.varwidth.erase_v2 final --concept "$C" --seed "$S" --qfile $Q --rule $R; done
python -m experiments.varwidth.integrity run --concept "$C" --seed "$S"
