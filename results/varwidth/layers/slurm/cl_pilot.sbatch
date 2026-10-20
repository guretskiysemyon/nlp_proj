#!/bin/bash
#SBATCH --job-name=cl_pilot
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=s-005,s-004
#SBATCH --output=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/logs/%x_%j.out
source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/results/varwidth/layers/slurm/header.sh
export GEMINI_TAG=collateral OMP_NUM_THREADS=4
D=results/varwidth/layers/collateral/drafts
python -m experiments.varwidth.collateral basecheck --draft $D/Harry_Potter/v0.json --base $D/base_answers.json
echo '{"Harry Potter": "'$D'/Harry_Potter/v0.json"}' > $D/pilot_chosen.json
python -m experiments.varwidth.collateral freeze --chosen $D/pilot_chosen.json --base $D/base_answers.json --out $D/pilot_questions.json
python -m experiments.varwidth.erase_v2 grid --concept "Harry Potter" --seed 42 --pilot --qfile $D/pilot_questions.json
