#!/bin/bash
# After every Part 2 Experiments job: redraw curves (X6), per-question split (X8), extrapolation effect (X9).
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm/p2x_final_%j.out
cd /home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure
set -a; source .env; set +a
export HF_HOME=/home/morg/NLP_2526b/semyong/.cache/huggingface HF_HUB_DISABLE_XET=1
UV=/home/morg/NLP_2526b/semyong/.local/bin/uv
for x in x6_strength_curves x7_question_split x8_test_mechanism x9_extrapolation_effect; do
  echo "[final] $x"; $UV run python "Part 2 Experiments/scripts/$x.py" || echo "[final] $x FAILED"
done
echo "[final] done"
