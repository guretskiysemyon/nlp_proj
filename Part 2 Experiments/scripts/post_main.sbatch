#!/bin/bash
# After the 4 main CINDER jobs: 5-language final test at the picked delta, the CINDER vs EMBER table, and the
# analyses that need the picks (Part 2 Experiments/scripts/post_analyses.sh). One RTX 2080 (Gemma only).
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/p2x_post_%j.out
set -uo pipefail
umask 002
cd /home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure
set -a; source .env; set +a
export HF_HOME=/home/morg/NLP_2526b/semyong/.cache/huggingface HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UV=/home/morg/NLP_2526b/semyong/.local/bin/uv
echo "[post] final test, 5 languages"
$UV run python -m cinder.final_test --concepts Cannabis Gun "Culture of Greece" Golf --languages en de es fr ru --overwrite --summary
echo "[post] compare"
$UV run python -m cinder.compare > "Part 2 Experiments/comparison_cinder_vs_ember.md" 2>&1
cp judge_rerun/Cinder/comparison.md "Part 2 Experiments/comparison_cinder_vs_ember.md" 2>/dev/null || true
echo "[post] analyses"
bash "Part 2 Experiments/scripts/post_analyses.sh"
echo "[post] done"
