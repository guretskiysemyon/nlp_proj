#!/bin/bash
#SBATCH --job-name=ml_data_nllb
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --exclude=s-004,s-005
#SBATCH --output=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml/results/ml/logs/ml_data_nllb_%j.log
# NLLB-200 3.3B worker. The HF-cache copy on NFS is incomplete (I/O error on the large shard), so the model is
# downloaded to node-local /tmp first (~17 GB, a few minutes). Args are passed to translate_nllb --stage run.
source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml/experiments/ml/slurm/header.sh
LOCAL=/tmp/ml_data_nllb_$USER
mkdir -p $LOCAL
SNAP=$(HF_HUB_OFFLINE=0 python -c "
from huggingface_hub import snapshot_download as s
print(s('facebook/nllb-200-3.3B', cache_dir='$LOCAL'))" | tail -1)
echo "model at $SNAP"
python -m experiments.ml.data.translate_nllb --stage run --model-path "$SNAP" "$@"
