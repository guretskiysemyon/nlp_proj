#!/bin/bash
#SBATCH -J ml_worker
#SBATCH -p studentkillable
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 23:00:00
#SBATCH --requeue
#SBATCH --exclude=s-004,s-005
#SBATCH -o results/ml/logs/%x_%j.out
source experiments/ml/slurm/header.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Node-local model copy: sequential cp from NFS is far faster than mmap-loading the shards over NFS. One copy per
# node (flock), reused by later jobs on the node; falls back to NFS if /tmp lacks space.
SNAP=$(ls -d $HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ | head -1)
LOC=/tmp/ml_llama_$USER
if [ $(df --output=avail -k /tmp | tail -1) -gt 20000000 ] || [ -f $LOC/.complete ]; then
    ( flock 9; [ -f $LOC/.complete ] || { mkdir -p $LOC && cp -L $SNAP/*.json $SNAP/*.safetensors $LOC/ && touch $LOC/.complete; } ) 9>/tmp/ml_llama_$USER.lock
    [ -f $LOC/.complete ] && export LLAMA_LOCAL=$LOC
fi
echo "LLAMA_LOCAL=${LLAMA_LOCAL:-none} $(date)"
python -m experiments.ml.worker "$@"
