umask 002
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj
set -a; source .env; set +a
export HF_HOME=/home/morg/NLP_2526b/oripetel/.cache/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PATH=$PWD/.venv/bin:$PATH
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID"
export TORCHDYNAMO_DISABLE=1
