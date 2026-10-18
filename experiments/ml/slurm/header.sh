umask 002
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml
set -a; source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.env; set +a
export HF_HOME=/home/morg/NLP_2526b/oripetel/.cache/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PATH=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.venv/bin:$PATH
export GEMINI_USAGE_LOG=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml/results/ml/gemini_usage.jsonl GEMINI_BUDGET_NIS=15 GEMINI_PRICE_IN_USD=0.25 GEMINI_PRICE_OUT_USD=2.50
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID gpus=$CUDA_VISIBLE_DEVICES"
export PYTHONPATH=/home/morg/NLP_2526b/oripetel/.cache/pylib_ml
