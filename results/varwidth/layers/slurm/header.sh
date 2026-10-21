umask 002
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj
set -a; source .env; set +a
export HF_HOME=/home/morg/NLP_2526b/oripetel/.cache/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PATH=$PWD/.venv/bin:$PATH
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID"
export TORCHDYNAMO_DISABLE=1
# node-local copy of the Gemma checkpoint (NFS reads are slow under load; Oct 21); falls back to the NFS cache
_LOC=/tmp/hf_oripetel
if [ "$(df --output=avail -k /tmp 2>/dev/null | tail -1)" -gt 12000000 ] 2>/dev/null; then
  ( flock 9
    if [ ! -f $_LOC/.done ]; then
      mkdir -p $_LOC/hub && rsync -a $HF_HOME/hub/models--google--gemma-2-2b-it $HF_HOME/hub/datasets--tatsu-lab--alpaca_eval $HF_HOME/hub/datasets--cais--mmlu $_LOC/hub/ && touch $_LOC/.done
    fi ) 9>/tmp/hf_oripetel.lock
  [ -f $_LOC/.done ] && export HF_HOME=$_LOC && echo "HF_HOME=$_LOC (node-local)"
fi
