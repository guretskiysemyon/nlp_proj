#!/bin/bash
# Run one evaldata step on the local machine (c-001, GPU 3 only, <= 4 threads), loading models
# from a fast LOCAL HF cache instead of NFS.
#   bash experiments/ml/evaldata/run_local.sh download          # NLLB + LaBSE into the local cache
#   bash experiments/ml/evaldata/run_local.sh nllb               # translate + back-translate
#   bash experiments/ml/evaldata/run_local.sh score --engine nllb ctx [--metrics labse]
#   bash experiments/ml/evaldata/run_local.sh madlad | build
set -euo pipefail
umask 002
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj_ml
set -a; source /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.env; set +a
export LOCAL_HF=${LOCAL_HF:-/tmp/claude-21644/-home-morg-NLP-2526b-oripetel-nlp-proj-nlp-proj/c5e716de-a1fa-45a5-9267-5521d11b2fe1/scratchpad/hf}
export HF_HOME=$LOCAL_HF HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj/.venv/bin/python
COMET_PY=/home/morg/NLP_2526b/semyong/.venv-comet/bin/python
step=$1; shift
case $step in
  download)
    unset HF_HUB_OFFLINE
    $PY -c "
from huggingface_hub import snapshot_download as s
print(s('facebook/nllb-200-3.3B', allow_patterns=['*.json', '*.bin', '*.model']), flush=True)
print(s('sentence-transformers/LaBSE', ignore_patterns=['*.h5', 'tf_model*', '*.onnx', 'onnx/*', 'openvino/*']), flush=True)
print('DONE')" ;;
  nllb)  export HF_HUB_OFFLINE=1; $PY -m experiments.ml.evaldata.translate --engine nllb "$@" ;;
  madlad) export HF_HUB_OFFLINE=1; $PY -m experiments.ml.evaldata.translate --engine madlad "$@" ;;
  score) export HF_HUB_OFFLINE=1; $COMET_PY -I experiments/ml/evaldata/score.py "$@" ;;
  build) $PY -m experiments.ml.evaldata.build "$@" ;;
  *) echo "unknown step $step"; exit 1 ;;
esac
