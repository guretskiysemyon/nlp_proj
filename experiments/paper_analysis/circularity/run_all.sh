#!/bin/bash
# WP7 fits on one local GPU (nvidia-smi GPU 3 = CUDA index 2 in PCI order on c-001), two workers sharing it.
# usage: bash experiments/paper_analysis/circularity/run_all.sh <shard 0|1>
cd /home/morg/NLP_2526b/oripetel/nlp_proj/nlp_proj
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-2}
PY=".venv/bin/python -m experiments.paper_analysis.circularity.fit --threads 3 --shard $1/2"
$PY --plan main --method fixed
$PY --plan etagamma --method fixed
$PY --plan main --method rn
