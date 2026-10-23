#!/bin/bash
# Several concepts in sequence on one local GPU: bash experiments/ml/run_local_many.sh <gpu> "<c1>" "<c2>" ...
# (STAGES as in run_local.sh). One concept's failure does not stop the others.
GPU=$1; shift
for c in "$@"; do bash experiments/ml/run_local.sh $GPU "$c" || echo "FAILED $c" >> results/ml/logs/local_failures.log; done
