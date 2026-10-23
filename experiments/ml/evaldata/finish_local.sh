#!/bin/bash
# Final pass, local c-001 GPU 3 (run_local.sh): wait for the SLURM NLLB jobs, fill any missing
# segments / option lists, LaBSE for NLLB + ctx, MADLAD for the low segments, LaBSE for MADLAD,
# then the full build + QUALITY.md. Every step is resumable.
set -euo pipefail
D=$(dirname "$0")
while squeue -u $USER -h -o %j | grep -q '^ml_eval_nllb'; do sleep 60; done
bash $D/run_local.sh nllb --phases rest ctx
bash $D/run_local.sh score --engine nllb ctx --metrics labse
bash $D/run_local.sh madlad
bash $D/run_local.sh score --engine madlad --metrics labse
bash $D/run_local.sh build
echo FINISHED
