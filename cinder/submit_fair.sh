#!/bin/bash
# Submit the fair comparison: one GPU job per (method, concept) with a kept feature; the no-feature
# concepts are written on the spot (no edit, pristine numbers, no GPU). Then, after all jobs:
#   python -m cinder.fair report
set -euo pipefail
cd "${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
export HF_HOME="${HF_HOME:-/home/morg/NLP_2526b/semyong/.cache/huggingface}"
UV="${UV:-/home/morg/NLP_2526b/semyong/.local/bin/uv}"
"$UV" run python - <<'PY' | while IFS=$'\t' read -r kind method concept; do
from cinder import fair as F
for method in F.ROOTS:
    for c in F.ALL_CONCEPTS:
        print("gpu" if F.kept_features(method, c) else "none", method, c, sep="\t")
PY
  if [[ $kind == gpu ]]; then
    echo "$method	$concept	$(sbatch --parsable --job-name="fair_${method}" cinder/run_fair.sbatch "$method" "$concept")"
  else
    "$UV" run python -m cinder.fair run --method "$method" --concept "$concept" >/dev/null
    echo "$method	$concept	no feature -> no edit (written, no GPU)"
  fi
done
