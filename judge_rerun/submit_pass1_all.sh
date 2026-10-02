#!/bin/bash
# Submit Pass 1 (judge_rerun/pass1_ember_delta.py) as one SLURM job per
# concept, for all 18 concepts. One job per concept (not one 18-concept job)
# because: (a) one concept takes ~65 min, 18 sequentially would be ~19.5h,
# over run_pass1.sbatch's 2h limit; (b) the `studentkillable` partition can
# preempt a running job, and losing one ~65min job is far cheaper than
# losing a ~19.5h one. pass1_ember_delta.py itself skips a concept whose
# grid_raw.csv already exists, so re-running this script after a partial
# failure only resubmits what's actually missing.
#
# This is NOT an sbatch script -- run it directly on the login node.
#
# Usage:
#   ./judge_rerun/submit_pass1_all.sh --dry-run
#   ./judge_rerun/submit_pass1_all.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

CONCEPTS=(
  "Ancient Rome" "Artificial intelligence" "Baseball" "COVID-19 pandemic"
  "Cannabis" "Culture of Greece" "Gambling" "Golf" "Gun" "Halloween"
  "Harry Potter" "Heroin" "Nazism" "Pornography" "Republic of Ireland"
  "Uranium" "Valentine's Day" "World War II"
)

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=1
fi

for concept in "${CONCEPTS[@]}"; do
  if [[ -n "$DRY" ]]; then
    printf '[dry-run] sbatch judge_rerun/run_pass1.sbatch --concept %q\n' "$concept"
  else
    sbatch judge_rerun/run_pass1.sbatch --concept "$concept"
  fi
done
