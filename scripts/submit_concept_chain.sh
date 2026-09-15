#!/bin/bash
# Submit one concept's full pipeline (Phase 1 grids + Phase 2 cross-lingual eval)
# as a Slurm dependency chain, so it runs unattended: each stage only starts
# once its prerequisite stage has actually completed successfully.
#
# Respects docs/progress.md's "one arm per job" rule (a job bundling two
# 144-cell SNMF grids risks the 12h SLURM time limit) by giving every arm its
# own sbatch submission, wired together with --dependency=afterok instead of
# being sequenced inside one script.
#
# Dependency graph (only real dependency: snmf_ember's grid reads ember's
# best_embed.csv for delta_embed -- see crosslingual/best_hp.py):
#
#   ember (grid)  ---------------------------+--> snmf_ember (grid) --+
#   snmf  (grid)  --------------------------------------------------+--> eval (Phase 2)
#
# ember and snmf run in parallel (neither needs the other); snmf_ember waits
# on ember only; eval waits on all three. If any job fails, its dependents are
# cancelled automatically rather than run on incomplete inputs -- check
# `sacct -u $USER` afterwards to see which stage needs a fix + resubmit.
#
# Usage:
#   ./scripts/submit_concept_chain.sh "World War II"
#   ./scripts/submit_concept_chain.sh "World War II" "Halloween" "Uranium"
#
# This only calls `sbatch`, so it returns immediately -- it does not wait for
# the jobs themselves to run.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 \"Concept One\" [\"Concept Two\" ...]" >&2
  exit 2
fi

for concept in "$@"; do
  echo "=============================================================="
  echo "[submit_concept_chain] concept=${concept}"
  echo "=============================================================="

  job_ember=$(sbatch --parsable scripts/run_erasure_grid.sh \
    --arms ember --concepts "$concept")
  echo "  ember grid        -> job ${job_ember}"

  job_snmf=$(sbatch --parsable scripts/run_erasure_grid.sh \
    --arms snmf --concepts "$concept")
  echo "  snmf grid         -> job ${job_snmf}"

  job_snmf_ef=$(sbatch --parsable --dependency=afterok:"${job_ember}" \
    scripts/run_erasure_grid.sh --arms snmf_ember --concepts "$concept")
  echo "  snmf_ember grid   -> job ${job_snmf_ef} (after ${job_ember})"

  job_eval=$(sbatch --parsable \
    --dependency=afterok:"${job_ember}":"${job_snmf}":"${job_snmf_ef}" \
    scripts/run_experiment.sh --concept "$concept" --phase eval \
    --arms "ember snmf snmf_ember")
  echo "  cross-lingual eval -> job ${job_eval} (after ${job_ember},${job_snmf},${job_snmf_ef})"

  echo "  track with: sacct -j ${job_ember},${job_snmf},${job_snmf_ef},${job_eval} --format=JobID,JobName,State,Elapsed,ExitCode"
  echo
done

echo "[submit_concept_chain] all chains submitted. Nothing further to do until"
echo "they finish or fail -- check back with: sacct -u \$USER"
