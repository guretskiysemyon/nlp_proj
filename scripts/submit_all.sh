#!/bin/bash
# Submit the FULL experiment for a set of concepts in one command, with SLURM
# dependencies wired so nothing starts before its inputs exist.
#
# This is NOT an sbatch script -- run it directly on the login node. It calls
# sbatch for you, 4 jobs per concept.
#
# Per concept the dependency graph is:
#
#     ember  ─┬─────────────► snmf_ember ─┐
#             │                            ├──► eval
#     snmf ───┴────────────────────────────┘
#
#   - ember and snmf are independent, so they start immediately, in parallel.
#   - snmf_ember waits for ember: its embedding step reads the best_embed.csv
#     that the ember run writes. Without the wait it would re-run the ember
#     grid inline and waste ~1.2h.
#   - eval waits for all three: crosslingual/run.py reads each arm's chosen HP.
#
# Everything is afterok, so a failed grid stops its dependents rather than
# feeding them a half-written hps.csv.
#
# Usage:
#   ./scripts/submit_all.sh --dry-run                 # print, submit nothing
#   ./scripts/submit_all.sh                           # batch-2 concepts
#   ./scripts/submit_all.sh --concepts "Golf|Baseball"
#   ./scripts/submit_all.sh --arms "ember snmf"       # skip an arm
#
# Resuming: everything is idempotent. If a job dies (time limit, node failure),
# resubmit the same command -- grids skip cells already in hps.csv and eval
# skips cells already on disk. Dependents that were cancelled by afterok just
# need resubmitting.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
cd "$REPO_DIR"

# Batch 2 -- chosen to break the surface-overlap vs edit-breadth confound left
# by batch 1. See the comment block in scripts/run_experiment.sh.
CONCEPTS=(
  "Golf"
  "Baseball"
  "Culture of Greece"
  "Gambling"
)
ARMS=(ember snmf snmf_ember)
DRY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --concepts) shift; IFS='|' read -r -a CONCEPTS <<< "$1"; shift ;;
    --arms)     shift; IFS=' ' read -r -a ARMS <<< "$1"; shift ;;
    --dry-run)  DRY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

has_arm() { local a; for a in "${ARMS[@]}"; do [[ "$a" == "$1" ]] && return 0; done; return 1; }

# ---------------------------------------------------------------------------
# Pre-flight: download SNMF features ONCE, here, before any job is submitted.
#
# Every grid job calls ember.data.ensure_features() on startup. When two jobs
# for the SAME concept start at the same instant (ember and snmf do, by design
# -- they have no dependency between them), both snapshot_download the same
# files into the shared mf_outputs/.cache/huggingface/download/ staging dir.
# They then race on shutil.move of the same *.incomplete temp file and one dies
# with FileNotFoundError. That is what killed the first Golf batch (jobs
# 895126/895127); the other concepts survived only on timing luck.
#
# Doing it once up front is GPU-free, takes ~1-2 min, and leaves every job with
# a warm cache -- so the race window never opens and full ember||snmf
# parallelism is preserved.
# ---------------------------------------------------------------------------
prefetch_features() {
  echo "--- pre-flight: ensuring SNMF features are cached (no GPU) ---"
  if [[ -n "$DRY" ]]; then
    echo "      (dry run) would prefetch features for: ${CONCEPTS[*]}"
    return 0
  fi
  local py="${PYTHON:-$REPO_DIR/.venv/bin/python}"
  if ! "$py" - "${CONCEPTS[@]}" <<'PYEOF'
import sys
from ember.data import ensure_features
concepts = sys.argv[1:]
ensure_features("google/gemma-2-2b-it", concepts, 100, 42)
print(f"[prefetch] features ready for {len(concepts)} concept(s)")
PYEOF
  then
    echo "ERROR: feature prefetch failed -- not submitting. Fix this first," >&2
    echo "       otherwise the grid jobs will race on the download and die." >&2
    exit 1
  fi
}

# Echo the command in dry-run, otherwise submit and return the job id.
submit() {
  if [[ -n "$DRY" ]]; then
    # printf %q so multi-word concepts ("Culture of Greece") print quoted and
    # the line is safe to copy-paste.
    printf '      sbatch' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    echo "DRYRUN"
  else
    sbatch --parsable "$@"
  fi
}

echo "=============================================================="
echo "Submitting ${#CONCEPTS[@]} concept(s) x ${#ARMS[@]} arm(s)${DRY:+  [DRY RUN]}"
echo "  concepts: ${CONCEPTS[*]}"
echo "  arms    : ${ARMS[*]}"
echo "=============================================================="
echo
prefetch_features

total=0
for c in "${CONCEPTS[@]}"; do
  echo
  echo "--- $c ---"
  deps=()          # everything eval must wait for
  jid_ember=""

  if has_arm ember; then
    jid_ember=$(submit scripts/run_erasure_grid.sh --arms "ember" --concepts "$c")
    echo "  ember       -> $jid_ember"
    deps+=("$jid_ember"); total=$((total+1))
  fi

  if has_arm snmf; then
    jid=$(submit scripts/run_erasure_grid.sh --arms "snmf" --concepts "$c")
    echo "  snmf        -> $jid"
    deps+=("$jid"); total=$((total+1))
  fi

  if has_arm snmf_ember; then
    # must follow ember (reads its best_embed.csv)
    dep_opt=()
    [[ -n "$jid_ember" && "$jid_ember" != "DRYRUN" ]] && dep_opt=(--dependency="afterok:$jid_ember")
    [[ -n "$jid_ember" && "$jid_ember" == "DRYRUN" ]] && dep_opt=(--dependency="afterok:<ember>")
    jid=$(submit "${dep_opt[@]}" scripts/run_erasure_grid.sh --arms "snmf_ember" --concepts "$c")
    echo "  snmf_ember  -> $jid  (after ember)"
    deps+=("$jid"); total=$((total+1))
  fi

  # eval waits on every grid submitted for this concept
  dep_opt=()
  if [[ ${#deps[@]} -gt 0 ]]; then
    if [[ -n "$DRY" ]]; then
      dep_opt=(--dependency="afterok:<all grids>")
    else
      dep_opt=(--dependency="afterok:$(IFS=:; echo "${deps[*]}")")
    fi
  fi
  # NOTE: "${ARMS[*]}" (star, not @) -- run_experiment.sh's parser reads --arms
  # as ONE space-separated string. Passing "${ARMS[@]}" expands to separate
  # words and the parser dies with `unknown argument: snmf` (exit 2), which is
  # what killed eval jobs 895133/895137/895141.
  jid=$(submit "${dep_opt[@]}" scripts/run_experiment.sh \
          --concept "$c" --arms "${ARMS[*]}" --phase eval)
  echo "  eval        -> $jid  (after all grids)"
  total=$((total+1))
done

echo
echo "=============================================================="
echo "${DRY:+would submit }$total job(s)${DRY:+ (dry run -- nothing submitted)}"
[[ -z "$DRY" ]] && echo "watch:  squeue -u \$(whoami)"
echo "results: results/cross_lang_test/summary.csv"
echo "=============================================================="
