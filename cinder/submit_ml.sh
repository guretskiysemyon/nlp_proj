#!/bin/bash
# Submit the native-text arms (cinder/ml_arms.py) end to end, as one dependency chain:
#   5 fit jobs (de / es / fr / ru at rank 100, en+ru+de+es+fr at rank 500; all 18 concepts)
#   -> build-ml (afterany: all fits; a missing fit leaves only that concept unbuilt)
#   -> 36 erasure jobs, cinder/run_fair.sbatch ml_joint|ml_sep <concept> (afterok: build)
#   -> report (afterany: all erasure jobs)
# Job ids: judge_rerun/Fair/ml_jobs.tsv. Logs: logs/slurm/{ml_fit,fair_cpu,fair}_<id>.out
set -euo pipefail
cd "${REPO_DIR:-/home/morg/NLP_2526b/semyong/EMBER-Embedding-Erasure}"
CONCEPTS=("Culture of Greece" "Golf" "Republic of Ireland" "Ancient Rome" "Baseball" "Uranium" "Harry Potter"
          "Cannabis" "Gambling" "Gun" "Pornography" "Valentine's Day" "World War II" "Nazism" "Heroin"
          "Halloween" "Artificial intelligence" "COVID-19 pandemic")
LOG=judge_rerun/Fair/ml_jobs.tsv
: > "$LOG"

# REUSE_FITS="de=1057648 ...": fit jobs already queued / running, used as dependencies instead of resubmitting.
# Fits ask for any 4 GPUs at submission (never change --gres of a queued job with scontrol: that drops the
# GPU allocation and the job runs on GPUs it does not own).
fits=()
for spec in "de 100" "es 100" "fr 100" "ru 100" "en+ru+de+es+fr 500"; do
  set -- $spec
  id=""
  for r in ${REUSE_FITS:-}; do [[ ${r%%=*} == "$1" ]] && id=${r#*=}; done
  [[ -n $id ]] || id=$(sbatch --parsable --gres=gpu:4 --job-name="ml_fit_$1" cinder/run_ml_fit.sbatch "$1" "$2")
  fits+=("$id"); printf 'fit\t%s r%s\t%s\n' "$1" "$2" "$id" | tee -a "$LOG"
done
build=$(sbatch --parsable --job-name=ml_build --dependency="afterany:$(IFS=:; echo "${fits[*]}")" \
        cinder/run_fair_cpu.sbatch build-ml)
printf 'build\tbuild-ml\t%s\n' "$build" | tee -a "$LOG"

runs=()
for method in ml_sep ml_joint; do
  for c in "${CONCEPTS[@]}"; do
    id=$(sbatch --parsable --job-name="fair_${method}" --dependency="afterok:$build" \
         cinder/run_fair.sbatch "$method" "$c")
    runs+=("$id"); printf 'run\t%s %s\t%s\n' "$method" "$c" "$id" | tee -a "$LOG"
  done
done
rep=$(sbatch --parsable --job-name=ml_report --dependency="afterany:$(IFS=:; echo "${runs[*]}")" \
      cinder/run_fair_cpu.sbatch report)
printf 'report\treport\t%s\n' "$rep" | tee -a "$LOG"
