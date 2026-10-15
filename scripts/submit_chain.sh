#!/bin/bash
# Queue N copies of a training job, each starting when the previous one ends (for any reason),
# so a run longer than the 24-hour job limit continues where it stopped.
# Usage, from the code directory on the cluster:
#   submit_chain.sh N <job script> [sbatch options --] <train.py arguments, including --job_name>
# The stud queue accepts at most 4 submitted jobs per user.
set -euo pipefail
n=$1; script=$2; shift 2
sbatch_opts=()
if [[ " $* " == *" -- "* ]]; then
  while [[ $1 != "--" ]]; do sbatch_opts+=("$1"); shift; done
  shift
fi
prev=""
for i in $(seq 1 "$n"); do
  dep=()
  [[ -n $prev ]] && dep=(--dependency=afterany:$prev)
  prev=$(sbatch --parsable --account="$(id -un)" "${dep[@]}" "${sbatch_opts[@]}" "$script" "$@")
  echo "part $i of $n: job $prev"
done
