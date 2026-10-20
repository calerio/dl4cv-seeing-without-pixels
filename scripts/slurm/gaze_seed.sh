#!/bin/bash
# Submits all runs of one seed of experiment 3 (docs/gaze.md, "Runs") as two GPU jobs (the queue allows 2
# running): GT, GT-shuf, GF on one GPU; G0, GF-shuf on the other. About 10 hours of training.
# Usage, on the login node after scripts/deploy.sh:  bash ~/dl4cv/gaze_seed.sh <seed> [--with_tests]
# Without --with_tests the runs are not scored on the test set (the pilot: seed 0 is tested only after the gate
# decision is written in docs/gaze.md). With it, each run is tested by eval_run.sh right after training.
set -euo pipefail
seed=$1
tests=${2:-}
cd ~/dl4cv/camformer
mkdir -p out err ~/dl4cv/run_ckpts

BASE="--dataset egoexo4d_pretrain_longseq --pose_encoding rel9d_grav --take_duration 8 --sample_dur --epochs 20 \
--seed $seed --val_every 5 --init_ckpt checkpoints/egoexo4d_dur8.pt"
G0=gz_G0_s$seed GT=gz_GT_s$seed GTS=gz_GTshuf_s$seed GF=gz_GF_s$seed GFS=gz_GFshuf_s$seed

# Every run of a seed must start fresh on this code revision: train.sh always resumes from an existing last.ckpt.
for r in $G0 $GT $GTS $GF $GFS; do
  if ls -d ~/data/logs/*/$r >/dev/null 2>&1; then
    echo "refusing: a log directory for $r already exists ($(ls -d ~/data/logs/*/$r)); move it away first" >&2
    exit 1
  fi
done

run() {  # job name, extra flags
  local cmd="bash ~/dl4cv/train.sh $BASE $2 --job_name $1"
  [ "$tests" = "--with_tests" ] && cmd="$cmd && bash ~/dl4cv/eval_run.sh $1"
  echo "$cmd"
}
gpu1="$(run $GT '--gaze real --gaze_input'); $(run $GTS '--gaze shuffled --gaze_input'); \
$(run $GF '--gaze real --gaze_input --freeze_encoder')"
gpu2="$(run $G0 ''); $(run $GFS '--gaze shuffled --gaze_input --freeze_encoder')"

for cmd in "$gpu1" "$gpu2"; do
  sbatch --parsable --account=3195720 --partition=stud --qos=stud --gres=gpu:1 --cpus-per-task=6 --mem=48G \
      --time=14:00:00 --job-name=gz_s$seed --output=out/%x_%j.out --error=err/%x_%j.err --wrap="$cmd"
done
