#!/bin/bash
# Submits the runs of one seed of experiment 3 (docs/gaze.md, "Runs") as two GPU jobs (the queue allows 2
# running): GT, GT-shuf, GF on one GPU; G0, GF-shuf on the other. About 10 hours of training.
#
# Usage, on the login node after scripts/deploy.sh:
#   bash ~/dl4cv/gaze_seed.sh smoke                    one-epoch smoke test of GT and GF only (times them; no gate)
#   bash ~/dl4cv/gaze_seed.sh <seed> [--with_tests] [--resume]
# A full seed is refused until the alignment decision is recorded: data_files/egoexo4d/gaze_alignment_decision.txt
# must exist in the deployed code (written, with a date, only after the decision in docs/gaze.md "Checks before
# training" is made). Without --with_tests the runs are not scored on the test set (the pilot: seed 0 is tested
# only after the gate decision is written in docs/gaze.md). With it, each run is tested by eval_run.sh right after
# training. --resume lets a killed chain be resubmitted: existing runs continue from their last.ckpt (train.sh
# always resumes), finished ones end at once.
set -euo pipefail
cd ~/dl4cv/camformer
mkdir -p out err ~/dl4cv/run_ckpts

sub() {  # job name suffix, command, time limit
  sbatch --parsable --account="$USER" --partition=stud --qos=stud --gres=gpu:1 --cpus-per-task=6 --mem=48G \
      --time=$3 --job-name=gz_$1 --output=out/%x_%j.out --error=err/%x_%j.err --wrap="$2"
}

if [ "${1:-}" = smoke ]; then
  S="--dataset egoexo4d_pretrain_longseq --pose_encoding rel9d_grav --take_duration 8 --sample_dur --epochs 1 \
--seed 0 --val_every 1 --init_ckpt checkpoints/egoexo4d_dur8.pt --gaze real --gaze_input"
  for r in gz_smoke_GT gz_smoke_GF; do
    if ls -d ~/data/logs/*/$r >/dev/null 2>&1; then echo "refusing: $r exists; move it away first" >&2; exit 1; fi
  done
  sub smoke "bash ~/dl4cv/train.sh $S --job_name gz_smoke_GT; bash ~/dl4cv/train.sh $S --freeze_encoder --job_name gz_smoke_GF" 01:30:00
  exit 0
fi

seed=$1; shift
tests=""; resume=""
for a in "$@"; do
  case $a in --with_tests) tests=1 ;; --resume) resume=1 ;; *) echo "unknown option $a" >&2; exit 1 ;; esac
done
if [ ! -s data_files/egoexo4d/gaze_alignment_decision.txt ]; then
  echo "refusing: no alignment decision recorded (data_files/egoexo4d/gaze_alignment_decision.txt); see docs/gaze.md" >&2
  exit 1
fi

BASE="--dataset egoexo4d_pretrain_longseq --pose_encoding rel9d_grav --take_duration 8 --sample_dur --epochs 20 \
--seed $seed --val_every 5 --init_ckpt checkpoints/egoexo4d_dur8.pt"
G0=gz_G0_s$seed GT=gz_GT_s$seed GTS=gz_GTshuf_s$seed GF=gz_GF_s$seed GFS=gz_GFshuf_s$seed

# Every run of a seed must start fresh on this code revision (train.sh always resumes from an existing last.ckpt),
# unless --resume says the chain is being continued on purpose.
if [ -z "$resume" ]; then
  for r in $G0 $GT $GTS $GF $GFS; do
    if ls -d ~/data/logs/*/$r >/dev/null 2>&1; then
      echo "refusing: a log directory for $r already exists ($(ls -d ~/data/logs/*/$r)); move it away or use --resume" >&2
      exit 1
    fi
  done
fi

run() {  # job name, extra flags
  # train.py records the code revision a run started on and refuses to resume it on another one, so no run mixes
  # two code versions. A finished run is not tested again.
  local d=~/data/logs/egoexo4d_pretrain_longseq/$1
  local cmd="bash ~/dl4cv/train.sh $BASE $2 --job_name $1"
  [ -n "$tests" ] && cmd="$cmd && { [ -f $d/test_questions.csv ] || bash ~/dl4cv/eval_run.sh $1; }"
  echo "$cmd"
}
gpu1="$(run $GT '--gaze real --gaze_input'); $(run $GTS '--gaze shuffled --gaze_input'); \
$(run $GF '--gaze real --gaze_input --freeze_encoder')"
gpu2="$(run $G0 ''); $(run $GFS '--gaze shuffled --gaze_input --freeze_encoder')"
for cmd in "$gpu1" "$gpu2"; do sub s$seed "$cmd" 14:00:00; done
