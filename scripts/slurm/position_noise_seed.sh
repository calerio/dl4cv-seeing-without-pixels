#!/bin/bash
# Submits the two runs of one seed of experiment 4 (docs/position_noise.md, D7) as two GPU jobs (the queue
# allows 2 running): A, the authors' setup, and N, the same with position noise on half the training clips.
# About 3 hours of training each.
#
# Usage, on the login node after scripts/deploy.sh:
#   bash ~/dl4cv/position_noise_seed.sh smoke                  one epoch of N, then one noisy test of it
#   bash ~/dl4cv/position_noise_seed.sh <seed> [--with_tests] [--resume]
# A seed is refused until the smoke run has finished on the deployed code revision.
# With --with_tests each run gets 13 test jobs (eval_run.sh, 1 hour each) that start once its training has
# finished: one on clean inputs (the standard test, comparable with the paper) and one per evaluation noise level
# of D7 and noise seed 0, 1, 2. Each noisy test writes test_questions_<flags>.csv next to test_questions.csv.
# --resume lets a killed chain be resubmitted: existing runs continue from their last.ckpt (train.sh always
# resumes), finished ones end at once; tests whose CSV exists are not run again.
# Noise levels are in metres (Aria poses; checked on the stored files, see docs/position_noise.md).
set -euo pipefail
cd ~/dl4cv/camformer
mkdir -p out err ~/dl4cv/run_ckpts

TRAIN_SIGMA=0.005                     # D7
EVAL_SIGMAS="0.005 0.01 0.02 0.05"    # D7; 0 is the clean test
NOISE="--position_noise_train_sigma $TRAIN_SIGMA --position_noise_probability 0.5"

sub() {  # job name suffix, command, time limit
  sbatch --parsable --account="$USER" --partition=stud --qos=stud --gres=gpu:1 --cpus-per-task=6 --mem=48G \
      --time=$3 --job-name=pn_$1 --output=out/%x_%j.out --error=err/%x_%j.err --wrap="$2"
}

if [ "${1:-}" = smoke ]; then
  S="--dataset egoexo4d_pretrain_longseq --pose_encoding rel9d_grav --take_duration 8 --sample_dur --epochs 1 \
--seed 0 --val_every 1 --init_ckpt checkpoints/egoexo4d_dur8.pt"
  if ls -d ~/data/logs/*/pn_smoke_N >/dev/null 2>&1; then echo "refusing: pn_smoke_N exists; move it away first" >&2; exit 1; fi
  j=$(sub smoke "bash ~/dl4cv/train.sh $S $NOISE --job_name pn_smoke_N" 01:00:00)
  sbatch --parsable --dependency=afterok:$j ~/dl4cv/eval_run.sh pn_smoke_N \
      --position_noise_eval_sigma $TRAIN_SIGMA --position_noise_eval_seed 0
  exit 0
fi

seed=$1; shift
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "usage: $0 smoke | <seed> [--with_tests] [--resume]" >&2; exit 1; }
smoke=~/data/logs/egoexo4d_pretrain_longseq/pn_smoke_N
sf="--position_noise_eval_sigma $TRAIN_SIGMA --position_noise_eval_seed 0"
if [ ! -f "$smoke/test_questions_$(echo "$sf" | tr -c 'a-zA-Z0-9' '_' | sed 's/^_*//; s/_*$//').csv" ] || \
   [ "$(cat $smoke/REVISION_AT_START 2>/dev/null)" != "$(cat REVISION)" ]; then
  echo "refusing: no finished smoke run (with its noisy test) on code revision $(cat REVISION); run '$0 smoke'" >&2
  exit 1
fi
tests=""; resume=""
for a in "$@"; do
  case $a in --with_tests) tests=1 ;; --resume) resume=1 ;; *) echo "unknown option $a" >&2; exit 1 ;; esac
done

BASE="--dataset egoexo4d_pretrain_longseq --pose_encoding rel9d_grav --take_duration 8 --sample_dur --epochs 20 \
--seed $seed --val_every 5 --init_ckpt checkpoints/egoexo4d_dur8.pt"
A=pn_A_s$seed N=pn_N_s$seed

# Every run of a seed must start fresh on this code revision, unless --resume says the chain is being continued.
if [ -z "$resume" ]; then
  for r in $A $N; do
    if ls -d ~/data/logs/*/$r >/dev/null 2>&1; then
      echo "refusing: a log directory for $r already exists ($(ls -d ~/data/logs/*/$r)); move it away or use --resume" >&2
      exit 1
    fi
  done
fi

train() {  # job name, extra flags; prints the job id
  sub ${1#pn_} "bash ~/dl4cv/train.sh $BASE $2 --job_name $1" 05:00:00
}
tests() {  # job name, training job id: 13 test jobs that start after training
  local d=~/data/logs/egoexo4d_pretrain_longseq/$1
  [ -f $d/test_questions.csv ] || sbatch --parsable --dependency=afterok:$2 ~/dl4cv/eval_run.sh $1
  for s in $EVAL_SIGMAS; do
    for k in 0 1 2; do
      local f="--position_noise_eval_sigma $s --position_noise_eval_seed $k"
      local csv="$d/test_questions_$(echo "$f" | tr -c 'a-zA-Z0-9' '_' | sed 's/^_*//; s/_*$//').csv"  # as eval_run.sh
      [ -f $csv ] || sbatch --parsable --dependency=afterok:$2 ~/dl4cv/eval_run.sh $1 $f
    done
  done
}
ja=$(train $A '')
jn=$(train $N "$NOISE")
echo "training: $A job $ja, $N job $jn"
if [ -n "$tests" ]; then tests $A $ja; tests $N $jn; fi
