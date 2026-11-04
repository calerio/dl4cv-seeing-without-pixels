#!/bin/bash
#SBATCH --job-name=followup_evals
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
# Tests for docs/followups.md: G on the Ego-Exo4D test, and every model on Nymeria (scored by session).
set -e
for s in 0 1 2; do bash ~/dl4cv/eval_run.sh p1_G_s$s; done
bash ~/dl4cv/eval_nymeria.sh released egoexo4d_dur16
bash ~/dl4cv/eval_nymeria.sh released egoexo4d_dur8
for s in 0 1 2; do for x in A C Bp G; do bash ~/dl4cv/eval_nymeria.sh run p1_${x}_s$s; done; done
echo "followup evals done"
