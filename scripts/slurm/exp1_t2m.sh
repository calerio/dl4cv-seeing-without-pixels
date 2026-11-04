#!/bin/bash
#SBATCH --job-name=exp1_t2m
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
# Experiment 1, text-to-motion (planned in docs/hard_negatives.md): per-seed accuracies and the three comparisons.
# Usage, from ~/dl4cv/camformer:  sbatch ~/dl4cv/exp1_t2m.sh   (job 737624 ran this script as audit/t2m_exp1.sh)
module load sw/miniconda3; eval "$(conda shell.bash hook)"; conda activate camformer
set -e
echo "code revision $(cat REVISION)"
E=~/data/logs/egoexo4d_pretrain_longseq
seeds() { for s in 0 1 2; do echo -n "$E/p1_$1_s$s/test_questions.csv "; done; }
python - <<PY
import pandas as pd
for x in ("A","C","Bp"):
  for s in range(3):
    q = pd.read_csv("$E/p1_%s_s%d/test_questions.csv" % (x, s))
    print(x, s, "t2m all %.2f" % (100*q.t2m.mean()), " ".join("%s %.2f" % (t, 100*g.t2m.mean()) for t, g in q.groupby("type")))
PY
for c in "A Bp" "C Bp" "A C"; do set -- $c; echo "== t2m $2 - $1"; python compare_runs.py --metric t2m --a $(seeds $1) --b $(seeds $2); done
