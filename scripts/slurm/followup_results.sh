#!/bin/bash
#SBATCH --job-name="followup_results"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Every number in results/followups.md, from the per-question files written by followup_evals.sh (CPU only).
# Usage, from ~/dl4cv/camformer:  sbatch ~/dl4cv/followup_results.sh

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer
set -euo pipefail
echo "code revision $(cat REVISION)"
E=~/data/logs/egoexo4d_pretrain_longseq
R=~/data/logs/released
f() { for s in 0 1 2; do echo -n "$E/p1_$1_s$s/$2 "; done; }

echo "== per-run accuracies"
python - "$E" "$R" <<'PY'
import sys
import pandas as pd
E, R = sys.argv[1:]
for x in ('A', 'C', 'Bp', 'G'):
  for s in range(3):
    q = pd.read_csv(f'{E}/p1_{x}_s{s}/test_questions.csv')
    by = {t: 100 * g.m2t.mean() for t, g in q.groupby('type')}
    byt = {t: 100 * g.t2m.mean() for t, g in q.groupby('type')}
    print(f"egoexo {x} s{s} m2t all {100*q.m2t.mean():.2f} proc {by['procedural']:.2f} phys {by['physical']:.2f}"
          f" | t2m all {100*q.t2m.mean():.2f} proc {byt['procedural']:.2f} phys {byt['physical']:.2f}")
names = {'a': 'body', 'b': 'hands', 'c': 'legs', 'd': 'focus'}
def nym(path, label):
  q = pd.read_csv(path)
  per = ' '.join(f"{names[t]} {100*g.m2t.mean():.2f}" for t, g in q.groupby('type'))
  print(f"nymeria {label} m2t {per} all {100*q.m2t.mean():.2f} | t2m all {100*q.t2m.mean():.2f} | sessions {q['take'].nunique()}")
for r in ('egoexo4d_dur16', 'egoexo4d_dur8'):
  nym(f'{R}/{r}/nymeria_questions.csv', r)
for x in ('A', 'C', 'Bp', 'G'):
  for s in range(3):
    nym(f'{E}/p1_{x}_s{s}/nymeria_questions.csv', f'{x} s{s}')
PY

cmp() { echo; echo "== $1"; shift; python compare_runs.py "$@"; }
cmp "Ego-Exo4D m2t B' - G" --a $(f G test_questions.csv) --b $(f Bp test_questions.csv)
cmp "Ego-Exo4D m2t G - C" --a $(f C test_questions.csv) --b $(f G test_questions.csv)
cmp "Ego-Exo4D m2t G - A (descriptive)" --a $(f A test_questions.csv) --b $(f G test_questions.csv)
cmp "Ego-Exo4D t2m B' - G (descriptive)" --metric t2m --a $(f G test_questions.csv) --b $(f Bp test_questions.csv)
cmp "Ego-Exo4D t2m G - C (descriptive)" --metric t2m --a $(f C test_questions.csv) --b $(f G test_questions.csv)
cmp "Nymeria m2t B' - A" --a $(f A nymeria_questions.csv) --b $(f Bp nymeria_questions.csv)
cmp "Nymeria m2t G - A" --a $(f A nymeria_questions.csv) --b $(f G nymeria_questions.csv)
echo "followup results done"
