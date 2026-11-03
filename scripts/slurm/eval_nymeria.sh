#!/bin/bash
#SBATCH --job-name="eval_nym"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Zero-shot Nymeria test (paper Table 2, right columns) of one model on all four narration types, through the
# authors' path (`--dataset nymeria_pretrain --test`). Writes one row per question for compare_runs.py:
# <out dir>/nymeria_questions.csv (4,000 rows; `type` = narration type a-d, `take` = recording).
# Usage, from ~/dl4cv/camformer:
#   sbatch ~/dl4cv/eval_nymeria.sh run <job_name>      # last checkpoint of one of our experiment-1 runs
#   sbatch ~/dl4cv/eval_nymeria.sh released <name>     # checkpoints/<name>.pt, e.g. egoexo4d_dur8
# Runs trained with gaze or the domain token are refused: this path feeds neither.

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

set -eo pipefail
kind=$1 name=$2
case $kind in
  run)
    out=~/data/logs/egoexo4d_pretrain_longseq/$name
    python - "$out/args.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
if a.get('gaze', 'none') != 'none' or a.get('domain_token'):
  sys.exit(f'{sys.argv[1]}: run uses gaze or the domain token; the Nymeria path feeds neither')
PY
    mkdir -p ~/dl4cv/run_ckpts
    ckpt=~/dl4cv/run_ckpts/$name.ckpt
    ln -sf "$out/checkpoints/last.ckpt" "$ckpt"
    load="--ckpt $ckpt" ;;
  released)
    out=~/data/logs/released/$name; mkdir -p "$out"
    load="--init_ckpt checkpoints/$name.pt" ;;
  *) echo "usage: eval_nymeria.sh run|released <name>"; exit 1 ;;
esac
echo "code revision $(cat REVISION 2>/dev/null || echo unknown); model: $load"

log=$(mktemp); trap 'rm -f "$log"' EXIT
for col in a b c d; do
  python train.py --dataset nymeria_pretrain --test --text_column $col --pose_encoding rel9d_grav \
      --num_gpus 1 --batch_size 1000 $load | tee "$log"
  d=$(sed -n 's/^Saved features.* to \(final_data\/.*\)$/\1/p' "$log" | tail -1)
  [ -n "$d" ] || { echo "FAIL: no features written for column $col"; exit 1; }
  python score_nymeria.py $col "$d" --out "$out/nymeria_$col.csv"
done
python - "$out" <<'PY'
import sys
import pandas as pd
out = sys.argv[1]
q = pd.concat([pd.read_csv(f'{out}/nymeria_{c}.csv') for c in 'abcd'], ignore_index=True)
q.to_csv(f'{out}/nymeria_questions.csv', index=False)
print(f'Nymeria all four: {len(q)} questions | m2t {100 * q.m2t.mean():.2f} | t2m {100 * q.t2m.mean():.2f}')
PY
