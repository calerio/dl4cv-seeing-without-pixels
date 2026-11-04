#!/bin/bash
#SBATCH --job-name="eval_nym"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Zero-shot Nymeria test (paper Table 2, right columns) of one model on all four narration types, through the
# authors' path (`--dataset nymeria_pretrain --test`). Writes one row per question for compare_runs.py:
# <out dir>/nymeria_questions.csv (4,000 rows; `type` = narration type a-d, `take` = recording).
# Usage, from ~/dl4cv/camformer:
#   sbatch ~/dl4cv/eval_nymeria.sh run <job_name>      # last checkpoint of one of our experiment-1 runs
#   sbatch ~/dl4cv/eval_nymeria.sh released <name>     # checkpoints/<name>.pt, e.g. egoexo4d_dur8
# Only finished runs with the authors' architecture (experiment 1) are accepted: this path feeds no gaze or domain
# token, and builds the model from train.py's defaults. The Nymeria loader reads the pose encoding and sample rate
# from the flags, not the clip duration (`--take_duration`, `--sample_dur` set Ego-Exo4D training windows only).

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

set -euo pipefail
kind=${1:-} name=${2:-}
[ -n "$kind" ] && [ -n "$name" ] || { echo "usage: eval_nymeria.sh run|released <name>"; exit 1; }
case $kind in
  run)
    out=~/data/logs/egoexo4d_pretrain_longseq/$name
    python - "$out/args.json" "$out/checkpoints/last.ckpt" <<'PY'
import json, sys
import torch
a = json.load(open(sys.argv[1]))
# The values the Nymeria test below builds the model with (train.py defaults, plus --pose_encoding rel9d_grav).
expected = {'d_model': 128, 'nhead': 4, 'num_layers': 4, 'dim_feedforward': 256,
            'pose_encoding': 'rel9d_grav', 'sample_rate': 1, 'gaze': 'none'}
odd = {k: a.get(k) for k in expected if a.get(k, expected[k]) != expected[k]}
odd.update({k: a[k] for k in ('gaze_input', 'domain_token', 'freeze_encoder', 'use_learnable_gravity',
                              'use_scenario_label', 'hier_text') if a.get(k)})
if odd:
  sys.exit(f'{sys.argv[1]}: run trained with {odd}; the Nymeria test would score a different model')
epoch = torch.load(sys.argv[2], map_location='cpu', weights_only=False)['epoch']
if epoch + 1 != a['epochs']:
  sys.exit(f'{sys.argv[2]}: checkpoint of epoch {epoch + 1} of {a["epochs"]}; training has not finished')
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
      --num_gpus 1 --batch_size 1000 --job_name nym_${name}_$col $load | tee "$log"
  d=$(sed -n 's/^Saved features.* to \(final_data\/.*\)$/\1/p' "$log" | tail -1)
  [ -n "$d" ] || { echo "FAIL: no features written for column $col"; exit 1; }
  python score_nymeria.py $col "$d" --out "$out/nymeria_$col.csv"
done
python - "$out" <<'PY'
import sys
import pandas as pd
out = sys.argv[1]
q = pd.concat([pd.read_csv(f'{out}/nymeria_{c}.csv') for c in 'abcd'], ignore_index=True)
if len(q) != 4000 or q.question.nunique() != 4000 or sorted(q.type.unique()) != list('abcd'):
  sys.exit(f'FAIL: expected 4,000 questions over types a-d, got {len(q)}')
q.to_csv(f'{out}/nymeria_questions.csv', index=False)
print(f'Nymeria all four: {len(q)} questions | m2t {100 * q.m2t.mean():.2f} | t2m {100 * q.t2m.mean():.2f}')
PY
