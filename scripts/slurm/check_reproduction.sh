#!/bin/bash
#SBATCH --job-name="check_repro"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Regression check: our code must still reproduce Table 2 of the paper from the released checkpoint
# (motion-to-text 44.81 overall on the Ego-Exo4D 5-way test). Run it after any change to the training,
# data or evaluation code, before spending GPU time on new runs. Fails (exit 1) on any difference.
# Usage, after scripts/deploy.sh:  cd ~/dl4cv/camformer && sbatch ~/dl4cv/check_reproduction.sh [extra train.py flags]
# Extra flags must leave the released model unchanged at step 0 (e.g. --domain_token, which starts at zero).

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

set -e
echo "code revision $(cat REVISION 2>/dev/null || echo unknown)"
start=$(mktemp); trap 'rm -f "$start"' EXIT
python train.py --dataset egoexo4d_pretrain_longseq --test --scenario all \
    --pose_encoding rel9d_grav --take_duration 8 --sample_dur \
    --num_gpus 1 --batch_size 1000 --init_ckpt checkpoints/egoexo4d_dur8.pt "$@"

f=$(find final_data/retrieval_features/ours/egoexo4d_pretrain_longseq -name frames.pt -newer "$start" | head -1)
[ -n "$f" ] || { echo "FAIL: no new test features written"; exit 1; }
out=$(mktemp)
python score_mcq.py "$(dirname "$f")" --out "$out"
python - "$out" <<'PY'
import sys
import pandas as pd
q = pd.read_csv(sys.argv[1])
expected = {'all': 44.81, 'procedural': 33.61}
got = {'all': 100 * q.m2t.mean()}
for t, g in q.groupby('type'):
  got[t] = 100 * g.m2t.mean()
bad = {k: round(got[k], 2) for k in expected if abs(got[k] - expected[k]) > 0.005}
print('reproduction', 'FAIL ' + str(bad) if bad else 'OK', {k: round(v, 2) for k, v in got.items()})
sys.exit(1 if bad else 0)
PY
