#!/bin/bash
#SBATCH --job-name="check_train_eq"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Checks that the current code trains like an older revision, which check_reproduction.sh cannot show (it only
# tests the released model). Trains one epoch of B' (experiment 1: same-take batches, verb mask, within-take loss;
# this covers every code path setup G uses) with seed 0: once with the old code and twice with the current code, on
# the same node. Two runs of the same code show how much the GPU alone makes them differ; the old and new code count
# as equivalent if they differ by no more than twice that (or both differences are below 1e-6). A fourth run, current
# code with seed 1, is the positive control: a change we know matters (another data order) must move the weights at
# least 5 times more than the GPU noise, or the check cannot tell a real change from noise and is reported as
# uninformative.
# Usage, from ~/dl4cv/camformer:  sbatch ~/dl4cv/check_training_equivalence.sh <dir with the old code> <tag>
# The old code dir is a copy of ~/dl4cv/camformer with src/camformer of the old revision on top and that revision
# in its REVISION file. Runs are written to ~/data/logs/egoexo4d_pretrain_longseq/eq_<tag>_{old,new1,new2,seed1}.

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp
export WANDB_MODE=offline

set -euo pipefail
old=${1:-} tag=${2:-}
[ -n "$old" ] && [ -n "$tag" ] && [ -f "$old/train.py" ] || {
  echo "usage: check_training_equivalence.sh <dir with the old code> <tag>"; exit 1; }
logs=~/data/logs/egoexo4d_pretrain_longseq
for r in old new1 new2 seed1; do
  [ ! -e "$logs/eq_${tag}_$r" ] || { echo "FAIL: $logs/eq_${tag}_$r exists; pick a new tag"; exit 1; }
done
new=$(pwd)
echo "node $(hostname); old code $(cat "$old/REVISION"); new code $(cat "$new/REVISION")"

flags="--dataset egoexo4d_pretrain_longseq --pose_encoding rel9d_grav --take_duration 8 --sample_dur --epochs 1
       --init_ckpt checkpoints/egoexo4d_dur8.pt --same_take_k 8 --same_take_gap 4 --neg_mask verb
       --within_take_weight 1 --num_gpus 1 --num_workers 6"
(cd "$old" && python train.py $flags --seed 0 --job_name eq_${tag}_old)
python train.py $flags --seed 0 --job_name eq_${tag}_new1
python train.py $flags --seed 0 --job_name eq_${tag}_new2
python train.py $flags --seed 1 --job_name eq_${tag}_seed1

python - "$logs/eq_${tag}" <<'PY'
import sys
import torch
base = sys.argv[1]
w = {r: torch.load(f'{base}_{r}/checkpoints/last.ckpt', map_location='cpu', weights_only=False)['state_dict']
     for r in ('old', 'new1', 'new2', 'seed1')}
for r in ('new1', 'new2', 'seed1'):
  if w[r].keys() != w['old'].keys():
    sys.exit(f'FAIL: parameter names differ between old and {r}: {sorted(set(w[r]) ^ set(w["old"]))[:10]}')

def maxdiff(a, b):
  return max((w[a][k].float() - w[b][k].float()).abs().max().item() for k in w[a])

gpu, code, ctrl = maxdiff('new1', 'new2'), maxdiff('old', 'new1'), maxdiff('new1', 'seed1')
print(f'max |weight difference| after one epoch: new1-new2 (same code) {gpu:.3g}, old-new1 (old vs new code) '
      f'{code:.3g}, new1-seed1 (positive control) {ctrl:.3g}')
if ctrl < 5 * max(gpu, 1e-6):
  print('training equivalence UNINFORMATIVE: a different seed is not clearly larger than the GPU noise')
  sys.exit(2)
ok = code <= 1e-6 if gpu <= 1e-6 else code <= 2 * gpu
print('training equivalence', 'OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
PY
