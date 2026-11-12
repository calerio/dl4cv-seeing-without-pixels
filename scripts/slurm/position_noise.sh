#!/bin/bash
#SBATCH --job-name=position_noise
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
set -euo pipefail
# From ~/dl4cv/camformer:
# sbatch ../position_noise.sh smoke exp4_v1 0.005 0
# After smoke checks pass:
# POSITION_UNITS_CONFIRMED=1 sbatch ../position_noise.sh train exp4_v1 0.005 0
# POSITION_UNITS_CONFIRMED=1 sbatch ../position_noise.sh eval exp4_v1 0.005 0
# Repeat smoke/train/eval for training seeds 1,2.
# Positions confirmed in metres. Training sigma 0.005 = 0.5 cm per coordinate.
module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer
cd "${CAMFORMER_CODE_DIR:-$HOME/dl4cv/camformer}"
export CAMFORMER_DATA="$HOME/dl4cv/data/camformer_data"
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export TMPDIR="$HOME/tmp"
export WANDB_MODE=offline
mkdir -p "$TMPDIR"
action=${1:?smoke/train/eval}; prefix=${2:?unique experiment prefix}; sigma=${3:?sigma in raw units}; seed=${4:-0}
[[ "$prefix" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid prefix'; exit 1; }
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'Invalid seed'; exit 1; }
python - "$sigma" <<'PY'
import sys
from utils.position_noise import validate_noise
s = float(sys.argv[1]); validate_noise(s)
assert s > 0, 'choose a positive training sigma'
PY
[ -s REVISION ] || { echo 'Deploy a committed revision with scripts/deploy.sh first'; exit 1; }
common=(--dataset egoexo4d_pretrain_longseq --pose_encoding rel9d_grav --take_duration 8 --test_take_duration 8 --sample_dur --batch_size 1024 --lr 0.0001 --num_gpus 1 --num_workers 6 --seed "$seed" --init_ckpt checkpoints/egoexo4d_dur8.pt)
base="$HOME/data/logs/egoexo4d_pretrain_longseq"
case "$action" in
 smoke)
   python train.py "${common[@]}" --epochs 1 --val_every 1 --job_name "${prefix}_smoke_s${seed}" --position_noise_train_sigma "$sigma" --position_noise_probability 0.5 --resume
   python check_finished.py "$base/${prefix}_smoke_s${seed}"
   ;;
 train)
   [ "${POSITION_UNITS_CONFIRMED:-0}" = 1 ] || { echo 'Confirm position units first'; exit 1; }
   python check_finished.py "$base/${prefix}_smoke_s${seed}"
   python - "$base/${prefix}_smoke_s${seed}" "$sigma" <<'PYSMOKE'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
a = json.loads((run / 'args.json').read_text())
assert a.get('position_noise_train_sigma') == float(sys.argv[2]), 'smoke sigma differs'
assert a.get('position_noise_probability') == 0.5, 'smoke probability differs'
assert (run / 'REVISION_AT_START').read_text().strip() == pathlib.Path('REVISION').read_text().strip(), 'smoke revision differs'
PYSMOKE
   python train.py "${common[@]}" --epochs 20 --val_every 1 --job_name "${prefix}_A_s${seed}" --resume
   python check_finished.py "$base/${prefix}_A_s${seed}"
   python train.py "${common[@]}" --epochs 20 --val_every 1 --job_name "${prefix}_N_s${seed}" --position_noise_train_sigma "$sigma" --position_noise_probability 0.5 --resume
   python check_finished.py "$base/${prefix}_N_s${seed}"
   ;;
 eval)
   [ "${POSITION_UNITS_CONFIRMED:-0}" = 1 ] || { echo 'Confirm position units first'; exit 1; }
   for setup in A N; do
          python eval_position_noise.py "$base/${prefix}_${setup}_s${seed}" --sigmas 0 0.005 0.01 0.02 0.05 --noise-seeds 0 1 2
   done
   ;;
 *) echo 'Expected smoke, train or eval'; exit 1;;
esac
