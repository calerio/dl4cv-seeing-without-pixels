#!/bin/bash
#SBATCH --job-name="eval_run"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Tests the last checkpoint of one of our training runs on the Ego-Exo4D 5-way test and scores it
# per activity type and visibility (score_mcq.py).
# Usage, from ~/dl4cv/camformer:  sbatch eval_run.sh <job_name of the training run> [extra train.py flags]
# The checkpoint is linked under a run-specific name, because the test code names its output folder
# after the checkpoint file and every run's file is called last.ckpt.
# Gaze runs (experiment 3) are tested with the gaze flags from their args.json. Extra flags come last and so
# override them (e.g. `--gaze zero`, to check that a model uses gaze); their scores go to
# test_questions_<extra flags>.csv instead of test_questions.csv.

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

set -eo pipefail
run=$1; shift
extra="$*"
dir=~/data/logs/egoexo4d_pretrain_longseq/$run
mkdir -p ~/dl4cv/run_ckpts
ckpt=~/dl4cv/run_ckpts/$run.ckpt
ln -sf "$dir/checkpoints/last.ckpt" "$ckpt"
gaze=$(python - "$dir/args.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
out = []
if a.get('gaze', 'none') != 'none':
  out.append(f"--gaze {a['gaze']}")
if a.get('gaze_input'):
  out.append('--gaze_input')
print(' '.join(out))
PY
)
echo "code revision $(cat REVISION 2>/dev/null || echo unknown); flags: $gaze $extra"
log=$(mktemp); trap 'rm -f "$log"' EXIT
python train.py --dataset egoexo4d_pretrain_longseq --test --scenario all \
    --pose_encoding rel9d_grav --take_duration 8 --sample_dur \
    --num_gpus 1 --batch_size 1000 --ckpt "$ckpt" $gaze $extra | tee "$log"

# Per-split table, plus one row per test question for paired comparisons between runs.
# The directory this job wrote, from its own log (two tests of one run may run at once).
d=$(sed -n 's/^Saved features.* to \(final_data\/.*\)$/\1/p' "$log" | tail -1)
[ -n "$d" ] || { echo "FAIL: no test features written"; exit 1; }
echo "== $d"
suffix=""
[ -n "$extra" ] && suffix="_$(echo "$extra" | tr -c 'a-zA-Z0-9' '_' | sed 's/^_*//; s/_*$//')"
python score_mcq.py "$d" --out "$dir/test_questions$suffix.csv"
