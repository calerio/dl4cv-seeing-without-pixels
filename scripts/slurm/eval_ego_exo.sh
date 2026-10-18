#!/bin/bash
#SBATCH --job-name="eval_ego_exo"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Tests the last checkpoint of an experiment-2 run on both test sets (Ego-Exo4D and DynPose), with the
# model and input flags the run was trained with (read from its args.json), and writes one row per
# question for compare_runs.py: <run dir>/test_questions_ego.csv and test_questions_exo.csv.
# Usage, from ~/dl4cv/camformer:  sbatch ~/dl4cv/eval_ego_exo.sh <dataset of the run> <job_name of the run>

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
dataset=$1 run=$2
dir=~/data/logs/$dataset/$run
mkdir -p ~/dl4cv/run_ckpts
ckpt=~/dl4cv/run_ckpts/$run.ckpt
ln -sf "$dir/checkpoints/last.ckpt" "$ckpt"
# Flags that change the model or its inputs, as the run used them.
flags=$(python - "$dir/args.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
out = [f"--pose_encoding {a['pose_encoding']}", f"--pose_source {a.get('pose_source', 'original')}",
       f"--take_duration {a['take_duration']}", f"--exo_resample_hz {a.get('exo_resample_hz', 0.0)}"]
for k in ('sample_dur', 'domain_token', 'exo_to_aria_axes'):
  if a.get(k):
    out.append(f'--{k}')
print(' '.join(out))
PY
)
echo "flags: $flags"

for domain in ego exo; do
  start=$(mktemp)
  python train.py --dataset egoexo4d_dynpose_longseq --test_domain $domain --test --scenario all \
      --num_gpus 1 --batch_size 1000 --ckpt "$ckpt" $flags
  if [ $domain = ego ]; then
    f=$(find final_data/retrieval_features/ours/egoexo4d_dynpose_longseq -path "*/$run/*" -name frames.pt -newer "$start" | head -1)
    python score_mcq.py "$(dirname "$f")" --out "$dir/test_questions_ego.csv"
  else
    f=$(find final_data/retrieval_features/ours/egoexo4d_dynpose_longseq_exo -path "*/$run/*" -name frames.pt -newer "$start" | head -1)
    python score_exo_mcq.py "$(dirname "$f")" --out "$dir/test_questions_exo.csv"
  fi
  rm -f "$start"
done
