#!/bin/bash
#SBATCH --job-name="exo_test"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Tests a checkpoint on DynPose through the long-sequence task (experiment 2) and scores it:
# the 5-way test with score_exo_mcq.py, or full retrieval on the validation clips the test does not use
# (--exo_eval_split val_nontest) with eval_full_retrieval.py.
# Usage, from ~/dl4cv/camformer:  sbatch ~/dl4cv/exo_test.sh <train.py flags>
#   e.g. released exo model, native rate:
#   sbatch ~/dl4cv/exo_test.sh --pose_encoding rel9d --pose_source vipe --init_ckpt checkpoints/dynpose100k_vipe.pt

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

set -e
echo "code revision $(cat REVISION 2>/dev/null || echo unknown)"
echo "flags: $*"
start=$(mktemp); trap 'rm -f "$start"' EXIT
python train.py --dataset dynpose_longseq --test --num_gpus 1 --batch_size 1000 "$@"

f=$(find final_data/retrieval_features/ours/dynpose_longseq_exo -name frames.pt -newer "$start" | head -1)
[ -n "$f" ] || { echo "FAIL: no new test features written"; exit 1; }
d=$(dirname "$f"); echo "== $d"
if [[ "$*" == *val_nontest* ]]; then
  python eval_full_retrieval.py "$d"
else
  python score_exo_mcq.py "$d"
fi
