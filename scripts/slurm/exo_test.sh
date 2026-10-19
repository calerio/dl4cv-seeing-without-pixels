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

# Tests a checkpoint on DynPose and scores it. Default: through the long-sequence task (experiment 2); with
# --dataset dynpose_pretrain: through the authors' exo task (as for S-exo). Scoring: the 5-way test with
# score_exo_mcq.py, or full retrieval on the validation clips the test does not use
# (--exo_eval_split val_nontest) with eval_full_retrieval.py.
# Usage, from ~/dl4cv/camformer:  [OUT=<per-question.csv>] sbatch ~/dl4cv/exo_test.sh <train.py flags>
#   e.g. released exo model, native rate:
#   sbatch ~/dl4cv/exo_test.sh --pose_encoding rel9d --pose_source vipe --init_ckpt checkpoints/dynpose100k_vipe.pt
#   the same through the authors' exo task: add --dataset dynpose_pretrain (must give 46.30, results/reproduction.md)

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

set -eo pipefail
echo "code revision $(cat REVISION 2>/dev/null || echo unknown)"
echo "flags: $*"
log=$(mktemp); trap 'rm -f "$log"' EXIT
python train.py --dataset dynpose_longseq --test --num_gpus 1 --batch_size 1000 "$@" | tee "$log"

# The directory this job wrote (from its own log, so jobs running side by side cannot mix up their files).
d=$(sed -n 's/^Saved features.* to \(final_data\/.*\)$/\1/p' "$log" | tail -1)
[ -n "$d" ] || { echo "FAIL: no test features written"; exit 1; }
echo "== $d"
if [[ "$*" == *val_nontest* ]]; then
  python eval_full_retrieval.py "$d"
else
  python score_exo_mcq.py "$d" ${OUT:+--out "$OUT"}
fi
