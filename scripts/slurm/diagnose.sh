#!/bin/bash
#SBATCH --job-name="diagnose"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Runs diagnose_within_take.py (no training) on each checkpoint given as an argument.
# Submit from ~/dl4cv/camformer:  sbatch diagnose.sh <ckpt> [<ckpt> ...]

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

echo "code revision $(cat REVISION 2>/dev/null || echo unknown)"
for ckpt in "$@"; do
  echo "== $ckpt"
  python diagnose_within_take.py "$ckpt"
done
