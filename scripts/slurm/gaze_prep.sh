#!/bin/bash
#SBATCH --job-name="gaze_prep"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Experiment 3 preparation and checks on CPU (gaze_prep.py): gaze cache, 1 kHz check, donor map, coverage over
# the model's input windows, and the vestibulo-ocular alignment check.
# Usage, from ~/dl4cv/camformer after scripts/deploy.sh:  sbatch ~/dl4cv/gaze_prep.sh [steps ...]

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export TMPDIR=~/tmp

set -eo pipefail
echo "code revision $(cat REVISION 2>/dev/null || echo unknown)"
python gaze_prep.py "$@"
