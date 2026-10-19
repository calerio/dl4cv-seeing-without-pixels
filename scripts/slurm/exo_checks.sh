#!/bin/bash
#SBATCH --job-name="exo_checks"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# CPU checks for experiment 2 (exo_checks.py): clip lengths, frame rate, speed, axes.
# Usage, after scripts/deploy.sh:  cd ~/dl4cv/camformer && sbatch ~/dl4cv/exo_checks.sh

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

echo "code revision $(cat REVISION 2>/dev/null || echo unknown)"
python exo_checks.py "$@"
