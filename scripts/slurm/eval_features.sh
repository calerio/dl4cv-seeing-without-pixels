#!/bin/bash
#SBATCH --job-name="eval_features"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Reproduces the paper's retrieval numbers from the released precomputed features (no GPU).
# Submit from the upstream repo:  cd ~/dl4cv/seeing_without_pixels && mkdir -p out err && sbatch <this file>

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

for split in egoexo4d dynpose_original dynpose_vipe nymeria_a nymeria_b nymeria_c nymeria_d; do
  echo "== $split"
  python eval_retrieval.py "retrieval_features/$split"
done
