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
# Usage, from ~/dl4cv/camformer:  sbatch eval_run.sh <job_name of the training run>
# The checkpoint is linked under a run-specific name, because the test code names its output folder
# after the checkpoint file and every run's file is called last.ckpt.

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

run=$1
mkdir -p ~/dl4cv/run_ckpts
ckpt=~/dl4cv/run_ckpts/$run.ckpt
ln -sf ~/data/logs/egoexo4d_pretrain_longseq/$run/checkpoints/last.ckpt "$ckpt"
python train.py --dataset egoexo4d_pretrain_longseq --test --scenario all \
    --pose_encoding rel9d_grav --take_duration 8 --sample_dur \
    --num_gpus 1 --batch_size 1000 --ckpt "$ckpt"

# Per-split table, plus one row per test question for paired comparisons between runs.
d=$(dirname "$(find final_data/retrieval_features/ours/egoexo4d_pretrain_longseq -path "*/$run/*" -name frames.pt | head -1)")
echo "== $d"
python score_mcq.py "$d" --out ~/data/logs/egoexo4d_pretrain_longseq/$run/test_questions.csv
