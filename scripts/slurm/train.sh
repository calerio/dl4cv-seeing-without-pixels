#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# One training job that continues from the run's last checkpoint if there is one.
# Arguments are passed to train.py; always give a fixed --job_name so restarts find the same run.
# Use scripts/submit_chain.sh to queue several of these back to back.

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp
export WANDB_MODE=offline   # compute nodes have no internet

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
python train.py --num_gpus 1 --num_workers 6 --resume "$@"
