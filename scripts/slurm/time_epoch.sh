#!/bin/bash
#SBATCH --job-name="time_epoch"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Short from-scratch training run on Ego-Exo4D (paper settings, 3 epochs), only to time one epoch on one GPU.
# Submit from the upstream repo:  cd ~/dl4cv/seeing_without_pixels && mkdir -p out err && sbatch <this file>

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp
export WANDB_MODE=offline   # compute nodes have no internet

nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py --dataset egoexo4d_pretrain_longseq \
    --pose_encoding rel9d_grav --take_duration 8 --sample_dur \
    --num_gpus 1 --num_workers 6 --epochs 3 --job_name time_epoch_${SLURM_JOB_ID}
nvidia-smi --query-gpu=memory.used --format=csv
