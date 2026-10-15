#!/bin/bash
#SBATCH --job-name="eval_ckpt"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Runs the released Ego-Exo4D checkpoint (8-second clips) on the test set and scores retrieval.
# The CLIP text model must already be in ~/.cache/clip (compute nodes have no internet).
# Submit from the upstream repo:  cd ~/dl4cv/seeing_without_pixels && mkdir -p out err && sbatch <this file>

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer

export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export TMPDIR=~/tmp

nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py --dataset egoexo4d_pretrain_longseq --test --scenario all \
    --pose_encoding rel9d_grav --take_duration 8 --sample_dur \
    --num_gpus 1 --batch_size 1000 \
    --init_ckpt checkpoints/egoexo4d_dur8.pt

# The features land in final_data/retrieval_features/ours/egoexo4d_pretrain_longseq/<...>/frames.pt and text.pt.
for f in $(find final_data/retrieval_features/ours/egoexo4d_pretrain_longseq -name frames.pt -newer checkpoints/egoexo4d_dur8.pt); do
  d=$(dirname "$f"); echo "== $d"; python eval_retrieval.py "$d"
done
