#!/bin/bash
# Submits all runs of one seed of experiment 2 (docs/ego_exo.md, "Runs"), each followed by its tests, as two
# GPU jobs (the queue allows 2 running): U1, S-exo, E->X on one GPU; U0, S-ego on the other. About 7 hours.
# Usage, on the login node after scripts/deploy.sh:  bash ~/dl4cv/ego_exo_seed.sh <seed> [v2]
# Two seeds in quotes ("0 2") run one after the other in the same two jobs (24 h limit), so that three seeds fit the
# queue's 4 submitted jobs: bash ego_exo_seed.sh "0 2" v2; bash ego_exo_seed.sh 1 v2.
# v2 (decision D6): all five runs again, on one code revision, with each exo clip's own frame rate
# (--exo_fps_table) instead of 30 FPS for all, under new names (ee2_*).
# E->X gets the ego flags --take_duration 8 --sample_dur too: DynPose ignores them, but the run's ego test reads its
# protocol from args.json, and without them it would score 8 s windows instead of the authors' test windows.
set -euo pipefail
seeds=$1
variant=${2:-v1}
n_seeds=$(echo $seeds | wc -w)
[ "$n_seeds" -ge 1 ] && [ "$n_seeds" -le 2 ] || { echo "one or two seeds per call" >&2; exit 1; }
[ "$variant" = v1 ] || [ "$variant" = v2 ] || { echo "variant must be v1 or v2" >&2; exit 1; }
cd ~/dl4cv/camformer
mkdir -p out err

EXO="--pose_source vipe --exo_resample_hz 20 --exo_to_aria_axes"
INIT_EGO="--init_ckpt checkpoints/egoexo4d_dur8.pt"

train() { echo "bash ~/dl4cv/train.sh $*"; }
test_longseq() { echo "bash ~/dl4cv/eval_ego_exo.sh $1 $2"; }  # dataset, run
test_sexo() {  # the authors' exo task, one row per question in the run's directory
  local dir=~/data/logs/dynpose_pretrain/$1
  echo "ln -sf $dir/checkpoints/last.ckpt ~/dl4cv/run_ckpts/$1.ckpt && OUT=$dir/test_questions_exo.csv bash ~/dl4cv/exo_test.sh --dataset dynpose_pretrain --pose_encoding rel9d --pose_source vipe --ckpt ~/dl4cv/run_ckpts/$1.ckpt"
}

prefix=ee
if [ "$variant" = v2 ]; then
  EXO="$EXO --exo_fps_table dynpose100k_fps/vipe_fps_v1.csv.gz"
  [ -f data_files/dynpose100k_fps/vipe_fps_v1.csv.gz ] || { echo "frame-rate table missing: deploy first" >&2; exit 1; }
  prefix=ee2
fi
J=egoexo4d_dynpose_longseq
mkdir -p ~/dl4cv/run_ckpts

# Every run must start fresh on this code revision (train.sh always resumes from an existing last.ckpt): check all
# runs of all requested seeds before submitting anything.
for seed in $seeds; do
  for r in ${prefix}_U0_s$seed ${prefix}_U1_s$seed ${prefix}_Sego_s$seed ${prefix}_Sexo_s$seed ${prefix}_EX_s$seed; do
    if ls -d ~/data/logs/*/$r >/dev/null 2>&1; then
      echo "refusing: a log directory for $r already exists ($(ls -d ~/data/logs/*/$r)); move it away first" >&2
      exit 1
    fi
  done
done

gpu1_all="" gpu2_all=""
for seed in $seeds; do
EGO="--pose_encoding rel9d_grav --take_duration 8 --sample_dur --epochs 20 --seed $seed --val_every 5"
U0=${prefix}_U0_s$seed U1=${prefix}_U1_s$seed SEGO=${prefix}_Sego_s$seed SEXO=${prefix}_Sexo_s$seed EX=${prefix}_EX_s$seed
gpu1="$(train --dataset $J $EGO $EXO $INIT_EGO --domain_token --job_name $U1) && $(test_longseq $J $U1); \
$(train --dataset dynpose_pretrain --pose_encoding rel9d --pose_source vipe --epochs 20 --seed $seed --val_every 5 \
  --init_ckpt checkpoints/dynpose100k_vipe.pt --job_name $SEXO) && $(test_sexo $SEXO); \
$(train --dataset dynpose_longseq --pose_encoding rel9d_grav --take_duration 8 --sample_dur --epochs 20 --seed $seed \
  --val_every 5 $EXO $INIT_EGO --job_name $EX) && $(test_longseq dynpose_longseq $EX)"
gpu2="$(train --dataset $J $EGO $EXO $INIT_EGO --job_name $U0) && $(test_longseq $J $U0); \
$(train --dataset egoexo4d_pretrain_longseq $EGO $INIT_EGO --job_name $SEGO) && $(test_longseq egoexo4d_pretrain_longseq $SEGO) ego"
gpu1_all="${gpu1_all:+$gpu1_all; }$gpu1" gpu2_all="${gpu2_all:+$gpu2_all; }$gpu2"
done

for cmd in "$gpu1_all" "$gpu2_all"; do
  sbatch --parsable --account="$USER" --partition=stud --qos=stud --gres=gpu:1 --cpus-per-task=6 --mem=48G \
      --time=$((12 * n_seeds)):00:00 --job-name=${prefix}_s$(echo $seeds | tr ' ' '+') --output=out/%x_%j.out \
      --error=err/%x_%j.err --wrap="$cmd"
done
