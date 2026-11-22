# What a Camera's Path Knows About Activity

Juan Ignacio Arguello, Valerio Costa, Facundo Lucero

Group project for 20600 Deep Learning for Computer Vision, Bocconi University, 2026-27

[Report (PDF)](report/main.pdf) | [Original paper](https://arxiv.org/abs/2511.21681) | [Original code](https://github.com/google-deepmind/seeing_without_pixels) | [Results](results/)

---

*Seeing without Pixels* (Xue et al., CVPR 2026) recognises what a person is doing from the path of their
head-mounted camera alone, without any image. Its model, CamFormer, is a four-layer
Transformer that reads a camera trajectory and matches it against text descriptions through a frozen CLIP text
encoder. We reproduce its main result exactly and then test four changes, each aimed at a limitation the authors
list or at how the model would behave on real tracking data.

## Results

All numbers are motion-to-text accuracy (%) on five-way multiple-choice questions, where chance is 20%. Every
experiment fine-tunes the authors' released checkpoint for 20 epochs, uses three training seeds unless noted, and is
compared with a baseline that gets the same extra training (the frozen-encoder gaze run is compared with the
released checkpoint). Intervals are 95% paired bootstrap intervals over seeds
and test recordings.

| Experiment | Question | Result |
|---|---|---|
| Reproduction | Does the released code give the paper's number? | Yes: 44.81 on Ego-Exo4D, matching the paper's 44.8 |
| 1. Same-recording wrong answers | Does training against wrong answers from the same recording help on procedural tasks? | Yes: procedural 33.36 → 36.91, +3.55 (+2.37 to +4.83), in every seed |
| 2. One model for both viewpoints | Can one model handle first-person and third-person trajectories? | Matches the first-person specialist (+0.03, −0.61 to +0.68); on third-person trajectories (DynPose-100K, 1,000 questions) −3.2, an interval too wide to resolve |
| 3. Eye gaze as a second input | Does adding where the person looks help? | No confirmed gain (one seed; +0.72, −0.39 to +1.84 on procedural) |
| 4. Noise on the camera positions | How robust is the model to tracking noise, and does training with noise help? | Accuracy holds up to 1 cm of noise per coordinate and drops about 5 points at 5 cm; training with noise gives no measurable gain (+0.21 averaged over noise levels, −0.08 to +0.51) |

Details are in the [report](report/main.pdf). Each experiment has a results file with the exact commands,
runs, job ids and code revisions behind every number:
[reproduction](results/reproduction.md), [1](results/hard_negatives.md) (plus [follow-ups](results/followups.md)),
[2](results/ego_exo.md), [3](results/gaze.md), [4](results/position_noise.md).

## Method

CamFormer turns each camera pose into a token, runs the sequence through a Transformer (4 layers, width 128,
about 0.6 M parameters) and trains it with a symmetric contrastive loss against frozen CLIP text embeddings of
narrations. Our additions, all behind command-line flags whose defaults give the authors' setup:

- `--same_take_k 8 --same_take_gap 4 --neg_mask verb --within_take_weight 1`: fill batches with groups of clips
  from the same recording and add a loss term that contrasts each clip only with its own recording (experiment 1).
- `--dataset egoexo4d_dynpose_longseq`, optionally with `--domain_token`: one model trained on ego and exo
  trajectories together (experiment 2).
- `--gaze`, optionally with `--freeze_encoder`: eye gaze as an extra per-frame input, training only the gaze layer
  or the whole model (experiment 3).
- `--position_noise_train_sigma` / `--position_noise_eval_sigma`: Gaussian noise on camera positions (experiment 4).

Every change to the authors' code is listed file by file in [`src/camformer/CHANGES.md`](src/camformer/CHANGES.md),
and every choice that affects the comparison with the paper is in [`docs/decisions.md`](docs/decisions.md).

## Setup

The code in `src/camformer/` is the authors' release with our changes on top, so their setup applies:

```bash
cd src/camformer
conda env create -f environment.yml
conda activate camformer
```

Camera trajectories, checkpoints and precomputed features come from the authors' release (see
[`src/camformer/DATA.md`](src/camformer/DATA.md)). Eye gaze and annotations come from Ego-Exo4D under its own
licence and are not redistributed here. Where the data sits and how it was fetched: [`data/README.md`](data/README.md).

## Reproducing the numbers

From `src/camformer/`, check the released checkpoint first (one GPU):

```bash
python train.py --dataset egoexo4d_pretrain_longseq --test --scenario all \
    --pose_encoding rel9d_grav --take_duration 8 --sample_dur \
    --num_gpus 1 --batch_size 1000 --init_ckpt checkpoints/egoexo4d_dur8.pt
python score_mcq.py <feature_dir>   # the folder train.py writes frames.pt to; expect 44.81 overall, 33.61 procedural
```

We ran everything on the Bocconi HPC cluster with SLURM. The job scripts in [`scripts/slurm/`](scripts/slurm/)
train and evaluate one setup and seed each (for example `position_noise_seed.sh`, `ego_exo_seed.sh`,
`gaze_seed.sh`); [`check_reproduction.sh`](scripts/slurm/check_reproduction.sh) is the check above, which we
reran before every new batch of training. Paired comparisons between runs use
[`compare_runs.py`](src/camformer/compare_runs.py).

## Repository layout

```
src/camformer/   the authors' code (Apache 2.0) with our changes; CHANGES.md lists them
scripts/         SLURM job scripts (slurm/), deployment to the cluster, small analysis scripts
results/         one file per experiment: tables, commands, runs, job ids, code revisions
docs/            design of each experiment, written before its runs, and the decisions log
data/            where the data comes from (the data itself is not in git)
report/          LaTeX source and PDF of the report (CVPR 2026 template)
```

How the team worked on the code: [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`REVIEW.md`](REVIEW.md). Dates and
progress: [`SCHEDULE.md`](SCHEDULE.md) and [`LOG.md`](LOG.md).

## Citation

This project builds on:

```bibtex
@inproceedings{xue2026seeing,
  title     = {Seeing without Pixels: Perception from Camera Trajectories},
  author    = {Xue, Zihui and Grauman, Kristen and Damen, Dima and Zisserman, Andrew and Han, Tengda},
  booktitle = {CVPR},
  year      = {2026}
}
```

## License

Apache 2.0, the same licence as the original code ([`LICENSE`](LICENSE)).
