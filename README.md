# Seeing without Pixels, extended

Group project for 20600 Deep Learning for Computer Vision (Bocconi, a.y. 2026-27).

**Team:** Juan Ignacio Arguello, Valerio Costa, Facundo Lucero

## The paper
We build on *Seeing without Pixels: Perception from Camera Trajectories* (Xue et al., CVPR 2026;
[paper](https://arxiv.org/abs/2511.21681), [code](https://github.com/google-deepmind/seeing_without_pixels)).
Its model, CamFormer, is a 4-layer Transformer with 0.3 M parameters. It never sees a pixel: it reads only the
camera's path through space and is trained contrastively against a frozen CLIP text encoder, so that a trajectory
can be matched to a description of what happens in the video.

## What we test
The authors name two limitations themselves: they train a separate model for egocentric and for exocentric video,
and results are weak on procedural activities. We ask how far a pixel-free model can be pushed on both.

1. **One model for both viewpoints.** A single encoder trained on egocentric and exocentric trajectories together,
   with and without a domain token, compared with the two released single-viewpoint checkpoints.
2. **Harder negatives for procedural activities.** The test set uses distractors from the same take, while training
   uses random in-batch negatives. We train with same-take negatives and measure on the procedural split.
3. **Eye gaze as a second input.** We add Ego-Exo4D gaze to the camera trajectory, comparing a frozen backbone with
   a newly trained gaze branch against full fine-tuning.

The metric throughout is the paper's own motion-to-text 5-way multiple-choice accuracy on Ego-Exo4D, DynPose-100K
and Nymeria (zero-shot). Every experiment starts from the authors' released checkpoints.

Background on what the release contains and what the authors already tried is in [`docs/feasibility.md`](docs/feasibility.md).
The plan of runs is in [`SCHEDULE.md`](SCHEDULE.md), and what has been done so far is in [`LOG.md`](LOG.md).

## Layout
```
src/            model and training code
configs/        experiment configs
scripts/slurm/  job scripts for the Bocconi HPC cluster
notebooks/      exploration only
data/           notes on where the data is and how to fetch it (the data itself is not in git)
docs/           background notes
results/        metrics tables and figures that go in the report
report/  poster/  presentation/
```

## Data
No video is used. The released camera-pose trajectories, checkpoints and precomputed features come from the
authors' public release; eye gaze and annotations come from Ego-Exo4D under its licence. Sizes, locations and
download commands are in [`data/README.md`](data/README.md).

## Deliverables
Presentation on 24 November 2026, poster session on 1 December 2026, and a written report.
