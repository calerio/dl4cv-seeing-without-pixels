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
The authors name two limitations themselves. They train a separate model for egocentric video (filmed by a camera the
person wears, such as smart glasses) and for exocentric video (filmed by someone watching, such as a phone or a
tripod). And results are weak on procedural activities (step-by-step tasks; in the paper: bike repair, cooking and
health tasks such as a COVID test), where the head moves much less than in basketball or bouldering. We ask how far a
pixel-free model can be pushed on both.

1. **One model for both viewpoints.** A single model trained on worn-camera and observer-camera trajectories together,
   with and without a domain token (one extra input that tells the model which kind of camera a trajectory comes
   from), compared with the two released single-viewpoint models.
2. **A harder training quiz for procedural activities.** The model learns by matching a camera motion to the right
   description among wrong ones. In training, the wrong ones come from random other videos, so they are easy to rule
   out (a cooking clip against a basketball caption). In the test, the wrong ones come from the same recording (in the
   same kitchen, "cracks an egg" against "stirs the pan"), which is much harder. We train on wrong answers from the
   same recording too, so that training matches the test, and measure the effect on the procedural clips.
3. **Eye gaze as a second input.** We give the model where the person is looking as well as how the camera moves.
   We compare two ways of adding it: keep the authors' trained model unchanged and train only the new gaze part, or
   retrain the whole model with gaze included.

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
