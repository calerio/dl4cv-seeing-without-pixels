# Seeing without Pixels, extended

Group project for 20600 Deep Learning for Computer Vision (Bocconi, a.y. 2026-27).

*Nullius in verba* (take nobody's word for it): every number here is reproduced from code in this repository, and every comparison was set before its runs.

**Team:** Juan Ignacio Arguello, Valerio Costa, Facundo Lucero

New here? Start with the plain-language [project wiki](https://github.com/calerio/dl4cv-seeing-without-pixels/wiki): status, FAQ and glossary.

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
2. **Harder wrong answers in training, for procedural activities.** The model learns by matching a camera motion to
   the right description among wrong ones. In training, as is standard for this kind of model, the wrong ones are the
   other clips that happen to be in the same batch, so they almost never come from the same recording. In the test,
   the authors deliberately take them from the same recording (the same person in the same kitchen, doing something
   else), which makes for a stricter benchmark. We ask whether also practising on same-recording wrong answers helps
   on procedural clips, where the differences between steps are subtle. As in the authors' test, a wrong answer must
   have a different main verb from the right one, so that a repeated step is not counted as wrong.
3. **Eye gaze as a second input.** We give the model where the person is looking as well as how the camera moves.
   We compare two ways of adding it: keep the authors' trained model unchanged and train only the new gaze part, or
   retrain the whole model with gaze included.

The metric throughout is the paper's own motion-to-text 5-way multiple-choice accuracy on Ego-Exo4D, DynPose-100K
and Nymeria (zero-shot). Every experiment starts from the authors' released checkpoints.

Background on what the release contains and what the authors already tried is in [`docs/feasibility.md`](docs/feasibility.md).
The plan of runs is in [`SCHEDULE.md`](SCHEDULE.md), and what has been done so far is in [`LOG.md`](LOG.md).

## Layout
```
src/camformer/  the authors' code (Apache 2.0), with our changes as separate commits
configs/        experiment configs
scripts/        job scripts for the Bocconi HPC cluster (slurm/) and chained submission
notebooks/      exploration only
data/           notes on where the data is and how to fetch it (the data itself is not in git)
docs/           background notes and the decisions log (decisions.md)
results/        metrics tables and figures that go in the report
report/  poster/  presentation/
```

## Data
No video is used. The released camera-pose trajectories, checkpoints and precomputed features come from the
authors' public release; eye gaze and annotations come from Ego-Exo4D under its licence. Sizes, locations and
download commands are in [`data/README.md`](data/README.md).

## Deliverables
Presentation on 24 November 2026, poster session on 1 December 2026, and a written report.
