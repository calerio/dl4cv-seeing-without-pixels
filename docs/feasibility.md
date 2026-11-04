# Feasibility check

Written after reading the paper, its supplementary material and all of the released code, before running anything.

## Problem statement
Camera motion is an almost free signal for understanding video: CamFormer has 0.3 M parameters by the paper's Table 2 (598,144 in the released checkpoint) and costs 0.02 GMACs,
against 150.7 M and 89.49 GMACs for the EgoVLPv2 video model. But the published model (a) needs a separate network
per viewpoint and (b) is weak on procedural activities: retrieval 34.3 / 32.7% on procedural clips against
56.1 / 46.4% on physical ones (chance 20%), and keystep recognition 14.07% against 29.17% for video.

## What is released
- Pretraining code for both domains and the 5-way retrieval evaluation.
- Four checkpoints.
- Pose archives: Ego-Exo4D 15 GB, DynPose-100K 2.3 GB, Nymeria 162 MB.
- Metadata CSVs: 159,186 egocentric and 89,678 exocentric training records (88,151 after the loader's pose-scale
  filter); both match the paper (p. 14). (Corrected 4 Nov: the first version gave line counts, 160,865 and
  89,683, which include headers and captions spread over several lines.)

Not released: code or data for keystep recognition and localization, proficiency, activity / event / attribute
classification, the pose-estimator comparisons, and counting. No eye gaze.

## Ideas considered
| Idea | Data in hand? | Feasibility | Notes |
|---|---|---|---|
| Reproduce the retrieval numbers | yes | high | the precomputed features reproduce the tables without a GPU |
| Single ego+exo encoder with a domain token | yes | high to build, payoff uncertain | `cm_encoder.py` already prepends a gravity token and an optional scenario token, so a domain token is a small change. The real work: two different training tasks (windowed pooling for ego, mean pooling for exo), different frame rates (20 vs 12/30 Hz), metric vs scale-ambiguous translation, Aria vs OpenCV axes |
| Same-take hard negatives for procedural clips | yes | high | the test set uses same-take distractors while training uses random in-batch negatives; closing that mismatch is cheap and measurable on the released procedural split |
| Eye gaze as a second input | via the Ego-Exo4D licence | medium | alignment can follow `data_prep/step1_extract_segments.py` |
| Keystep benchmark | via the Ego-Exo4D licence | medium | needs our own evaluation code; may need the raw 1 kHz trajectories, which are very large |
| Repetition counting | no | low | no egocentric counting labels |
| Fixing pose-estimator failures | no | low | pose-estimation research, outside our scope |

We take the second, third and fourth rows, after the first as a check of the setup.

## Already tried by the authors
Not ours to claim: fine-tuning the text encoder (45.42 vs 44.81%, Table 10), longer context (Fig. 6), fusion with
video features (Table 2), and three other pose estimators (Table 3). The released code also has options for a
context-aware text encoder and an auxiliary activity loss (`--hier_text`, `--cls_loss_weight`), with no result in
the paper.

## Compute
The paper used 8 A100s with batch size 1024, and the script defaults to 500 epochs. The text encoder is frozen, so
its embeddings can be computed once and cached, leaving a model of about 0.6 M parameters to train. We expect a run to fit on
one GPU; the first timing run will confirm it.
