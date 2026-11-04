# Experiment 2: one model for egocentric and exocentric trajectories (design, 17 Oct 2026)

Written before any code, while the cluster is unreachable. Items marked **Open** are checked on the cluster
before the first training job; each says how.

## Question
The paper trains a separate CamFormer per viewpoint: one on Ego-Exo4D (camera worn on the head, Aria glasses)
and one on DynPose-100K (camera held by an observer, poses estimated from video). Its limitations section names
the next step: "a single, unified trajectory encoder ... introducing an explicit conditional domain token"
(supplementary material, section 3.3). We test that suggestion:

1. Can one model, trained on both domains, match the two separate models on both test sets?
2. Does telling the model which domain a clip comes from (a domain token) help, compared with the same model
   without it?

Hypothesis: both domains share basic camera motions (walking, turning, stopping, looking down), so a shared
model can reuse them; but the two domains differ in pose source, scale and noise, so training on both can also
interfere. The runs below separate the two effects.

## What the paper did (and what the code shows)
| | Ego-Exo4D (ego) | DynPose-100K (exo) |
|---|---|---|
| Training pairs | 159,186 (narrations) | 88,151 after the scale filter (Panda70M captions) |
| Pose source | Aria glasses, downsampled to 20 FPS (supp. 2.2) | video estimates: dataset's own at 12 FPS, ViPE at 30 FPS per supp. 2.2, in fact at each video's rate: mostly 30, 25 or 24, a few 15, and some clips that fit no standard rate (D6) |
| Pose encoding | relative to the middle frame, 6D rotation, plus a gravity token (`rel9d_grav`) | same without gravity (`rel9d`); gravity "is omitted when processing poses estimated from monocular videos" (supp. 2.3) |
| Clip | a context window of up to 8 s around the narration; only the labelled sub-window is pooled (`tasks/pretrain_longseq.py`) | the whole short clip, mean-pooled (`tasks/pretrain.py`) |
| Translation | metric (SLAM) | scale from the estimator; clips with `pose_scale` ≥ 20 are dropped |
| Axes | supp. 2.3 says x left, y up, z forward; the stored trajectories measure x down, y left, z forward (Design 6) | not stated; ViPE's first pose is the identity and its vertical axis is y (Design 6) |
| Test | 7,079 five-way questions, all options from one take | 1,000 five-way questions (`samevideo_test5000`); in 902 of them one distractor comes from the answer's video |
| Released result (motion-to-text) | 44.81 (`egoexo4d_dur8.pt`) | ViPE 46.3 (`dynpose100k_vipe.pt`), original 36.2 in Table 4 (39.2 from the released features) |

The ViPE poses are used for exo throughout: they give the stronger released model, keep more clips (88,151
against 81,975 after the scale filter; the paper's training count is the ViPE one) and ViPE estimates metric
depth, which brings the translation scale closer to the Aria data.

Facts checked on the released CSVs (all counts in this note are recomputed by
`python scripts/ego_exo_facts.py <upstream data_files/>`): all 1,000 DynPose test questions use validation clips, none use
training clips; but 236 of the 1,379 test videos also have other segments in the training split. This is the
authors' split and we keep it, so our numbers stay comparable; the report mentions it.

## Design
Every change sits behind a flag whose default reproduces the authors' code.

1. **Start from the released ego model** (`egoexo4d_dur8.pt`), fine-tuned at the authors' learning rate
   (1e-4), as in experiment 1 and as the course recommends (Lecture 7, transfer learning). Ego-Exo4D models are
   the only released models with the gravity branch (`egoexo4d_dur8.pt` and `egoexo4d_dur16.pt`), which a shared
   model needs for ego clips; we take the 8-second one, which matches the ego test. The released exo model
   cannot be merged into it, so the shared model meets exo data starting from the ego weights. Control E→X
   below measures what that costs.
2. **Exo clips go through the ego architecture.** Each exo clip is encoded like an ego clip whose labelled
   window is the whole clip. This is not the authors' exo computation. Their exo task mean-pools the encoder
   states first and then applies LayerNorm and the output projection (`cm_encoder.py`, lines 161-173); the
   long-sequence task applies LayerNorm and the projection to each frame and then pools over the window
   (`tasks/pretrain_longseq.py`, lines 137-141). LayerNorm is not linear, so the two give different embeddings
   from the same weights. Check before training: run the released exo model through the long-sequence path on
   its own test and record the gap to 46.3. This is a measurement, not a gate. **Measured 18 Oct:** 46.20
   (job 706944) against 46.30 through the authors' exo task (job 706967, which also confirms our copy of
   their path): a gap of 0.1 points, one question.
3. **No gravity for exo.** The released ego model was trained with gravity dropout: half of the time its
   gravity token was replaced by a learned "gravity unknown" token (`learnable_gravity` in `cm_encoder.py`).
   Exo clips always get that learned token. This follows the paper's rule (no gravity from monocular video)
   with a token the model already knows. Code: a per-clip `has_gravity` mask instead of the global
   `--use_learnable_gravity`.
4. **Domain token** (`--domain_token`). A learned vector per domain (2 × 128) is added to every input token,
   like the segment embedding in BERT. It starts at zero, so at step 0 the model is exactly the released ego
   model; the reproduction check verifies this (44.81 with the flag on). We add rather than prepend, because a
   prepended token would shift every pose frame by one position and change the released model before
   training starts.
5. **Common pose rate** (`--exo_resample_hz`, 0 = native). ViPE poses (30 FPS) are resampled to the ego rate,
   20 FPS (linear for translation, spherical for rotation), because the positional encoding counts frames, not
   seconds: at native rates one position is 50 ms for ego and 33 ms for exo. **Open:** the released exo model was
   trained at 30 FPS. Before training, compare it at native and resampled rates on validation clips that are not
   in the test set (7,925 remain) with full retrieval, never on the test. If it loses more than 2 points of
   recall at 5, resampling becomes an ablation instead of the default. **Measured 18 Oct** (jobs 706959,
   706960, 706966; `exo_test.sh ... --exo_eval_split val_nontest`, then `eval_full_retrieval.py`): recall at 5
   is 0.67% native and 0.56% at 20 FPS. Over 7,925 clips recall at 5 is below 1% either way, so the 2-point
   rule could not have failed and says little. On the same clips, a 5-way choice with random distractors
   (chance 20%) gives 50.42 native and 49.38 at 20 FPS: one point lost. Resampling stays the default; the
   rule's metric is reported as chosen badly.
   **Corrected 4 Nov (D6).** ViPE poses come at each video's own rate: about a third of the clips are 24 or 25
   FPS, not 30. The first runs and the measurements above took 30 FPS for every clip, so those clips were resampled
   to 16-17 poses per second instead of 20. Version 2 of the runs uses each clip's rate (`--exo_fps_table`).
6. **Common axes** (`--exo_to_aria_axes`). Exo camera frames are turned into the ego camera frame before the
   relative poses are computed. ViPE stores OpenCV camera-to-world matrices (its writer, `vipe/utils/io.py`:
   "Save OpenCV cam2world matrices as 4x4 matrix in npz file"), which fixes the y-down sign that the exo data
   alone cannot. **Checked 18 Oct** (`exo_checks.py`, job 706937, 2,000 training clips per domain):
   - Ego, as stored: the camera x axis points down (mean world-up component −0.81; y +0.11, z −0.19), and ego
     motion lies mostly along ±y and ±z (x carries 12% of it). So the stored frame is x down, y left,
     z forward, not the "x left, y up" of supp. 2.3.
   - Exo (ViPE): the first pose is the identity (world = first camera); motion lies mostly along ±z and ±x
     (y carries 13%), with +z more often than −z (32% against 20%). So y is vertical and z forward, as in the
     OpenCV convention (x right, y down) that we assume; without gravity the sign of y cannot be measured.
   - Hence the conversion is a 90° turn about z (ego x = OpenCV y, ego y = −OpenCV x), not the 180° turn the
     earlier draft took from supp. 2.3. Pi3 files are not used here; their first poses are not the identity,
     so they come in some other world frame, left as found.
7. **Translation scale.** No normalisation in the main runs. Rule: if the median speeds differ by more than a
   factor of 3, add one scale constant. **Checked 18 Oct** (job 706937): per-clip median speed, ego 0.31 m/s
   (p90 1.12), exo 0.19 (p90 0.95); ratio 1.6. No scale constant. (Exo speeds were computed at 30 FPS for every
   clip; with each clip's rate (D6) they are up to 20% lower for a third of the clips, and the ratio stays far below
   3.)
8. **Batches stay within one domain.** Each step draws a full batch of 1,024 from one domain. A mixed batch would
   pit ego clips against exo captions, which are trivially easy to rule out, while both tests are within one
   domain. Each epoch runs every ego batch (155) and every exo batch (86) once, in a random order fixed by the
   seed: 241 steps. The loss is the authors' (symmetric InfoNCE, temperature 0.07) for both domains;
   experiment 1's changes stay off, so the two experiments do not interact.

## Runs
All runs: 20 epochs of their domain(s), batch 1,024, learning rate 1e-4, last checkpoint tested (no checkpoint
is picked on validation, because both test sets are built from validation clips), validation every 5 epochs
for monitoring only. Ego data always uses the same flags (`--take_duration 8 --sample_dur`), and all runs of one
seed use one code revision. Each domain gets the same
number of updates in every run that trains on it: 3,100 ego steps and 1,720 exo steps.

| Run | Starts from | Trains on | Domain token | Purpose |
|---|---|---|---|---|
| R. Released | the two released models | none | n/a | Reference rows (44.81 / 46.3) |
| Z. Zero-shot | released ego | none | n/a | How far the ego model reaches on exo with no exo training |
| S-ego | released ego | ego | n/a | Ego specialist with the same extra training. Same settings as run A of experiment 1, but retrained for every seed on the same code revision as U0 and U1, so the comparison does not mix code versions |
| S-exo | released exo | exo | n/a | Exo specialist with the same extra training |
| E→X | released ego | exo | n/a | Exo from the ego weights: separates "starting from ego" from "sharing one model" |
| U0 | released ego | ego + exo | off | One model, no explicit token |
| U1 | released ego | ego + exo | on | One model with the domain token |

S-exo uses the authors' exo setup (`rel9d`, native rate unless item 5 changes it, their pooling). U0, U1, E→X
and Z use items 2 to 7. So on the exo test, U1 against S-exo mixes four differences (starting weights, input
preparation, pooling and architecture, sharing), while E→X has the same start, inputs and exo steps as the joint
models. U0 against E→X is the comparison closest to "sharing only"; U1 against E→X also includes the domain token,
which E→X does not have. (Corrected 4 Nov: this paragraph said U1 against E→X isolates sharing.)

U0 is not blind to the domain. Exo clips always carry the "gravity unknown" token, and translation scale and
clip length differ between domains, so the encoder can tell them apart without the domain token. Our prior is
that U1 − U0 is close to zero. A null result means the explicit token is redundant given these cues; it does
not mean a shared model needs no conditioning.

## Evaluation
- Ego-Exo4D five-way test, motion-to-text and text-to-motion, by type and view (`score_mcq.py`).
- DynPose-100K ViPE five-way test, 1,000 questions, motion-to-text and text-to-motion (`eval_retrieval.py`, plus
  a per-question dump for the paired bootstrap).
- Seeds 0, 1, 2. For each comparison (U1 − S, U1 − U0, U1 − E→X) and each test, one 95% interval from
  `compare_runs.py`: a paired bootstrap that resamples seeds, then takes (Ego-Exo4D) or videos (DynPose). With
  three seeds the seed level is rough, so per-seed results are reported next to it.
- Honest limit: with 1,000 DynPose questions one run's accuracy has a standard error of about 1.6 points, so
  differences under about 3 points are not resolvable on that test from a single seed.

**Reporting rule, fixed now.**
- "One model matches the specialists" is claimed only if the 95% interval for U1 − S lies inside ±1.5 points
  on Ego-Exo4D and inside ±3 points on DynPose. The margins are the smallest differences each test can resolve
  (7,079 against 1,000 questions).
- "The domain token helps" is claimed only if the 95% interval for U1 − U0 on DynPose excludes zero and is
  positive, with U1 − U0 not negative on Ego-Exo4D. DynPose is named in advance because exo is the domain the
  token should help: the ego weights already fit ego clips. One interval, one test, so there is one chance of a
  false positive, not six.
- Anything else is reported as measured, including a negative result.

**Pilot gate.** Seed 0 of S-ego, S-exo, E→X, U0 and U1 runs first, all on one code revision. Seeds 1
and 2 run only if all seed-0 runs finished without errors and U1 is within 5 points of its specialist on both
tests. Otherwise we stop, report seed 0 with that diagnosis, and move the GPUs to experiment 3.

## Compute
Measured in the smoke test of U1 (job 706961, one epoch): a median of 2.7 s per ego step and 2.8 s per exo step
(compute only), so an exo step costs about as much as an ego step. One joint epoch (155 ego + 86 exo steps)
takes about 11 min of training; a validation pass on both domains about 4 min.

| Run | Per seed |
|---|---|
| U0, U1 | about 4 h each (20 joint epochs, validation every 5 epochs) |
| S-exo, E→X | about 1.5 h each |
| S-ego | about 2.5 h (20 ego epochs) |
| Tests (about 3 min each) and Z | under 1 h in all |

About 14 GPU-hours per seed, 42 for three seeds: about 21 hours of wall time on our 2 GPUs. The pilot (seed 0)
is about 7 hours of wall time. Every run of one seed uses the same code revision, which is recorded with its
results; run A of experiment 1 is not reused, because it was trained on an earlier revision.

## Code changes (each behind a flag; defaults keep the authors' behaviour)
1. `datasets/dynpose.py`: an option to return the long-sequence sample format (12 columns, with zeros for the
   unused gravity columns; window = whole clip; the video id in place of the take name), plus
   `--exo_resample_hz` and `--exo_to_aria_axes`.
2. `models/cm_encoder.py`: per-clip `has_gravity` mask; `--domain_token` (zero-initialised additive embedding).
3. `loader.py`: a batch order that interleaves complete single-domain batches, seeded.
4. `tasks/pretrain_longseq.py`: domain ids in the batch, loss logged per domain, one validation set per domain.
5. `train.py`: `--dataset egoexo4d_dynpose_longseq` for joint training, and `--test_domain {ego,exo}` to choose
   the test set.
6. `compare_runs.py` and a per-question dump for DynPose: the video id takes the place of the take, and the
   type and view columns become optional.
7. A CPU script for the open checks: clip lengths, rates, translation per second, axis check.

Checks before the first run, all done on 18 Oct:

| Check | Job | Result |
|---|---|---|
| Unit tests (`test_ego_exo.py`: token at zero changes nothing, gravity mask, resampling, axes, sampler) | 706958 | pass |
| Reproduction check, all flags off | 706935 | 44.81 / 33.61, OK |
| Reproduction check with `--domain_token` | 706936 | 44.81 / 33.61, OK |
| Released exo model through the authors' exo task (our copy of their path) | 706967 | 46.30 / 48.90, matches |
| Released exo model through the long-sequence path, native rate | 706944 | 46.20 / 48.30 (gap 0.1) |
| Pose rate on validation clips outside the test (Design 5) | 706959, 706960, 706966 | resampling kept |
| CPU checks: lengths, rate, speed, axes (Design 6, 7) | 706937 | see Design 6 and 7 |
| Z: released ego model on the DynPose test (20 FPS, ego axes) | 706968 | 26.30 / 25.40 (chance 20) |
| Smoke test of U1, one epoch | 706961 | runs; both validation sets logged; timing above |
| Tests of the U1 smoke checkpoint on both test sets (`eval_ego_exo.sh`) | 706978 | ego 44.89, exo 34.60 after one epoch; per-question files written |
| S-exo path: one epoch of the authors' exo task, then its test (`exo_test.sh --dataset dynpose_pretrain`) | 706979 | 46.20; about 4 min per epoch |

## Risks and what we cut first
- **Exo starts from ego weights.** If U1 trails S-exo but matches E→X, the gap comes from the starting point, not
  from sharing; the report says so rather than claiming sharing hurts.
- **Different clip lengths.** Exo clips are whole short videos; ego windows are up to 8 s. **Checked 18 Oct**
  (job 706937): exo clips are 3.8 to 7.6 s (p10 to p90; median 5.0 s, at most 256 frames = 8.5 s), every
  video frame (index step 1). No cropping. (Seconds computed at 30 FPS; clips from 24 or 25 FPS videos are up to
  25% longer, at most about 10.7 s, still one window.)
- **1,000 exo questions are few.** Small exo differences will not be significant; we say so instead of reading
  them.
- **If time runs short**, cut in this order: seeds 1 and 2 of E→X, then the resampling ablation, then Z,
  then validation altogether. U0, U1, S-ego and S-exo are the experiment and are not cut.
