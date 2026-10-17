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
| Pose source | Aria glasses, downsampled to 20 FPS (supp. 2.2) | video estimates: dataset's own at 12 FPS, ViPE at 30 FPS (supp. 2.2) |
| Pose encoding | relative to the middle frame, 6D rotation, plus a gravity token (`rel9d_grav`) | same without gravity (`rel9d`); gravity "is omitted when processing poses estimated from monocular videos" (supp. 2.3) |
| Clip | a context window of up to 8 s around the narration; only the labelled sub-window is pooled (`tasks/pretrain_longseq.py`) | the whole short clip, mean-pooled (`tasks/pretrain.py`) |
| Translation | metric (SLAM) | scale from the estimator; clips with `pose_scale` ≥ 20 are dropped |
| Axes | Aria camera frame: x left, y up, z forward (supp. 2.3) | not stated; OpenCV is x right, y down, z forward |
| Test | 7,079 five-way questions, all options from one take | 1,000 five-way questions (`samevideo_test5000`); in 902 of them one distractor comes from the answer's video |
| Released result (motion-to-text) | 44.81 (`egoexo4d_dur8.pt`) | ViPE 46.3 (`dynpose100k_vipe.pt`), original 36.2 in Table 4 (39.2 from the released features) |

The ViPE poses are used for exo throughout: they give the stronger released model, keep more clips (88,151
against 81,975 after the scale filter; the paper's training count is the ViPE one) and ViPE estimates metric
depth, which brings the translation scale closer to the Aria data.

Facts checked on the released CSVs (17 Oct): all 1,000 DynPose test questions use validation clips, none use
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
   its own test and record the gap to 46.3. This is a measurement, not a gate.
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
   in the test set (5,452 remain) with full retrieval, never on the test. If it loses more than 2 points of
   recall at 5, resampling becomes an ablation instead of the default.
6. **Common axes** (`--exo_to_aria_axes`). Exo poses are turned into the Aria camera frame by a fixed 180°
   rotation about the forward axis (x → −x, y → −y), applied to the camera frame before the relative poses are
   computed. The authors did exactly this when they fed video-estimated poses to the Aria-trained model
   (supp. 2.3). **Open:** which frame ViPE and DynPose use. Check on the cluster: in upright handheld video the
   camera's "up" axis should point against gravity in the estimator's world frame; compare the mean camera y
   axis with the estimator's stated world up. Note also that our copy of the Pi3 loader converts nothing,
   so either the Pi3 files are stored already converted or the conversion is missing there; the files tell.
7. **Translation scale.** No normalisation in the main runs. **Open:** compare the distribution of translation
   per second (median and 90th percentile) between ego training clips and resampled ViPE clips on the cluster
   (CPU job). If the medians differ by more than a factor of 3, add `--exo_translation_scale` (one constant from
   the ratio of medians, fixed before training) and record it as a decision.
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
| Z'. Zero-shot | released exo | none | n/a | The reverse: the exo model on the ego test |
| S-ego | released ego | ego | n/a | Ego specialist with the same extra training. Same settings as run A of experiment 1, but retrained for every seed on the same code revision as U0 and U1, so the comparison does not mix code versions |
| S-exo | released exo | exo | n/a | Exo specialist with the same extra training |
| E→X | released ego | exo | n/a | Exo from the ego weights: separates "starting from ego" from "sharing one model" |
| U0 | released ego | ego + exo | off | One model, no explicit token |
| U1 | released ego | ego + exo | on | One model with the domain token |

S-exo uses the authors' exo setup (`rel9d`, native rate unless item 5 changes it, their pooling). U0, U1, E→X
and Z use items 2 to 7. So on the exo test, U1 against S-exo mixes four differences (starting weights, input
preparation, pooling and architecture, sharing), while U1 against E→X isolates sharing: same start, same
inputs, same architecture, same exo steps. The report reads E→X first.

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
Measured: one ego epoch is about 5.8 min of training and 2.5 min of validation on one A100. The cost of an exo
step is not known: whole ViPE clips at 30 FPS may be longer than ego windows, and attention cost grows with the
square of the length. **Open:** time it in the smoke test. The table assumes an exo step costs about as much as
an ego step.

| Run | Per seed |
|---|---|
| U0, U1 | about 3 h each (20 × 241 steps, validation every 5 epochs) |
| S-exo, E→X | about 1.2 h each |
| S-ego | about 2 h (20 ego epochs) |
| Tests (about 3 min each), Z and Z' | under 1 h in all |

About 12 GPU-hours per seed, 36 for three seeds: about 18 hours of wall time on our 2 GPUs. The pilot (seed 0)
is about 6 hours of wall time. If an exo step costs twice an ego step, add about 3 GPU-hours per seed. Every run of one seed uses the same code revision, which is recorded with its
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

Checks before the first run: the reproduction check with all flags off and with `--domain_token` on (both must
give 44.81); a one-epoch smoke test of U1; the released exo model on its own test with our loader in long-sequence
format and native rate (the gap to 46.3 is recorded, see Design item 2); Z as a sanity row.

## Risks and what we cut first
- **Exo starts from ego weights.** If U1 trails S-exo but matches E→X, the gap comes from the starting point, not
  from sharing; the report says so rather than claiming sharing hurts.
- **Different clip lengths.** Exo clips are whole short videos; ego windows are up to 8 s. **Open:** the exo
  length distribution (CPU check). If exo clips are much longer than 8 s at 20 FPS, crop to the centre 8 s and
  record it as a decision.
- **1,000 exo questions are few.** Small exo differences will not be significant; we say so instead of reading
  them.
- **If time runs short**, cut in this order: seeds 1 and 2 of E→X, then the resampling ablation, then Z and Z',
  then validation altogether. U0, U1, S-ego and S-exo are the experiment and are not cut.
