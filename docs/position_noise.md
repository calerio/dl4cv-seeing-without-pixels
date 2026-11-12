# Experiment 4: robustness to noise in the camera positions (design, 12 Nov 2026)

Written before any GPU job, following the same steps as experiments 1 to 3: the table to fill, the seeds, the
reason for each choice and the reporting rule are fixed here (and in D7), and a review comes before the first run.

## Question
Real camera tracking is never exact. Does fine-tuning CamFormer on positions with small random errors make it
pick the right caption more often when the test positions are noisy too, without costing accuracy on clean inputs?
This measures robustness to synthetic noise, not to any real tracking failure.

## What the paper did
The paper always uses the stored Aria trajectories as they are. It reports results on estimated poses (Table 3,
p. 6), but never adds noise on purpose. So this experiment has no reference number in the paper; the comparison
is between our own runs, all starting from the released checkpoint. The clean test is the standard one, so clean
scores stay comparable with the paper's 44.81.

## Data
No new data. The Ego-Exo4D training and test files and the released checkpoint already on the cluster.

**Units and scale (checked 12 Nov 2026, `src/camformer/position_units_check.py`, 20 random training takes,
seed 0; output, take list and code revision in `results/position_units_check.txt`).** Positions are in metres: all float64, all finite, unit quaternions, a clip moves 0.2 to 6.6 m in total.
The median step between stored frames ranges from 0.21 cm (cooking, a person standing at a counter) to 3.7 cm
(basketball), median 1.44 cm over takes. Noise of sigma per coordinate on each frame puts noise of about
sqrt(2) x sqrt(3) x sigma on each step: 1.2 cm at sigma 0.005 and 2.4 cm at 0.01. At the training level 0.005 the
step noise is already larger than the median step in 7 of the 20 takes (three cooking, one COVID test, one bike
repair and two bouldering takes, so five of the seven are procedural activities); at 0.01 in 13 of 20, and at
0.02 and above in all 20. Even the training level therefore buries the fine motion of slow takes, and procedural
activities are where the paper's trajectory model scores lowest (Table 2, p. 6).

## Method
In the dataset, after the clip's frames are chosen and before the usual conversion to relative poses
(`transform_camera_pose`), independent Gaussian noise with standard deviation sigma is added to x, y and z of
every frame. Rotations are untouched. Because the relative poses are taken from the noisy middle frame, the
errors in the relative positions are correlated, as they would be with real noisy absolute positions.

- **Training:** each sampled clip gets noise with probability 0.5. The noise has its own random generator, so
  turning it on does not change which clips and windows are drawn.
- **Testing:** the noise of a clip is fixed by (noise seed, clip id, frame indices), so A and N see exactly the
  same noisy inputs, whatever the batching or number of workers.
- **Validation** stays clean and is only watched, never used to choose anything (the test questions are built
  from the validation takes).
- Only the Aria ego setup; the flags are refused with gaze, Pi3 poses or other datasets.

Flags (`train.py`), all off by default:

| Flag | Default | Meaning |
|---|---|---|
| `--position_noise_train_sigma` | 0 (off) | noise std on training clips, metres |
| `--position_noise_probability` | 0.5 | fraction of training clips that get noise |
| `--position_noise_eval_sigma` | 0 (off) | noise std at test time, with `--test` only |
| `--position_noise_eval_seed` | 0 | which fixed test noise |

## Runs
Both runs fine-tune `egoexo4d_dur8.pt` with the authors' settings (learning rate 1e-4, batch 1,024, 8-second
clips, sampled durations) for 20 epochs, seeds 0, 1 and 2, on the same code revision, and are tested from their
last checkpoint.

| Run | Training noise | Purpose |
|---|---|---|
| A | none (authors' setup) | Same extra training, so a gain is not just "more training" |
| N | sigma 0.005 m on half the clips (D7) | The experiment |

`bash ~/dl4cv/position_noise_seed.sh <seed> --with_tests` submits A and N as two jobs, about 3 hours of training
each followed by the tests. The account queues 4 jobs, so at most two seeds are queued at a time; the launcher
refuses a seed already in the queue. Each run is tested on clean inputs and at sigma 0.005, 0.01, 0.02 and 0.05 m with noise seeds 0, 1 and 2,
through `eval_run.sh` (the standard 4-second test window). `bash ~/dl4cv/position_noise_seed.sh smoke` runs one
epoch of N first, then one noisy test of it (sigma 0.005, noise seed 0), so the noisy test path is exercised
before any full run; a seed is refused until that smoke run has finished on the deployed code revision.

## Table to fill
Motion-to-text accuracy, overall and procedural, per seed and mean over the three seeds:

| Run | clean | 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|---|
| A | | | | | |
| N | | | | | |

This table is descriptive: at each noise level the number is the mean over the three noise seeds, which are
repeated measurements of the same model, not extra training runs. The claims rest only on the rule below.

## Reporting rule (fixed before any run)
Written before any full run. The training level was changed from 0.01 to 0.005 before any noisy job had run on this
code. The smoke run does produce a one-epoch noisy accuracy; it only checks that the pipeline works, and nothing
(training level, test levels or this rule) was chosen from it.
- Main comparison: N against A on **mean accuracy over the four noise levels** (0.005, 0.01, 0.02, 0.05 m),
  noise seed 0, motion-to-text overall, each level weighted equally. `average_noise_levels.py` averages each
  question's score over the four levels per run, and `compare_runs.py` runs the paired bootstrap over seeds and
  recordings on those files.
  This is a screening test: N is called more robust if the interval of the difference is above zero. With 3 seeds
  only a gain of roughly 1.5 points or more is likely to be resolved (experiment 1's procedural intervals were about 2.5 points
  wide); a smaller gain would be reported as not resolved, not as absent. This follows the standard protocol
  for corruption robustness (Hendrycks and Dietterich 2019), which scores a model across all severity levels of a
  corruption rather than at one chosen level, so no single level can be picked after the fact. Accuracy at each
  level, including the matched level 0.005, is reported as a curve without separate claims.
- Clean cost: N against A on clean inputs, same test, reported as the difference with its interval. If the
  interval includes zero it is reported as "no cost resolved, within about ±1.5 points", not as "no cost".
- Noise seeds 1 and 2 are reported as a check that the result does not depend on one draw of noise.

## Limitations
- Independent noise on every frame is a simple stress test. Real tracking error is likely to be smoother (Meta's
  Aria accuracy is a global error of whole trajectories, and Engel et al. 2023 describe Aria's open-loop error as
  accumulated drift), which
  would disturb frame-to-frame motion less; we have not measured its shape. Drift and tracking loss are not tested.
- N is trained on the same kind of noise it is tested on, so this measures robustness to a known corruption,
  not to unseen ones (the corruption benchmark keeps its test corruptions out of training; Hendrycks and
  Dietterich 2019). Whether it carries over to real tracking error is not tested here.
- Per-frame jitter is "one of the most commonly employed" augmentations for motion sensor data (Wu et al. 2025),
  and has been found not to help a wrist-accelerometer classifier (Um et al. 2017) and to hurt IMU-based pose
  estimation (Wu et al. 2025). It was chosen knowingly as the
  simplest common perturbation.

- Missing frames and rotation noise are not tested.

## References
- Hendrycks and Dietterich 2019. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.*
  ICLR 2019, arXiv:1903.12261. Scores each corruption across five severity levels and averages them (mCE);
  networks "should not be trained on" the test corruptions.
- Engel et al. 2023. *Project Aria: A New Tool for Egocentric Multi-Modal AI Research.* arXiv:2308.13561. Measures
  closed-loop Aria trajectories: "typical global RMSE translation error of no more than 1.5 cm in room-scale
  scenarios" (a whole-trajectory error, not per-frame noise).
- Um et al. 2017. *Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional
  Neural Networks.* ICMI 2017, arXiv:1706.00527. Wrist accelerometer data: jittering, described as simulating
  additive sensor noise, "fails to improve" their classifier because the fast fluctuations resemble the symptom
  being classified.
- Wu, Guo and Qin 2025. *MODA: Motion-Drift Augmentation for Inertial Human Motion Analysis.* CVPR 2025. Body-worn
  IMUs: calls jittering "one of the most commonly employed" augmentations; in their fully supervised pose-estimation
  experiments, jittering and moving averages have "adverse effects", while simulated sensor displacement (drift)
  improves results.
