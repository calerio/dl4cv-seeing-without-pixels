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
seed 0).** Positions are in metres: all float64, all finite, unit quaternions, a clip moves 0.2 to 6.6 m in total.
The median step between stored frames ranges from 0.21 cm (cooking, a person standing at a counter) to 3.7 cm
(basketball), median 1.44 cm over takes. Noise of sigma per coordinate on each frame puts noise of about
sqrt(2) x sqrt(3) x sigma on each step: 1.2 cm at sigma 0.005 and 2.4 cm at 0.01. At the training level 0.005 the
step noise is already larger than the median step in 7 of the 20 takes (the slow, mostly procedural ones); at 0.01
in 13 of 20, and at 0.02 and above in all 20. Even the training level therefore buries the fine motion of slow
takes, which is where the paper finds the trajectory weakest.

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

`bash ~/dl4cv/position_noise_seed.sh <seed> --with_tests` submits A and N as two jobs (about 3 hours of training
each) and tests each run on clean inputs and at sigma 0.005, 0.01, 0.02 and 0.05 m with noise seeds 0, 1 and 2,
through `eval_run.sh` (the standard 4-second test window). `bash ~/dl4cv/position_noise_seed.sh smoke` runs one
epoch of N first.

## Table to fill
Motion-to-text accuracy, overall and procedural, per seed and mean over the three seeds:

| Run | clean | 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|---|
| A | | | | | |
| N | | | | | |

This table is descriptive: at each noise level the number is the mean over the three noise seeds, which are
repeated measurements of the same model, not extra training runs. The claims rest only on the rule below.

## Reporting rule (fixed before any run)
Written before any noisy test result existed: when the training level was changed from 0.01 to 0.005, only the
smoke test had run, and it gives no accuracy.
- Main comparison: N against A on **mean accuracy over the four noise levels** (0.005, 0.01, 0.02, 0.05 m),
  noise seed 0, motion-to-text overall, with the paired bootstrap of `compare_runs.py` over seeds and recordings.
  N is called more robust if the interval of the difference is above zero. This follows the standard protocol
  for corruption robustness (Hendrycks and Dietterich 2019), which scores a model across all severity levels of a
  corruption rather than at one chosen level, so no single level can be picked after the fact. Accuracy at each
  level, including the matched level 0.005, is reported as a curve without separate claims.
- Clean cost: N against A on clean inputs, same test. N is said to keep clean accuracy only if the whole interval
  lies above -1.0 point (the same bar as experiment 3). With 3 seeds an interval about 2 points wide is typical
  (experiment 1), so a cost under 1 point may not be detectable; that is said if it happens.
- Noise seeds 1 and 2 are reported as a check that the result does not depend on one draw of noise.

## Limitations
- Independent noise on every frame is a simple stress test. Real tracking error is likely to be smoother (Meta's
  Aria accuracy is a global error of whole trajectories, and Aria's open-loop error is described as drift), which
  would disturb frame-to-frame motion less; we have not measured its shape. Drift and tracking loss are not tested.
- N is trained on the same kind of noise it is tested on, so this measures robustness to a known corruption,
  not to unseen ones (the corruption benchmark keeps its test corruptions out of training; Hendrycks and
  Dietterich 2019). Whether it carries over to real tracking error is not tested here.
- Per-frame jitter is "one of the most commonly employed" augmentations for motion sensor data (Wu et al. 2025),
  and has been found not to help a wrist-accelerometer classifier (Um et al. 2017) and to hurt IMU-based pose
  estimation (Wu et al. 2025). It was chosen knowingly as the
  standard, simplest perturbation.

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
