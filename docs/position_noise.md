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

**Units (checked 12 Nov 2026 on the stored files).** Positions are in metres: six random training takes are all
float64, all finite, with unit quaternions; a clip moves 1 to 3 m in total, and the median step between stored
frames is 0.9 to 3.4 cm (99th percentile 5 to 11 cm). So the training noise of 0.005 is 0.5 cm per coordinate (about 0.9 cm in total), below one ordinary
frame-to-frame step and below the accuracy Meta reports for Aria's trajectories (no more than 1.5 cm in
room-scale scenes; Engel et al. 2023, arXiv:2308.13561); the test level 0.01 is about that accuracy.

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

At each noise level the number is the mean over the three noise seeds; the noise seeds are repeated measurements
of the same model, not extra training runs.

## Reporting rule (fixed before any run)
- Main comparison: N against A at sigma 0.01 (Aria's reported accuracy), noise seed 0, with the paired bootstrap of `compare_runs.py` over
  seeds and recordings. N is called more robust if the interval of the difference is above zero.
- Clean cost: N against A on clean inputs, same test. A drop is reported with its interval; "no harm" is not
  claimed just because the interval includes zero.
- The other noise levels and seeds are reported as a curve, without separate claims.

## Limitations
- Independent noise on every frame is a simple stress test. Real tracking error is mostly smooth drift, which
  disturbs frame-to-frame motion less; drift and tracking loss are not tested.
- At 1 cm the noise is about as large as an ordinary step between frames, and 5 cm several times larger, so the
  higher levels mostly test how much fine motion the model can lose.
- Missing frames and rotation noise are not tested.
