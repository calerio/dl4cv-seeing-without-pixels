# Experiment 4 results: position noise (13 Nov 2026)

Design, flags and the reporting rule: `docs/position_noise.md` and D7, fixed before any run.

## Runs
All runs fine-tune `egoexo4d_dur8.pt` for 20 epochs on code revision af6c5c5 (`REVISION_AT_START` of every run).
Reproduction check on that revision: job 757950, 44.81 overall, 33.61 procedural.

| Run | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| A (no training noise) | 757982 (training, 12 tests), 758588 (2 tests) | 760227 | 758759 |
| N (sigma 0.005 m, half the clips) | 757983 (training, clean test), 758589 (12 tests) | 758594 | 758760 |

Seed 0's tests were finished in separate test-only jobs because the training jobs were stopped by a full home
directory on the cluster. N seed 0's `args.json` was emptied by the same problem and was rewritten from the arguments
stored in its own checkpoint; it differs from A seed 0 only in the noise sigma and the job name. A seed 1 was
restarted from scratch (first attempt 758574 made no progress and was stopped by the time limit).

Commands (on the cluster, from `~/dl4cv/camformer`):

    python average_noise_levels.py --run_dir <log dir of run> --out <run>_meanlevels.csv     # per run
    python compare_runs.py --a A_s{0,1,2}_meanlevels.csv --b N_s{0,1,2}_meanlevels.csv       # main
    python compare_runs.py --a <A runs>/test_questions.csv --b <N runs>/test_questions.csv   # clean cost

## Main comparison (rule fixed before the runs)
Motion-to-text accuracy, mean over the four noise levels (noise seed 0), N - A, paired bootstrap over seeds and
takes, 95% interval. Per-seed ranges in brackets.

| Split | A | N | N - A | interval |
|---|---|---|---|---|
| overall | 43.43 (43.30-43.58) | 43.64 (43.49-43.76) | +0.21 | -0.08 to +0.51 |
| procedural | 30.49 (30.33-30.71) | 30.72 (30.49-31.08) | +0.23 | -0.28 to +0.76 |

**No gain resolved.** The interval includes zero, so N is not called more robust. The interval is about 0.6 points
wide, so a gain larger than about half a point would likely have been resolved. Noise seeds 1 and 2 give the same
picture (+0.15, interval -0.14 to +0.44; +0.22, interval -0.03 to +0.46).

## Clean cost
| Split | A | N | N - A | interval |
|---|---|---|---|---|
| overall | 44.93 (44.67-45.16) | 44.89 (44.81-45.06) | -0.03 | -0.44 to +0.38 |
| procedural | 33.55 (33.31-33.69) | 33.40 (33.12-33.57) | -0.15 | -0.76 to +0.42 |

No cost resolved, with the interval above.

## Accuracy at each noise level (descriptive)
Motion-to-text overall, mean over the three training seeds and the three noise seeds.

| Run | clean | 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|---|
| A | 44.93 | 44.80 | 44.91 | 44.28 | 39.71 |
| N | 44.89 | 44.91 | 44.95 | 44.70 | 39.89 |

Both models keep their clean accuracy up to 1 cm of noise per coordinate and lose about 5 points at 5 cm. Training
with noise moves the curve by less than half a point at every level.
