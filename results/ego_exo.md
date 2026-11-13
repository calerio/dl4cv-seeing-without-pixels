# Experiment 2 results: one model for ego and exo trajectories

Design, runs and the reporting rule in [`docs/ego_exo.md`](../docs/ego_exo.md); input choices in D4
([`docs/decisions.md`](../docs/decisions.md)). Motion-to-text accuracy (%) on the Ego-Exo4D 5-way test (7,079
questions) and the DynPose-100K ViPE 5-way test (1,000 questions).

## Seed 0 (pilot)
Launched with `bash ~/dl4cv/ego_exo_seed.sh 0` on code 402cc33 (jobs 707658, 707659). Each run's tests write
`test_questions_ego.csv` / `test_questions_exo.csv` in its log directory.

| Run | Ego test | Exo test |
|---|---|---|
| S-ego (`ee_Sego_s0`) | 44.65 | - |
| S-exo (`ee_Sexo_s0`) | - | 47.30 |
| E->X (`ee_EX_s0`) | 32.32 | 46.70 |
| U0, no explicit token (`ee_U0_s0`) | 45.15 | 42.10 |
| U1, domain token (`ee_U1_s0`) | 45.35 | 42.90 |

Single-seed differences, `python compare_runs.py --a <A file> --b <U1 file>` (interval over test takes only):

| Comparison | Test | U1 minus | 95% interval |
|---|---|---|---|
| U1 - S-exo | exo | -4.40 | -7.30 to -1.46 |
| U1 - E->X | exo | -3.80 | -6.54 to -1.11 |
| U1 - U0 | exo | +0.80 | -1.19 to +2.71 |
| U1 - S-ego | ego | +0.69 | -0.09 to +1.42 |
| U1 - U0 | ego | +0.20 | -0.33 to +0.73 |

**Pilot rule** (`docs/ego_exo.md`): seeds 1 and 2 run if every seed-0 run finished and U1 is within 5 points of each
specialist on its test. All runs finished; U1 is +0.69 against S-ego (from unrounded accuracies) and -4.40 against S-exo. Met, with the exo gap
close to the limit. Seeds 1 and 2 were launched on the same code (jobs 708367-708370).

These are single-seed numbers: no claim is made from them. The final claims follow the rule in `docs/ego_exo.md`
once seeds 1 and 2 are in.

## Seeds 0-2 (final)
Seeds 1 and 2: `bash ~/dl4cv/ego_exo_seed.sh 1` and `2`, same code 402cc33. Seed 1: jobs 708367 (U1, S-exo, E->X)
and 708368 (U0, S-ego); seed 2: jobs 708369 and 708370, same split. Seed 0 likewise: 707658 and 707659.

| Run | Seed | Ego test | Exo test |
|---|---|---|---|
| S-ego | 0 / 1 / 2 | 44.65 / 45.26 / 44.95 | - |
| S-exo | 0 / 1 / 2 | - | 47.30 / 46.70 / 46.20 |
| E->X | 0 / 1 / 2 | 32.32 / 32.42 / 31.91 | 46.70 / 44.80 / 45.80 |
| U0 | 0 / 1 / 2 | 45.15 / 44.95 / 45.03 | 42.10 / 46.40 / 45.30 |
| U1 | 0 / 1 / 2 | 45.35 / 45.02 / 44.91 | 42.90 / 43.80 / 45.70 |

Three-seed comparisons, `python compare_runs.py --a <three A files> --b <three B files>` (two-level bootstrap over
seeds and test videos or takes, 10,000 resamples, seed 0; with three seeds the seed level is rough):

| Comparison | Test | Means (A, B) | B - A | 95% interval |
|---|---|---|---|---|
| U1 - S-ego | ego | 44.95, 45.09 | +0.14 | -0.52 to +0.81 |
| U1 - S-exo | exo | 46.73, 44.13 | -2.60 | -5.63 to +0.40 |
| U1 - E->X | exo | 45.77, 44.13 | -1.63 | -4.49 to +1.09 |
| U1 - U0 | exo | 44.60, 44.13 | -0.47 | -3.40 to +2.51 |
| U1 - U0 | ego | 45.04, 45.09 | +0.05 | -0.42 to +0.54 |
| U0 - S-exo (descriptive, not pre-specified) | exo | 46.73, 44.60 | -2.13 | -5.79 to +1.31 |
| E->X - S-ego (descriptive, not pre-specified) | ego | 44.95, 32.22 | -12.74 | -15.48 to -9.77 |

### Against the reporting rule (fixed in `docs/ego_exo.md` before the runs)
- "One model matches the specialists": needs U1 - S inside +-1.5 on Ego-Exo4D and inside +-3 on DynPose. Ego:
  -0.52 to +0.81, inside. Exo: -5.63 to +0.40, not inside. **Not claimed.** The ego interval lies inside the +-1.5
  margin, but the claim needs the exo test too, which fails. On third-person video the test cannot tell whether there is a
  gap, nor rule out one of up to 5.6 points. Read first, as the design note asks, the clean test of sharing: U1
  against E->X, which shares starting weights, input path, rate and axes, is -1.63 (-4.49 to +1.09). Of the 2.6
  points between U1 and S-exo, about 1 is already there without any sharing (E->X is 45.77 against S-exo's 46.73),
  so the gap cannot be read as a cost of sharing.
- "The domain token helps": needs U1 - U0 on DynPose above zero. -0.47 (-3.40 to +2.51). **Not claimed.** As the
  design note expected, the token adds nothing measurable here. A possible reason, not tested, is that the encoder
  already separates the domains through the gravity-unknown token, the translation scale and the clip length.
- Reported as measured (descriptive): after training on exo data alone, E->X scores 32.2 on the ego test, against
  44.81 for the released ego model and 44.95 for S-ego; joint training (U0, U1) keeps ego accuracy at about 45.
  U0 and U1 do not show that drop.

### Limits
- The exo test has 1,000 questions, so exo differences under about 3 points cannot be resolved; seed spread on exo
  is large (U0: 42.1 to 46.4).
- U1 - S-exo also differs in input path, rate and axes (D4); U1 - E->X shares them and gives the same direction
  (-1.6, interval includes zero).

## Version 2: each exo clip at its own frame rate (final)
Version 1 above resampled every exo clip as if it were 30 FPS; about a third are 24 or 25 FPS, so those clips
entered at 16 to 17 poses per second instead of 20 (D6 in [`docs/decisions.md`](../docs/decisions.md)). All five
setups were trained again, seeds 0-2, with each clip's rate (`--exo_fps_table dynpose100k_fps/vipe_fps_v1.csv.gz`):
runs `ee2_{Sego,Sexo,EX,U0,U1}_s{0,1,2}`, `bash ~/dl4cv/ego_exo_seed.sh <seed> v2`, code 17a1a42 (regression check
738231: `reproduction OK`; one-epoch smoke test 738195). Jobs 738234 (seed 0), 738235 (seed 2), 738236 and 738237
(seed 1). All 15 runs finished their 20 epochs on 17a1a42. Every number below comes from
`scripts/slurm/ego_exo_results.sh` as in this commit (4825ad4), job 761251 on deployed code af6c5c5; it reproduces every number of the earlier job 740932. Z with the rate table: job 740931.

Version 1 stays above for the record. As D6 fixes, version 2 is the experiment's result whatever it shows. The
version 1 results, including the seed-0 pilot, were known when the rerun was decided; the rerun follows from the
rate bug, and the reporting rule is the one fixed in `docs/ego_exo.md` before the first runs. Version 2 had no
separate pilot gate: the three seeds were launched together.

| Run | Seed | Ego test | Exo test |
|---|---|---|---|
| S-ego | 0 / 1 / 2 | 44.65 / 45.33 / 44.98 | - |
| S-exo | 0 / 1 / 2 | - | 46.70 / 46.70 / 46.20 |
| E->X | 0 / 1 / 2 | 32.89 / 32.70 / 32.15 | 47.40 / 45.50 / 46.20 |
| U0 | 0 / 1 / 2 | 45.30 / 45.26 / 45.32 | 41.60 / 45.10 / 44.90 |
| U1 | 0 / 1 / 2 | 45.08 / 45.13 / 44.85 | 41.70 / 42.50 / 45.80 |
| Z, released ego model, no exo training | - | - | 26.40 |

Three-seed comparisons, motion-to-text, same bootstrap as above (seeds, then takes or videos; 10,000 resamples):

| Comparison | Test | Means (A, B) | B - A | 95% interval |
|---|---|---|---|---|
| U1 - S-ego | ego | 44.99, 45.02 | +0.03 | -0.61 to +0.68 |
| U1 - S-exo | exo | 46.53, 43.33 | -3.20 | -6.49 to +0.20 |
| U1 - E->X | exo | 46.37, 43.33 | -3.03 | -6.17 to +0.10 |
| U1 - U0 | exo | 43.87, 43.33 | -0.53 | -3.61 to +2.78 |
| U1 - U0 | ego | 45.29, 45.02 | -0.27 | -0.74 to +0.21 |
| U0 - E->X (descriptive) | exo | 46.37, 43.87 | -2.50 | -5.69 to +0.58 |
| U0 - S-exo (descriptive) | exo | 46.53, 43.87 | -2.67 | -6.08 to +0.56 |
| E->X - S-exo (descriptive) | exo | 46.53, 46.37 | -0.17 | -3.02 to +2.66 |
| E->X - S-ego (descriptive) | ego | 44.99, 32.58 | -12.41 | -15.33 to -9.27 |
| U0 - S-ego (descriptive) | ego | 44.99, 45.29 | +0.31 | -0.34 to +0.99 |

### Against the reporting rule
- "One model matches the specialists": ego -0.61 to +0.68, inside +-1.5; exo -6.49 to +0.20, not inside +-3.
  **Not claimed.** As in version 1, the exo test cannot exclude a gap of up to about 6.5 points (bootstrap), or 8.5
  with the seed-level interval below (Limits).
- "The domain token helps": U1 - U0 on DynPose is -0.53 (-3.61 to +2.78), and on Ego-Exo4D it is -0.27.
  **Not claimed.**
- Reported as measured: U0 against E->X, the comparison closest to sharing alone (same start, inputs and exo
  steps, no token), is -2.50 (-5.69 to +0.58). E->X is level with S-exo within about +-3 (-0.17, -3.02 to
  +2.66); E->X and S-exo also differ in inputs, pooling and architecture (`docs/ego_exo.md`), so this does not
  isolate the starting weights. U0 sits 2.5 below E->X, with an interval that includes zero. The data do not say
  where the gap between the joint models and S-exo comes from. Training on exo alone drops the ego score to 32.6 (E->X);
  joint training keeps it at about 45.

### Text-to-motion (descriptive, not part of the rule)
| Comparison | Test | Means (A, B) | B - A | 95% interval |
|---|---|---|---|---|
| U1 - S-ego | ego | 47.33, 47.53 | +0.19 | -0.77 to +1.18 |
| U1 - S-exo | exo | 48.70, 45.77 | -2.93 | -5.95 to +0.13 |
| U1 - U0 | exo | 46.60, 45.77 | -0.83 | -2.73 to +1.11 |
| U1 - U0 | ego | 47.66, 47.53 | -0.13 | -1.01 to +0.69 |
| U1 - E->X | exo | 47.53, 45.77 | -1.77 | -4.34 to +0.75 |

Same direction as motion-to-text.

### Version 2 against version 1
Per setup, version 2 minus version 1 on the same test questions: exo S-exo -0.20, E->X +0.60, U0 -0.73, U1 -0.80;
ego S-ego +0.03, E->X +0.36, U0 +0.25, U1 -0.07. Every interval includes zero (widest exo: U0, -3.78 to +2.31).
No change from correcting the rate was resolved; the exo intervals are about +-3 points, so a shift of that size
would not be detected. S-ego and S-exo do not use the rate, so their rows show how much a rerun on newer code
moves a setup.

### DynPose: test clips from videos that are also in training
No test clip is a training clip, but in the authors' split 941 of the 5,000 test clips (236 of 1,379 test videos)
come from videos with other clips in training; after the scale filter, 845 clips, 212 videos, touching 521 of the
1,000 questions. On the 479 questions with no clip from a training video:

| Setup | All 1,000 | The 479 | Seeds on the 479 |
|---|---|---|---|
| S-exo | 46.53 | 47.53 | 47.18 / 47.81 / 47.60 |
| E->X | 46.37 | 47.39 | 48.23 / 47.81 / 46.14 |
| U0 | 43.87 | 44.89 | 41.96 / 46.35 / 46.35 |
| U1 | 43.33 | 45.44 | 44.05 / 45.51 / 46.76 |

Every setup scores about 1 to 2 points higher on the unseen-video questions, so the overlap does not inflate the
scores. On them U1 - S-exo is -2.09, U1 - E->X -1.95, U0 - E->X -2.51, U1 - U0 +0.56: the same picture.

### Limits
- Three seeds understate the seed spread on exo. A Welch t interval over the three seed means alone, test
  questions held fixed, is wider than the bootstrap: U1 - S-exo -3.20 (-8.47 to +2.07), U1 - E->X -3.03 (-7.63 to
  +1.56), U0 - E->X -2.50 (-6.60 to +1.60), U1 - U0 -0.53 (-5.25 to +4.18). On ego the two agree (U1 - S-ego
  +0.03, -0.68 to +0.75). The exo intervals in the tables above should be read as too narrow, not too wide.
- The exo test has 1,000 questions and the joint models vary by up to 4 points between seeds (U1: 41.7 to 45.8).
- U1 - S-exo also differs in input path, rate and axes (D4); E->X shares them with the joint models.
