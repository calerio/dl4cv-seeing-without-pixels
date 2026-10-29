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
