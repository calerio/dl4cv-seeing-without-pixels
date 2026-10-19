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
specialist on its test. All runs finished; U1 is +0.69 against S-ego and -4.40 against S-exo. Met, with the exo gap
close to the limit. Seeds 1 and 2 were launched on the same code (jobs 708367-708370).

These are single-seed numbers: no claim is made from them. The final claims follow the rule in `docs/ego_exo.md`
once seeds 1 and 2 are in.
