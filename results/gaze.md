# Experiment 3 results: eye gaze as a second input (seed 0 only)

Design, gate and reporting rule in [`docs/gaze.md`](../docs/gaze.md). The pilot gate (training takes only) was not
met (`docs/gaze.md`, "Pilot gate decision"), so seeds 1 and 2 were not trained. As fixed in advance, seed 0 of all
five runs is scored on the test and reported here, **single-seed: no claim is made from these numbers**.

Training: `bash ~/dl4cv/gaze_seed.sh 0`, code d8fc889, jobs 710385-710386. Test: `eval_run.sh gz_<run>_s0` for each
run (gaze flags read from the run's `args.json`), code ccc49bc, job 713312. Ego-Exo4D 5-way test, motion-to-text (%).

| Run | Encoder | Gaze | Procedural | Physical | Overall |
|---|---|---|---|---|---|
| G0 | trains | none | 33.35 | 51.44 | 44.74 |
| GT | trains | real | 34.07 | 52.18 | 45.47 |
| GT-shuf | trains | wrong (donor) | 33.23 | 51.14 | 44.51 |
| GF | frozen | real | 33.54 | 51.84 | 45.06 |
| GF-shuf | frozen | wrong (donor) | 33.46 | 51.23 | 44.65 |

Single-seed differences, `python compare_runs.py --a <A>/test_questions.csv --b <B>/test_questions.csv` (95%
interval over test takes only; says nothing about retraining):

| Comparison | Procedural | Physical | Overall |
|---|---|---|---|
| GT - G0 | +0.72 (-0.39 to +1.84) | +0.74 (-0.07 to +1.60) | +0.73 (+0.08 to +1.43) |
| GT - GT-shuf | +0.84 (-0.12 to +1.77) | +1.03 (+0.40 to +1.67) | +0.96 (+0.44 to +1.50) |
| GF - G0 | +0.19 (-0.66 to +1.08) | +0.40 (-0.29 to +1.09) | +0.32 (-0.22 to +0.89) |
| GF - GF-shuf | +0.08 (-0.51 to +0.66) | +0.61 (+0.16 to +1.04) | +0.41 (+0.05 to +0.77) |

## Reading
- Descriptive only (one seed). The direction matches the training-take gate: real gaze with full fine-tuning is
  slightly ahead of no gaze and of wrong gaze, by under one point; the frozen variant changes little.
- The reporting rule in `docs/gaze.md` needs three seeds per run, so none of its claims ("gaze helps", "frozen is
  enough", "no cost on physical") can be evaluated.
- Alignment of gaze and pose was not verified (`data_files/egoexo4d/gaze_alignment_decision.txt`); small or absent
  gains are "no gain with gaze as aligned here", not evidence that gaze cannot help.
