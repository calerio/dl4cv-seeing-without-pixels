# Experiment 3 results: eye gaze as a second input (seed 0 only)

Design, gate and reporting rule in [`docs/gaze.md`](../docs/gaze.md). The pilot gate (training takes only) was not
met (`docs/gaze.md`, "Pilot gate decision"), so seeds 1 and 2 were not trained. As fixed in advance, seed 0 of all
five runs is scored on the test and reported here, **single-seed: no claim is made from these numbers**.

Training: `bash ~/dl4cv/gaze_seed.sh 0`, code d8fc889, jobs 710385-710386. Test: `eval_run.sh gz_<run>_s0` for each
run (gaze flags read from the run's `args.json`), code ccc49bc, job 713312. Between d8fc889 and ccc49bc only the diagnostics and the new error on a dropped
gaze layer changed (`git diff --stat d8fc889 ccc49bc -- src/camformer`); test scoring is unchanged. Ego-Exo4D 5-way test, motion-to-text (%).

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
- Procedural, the split named in advance: all four single-seed intervals include zero. The point estimates are
  positive (GT - G0 +0.72, GT - GT-shuf +0.84, GF - G0 +0.19, GF - GF-shuf +0.08).
- Some overall and physical intervals exclude zero (e.g. GF - GF-shuf overall +0.41, +0.05 to +0.77), but they are
  not the primary split and are picked from 12 intervals with no correction. All intervals cover test takes only,
  not retraining.
- The reporting rule in `docs/gaze.md` needs three seeds per run, so none of its claims ("gaze helps", "frozen is
  enough", "no cost on physical") can be evaluated.
- The shuffled controls are not the same kind on both splits, as the design note says (`docs/gaze.md`, "Donor
  map" and "Shuffled input"). On test procedural clips 94.3% of donors come from another moment of the same take; on
  test physical clips 42.6% come from another take of the same scenario, i.e. another person and session. A -shuf run
  also sees the validity flag off on more frames than its real twin: from 0.2% more (test procedural) to 2.4% more
  (training and validation physical). The intervals above that exclude zero are all physical or overall; that is
  also where the controls differ most from real gaze, and this note does not say whether that explains them.
- GF - G0 changes two things: G0 fine-tunes the whole encoder, GF freezes it and trains only the gaze layer. The
  matching no-gaze model for GF is the released checkpoint (44.81 overall, 33.61 procedural); GF - GF-shuf is the
  matched comparison for the frozen runs (with the control caveats above).
- Alignment of gaze and pose was not verified (`data_files/egoexo4d/gaze_alignment_decision.txt`); small or absent
  gains are "no gain with gaze as aligned here", not evidence that gaze cannot help.
