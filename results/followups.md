# Follow-ups to experiment 1: grouping only (G), and Nymeria zero-shot

Design, tables and rules in [`docs/followups.md`](../docs/followups.md) (decision D5), fixed before these results
were read. Motion-to-text accuracy (%) unless said otherwise.

## Runs and jobs
- **G** (`--same_take_k 8 --same_take_gap 4 --neg_mask verb`, no within-take loss; other flags as in
  `results/hard_negatives.md`): seeds 0-2, `sbatch train.sh ... --seed <s> --job_name p1_G_s<s>`, jobs 736106,
  736107, 736109, code 3d394e7 (3 h 21 min to 3 h 37 min each).
- **Tests:** `scripts/slurm/followup_evals.sh`, job 737586, code 82b14dd: `eval_run.sh` on G (Ego-Exo4D test) and
  `eval_nymeria.sh` on the released 16 s and 8 s checkpoints and on A, C, B', G (seeds 0-2).
- **Every number below:** `scripts/slurm/followup_results.sh`, job 737920 (two-level bootstrap over seeds and test
  takes or sessions, 10,000 resamples, seed 0). Experiment 1's text-to-motion comparisons: job 737624, same method.
- **Training equivalence** (`scripts/slurm/check_training_equivalence.sh`, job 737619, training code 82b14dd against
  bb5dc84; one epoch of B', seed 0): largest weight difference between two runs of the current code 0.000142, between
  old and current code 0.000166, between seed 1 and seed 0 of the current code (positive control) 0.00592, about 42
  times the noise. The check can tell a real change from GPU noise, and old and new code train alike (0.000166 <= 2 x
  0.000142). So B' - G and G - C below are controlled comparisons. (A first run without the control, job 737585, gave
  0.0002 and 0.000162.) The check covers one epoch only.

## 1. Grouping only (G)

| Setup | Seed | Procedural | Physical | Overall |
|---|---|---|---|---|
| G | 0 | 33.96 | 51.68 | 45.12 |
| G | 1 | 34.11 | 52.27 | 45.54 |
| G | 2 | 33.12 | 51.75 | 44.85 |

For reference (`results/hard_negatives.md`): procedural A 33.27 / 33.61 / 33.19, C 33.50 / 33.57 / 33.16,
B' 37.09 / 36.44 / 37.20.

| Comparison | Split | Difference | 95% interval |
|---|---|---|---|
| B' - G | procedural | +3.18 | +1.95 to +4.49 |
| B' - G | physical | +1.50 | +0.38 to +2.43 |
| B' - G | overall | +2.12 | +1.28 to +2.89 |
| G - C | procedural | +0.32 | -0.60 to +1.23 |
| G - C | physical | -0.11 | -0.88 to +0.59 |
| G - C | overall | +0.05 | -0.61 to +0.67 |
| G - A (descriptive) | procedural | +0.37 | -0.66 to +1.35 |

**Against the rule.**
- "The within-take loss adds to grouping": interval of B' - G on procedural above zero (+1.95 to +4.49), and B'
  above G in each seed (37.09 vs 33.96, 36.44 vs 34.11, 37.20 vs 33.12). **Met.**
- "Grouping alone helps": interval of G - C on procedural includes zero (-0.60 to +1.23), and G is below C in seed 2
  (33.12 vs 33.16). **Not met:** no gain from grouping alone was detected.

**Reading.** Nearly all of B''s gain over the baseline comes with the within-take loss term: grouping clips by
recording without the term is indistinguishable from C (and from A). This matches the measurement behind D3 that
grouped negatives carry under 3% of the loss's weight on negatives unless a term focuses on them. It does not show
that grouping is unnecessary: the term needs same-recording clips in the batch to work on.

**Text-to-motion (descriptive).** B' - G: procedural +0.71 (-0.78 to +2.24), overall +0.81 (-0.10 to +1.73);
G - C: procedural -0.24 (-1.50 to +0.95), overall -0.41 (-1.23 to +0.38). Per seed, procedural t2m: G 40.10 /
38.80 / 39.22. As for B' - A (`results/hard_negatives.md`), the gain shows in the motion-to-text direction only.

## 2. Nymeria zero-shot (descriptive only)

Per narration type and pooled over 4,000 questions; for our setups, the mean of 3 seeds with the per-seed range of
the pooled figure.

| Model | Body posture | Hands/arms | Legs/feet | Focus | All (range) |
|---|---|---|---|---|---|
| Released 16 s (the paper's Nymeria model) | 36.30 | 26.00 | 30.80 | 33.20 | 31.57 |
| Released 8 s | 33.20 | 21.90 | 29.40 | 31.30 | 28.95 |
| A | 32.60 | 23.17 | 28.83 | 30.87 | 28.87 (28.45-29.43) |
| C | 32.83 | 23.13 | 28.87 | 31.20 | 29.01 (28.80-29.23) |
| B' | 32.70 | 22.47 | 28.17 | 30.87 | 28.55 (27.80-29.07) |
| G | 32.90 | 23.13 | 28.87 | 30.57 | 28.87 (28.35-29.23) |

The released 16 s model reproduces all four of the paper's Nymeria numbers and its pooled 31.6.

| Comparison | Body posture | Hands/arms | Legs/feet |
|---|---|---|---|
| B' - A | +0.10 (-1.75 to +1.93) | -0.70 (-2.54 to +1.02) | -0.67 (-2.01 to +0.65) |
| G - A | +0.30 (-1.47 to +2.02) | -0.03 (-1.69 to +1.66) | +0.03 (-1.61 to +1.61) |

No interval for focus of attention (4 sessions) or for the pooled number (about 11 effective sessions), as the rule
fixed.

**Reading.** No claim is made from Nymeria. Every interval includes zero: neither same-recording negatives nor
grouping changes zero-shot transfer measurably. All our fine-tuned models score about where the released 8-second
model does (28.95 pooled), about 2.6 points below the 16-second model behind the paper's Nymeria column, so
comparing our numbers with the paper's would mostly measure the context length, not our changes. B''s procedural
gain on Ego-Exo4D does not carry over to Nymeria's narration types; whether that is because the gain is specific to
Ego-Exo4D's question format or because Nymeria's narrations (body posture, limb motion, attention) are a different
task cannot be told from these runs.

## Limits
- Three seeds per setup; the seed part of each interval is rough. The per-seed numbers are shown.
- The training-equivalence check covers one epoch.
- Nymeria's test clusters into few sessions (one gives 1,132 of the 4,000 questions), so its intervals are wide and
  the pooled one is not reported.
