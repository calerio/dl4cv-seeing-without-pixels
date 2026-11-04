# Experiment 1 results: same-take hard negatives

Design in [`docs/hard_negatives.md`](../docs/hard_negatives.md); decisions, gate and the rule for the final claim in
[`docs/decisions.md`](../docs/decisions.md) (D3). Follow-ups (grouping without the loss term, Nymeria zero-shot):
[`results/followups.md`](followups.md); the within-take loss term carries the gain, grouping alone shows none. Ego-Exo4D 5-way test, motion-to-text accuracy (%).

## Runs
All fine-tune `checkpoints/egoexo4d_dur8.pt` for 20 epochs, batch 1,024, lr 1e-4, one A100; last checkpoint tested.
Common flags: `--dataset egoexo4d_pretrain_longseq --pose_encoding rel9d_grav --take_duration 8 --sample_dur
--epochs 20 --init_ckpt checkpoints/egoexo4d_dur8.pt --seed <s>`.

| Setup | Extra flags | Runs (job name) |
|---|---|---|
| A | none (authors' setup) | `p1_A_s0`, `p1_A_s1`, `p1_A_s2` |
| C | `--neg_mask verb` | `p1_C_s0`, `p1_C_s1`, `p1_C_s2` |
| B' | `--same_take_k 8 --same_take_gap 4 --neg_mask verb --within_take_weight 1` | `p1_Bp_s0`, `p1_Bp_s1`, `p1_Bp_s2` |

Seed 0 of A and B' (the pilot): jobs 704740-704743, code a8fc1ec. All other runs: jobs 705874-705877, code bb5dc84
(the only difference under `src/camformer/` is the new file `compare_runs.py`, see D3). Job wall times (`sacct -X -o JobID,Elapsed`): 705874 5:39:48 and 705875 6:40:49 and 705876 5:45:52 (two runs each),
705877 3:20:29 (one run). Test split sizes (`score_mcq.py`, `n` column): 2,621 procedural, 4,458 physical, 7,079 in all.
Each run is tested by
`scripts/slurm/eval_run.sh`, which writes `test_questions.csv` (one row per question) in the run's log directory.

## Per seed

| Setup | Seed | Procedural | Physical | Overall |
|---|---|---|---|---|
| A | 0 | 33.27 | 51.73 | 44.89 |
| A | 1 | 33.61 | 52.38 | 45.43 |
| A | 2 | 33.19 | 51.50 | 44.72 |
| C | 0 | 33.50 | 52.09 | 45.20 |
| C | 1 | 33.57 | 52.40 | 45.43 |
| C | 2 | 33.16 | 51.55 | 44.74 |
| B' | 0 | 37.09 | 53.34 | 47.32 |
| B' | 1 | 36.44 | 53.86 | 47.41 |
| B' | 2 | 37.20 | 52.98 | 47.14 |
| Released model | - | 33.61 | 51.39 | 44.81 |

## Text-to-motion (descriptive)
The design (`docs/hard_negatives.md`) planned both directions; the claims below are on motion-to-text, the paper's
metric. Text-to-motion (a caption picks among five trajectories), job 737624 (`scripts/slurm/t2m_exp1.sh`, the file that ran, byte for byte):

| Setup | Procedural (seeds 0 / 1 / 2) | Physical, mean | Overall, mean |
|---|---|---|---|
| A | 39.03 / 39.64 / 39.37 | 52.36 | 47.54 |
| C | 38.99 / 39.68 / 40.18 | 52.67 | 47.84 |
| B' | 40.82 / 39.49 / 39.95 | 53.03 | 48.24 |

| Comparison | Split | Difference | 95% interval |
|---|---|---|---|
| B' - A | procedural | +0.74 | -0.73 to +2.21 |
| B' - A | overall | +0.70 | -0.21 to +1.64 |
| B' - C | procedural | +0.47 | -1.08 to +2.03 |
| C - A | procedural | +0.27 | -0.74 to +1.19 |

The gain is much smaller in this direction, every interval includes zero, and in seed 1 B' is below A and C on
procedural questions. By reading the code, both loss terms are symmetric, averaging the
motion-to-text and text-to-motion cross-entropies (`tasks/pretrain.py`, `CLIPLoss` and `within_take_loss`), so the
asymmetry does not come from training one direction only; no run logged the two directions' losses separately, and
why the gain is one-sided was not tested. B'
trains on the test's own construction (wrong options from the same recording with another verb), which a reader
should keep in mind; the Nymeria follow-up (`results/followups.md`) finds no measurable change there either.

## Comparisons over three seeds
`python compare_runs.py --a <A or C files> --b <B' files>` (two-level bootstrap over seeds and takes, 10,000
resamples, seed 0). With three seeds per setup the seed level of the interval is rough; the per-seed rows above are
the more direct evidence.

| Comparison | Split | Difference | 95% interval |
|---|---|---|---|
| B' - A | procedural | +3.55 | +2.37 to +4.83 |
| B' - A | physical | +1.53 | +0.31 to +2.64 |
| B' - A | overall | +2.27 | +1.38 to +3.12 |
| B' - C | procedural | +3.50 | +2.32 to +4.74 |
| B' - C | physical | +1.38 | +0.11 to +2.50 |
| C - A | overall | +0.11 | -0.57 to +0.74 |

## Against the rule fixed on 18 Oct (D3), before seeds 1 and 2 were seen
- Procedural gain: interval of B' - A above zero (+2.37 to +4.83), and B' beats A in seed 1 (36.44 vs 33.61) and in
  seed 2 (37.20 vs 33.19). Met.
- More than the verb rule: B' - C on procedural above zero (+2.32 to +4.74). Met.
- No cost on physical activities: lower end of B' - A on physical above -1 (+0.31). Met.

## Limits
- The go decision for seeds 1 and 2 rested on seed 0's test result and on a post-hoc reading of one gate criterion
  (D3). Seeds 1 and 2, trained after that decision, each show the gain on their own.
- B' changes two things together: batches grouped by take, and the within-take loss. The loss needs the grouping to
  have same-take pairs to work on. Grouping without the loss gave no gain in an earlier, unseeded run (16 Oct), which
  is supporting evidence only.
- Seeds vary training only; all runs share one test set of 7,079 questions from 671 takes.
