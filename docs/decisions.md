# Decisions

Choices that change how we run or compare experiments, with the reason for each. Newest first.
Each one is meant to be cited in the report.

## D1. Training runs are split into chained 24-hour jobs (15 Oct 2026)

**Situation.** The authors trained on 8 GPUs for 500 epochs (the defaults in `train.py`). We have one
student account on the Bocconi cluster: at most 2 GPUs, 2 running jobs and 24 hours per job. On one
A100, one Ego-Exo4D epoch takes about 8 minutes: under 6 minutes of training and about 2.5 minutes
for the validation pass that runs after every epoch (timing run, 15 Oct). The full 500-epoch schedule
would take about 70 hours: nearly three times the job limit. The released code saves a checkpoint
after every epoch, but it has no way to continue training from one: `--ckpt` only re-evaluates.

**Decision.** We added a `--resume` option to `train.py`. With it, training continues from the run's
last checkpoint (model weights, optimizer state and epoch number) when one exists, and starts fresh
otherwise. A long run is then submitted as a chain of jobs with the same run name, each starting when
the previous one ends (`scripts/submit_chain.sh`, which uses SLURM job dependencies).

**What it changes.** Nothing about the model, the loss or the data. A run that is cut off at the time
limit repeats at most the epoch it was in, about 8 minutes per 24-hour job. Other differences from the
authors' setup remain and are reported separately. The main one: with eight GPUs, each GPU takes its
own batch of 1,024 clips and computes the contrastive loss within it (the code does not pool clips
across GPUs), and the eight gradients are averaged into one update. So each clip is compared against the
same 1,023 others as in our runs, but one of our epochs makes eight times as many, smaller updates
(155 instead of about 20).

**Test (15 Oct 2026).** A 4-epoch Ego-Exo4D run was split across two 25-minute jobs (702170, 702171).
The first job finished epochs 0 and 1 and was stopped by the time limit during epoch 2. The second job
started on its own, restored the saved state and continued from epoch 2 to the end. Validation loss
against an uninterrupted run with the same settings (job 702054):

| Epoch | Uninterrupted | Split across two jobs |
|---|---|---|
| 0 | 6.52 | 6.52 |
| 1 | 6.40 | 6.42 |
| 2 | 6.36 | 6.37 (after the restart) |
| 3 | not run | 6.35 |

The loss carries on from where it stopped rather than starting again. The small differences come from
random data sampling, which is not seeded in the released code. The interrupted epoch was repeated in
full, which cost about 8 minutes.
