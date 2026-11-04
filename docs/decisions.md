# Decisions

Choices that change how we run or compare experiments, with the reason for each. Newest first.
Each one is meant to be cited in the report.

## D4. Experiment 2 puts exo trajectories through the ego model's input path (18 Oct 2026)

**Situation.** The joint model (docs/ego_exo.md) needs exo (DynPose, ViPE poses) and ego (Ego-Exo4D) clips in one
encoder. The authors trained them with different tasks, rates and axes.

**Decision.** In the joint runs and in E->X, exo clips go through the long-sequence (ego) path with the whole clip
as one window, are resampled from 30 to 20 FPS (`--exo_resample_hz 20`), and are turned into the ego camera axes
by a 90 degree turn about the forward axis (`--exo_to_aria_axes`). S-exo keeps the authors' exo task unchanged.

**Evidence.** The released exo model scores 46.30 through the authors' path and 46.20 through ours at the native
rate (jobs 706967, 706944). Resampling costs about one point on a 5-way check over 7,925 validation clips outside
the test (50.42 against 49.38, job 706966); it is kept as a choice, not as a gain. The axis turn follows from the
stored ego frames (camera x has a world-up component of -0.81, `exo_checks.py`, jobs 706934, 706937) and ViPE's
OpenCV convention.

**Consequence for comparisons.** U1 against S-exo differs in path, rate and axes as well as in sharing; U1 against
E->X shares all three and is the clean test of sharing.

## D3. Hard negatives: fine-tune, add a within-take loss term, and gate the full runs on one pilot (16 Oct 2026)

**Situation.** Two first fine-tuning runs (20 epochs from the released checkpoint, without a fixed seed) gave
the same test accuracy as the released model whether batches were random or grouped by take (44.79 and 44.84
against 44.81 motion-to-text). Before changing anything we measured why, on the released checkpoint and both
runs, without training (`diagnose_within_take.py`):

| Batches | Same-take negatives per clip (different verb) | Share of the softmax over negatives that they carry |
|---|---|---|
| random | 1.1 of 1,023 | 0.6% |
| 8 clips per take | 6.0 of 1,023 | 2.8% |

So grouping 8 clips per take changed less than 3% of what the loss pushes on: the hard negatives were
diluted, not ineffective. The model is also far from solving the within-take task where it was trained: on
5,000 within-take questions built from training takes with the paper's rule, it picks the right caption 59.8%
of the time (60.2% and 60.4% after the two runs). There is room to learn, and the first runs did not use it.

**Decision.** We keep fine-tuning the released checkpoint at the authors' learning rate, as the course
recommends for a model that is already near an optimum (Lecture 7, transfer learning). Retraining from scratch
would not change the dilution and would cost about 70 GPU-hours (D1). Instead we add a second loss term
(`--within_take_weight`, weight 1): each clip must pick its own caption among the clips of its own take that
have a different verb, which is close to the format of the test. The authors' loss is kept unchanged next to it.

**Gate, fixed before the pilot.** One pilot pair, seed 0, 20 epochs: A (random batches, authors' loss) and B'
(8 clips per take, verb mask, within-take term). The three-seed runs of A, B' and C (random batches, verb mask)
go ahead only if B' against A shows all three of: a clearly lower within-take loss in training, higher accuracy
on the within-take questions from training takes, and at least 1 point more on the procedural test questions.
Otherwise we report the negative result with these measurements and stop this experiment.

**Gate outcome (18 Oct 2026).** Pilot runs `p1_A_s0` and `p1_Bp_s0` (seed 0, jobs 704740-704743, code
a8fc1ec), diagnostics job 705850 (`diagnose.sh` on both `last.ckpt` files, defaults: 20 batches, K=8, 5,000
questions, seed 0):

| Criterion | A | B' | Result |
|---|---|---|---|
| 1. Within-take loss on training batches, 8 clips per take | 1.467 | 1.405 | lower, but see below |
| 2. Within-take 5-way MCQ on training takes (m2t) | 60.50 | 61.92 | passed |
| 3. Procedural test, m2t, at least +1 point | 33.27 | 37.09 | passed |

Criterion 1 said "clearly lower" with no number. The drop is 4%; for scale, plain fine-tuning (released model to
A) moved it from 1.473 to 1.467. Calling 4% "clear" is a reading made after the measurement, so by the letter of
the rule criterion 1 is undecided, and the gate passed only under that post-hoc reading. We went ahead with the
seed runs on that basis and say so here.

Two limits of the gate itself, which we did not see when we set it:
- Criterion 3 uses the test set to decide whether to continue. It tunes no model, but our final result is
  conditional on seed 0 having looked good on the test. To keep the evidence separate, the report gives seeds
  1 and 2, trained after the decision, on their own as well as the three-seed mean.
- Seed 0 alone is not evidence of an effect: its intervals below cover test sampling only.

For the record, the seed-0 comparison on the test (`python compare_runs.py --a p1_A_s0.csv --b p1_Bp_s0.csv`, the
files being each run's `test_questions.csv` from `eval_run.sh`; 10,000 take resamples, seed 0): B' minus A is
+3.82 points on procedural questions (95% interval +2.50 to +5.18) and +2.43 overall (+1.25 to +3.48).

**Rule for the final claim, fixed 18 Oct before any result of seeds 1 and 2 is seen.** On motion-to-text
accuracy, comparing B' with A over seeds 0-2 with `compare_runs.py` (two-level bootstrap over seeds and takes):
- "Same-take negatives improve procedural accuracy" is claimed only if the 95% interval of B' minus A on
  procedural questions is above zero, and B' beats A on procedural questions in seed 1 and in seed 2 separately
  (the seeds trained after the go decision).
- "The gain comes from more than the verb rule" is claimed only if, by the same test, B' also beats C.
- "No cost on physical activities" is claimed only if the interval of B' minus A on physical questions has a
  lower end above -1 point.
- Anything else is reported as measured, including a negative result.

The regression check passed before the seed runs were queued (job 705870, `reproduction OK` at 44.81 / 33.61) on
revision bb5dc84. Between a8fc1ec (the pilot) and bb5dc84 the only change under `src/camformer/` is the new
file `compare_runs.py` (`git diff --stat a8fc1ec bb5dc84 -- src/camformer`), which training and testing do not
import, so the check covers the pilot's code too.

## D2. Checkpoints leave out the frozen CLIP text model (16 Oct 2026)

**Situation.** Each Lightning checkpoint was about 620 MB, and a run keeps two (best and last). Almost all of
that is the CLIP ViT-B/32 model, which is frozen and loaded again from its own file every time training
starts. Our cluster quota is 100 GB, already three-quarters full. W&B was also set to keep a copy of each
checkpoint (`log_model=True`).

**Decision.** When CLIP is frozen, its weights are removed from the checkpoint before it is saved and filled
back in from the freshly loaded model when a checkpoint is loaded (`drop_frozen_clip` and
`restore_frozen_clip` in `tasks/pretrain.py`). Full checkpoints from before this change still load. If CLIP is
fine-tuned (`--finetune_clip`), nothing is removed. W&B no longer stores checkpoints (`log_model=False`).

**What it changes.** Nothing about training. A checkpoint is now 12 MB instead of 620 MB.

**Test (16 Oct 2026).** A 2-epoch run was split across two 16-minute jobs (703246, 703247). The first was
stopped by the time limit during epoch 2. The second restored model, optimizer and epoch from the 12 MB
checkpoint and finished the run.

## D1. Training runs are split into chained 24-hour jobs (15 Oct 2026)

**Situation.** The authors trained on 8 GPUs (supp. 2.3); `train.py` defaults to 500 epochs, and the released 8-second
checkpoint is from epoch 498 (its `source_ckpt` field). The paper gives no epoch count. We have one
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
