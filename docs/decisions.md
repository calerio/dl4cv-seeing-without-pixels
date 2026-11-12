# Decisions

Choices that change how we run or compare experiments, with the reason for each. Newest first.
Each one is meant to be cited in the report.

## D7. Experiment 4 adds noise to camera positions, sigma 0.01 m on half the training clips (12 Nov 2026)

**Situation.** Experiment 4 (`docs/position_noise.md`) asks whether training on noisy positions makes the model
more robust to noisy test positions. The paper never adds noise, so the choice of noise has no reference.

**Decision.** Gaussian noise, independent per frame and coordinate, added to raw x, y, z before the relative-pose
conversion; rotations untouched. Training: standard deviation 0.01 m, on 50% of training clips. Testing: 0 (the
standard test), 0.005, 0.01, 0.02 and 0.05 m, noise seeds 0, 1 and 2, the same noisy inputs for both models.
Training seeds 0, 1 and 2; last checkpoint. Positions were checked to be in metres on the stored files (12 Nov).

**Why.** 0.01 m per coordinate is about 1.7 cm of total position error, close to the accuracy Meta reports for
Aria's closed-loop trajectories ("no more than 1.5 cm" global RMSE in room-scale scenes; Engel et al. 2023,
arXiv:2308.13561). The other levels make a sensitivity curve around it (a better tracker, then worse ones); they
are not a measured error distribution, and real tracking error is smooth drift, not independent per-frame noise. 0.01 m is about one ordinary frame-to-frame step (median 0.9 to 3.4 cm), so it is
large enough to matter without erasing the clip's overall path (1 to 3 m).

**Effect on comparisons with the paper.** None on the clean test: A and N are tested on the standard benchmark
without noise, comparable with 44.81. All noisy results are a separate experiment with no paper counterpart.

## D6. Experiment 2 is run again, with each exo clip's own frame rate (4 Nov 2026)

**Situation.** D4 resamples exo clips "from 30 to 20 FPS", taking supp. 2.2's 30 FPS for every ViPE clip. ViPE
gives one pose per video frame at the video's own rate, and about a third of the DynPose-100K videos are 24 or 25
FPS (`src/camformer/dynpose_fps.py`: the rate follows from the number of ViPE poses and the length of the
dataset's own 12 FPS frame sequence; the estimates fall into tight groups). Table
`src/camformer/data_files/dynpose100k_fps/vipe_fps_v1.csv.gz` (job 737943, code as in this decision's PR; sha256 of
the uncompressed CSV `1f9cb977e60166c8f05876392f54497a5fb6f55a67578d5a965d99a3e18c13e0`): 99,501 clips, 64,866 at 30
FPS, 18,123 at 25, 16,443 at 24 and 46 at 15, and 23 that fit no standard rate and keep their own estimate. 4,778
(4.8%) are marked `ambiguous`: estimates between the groups (between 25 and 30, 1,456 assigned 25 and 60 assigned 30
by each clip's rounding margin; above 30.5, 30) and the 23 outside every group. An earlier build (job 737912) sent
every estimate from 27.5 to 29.5 to 30 FPS; review found that this contradicts the rounding bound, and the rule was
changed before any run used the table. In E->X, U0 and U1
those clips therefore entered at 16 to 17 poses per second instead of 20. Training and test handled them the same
way, so nothing was mis-scored, but D4's statement was false for those clips and the joint models saw two
different pose rates for exo data.

**Decision.** Run all five setups again, seeds 0-2 (runs `ee2_*`, `bash ~/dl4cv/ego_exo_seed.sh <seed> v2`), with
each clip's rate (`--exo_fps_table dynpose100k_fps/vipe_fps_v1.csv.gz`), on one code revision after the
regression check. S-ego and S-exo do not resample and are rerun only so that every comparison is within one code
revision. The reporting rule in `docs/ego_exo.md`, fixed before the first runs, applies unchanged. The rerun follows
from the bug, not from the results; the first version's results were known when it was decided. They stay in
`results/ego_exo.md` as version 1, and version 2 replaces them as the experiment's result whatever it shows.

## D5. Follow-ups: a grouping-only setup, and Nymeria scored against the 8-second model (3 Nov 2026)

**Situation.** Experiment 1's gain mixes two changes (grouping by take, within-take loss), and the README promises
Nymeria zero-shot results. The paper's Nymeria figures match the authors' 16-second checkpoint (their README pairs
it with Nymeria; the paper itself describes only an 8-second context), while all our runs start from the 8-second
one.

**Decision.** Train G (grouping and verb mask, no loss term; seeds 0-2) on current code, and check separately that
the current code trains like the revision B' and C were trained on (`check_training_equivalence.sh`; the regression
check covers only the released model). Score experiment-1 models on Nymeria through the authors' test path,
resampling recording sessions, compared only with the released 8-second model scored the same way; the 16-second
model is scored once to show the gap. Experiment 2 and 3 models are not
scored on Nymeria (that path feeds neither the domain token nor gaze). Design, tables and rules in
[`followups.md`](followups.md).

## D4. Experiment 2 puts exo trajectories through the ego model's input path (18 Oct 2026)

**Situation.** The joint model (docs/ego_exo.md) needs exo (DynPose, ViPE poses) and ego (Ego-Exo4D) clips in one
encoder. The authors trained them with different tasks, rates and axes.

**Decision.** In the joint runs and in E->X, exo clips go through the long-sequence (ego) path with the whole clip
as one window, are resampled to 20 FPS (`--exo_resample_hz 20`; the first runs took every clip as 30 FPS, which is
wrong for about a third of them, see D6), and are turned into the ego camera axes
by a 90 degree turn about the forward axis (`--exo_to_aria_axes`). S-exo keeps the authors' exo task unchanged.

**Evidence.** The released exo model scores 46.30 through the authors' path and 46.20 through ours at the native
rate (jobs 706967, 706944). Resampling costs about one point on a 5-way check over 7,925 validation clips outside
the test (50.42 against 49.38, job 706966); it is kept as a choice, not as a gain. The axis turn follows from the
stored ego frames (camera x has a world-up component of -0.81, `exo_checks.py`, jobs 706934, 706937) and ViPE's
OpenCV convention.

**Consequence for comparisons.** U1 against S-exo differs in path, rate and axes as well as in sharing. E->X shares
path, rate and axes with the joint models, but not the domain token: U0 against E->X is the comparison closest to
"sharing only" (descriptive, as it was not one of the comparisons fixed in advance); U1 against E->X also includes
the token. (Corrected 4 Nov: this paragraph called U1 against E->X "the clean test of sharing".)

## D3. Hard negatives: fine-tune, add a within-take loss term, and gate the full runs on one pilot (16 Oct 2026)

**Situation.** Two first fine-tuning runs (20 epochs from the released checkpoint, without a fixed seed) gave
the same test accuracy as the released model whether batches were random or grouped by take (44.79 and 44.84
against 44.81 motion-to-text). These two runs were scored on the test, so the redesign below followed a test
result. Before changing anything we measured why, on the released checkpoint and both
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

**Rule for the final claim, written 18 Oct at 11:39 (commit 7d415c8; all times Central European Summer Time, as
on the cluster).** The go decision for seeds 1 and 2 was committed at 00:46 (44f1978) and the seeds were submitted
at 00:47 (jobs 705874-705877). By 11:39 seeds 1 and 2 of A and B' had finished (test tables written between 03:34
and 07:27), and so had C's seeds 0 and 2 (09:17, 10:47); C's seed 1 finished at 12:12. None of these results had
been read when the rule was written; its thresholds were set knowing seed 0's test result. (Corrected 4 Nov: this
heading said "fixed 18 Oct before any result of seeds 1 and 2 is seen".) On motion-to-text
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
