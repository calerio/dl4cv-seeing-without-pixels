# Experiment 3: eye gaze as a second input (design, 20 Oct 2026)

Written before any code, following the same steps as experiments 1 and 2: the table to fill, the seeds, the reason
for each choice and the reporting rule are fixed here, and an independent review comes before the first GPU job.

## Question
Does knowing where the wearer looks, in addition to how the head moves, help CamFormer pick the right caption on
Ego-Exo4D? And is it enough to train only a small new gaze input on top of the released model, or does the whole
encoder need retraining?

**Hypothesis.** The paper finds the trajectory weakest on procedural activities, "where camera motions are more
localized and subtle" (Sec. 4, discussion of Table 2). In those tasks the head moves little while the eyes move between
the hands, a tool and the object being worked on. Gaze should carry part of what the head no longer shows, so the
gain, if any, should be largest on procedural questions. If gaze only duplicates head motion, the gain is zero.

## What the paper did
The paper uses camera trajectory only. It never uses gaze; the word does not appear in the main paper or the
supplement. Its comparisons with other modalities are with video (Table 2: fusing trajectory and EgoVLPv2 video
features helps on procedural activities). So this experiment has no reference number in the paper: the comparison is
entirely between our own runs, all starting from the released checkpoint.

## Data (measured, `results/gaze_check.md`)
- Source: `takes/<take>/eye_gaze/general_eye_gaze.csv` (Aria MPS), already on the cluster for every take. Columns used:
  `tracking_timestamp_us`, `left_yaw_rads_cpf`, `right_yaw_rads_cpf`, `pitch_rads_cpf`, `depth_m` (header recorded
  in `results/gaze_check.md`, "Gaze file columns").
- 10 Hz, starting 0.03 s (median) after the trajectory, on the same device clock.
- Coverage of the labelled clip window (`start_time` to `end_time`): every validation and test clip; 359 of 159,186
  training clips (4 takes) have no gaze. The model's input window is wider than the labelled clip
  (`datasets/egoexo4d_pretrain.py:212-214`), so frames near a take's start or end may lack gaze even where the clip is
  covered; the CPU check (code change 6) measures the validity-flag share over the input window, per split.

## Design
1. **Start from the released model, not from experiment 1's B'.** Each experiment changes one thing against the
   released checkpoint, so the gaze effect is measured on its own and stays comparable with the paper's numbers and
   with experiment 1's baseline A. Gaze on top of B' would mix two changes; it is an optional extra run (G+B'), cut first.
2. **Gaze features per pose frame.** For each pose frame (20 Hz), gaze is linearly interpolated from the 10 Hz samples
   at the frame's time: yaw (mean of left and right eye), pitch, and log depth, plus a validity flag (0 where no gaze
   sample lies within 0.2 s, and for the 359 clips without gaze). Gaze is cut with the same `random_start`,
   `random_end` and stride as the poses (`egoexo4d_pretrain.py:212-217`), and follows the same path when a sample is
   redrawn (line 223). Angles are already in the Aria central-pupil frame
   (`_cpf`), which moves with the head, so gaze is "where the eyes point relative to the head": exactly the part the
   trajectory does not carry. 4 numbers per frame.
3. **Alignment.** Frame index i of the stored trajectory corresponds to time i x 50 ms after the trajectory's first row
   (20 Hz, `pre_sample_rate`). Gaze time is `tracking_timestamp_us` minus the trajectory's first timestamp
   (`scripts/gaze_check/fetch_traj_t0.py`). **Open:** check that i x 50 ms matches the clip's `start_time` in the CSVs
   for a sample of clips (CPU job), before training.
4. **How gaze enters the model.** A new linear layer `gaze_proj` (4 -> 128) whose output is added to the pose tokens
   `h1` right after `in_proj` (`models/cm_encoder.py:121`), before the gravity token is prepended (line 140), so gaze
   frame i lands on pose token i; the branch without gravity (line 146) gets the same add. A unit test checks that
   alignment. It is initialised to zero, so at step 0 the model is exactly the released one (same idea as the domain token, line 101).
   The gravity token is untouched.
5. **Two ways to train it** (README item 3):
   - **Frozen (GF):** the released encoder is frozen; only `gaze_proj` trains (640 parameters). Asks: can gaze be
     bolted on without touching what the model knows?
   - **Full (GT):** encoder and `gaze_proj` train together, as in experiment 1.
6. **Control for "more parameters / more training".** Each gaze run has a twin that gets gaze from a different clip of
   the same take (GF-shuf, GT-shuf): same input statistics, same parameters and training, no information about this
   clip. If a gain is real, gaze must beat the shuffled twin, not only the no-gaze baseline. Details, fixed now: the
   donor clip is a fixed random choice per clip (seeded), in training and at test time alike; the donor's gaze is
   cut over a window of the same length starting at the donor's own window start, and padded with the validity flag
   off if the donor's window is shorter; donors are drawn only among clips of the take that have gaze.
7. **Everything else** as experiment 1's A: 20 epochs, batch 1,024, lr 1e-4, random batches, authors' loss, last
   checkpoint tested. Frozen runs use the same lr for `gaze_proj`.

## Runs
| Run | Encoder | Gaze input | Purpose |
|---|---|---|---|
| G0 | trains | none | Baseline: the same extra training without gaze (as A, on this code revision) |
| GF | frozen | real | Frozen variant |
| GF-shuf | frozen | shuffled within take | Control for GF |
| GT | trains | real | Full fine-tuning variant |
| GT-shuf | trains | shuffled within take | Control for GT |
| G+B' (optional) | trains, from B' setup | real | Only if time allows; cut first |

Seeds 0, 1, 2 for G0, GF, GF-shuf, GT, GT-shuf. All runs of a seed on one code revision.

## Evaluation
Ego-Exo4D 5-way test, motion-to-text, by type (procedural / physical) and overall (`score_mcq.py`; 2,621 procedural and 4,458 physical
questions, recorded in `results/hard_negatives.md`), and `compare_runs.py` with three seeds per setup (its
two-level bootstrap over seeds and takes) for every interval below, including V - V-shuf. Nymeria and DynPose have no gaze and are not used.

**Reporting rule, fixed now.** Primary split named in advance: procedural (the hypothesis). Expected interval
widths come from experiment 1's three-seed intervals (`results/hard_negatives.md`): half-widths about 1.2 points on
procedural, 1.2 on physical and 0.9 overall. The margins below are set from those: the equivalence margin of +-2 is
wider than one half-width so that it can pass when two variants are truly equal, and the -1 floor is the one used in
experiment 1.
- "Gaze helps" (for a variant V in {GF, GT}) is claimed only if, on procedural questions, the 95% interval of
  V - G0 is above zero **and** the 95% interval of V - V-shuf is above zero. Both must hold; the second rules out a
  gain from extra parameters or training alone.
- "Frozen is enough" is claimed only if "gaze helps" holds for GF and the interval of GF - GT on procedural lies
  inside +-2 points.
- "No cost on physical activities" is claimed only if the interval of V - G0 on physical questions has a lower end
  above -1.
- Anything else is reported as measured, including a negative result.

**Pilot gate, on training data only.** Seed 0 of all five runs first. Seeds 1 and 2 run only if (a) every run
finished, and (b) on the within-take 5-way questions built from training takes (`diagnose_within_take.py`, 5,000
questions, seed 0, extended to pass gaze), GT beats both G0 and GT-shuf by at least 1.5 points. One accuracy on 5,000
questions has a standard error of about 0.7 points, and plain fine-tuning moved this score by about 0.6 (D3), so 1.5
is about twice the noise. The frozen runs follow GT's decision. If (b) fails, we report seed 0 with the diagnosis and
stop. The pilot runs are **not** scored on the test until the gate decision is written in this file: the training
jobs run without the automatic test step, and testing is submitted after the decision. (Training-take accuracy can
overstate a gain; that is acceptable for a go/no-go, never for a claim.)

## Compute
Measured: one Ego-Exo4D epoch is about 8.3 min on one A100 (D1); experiment 1's 20-epoch runs took 3.3 h each
including their test (job times recorded in `results/hard_negatives.md`). Full runs (G0, GT,
GT-shuf): about 3.3 h each, test included. Frozen runs (GF, GF-shuf): no encoder gradients, **Open:** time them in the smoke test; budget
assumes 3 h. Per seed about 16 GPU-hours; three seeds about 48, about 24 h of wall time on our 2 GPUs (2 jobs
running, 4 queued at most), plus the pilot's diagnostics (about 1 h). The pilot alone is about 8 h.

## Code changes (each behind a flag; defaults keep the authors' behaviour)
1. `datasets/egoexo4d_pretrain.py`: `--gaze {none,real,shuffled}`; load each take's gaze CSV once, return the 4-D
   gaze features for the window's frames next to the poses (window code at lines 207-217).
2. `loader.py`: pad and batch the gaze features like the poses.
3. `models/cm_encoder.py`: `--gaze_input` adds the zero-initialised `gaze_proj` (Design 4).
4. `tasks/pretrain_longseq.py`: pass gaze to the encoder; `--freeze_encoder` trains only `gaze_proj`.
5. `diagnose_within_take.py`: pass gaze, for the gate.
6. A CPU check script for Design 3 (alignment) and the validity-flag share per split.

Checks before the first run: the reproduction check with all flags off and with `--gaze_input` on and `--gaze none`
(both must give 44.81: the zero layer changes nothing); a unit test that the shuffled gaze comes from the same take
and a different clip; a one-epoch smoke test of GT and GF timing both.

## Risks and what we cut first
- **Gaze may add nothing.** Then the shuffled twins match the real ones and we report that, with the training-take
  diagnostics.
- **Alignment errors** would silently destroy the signal; Design 3's check comes before any training.
- **Frozen gaze may be too weak to learn** with 640 parameters; that is a result, not a bug, if the smoke test shows
  the layer's weights moving.
- **If time runs short**, cut in this order: G+B', then seeds 1 and 2 of GF-shuf, then the frozen variant entirely.
  Keep G0, GT and GT-shuf.
