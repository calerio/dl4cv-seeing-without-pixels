# Experiment 3: eye gaze as a second input (design, 20 Oct 2026)

Written before any code, following the same steps as experiments 1 and 2: the table to fill, the seeds, the reason
for each choice and the reporting rule are fixed here, and an independent review comes before the first GPU job.

## Question
Does knowing where the wearer looks, in addition to how the head moves, help CamFormer pick the right caption on
Ego-Exo4D? And is it enough to train only a small new gaze input on top of the released model, or does the whole
encoder need retraining?

**Hypothesis.** The paper finds the trajectory weakest on procedural activities, "where camera motions are more
localized and subtle" (Sec. 4, in the activity-classification discussion around Fig. 3c; Table 2 shows the same
pattern on retrieval). Its Limitations add that the trajectory carries no noun semantics; gaze will not add those
either, so the hypothesis is at the level of verbs (what the hands are doing), which is what the 5-way questions test. In those tasks the head moves little while the eyes move between
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
- Coverage: every validation and test clip; 359 of 159,186 training clips (4 takes) have no gaze. The coverage check
  used the CSVs' `start_time`/`end_time`, which are the bounds of the stored trajectory window
  (`data_prep/step3_build_longseq.py:79-80`), and every model input window lies inside it
  (`datasets/egoexo4d_pretrain.py:212-214`), so input windows are covered too. Gaze availability is therefore not
  correlated with activity type in validation or test.

## Design
1. **Start from the released model, not from experiment 1's B'.** Each experiment changes one thing against the
   released checkpoint, so the gaze effect is measured on its own and stays comparable with the paper's numbers and
   with experiment 1's baseline A. Gaze on top of B' would mix two changes; it is an optional extra run (G+B'), cut first.
2. **Gaze features per pose frame.** For each pose frame (20 Hz), gaze is linearly interpolated from the 10 Hz samples
   at the frame's time: yaw (mean of left and right eye), pitch, and log depth (`depth_m` clamped to at least 0.1 m
   before the log; NaN or missing depth gives depth 0 and flag 0; Aria depth is unreliable beyond about 1 m, so it
   is a weak feature by design), plus a validity flag (0 where no gaze sample lies within 0.2 s, and for the 359
   clips without gaze). Gaze is cut with the same `random_start`,
   `random_end` and stride as the poses (`egoexo4d_pretrain.py:212-217`), and follows the same path when a sample is
   redrawn (line 223). Angles are already in the Aria central-pupil frame
   (`_cpf`), which moves with the head, so gaze is "where the eyes point relative to the head": exactly the part the
   trajectory does not carry. 4 numbers per frame.
3. **Alignment.** The stored arrays are per-clip windows, not whole takes: `step3_build_longseq.py:79-85` cuts rows
   `int(start_time*1e3)` to `int(end_time*1e3)` of the 1 kHz trajectory with stride 50, and `_load_cam_traj` loads
   that window by `save_id` (`egoexo4d_pretrain.py:182-183`). So stored frame j is at `start_time + 0.05*j` seconds
   after the trajectory's first row, `start_time` being the CSV column; the dataset then takes frames
   `random_start + k*sample_rate` (lines 217-218). The gaze time of pose frame k is therefore
   `traj_t0_us + (start_time + 0.05*(random_start + k*sample_rate))*1e6`, with `traj_t0_us` from
   `scripts/gaze_check/fetch_traj_t0.py`. Two checks before training (CPU job, code change 6):
   - **1 kHz rows.** For the last clip of a take, `end_time` must equal `(traj_t1_us - traj_t0_us)/1e6` within
     a few ms (the rows-are-milliseconds assumption).
   - **Positive check, vestibulo-ocular reflex.** When the head turns, the eyes counter-rotate in the head. Over a
     sample of clips, eye-in-head yaw rate must anti-correlate with head yaw rate from the poses, with the peak at lag
     0 (+-1 frame). A misaligned gaze shows the peak elsewhere or none at all; this is the only check that proves
     alignment, since the zero-initialised layer hides any gaze in the reproduction checks.
4. **How gaze enters the model.** A new linear layer `gaze_proj` (4 -> 128) whose output is added to the pose tokens
   `h1` right after `in_proj` (`models/cm_encoder.py:121`), before the gravity token is prepended (line 140), so gaze
   frame i lands on pose token i; the branch without gravity (line 146) gets the same add. A unit test checks that
   alignment. Weight and bias are both initialised to zero, so at step 0 the model is exactly the released one (same
   idea as the domain token, line 101). The gravity token is untouched.
5. **Two ways to train it** (README item 3):
   - **Frozen (GF):** the released encoder's parameters are frozen; only `gaze_proj` trains (640 parameters). The
     model stays in training mode, so dropout and gravity dropout (`cm_encoder.py:123-126`) behave as in GT, and GF
     and GT differ only in which parameters update. Asks: can gaze be bolted on without touching what the model knows?
   - **Full (GT):** encoder and `gaze_proj` train together, as in experiment 1.
6. **Control for "more parameters / more training".** Each gaze run has a twin that gets gaze from a different clip of
   the same take (GF-shuf, GT-shuf): same input statistics, same parameters and training, no information about this
   clip. If a gain is real, gaze must beat the shuffled twin, not only the no-gaze baseline. Details, fixed now:
   - **One donor map for everything.** Each clip's donor is drawn once with a fixed data seed (independent of the
     training `--seed`) and written to a committed CSV (`data_files/egoexo4d/gaze_donors_v1.csv`) for train,
     validation and test. Training, testing and the gate's diagnostics all read that file, so a shuffled run sees the
     same donor at every stage.
   - **No temporal overlap.** Stored windows span +-duration seconds around each caption, so neighbouring clips share
     real seconds of gaze. A donor must be at least one stored window away from the clip (the same rule as
     `min_gap` in `loader.py:35-60`); the unit test asserts it for every clip.
   - **Donor slice.** The donor's gaze is taken over the donor's own stored window at the same relative frames
     (`random_start + k*sample_rate`), so in training it is redrawn each epoch together with the clip's own window,
     exactly as real gaze would be; at test time the window is the deterministic test one. Padded with the validity
     flag off if the donor's window is shorter. Donors are drawn only among clips of the take that have gaze.
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
Ego-Exo4D 5-way test, motion-to-text, by type (procedural / physical) and overall (`score_mcq.py`; 2,621 procedural
and 4,458 physical questions, recorded in `results/hard_negatives.md`), and `compare_runs.py` with three seeds per
setup (its two-level bootstrap over seeds and takes) for every interval below, including V - V-shuf. Nymeria and
DynPose have no gaze and are not used. Secondary split, named now and never used for a claim: in-view vs
out-of-view (`score_mcq.py:47`); gaze should help more where the hands are in view.

**Reporting rule, fixed now.** Primary split named in advance: procedural (the hypothesis). Expected interval
widths come from experiment 1's three-seed intervals (`results/hard_negatives.md`): half-widths about 1.2 points on
procedural, 1.2 on physical and 0.9 overall. The margins below are set from those: the equivalence margin of +-2 is
wider than one half-width so that it can pass when two variants are truly equal, and the -1 floor is the one used in
experiment 1. A claim for a variant V needs three seeds of V **and** of V-shuf.
- "Gaze helps" (for a variant V in {GF, GT}) is claimed only if, on procedural questions, the 95% interval of
  V - G0 is above zero **and** the 95% interval of V - V-shuf is above zero. Both must hold; the second rules out a
  gain from extra parameters or training alone.
- "Frozen is enough" is claimed only if "gaze helps" holds for GF and the interval of GF - GT on procedural lies
  inside +-2 points. Because +-2 would still pass with GT 1.9 points better, the report gives GF - GT with its
  interval next to the claim and says "frozen is within 2 points of full", not "as good as full".
- "No cost on physical activities" is claimed only if the interval of V - G0 on physical questions has a lower end
  above -1.
- Anything else is reported as measured, including a negative result.

**Pilot gate, on training data only.** Seed 0 of all five runs first. Seeds 1 and 2 run only if (a) every run
finished, and (b) on the within-take 5-way questions built from training takes (`diagnose_within_take.py`, 5,000
questions, seed 0, reading the committed donor map), GT beats both G0 and GT-shuf by at least **1.0** point.
Calibration, written here so the threshold is not set after the fact: in D3, plain fine-tuning moved this score by
+0.7 (59.8 to 60.50), and B' moved it by +1.42 over A (61.92 against 60.50) while gaining +3.8 on the procedural
test. A threshold of 1.0 would have let B' through; 1.5 would have stopped a real effect. One accuracy on 5,000
questions has a standard error of about 0.7 points, so 1.0 accepts some risk of continuing on noise, which seeds 1
and 2 then test. The frozen runs follow GT's decision.
- A failed gate is a budget decision (no seeds 1 and 2), not a null result. Either way, after the decision is
  written in this file, seed 0 of all five runs is scored on the test and reported, labelled single-seed.
- The pilot runs are **not** scored on the test until the decision is written: the training jobs run without the
  automatic test step. (Training-take accuracy can overstate a gain; acceptable for a go/no-go, never for a claim.)

## Compute
Measured: one Ego-Exo4D epoch is about 8.3 min on one A100 (D1); experiment 1's 20-epoch runs took 3.3 h each
including their test (job times recorded in `results/hard_negatives.md`). Full runs (G0, GT,
GT-shuf): about 3.3 h each, test included. Frozen runs (GF, GF-shuf): no encoder gradients, **Open:** time them in the smoke test; budget
assumes 3 h. Per seed about 16 GPU-hours; three seeds about 48, about 24 h of wall time on our 2 GPUs (2 jobs
running, 4 queued at most), plus the pilot's diagnostics (about 1 h). The pilot alone is about 8 h. These wall
times assume both GPUs are free: anything else of ours on the cluster (follow-up runs of experiments 1 or 2)
pushes the 4 Nov date in `SCHEDULE.md` back by its own length.

## Code changes (each behind a flag; defaults keep the authors' behaviour)
1. `datasets/egoexo4d_pretrain.py`: `--gaze {none,real,shuffled}`; load each take's gaze CSV once, return the 4-D
   gaze features for the window's frames next to the poses (window code at lines 207-217).
2. `loader.py`: pad and batch the gaze features like the poses.
3. `models/cm_encoder.py`: `--gaze_input` adds the zero-initialised `gaze_proj` (Design 4).
4. `tasks/pretrain_longseq.py`: pass gaze to the encoder; `--freeze_encoder` freezes encoder parameters only
   (training mode kept).
5. **Test-time path.** `train.py --test` accepts `--gaze`/`--gaze_input`; `scripts/slurm/eval_run.sh` (and the seed
   launch script) read them from the run's `args.json`, as `eval_ego_exo.sh` already does for experiment 2, so a
   gaze checkpoint is always tested with its gaze and a shuffled run with the committed donor map.
   `diagnose_within_take.py` builds its arguments the same way (today it hard-codes them, lines 30-33) and applies
   the shuffle for the -shuf runs.
6. A CPU check script: Design 3's two alignment checks, building and checking the donor map (same take, different
   clip, at least one window apart), and the validity-flag share per split.

Checks before the first run:
- The reproduction check with all flags off and with `--gaze_input` on and `--gaze none`: both must give 44.81. This
  proves only that the plumbing changes nothing when the layer is zero; it says nothing about alignment.
- Design 3's 1 kHz and vestibulo-ocular checks (alignment).
- Unit tests: gaze frame k lands on pose token k; every donor is from the same take, a different clip, and at least
  one window apart; the donor map is identical whether read in training, test or diagnostics.
- A one-epoch smoke test of GT and GF, timing both. Then, on that GT checkpoint, the test scored with real gaze and
  with gaze zeroed must differ; equal scores mean the test path dropped gaze.

## Risks and what we cut first
- **Gaze may add nothing.** Then the shuffled twins match the real ones and we report that, with the training-take
  diagnostics.
- **Alignment errors** would silently destroy the signal; Design 3's check comes before any training.
- **Frozen gaze may be too weak to learn** with 640 parameters; that is a result, not a bug, if the smoke test shows
  the layer's weights moving.
- **If time runs short**, cut in this order: G+B', then the frozen pair (GF and GF-shuf together, all seeds: GF
  alone cannot support a claim without three seeds of GF-shuf). Keep G0, GT and GT-shuf.
