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
  used the labelled clip windows (`t0`-`t1`) of the released CSVs. The model's input window is wider, up to the
  stored window (`data_prep/step3_build_longseq.py:80-81`; `datasets/egoexo4d_pretrain.py:212-214`), so coverage
  there is not yet known. The CPU check (code change 6) measures the share of valid gaze frames over the model's
  actual input windows, per split and per activity type, before training.

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
   - **1 kHz rows.** In each take, `max(end_time)` must be at most `(traj_t1_us - traj_t0_us)/1e6`, and equal to it
     within a few ms for a clip cut at the take's end (the rows-are-milliseconds assumption).
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
     training `--seed`) and written to a committed CSV (`data_files/egoexo4d/gaze_donors_v1.csv.gz`) for train,
     validation and test. Training, testing and the gate's diagnostics all read that file, so a shuffled run sees the
     same donor at every stage.
   - **No temporal overlap.** Stored windows span +-duration seconds around each caption, so neighbouring clips share
     real seconds of gaze. A donor must be at least one stored window away from the clip (the same rule as
     `min_gap` in `loader.py:35-60`); the unit test asserts it for every clip.
   - **Donor slice.** The donor's gaze is taken over the donor's own stored window at the same relative frames
     (`random_start + k*sample_rate`), so in training it is redrawn each epoch together with the clip's own window,
     exactly as real gaze would be; at test time the window is the deterministic test one. Padded with the validity
     flag off if the donor's window is shorter.
   - **Which clips get which donor (decided 20 Oct, before any gaze run).** Every clip in a -shuf run gets realistic
     but wrong gaze; none gets zeros. Rule 1, used whenever possible: a different stored window of the **same take**
     that has gaze and is at least one window away. Rule 2, only for clips whose take has no such window (mostly
     short takes): a window of a **different take of the same scenario and the same split group** (e.g. another
     Cooking take; training clips only from training takes, validation and test clips only from validation/test
     takes, so no gaze crosses the train/eval boundary), drawn with the same fixed data seed. The donor CSV records the donor's take and which rule supplied it (`donor_rule`:
     `same_take` or `same_scenario`). Counts per split and activity type are under "Checks before training".
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
training only (job times recorded in `results/hard_negatives.md`); testing adds up to about 1 h per run (the test
jobs' time limit). Full runs (G0, GT, GT-shuf): about 3.3 h of training plus test each. Frozen runs (GF, GF-shuf): no encoder gradients, **Open:** time them in the smoke test; budget
assumes 3 h. Per seed about 16 GPU-hours of training plus up to 5 of testing; three seeds about 63 in all, about 32 h of wall time on our 2 GPUs (2 jobs
running, 4 queued at most), plus the pilot's diagnostics (about 1 h). The pilot's seed-0 training is about 8 h; its testing (about 2.5 h) runs only after the gate decision. All of
it fits well before the 4 Nov date in `SCHEDULE.md`. These wall
times assume both GPUs are free: anything else of ours on the cluster (follow-up runs of experiments 1 or 2)
pushes the 4 Nov date in `SCHEDULE.md` back by its own length.

## Checks before training: results (20 Oct 2026)
CPU job 709477 (`sbatch ~/dl4cv/gaze_prep.sh`, `gaze_prep.py`, code d900001); unit tests `python test_gaze.py` on
the login node, all passing (zero-init identity, gaze frame k on pose token k, dataset time of frame k, donor map,
collate).
- **Gaze cache:** 3,597 takes; 4 without a gaze file or trajectory start (the 359 training clips noted above).
- **1 kHz rows: passed.** Over 3,601 takes, max(`end_time`) minus trajectory length is at most +0.001 s; no clip
  ends after its trajectory; 3,360 takes have a clip cut within 5 ms of the end.
- **Coverage over the model's input windows: passed.** Valid gaze share, mean per clip: train 99.67% physical and
  100.00% procedural (30,000 sampled training clips, seed 0); validation and test 100.00% in both types. No gap
  between types, so no secondary analysis is needed.
  Shuffled input (job 709883, code ee408bd, same clips): mean valid share 97.6% (train and validation physical) to
  99.8% (test procedural), against 99.7-100% for real gaze; same-take and cross-take donors within 2.2 points of
  each other in every split and type (`results/gaze_checks.md`). So a -shuf run sees the validity flag off on about
  0.2-2.4% more frames than its real twin, most on physical clips; reported with the results, small next to the
  content difference the control is for.
- **Donor map (rebuilt 20 Oct with the two rules in Design 6):** `data_files/egoexo4d/gaze_donors_v1.csv.gz`,
  207,176 clip windows, every one with a donor; checks asserted (same-take donors: different window, no overlap in
  time; cross-take donors: different take, same scenario, same split group; every donor take has gaze). Built by
  job 709883 running code ee408bd (`sbatch ~/dl4cv/gaze_prep.sh donors coverage`). The committed file is that
  job's output: `zcat src/camformer/data_files/egoexo4d/gaze_donors_v1.csv.gz | sha256sum` must print the hash in
  the job's log (the hash is of the uncompressed CSV text):
  2172ba9551d8d09b40b9fa9f82d0a177fc5bb0692d6f9c22e5c938383b848ffa. Valid-gaze share of the shuffled input, by rule: see the coverage results and `results/gaze_checks.md`. Clips (CSV rows, as the model sees them) by rule:

  | Split | Type | Clips | Same take | Other take, same scenario |
  |---|---|---|---|---|
  | train | physical | 104,920 | 42,146 (40.2%) | 62,774 (59.8%) |
  | train | procedural | 54,266 | 51,050 (94.1%) | 3,216 (5.9%) |
  | val | physical | 26,493 | 9,684 (36.6%) | 16,809 (63.4%) |
  | val | procedural | 42,580 | 41,670 (97.9%) | 910 (2.1%) |
  | test | physical | 22,290 | 12,787 (57.4%) | 9,503 (42.6%) |
  | test | procedural | 13,105 | 12,356 (94.3%) | 749 (5.7%) |

  Physical takes are short, so most physical clips need rule 2; procedural clips almost all use rule 1. The
  procedural comparison (the pre-named split) is therefore nearly a pure same-take control, while on physical
  clips the shuffled gaze also comes from a different person and session. Reported with the results.
- **Vestibulo-ocular check: FAILED the criterion fixed above** (peak at lag 0, +-1 frame). Correlation of eye-in-head
  yaw rate (frame k) with head rotation rate about device axis 0 (frame k + lag), 1,000 validation clips:

  | lag (frames of 50 ms) | -2 | -1 | 0 | +1 | **+2** | +3 | +4 |
  |---|---|---|---|---|---|---|---|
  | real gaze | +0.053 | -0.028 | -0.132 | -0.218 | **-0.247** | -0.224 | -0.180 |
  | donor gaze (control) | +0.001 | +0.002 | +0.002 | +0.002 | +0.002 | +0.002 | +0.002 |

  Real gaze correlates with head motion at r about -0.25 on device axis 0 (about 0 with donor gaze), but the peak is at +2 frames: the gaze at a pose frame's time matches head motion about 100 ms later. Per
  the rule above, training does not start until this is understood. (Job 709477, code d900001,
  `sbatch ~/dl4cv/gaze_prep.sh`.) The follow-up below shows this check cannot tell a timing error from normal
  eye-head behaviour.
- **Native-rate follow-up (20 Oct): the offset is not a pipeline artefact, and it is not a clean clock offset.
  Training stays on hold.** `sbatch ~/dl4cv/gaze_prep.sh vor_native --vor_clips 3000`, job 709871, code e0abf27:
  eye-in-head rate from the native 10 Hz gaze samples (central differences, no interpolation) against head rate
  over the same two sample times shifted by the lag, -300 to +300 ms in 10 ms steps; 3,000 validation clips, 627
  takes, 1,558,780 gaze samples.

  | Scan | Most negative r at | r there | r at 0 ms |
  |---|---|---|---|
  | yaw vs head axis 0, all samples | +120 ms | -0.277 | -0.161 |
  | pitch vs head axis 1, all samples | +130 ms | -0.258 | -0.163 |
  | yaw, "VOR-only" samples (eye < 100 deg/s, head > 15 deg/s) | +150 ms | -0.078 | +0.035 |
  | 20 Hz model features, yaw, central differences, all | +100 ms | -0.263 | -0.149 |

  - Not the feature pipeline: native samples give the same offset as the 20 Hz features.
  - Per take, among the 160 of 627 takes with a clear negative peak (r <= -0.2 with at least 40 samples; a selected
    subset), the peak is always positive but spread from +90 to +250 ms (p10 to p90, median +140), with activity
    medians from +100 ms (Dance, 29 takes) to +230 ms (Soccer, 13 takes). A pure clock offset would give the same
    lag for every activity; these medians suggest it is not one, but with these take counts and a selected subset
    they do not prove it.
  - Two weaknesses of the scan itself: it takes the most negative r, which is the wrong lobe if the sign
    conventions of eye yaw and head axis 0 are opposite; and the per-take statistics use a selected subset.
  - The reflex itself is not isolated at 10 Hz: on the "VOR-only" samples the eye/head gain is -0.03, where the
    reflex would give about -1, and the correlation there is weak. At 10 Hz, eye movement is dominated by gaze
    shifts, in which the eyes move ahead of the head by an amount that depends on the task, which fits an
    activity-dependent lag of 100-250 ms. The sign conventions of eye yaw and head axis 0 are not pinned down, so
    the scan cannot tell "eyes counter-rotate" from "eyes lead in the same direction".
  - Consequence: the alignment check fixed above (peak at lag 0 +-1 frame) **failed and stays failed**; it rested
    on the 10 Hz signal being dominated by the reflex, which it is not, so it cannot confirm or rule out a timing
    error. No shift is applied. **Needs a decision before training, recorded here before anything else is run:**
    either (a) a replacement alignment test with its pass/fail threshold written down first (for example the sign
    pinned with a synthetic known rotation, then a scan of |r|; or gaze projected into the RGB frame, which needs
    video we do not have), or (b) proceeding on the documented timestamps (same device clock, Aria docs) as an
    **untested assumption**, stated as such in the report, not as a passed check. Logs: `results/gaze_checks.md`.
    The choice is recorded, with its date and (for (a)) the test and its threshold, in
    `src/camformer/data_files/egoexo4d/gaze_alignment_decision.txt`, committed before any run; `gaze_seed.sh`
    refuses a full seed until that file exists. The one-epoch smoke test (`bash ~/dl4cv/gaze_seed.sh smoke`) does
    not need it.


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
   clip, at least one window apart), and the validity-flag share over the input windows, per split and per activity type
   (procedural / physical). If the procedural and physical shares differ by more than 5 points, the gap is reported
   and the comparison is repeated on clips with at least 90% valid gaze as a secondary analysis.

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

## Alignment decision (20 Oct 2026)
Proceed on the documented timestamps, no shift (`data_files/egoexo4d/gaze_alignment_decision.txt`). The vestibulo-
ocular criterion fixed above stays failed and is reported as such; alignment is an untested assumption in the report.
Consequence for the results: if gaze is offset from the poses by 2 to 3 frames, a null or weak GT - G0 result is
partly confounded with misalignment, so a null is reported as "no gain with gaze as aligned here", not as "gaze does
not help".
