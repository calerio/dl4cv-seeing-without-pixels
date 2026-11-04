# Eye-gaze check (16 Oct 2026)

Question: can the Ego-Exo4D eye-gaze recordings be used as a second input next to the released camera-pose clips?
Scripts: [`scripts/gaze_check/`](../scripts/gaze_check/).

## Method
- Gaze: `takes/<take>/eye_gaze/general_eye_gaze.csv` (Aria MPS output, already downloaded for all 4,990 takes).
  The personalized version exists for only 820 takes, so we use the general one.
- Clip windows: `start_time` and `end_time` in the released `train_v2.csv`, `val_v2.csv` and `test_alltasks_mcqv0.csv`.
  The authors' `data_prep/step1_extract_segments.py` measures these in seconds from the first row of the take's
  `trajectory/closed_loop_trajectory.csv`.
- The full trajectory download is 474 GB, but only that first timestamp is needed. `fetch_traj_t0.py` reads just the
  first and last 4 KB of each take's trajectory file (byte-range requests), for all 4,994 takes, in 3.5 minutes.
  Gaze and trajectory are both stamped in Aria device time (`tracking_timestamp_us`), so no conversion between
  clocks is needed. (Note, 4 Nov: whether the two streams are aligned within a frame was tested later and could not
  be verified; see `src/camformer/data_files/egoexo4d/gaze_alignment_decision.txt` and `docs/gaze.md`.)
- For each clip, `check_gaze.py` counts the gaze samples inside its window and the share of the window not covered
  by a gap longer than 0.25 s.

## Results
- Gaze is sampled at a steady 10 Hz (spacing 100 ms at the 1st and 99th percentiles). The pose clips are at 20 Hz,
  so each gaze sample spans two pose frames.
- Gaze starts 0.03 s after the trajectory (median; at most 0.10 s), so the two streams cover the same time span.
  (This checks coverage, not alignment within a frame; see the note above.)

| Split | Clips | Takes | Clips without gaze | Clips with ≥95% coverage | Longest gap (99th pct.) |
|---|---|---|---|---|---|
| Train | 159,186 | 2,820 | 359 (4 takes) | 99.77% | 0.1 s |
| Val | 69,073 | 781 | 0 | 100% | 0.1 s |
| Test | 35,395 | 671 | 0 | 100% | 0.1 s |

## Conclusion
Gaze covers almost every released clip and needs no further download. The 359 training clips without gaze
are dropped (or given a zero gaze input) in the gaze experiment.

## Gaze file columns (20 Oct 2026)
`head -1 ~/dl4cv/data/egoexo4d/takes/cmu_bike01_2/eye_gaze/general_eye_gaze.csv` on the cluster:

```
tracking_timestamp_us,left_yaw_rads_cpf,right_yaw_rads_cpf,pitch_rads_cpf,depth_m,left_yaw_low_rads_cpf,right_yaw_low_rads_cpf,pitch_low_rads_cpf,left_yaw_high_rads_cpf,right_yaw_high_rads_cpf,pitch_high_rads_cpf,tx_left_eye_cpf,ty_left_eye_cpf,tz_left_eye_cpf,tx_right_eye_cpf,ty_right_eye_cpf,tz_right_eye_cpf,session_uid
```
